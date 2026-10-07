from __future__ import annotations

import json
import logging
import os
import threading
from typing import List, Optional, Sequence

import cv2
import numpy as np

from . import MODELS_DIR, torch_device, wait_gpu

log = logging.getLogger("fraudshield.custom")

CUSTOM_DIR = MODELS_DIR / "custom"
HF_CACHE = MODELS_DIR / "hf-cache"
SR = 16000

VOICE_BACKBONE = "facebook/wav2vec2-xls-r-300m"
VOICE_LAYERS = (3, 6, 9, 12, 15, 18, 21, 24)
FACE_BACKBONE = "openai/clip-vit-large-patch14"
FACE_LAYERS = (8, 12, 16, 20, 24)
CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], np.float32)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], np.float32)

def enabled() -> bool:
    return os.environ.get("FRAUDSHIELD_CUSTOM", "0") == "1"

def head_path(kind: str):
    return CUSTOM_DIR / f"{kind}_head.npz"

CHANNELS = {"none": None, "call": (80.0, 7000.0), "phone": (300.0, 3400.0)}

def normalize_channel(y16: np.ndarray, channel: str = "call") -> np.ndarray:

    band = CHANNELS.get(channel)
    if band is None:
        return np.asarray(y16, np.float32)
    from scipy.signal import butter, sosfiltfilt
    x = np.asarray(y16, np.float64)
    sos = butter(4, [band[0], band[1]], btype="bandpass", fs=SR, output="sos")
    x = sosfiltfilt(sos, x) if len(x) > 60 else x
    rms = float(np.sqrt(np.mean(x ** 2))) + 1e-9
    x = x * (10 ** (-23 / 20) / rms)
    rng = np.random.default_rng(len(x))
    x = x + rng.standard_normal(len(x)) * 10 ** (-23 / 20) * 10 ** (-50 / 20)
    return np.clip(x, -1.0, 1.0).astype(np.float32)

def _half_ok(device: str) -> bool:
    return device == "cuda" and os.environ.get("FRAUDSHIELD_FP32", "0") != "1"

def _finite_or_fp32(bb, run):
    res = run()
    if bb.dtype != bb.torch.float32 and not bool(bb.torch.isfinite(res).all()):
        log.warning("%s: fp16 produced NaN/inf features, switching to float32", bb.repo)
        bb.dtype = bb.torch.float32
        bb.model = bb.model.float()
        res = run()
    return res

class Head:

    def __init__(self, params: dict):
        self.p = params
        self.kind = str(params["kind"])
        self.layer = int(params["layer"])
        self.backbone = str(params["backbone"])
        self.meta = json.loads(str(params.get("meta", "{}")))

    @classmethod
    def load(cls, path) -> "Head":
        with np.load(path, allow_pickle=False) as z:
            return cls({k: z[k] for k in z.files})

    def save(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        out = {k: v for k, v in self.p.items() if k != "meta"}
        out["meta"] = np.array(json.dumps(self.meta))
        np.savez(path, **out)

    def logits(self, X: np.ndarray) -> np.ndarray:
        X = (np.asarray(X, np.float32) - self.p["mu"]) / self.p["sd"]
        if self.kind == "logreg":
            return X @ self.p["w"] + self.p["b"]
        h = np.maximum(X @ self.p["W1"] + self.p["b1"], 0.0)
        return h @ self.p["W2"] + self.p["b2"]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        z = np.clip(self.logits(X), -30, 30)
        return (1.0 / (1.0 + np.exp(-z))).astype(np.float32)

    @property
    def auc(self) -> Optional[float]:
        return self.meta.get("test_auc")

    @property
    def label(self) -> str:
        return self.meta.get("label") or "Custom detector (ours)"

class VoiceBackbone:

    def __init__(self, repo: str = VOICE_BACKBONE, max_layer: Optional[int] = None, device: Optional[str] = None):
        import torch
        from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

        self.torch = torch
        self.device = device or torch_device()
        try:
            self.fe = Wav2Vec2FeatureExtractor.from_pretrained(repo, cache_dir=HF_CACHE)
        except OSError:
            self.fe = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=SR, padding_value=0.0,
                                               do_normalize=True, return_attention_mask=True)
        model = Wav2Vec2Model.from_pretrained(repo, cache_dir=HF_CACHE)
        n = len(model.encoder.layers)
        if max_layer is not None and max_layer < n:
            model.encoder.layers = model.encoder.layers[: max_layer + 1]
        self.dtype = torch.float16 if _half_ok(self.device) else torch.float32
        self.model = model.eval().to(self.device, self.dtype)
        self.repo = repo
        self._lock = threading.Lock()

    def features(self, segments_16k: Sequence[np.ndarray], layers: Sequence[int]) -> np.ndarray:
        torch = self.torch
        if not len(segments_16k):
            return np.zeros((0, len(layers), self.model.config.hidden_size), np.float32)

        def run():
            out = []
            for seg in segments_16k:
                inp = self.fe(np.asarray(seg, np.float32), sampling_rate=SR, return_tensors="pt")
                x = inp["input_values"].to(self.device, self.dtype)
                hs = self.model(x, output_hidden_states=True).hidden_states
                out.append(torch.stack([hs[l][0].float().mean(0) for l in layers]))
            return torch.stack(out)

        with self._lock, torch.inference_mode():
            res = _finite_or_fp32(self, run)
            wait_gpu(torch, self.device)
            return res.cpu().numpy()

def clip_preprocess(crop_rgb: np.ndarray, size: int = 224) -> np.ndarray:
    h, w = crop_rgb.shape[:2]
    s = max(h, w)
    canvas = np.zeros((s, s, 3), np.uint8)
    canvas[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = crop_rgb
    img = cv2.resize(canvas, (size, size), interpolation=cv2.INTER_CUBIC if s < size else cv2.INTER_AREA)
    x = (img.astype(np.float32) / 255.0 - CLIP_MEAN) / CLIP_STD
    return x.transpose(2, 0, 1)

class FaceBackbone:

    def __init__(self, repo: str = FACE_BACKBONE, max_layer: Optional[int] = None, device: Optional[str] = None):
        import torch
        from transformers import CLIPVisionModel

        self.torch = torch
        self.device = device or torch_device()
        model = CLIPVisionModel.from_pretrained(repo, cache_dir=HF_CACHE)
        enc = model.vision_model.encoder
        if max_layer is not None and max_layer < len(enc.layers):
            enc.layers = enc.layers[:max_layer]
        self.dtype = torch.float16 if _half_ok(self.device) else torch.float32
        self.model = model.eval().to(self.device, self.dtype)
        self.size = int(model.config.image_size)
        self.repo = repo
        self._lock = threading.Lock()

    def features(self, crops_rgb: Sequence[np.ndarray], layers: Sequence[int], batch: int = 16) -> np.ndarray:
        torch = self.torch
        if not len(crops_rgb):
            return np.zeros((0, len(layers), self.model.config.hidden_size), np.float32)
        pix = np.stack([clip_preprocess(c, self.size) for c in crops_rgb])

        def run():
            out = []
            for i in range(0, len(pix), batch):
                x = torch.from_numpy(pix[i:i + batch]).to(self.device, self.dtype)
                hs = self.model(pixel_values=x, output_hidden_states=True).hidden_states
                out.append(torch.stack([hs[l][:, 0].float() for l in layers], 1))
            return torch.cat(out)

        with self._lock, torch.inference_mode():
            res = _finite_or_fp32(self, run)
            wait_gpu(torch, self.device)
            return res.cpu().numpy()

class _CustomDetector:
    kind = ""
    _instance = None
    _init_lock: threading.Lock
    status = "not loaded"

    @classmethod
    def get(cls, wait: bool = True):
        if cls._instance is not None:
            return cls._instance
        if not enabled():
            cls.status = "off"
            return None
        if not head_path(cls.kind).exists():
            cls.status = "not trained"
            return None
        if not wait and cls._init_lock.locked():
            return None
        with cls._init_lock:
            if cls._instance is None and not cls.status.startswith("unavailable"):
                try:
                    cls.status = "loading"
                    from . import LOAD_LOCK
                    with LOAD_LOCK:
                        cls._instance = cls()
                    cls.status = "ready"
                except Exception as e:
                    cls.status = f"unavailable: {e}"
                    log.warning("custom %s detector unavailable: %s", cls.kind, e)
        return cls._instance

    @classmethod
    def warmup_async(cls):
        if enabled() and head_path(cls.kind).exists():
            threading.Thread(target=cls.get, name=f"fs-custom-{cls.kind}-load", daemon=True).start()

    @classmethod
    def ensure_loading(cls):
        det = cls.get(wait=False)
        if det is None and cls.status == "not loaded" and head_path(cls.kind).exists() and enabled():
            cls.warmup_async()
        return det

class CustomVoiceDetector(_CustomDetector):
    kind = "voice"
    _instance: Optional["CustomVoiceDetector"] = None
    _init_lock = threading.Lock()
    status = "not loaded"

    def __init__(self):
        self.head = Head.load(head_path("voice"))
        self.backbone = VoiceBackbone(self.head.backbone, max_layer=self.head.layer)
        self.name = f"{self.head.label} · XLS-R layer {self.head.layer} + {self.head.kind} · {self.backbone.device}"

    def predict(self, segments_16k: List[np.ndarray]) -> np.ndarray:
        if not segments_16k:
            return np.zeros(0, np.float32)
        ch = self.head.meta.get("channel", "none")
        segs = [normalize_channel(s, ch) for s in segments_16k]
        X = self.backbone.features(segs, [self.head.layer])[:, 0]
        return self.head.predict_proba(X)

class CustomFaceDetector(_CustomDetector):
    kind = "face"
    _instance: Optional["CustomFaceDetector"] = None
    _init_lock = threading.Lock()
    status = "not loaded"

    def __init__(self):
        self.head = Head.load(head_path("face"))
        self.backbone = FaceBackbone(self.head.backbone, max_layer=self.head.layer)
        self.name = f"{self.head.label} · CLIP layer {self.head.layer} + {self.head.kind} · {self.backbone.device}"

    def predict(self, crops_rgb: List[np.ndarray]) -> np.ndarray:
        if not crops_rgb:
            return np.zeros(0, np.float32)
        X = self.backbone.features(crops_rgb, [self.head.layer])[:, 0]
        return self.head.predict_proba(X)

def trust(head: Head) -> float:
    auc = head.auc
    if auc is None:
        return 0.3
    return float(np.clip((auc - 0.5) / 0.4, 0.0, 1.0))

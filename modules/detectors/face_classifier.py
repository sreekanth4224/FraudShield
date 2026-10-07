from __future__ import annotations

import logging
import os
import threading
import time
from typing import Dict, List, Optional

import cv2
import numpy as np

from . import MODELS_DIR, torch_device, wait_gpu

log = logging.getLogger("fraudshield.face_model")

CHECKPOINTS = {
    "b0-ff++": ("KoreaPeter/ms-eff-gcvit-deepfake-b0-ff-plus-plus", "b0"),
    "b0-celeb": ("KoreaPeter/ms-eff-gcvit-deepfake-b0-celeb-df-v2", "b0"),
    "b5-ff++": ("KoreaPeter/ms-eff-gcvit-deepfake-b5-ff-plus-plus", "b5"),
    "b5-celeb": ("KoreaPeter/ms-eff-gcvit-deepfake-b5-celeb-df-v2", "b5"),
    "gend-clip": ("HoopitAI/video-deepfake-detection-GenD_CLIP_L_14_FF", "gend"),
}
DEFAULT_SET = "b0-ff++,b0-celeb"
DEFAULT_SET_GPU = "b0-ff++,b0-celeb,gend-clip"
GEND_EVERY_S = 0.25
CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], np.float32)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], np.float32)

def gend_crop(crop_rgb: np.ndarray) -> np.ndarray:
    h, w = crop_rgb.shape[:2]
    side = max(2, int(round(max(h, w) * 1.3 / 1.4)))
    cy, cx = h / 2.0, w / 2.0
    y0, x0 = int(round(cy - side / 2)), int(round(cx - side / 2))
    out = np.zeros((side, side, 3), np.uint8)
    sy0, sx0 = max(0, y0), max(0, x0)
    sy1, sx1 = min(h, y0 + side), min(w, x0 + side)
    out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = crop_rgb[sy0:sy1, sx0:sx1]
    return out

def clip_preprocess(img_rgb: np.ndarray, size: int = 224) -> np.ndarray:
    h, w = img_rgb.shape[:2]
    s = size / min(h, w)
    img = cv2.resize(img_rgb, (max(size, round(w * s)), max(size, round(h * s))), interpolation=cv2.INTER_CUBIC)
    h, w = img.shape[:2]
    y0, x0 = (h - size) // 2, (w - size) // 2
    img = img[y0:y0 + size, x0:x0 + size]
    x = (img.astype(np.float32) / 255.0 - CLIP_MEAN) / CLIP_STD
    return x.transpose(2, 0, 1)
VARIANTS = {
    "b0": dict(
        model_name="tf_efficientnet_b0.ns_jft_in1k", img_size=[224, 224],
        l_dim=24, h_dim=256, l_depths=[2, 2, 4, 2], h_depths=[4],
        l_windows=[7, 7, 14, 7], h_windows=[7], l_heads=[1, 2, 4, 8], h_heads=[4],
        l_ratio=[4, 4, 4, 4], h_ratio=[4], h_drop=0.05, l_attn_drop=0.05,
        l_drop_path=0.1, h_drop_path=0.05),
    "b5": dict(
        model_name="tf_efficientnet_b5.ns_jft_in1k", img_size=[384, 384],
        l_dim=48, h_dim=512, l_depths=[2, 2, 6, 2], h_depths=[6],
        l_windows=[12, 12, 24, 12], h_windows=[12], l_heads=[2, 4, 8, 16], h_heads=[16],
        l_ratio=[3, 3, 3, 3], h_ratio=[3], h_drop=0.1, l_attn_drop=0.1,
        l_drop_path=0.15, h_drop_path=0.1),
}
MARGIN = 0.2
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)

def face_crop(frame_bgr: np.ndarray, box, margin: float = MARGIN) -> Optional[np.ndarray]:
    H, W = frame_bgr.shape[:2]
    x, y, w, h = box
    pw, ph = int(w * margin), int(h * margin)
    x0, y0 = max(int(x - pw), 0), max(int(y - ph), 0)
    x1, y1 = min(int(x + w + pw), W), min(int(y + h + ph), H)
    if x1 - x0 < 16 or y1 - y0 < 16:
        return None
    return cv2.cvtColor(frame_bgr[y0:y1, x0:x1], cv2.COLOR_BGR2RGB)

def effective_scale(crop_rgb: np.ndarray) -> float:

    g = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
    h, w = g.shape
    if min(h, w) < 24:
        return 1.0
    energy = lambda im: float(np.mean(np.abs(cv2.Laplacian(im, cv2.CV_32F)))) + 1e-6
    half = cv2.resize(g, (max(8, w // 2), max(8, h // 2)), interpolation=cv2.INTER_AREA)
    kept = energy(cv2.resize(half, (w, h), interpolation=cv2.INTER_LINEAR)) / energy(g)
    return float(np.clip((1.0 - kept) / 0.45, 0.25, 1.0))

def preprocess(crop_rgb: np.ndarray, size: int = 224) -> np.ndarray:
    h, w = crop_rgb.shape[:2]
    s = size / max(h, w)
    nh, nw = max(1, round(h * s)), max(1, round(w * s))
    img = cv2.resize(crop_rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas = np.zeros((size, size, 3), np.uint8)
    canvas[top:top + nh, left:left + nw] = img
    x = (canvas.astype(np.float32) / 255.0 - MEAN) / STD
    return x.transpose(2, 0, 1)

class FaceDeepfakeClassifier:

    _instance: Optional["FaceDeepfakeClassifier"] = None
    _init_lock = threading.Lock()
    status = "not loaded"

    @classmethod
    def get(cls, wait: bool = True) -> Optional["FaceDeepfakeClassifier"]:
        if cls._instance is not None:
            return cls._instance
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
                    log.warning("face deepfake model unavailable: %s", e)
        return cls._instance

    @classmethod
    def warmup_async(cls):
        threading.Thread(target=cls.get, name="fs-face-model-load", daemon=True).start()

    def __init__(self, names: Optional[List[str]] = None):
        import torch
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        from .deepguard import MultiScaleEffGCViT

        self.torch = torch
        self.device = torch_device()
        torch.set_num_threads(2)
        default = DEFAULT_SET_GPU if self.device == "cuda" else DEFAULT_SET
        names = names or [n.strip() for n in os.environ.get("FRAUDSHIELD_FACE_MODELS", default).split(",") if n.strip()]
        self.models: Dict[str, "torch.nn.Module"] = {}
        self.sizes: Dict[str, int] = {}
        self.gend: Dict[str, "torch.nn.Module"] = {}
        self.half = self.device == "cuda" and os.environ.get("FRAUDSHIELD_FP32", "0") != "1"
        for name in names:
            repo, variant = CHECKPOINTS[name]
            if variant == "gend":
                try:
                    self.gend[name] = self._load_gend(repo)
                except Exception as e:
                    log.warning("%s unavailable, continuing without it: %s", name, e)
                continue
            local = MODELS_DIR / repo.split("/")[1] / "model.safetensors"
            path = local if local.exists() else hf_hub_download(repo, "model.safetensors", local_dir=local.parent)
            state = {k.removeprefix("model."): v for k, v in load_file(path).items()}
            net = MultiScaleEffGCViT(**VARIANTS[variant])
            missing, unexpected = net.load_state_dict(state, strict=False)
            if missing or unexpected:
                raise RuntimeError(f"{name}: checkpoint mismatch ({len(missing)} missing, {len(unexpected)} unexpected keys)")
            self.models[name] = net.eval().to(self.device)
            self.sizes[name] = VARIANTS[variant]["img_size"][0]
        if not self.models and not self.gend:
            raise RuntimeError("no face checkpoint could be loaded")
        self._lock = threading.Lock()
        self._gend_last: Dict[str, tuple] = {}
        self.names = list(self.models) + list(self.gend)
        self._graph = None
        self.predict([np.zeros((64, 64, 3), np.uint8)])
        if self.device == "cuda" and os.environ.get("FRAUDSHIELD_CUDA_GRAPHS", "1") != "0":
            try:
                self._build_graph()
            except Exception as e:
                self._graph = None
                log.warning("CUDA graph capture failed, using eager mode: %s", e)
        self.name = (f"face ensemble ({' + '.join(self.names)}) · {self.device}"
                     + (" · CUDA graph" if self._graph is not None else ""))

    def _forward(self, batches):
        torch = self.torch
        return torch.stack([torch.sigmoid(net(batches[self.sizes[name]])).float()[:, 0]
                            for name, net in self.models.items()], 1)

    def _build_graph(self):

        torch = self.torch
        self._static_in = {sz: torch.zeros(1, 3, sz, sz, device=self.device) for sz in set(self.sizes.values())}
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.no_grad(), torch.cuda.stream(side):
            for _ in range(3):
                self._forward(self._static_in)
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.no_grad(), torch.cuda.graph(graph):
            self._static_out = self._forward(self._static_in)
        self._graph = graph

    def _load_gend(self, repo):
        import torch
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        from transformers import AutoConfig
        from transformers.dynamic_module_utils import get_class_from_dynamic_module
        cache = MODELS_DIR / "hf-cache"
        cfg = AutoConfig.from_pretrained(repo, trust_remote_code=True, cache_dir=cache)
        cls = get_class_from_dynamic_module("modeling_gend.GenD", repo, cache_dir=cache)
        if hasattr(torch, "get_default_device") and str(torch.get_default_device()) == "meta":
            torch.set_default_device("cpu")
        model = cls(cfg)
        state = load_file(hf_hub_download(repo, "model.safetensors", cache_dir=cache))
        own = model.state_dict()
        for pre in ("", "model.", "module."):
            mapped = {k[len(pre):]: v for k, v in state.items() if k.startswith(pre)}
            if sum(k in own for k in mapped) >= 0.9 * len(own):
                state = mapped
                break
        missing, unexpected = model.load_state_dict(state, strict=False)
        loaded = len(own) - len(missing)
        if loaded < 0.9 * len(own):
            raise RuntimeError(f"GenD weights didn't match the model ({loaded}/{len(own)} tensors loaded)")
        if missing:
            log.info("GenD: %d tensors not in the checkpoint (kept from the CLIP backbone): %s",
                     len(missing), ", ".join(list(missing)[:5]))
        model = model.eval().to(self.device)
        if self.half:
            model = model.half()
        return model

    def _gend_probs(self, model, crops_rgb) -> np.ndarray:
        torch = self.torch
        faces = [clip_preprocess(gend_crop(c)) for c in crops_rgb]
        x = np.stack(faces + [f[:, :, ::-1].copy() for f in faces])
        x = torch.from_numpy(x).to(self.device)
        if self.half:
            x = x.half()
        with torch.inference_mode():
            out = model(x)
            logits = out.logits if hasattr(out, "logits") else (out["logits"] if isinstance(out, dict) else out)
            logits = logits.float().reshape(2 * len(crops_rgb), -1)
            if not bool(torch.isfinite(logits).all()):
                model.float()
                logits = model(x.float())
                logits = (logits.logits if hasattr(logits, "logits") else logits).float().reshape(2 * len(crops_rgb), -1)
                model.half()
            p = torch.softmax(logits, -1)[:, 1]
            p = 0.5 * (p[:len(crops_rgb)] + p[len(crops_rgb):])
            wait_gpu(torch, self.device)
            return p.cpu().numpy()

    def predict(self, crops_rgb: List[np.ndarray]) -> np.ndarray:
        if not crops_rgb:
            return np.zeros((0, len(self.models) + len(self.gend)), np.float32)
        cols = [self._predict_ms(crops_rgb)] if self.models else []
        with self._lock:
            now = time.monotonic()
            live = len(crops_rgb) == 1
            for name, model in self.gend.items():
                last = self._gend_last.get(name)
                if live and last is not None and now - last[0] < GEND_EVERY_S:
                    p = np.array([last[1]], np.float32)
                else:
                    p = self._gend_probs(model, crops_rgb)
                    if live:
                        self._gend_last[name] = (now, float(p[0]))
                cols.append(p[:, None])
        return np.hstack(cols).astype(np.float32)

    def _predict_ms(self, crops_rgb: List[np.ndarray]) -> np.ndarray:
        torch = self.torch
        with self._lock:
            if self._graph is not None and len(crops_rgb) == 1:
                for sz, buf in self._static_in.items():
                    buf.copy_(torch.from_numpy(preprocess(crops_rgb[0], sz)[None]))
                self._graph.replay()
                wait_gpu(torch, self.device)
                return self._static_out.cpu().numpy()
            batches = {sz: torch.from_numpy(np.stack([preprocess(c, sz) for c in crops_rgb])).to(self.device)
                       for sz in set(self.sizes.values())}
            with torch.inference_mode():
                out = self._forward(batches)
                wait_gpu(torch, self.device)
                return out.cpu().numpy()

class AsyncFaceScorer:

    CUSTOM_EVERY_S = {"cuda": 0.5, "cpu": 1.5}

    def __init__(self, on_result, on_custom=None):
        self.on_result = on_result
        self.on_custom = on_custom
        self._last_custom = -1e9
        self._slot = None
        self._cv = threading.Condition()
        self._stop = False
        self._thread = threading.Thread(target=self._run, name="fs-face-model", daemon=True)
        self._thread.start()

    def submit(self, t: float, crop_rgb: np.ndarray, meta: dict):
        with self._cv:
            self._slot = (t, crop_rgb, meta)
            self._cv.notify()

    def _run(self):
        clf = FaceDeepfakeClassifier.get()
        while True:
            with self._cv:
                while self._slot is None and not self._stop:
                    self._cv.wait()
                if self._stop:
                    return
                t, crop, meta = self._slot
                self._slot = None
            if clf is not None:
                t0 = time.perf_counter()
                p = clf.predict([crop])[0]
                self.on_result(t, p, {**meta, "ms": 1000 * (time.perf_counter() - t0)})
            self._score_custom(t, crop, meta)

    def _score_custom(self, t, crop, meta):
        if self.on_custom is None:
            return
        from .custom_heads import CustomFaceDetector
        det = CustomFaceDetector.ensure_loading()
        if det is None or t - self._last_custom < self.CUSTOM_EVERY_S.get(det.backbone.device, 1.5):
            return
        self._last_custom = t
        try:
            self.on_custom(t, float(det.predict([crop])[0]), meta)
        except Exception as e:
            log.warning("custom face head failed: %s", e)

    def close(self):
        with self._cv:
            self._stop = True
            self._cv.notify()

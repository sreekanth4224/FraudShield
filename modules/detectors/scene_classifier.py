from __future__ import annotations

import logging
import os
import threading
from typing import List, Optional

import cv2
import numpy as np

from . import MODELS_DIR, torch_device, wait_gpu

log = logging.getLogger("fraudshield.scene_model")

REPO = "buildborderless/CommunityForensics-DeepfakeDet-ViT"
SIZE, RESIZE = 384, 440
CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], np.float32)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], np.float32)
SCENE_EVERY_S = 0.5
CONTEXT = 2.2

def enabled() -> bool:
    return os.environ.get("FRAUDSHIELD_SCENE", "1") != "0"

def scene_crop(frame_bgr: np.ndarray, box) -> Optional[np.ndarray]:
    H, W = frame_bgr.shape[:2]
    x, y, w, h = box
    side = CONTEXT * max(w, h)
    cx, cy = x + w / 2, y + h / 2 + 0.12 * h
    x0, y0 = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
    x1, y1 = int(min(W, cx + side / 2)), int(min(H, cy + side / 2))
    if x1 - x0 < 64 or y1 - y0 < 64:
        return None
    return cv2.cvtColor(frame_bgr[y0:y1, x0:x1], cv2.COLOR_BGR2RGB)

def preprocess(img_rgb: np.ndarray) -> np.ndarray:
    h, w = img_rgb.shape[:2]
    s = RESIZE / min(h, w)
    img = cv2.resize(img_rgb, (max(SIZE, round(w * s)), max(SIZE, round(h * s))), interpolation=cv2.INTER_CUBIC)
    h, w = img.shape[:2]
    y0, x0 = (h - SIZE) // 2, (w - SIZE) // 2
    img = img[y0:y0 + SIZE, x0:x0 + SIZE]
    return ((img.astype(np.float32) / 255.0 - CLIP_MEAN) / CLIP_STD).transpose(2, 0, 1)

class SceneDeepfakeClassifier:
    _instance: Optional["SceneDeepfakeClassifier"] = None
    _init_lock = threading.Lock()
    status = "not loaded"

    @classmethod
    def get(cls, wait: bool = True) -> Optional["SceneDeepfakeClassifier"]:
        if cls._instance is not None:
            return cls._instance
        if not enabled():
            cls.status = "off"
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
                    log.warning("AI-generated video detector unavailable: %s", e)
        return cls._instance

    @classmethod
    def warmup_async(cls):
        if enabled():
            threading.Thread(target=cls.get, name="fs-scene-model-load", daemon=True).start()

    def __init__(self):
        import torch
        from transformers import ViTForImageClassification
        self.torch = torch
        self.device = torch_device()
        model = ViTForImageClassification.from_pretrained(REPO, cache_dir=MODELS_DIR / "hf-cache")
        self.half = self.device == "cuda" and os.environ.get("FRAUDSHIELD_FP32", "0") != "1"
        self.model = model.eval().to(self.device)
        if self.half:
            self.model = self.model.half()
        n = int(model.config.num_labels)
        labels = {int(k): str(v).lower() for k, v in (model.config.id2label or {}).items()}
        self.fake_idx = next((i for i, v in labels.items() if any(w in v for w in ("fake", "ai", "generated", "synthetic"))),
                             n - 1)
        self.n_labels = n
        self._lock = threading.Lock()
        self.name = f"Community Forensics ViT (AI-generated images, 4,800 generators) · {self.device}"

    def predict(self, crops_rgb: List[np.ndarray]) -> np.ndarray:
        if not crops_rgb:
            return np.zeros(0, np.float32)
        torch = self.torch
        x = torch.from_numpy(np.stack([preprocess(c) for c in crops_rgb])).to(self.device)
        if self.half:
            x = x.half()
        with self._lock, torch.inference_mode():
            logits = self.model(pixel_values=x).logits.float()
            if not bool(torch.isfinite(logits).all()):
                self.model.float()
                logits = self.model(pixel_values=x.float()).logits.float()
                self.model.half()
            p = torch.sigmoid(logits[:, 0]) if self.n_labels == 1 else torch.softmax(logits, -1)[:, self.fake_idx]
            wait_gpu(torch, self.device)
            return p.cpu().numpy().astype(np.float32)

class AsyncSceneScorer:

    def __init__(self, on_result):
        self.on_result = on_result
        self._slot = None
        self._cv = threading.Condition()
        self._stop = False
        threading.Thread(target=self._run, name="fs-scene-model", daemon=True).start()

    def submit(self, t, crop):
        with self._cv:
            self._slot = (t, crop)
            self._cv.notify()

    def _run(self):
        clf = SceneDeepfakeClassifier.get()
        while True:
            with self._cv:
                while self._slot is None and not self._stop:
                    self._cv.wait()
                if self._stop:
                    return
                t, crop = self._slot
                self._slot = None
            if clf is not None:
                try:
                    self.on_result(t, float(clf.predict([crop])[0]))
                except Exception as e:
                    log.warning("scene detector failed: %s", e)

    def close(self):
        with self._cv:
            self._stop = True
            self._cv.notify()

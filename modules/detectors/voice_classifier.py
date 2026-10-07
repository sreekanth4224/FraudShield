from __future__ import annotations

import logging
import os
import threading
from typing import List, Optional

import numpy as np

from . import MODELS_DIR, torch_device, wait_gpu

log = logging.getLogger("fraudshield.voice_model")

REPO = "abhishtagatya/wav2vec2-base-960h-itw-deepfake"
ARENA = {
    "arena-500m": "Speech-Arena-2025/DF_Arena_500M_V_1",
    "arena-1b": "Speech-Arena-2025/DF_Arena_1B_V_1",
    "arena-100m": "Speech-Arena-2025/DF_Arena_100M_V_1",
}
ARENA_LEN = 64600
MAX_WINDOWS = int(os.environ.get("FRAUDSHIELD_VOICE_WINDOWS", "1"))
SR = 16000

def fill_digital_silence(seg: np.ndarray) -> np.ndarray:
    x = np.asarray(seg, np.float32)
    if len(x) < 160:
        return x
    rms = float(np.sqrt(np.mean(x ** 2))) + 1e-9
    hop = 160
    nb = len(x) // hop
    peaks = np.abs(x[: nb * hop]).reshape(nb, hop).max(1)
    gated_b = peaks < rms * 10 ** (-60 / 20)
    if gated_b.mean() < 0.02:
        return x
    gated = np.zeros(len(x), bool)
    gated[: nb * hop] = np.repeat(gated_b, hop)
    rng = np.random.default_rng(len(x))
    noise = rng.standard_normal(len(x)).astype(np.float32) * rms * 10 ** (-50 / 20)
    return np.where(gated, x + noise, x).astype(np.float32)

def auto_backends() -> List[str]:
    try:
        import torch
        if torch.cuda.is_available():
            gb = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
            return ["arena-500m", "arena-1b"] if gb >= 5.5 else ["arena-1b"] if gb >= 3.5 else ["arena-500m"]
    except Exception:
        pass
    return ["arena-500m"]

CHOICE_FILE = MODELS_DIR / "voice_choice.json"

def chosen_backend() -> Optional[str]:
    try:
        import json
        return str(json.loads(CHOICE_FILE.read_text(encoding="utf-8"))["backend"])
    except Exception:
        return None

def backend_names() -> List[str]:
    raw = os.environ.get("FRAUDSHIELD_VOICE_MODEL", "").lower().strip()
    if raw == "auto":
        return auto_backends()
    if not raw:
        raw = chosen_backend() or "itw"
    names = [b.strip() for b in raw.split(",") if b.strip() in ARENA or b.strip() == "itw"]
    return names or ["itw"]

def backend_name() -> str:
    return ",".join(backend_names())

class VoiceDeepfakeClassifier:
    _instance: Optional["VoiceDeepfakeClassifier"] = None
    _init_lock = threading.Lock()
    status = "not loaded"

    @classmethod
    def get(cls, wait: bool = True) -> Optional["VoiceDeepfakeClassifier"]:
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
                    log.warning("voice deepfake model unavailable: %s", e)
        return cls._instance

    @classmethod
    def warmup_async(cls):
        threading.Thread(target=cls.get, name="fs-voice-model-load", daemon=True).start()

    def __init__(self, backend: Optional[str] = None):
        import torch
        self.torch = torch
        self.device = torch_device()
        self._lock = threading.Lock()
        self.half = self.device == "cuda" and os.environ.get("FRAUDSHIELD_FP32", "0") != "1"
        wanted = [b.strip() for b in backend.split(",")] if backend else backend_names()
        self.members = []
        self.failed = {}
        for b in wanted:
            if b in ARENA:
                try:
                    self.members.append((b,) + self._load_arena(ARENA[b]))
                except Exception as e:
                    log.warning("%s unavailable (%s)", b, e)
                    self.failed[b] = str(e)
        if self.members:
            self.kind = "arena"
            self.backend = ",".join(m[0] for m in self.members)
            sizes = " + ".join(m[0].split("-")[1].upper() for m in self.members)
            self.name = (f"DF-Arena {sizes}{' ensemble' if len(self.members) > 1 else ''} "
                         f"(XLS-R + Conformer, multi-corpus) · {self.device}{' fp16' if self.half else ''}")
            if "itw" in wanted:
                self._init_itw()
                self.kind = "arena"
                self.backend += ",itw"
                self.name = self.name.replace(") ·", ") + wav2vec2-ITW ·")
            return
        if self.failed:
            log.warning("no DF-Arena model loaded — falling back to the In-the-Wild wav2vec2 model")
            self.fallback_reason = "; ".join(f"{k}: {v}" for k, v in self.failed.items())
        self.backend = "itw"
        self._init_itw()
        self.kind = "itw"

    def _load_arena(self, repo):
        import einops
        from transformers import AutoModel
        model = AutoModel.from_pretrained(repo, trust_remote_code=True, cache_dir=MODELS_DIR / "hf-cache")
        model = model.eval().to(self.device)
        if self.half:
            model = model.half()
        labels = {v.lower(): int(k) for k, v in model.config.id2label.items()}
        return model, labels.get("spoof", 0)

    def _init_itw(self):
        from huggingface_hub import snapshot_download
        from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification

        path = MODELS_DIR / REPO.split("/")[1]
        if not (path / "model.safetensors").exists() or not (path / "config.json").exists():
            path = snapshot_download(REPO, local_dir=path, allow_patterns=["*.json", "*.safetensors"])
        self.fe = Wav2Vec2FeatureExtractor.from_pretrained(path)
        self.model = Wav2Vec2ForSequenceClassification.from_pretrained(path).eval().to(self.device)
        labels = {v.lower(): int(k) for k, v in self.model.config.id2label.items()}
        fake = [i for name, i in labels.items() if any(w in name for w in ("fake", "spoof"))]
        if len(fake) != 1:
            raise RuntimeError(f"can't tell which output is 'fake' in {self.model.config.id2label}")
        self.fake_idx = fake[0]
        self.itw_name = f"wav2vec2-base · In-the-Wild · {self.device}"
        if not getattr(self, "members", None):
            self.name = self.itw_name

    @staticmethod
    def _windows(seg: np.ndarray) -> List[np.ndarray]:
        seg = np.asarray(seg, np.float32)
        seg = seg - float(np.mean(seg)) if len(seg) else seg
        peak = float(np.max(np.abs(seg))) if len(seg) else 0.0
        if 1e-4 < peak < 0.05:
            seg = seg * (0.5 / peak)
        L = ARENA_LEN
        if len(seg) <= L:
            reps = L // max(1, len(seg)) + 1
            return [np.tile(seg, reps)[:L]]
        hop = 160
        e = np.convolve(seg.astype(np.float64) ** 2, np.ones(hop), "valid")[::hop]
        csum = np.concatenate([[0.0], np.cumsum(e)])
        n_win = L // hop
        starts = np.arange(0, max(1, len(e) - n_win + 1))
        energy = csum[np.minimum(starts + n_win, len(e))] - csum[starts]
        s1 = int(starts[np.argmax(energy)])
        out = [seg[s1 * hop: s1 * hop + L]]
        ok = np.abs(starts - s1) >= int(0.75 * n_win)
        if MAX_WINDOWS > 1 and ok.any():
            s2 = int(starts[ok][np.argmax(energy[ok])])
            if energy[starts == s2][0] > 0.25 * energy[starts == s1][0]:
                out.append(seg[s2 * hop: s2 * hop + L])
        return [w if len(w) == L else np.pad(w, (0, L - len(w))) for w in out]

    def _arena_logodds(self, model, fake_idx, wav: np.ndarray) -> float:
        torch = self.torch
        x = torch.from_numpy(wav).to(self.device)
        if self.half:
            logits = model(input_values=x.half())["logits"].float()
            if not bool(torch.isfinite(logits).all()):
                model.float()
                logits = model(input_values=x)["logits"].float()
                model.half()
        else:
            logits = model(input_values=x)["logits"].float()
        lp = torch.log_softmax(logits.reshape(-1, logits.shape[-1]), -1)[0]
        return float(lp[fake_idx] - torch.logsumexp(torch.cat([lp[:fake_idx], lp[fake_idx + 1:]]), 0))

    def _itw_logodds(self, segments_16k) -> np.ndarray:
        torch = self.torch
        inp = self.fe([s.astype(np.float32) for s in segments_16k], sampling_rate=SR,
                      return_tensors="pt", padding=True)
        logits = self.model(**{k: v.to(self.device) for k, v in inp.items()}).logits.float()
        lp = torch.log_softmax(logits, -1)
        p = lp[:, self.fake_idx].exp().clamp(1e-6, 1 - 1e-6)
        return (torch.log(p) - torch.log1p(-p)).cpu().numpy()

    def predict(self, segments_16k: List[np.ndarray]) -> np.ndarray:
        if not segments_16k:
            return np.zeros(0, np.float32)
        segments_16k = [fill_digital_silence(s) for s in segments_16k]
        torch = self.torch
        with self._lock, torch.inference_mode():
            if self.kind == "arena":
                z = np.zeros((len(segments_16k), 0))
                for _, model, fake_idx in self.members:
                    col = [np.mean([self._arena_logodds(model, fake_idx, w) for w in self._windows(seg)])
                           for seg in segments_16k]
                    z = np.column_stack([z, col])
                if getattr(self, "fe", None) is not None:
                    z = np.column_stack([z, self._itw_logodds(segments_16k)])
                zm = z.mean(1)
            else:
                zm = self._itw_logodds(segments_16k)
            wait_gpu(torch, self.device)
            return (1.0 / (1.0 + np.exp(-np.clip(zm, -30, 30)))).astype(np.float32)

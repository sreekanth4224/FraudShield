from __future__ import annotations

import json
import math
from typing import Dict, Sequence

from .detectors import CALIBRATION_FILE

DEFAULTS: Dict[str, dict] = {
    "face_model": {"w": [0.394, 0.323], "b": 0.609},
    "voice_model": {"w": [1.0], "b": 0.0},
    "face_model[b0-ff++,b0-celeb,gend-clip]": {"w": [0.24, 0.19, 1.0], "b": 0.35},
}
DEFAULT_FACE_SET = ["b0-ff++", "b0-celeb"]
_cache = None

def key(name: str, models=None) -> str:
    if not models or (name == "face_model" and list(models) == DEFAULT_FACE_SET):
        return name
    return f"{name}[{','.join(models)}]"

def params() -> Dict[str, dict]:
    global _cache
    if _cache is None:
        _cache = {k: dict(v) for k, v in DEFAULTS.items()}
        try:
            data = json.loads(CALIBRATION_FILE.read_text(encoding="utf-8"))
            for k, v in data.items():
                if k.split("[")[0] in DEFAULTS and "w" in v and "b" in v:
                    _cache[k] = {**v, "w": [float(x) for x in v["w"]], "b": float(v["b"])}
        except (OSError, ValueError):
            pass
    return _cache

def reload():
    global _cache
    _cache = None
    return params()

def logit(p: float) -> float:
    p = min(1 - 1e-4, max(1e-4, float(p)))
    return math.log(p / (1 - p))

def risk(name: str, probs: Sequence[float], models=None) -> float:
    c = params().get(key(name, models))
    if c is None or len(c["w"]) != len(probs):
        c = {"w": [1.0 / len(probs)] * len(probs), "b": 0.0}
    z = sum(w * logit(p) for w, p in zip(c["w"], probs)) + c["b"]
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))

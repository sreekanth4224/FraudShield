from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "data" / "evidence" / "evidence.jsonl"
MODEL = ROOT / "models" / "evidence_model.json"
LEARNED_WEIGHT = 0.25
MIN_REL = 0.2

def features(mods: Dict[str, dict]) -> Dict[str, float]:
    out = {}
    for m, mod in mods.items():
        sc = mod.get("score")
        out[f"{m}.score"] = float("nan") if sc is None else float(sc) / 100.0
        for s in mod.get("signals", []) or []:
            ok = (s.get("reliability") or 0.0) >= MIN_REL
            out[f"{m}.{s['key']}"] = float(s["risk"]) if ok else float("nan")
    return out

def log_example(path: str, label: str, mods: Dict[str, dict], verdict: Optional[float]):
    LOG.parent.mkdir(parents=True, exist_ok=True)
    row = {"file": Path(path).name, "label": label, "verdict": verdict, "t": time.strftime("%Y-%m-%d %H:%M"),
           "features": {k: (None if math.isnan(v) else round(v, 4)) for k, v in features(mods).items()}}
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")

_cache = {"mtime": None, "model": None}

def load_model() -> Optional[dict]:
    try:
        mt = MODEL.stat().st_mtime
    except OSError:
        return None
    if _cache["mtime"] != mt:
        try:
            _cache["model"] = json.loads(MODEL.read_text(encoding="utf-8"))
        except Exception:
            _cache["model"] = None
        _cache["mtime"] = mt
    m = _cache["model"]
    return m if m and m.get("trusted") else None

def vectorize(feats: Dict[str, float], model: dict) -> np.ndarray:
    x = np.array([feats.get(k, float("nan")) for k in model["keys"]], float)
    fill = np.array(model["fill"], float)
    x = np.where(np.isfinite(x), x, fill)
    return (x - np.array(model["mu"])) / np.array(model["sd"])

def predict(mods: Dict[str, dict]) -> Optional[tuple]:
    m = load_model()
    if m is None:
        return None
    feats = features(mods)
    measured = float(np.mean([np.isfinite(feats.get(k, float("nan"))) for k in m["keys"]]))
    z = float(vectorize(feats, m) @ np.array(m["w"]) + m["b"])
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z)))), measured

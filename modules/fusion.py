from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

MODULE_WEIGHTS = {"face": 0.50, "voice": 0.30, "sync": 0.20}
VOICE_CLEAR_FACTOR = 1.0

def module_weight(k: str, r: "Reading") -> float:
    w = MODULE_WEIGHTS[k]
    if k == "voice":
        t = min(1.0, max(0.0, (r.score - 35.0) / 30.0))
        w *= VOICE_CLEAR_FACTOR + (1.0 - VOICE_CLEAR_FACTOR) * t
    return w

GENUINE_BELOW = 26.0
DEEPFAKE_FROM = 55.0
HYSTERESIS = 4.0
MIN_CONFIDENCE = 0.18
MIN_EVIDENCE_S = 5.0
CORROBORATE_CONF = 0.45

VERDICTS = {
    "idle": ("Idle", "Share your screen to start monitoring the video KYC call."),
    "searching": ("Looking for a face", "Bring the customer's video tile into view — FraudShield is scanning the screen."),
    "calibrating": ("Calibrating", "Collecting evidence — keep the customer on camera and speaking."),
    "genuine": ("Likely genuine", "No deepfake indicators. Continue the KYC."),
    "suspicious": ("Suspicious", "Run a liveness challenge before proceeding."),
    "deepfake": ("Likely deepfake", "Stop the KYC and escalate to the fraud team."),
}

@dataclass
class Reading:
    score: Optional[float]
    confidence: float

def red_floor(active: Dict[str, Reading]) -> float:

    floor = 0.0
    for k, r in active.items():
        if r.score < DEEPFAKE_FROM or r.confidence < 0.5:
            continue
        support = max((min(1.0, max(0.0, (o.score - 40.0) / 25.0))
                       for j, o in active.items() if j != k and o.confidence >= CORROBORATE_CONF), default=0.0)
        base = 0.92 if (k == "sync" and r.confidence >= 0.8) else 0.85
        floor = max(floor, r.score * (base + (1.0 - base) * support) + 10.0 * support)
    return floor

def fuse(readings: Dict[str, Reading], learned: Optional[Tuple[float, float]] = None) -> Tuple[Optional[float], float]:

    active = {k: r for k, r in readings.items() if r.score is not None and r.confidence > 0.05}
    if not active:
        return None, 0.0
    w = {k: module_weight(k, r) * (0.35 + 0.65 * r.confidence) for k, r in active.items()}
    score = sum(w[k] * r.score for k, r in active.items()) / sum(w.values())
    if learned is not None and learned[1] >= 0.5:
        from .evidence import LEARNED_WEIGHT
        a = LEARNED_WEIGHT * learned[1]
        score = (1 - a) * score + a * 100.0 * learned[0]
    score = max(score, red_floor(active))
    conf = sum(MODULE_WEIGHTS[k] * r.confidence for k, r in active.items()) / sum(MODULE_WEIGHTS.values())
    return float(min(100.0, max(0.0, score))), float(min(1.0, conf))

def verdict_confidence(score: Optional[float], readings: Dict[str, Reading]) -> float:

    if score is None:
        return 0.0
    cov_num, cov_den = 0.0, 0.0
    for k, w in MODULE_WEIGHTS.items():
        r = readings.get(k)
        has = r is not None and r.score is not None and r.confidence > 0.05
        c = min(1.0, r.confidence / 0.6) if has else 0.0
        if has or k == "face":
            cov_num += w * c
            cov_den += w
    coverage = cov_num / cov_den if cov_den else 0.0
    if score >= DEEPFAKE_FROM:
        margin = min(1.0, (score - DEEPFAKE_FROM) / 25.0)
    elif score < GENUINE_BELOW:
        margin = min(1.0, (GENUINE_BELOW - score) / 16.0)
    else:
        margin = -0.5 + min(abs(score - GENUINE_BELOW), abs(score - DEEPFAKE_FROM)) / (DEEPFAKE_FROM - GENUINE_BELOW)
    return float(max(0.0, min(0.99, coverage * (0.75 + 0.25 * margin))))

class LiveFusion:
    def __init__(self, tau_up: float = 0.75, tau_down: float = 8.0):
        self.tau_up, self.tau_down = tau_up, tau_down
        self.reset()

    def reset(self):
        self.value: Optional[float] = None
        self.verdict = "idle"
        self._t: Optional[float] = None
        self.evidence_s = 0.0
        self.locked: Optional[float] = None

    def update(self, t: float, raw: Optional[float], conf: float, face_present: bool,
               readings: Optional[Dict[str, Reading]] = None, cap: Optional[float] = None) -> dict:
        dt = 1.0 if self._t is None else max(0.05, min(5.0, t - self._t))
        self._t = t
        if cap is not None and raw is not None:
            raw = min(raw, cap)
        if raw is not None and conf >= MIN_CONFIDENCE:
            self.evidence_s += dt
            if self.value is None:
                self.value = raw
            else:
                tau = self.tau_up if raw > self.value else self.tau_down
                self.value += (1 - math.exp(-dt / tau)) * (raw - self.value)
        if cap is None:
            self.locked = None
        else:
            base = cap if self.value is None else min(self.value, cap)
            self.locked = base if self.locked is None else min(self.locked, cap)
            self.value = self.locked
            self.evidence_s = max(self.evidence_s, MIN_EVIDENCE_S)

        if self.value is None or self.evidence_s < MIN_EVIDENCE_S:
            self.verdict = "calibrating" if face_present or raw is not None else "searching"
        else:
            v, prev = self.value, self.verdict
            lo = GENUINE_BELOW + (HYSTERESIS if prev == "genuine" else -HYSTERESIS if prev == "suspicious" else 0)
            hi = DEEPFAKE_FROM + (-HYSTERESIS if prev == "deepfake" else HYSTERESIS if prev == "suspicious" else 0)
            self.verdict = "genuine" if v < lo else "deepfake" if v >= hi else "suspicious"

        label, action = VERDICTS[self.verdict]
        return {"score": None if self.value is None else round(self.value, 1),
                "raw": None if raw is None else round(raw, 1),
                "confidence": round(verdict_confidence(self.value, readings or {}) if self.verdict in
                                    ("genuine", "suspicious", "deepfake") else 0.0, 2),
                "evidence": round(conf, 2), "verdict": self.verdict, "label": label, "action": action,
                "evidence_s": round(self.evidence_s, 1)}

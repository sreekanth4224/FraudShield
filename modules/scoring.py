from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import List, Optional, Tuple

@dataclass
class Signal:
    key: str
    label: str
    value: str
    risk: float
    weight: float
    reliability: float
    message: str
    decisive: bool = True
    group: str = "liveness"
    extra: dict = field(default_factory=dict)

    @property
    def status(self) -> str:
        if self.reliability < 0.2:
            return "n/a"
        if self.risk >= 0.65:
            return "alert"
        if self.risk >= 0.35:
            return "watch"
        return "clear"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["status"] = self.status
        return d

def clip01(x: float) -> float:
    return float(min(1.0, max(0.0, x)))

def ramp(x: float, x0: float, x1: float, r0: float, r1: float) -> float:
    if x1 == x0:
        return r1
    t = clip01((x - x0) / (x1 - x0))
    return float(r0 + t * (r1 - r0))

def aggregate(signals: List[Signal]) -> Tuple[float, float, List[str]]:
    num = sum(s.weight * s.reliability * s.risk for s in signals)
    den = sum(s.weight * s.reliability for s in signals)
    total_w = sum(s.weight for s in signals) or 1.0

    if den < 1e-6:
        return 50.0, 0.0, ["Not enough usable evidence in this clip — defaulted to medium risk"]

    score = num / den

    strong = [s for s in signals if s.reliability >= 0.5 and s.risk >= 0.65]
    decisive = [s for s in strong if s.decisive]
    if decisive:
        floor = max(s.extra.get("floor", 0.8) * s.risk * min(1.0, s.reliability + 0.2) for s in decisive)
        score = max(score, floor)
    if len(strong) >= 2:
        score += 0.08 * (len(strong) - 1)

    score = clip01(score)
    confidence = clip01(den / total_w)

    ordered = sorted(
        signals,
        key=lambda s: (s.reliability >= 0.2, s.weight * s.reliability * s.risk),
        reverse=True,
    )
    reasons = [s.message for s in ordered]
    return round(100.0 * score, 1), round(confidence, 2), reasons

def group_score(signals: List[Signal]) -> Tuple[Optional[float], float, List[str], dict]:

    groups = {}
    for g in ("liveness", "synthesis"):
        sig = [s for s in signals if s.group == g]
        usable = sum(s.weight * s.reliability for s in sig)
        if sig and usable > 1e-6:
            sc, conf, _ = aggregate(sig)
            groups[g] = {"score": sc, "confidence": conf}
    _, _, reasons = aggregate(signals) if signals else (0, 0, [])
    if not groups:
        return None, 0.0, reasons, groups
    worst = max(groups, key=lambda g: groups[g]["score"])
    conf = max(groups[worst]["confidence"], max(v["confidence"] for v in groups.values()) * 0.8)
    return groups[worst]["score"], round(conf, 2), reasons, {**groups, "driver": worst}

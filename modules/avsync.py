from __future__ import annotations

import numpy as np
from scipy import signal as sps

from .scoring import Signal, aggregate, clip01, ramp
from .voice_module import _resample, _vad, SR

GRID_DT = 0.04
MAX_LAG_S = 0.4
WINDOW_S = 10.0
STATIC_SPREAD = 0.04
SMOOTH_S = 0.2
MOVING_SPREAD = 0.09
NULL_SHIFT_S = 1.2

def _peak_corr(mz, ez, L):
    n = len(mz)
    curve = []
    for k in range(-L, L + 1):
        a = mz[max(0, -k): n - max(0, k)]
        b = ez[max(0, k): n - max(0, -k)]
        curve.append(float(np.mean(a * b)) if len(a) > 10 else 0.0)
    kb = int(np.argmax(curve))
    return curve[kb], kb, curve

def _chance_level(mz, ez, L):
    n = len(mz)
    lo, step = int(NULL_SHIFT_S / GRID_DT), max(1, int(0.2 / GRID_DT))
    peaks = [_peak_corr(mz, np.roll(ez, sh), L)[0] for sh in range(lo, n - lo, step)]
    return float(np.percentile(peaks, 95)) if len(peaks) >= 5 else None

def _envelope(y, sr, t_end):
    y = _resample(y.astype(np.float32), sr)
    sos = sps.butter(4, [250, 3500], btype="band", fs=SR, output="sos")
    y = sps.sosfilt(sos, y)
    hop, win = SR // 100, SR // 50
    n = 1 + max(0, len(y) - win) // hop
    idx = np.arange(win)[None, :] + hop * np.arange(n)[:, None]
    rms = np.sqrt(np.mean(y[idx] ** 2, 1)) + 1e-9
    db = 20 * np.log10(rms)
    speech, *_ = _vad(db)
    t = t_end - (len(y) - (hop * np.arange(n) + win / 2)) / SR
    return t, np.sqrt(rms), speech

def analyze_sync(face_t, mouth, audio, sr, audio_t_end, window_s: float = WINDOW_S) -> dict:
    out = {"status": "waiting", "score": None, "confidence": 0.0, "signals": [], "reasons": [], "details": {}}
    na = lambda why: {**out, "signals": [Signal("sync", "Lip-sync", "—", 0.3, 0.9, 0.0, why).to_dict()],
                      "reasons": [why]}
    if audio is None or not sr or len(audio) < sr * 2 or audio_t_end is None:
        return na("Needs the call's system audio to compare lips with voice")
    if len(face_t) < 20 or np.isfinite(mouth).sum() < 20:
        return na("Needs a tracked face to compare lips with voice")

    at, env, speech = _envelope(audio, sr, audio_t_end)
    t0 = max(face_t[0], at[0], max(face_t[-1], at[-1]) - window_s)
    t1 = min(face_t[-1], at[-1])
    if t1 - t0 < 3.0:
        return na("Waiting for overlapping audio and video")
    grid = np.arange(t0, t1, GRID_DT)

    ok = np.isfinite(mouth)
    in_win = (face_t >= t0) & (face_t <= t1)
    valid_frac = float(ok[in_win].mean()) if in_win.any() else 0.0
    if valid_frac < 0.6:
        return na("Face tracking too patchy for a lip-sync check")
    m = np.interp(grid, face_t[ok], mouth[ok])
    k = max(1, int(round(SMOOTH_S / GRID_DT)))
    m = np.convolve(np.pad(m, (k // 2, k - 1 - k // 2), mode="edge"), np.ones(k) / k, "valid")
    e = np.interp(grid, at, env)
    sp = np.interp(grid, at, speech.astype(float)) > 0.5
    speech_s = float(sp.sum() * GRID_DT)
    out["details"] = {"speech_s": round(speech_s, 1)}
    if speech_s < 2.5:
        out["status"] = "listening"
        return na("Waiting for the customer to speak")

    mouth_move = float(np.std(m[sp]))
    spread = float(np.percentile(m[sp], 95) - np.percentile(m[sp], 5))
    rel = clip01((speech_s - 2.0) / 5.0) * clip01((valid_frac - 0.5) / 0.4)
    lag_ms, best, curve, chance = None, None, [], None
    if spread < STATIC_SPREAD or mouth_move < 0.012:
        s = Signal("sync", "Lip-sync", f"lips {100 * spread:.0f}%", 0.85, 0.9, rel,
                   f"Voice is heard for {speech_s:.0f}s but the lips barely move (opening spread "
                   f"{100 * spread:.0f}% of eye distance) — the audio is not coming from this face "
                   f"(voice-over, dubbed or frozen video)")
    else:
        mz = (m - m.mean()) / (m.std() + 1e-9)
        ez = (e - e.mean()) / (e.std() + 1e-9)
        L = int(round(MAX_LAG_S / GRID_DT))
        best, kbest, curve = _peak_corr(mz, ez, L)
        lag_ms = (kbest - L) * GRID_DT * 1000
        chance = _chance_level(mz, ez, L)
        excess = best - (chance if chance is not None else 0.20)
        risk = max(ramp(best, 0.10, 0.30, 0.7, 0.05),
                   ramp(excess, 0.0, 0.12, 0.75, 0.05))
        risk = max(risk, ramp(spread, STATIC_SPREAD, MOVING_SPREAD, 0.6, 0.0) * clip01(1.0 - excess / 0.2))
        if spread < MOVING_SPREAD and excess < 0.08:
            risk = max(risk, 0.8)
        ch = f", chance level {chance:.2f}" if chance is not None else ""
        if spread < MOVING_SPREAD and excess < 0.08:
            msg = (f"Voice is heard for {speech_s:.0f}s but the lips hardly move and don't follow it "
                   f"(correlation {best:.2f}{ch}) — the audio is not coming from this face")
        elif risk >= 0.5:
            msg = (f"Lip movement doesn't follow the voice (correlation {best:.2f}{ch}) — audio and face may "
                   f"come from different sources")
        elif risk >= 0.3:
            msg = f"Weak lip-voice coupling (correlation {best:.2f}{ch}, offset {lag_ms:+.0f} ms)"
        else:
            msg = f"Lips move in step with the voice (correlation {best:.2f}{ch}, offset {lag_ms:+.0f} ms)"
        s = Signal("sync", "Lip-sync", f"r {best:.2f}", risk, 0.9, rel, msg, decisive=risk >= 0.65)

    score, conf, reasons = aggregate([s])
    out.update(status="analyzing", score=score, confidence=conf, reasons=reasons, signals=[s.to_dict()])
    out["details"].update(corr=best, lag_ms=lag_ms, mouth_move=round(mouth_move, 4), spread=round(spread, 4),
                          chance=None if chance is None else round(chance, 3),
                          curve=[round(c, 3) for c in curve])
    return out

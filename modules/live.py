from __future__ import annotations

import math
import random
import threading
import time
from collections import deque
from typing import Optional

import cv2
import numpy as np

from .avsync import analyze_sync
from .challenge import ChallengeManager
from .scoring import Signal, aggregate, group_score
from .detectors.face_classifier import FaceDeepfakeClassifier
from .detectors.scene_classifier import SceneDeepfakeClassifier
from .detectors.voice_classifier import VoiceDeepfakeClassifier
from .detectors.custom_heads import CustomFaceDetector, CustomVoiceDetector
from .face_module import FaceStream, RECENT_MODEL_S
from .evidence import predict as predict_learned
from .fusion import LiveFusion, Reading, fuse
from .voice_module import VoiceStream

EVENT_COOLDOWN_S = 20.0
UNLOCK_FAKE = 0.75
UNLOCK_S = 3.0
JUMP_COOLDOWN_S = 8.0
MARK_KEYS = {"model", "hand", "profile", "sync"}
LIVE_WINDOW_S = 20.0
VOICE_EVERY_S = 2.0
SENTENCES = [
    "The name on my ID card is my full legal name",
    "I am opening this account for myself",
    "Today I am completing my video KYC",
]

def clean(o):
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return f if math.isfinite(f) else None
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    return o

def _top_finding(mod: dict) -> str:
    sigs = [s for s in mod.get("signals", []) if s["status"] != "n/a"]
    for want in ("alert", "watch"):
        hits = sorted((s for s in sigs if s["status"] == want), key=lambda s: -s["risk"] * s["weight"])
        if hits:
            return hits[0]["message"]
    return mod["reasons"][0] if mod.get("reasons") else ""

class LiveSession:
    def __init__(self):
        self.face = FaceStream()
        self.voice = VoiceStream(keep_s=LIVE_WINDOW_S + 15, window_s=LIVE_WINDOW_S)
        self.fusion = LiveFusion()
        self.challenges = ChallengeManager()
        self.lock = threading.Lock()
        self.has_audio: Optional[bool] = None
        self.carry = None
        self._new_challenge()
        self._reset_counters()

    def _new_challenge(self):
        digits = " ".join(str(random.randint(0, 9)) for _ in range(4))
        self.challenge = f"“{random.choice(SENTENCES)} — reference {digits}”"

    def _reset_counters(self):
        self.t0: Optional[float] = None
        self.t_media: Optional[float] = None
        self.frames = 0
        self.proc_ms = deque(maxlen=30)
        self.frame_ts = deque(maxlen=40)
        self.pending_events = []
        self._sig_status = {}
        self._last_alert = {}
        self._verdict = None
        self._voice = None
        self._voice_t = -1e9
        self._face_status = None
        self._voice_status = None

    def configure(self, msg: dict):
        if "roi" in msg:
            self.face.set_roi(msg["roi"])
        if "has_audio" in msg:
            self.has_audio = bool(msg["has_audio"])

    def start_challenge(self, kind="phrase", custom=None) -> dict:
        kind = kind if kind in ("phrase", "hand", "profile") else "phrase"
        with self.lock:
            t = self.t_media or 0.0
        st = self.challenges.start(kind, t, custom)
        if kind == "hand":
            from .challenge import PREP_S, ACTION
            self.face.watch_hands(t + PREP_S + ACTION["hand"]["seconds"] + 5.0)
        self._event("sync" if kind == "phrase" else "face", "info", f"Challenge started — {st['title']}: “{st['phrase']}”")
        return self.challenges.public()

    def cancel_challenge(self):
        self.challenges.cancel()

    def reset(self):
        self.carry = None
        self.challenges.reset()
        self.face.reset()
        self.voice.reset()
        self.fusion.reset()
        self._new_challenge()
        with self.lock:
            self._reset_counters()

    def close(self):
        self.face.close()

    def _clock(self, ts_ms: float) -> float:
        t = ts_ms / 1000.0
        with self.lock:
            if self.t0 is None:
                self.t0 = t
            self.t_media = t if self.t_media is None else max(self.t_media, t)
        return t

    def process_frame(self, ts_ms: float, jpeg: bytes, origin=(0, 0), full_size=None) -> dict:
        t = self._clock(ts_ms)
        t_start = time.perf_counter()
        frame = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            return {"type": "ack", "error": "bad frame"}
        ov = self.face.process(t, frame, origin, full_size)
        size = list(full_size) if full_size else [frame.shape[1], frame.shape[0]]
        ms = 1000 * (time.perf_counter() - t_start)
        with self.lock:
            self.frames += 1
            self.proc_ms.append(ms)
            self.frame_ts.append(t)
            for kind, text in self.face.events:
                if kind == "new_customer":
                    failed = [st["title"] for st in self.challenges.results.values() if st.get("verdict") == "fail"]
                    bad = self._verdict in ("suspicious", "deepfake") or failed or self.carry is not None
                    if bad:
                        if self.carry is None:
                            self.carry = {"verdict": self._verdict, "score": self.fusion.value or 0.0,
                                          "failed": failed}
                        self.voice.reset()
                        self.challenges.cancel()
                        why = ", ".join(failed) + " FAILED" if failed else f"ended as {self._verdict}"
                        self._event("fusion", "alert",
                                    f"The face left and came back — the previous person on this call {why}. "
                                    f"Evidence kept; press New customer only if this is really a different person",
                                    mark=True)
                        continue
                    self.fusion.reset()
                    self.voice.reset()
                    self.challenges.reset()
                    self._verdict = None
                    self._new_challenge()
                    self._event("face", "info", text, mark=True)
                    continue
                self._event("face", "info" if kind == "acquire" else "warn", text)
            self.face.events.clear()
        return clean({"type": "ack", "t": ts_ms, "proc_ms": round(ms, 1), "size": size, **ov})

    def add_audio(self, ts_ms: float, sr: int, pcm16: bytes):
        t = self._clock(ts_ms)
        pcm = np.frombuffer(pcm16, dtype="<i2").astype(np.float32) / 32768.0
        self.voice.push(t, int(sr), pcm)

    def _event(self, module, level, text, t=None, mark=False):
        t = self.t_media if t is None else t
        rel = 0.0 if t is None or self.t0 is None else t - self.t0
        self.pending_events.append({"t": round(rel, 1), "module": module, "level": level, "text": text,
                                    "at": time.strftime("%H:%M:%S"), "mark": bool(mark)})

    def _track_signals(self, module, mod):
        now = self.t_media or 0.0
        for s in mod.get("signals", []):
            key = f"{module}.{s['key']}"
            prev = self._sig_status.get(key)
            self._sig_status[key] = s["status"]
            if s["status"] == "alert" and prev != "alert" and s["reliability"] >= 0.5 \
                    and now - self._last_alert.get(key, -1e9) > EVENT_COOLDOWN_S:
                self._last_alert[key] = now
                self._event(module, "alert", f"{s['label']}: {s['message']}", mark=s["key"] in MARK_KEYS)

    def analyze(self) -> Optional[dict]:
        if self.t_media is None:
            return None
        face = self.face.analyze_window()
        y, sr, a_end = self.voice.snapshot(self.voice.window_s)
        if self._voice is None or self.t_media - self._voice_t >= VOICE_EVERY_S:
            self._voice, self._voice_t = self.voice.analyze_window(y, sr, a_end), self.t_media
        voice = dict(self._voice)
        if self.has_audio is False:
            voice["status"] = "no_track"
        mt, mouth = self.face.mouth_series(LIVE_WINDOW_S)
        sync = analyze_sync(mt, mouth, y, sr, a_end, window_s=LIVE_WINDOW_S)
        prev = (self.challenges.state or {}).get("status")
        ch = self.challenges.update(self.t_media, self.voice, self.face, analyze_sync, face_score=face.get("score"))
        if ch and ch.get("status") == "done" and prev != "done":
            lvl = {"fail": "alert", "pass": "ok"}.get(ch.get("verdict"), "warn")
            self._event("sync" if ch["kind"] == "phrase" else "face", lvl,
                        f"{ch['title']} challenge: {(ch.get('verdict') or '').upper()} — {ch['message']}"
                        + (f" (heard: “{ch['transcript']}”)" if ch.get("transcript") else ""), mark=lvl == "alert")
        extra = {"face": [], "sync": []}
        for module, sig in self.challenges.signals():
            extra[module].append(sig)
        if extra["sync"]:
            sigs = [Signal(**{k: v for k, v in d.items() if k != "status"}) for d in sync.get("signals", [])
                    if d.get("reliability", 0) > 0] + extra["sync"]
            sc, cf, reasons = aggregate(sigs)
            sync = {**sync, "status": "analyzing", "score": sc, "confidence": cf,
                    "signals": sync.get("signals", []) + [x.to_dict() for x in extra["sync"]], "reasons": reasons}
        if extra["face"]:
            sigs = [Signal(**{k: v for k, v in d.items() if k != "status"}) for d in face.get("signals", [])] \
                + extra["face"]
            sc, cf, reasons, groups = group_score(sigs)
            face = {**face, "score": sc, "confidence": cf, "reasons": reasons,
                    "signals": face.get("signals", []) + [x.to_dict() for x in extra["face"]]}
        mods = {"face": face, "voice": voice, "sync": sync}

        readings = {k: Reading(m["score"], m["confidence"]) for k, m in mods.items()}
        learned = predict_learned(mods)
        raw, conf = fuse(readings, learned)
        cap, passed = self.challenges.pass_credit(self.t_media)
        if passed and self._fake_after_pass():
            self.challenges.invalidate_passes()
            cap, passed = None, []
            self._event("face", "alert", f"The face looks fake AFTER the challenge passed (detector ≥ {UNLOCK_FAKE:.2f} "
                                         f"for {UNLOCK_S:.0f} s) — the pass no longer counts", mark=True)
        if self.carry is not None and raw is not None:
            raw = max(raw, float(self.carry["score"]))

        with self.lock:
            t_rel = self.t_media - self.t0
            overall = self.fusion.update(self.t_media, raw, conf, face["status"] in ("tracking", "warming"),
                                         readings, cap=cap)
            if cap is not None and overall["verdict"] == "genuine":
                names = " + ".join(passed)
                overall["confidence"] = round(min(0.97, 0.93 + 0.02 * (len(passed) - 1)), 2)
                overall["action"] = f"Liveness challenge passed ({names}). Continue the KYC."
            overall["challenge_pass"] = passed or None
            if self.carry is not None:
                c = self.carry
                why = ", ".join(c["failed"]) + " challenge FAILED" if c["failed"] else \
                    f"ended as {(c['verdict'] or 'suspicious')} (score {c['score']:.0f})"
                overall["action"] = (f"Warning: the previous person on this call {why}. Evidence kept — press "
                                     f"New customer only if this is really a different person.")
                overall["carry"] = why

            if face["status"] != self._face_status and face["status"] == "searching" and self._face_status:
                self._event("face", "warn", "No customer face on screen")
            self._face_status = face["status"]
            if voice["status"] == "analyzing" and self._voice_status not in (None, "analyzing"):
                self._event("voice", "info", "Customer speech detected — voice analysis running")
            self._voice_status = voice["status"]
            for k, m in mods.items():
                self._track_signals(k, m)
            jump = ((face.get("details") or {}).get("model") or {}).get("jump")
            if jump and jump[1] >= 0.65 and jump[1] - jump[0] >= 0.3 \
                    and self.t_media - self._last_alert.get("face.jump", -1e9) > JUMP_COOLDOWN_S:
                self._last_alert["face.jump"] = self.t_media
                self._event("face", "alert", f"Sudden change: face fake score jumped {jump[0]:.2f} → {jump[1]:.2f} "
                                             f"in the last {1000 * RECENT_MODEL_S:.0f} ms", mark=True)
                if passed:
                    self._event("face", "warn", "Sudden face change after the challenge passed — "
                                                "the score stays lowered; repeat a challenge if you are unsure")
            if overall["verdict"] != self._verdict and overall["verdict"] in ("genuine", "suspicious", "deepfake"):
                lvl = {"genuine": "ok", "suspicious": "warn", "deepfake": "alert"}[overall["verdict"]]
                self._event("fusion", lvl, f"{overall['label']} (risk {overall['score']:.0f})")
            self._verdict = overall["verdict"]
            events, self.pending_events = self.pending_events, []

            ft = list(self.frame_ts)
            fps = (len(ft) - 1) / (ft[-1] - ft[0]) if len(ft) > 2 and ft[-1] > ft[0] else 0.0
            stats = {"fps": round(fps, 1), "proc_ms": round(float(np.mean(self.proc_ms)), 1) if self.proc_ms else None,
                     "frames": self.frames, "audio_s": round(self.voice.received_s, 1),
                     "audio_sr": self.voice.sr,
                     "models": {"face": FaceDeepfakeClassifier.status, "voice": VoiceDeepfakeClassifier.status,
                                "scene": SceneDeepfakeClassifier.status,
                                "custom_face": CustomFaceDetector.status, "custom_voice": CustomVoiceDetector.status}}

        for k, m in mods.items():
            m["finding"] = _top_finding(m)
            m.pop("reasons", None)

        return clean({
            "type": "state", "t": round(t_rel, 2),
            "overall": {**overall, "learned": None if learned is None else round(100 * learned[0], 1)},
            "modules": mods,
            "point": [round(t_rel, 2), overall["score"], face["score"], voice["score"], sync["score"]],
            "events": events,
            "prompts": self._prompts(mods, overall),
            "challenge": self.challenges.public(),
            "stats": stats,
        })

    def _fake_after_pass(self) -> bool:
        from .face_module import _crop_risks
        t_pass = max((st.get("t_done", 0.0) for st in self.challenges.results.values()
                      if st.get("verdict") == "pass" and not st.get("stale")), default=None)
        if t_pass is None or self.t_media - t_pass < UNLOCK_S:
            return False
        with self.face.lock:
            recs = [r for r in self.face.model_recs if r[0] >= max(t_pass, self.t_media - UNLOCK_S)]
        if len(recs) < 8:
            return False
        return float(np.median(_crop_risks(recs, self.face.model_names))) >= UNLOCK_FAKE

    def _prompts(self, mods, overall):
        face, voice = mods["face"], mods["voice"]
        sig = {s["key"]: s for m in mods.values() for s in m.get("signals", [])}
        out = []
        if face["status"] == "searching":
            out.append(("screen", "Bring the video call into view",
                        "No face on the shared screen. Un-minimise the call or maximise the customer's video."))
        if voice["status"] == "no_track":
            out.append(("audio", "Share the call's audio",
                        "Stop and share again with “Also share tab audio” (tab) or “Also share system audio” "
                        "(entire screen) ticked to analyse the voice."))
        elif voice["status"] in ("listening", "silent", "no_audio") and face["status"] == "tracking":
            out.append(("speak", "Ask the customer to read aloud", self.challenge))
        if face["status"] == "tracking":
            par = sig.get("parallax")
            if par and par["reliability"] < 0.3:
                out.append(("turn", "Ask them to turn their head slowly left and right",
                            "Needed for the 3-D depth test — flat photos and screens fail it."))
            pulse = sig.get("pulse")
            if pulse and pulse["reliability"] < 0.2 and face["details"].get("tracked_s", 0) > 8:
                out.append(("still", "Ask them to hold still facing the light for 10 s",
                            "Lets the remote-pulse check find a heartbeat in the skin."))
        doubtful = overall["verdict"] in ("suspicious", "deepfake")
        settled = face["status"] == "tracking" and face["details"].get("tracked_s", 0) > 5
        hand, prof = sig.get("hand"), sig.get("profile")
        if (doubtful or settled) and (hand is None or hand["reliability"] == 0):
            out.append(("hand", "Ask them to wave a hand slowly across their face",
                        "Measured: a real face comes back unchanged; a real-time face swap glitches as it re-locks."))
        if (doubtful or settled) and (prof is None or prof["reliability"] == 0):
            out.append(("profile", "Ask for a full side profile, then back",
                        "Measured: face-swap models are trained on frontal faces and break on side views."))
        return [{"icon": i, "title": t, "text": x} for i, t, x in out[:4]]

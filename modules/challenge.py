from __future__ import annotations

import difflib
import logging
import os
import random
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import numpy as np

log = logging.getLogger("fraudshield.challenge")

WORDS = ["apple", "river", "orange", "garden", "silver", "window", "pencil", "rocket", "yellow", "forest",
         "candle", "button", "planet", "bottle", "tiger", "jacket", "mango", "ladder", "summer", "doctor",
         "market", "camera", "purple", "pillow", "island", "basket", "copper", "lemon", "hammer", "violin"]
DIGIT_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
NUM_ALIASES = {"oh": "zero", "o": "zero", "to": "two", "too": "two", "for": "four", "won": "one",
               "ate": "eight", "free": "three", "tree": "three", "nein": "nine", "sex": "six", "fore": "four"}

PREP_S = 5.0
MAX_S = 20.0
END_SILENCE_S = 2.0
MIN_SPEECH_S = 1.2

def make_phrase(rng: Optional[random.Random] = None):
    rng = rng or random.SystemRandom()
    words = rng.sample(WORDS, 2)
    digits = [rng.randrange(10) for _ in range(4)]
    tokens = words + [DIGIT_WORDS[d] for d in digits]
    display = f"{words[0].capitalize()} {words[1]} · {' '.join(map(str, digits))}"
    return display, tokens

def normalize(text: str):
    text = text.lower()
    text = re.sub(r"\d", lambda m: f" {DIGIT_WORDS[int(m.group())]} ", text)
    toks = re.findall(r"[a-z]+", text)
    return [NUM_ALIASES.get(t, t) for t in toks]

def match_score(expected, heard):
    if not expected:
        return 0.0, []
    sm = difflib.SequenceMatcher(a=expected, b=heard, autojunk=False)
    hit = set()
    for a, b, n in sm.get_matching_blocks():
        hit.update(range(a, a + n))
    for i, tok in enumerate(expected):
        if i not in hit and any(difflib.SequenceMatcher(a=tok, b=h).ratio() >= 0.8 for h in heard):
            hit.add(i)
    missed = [t for i, t in enumerate(expected) if i not in hit]
    return len(hit) / len(expected), missed

class Transcriber:
    _instance = None
    _lock = threading.Lock()
    status = "not loaded"

    @classmethod
    def get(cls):
        with cls._lock:
            if cls._instance is None and not cls.status.startswith("unavailable"):
                try:
                    cls.status = "loading"
                    from .detectors import LOAD_LOCK
                    with LOAD_LOCK:
                        cls._instance = cls()
                    cls.status = "ready"
                except Exception as e:
                    cls.status = f"unavailable: {e}"
                    log.warning("speech-to-text unavailable: %s", e)
        return cls._instance

    @classmethod
    def warmup_async(cls):
        threading.Thread(target=cls.get, name="fs-asr-load", daemon=True).start()

    def __init__(self):
        import torch
        from transformers import pipeline
        from .detectors import MODELS_DIR
        cuda = torch.cuda.is_available()
        repo = os.environ.get("FRAUDSHIELD_ASR") or ("openai/whisper-small" if cuda else "openai/whisper-base")
        self.pipe = pipeline("automatic-speech-recognition", model=repo, device=0 if cuda else -1,
                             torch_dtype=torch.float16 if cuda else torch.float32,
                             model_kwargs={"cache_dir": str(MODELS_DIR / "hf-cache")})
        self.name = f"Whisper ({repo.split('/')[-1]}) · {'cuda' if cuda else 'cpu'}"
        self._run = threading.Lock()

    def __call__(self, y16: np.ndarray) -> str:
        with self._run:
            out = self.pipe({"raw": np.asarray(y16, np.float32), "sampling_rate": 16000},
                            generate_kwargs={"language": "english", "task": "transcribe"})
        return str(out.get("text", "")).strip()

_POOL = ThreadPoolExecutor(1, thread_name_prefix="fs-asr")

def judge(words: float, sync: Optional[dict], speech_s: float, reaction_s: Optional[float]):
    sync_risk = None if not sync or sync.get("score") is None else sync["score"] / 100.0
    sync_conf = 0.0 if not sync else float(sync.get("confidence") or 0.0)
    if speech_s < MIN_SPEECH_S:
        return None, "no_answer", "No answer heard — repeat the challenge (is the call audio shared?)"
    if words < 0.5:
        return 0.75, "fail", (f"Said something else: only {100 * words:.0f}% of the phrase was heard — a recording or a "
                              f"pre-made fake can't answer a random phrase (or ask them to repeat clearly)")
    if sync_risk is not None and sync_conf >= 0.3 and sync_risk >= 0.65:
        return 0.9, "fail", (f"The phrase was said ({100 * words:.0f}% of words) but the lips did not move with it — "
                             f"the voice is not coming from this face")
    if words < 0.8:
        return 0.45, "unclear", f"Only part of the phrase was heard ({100 * words:.0f}%) — ask them to repeat it"
    slow = reaction_s is not None and reaction_s > 4.0
    msg = (f"Phrase read correctly ({100 * words:.0f}% of words)"
           + (", lips moved with it" if sync_risk is not None and sync_risk < 0.35 else "")
           + (f" — but slow to start ({reaction_s:.1f} s)" if slow else ""))
    return (0.3 if slow else 0.05), "pass", msg

class SpokenChallenge:

    def __init__(self):
        self.state = None

    def start(self, t_now: float, custom: Optional[str] = None):
        if custom and normalize(custom):
            display, tokens = custom.strip(), normalize(custom)
        else:
            display, tokens = make_phrase()
        self.state = {"status": "ready", "phrase": display, "tokens": tokens, "t0": t_now,
                      "t_go": t_now + PREP_S, "prep_s": PREP_S, "ready_left_s": PREP_S, "seconds": MAX_S,
                      "say": " ".join(tokens), "custom": bool(custom and normalize(custom)),
                      "transcript": None, "match": None, "missed": [], "risk": None, "verdict": None,
                      "message": "Get ready — read the phrase to the customer", "reaction_s": None}
        if Transcriber.status == "not loaded":
            Transcriber.warmup_async()
        return self.state

    def cancel(self):
        self.state = None

    def update(self, t_now, voice, face, analyze_sync):
        st = self.state
        if not st:
            return None
        if st["status"] == "ready":
            st["ready_left_s"] = round(max(0.0, st["t_go"] - t_now), 1)
            if t_now >= st["t_go"]:
                st["status"] = "listening"
                st["message"] = "Listening — ask the customer to read the phrase aloud"
        if st["status"] == "listening":
            y, sr, a_end = voice.snapshot(MAX_S + PREP_S + 2)
            if sr and a_end is not None and len(y):
                dur = min(len(y) / sr, max(0.0, a_end - st["t0"]))
                seg = y[len(y) - int(dur * sr):] if dur > 0 else y[:0]
                sp, t_sp = _speech_mask(seg, sr)
                speech_s = float(sp.sum() * 0.01)
                if sp.any() and st["reaction_s"] is None:
                    st["reaction_s"] = round(max(0.0, float(np.argmax(sp)) * 0.01 - PREP_S), 2)
                done_talking = sp.any() and speech_s >= MIN_SPEECH_S and \
                    (len(sp) - 1 - np.flatnonzero(sp)[-1]) * 0.01 >= END_SILENCE_S
                timeout = a_end - st["t_go"] >= MAX_S
                if done_talking or timeout:
                    st["status"] = "checking"
                    st["message"] = "Checking what was said…"
                    mt, mouth = face.mouth_series(MAX_S + PREP_S + 2)
                    keep = mt >= st["t0"] if len(mt) else np.zeros(0, bool)
                    sync = analyze_sync(mt[keep], mouth[keep], seg, sr, a_end, window_s=max(3.0, dur)) \
                        if keep.sum() >= 20 else None
                    _POOL.submit(self._finish, seg.copy(), sr, speech_s, sync)
        return st

    def _finish(self, seg, sr, speech_s, sync):
        st = self.state
        if st is None:
            return
        try:
            from .voice_module import _resample
            asr = Transcriber.get()
            if asr is None:
                st.update(status="error", message=f"Speech-to-text not available ({Transcriber.status})")
                return
            text = asr(_resample(seg, sr)) if speech_s >= MIN_SPEECH_S else ""
            words, missed = match_score(st["tokens"], normalize(text))
            risk, verdict, msg = judge(words, sync, speech_s, st["reaction_s"])
            st.update(status="done", transcript=text, match=round(words, 2), missed=missed, risk=risk,
                      verdict=verdict, message=msg,
                      sync=None if not sync else {"score": sync.get("score"), "confidence": sync.get("confidence")})
        except Exception as e:
            log.warning("challenge failed: %s", e)
            st.update(status="error", message=f"Challenge check failed: {e}")

    def signal(self):
        from .scoring import Signal
        st = self.state
        if not st or st.get("status") != "done" or st.get("risk") is None:
            return None
        risk = float(st["risk"])
        return Signal("phrase", "Spoken challenge", f"{100 * (st['match'] or 0):.0f}% words", risk, 2.5,
                      1.0 if risk >= 0.65 else 0.6, st["message"], decisive=risk >= 0.65, group="liveness",
                      extra={"floor": 0.95})

def _speech_mask(y, sr):
    if not len(y) or not sr:
        return np.zeros(0, bool), None
    hop = max(1, int(sr * 0.01))
    n = len(y) // hop
    if n < 5:
        return np.zeros(0, bool), None
    e = 10 * np.log10(np.mean(y[: n * hop].reshape(n, hop).astype(np.float64) ** 2, 1) + 1e-10)
    thr = max(np.percentile(e, 10) + 10.0, np.percentile(e, 98) - 30.0)
    sp = e > thr
    idx = np.flatnonzero(sp)
    for a, b in zip(idx[:-1], idx[1:]):
        if 1 < b - a <= 25:
            sp[a:b] = True
    return sp, thr

ACTION = {
    "hand": {"title": "Hand across the face",
             "ask": "Please wave your hand slowly across your face, once.",
             "seconds": 15.0},
    "profile": {"title": "Side profile",
                "ask": "Please turn your head slowly to one side until I see your side profile, then back.",
                "seconds": 15.0},
}
PROFILE_REACH = 25.0
PASS_BELOW = 0.42
FAIL_FROM = 0.50
FACE_PASS_BELOW = 42.0
FACE_FAIL_FROM = 50.0

def _face_window(face, t_from):
    with face.lock:
        recs = [r for r in face.records if r.t >= t_from]
        mrecs = [m for m in face.model_recs if m[0] >= t_from]
    ts = np.array([r.t for r in recs], float)
    present = np.array([r.lm is not None for r in recs], bool)
    occ = np.array([getattr(r, "occ", np.nan) for r in recs], float)
    jit = np.array([getattr(r, "jit", np.nan) for r in recs], float)
    yaw = np.array([getattr(r, "yaw", np.nan) for r in recs], float)
    hand = np.array([getattr(r, "hand_x", np.nan) for r in recs], float)
    return ts, present, occ, jit, yaw, mrecs, hand

def _face_realness(mrecs, names, sel):
    from .face_module import _window_risk
    sel = np.asarray(sel, bool)
    if len(sel) == 0 or sel.sum() < 6:
        return None
    return float(_window_risk(mrecs, names, sel))

LOG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "challenge_log.jsonl")

def _log(entry):
    try:
        import json
        import time
        entry = {"at": time.strftime("%Y-%m-%d %H:%M:%S"), **entry}
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=lambda o: float(o) if hasattr(o, "__float__") else str(o)) + "\n")
    except Exception:
        pass

class ActionChallenge:
    def __init__(self, kind):
        self.kind = kind

    def start(self, t_now):
        a = ACTION[self.kind]
        return {"kind": self.kind, "status": "ready", "title": a["title"], "phrase": a["ask"], "say": "",
                "t_ask": t_now, "t0": t_now + PREP_S, "prep_s": PREP_S, "ready_left_s": PREP_S,
                "seconds": a["seconds"], "left_s": a["seconds"], "progress": "Get ready…",
                "risk": None, "verdict": None, "message": "Get ready — ask the customer now", "transcript": None,
                "match": None, "reaction_s": None, "custom": False}

    def update(self, st, t_now, face, face_score=None):
        self.face_score = st.get("face_before")
        if st["status"] == "ready":
            st["ready_left_s"] = round(max(0.0, st["t0"] - t_now), 1)
            if t_now < st["t0"]:
                return
            st["status"] = "watching"
            st["message"] = "Watching…"
        if st["status"] != "watching":
            return
        from .face_module import occlusion_episodes, analyze_hand_test, analyze_profile_test, hand_cover_check
        t0, t_ask = st["t0"], st["t_ask"]
        elapsed = t_now - t0
        st["left_s"] = round(max(0.0, st["seconds"] - elapsed), 1)
        ts, present, occ, jit, yaw, mrecs, hand = _face_window(face, t_ask - 5.0)
        self.names, self.mrecs, self.t_ask = face.model_names, mrecs, t_ask
        if not len(ts):
            st["progress"] = "No face on screen yet"
            if elapsed >= st["seconds"] + 3:
                self._done(st, None, "no_answer", "No face was tracked — bring the customer's video into view and repeat")
            return
        names = face.model_names
        if self.kind == "hand":
            eps = [e for e in occlusion_episodes(ts, present, occ) if e[0] >= t_ask - 0.2]
            hc = hand_cover_check(ts, present, occ, hand, mrecs, face.model_names, t_ask - 0.2)
            if hc["painted"]:
                self._done(st, 0.92, "fail",
                           "The hand crossed in front of the face, but the face was never covered — the face "
                           "is being drawn on top of the hand (a real-time face swap)")
                return
            st["progress"] = "Hand pass seen ✓ — checking the face during and after it…" if eps \
                else "Watching for the hand…"
            finished = (eps and t_now - eps[-1][1] >= 2.5) or (hc["crossed"] and elapsed >= 2.0)
            if not (finished or elapsed >= st["seconds"]):
                return
            if not eps:
                gone = (~present[ts >= t_ask]).mean() if (ts >= t_ask).any() else 1.0
                if gone > 0.7:
                    self._done(st, 0.5, "unclear", "The face disappeared and did not come back cleanly — repeat; "
                                                   "a swap that cannot re-lock is suspicious")
                else:
                    self._done(st, None, "not_done", "No hand pass seen — ask them to move the whole hand slowly "
                                                     "across the face, then repeat")
                return
            sig, info = analyze_hand_test(ts, present, occ, jit, mrecs, names)
            w = info.get("worst") or {}
            hand_risk = sig.risk if sig.reliability > 0 else None
            self.diag = {"hand_cover": hc, "hand_test": info}
            mt = np.array([r[0] for r in mrecs], float)
            busy = np.zeros(len(mt), bool)
            for e0, e1 in occlusion_episodes(ts, present, occ) + list(eps):
                busy |= (mt >= e0 - 0.5) & (mt <= e1 + 0.8)
            occ_at = np.interp(mt, ts, np.nan_to_num(occ, nan=0.0)) if len(ts) else np.zeros(len(mt))
            self.clean = (mt >= t_ask - 8.0) & ~busy & (occ_at < 0.2)
            detail = []
            if w.get("model_jump") is not None:
                detail.append(f"fake score {'+' if w['model_jump'] >= 0 else ''}{w['model_jump']:.2f} after the hand")
            if w.get("jitter_ratio") is not None:
                detail.append(f"landmark wobble ×{w['jitter_ratio']:.1f}")
            self._judge(st, hand_risk, ", ".join(detail),
                        pass_msg="Hand hid the face naturally and the face came back unchanged — real face",
                        fail_msg="The face broke up after the hand — a face swap re-locking")
        else:
            ok = np.isfinite(yaw) & present
            base = np.nanmedian(yaw[ok & (ts < t_ask)]) if (ok & (ts < t_ask)).sum() >= 5 else \
                (np.nanmedian(yaw[ok]) if ok.any() else np.nan)
            turned = np.abs(yaw - base) if np.isfinite(base) else np.zeros(len(yaw))
            in_win = ok & (ts >= t_ask)
            reach = float(np.nanmax(turned[in_win])) if in_win.any() else 0.0
            back = in_win & (turned < PROFILE_REACH / 2)
            reached = reach >= PROFILE_REACH
            st["progress"] = (f"Side view reached ({reach:.0f}°) ✓ — wait for them to turn back…" if reached
                              else f"Watching the head turn… {reach:.0f}° of {PROFILE_REACH:.0f}°")
            t_reach = ts[in_win][np.argmax(turned[in_win])] if in_win.any() else t_ask
            came_back = reached and (back & (ts > t_reach)).sum() >= 8
            if not (came_back or elapsed >= st["seconds"]):
                return
            if not reached:
                lost = (~present[ts >= t_ask]).mean() if (ts >= t_ask).any() else 0.0
                if lost > 0.5:
                    self._done(st, 0.5, "unclear", "The face was lost during the turn — repeat slower; a swap that "
                                                   "vanishes on a side view is suspicious")
                else:
                    self._done(st, None, "not_done", f"They only turned {reach:.0f}° — ask for a full side profile "
                                                     f"(ear facing the camera), then repeat")
                return
            sig, info = analyze_profile_test(mrecs, names)
            d = info.get("delta")
            self.diag = {"profile_test": info, "reach": reach}
            myaw = np.array([r[3] if len(r) > 3 else np.nan for r in mrecs], float)
            mt = np.array([r[0] for r in mrecs], float)
            self.clean = (mt >= t_ask - 8.0) & np.isfinite(myaw) & (np.abs(myaw - base) < 10.0) \
                if np.isfinite(base) else (mt >= t_ask - 8.0) & (mt < t_ask)
            self._judge(st, sig.risk if sig.reliability > 0 else None,
                        "" if d is None else f"fake score side {info['side']:.2f} vs facing {info['front']:.2f}",
                        pass_msg=f"Side view ({reach:.0f}°) looked as natural as the frontal view — real face",
                        fail_msg="The face looked fake on the side view — face swaps break when the head turns")

    def _judge(self, st, risk, detail, pass_msg, fail_msg):
        real = _face_realness(getattr(self, "mrecs", []), getattr(self, "names", None),
                              getattr(self, "clean", np.zeros(0, bool)))
        fs = getattr(self, "face_score", None)
        st["face_fake"] = None if real is None else round(real, 2)
        parts = [v for v in (real, None if fs is None else fs / 100.0) if v is not None]
        fake = float(np.mean(parts)) if parts else None
        st["fakeness"] = None if fake is None else round(fake, 2)
        how = "" if fake is None else (f"fakeness {fake:.2f} = detector {real:.2f}" if real is not None else
                                       f"fakeness {fake:.2f}") + (f" + face score {fs:.0f}" if fs is not None else "")
        _log({"kind": st["kind"], "action_risk": risk, "face_fake_clean": real, "face_score_before": fs,
              "fakeness": fake, "n_clean": int(np.sum(getattr(self, "clean", []))), "detail": detail,
              **getattr(self, "diag", {})})
        if fake is None:
            self._done(st, None, "unclear", "Action seen, but the AI face detector had too few clear frames of the "
                                            "face — repeat (is the face large enough on screen?)")
        elif risk is not None and risk >= 0.65:
            self._done(st, 0.9, "fail", fail_msg + (f" ({detail})" if detail else ""))
        elif fake >= FAIL_FROM:
            self._done(st, 0.9, "fail", f"The action was done, but the face itself looks fake ({how}) — a live "
                                        f"face swap can do the action; it cannot make the face real")
        elif fake < PASS_BELOW and (risk is None or risk < 0.35):
            self._done(st, 0.05, "pass", pass_msg + f" ({how}" + (f", {detail}" if detail else "") + ")")
        else:
            self._done(st, 0.5, "unclear", f"Not clear-cut ({how}" + (f", {detail}" if detail else "")
                       + ") — repeat once, face closer to the camera and well lit")

    @staticmethod
    def _done(st, risk, verdict, message):
        st.update(status="done", risk=risk, verdict=verdict, message=message, left_s=0.0)

PASS_CAP = (18.0, 12.0, 8.0)
PASS_HOLD_S = float("inf")

class ChallengeManager:

    def __init__(self):
        self.spoken = SpokenChallenge()
        self.state = None
        self.results = {}

    def start(self, kind, t_now, custom=None):
        self._n = getattr(self, "_n", 0) + 1
        if kind == "phrase":
            st = self.spoken.start(t_now, custom)
            st.update(kind="phrase", title="Read a phrase aloud")
            self.state = st
        else:
            self.spoken.cancel()
            self.state = ActionChallenge(kind).start(t_now)
        self.state["id"] = self._n
        return self.state

    def cancel(self):
        self.spoken.cancel()
        self.state = None

    def reset(self):
        self.cancel()
        self.results = {}

    def update(self, t_now, voice, face, analyze_sync, face_score=None):
        st = self.state
        if st is not None and "face_before" not in st:
            st["face_before"] = face_score
        if not st:
            return None
        prev = st.get("status")
        if st["kind"] == "phrase":
            self.spoken.update(t_now, voice, face, analyze_sync)
            st = self.state = self.spoken.state or st
            if st.get("status") in ("ready", "listening"):
                st["left_s"] = round(max(0.0, MAX_S - max(0.0, t_now - st["t_go"])), 1)
        else:
            ActionChallenge(st["kind"]).update(st, t_now, face, face_score)
        fb = st.get("face_before")
        if st.get("status") == "done" and prev != "done" and st["kind"] == "phrase" and st.get("verdict") == "pass" \
                and fb is not None and fb >= FACE_PASS_BELOW:
            face_score = fb
            fail = face_score >= FACE_FAIL_FROM
            st.update(verdict="fail" if fail else "unclear", risk=0.9 if fail else 0.5,
                      message=st["message"] + f" — but the face looks {'fake' if fail else 'doubtful'} "
                                              f"(face score {face_score:.0f}); a live face swap can read a phrase")
        if st.get("status") == "done" and prev != "done" and st.get("risk") is not None:
            st["t_done"] = t_now
            if st.get("verdict") in ("pass", "fail"):
                self.results[st["kind"]] = dict(st)
        return st

    def invalidate_passes(self):
        for st in self.results.values():
            if st.get("verdict") == "pass":
                st["stale"] = True

    def pass_credit(self, t_now):

        if any(st.get("verdict") == "fail" for st in self.results.values()):
            return None, []
        ok = [st for st in self.results.values() if st.get("verdict") == "pass" and not st.get("stale")
              and t_now - st.get("t_done", t_now) <= PASS_HOLD_S]
        if not ok:
            return None, []
        return PASS_CAP[min(len(ok), len(PASS_CAP)) - 1], [st["title"] for st in ok]

    def public(self):
        st = self.state
        if not st:
            return None
        return {k: v for k, v in st.items() if k not in ("tokens", "t0", "t_ask", "t_go", "t_done", "face_before")}

    def signals(self):
        from .scoring import Signal
        out = []
        for kind, st in self.results.items():
            if st.get("verdict") not in ("pass", "fail"):
                continue
            risk = float(st["risk"])
            fail = risk >= 0.65
            if kind == "phrase":
                label, module, group = "Spoken challenge", "sync", "liveness"
            else:
                label, module, group = ACTION[kind]["title"] + " challenge", "face", "synthesis"
            out.append((module, Signal(f"ch_{kind}", label, st.get("verdict", "").upper(), risk, 2.5,
                                       1.0 if fail else 0.6, st["message"], decisive=fail, group=group,
                                       extra={"floor": 0.95})))
        return out

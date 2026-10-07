"""
Run the WHOLE detector on video files — face, voice, lip-sync and the final verdict — without
screen sharing, and print every check. Same code as the live app, so the numbers show exactly
what each module sees; no window size / tab audio / capture quality in the way.

    python -m tools.check_video "C:\\path\\video_1.mp4"
    python -m tools.check_video data\\test\\*.mp4 --json report.json

Needs ffmpeg for the audio track (winget install Gyan.FFmpeg).
"""

from __future__ import annotations

import argparse
import glob
import json
import time
from pathlib import Path

import cv2
import numpy as np

from modules import evidence
from modules.avsync import analyze_sync
from modules.face_module import FaceStream
from modules.fusion import DEEPFAKE_FROM, GENUINE_BELOW, Reading, fuse, verdict_confidence
from modules.scoring import group_score
from modules.voice_module import _resample, _voice_model, analyze, model_signal, speech_segments

def load_audio_track(path):
    import shutil
    import subprocess
    import tempfile
    from modules.voice_module import load_audio
    ff = shutil.which("ffmpeg")
    if ff is None:
        raise RuntimeError("ffmpeg not found (winget install Gyan.FFmpeg, then open a new Command Prompt)")
    with tempfile.TemporaryDirectory() as d:
        wav = Path(d) / "a.wav"
        r = subprocess.run([ff, "-v", "error", "-y", "-i", str(path), "-vn", "-ac", "1", "-ar", "48000", str(wav)],
                           capture_output=True)
        if r.returncode != 0 or not wav.exists():
            return None, None
        return load_audio(str(wav), max_seconds=600)

def face_pass(path, seconds, fps=15.0):
    cap = cv2.VideoCapture(str(path))
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(src_fps) or src_fps < 2 or src_fps > 240:
        src_fps = 30.0
    step = max(1, int(round(src_fps / fps)))
    fs = FaceStream(keep_s=seconds + 5, model="sync")
    i, n = 0, 0
    try:
        while i < seconds * src_fps:
            ok, frame = cap.read()
            if not ok:
                break
            if i % step == 0:
                fs.process(i / src_fps, frame)
                n += 1
            i += 1
        face = fs.analyze_window(window_s=seconds)
        mt, mouth = fs.mouth_series(window_s=seconds)
        names = fs.model_names
    finally:
        fs.close()
        cap.release()
    return face, mt, mouth, names, n, i / src_fps

def voice_pass(y, sr):
    signals, det = analyze(y, sr, channel="call")
    cls = _voice_model()
    clf = cls.get()
    if det.get("speech") and clf is not None:
        segs = speech_segments(_resample(y, sr), np.array(det["speech"]), max_n=6)
        probs = clf.predict(segs) if segs else []
        s_m, _ = model_signal(probs, cls.status)
        signals.append(s_m)
    score, conf, reasons, groups = group_score(signals)
    return {"score": score, "confidence": conf, "signals": [s.to_dict() for s in signals],
            "model": getattr(clf, "name", cls.status)}

def verdict(v):
    return "Likely deepfake" if v >= DEEPFAKE_FROM else "Likely genuine" if v < GENUINE_BELOW else "Suspicious"

def show(title, mod):
    sc = mod.get("score")
    print(f"\n  {title}: {'—' if sc is None else f'{sc:.0f}'}   (confidence {100 * (mod.get('confidence') or 0):.0f}%)")
    for s in sorted(mod.get("signals", []), key=lambda s: -s["risk"] * s["reliability"]):
        st = s["status"].upper()
        print(f"    {st:<6} {s['label']:<26} {100 * s['risk']:5.0f}  rel {s['reliability']:.2f}  {s['message'][:110]}")

def main():
    import sys
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--seconds", type=float, default=20.0, help="analyse the first N seconds (default 20)")
    ap.add_argument("--json", help="also save every number to this JSON file")
    ap.add_argument("--label", choices=["real", "fake"],
                    help="you KNOW this video is real / fake: save its check results (numbers only, no video) "
                         "to data/evidence/evidence.jsonl for python -m tools.train_evidence")
    args = ap.parse_args()
    files = [Path(f) for v in args.videos for f in (glob.glob(v) or [v])]
    report = []
    for path in files:
        if not path.exists():
            print(f"not found: {path}")
            continue
        t0 = time.time()
        print(f"\n=== {path.name}")
        face, mt, mouth, names, nframes, dur = face_pass(path, args.seconds)
        print(f"  {nframes} frames analysed over {dur:.1f} s; face models: {', '.join(names or []) or 'none loaded'}")
        y, sr = load_audio_track(path)
        if y is not None:
            y = y[: int(args.seconds * sr)]
            voice = voice_pass(y, sr)
            sync = analyze_sync(mt, mouth, y, sr, len(y) / sr, window_s=args.seconds) if len(mt) else \
                {"score": None, "confidence": 0.0, "signals": []}
            print(f"  voice model: {voice['model']}")
        else:
            print("  no audio track (or ffmpeg missing) — voice and lip-sync skipped")
            voice = sync = {"score": None, "confidence": 0.0, "signals": []}
        show("FACE", face)
        show("VOICE", voice)
        show("LIP-SYNC", sync)
        readings = {k: Reading(m.get("score"), float(m.get("confidence") or 0.0))
                    for k, m in (("face", face), ("voice", voice), ("sync", sync))}
        mods = {"face": face, "voice": voice, "sync": sync}
        learned = evidence.predict(mods)
        raw, conf = fuse(readings, learned)
        if learned is not None:
            print(f"\n  LEARNED FROM YOUR VIDEOS: P(fake) {learned[0]:.2f}  ({100 * learned[1]:.0f}% of its checks measured)")
        if args.label:
            evidence.log_example(str(path), args.label, mods, None if raw is None else round(raw, 1))
            print(f"  saved as '{args.label}' example -> {evidence.LOG}")
        if raw is None:
            print("\n  VERDICT: not enough evidence")
        else:
            print(f"\n  VERDICT: {raw:.0f} — {verdict(raw)}   (confidence {100 * verdict_confidence(raw, readings):.0f}%, "
                  f"evidence {100 * conf:.0f}%)   "
                  f"[{time.time() - t0:.0f} s]")
        report.append({"file": str(path), "verdict": None if raw is None else round(raw, 1),
                       "face": face, "voice": voice, "sync": sync})
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
        print(f"\nsaved {args.json}")

if __name__ == "__main__":
    main()

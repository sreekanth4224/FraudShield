from __future__ import annotations

import argparse
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
problems = []

def bad(msg):
    problems.append(msg)
    print(f"  !! {msg}")

def timeit(fn, n=10, warm=2):
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append(1000 * (time.perf_counter() - t0))
    return statistics.median(ts)

def section(t):
    print(f"\n=== {t}")

def env():
    section("environment")
    print(f"  python {platform.python_version()}  {platform.platform()}  cpus {os.cpu_count()}")
    print(f"  project: {ROOT}")
    if "onedrive" in str(ROOT).lower():
        bad("project is inside OneDrive: OneDrive re-uploads models/, data/ and features/ (GBs) "
            "while you work -> disk + CPU load. Pause OneDrive or move the folder to C:\\FraudShield")
    try:
        out = subprocess.run(["powercfg", "/getactivescheme"], capture_output=True, text=True, timeout=5).stdout
        print(f"  power plan: {out.strip()}")
        if "saver" in out.lower():
            bad("Windows is on a power-saver plan: switch to Best performance and plug in the charger")
    except Exception:
        pass
    try:
        import torch
        print(f"  torch {torch.__version__}  cuda {torch.cuda.is_available()}  threads {torch.get_num_threads()}")
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            print(f"  GPU {p.name}  {p.total_memory / 2**30:.1f} GB")
        else:
            bad("PyTorch can't use the GPU: every model runs on the CPU (reinstall the cu124 build)")
    except Exception as e:
        bad(f"torch not importable: {e}")
    try:
        import mediapipe as mp
        ok = hasattr(mp, "solutions")
        print(f"  mediapipe {mp.__version__}  solutions={'yes' if ok else 'NO'}")
        if not ok:
            bad("mediapipe has no mp.solutions: face tracking is OFF (pip install mediapipe==0.10.14)")
    except Exception as e:
        bad(f"mediapipe not importable: {e}")
    print(f"  opencv threads {cv2.getNumThreads()}")
    for k in ("FRAUDSHIELD_CUSTOM", "FRAUDSHIELD_FACE_MODELS", "FRAUDSHIELD_CUDA_GRAPHS", "FRAUDSHIELD_FP32"):
        if os.environ.get(k):
            print(f"  {k}={os.environ[k]}")

def models():
    section("models: load + speed of one inference")
    from modules.detectors.custom_heads import CustomFaceDetector, CustomVoiceDetector
    from modules.detectors.face_classifier import FaceDeepfakeClassifier
    from modules.detectors.voice_classifier import VoiceDeepfakeClassifier
    rng = np.random.default_rng(0)
    crop = (rng.random((180, 150, 3)) * 255).astype(np.uint8)
    seg = (0.1 * rng.standard_normal(6 * 16000)).astype(np.float32)
    got = {}
    for cls, arg, what, limit in ((FaceDeepfakeClassifier, crop, "per face crop", 60),
                                  (VoiceDeepfakeClassifier, seg, "per 6 s segment", 150),
                                  (CustomVoiceDetector, seg, "per 6 s segment", 150),
                                  (CustomFaceDetector, crop, "per face crop", 80)):
        t0 = time.perf_counter()
        det = cls.get()
        load = time.perf_counter() - t0
        if det is None:
            print(f"  {cls.__name__:<24} {cls.status}")
            continue
        ms = timeit(lambda: det.predict([arg]))
        got[cls.__name__] = det
        print(f"  {cls.__name__:<24} {ms:7.1f} ms {what}   (load {load:.1f} s)  {det.name}")
        if ms > limit:
            bad(f"{cls.__name__} is slow ({ms:.0f} ms {what}); expected < {limit} ms on a GPU")
        if "cpu" in det.name and "cuda" not in det.name:
            bad(f"{cls.__name__} runs on the CPU")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  GPU memory in use: {torch.cuda.memory_allocated() / 2**20:.0f} MB")
    except Exception:
        pass
    return got

def face_pipeline(use_webcam):
    section("face pipeline (per video frame)")
    from modules.face_module import FaceStream, _ScreenFaceFinder
    frame = None
    if use_webcam:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        frames = []
        for _ in range(40):
            ok, f = cap.read()
            if ok:
                frames.append(f)
        cap.release()
        if frames:
            frame = frames[-1]
            print(f"  webcam frame {frame.shape[1]}x{frame.shape[0]} (look at the camera)")
    if frame is None:
        print("  no webcam frame: timing on a blank 1080p screen (search cost only)")
        frame = np.full((1080, 1920, 3), 40, np.uint8)
    screen = np.full((1080, 1920, 3), 30, np.uint8)
    h, w = frame.shape[:2]
    s = min(1.0, 900 / max(h, w))
    small = cv2.resize(frame, (int(w * s), int(h * s)))
    screen[60:60 + small.shape[0], 900:900 + small.shape[1]] = small
    ok, jpg = cv2.imencode(".jpg", screen, [cv2.IMWRITE_JPEG_QUALITY, 80])
    print(f"  JPEG decode 1080p       {timeit(lambda: cv2.imdecode(jpg, cv2.IMREAD_COLOR)):7.1f} ms")
    finder = _ScreenFaceFinder()
    ms = timeit(lambda: finder.find(screen), n=5)
    faces = finder.find(screen)
    print(f"  full-screen face search {ms:7.1f} ms  ({finder.name}, found {len(faces)} face(s))")
    if ms > 250:
        bad(f"full-screen face search takes {ms:.0f} ms; share just the call TAB, not the entire screen")
    fs = FaceStream(model="async")
    t, proc = 0.0, []
    for i in range(90):
        t += 1 / 15
        f = screen if i % 30 == 0 else screen
        t0 = time.perf_counter()
        fs.process(t, f)
        proc.append(1000 * (time.perf_counter() - t0))
    tracked = fs._target is not None
    print(f"  per-frame processing    {statistics.median(proc[20:]):7.1f} ms median, "
          f"{max(proc[20:]):.0f} ms worst  (face {'tracked' if tracked else 'NOT found'})")
    t0 = time.perf_counter()
    out = fs.analyze_window()
    print(f"  1 s analysis window     {1000 * (time.perf_counter() - t0):7.1f} ms  (status {out['status']})")
    if statistics.median(proc[20:]) > 45:
        bad("per-frame processing > 45 ms: the engine can't keep up with 15 fps")
    if not tracked and use_webcam:
        print("  (no face tracked: is the webcam covered / in use by another app?)")
    fs.close()

def voice_check(dets):
    section("your own recordings (data/own/real, data/own/fake)")
    own = ROOT / "data" / "own"
    if not own.exists():
        print("  data/own not found: record yourselves (see TRAINING.md) to test real-world accuracy")
        return
    from tools.extract_voice_features import segments_from
    from tools.score_head import files_in, load_any
    v = dets.get("VoiceDeepfakeClassifier")
    c = dets.get("CustomVoiceDetector")
    print(f"  {'file':<28} {'label':<5}  {'pretrained':>12}  {'ours':>6}   (P(fake); >0.5 = flagged)")
    rows = []
    for lab in ("real", "fake"):
        for f in files_in([own / lab]):
            try:
                y, sr = load_any(f)
                segs = segments_from(y, sr, 6.0, 5)
            except Exception as e:
                print(f"  {f.name[:28]:<28} {lab:<5}  error: {e}")
                continue
            if not segs:
                print(f"  {f.name[:28]:<28} {lab:<5}  no speech found")
                continue
            pv = float(np.mean(v.predict(segs))) if v else float("nan")
            pc = float(np.mean(c.predict(segs))) if c else float("nan")
            rows.append((lab, pv, pc))
            print(f"  {f.name[:28]:<28} {lab:<5}  {pv:12.2f}  {pc:6.2f}")
    for name, k in (("pretrained", 1), ("ours", 2)):
        r = [x[k] for x in rows if x[0] == "real" and not np.isnan(x[k])]
        fk = [x[k] for x in rows if x[0] == "fake" and not np.isnan(x[k])]
        if r:
            fl = sum(p >= 0.5 for p in r)
            print(f"  {name}: real flagged {fl}/{len(r)}" + (f", fakes caught {sum(p >= 0.5 for p in fk)}/{len(fk)}" if fk else ""))
            if fl > len(r) / 4:
                bad(f"{name} flags {fl}/{len(r)} of your REAL voices: don't trust it in the demo "
                    + ("(set FRAUDSHIELD_CUSTOM=0)" if name == "ours" else ""))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-webcam", action="store_true")
    args = ap.parse_args()
    sys.path.insert(0, str(ROOT))
    env()
    dets = models()
    face_pipeline(not args.no_webcam)
    voice_check(dets)
    section("summary")
    if problems:
        for p in problems:
            print(f"  - {p}")
    else:
        print("  no problems found")

if __name__ == "__main__":
    main()

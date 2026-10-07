"""
Which "AI-generated picture" detector actually works on YOUR videos?

Runs several public AI-image detectors on frames of labelled videos and prints, per model,
the average P(AI-generated) for every video and how cleanly it separates real from fake.

    python -m tools.try_scene_models --real "C:\\...\\fs\\real\\*.mp4" --fake "C:\\...\\fs\\fake\\video 5.mp4"

Only put FULLY generated videos (Sora / Veo / Kling style) under --fake here: face-swap fakes have
a real body and background, which these detectors are right to call real.
Each model downloads once (100-400 MB). Models that fail to load are skipped.
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import cv2
import numpy as np

CANDIDATES = [
    "buildborderless/CommunityForensics-DeepfakeDet-ViT",
    "Ateeqq/ai-vs-human-image-detector",
    "Organika/sdxl-detector",
    "umm-maybe/AI-image-detector",
]
FAKE_WORDS = ("ai", "fake", "artificial", "generated", "synthetic", "sdxl", "machine")

def frames_of(path, n=8, seconds=15.0):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total = int(min(cap.get(cv2.CAP_PROP_FRAME_COUNT) or fps * seconds, fps * seconds))
    out = []
    for k in np.linspace(0, max(0, total - 1), n).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(k))
        ok, f = cap.read()
        if ok:
            out.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    return out

def p_fake(result):
    if len(result) == 1:
        return float(result[0]["score"])
    fake = [r["score"] for r in result if any(w in str(r["label"]).lower() for w in FAKE_WORDS)
            and "human" not in str(r["label"]).lower() and "real" not in str(r["label"]).lower()]
    return float(sum(fake)) if fake else float("nan")

def main():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--real", nargs="+", required=True)
    ap.add_argument("--fake", nargs="+", required=True)
    ap.add_argument("--models", nargs="*", default=CANDIDATES)
    args = ap.parse_args()
    expand = lambda xs: [Path(f) for x in xs for f in (glob.glob(x) or [x]) if Path(f).exists()]
    vids = [(v, 0) for v in expand(args.real)] + [(v, 1) for v in expand(args.fake)]
    if not vids:
        raise SystemExit("no videos found")
    frames = {v: frames_of(v) for v, _ in vids}

    import torch
    from transformers import pipeline
    dev = 0 if torch.cuda.is_available() else -1
    from PIL import Image
    summary = []
    for repo in args.models:
        print(f"\n=== {repo}")
        try:
            clf = pipeline("image-classification", model=repo, device=dev, top_k=None)
        except Exception as e:
            print(f"  could not load: {str(e)[:150]}")
            continue
        scores = {}
        for v, lab in vids:
            ps = [p_fake(clf(Image.fromarray(f))) for f in frames[v]]
            ps = [p for p in ps if np.isfinite(p)]
            scores[v] = float(np.mean(ps)) if ps else float("nan")
            print(f"  {'FAKE' if lab else 'real'}  P(AI) {scores[v]:.2f}   {v.name}")
        real = [scores[v] for v, l in vids if l == 0 and np.isfinite(scores[v])]
        fake = [scores[v] for v, l in vids if l == 1 and np.isfinite(scores[v])]
        if real and fake:
            gap = min(fake) - max(real)
            print(f"  -> highest real {max(real):.2f}, lowest fake {min(fake):.2f}: "
                  + ("SEPARATES them" if gap > 0.1 else "does NOT separate them"))
            summary.append((gap, repo, max(real), min(fake)))
        del clf
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if summary:
        print("\nbest first (gap = lowest fake - highest real; bigger is better):")
        for gap, repo, r, f in sorted(summary, reverse=True):
            print(f"  gap {gap:+.2f}   real max {r:.2f}   fake min {f:.2f}   {repo}")

if __name__ == "__main__":
    main()

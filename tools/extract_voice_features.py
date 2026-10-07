"""
Step 1 (voice): turn real / fake speech clips into XLS-R feature vectors.

    python -m tools.extract_voice_features --real data\\voice\\real --fake data\\voice\\fake --augment
    python -m tools.extract_voice_features --manifest data\\voice\\manifest.csv --augment

Each clip -> up to 3 speech segments of 6 s (the same window the live engine
scores) -> the frozen XLS-R 300M model -> one 1024-number vector per segment
for each of 8 layers. Saved to features/voice.npz. Re-running resumes where it
stopped.

--augment adds a "phone-call" copy of every segment (8 kHz band limit + noise),
applied to real and fake alike, so the head learns fake-vs-real rather than
studio-vs-phone.

First run downloads XLS-R (~1.3 GB) into models/hf-cache.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from modules.detectors.custom_heads import CHANNELS, VOICE_BACKBONE, VOICE_LAYERS, VoiceBackbone, normalize_channel
from modules.voice_module import HOP, SR, _frames, _resample, load_audio, speech_segments
from tools.custom_common import AUDIO_EXT, collect, load_any_audio, load_partial, save_features

def speech_mask(y16: np.ndarray) -> np.ndarray:
    e = 10 * np.log10(np.mean(_frames(y16, 400, HOP) ** 2, 1) + 1e-10)
    if not len(e):
        return np.zeros(0, bool)
    thr = max(np.percentile(e, 10) + 6.0, np.percentile(e, 95) - 30.0)
    return e > thr

def call_like(seg: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    x = seg.astype(np.float32)
    if rng.random() < 0.6:
        x = _resample(_resample(x, SR, 8000), 8000, SR)[: len(seg)]
    p = float(np.mean(x ** 2)) + 1e-10
    snr = rng.uniform(15, 35)
    x = x + rng.normal(0, np.sqrt(p / 10 ** (snr / 10)), len(x)).astype(np.float32)
    return x

def segments_for(path: str, seg_s: float, max_n: int):
    y, sr = load_any_audio(path, max_seconds=max(40.0, seg_s * max_n + 5))
    return segments_from(y, sr, seg_s, max_n)

def segments_from(y: np.ndarray, sr: int, seg_s: float = 6.0, max_n: int = 3):
    if len(y) < 2 * sr:
        return []
    y16 = _resample(y, sr)
    peak = float(np.max(np.abs(y16))) or 1.0
    y16 = (y16 / peak * 0.9).astype(np.float32)
    return speech_segments(y16, speech_mask(y16), seg_s=seg_s, max_n=max_n, min_s=2.0)

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--real", nargs="*", help="folders of genuine speech")
    ap.add_argument("--fake", nargs="*", help="folders of synthetic / cloned speech")
    ap.add_argument("--manifest", help="CSV with path,label,group[,split][,tag]")
    ap.add_argument("--out", default="features/voice.npz")
    ap.add_argument("--seg", type=float, default=6.0, help="segment length in seconds (live engine: 6)")
    ap.add_argument("--per-file", type=int, default=3, help="max segments per clip")
    ap.add_argument("--augment", action="store_true", help="add a phone-call copy of every segment")
    ap.add_argument("--limit", type=int, default=0, help="only the first N files per label (quick test)")
    ap.add_argument("--backbone", default=VOICE_BACKBONE)
    ap.add_argument("--channel", choices=list(CHANNELS), default="call",
                    help="channel normalisation applied to EVERY segment (and live): call = 80-7000 Hz + "
                         "loudness + noise floor (default), phone = 300-3400 Hz, none = raw audio")
    args = ap.parse_args()

    items = collect(args.real, args.fake, args.manifest, AUDIO_EXT)
    if args.limit:
        items = [it for lab in (0, 1) for it in [i for i in items if i["label"] == lab][: args.limit]]
    out = Path(args.out)
    rows = load_partial(out, args.channel)
    done = set(rows["path"])
    todo = [it for it in items if it["path"] not in done]
    print(f"{len(todo)} clips to process -> {out}")

    bb = VoiceBackbone(args.backbone, max_layer=max(VOICE_LAYERS))
    print(f"backbone {args.backbone} on {bb.device}")
    rng = np.random.default_rng(0)
    t0, skipped = time.time(), 0
    for i, it in enumerate(todo, 1):
        try:
            segs = segments_for(it["path"], args.seg, args.per_file)
        except Exception as e:
            print(f"  skip {it['path']}: {e}")
            segs = []
        if not segs:
            skipped += 1
            continue
        variants = [(normalize_channel(s, args.channel), 0) for s in segs]
        if args.augment:
            variants += [(normalize_channel(call_like(s, rng), args.channel), 1) for s in segs]
        X = bb.features([v[0] for v in variants], VOICE_LAYERS)
        for x, (_, aug) in zip(X, variants):
            rows["X"].append(x)
            rows["y"].append(it["label"])
            rows["aug"].append(aug)
            for k in ("group", "split", "tag", "path"):
                rows[k].append(it[k])
        if i % 100 == 0 or i == len(todo):
            save_features(out, rows, VOICE_LAYERS, args.backbone, "voice", args.channel)
            rate = i / (time.time() - t0)
            print(f"  {i}/{len(todo)} clips  ({rate:.1f}/s, ~{(len(todo) - i) / rate / 60:.0f} min left)")
    save_features(out, rows, VOICE_LAYERS, args.backbone, "voice", args.channel)
    print(f"done: {len(rows['y'])} segments from {len(set(rows['path']))} clips "
          f"({skipped} clips skipped: <2 s or no speech) -> {out}")

if __name__ == "__main__":
    main()

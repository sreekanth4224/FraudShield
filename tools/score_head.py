"""
Score audio files with the app's voice detector and OUR trained head, e.g. your own phone recordings.

    python -m tools.score_head --real data\own\real --fake data\own\fake
    python -m tools.score_head my_voice.wav cloned_voice.wav

Prints P(fake) per file (mean over up to 5 speech segments of 6 s) and, when
both --real and --fake are given, the AUC and how many files each side would
be flagged at P(fake) >= 0.5 — the honest check of whether the head learned
"real vs fake" or just "which dataset".

WAV / FLAC work directly. Phone formats (.m4a, .mp3, .aac, .opus, .ogg) are
converted with ffmpeg if it is installed (winget install Gyan.FFmpeg).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from modules.detectors.custom_heads import CustomVoiceDetector, head_path
from tools.custom_common import load_any_audio
from tools.extract_voice_features import segments_from

EXT = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aac", ".opus", ".webm", ".mp4", ".3gp", ".amr"}

def load_any(path: Path):
    return load_any_audio(path, 60.0)

def files_in(paths):
    out = []
    for p in paths or []:
        p = Path(p)
        if p.is_dir():
            out += sorted(f for f in p.rglob("*") if f.suffix.lower() in EXT)
        elif p.exists():
            out.append(p)
        else:
            print(f"not found: {p}")
    return out

def auc(scores, labels):
    s, y = np.asarray(scores, float), np.asarray(labels, int)
    pos, neg = s[y == 1], s[y == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    return float(((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum())
                 / (len(pos) * len(neg)))

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*", help="audio files or folders (unlabelled)")
    ap.add_argument("--real", nargs="*", help="folders / files of genuine speech")
    ap.add_argument("--fake", nargs="*", help="folders / files of cloned / synthetic speech")
    ap.add_argument("--which", choices=["both", "pretrained", "ours"], default="both",
                    help="pretrained = the app's main voice detector (DF-Arena by default), ours = our head")
    args = ap.parse_args()
    from modules.detectors.voice_classifier import VoiceDeepfakeClassifier
    models = []
    if args.which in ("both", "pretrained"):
        d = VoiceDeepfakeClassifier.get()
        if d is None:
            print(f"pretrained voice detector failed to load: {VoiceDeepfakeClassifier.status}")
        else:
            models.append(("pretrained", d))
    if args.which in ("both", "ours"):
        if not head_path("voice").exists():
            print("no trained head of our own yet (tools.train_head) - scoring the pretrained detector only")
        else:
            d = CustomVoiceDetector.get()
            if d is None:
                print(f"our voice head failed to load: {CustomVoiceDetector.status}")
            else:
                models.append(("ours", d))
    if not models:
        raise SystemExit("nothing to score with")
    for name, d in models:
        print(f"{name:<11} {d.name}")
    print()

    jobs = [(f, None) for f in files_in(args.files)] + [(f, 0) for f in files_in(args.real)] + \
           [(f, 1) for f in files_in(args.fake)]
    if not jobs:
        raise SystemExit("no audio files given")
    scores = {n: [] for n, _ in models}
    labels = []
    print("".join(f"{n:>11}" for n, _ in models) + f"  {'label':<6} file      (P(fake); >= 0.5 = flagged)")
    for f, lab in jobs:
        try:
            y, sr = load_any(f)
            segs = segments_from(y, sr, 6.0, 5)
        except Exception as e:
            print(f"{'error':>11}  {f.name}: {e}")
            continue
        if not segs:
            print(f"{'no speech':>11}  {f.name} (needs >= 2 s of speech)")
            continue
        ps = {n: float(np.mean(d.predict(segs))) for n, d in models}
        tag = "" if lab is None else ("fake" if lab else "real")
        print("".join(f"{ps[n]:>11.3f}" for n, _ in models) + f"  {tag:<6} {f.name}")
        if lab is not None:
            labels.append(lab)
            for n in ps:
                scores[n].append(ps[n])
    if labels:
        y = np.array(labels)
        for n, _ in models:
            s = np.array(scores[n])
            print(f"\n{n}:")
            if (y == 0).any():
                print(f"  real files flagged (P >= 0.5): {int((s[y == 0] >= 0.5).sum())}/{int((y == 0).sum())}")
            if (y == 1).any():
                print(f"  fake files caught  (P >= 0.5): {int((s[y == 1] >= 0.5).sum())}/{int((y == 1).sum())}")
            if (y == 0).any() and (y == 1).any():
                print(f"  AUC on these files: {auc(s, y):.3f}")

if __name__ == "__main__":
    main()

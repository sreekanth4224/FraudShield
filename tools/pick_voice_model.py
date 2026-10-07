"""
MEASURE which voice-deepfake model is most accurate on YOUR labelled clips, make the app use it,
and calibrate its scores — instead of guessing.

    python -m tools.pick_voice_model
    python -m tools.pick_voice_model --manifest data\\voice\\manifest.csv --real data\\own\\real --fake data\\own\\fake

Candidates: itw (wav2vec2 In-the-Wild, the original), arena-500m, arena-1b, and every ensemble
of them (log-odds averaged — computed from the single-model scores, no extra inference).
Each clip is scored twice: as recorded, and through a simulated call channel (8 kHz band
limit + noise), because the live app hears call audio.

Prints AUC, EER, % of real clips flagged and % of fakes caught for every candidate, then:
  * models/voice_choice.json   <- the winner; the app loads it on the next start
  * models/calibration.json    <- "voice_model" mapping for the winner, so a real voice lands
                                  in the green zone and a fake one in the red zone
Set FRAUDSHIELD_VOICE_MODEL to override the choice by hand.
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import time
from datetime import date
from pathlib import Path

import numpy as np

from modules import calibration
from modules.detectors import CALIBRATION_FILE, MODELS_DIR
from modules.detectors.voice_classifier import CHOICE_FILE, VoiceDeepfakeClassifier
from modules.voice_module import HOP, SR, _frames, _resample, load_audio, speech_segments
from tools.custom_common import collect

AUDIO_EXT = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".opus", ".aac", ".webm", ".3gp", ".amr"}

def load_any_audio(path, max_seconds=40.0):
    import shutil
    import subprocess
    import tempfile
    path = Path(path)
    if path.suffix.lower() in (".wav", ".flac"):
        return load_audio(str(path), max_seconds=max_seconds)
    ff = shutil.which("ffmpeg")
    if ff is None:
        return load_audio(str(path), max_seconds=max_seconds)
    with tempfile.TemporaryDirectory() as d:
        wav = Path(d) / "a.wav"
        subprocess.run([ff, "-v", "error", "-y", "-i", str(path), "-ac", "1", "-ar", "16000",
                        "-t", str(max_seconds), str(wav)], check=True)
        return load_audio(str(wav), max_seconds=max_seconds)

def segments_from(y, sr, seg_s=6.0, max_n=2):
    if len(y) < 2 * sr:
        return []
    y16 = _resample(y, sr)
    peak = float(np.max(np.abs(y16))) or 1.0
    y16 = (y16 / peak * 0.9).astype(np.float32)
    e = 10 * np.log10(np.mean(_frames(y16, 400, HOP) ** 2, 1) + 1e-10)
    mask = e > max(np.percentile(e, 10) + 6.0, np.percentile(e, 95) - 30.0)
    return speech_segments(y16, mask, seg_s=seg_s, max_n=max_n, min_s=2.0)

def call_like(seg, rng):
    x = seg.astype(np.float32)
    if rng.random() < 0.6:
        x = _resample(_resample(x, SR, 8000), 8000, SR)[: len(seg)]
    p = float(np.mean(x ** 2)) + 1e-10
    snr = rng.uniform(15, 35)
    return x + rng.normal(0, np.sqrt(p / 10 ** (snr / 10)), len(x)).astype(np.float32)

SINGLES = ["itw", "arena-500m", "arena-1b"]

def auc(s, y):
    s, y = np.asarray(s, float), np.asarray(y, int)
    pos, neg = s[y == 1], s[y == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    return float(((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum())
                 / (len(pos) * len(neg)))

def eer(s, y):
    s, y = np.asarray(s, float), np.asarray(y, int)
    best = (2.0, 1.0)
    for t in np.unique(np.quantile(s, np.linspace(0, 1, 201))):
        far, frr = float(np.mean(s[y == 0] >= t)), float(np.mean(s[y == 1] < t))
        if abs(far - frr) < best[0]:
            best = (abs(far - frr), (far + frr) / 2)
    return best[1]

def sig(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

def logit(p):
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))

def sample(items, n, seed):
    rng = random.Random(seed)
    out = []
    for lab in (0, 1):
        pool = [it for it in items if it["label"] == lab]
        by = {}
        for it in pool:
            by.setdefault(it["group"], []).append(it)
        for g in by.values():
            rng.shuffle(g)
        groups = list(by.values())
        rng.shuffle(groups)
        picked = []
        while len(picked) < n and any(groups):
            for g in groups:
                if g and len(picked) < n:
                    picked.append(g.pop())
        out += picked
    return out

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", nargs="*", default=None,
                    help="manifest CSV(s) (default: data/voice/manifest.csv if it exists)")
    ap.add_argument("--real", nargs="*", help="extra folders of genuine speech (e.g. data/own/real)")
    ap.add_argument("--fake", nargs="*", help="extra folders of cloned speech (e.g. data/own/fake)")
    ap.add_argument("--n", type=int, default=150, help="clips per label from the manifest (default 150)")
    ap.add_argument("--candidates", default=",".join(SINGLES), help="single models to try")
    ap.add_argument("--no-call", action="store_true", help="skip the simulated call-channel copies")
    ap.add_argument("--dry-run", action="store_true", help="report only, don't change the app's choice")
    args = ap.parse_args()

    items = []
    manifests = args.manifest if args.manifest is not None else (
        ["data/voice/manifest.csv"] if Path("data/voice/manifest.csv").exists() else [])
    for m in manifests:
        items += sample(collect(None, None, m, AUDIO_EXT), args.n, seed=0)
    own = []
    if args.real or args.fake:
        dirs = [d for d in (args.real or []) + (args.fake or []) if Path(d).exists()]
        try:
            own = collect([d for d in (args.real or []) if Path(d).exists()],
                          [d for d in (args.fake or []) if Path(d).exists()], None, AUDIO_EXT) if dirs else []
        except SystemExit:
            print("  (no files in the --real/--fake folders yet — using the manifest only)")
    for it in own:
        it["tag"] = "own"
    items += own
    if not items:
        raise SystemExit("no labelled clips: give --manifest and/or --real/--fake folders")

    rng = np.random.default_rng(0)
    clips = []
    t0 = time.time()
    for i, it in enumerate(items, 1):
        try:
            y, sr = load_any_audio(it["path"], 40.0)
            segs = segments_from(y, sr, 6.0, 2)
        except Exception as e:
            print(f"  skip {Path(it['path']).name}: {e}")
            continue
        if not segs:
            continue
        tag = it["tag"] or "data"
        clips.append((it["label"], tag, "clean", segs))
        if not args.no_call:
            clips.append((it["label"], tag, "call", [call_like(s, rng) for s in segs]))
    y = np.array([c[0] for c in clips])
    cond = np.array([c[2] for c in clips])
    tags = np.array([c[1] for c in clips])
    print(f"{len(clips)} clip versions ({int((y == 0).sum())} real / {int((y == 1).sum())} fake) "
          f"from {len(items)} files  [{time.time() - t0:.0f} s]")
    if y.min() == y.max():
        raise SystemExit("need both real and fake clips")

    Z = {}
    for name in [c.strip() for c in args.candidates.split(",") if c.strip()]:
        t0 = time.time()
        try:
            clf = VoiceDeepfakeClassifier(backend=name)
        except Exception as e:
            print(f"  {name}: could not load ({e})")
            continue
        if clf.backend != name:
            print(f"  {name}: could not load ({getattr(clf, 'fallback_reason', 'fell back to ' + clf.backend)})")
            del clf
            continue
        z = []
        for k, c in enumerate(clips, 1):
            z.append(float(np.mean(logit(clf.predict(c[3])))))
            if k % 100 == 0:
                print(f"  {name}: {k}/{len(clips)}")
        Z[name] = np.array(z)
        print(f"  {name}: scored {len(clips)} in {time.time() - t0:.0f} s  ({clf.name})")
        del clf
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
    if not Z:
        raise SystemExit("no model could be loaded")
    singles = list(Z)
    for r in range(2, len(singles) + 1):
        for combo in itertools.combinations(singles, r):
            Z[",".join(combo)] = np.mean([Z[c] for c in combo], axis=0)

    conds = [c for c in ("clean", "call") if (cond == c).any()]
    print(f"\n{'model':<28}" + "".join(f"{'AUC ' + c:>11}" for c in conds)
          + f"{'EER':>8}{'real flagged':>14}{'fakes caught':>14}")
    rows = []
    for name, z in Z.items():
        p = sig(z)
        aucs = [auc(z[cond == c], y[cond == c]) for c in conds]
        r = {"name": name, "auc": float(np.nanmean(aucs)), "aucs": aucs, "eer": eer(z, y),
             "real_flagged": float(np.mean(p[y == 0] >= 0.5)), "fake_caught": float(np.mean(p[y == 1] >= 0.5))}
        rows.append(r)
    rows.sort(key=lambda r: (-r["auc"], r["eer"]))
    for r in rows:
        print(f"{r['name']:<28}" + "".join(f"{a:>11.3f}" for a in r["aucs"])
              + f"{100 * r['eer']:>7.1f}%{100 * r['real_flagged']:>13.0f}%{100 * r['fake_caught']:>13.0f}%")
    best = rows[0]
    for t in sorted(set(tags.tolist())):
        m = tags == t
        print(f"   [{t}] {best['name']}: AUC {auc(Z[best['name']][m], y[m]):.3f}  "
              f"({int((y[m] == 0).sum())} real / {int((y[m] == 1).sum())} fake)")

    from tools.evaluate import fit_logistic
    zb = Z[best["name"]]
    w, b = fit_logistic([[float(v)] for v in sig(zb)], y.tolist(), prior_w=[1.0], lam=8.0)
    risk = sig(w[0] * zb + b)
    print(f"\nwinner: {best['name']}   mean AUC {best['auc']:.3f}, EER {100 * best['eer']:.1f} %")
    print(f"calibrated (risk = sigmoid({w[0]:.2f}·logodds + {b:.2f})):")
    for thr, lab in ((0.35, "Suspicious"), (0.65, "Likely fake")):
        print(f"  voice risk ≥ {int(100 * thr)} ({lab}): {100 * np.mean(risk[y == 0] >= thr):.0f} % of real clips, "
              f"{100 * np.mean(risk[y == 1] >= thr):.0f} % of fakes")
    if best["auc"] < 0.75:
        print("note: even the best model separates these clips poorly — check the labels / audio")
    if args.dry_run:
        print("\n(dry run — nothing saved)")
        return

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    CHOICE_FILE.write_text(json.dumps({"backend": best["name"], "auc": best["auc"], "eer": best["eer"],
                                       "measured": date.today().isoformat(), "clips": len(clips),
                                       "all": {r["name"]: round(r["auc"], 4) for r in rows}}, indent=2),
                           encoding="utf-8")
    try:
        cal = json.loads(CALIBRATION_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        cal = {}
    entry = {"w": [round(float(w[0]), 4)], "b": round(float(b), 4), "backend": best["name"],
             "fitted_on": f"{len(clips)} clips, tools.pick_voice_model"}
    cal["voice_model"] = entry
    cal[calibration.key("voice_model", [best["name"]])] = entry
    CALIBRATION_FILE.write_text(json.dumps(cal, indent=2), encoding="utf-8")
    print(f"\nsaved {CHOICE_FILE.name} (app now uses: {best['name']}) and the voice calibration in "
          f"{CALIBRATION_FILE.name}.  Restart `python app.py`.")

if __name__ == "__main__":
    main()

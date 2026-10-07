"""
Learn from your labelled videos (no video is stored — see modules/evidence.py).

1. Label videos as you check them (each run adds one line of check results):
       python -m tools.check_video "C:\\videos\\real\\*.mp4" --label real
       python -m tools.check_video "C:\\videos\\fake\\*.mp4" --label fake
2. Train:
       python -m tools.train_evidence

Prints leave-one-out results (each video scored by a model that never saw it) and saves
models/evidence_model.json. The live app uses it only if it is "trusted": at least
MIN_PER_CLASS real and fake videos and a leave-one-out AUC >= MIN_AUC. Retrain whenever you add videos.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from modules.evidence import LOG, MODEL
from tools.train_head import auc, train_logreg

MIN_PER_CLASS = 10
MIN_AUC = 0.80

def load_rows(path):
    rows = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line:
            r = json.loads(line)
            if r.get("label") in ("real", "fake"):
                rows.append(r)
    latest = {}
    for r in rows:
        latest[(r["file"], r["label"])] = r
    return list(latest.values())

def fit(X, y, keys, lam):
    fill = np.nanmedian(X, axis=0)
    fill = np.where(np.isfinite(fill), fill, 0.5)
    Xf = np.where(np.isfinite(X), X, fill)
    mu, sd = Xf.mean(0), Xf.std(0) + 1e-3
    m = train_logreg((Xf - mu) / sd, y, lam)
    return {"keys": keys, "fill": fill.tolist(), "mu": mu.tolist(), "sd": sd.tolist(),
            "w": np.asarray(m["w"], float).tolist(), "b": float(m["b"])}

def score(model, X):
    fill, mu, sd = (np.array(model[k]) for k in ("fill", "mu", "sd"))
    Xf = np.where(np.isfinite(X), X, fill)
    z = ((Xf - mu) / sd) @ np.array(model["w"]) + model["b"]
    return 1 / (1 + np.exp(-np.clip(z, -30, 30)))

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", default=str(LOG))
    ap.add_argument("--lam", type=float, default=0.05, help="L2 strength (bigger = simpler model)")
    args = ap.parse_args()
    from pathlib import Path
    if not Path(args.log).exists():
        raise SystemExit("no labelled videos yet. First run, for videos you KNOW are real / fake:\n"
                         "  python -m tools.check_video \"<folder with real videos>\\*.mp4\" --label real\n"
                         "  python -m tools.check_video \"<folder with fake videos>\\*.mp4\" --label fake")
    rows = load_rows(args.log)
    y = np.array([1 if r["label"] == "fake" else 0 for r in rows])
    n_real, n_fake = int((y == 0).sum()), int(y.sum())
    print(f"{len(rows)} labelled videos: {n_real} real, {n_fake} fake  ({args.log})")
    if n_real < 2 or n_fake < 2:
        raise SystemExit("need at least 2 real and 2 fake videos (aim for 10+ of each)")
    allk = sorted({k for r in rows for k in r["features"]})
    keys = [k for k in allk if np.mean([r["features"].get(k) is not None for r in rows]) >= 0.3]
    X = np.array([[np.nan if r["features"].get(k) is None else r["features"][k] for k in keys] for r in rows], float)

    loo = np.zeros(len(rows))
    for i in range(len(rows)):
        tr = np.arange(len(rows)) != i
        if len(set(y[tr])) < 2:
            loo[i] = 0.5
            continue
        loo[i] = score(fit(X[tr], y[tr], keys, args.lam), X[i:i + 1])[0]
    a = auc(loo, y)
    caught = float(np.mean(loo[y == 1] >= 0.5))
    flagged = float(np.mean(loo[y == 0] >= 0.5))
    print(f"\nleave-one-out (each video judged by a model that never saw it):")
    print(f"  AUC {a:.3f}   fakes caught {100 * caught:.0f}%   real flagged {100 * flagged:.0f}%")
    for r, p, yy in sorted(zip(rows, loo, y), key=lambda t: -t[1]):
        mark = "  <- wrong" if (p >= 0.5) != bool(yy) else ""
        print(f"    {p:5.2f}  {r['label']:<4}  {r['file']}{mark}")

    model = fit(X, y, keys, args.lam)
    trusted = bool(n_real >= MIN_PER_CLASS and n_fake >= MIN_PER_CLASS and a >= MIN_AUC)
    model.update(trusted=trusted, loo_auc=a, n_real=n_real, n_fake=n_fake, lam=args.lam)
    w = np.array(model["w"])
    print("\nchecks that matter most (+ = points to fake):")
    for k, wk in sorted(zip(keys, w), key=lambda t: -abs(t[1]))[:8]:
        print(f"    {wk:+.2f}  {k}")
    MODEL.parent.mkdir(parents=True, exist_ok=True)
    MODEL.write_text(json.dumps(model, indent=1), encoding="utf-8")
    print(f"\nsaved {MODEL}")
    if trusted:
        print("TRUSTED: the live app and check_video will use it (restart python app.py).")
    else:
        why = []
        if n_real < MIN_PER_CLASS or n_fake < MIN_PER_CLASS:
            why.append(f"needs >= {MIN_PER_CLASS} real and {MIN_PER_CLASS} fake videos")
        if not a >= MIN_AUC:
            why.append(f"leave-one-out AUC {a:.2f} < {MIN_AUC}")
        print("NOT used yet: " + "; ".join(why) + ". Label more videos and train again.")

if __name__ == "__main__":
    main()

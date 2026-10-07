"""
Step 2: train our own detector head on the extracted features.

    python -m tools.train_head features\\voice.npz --label "Indian-language voice detector (ours)"
    python -m tools.train_head features\\face.npz  --label "Webcam deepfake detector (ours)"

What it does
  * splits by GROUP (speaker / person): test speakers are never heard in training,
    so the test score is honest. A `split` column in the manifest overrides this.
  * for every saved backbone layer it trains
      - logistic regression (4 regularisation strengths)
      - a small MLP (1024 -> 256 -> 1, dropout, early stopping)
    and picks the best by AUC on a validation slice of the TRAINING speakers
  * refits the winner on all training data and reports it ONCE on the test set
  * saves models/custom/<voice|face>_head.npz + a JSON report; the live engine
    picks it up on the next start (no code changes)

Pure numpy + scipy: no GPU needed for this step, it takes a few minutes.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from datetime import date
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.stats import rankdata

from modules.detectors.custom_heads import Head, head_path
from tools.custom_common import hashed

def auc(scores, labels) -> float:
    s, y = np.asarray(scores, float), np.asarray(labels, int)
    n1, n0 = int(y.sum()), int(len(y) - y.sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(s)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))

def eer(scores, labels) -> float:
    s, y = np.asarray(scores, float), np.asarray(labels, int)
    if y.min() == y.max():
        return float("nan")
    best = (2.0, 1.0)
    for t in np.unique(np.quantile(s, np.linspace(0, 1, 401))):
        far = float(np.mean(s[y == 0] >= t))
        frr = float(np.mean(s[y == 1] < t))
        if abs(far - frr) < best[0]:
            best = (abs(far - frr), (far + frr) / 2)
    return best[1]

def per_clip(p, y, paths):
    acc = defaultdict(list)
    lab = {}
    for pi, yi, pa in zip(p, y, paths):
        acc[pa].append(pi)
        lab[pa] = yi
    keys = list(acc)
    return np.array([np.mean(acc[k]) for k in keys]), np.array([lab[k] for k in keys]), keys

def _sig(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

def balanced_weights(y):
    y = np.asarray(y)
    n1 = max(1, int(y.sum()))
    n0 = max(1, int(len(y) - y.sum()))
    return np.where(y == 1, len(y) / (2 * n1), len(y) / (2 * n0)).astype(np.float64)

def train_logreg(X, y, lam):
    X = X.astype(np.float64)
    y = y.astype(np.float64)
    sw = balanced_weights(y)
    S = sw.sum()

    def f(th):
        w, b = th[:-1], th[-1]
        z = X @ w + b
        loss = float(np.sum(sw * (np.logaddexp(0, z) - y * z)) / S + lam * w @ w)
        g = sw * (_sig(z) - y) / S
        return loss, np.concatenate([X.T @ g + 2 * lam * w, [g.sum()]])

    th = minimize(f, np.zeros(X.shape[1] + 1), jac=True, method="L-BFGS-B", options={"maxiter": 500}).x
    return {"kind": "logreg", "w": th[:-1].astype(np.float32), "b": np.float32(th[-1])}

def mlp_forward(P, X):
    h = np.maximum(X @ P["W1"] + P["b1"], 0.0)
    return h @ P["W2"] + P["b2"]

def train_mlp(X, y, epochs=80, Xv=None, yv=None, hidden=256, lr=1e-3, wd=1e-4, drop=0.3,
              patience=10, batch=256, seed=0):
    rng = np.random.default_rng(seed)
    X = X.astype(np.float32)
    y = y.astype(np.float32)
    sw = balanced_weights(y).astype(np.float32)
    d = X.shape[1]
    P = {"W1": rng.normal(0, np.sqrt(2 / d), (d, hidden)).astype(np.float32), "b1": np.zeros(hidden, np.float32),
         "W2": rng.normal(0, np.sqrt(1 / hidden), hidden).astype(np.float32), "b2": np.float32(0.0)}
    m = {k: np.zeros_like(v) for k, v in P.items()}
    v2 = {k: np.zeros_like(v) for k, v in P.items()}
    step, best, best_ep, bad = 0, (-1.0, None), 0, 0
    for ep in range(1, epochs + 1):
        for idx in np.array_split(rng.permutation(len(X)), max(1, len(X) // batch)):
            xb, yb, wb = X[idx], y[idx], sw[idx]
            z1 = xb @ P["W1"] + P["b1"]
            a1 = np.maximum(z1, 0.0)
            mask = (rng.random(a1.shape) > drop).astype(np.float32) / (1 - drop)
            a1d = a1 * mask
            p = _sig(a1d @ P["W2"] + P["b2"])
            g2 = (wb * (p - yb) / wb.sum()).astype(np.float32)
            ga1 = np.outer(g2, P["W2"]) * mask * (z1 > 0)
            G = {"W2": a1d.T @ g2 + wd * P["W2"], "b2": np.float32(g2.sum()),
                 "W1": xb.T @ ga1 + wd * P["W1"], "b1": ga1.sum(0)}
            step += 1
            for k in P:
                m[k] = 0.9 * m[k] + 0.1 * G[k]
                v2[k] = 0.999 * v2[k] + 0.001 * G[k] ** 2
                mh, vh = m[k] / (1 - 0.9 ** step), v2[k] / (1 - 0.999 ** step)
                P[k] = (P[k] - lr * mh / (np.sqrt(vh) + 1e-8)).astype(np.float32)
        if Xv is not None:
            a = auc(mlp_forward(P, Xv), yv)
            if a > best[0] + 1e-4:
                best, best_ep, bad = (a, {k: np.copy(v) for k, v in P.items()}), ep, 0
            else:
                bad += 1
                if bad >= patience:
                    break
    if Xv is not None:
        return {"kind": "mlp", **best[1]}, best_ep, best[0]
    return {"kind": "mlp", **P}, epochs, None

def predict(model, X):
    if model["kind"] == "logreg":
        return _sig(X @ model["w"] + model["b"])
    return _sig(mlp_forward(model, X))

def load(paths):
    parts = [dict(np.load(p, allow_pickle=False)) for p in paths]
    bb = {str(p["backbone"]) for p in parts}
    lay = {tuple(p["layers"].tolist()) for p in parts}
    if len(bb) > 1 or len(lay) > 1:
        raise SystemExit("feature files come from different backbones / layers — extract them the same way")
    ch = {str(p["channel"]) if "channel" in p else "none" for p in parts}
    if len(ch) > 1:
        raise SystemExit(f"feature files use different --channel settings {sorted(ch)} — re-extract them the same way")
    D = {k: np.concatenate([p[k] for p in parts]) for k in ("X", "y", "group", "split", "tag", "path", "aug")}
    D.update(layers=list(lay.pop()), backbone=bb.pop(), kind=str(parts[0]["kind"]), channel=ch.pop())
    return D

def assign_split(D, test_pct):
    sp = np.array([s.lower() for s in D["split"]])
    auto = np.array([hashed(g) < test_pct for g in D["group"]])
    test = np.where(sp == "test", True, np.where(sp == "train", False, auto))
    val = ~test & np.array([hashed(g, "val") < 20 for g in D["group"]])
    return test, val

def describe(name, mask, D):
    y = D["y"][mask]
    return (f"{name:<6} {mask.sum():>6} rows  {int((y == 0).sum()):>6} real / {int(y.sum()):>6} fake  "
            f"{len(set(D['group'][mask])):>4} groups  {len(set(D['path'][mask])):>5} clips")

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("features", nargs="+", help="features/*.npz from tools.extract_*_features")
    ap.add_argument("--label", default=None, help="name shown on the dashboard")
    ap.add_argument("--model", choices=["both", "logreg", "mlp"], default="both")
    ap.add_argument("--test-pct", type=int, default=20, help="%% of groups held out for testing")
    ap.add_argument("--clean-only", action="store_true", help="ignore the augmented (call-like) copies")
    ap.add_argument("--out", default=None, help="head file (default models/custom/<kind>_head.npz)")
    ap.add_argument("--holdout", nargs="*", default=[],
                    help="features of a SEPARATE real-world set (e.g. your own recordings) — never trained on; "
                         "used to pick the model and to set how much the app trusts it")
    args = ap.parse_args()

    D = load(args.features)
    if args.clean_only:
        keep = D["aug"] == 0
        D = {k: (v[keep] if isinstance(v, np.ndarray) else v) for k, v in D.items()}
    kind = D["kind"]
    test, val = assign_split(D, args.test_pct)
    train = ~test
    fit = train & ~val
    print(f"\n{kind} features: {D['X'].shape[0]} rows, layers {D['layers']}, backbone {D['backbone']}")
    for name, msk in (("fit", fit), ("val", val), ("test", test)):
        print("  " + describe(name, msk, D))
    for name, msk in (("fit", fit), ("val", val), ("test", test)):
        if len(set(D["y"][msk].tolist())) < 2:
            raise SystemExit(f"the {name} split has only one class — add more groups (speakers / people) "
                             f"of both real and fake, or lower --test-pct")

    y = D["y"].astype(int)
    H = None
    if args.holdout:
        H = load(args.holdout)
        if H["channel"] != D["channel"] or H["layers"] != D["layers"] or H["backbone"] != D["backbone"]:
            raise SystemExit("--holdout features were extracted differently (channel / backbone) — re-extract them")
        if len(set(H["y"].tolist())) < 2:
            print(f"note: holdout has only {'real' if H['y'].max() == 0 else 'fake'} clips — "
                  f"reporting how many get flagged, not AUC")
        print("  " + describe("hold", np.ones(len(H["y"]), bool), H))

    def holdout_eval(mdl, li, mu, sd):
        if H is None:
            return float("nan"), None, None
        ph = predict(mdl, (H["X"][:, li].astype(np.float32) - mu) / sd)
        pc, yc, _ = per_clip(ph, H["y"].astype(int), H["path"])
        rf = float(np.mean(pc[yc == 0] >= 0.5)) if (yc == 0).any() else None
        fc = float(np.mean(pc[yc == 1] >= 0.5)) if (yc == 1).any() else None
        return auc(pc, yc), rf, fc

    def score(r):
        v = np.nan_to_num(r["val_auc"], nan=0.0)
        if H is None:
            return v
        if not np.isnan(r["hold_auc"]):
            return 0.5 * v + 0.5 * r["hold_auc"]
        if r["hold_real_flagged"] is not None:
            return 0.5 * v + 0.5 * (1.0 - r["hold_real_flagged"])
        return v

    results = []
    t0 = time.time()
    print("\nlayer  model                 val AUC" + ("   holdout AUC  real flagged" if H is not None else ""))

    def show(r, name):
        line = f"{r['layer']:>5}  {name:<21} {r['val_auc']:.4f}"
        if H is not None:
            ha = "     —" if np.isnan(r["hold_auc"]) else f"{r['hold_auc']:.4f}"
            rf = "    —" if r["hold_real_flagged"] is None else f"{100 * r['hold_real_flagged']:4.0f}%"
            line += f"   {ha:>11}  {rf:>12}"
        print(line)

    for li, layer in enumerate(D["layers"]):
        X = D["X"][:, li].astype(np.float32)
        mu, sd = X[fit].mean(0), X[fit].std(0) + 1e-6
        Z = (X - mu) / sd
        if args.model in ("both", "logreg"):
            for lam in (1e-4, 1e-3, 1e-2, 1e-1):
                mdl = train_logreg(Z[fit], y[fit], lam)
                a = auc(predict(mdl, Z[val]), y[val])
                ha, rf, fc = holdout_eval(mdl, li, mu, sd)
                r = {"layer": layer, "li": li, "model": "logreg", "lam": lam, "val_auc": a,
                     "hold_auc": ha, "hold_real_flagged": rf, "hold_fake_caught": fc}
                results.append(r)
                show(r, f"logreg λ={lam:g}")
        if args.model in ("both", "mlp"):
            mdl, ep, a = train_mlp(Z[fit], y[fit], Xv=Z[val], yv=y[val])
            ha, rf, fc = holdout_eval(mdl, li, mu, sd)
            r = {"layer": layer, "li": li, "model": "mlp", "epochs": ep, "val_auc": a,
                 "hold_auc": ha, "hold_real_flagged": rf, "hold_fake_caught": fc}
            results.append(r)
            show(r, f"mlp (best epoch {ep})")
    best = max(results, key=lambda r: (score(r), r["model"] == "logreg"))
    how = "validation + holdout" if H is not None else "validation"
    print(f"\nbest on {how}: layer {best['layer']} {best['model']} (val AUC {best['val_auc']:.4f}) "
          f"[{time.time() - t0:.0f} s]")

    X = D["X"][:, best["li"]].astype(np.float32)
    mu, sd = X[train].mean(0), X[train].std(0) + 1e-6
    Z = (X - mu) / sd
    if best["model"] == "logreg":
        mdl = train_logreg(Z[train], y[train], best["lam"])
    else:
        mdl, _, _ = train_mlp(Z[train], y[train], epochs=best["epochs"])
    p = predict(mdl, Z[test])
    yt, paths, tags, aug = y[test], D["path"][test], D["tag"][test], D["aug"][test]
    pc, yc, _ = per_clip(p, yt, paths)
    clean = aug == 0
    pcc, ycc, _ = per_clip(p[clean], yt[clean], paths[clean]) if clean.any() else (pc, yc, None)
    rep = {
        "kind": kind, "backbone": D["backbone"], "layer": best["layer"], "model": best["model"],
        "val_auc": best["val_auc"],
        "test_auc_segment": auc(p, yt), "test_auc": auc(pc, yc), "test_auc_clean": auc(pcc, ycc),
        "test_eer": eer(pc, yc),
        "fake_caught_at_0.5": float(np.mean(pc[yc == 1] >= 0.5)),
        "real_flagged_at_0.5": float(np.mean(pc[yc == 0] >= 0.5)),
        "n_train_rows": int(train.sum()), "n_test_clips": int(len(yc)),
        "train_groups": len(set(D["group"][train])), "test_groups": len(set(D["group"][test])),
        "per_tag": {},
    }
    for t in sorted(set(tags.tolist()) - {""}):
        m = tags == t
        tc, tyc, _ = per_clip(p[m], yt[m], paths[m])
        rep["per_tag"][t] = {"clips": int(len(tyc)), "auc": auc(tc, tyc),
                             "fake_caught": float(np.mean(tc[tyc == 1] >= 0.5)) if (tyc == 1).any() else None,
                             "real_flagged": float(np.mean(tc[tyc == 0] >= 0.5)) if (tyc == 0).any() else None}

    hold = holdout_eval(mdl, best["li"], mu, sd)
    rep["holdout_auc"], rep["holdout_real_flagged"], rep["holdout_fake_caught"] = hold
    rep["channel"] = D["channel"]

    print("\n==== TEST (speakers / people never seen in training)")
    print(f"  clips: {rep['n_test_clips']}  groups: {rep['test_groups']}")
    print(f"  AUC per clip       {rep['test_auc']:.4f}   (clean audio/video only: {rep['test_auc_clean']:.4f})")
    print(f"  AUC per segment    {rep['test_auc_segment']:.4f}")
    print(f"  EER                {100 * rep['test_eer']:.1f} %")
    print(f"  at P(fake) ≥ 0.5:  {100 * rep['fake_caught_at_0.5']:.0f} % of fakes caught, "
          f"{100 * rep['real_flagged_at_0.5']:.0f} % of real clips flagged")
    for t, r in rep["per_tag"].items():
        fc = "—" if r["fake_caught"] is None else f"{100 * r['fake_caught']:.0f}%"
        rf = "—" if r["real_flagged"] is None else f"{100 * r['real_flagged']:.0f}%"
        a = "—" if np.isnan(r["auc"]) else f"{r['auc']:.3f}"
        print(f"    {t:<16} clips {r['clips']:>5}  AUC {a:>6}  fakes caught {fc:>5}  real flagged {rf:>5}")

    if H is not None:
        ha, rf, fc = hold
        print("\n==== HOLDOUT (separate real-world recordings — the number that matters for the demo)")
        print(f"  clips: {len(set(H['path']))}   AUC {'—' if np.isnan(ha) else f'{ha:.3f}'}   "
              f"real flagged {'—' if rf is None else f'{100 * rf:.0f}%'}   "
              f"fakes caught {'—' if fc is None else f'{100 * fc:.0f}%'}")

    trust_auc = rep["test_auc"]
    if H is not None:
        if not np.isnan(hold[0]):
            trust_auc = min(trust_auc, hold[0])
        elif hold[1] is not None:
            trust_auc = min(trust_auc, 1.0 - 0.5 * hold[1])

    label = args.label or ("Custom voice detector (ours)" if kind == "voice" else "Custom face detector (ours)")
    params = {"kind": np.array(mdl["kind"]), "layer": np.int16(best["layer"]), "backbone": np.array(D["backbone"]),
              "mu": mu.astype(np.float32), "sd": sd.astype(np.float32),
              **{k: np.asarray(v, np.float32) for k, v in mdl.items() if k != "kind"}}
    head = Head(params)
    head.meta = {"label": label, "test_auc": float(trust_auc), "dataset_auc": rep["test_auc"],
                 "holdout_auc": rep["holdout_auc"], "channel": D["channel"],
                 "val_auc": rep["val_auc"], "eer": rep["test_eer"],
                 "trained": date.today().isoformat(), "features": [str(f) for f in args.features],
                 "train_groups": rep["train_groups"], "test_groups": rep["test_groups"]}
    out = Path(args.out) if args.out else head_path(kind)
    head.save(out)
    report = out.with_name(f"{kind}_report.json")
    report.write_text(json.dumps({**rep, "search": results}, indent=2, default=float), encoding="utf-8")
    print(f"\nsaved {out}  (+ {report.name}). Restart `python app.py` to use it.")
    print(f"live-app trust: based on AUC {trust_auc:.3f} "
          f"({'worse of dataset test and holdout' if H is not None else 'dataset test only — add --holdout'})")
    if trust_auc < 0.75:
        print("note: test AUC is low — the engine will trust this head only a little "
              "(its weight scales with the test AUC).")

if __name__ == "__main__":
    main()

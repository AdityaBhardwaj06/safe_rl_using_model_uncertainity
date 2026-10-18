"""
Experiment 2 -- Does ensemble disagreement sigma_c actually flag unfamiliar / hazardous states?

This tests the central claim of the report ("in unvisited states the heads disagree, raising
c_tilde and warning the agent before it crashes") on REAL Safety-Gymnasium data.

Setup A  "never-seen hazards":
    Train ONLY on states farther than d_hold from every hazard boundary (so the ensemble has never
    seen a hazard contact). Test on held-out episodes, bucketed by true distance:
        far (ID)  : dist > d_hold          near : 0 < dist <= d_hold          contact : dist <= 0
    Question: is sigma (or c_tilde) higher at contacts than in familiar far states?

Setup B  "data scarcity" (replaces the unsupported N-axis of the old Table 3):
    Train on the first N transitions (N in {1k, 5k, 20k, all}) of normal data that includes contacts.
    Question: does c_tilde = mu + kappa*sigma catch more true hazard states than mu alone, at the
    same false-alarm rate?  Does sigma predict where the mean prediction is wrong?

Variants (K = 5 heads, 3 seeds each):
    plain     : different random initialisations only
    bootstrap : + Poisson(1) bootstrap weights per head
    boot+prior: + randomized prior functions (prior_scale = 1)

Metrics: AUROC (threshold-free), TPR at 1% / 5% FPR, mean sigma per bucket. We also report
sigma measured in LOGIT space, because sigmoid saturation can hide disagreement.

Usage:  python scripts/exp2_uncertainty_detection.py --data data/pointgoal1_random.npz
"""
import argparse
import json
import os

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from cost_data import Normalizer, load_split, predict, tpr_at_fpr, train_ensemble

VARIANTS = {
    "plain": dict(bootstrap="none", prior_scale=0.0),
    "bootstrap": dict(bootstrap="poisson", prior_scale=0.0),
    "boot+prior": dict(bootstrap="poisson", prior_scale=1.0),
}


def auroc(pos, neg):
    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    return float(roc_auc_score(y, np.r_[pos, neg]))


def setup_a(args, seed, variant):
    S = load_split(args.data, seed=seed)
    keep = S["dist_tr"] > args.d_hold
    norm = Normalizer(S["X_tr"][keep])
    ens = train_ensemble(norm(S["X_tr"][keep]).numpy(), S["y_tr"][keep], num_heads=args.K,
                         steps=args.steps, seed=seed, **VARIANTS[variant])
    p = predict(ens, norm(S["X_te"]), kappa=args.kappa)
    d = S["dist_te"]
    far, near, con = d > args.d_hold, (d > 0) & (d <= args.d_hold), d <= 0
    out = dict(setup="A", variant=variant, seed=seed, n_train=int(keep.sum()),
               train_contacts=int(S["y_tr"][keep].sum()), n_contact_test=int(con.sum()))
    for b, m in [("far", far), ("near", near), ("contact", con)]:
        out[f"sigma_{b}"] = float(p["sigma"][m].mean())
        out[f"sigma_logit_{b}"] = float(p["sigma_logit"][m].mean())
        out[f"mu_{b}"] = float(p["mu"][m].mean())
    for s in ["mu", "sigma", "c_tilde", "sigma_logit", "c_tilde_logit"]:
        out[f"auroc_{s}_contact_vs_far"] = auroc(p[s][con], p[s][far])
        out[f"tpr1_{s}"] = tpr_at_fpr(p[s][con], p[s][far], 0.01)
    return out


def setup_b(args, seed, variant, n):
    S = load_split(args.data, seed=seed)
    Xtr, ytr = S["X_tr"][:n], S["y_tr"][:n]
    norm = Normalizer(Xtr)
    ens = train_ensemble(norm(Xtr).numpy(), ytr, num_heads=args.K, steps=args.steps, seed=seed,
                         **VARIANTS[variant])
    out = dict(setup="B", variant=variant, seed=seed, n_train=int(len(Xtr)), train_contacts=int(ytr.sum()))
    y = S["y_te"]
    for kappa in [0.0, 1.0, 2.0]:
        p = predict(ens, norm(S["X_te"]), kappa=kappa)
        sc = p["c_tilde"]
        out[f"auroc_k{kappa:g}"] = float(roc_auc_score(y, sc))
        out[f"tpr5_k{kappa:g}"] = tpr_at_fpr(sc[y == 1], sc[y == 0], 0.05)
        out[f"tpr1_k{kappa:g}"] = tpr_at_fpr(sc[y == 1], sc[y == 0], 0.01)
        sl = p["c_tilde_logit"]
        out[f"auroc_logit_k{kappa:g}"] = float(roc_auc_score(y, sl))
        out[f"tpr5_logit_k{kappa:g}"] = tpr_at_fpr(sl[y == 1], sl[y == 0], 0.05)
    err = np.abs(p["mu"] - y)
    big = err > 0.5
    out["auroc_sigma_predicts_error"] = float(roc_auc_score(big, p["sigma"])) if 0 < big.sum() < len(big) else float("nan")
    out["spearman_sigma_abs_error"] = float(spearmanr(p["sigma"], err).correlation)
    out["mean_sigma"] = float(p["sigma"].mean())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/pointgoal1_random.npz")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--kappa", type=float, default=1.0)
    ap.add_argument("--d_hold", type=float, default=0.25)
    ap.add_argument("--Ns", default="1000,5000,20000,1000000")
    ap.add_argument("--only", default="AB")
    ap.add_argument("--out", default="results/exp2_uncertainty.json")
    args = ap.parse_args()

    rows = []
    if "A" in args.only:
        for variant in VARIANTS:
            for seed in range(args.seeds):
                r = setup_a(args, seed, variant)
                rows.append(r)
                print(f"[A {variant:>10} s{seed}] sigma far/near/contact = {r['sigma_far']:.4f}/"
                      f"{r['sigma_near']:.4f}/{r['sigma_contact']:.4f} | AUROC sigma {r['auroc_sigma_contact_vs_far']:.3f}"
                      f" | AUROC c_tilde {r['auroc_c_tilde_contact_vs_far']:.3f} | AUROC sigma_logit "
                      f"{r['auroc_sigma_logit_contact_vs_far']:.3f} | TPR@1%FPR c_tilde {r['tpr1_c_tilde']:.3f}", flush=True)
    if "B" in args.only:
        for n in [int(x) for x in args.Ns.split(",")]:
            for variant in ["bootstrap", "boot+prior"]:
                for seed in range(args.seeds):
                    r = setup_b(args, seed, variant, n)
                    rows.append(r)
                    print(f"[B N={r['n_train']:>6} {variant:>10} s{seed}] contacts in train {r['train_contacts']:>4} |"
                          f" AUROC k0/k1/k2 = {r['auroc_k0']:.3f}/{r['auroc_k1']:.3f}/{r['auroc_k2']:.3f} |"
                          f" TPR@5%FPR k0/k1/k2 = {r['tpr5_k0']:.3f}/{r['tpr5_k1']:.3f}/{r['tpr5_k2']:.3f} |"
                          f" logit-pess AUROC k1 {r['auroc_logit_k1']:.3f} | sigma->error AUROC {r['auroc_sigma_predicts_error']:.3f}", flush=True)
            json.dump(dict(config=vars(args), rows=rows), open(args.out, "w"), indent=2)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(dict(config=vars(args), rows=rows), open(args.out, "w"), indent=2)
    print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()

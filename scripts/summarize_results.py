"""
Build results/RESULTS.md from the logged result files (so every number in the report is traceable).

Usage:  python scripts/summarize_results.py
"""
import glob
import json
import os
import re

import numpy as np
import pandas as pd

RES = "results"
L = ["# Results summary (auto-generated from results/)\n"]


def ms(x):
    x = np.asarray(x, dtype=float)
    return f"{x.mean():.3f} ± {x.std():.3f}" if len(x) > 1 else f"{x.mean():.3f}"


# ---------- integration test ----------
for name, path in [("fixed, b=0", "smoke_test_fixed.csv"), ("buggy (original), b=0", "smoke_test_buggy.csv"),
                   ("fixed, b=25", "smoke_test_fixed_b25.csv")]:
    p = os.path.join(RES, path)
    if os.path.exists(p):
        if name.startswith("fixed, b=0"):
            L.append("## Integration test (real data, H=15, K=5, kappa=1)\n")
            L.append("| run | iters passing checks | final img. reward (H) | max img. reward | final img. cost (H) | final lambda |")
            L.append("|---|---|---|---|---|---|")
        d = pd.read_csv(p)
        L.append(f"| {name} | {d.passed.sum()}/{len(d)} | {d.img_reward_H.iloc[-1]:.3g} | {d.img_reward_H.max():.3g} | "
                 f"{d.img_cost_H.iloc[-1]:.3g} | {d.lam.iloc[-1]:.2f} |")
L.append("")

# ---------- exp1 ----------
p = os.path.join(RES, "exp1_noise_floor.json")
if os.path.exists(p):
    S = json.load(open(p))["summary_mean_std"]
    L.append("## Exp 1: single cost head on held-out layouts (3 seeds, mean ± std)\n")
    L.append("| head | mean ĉ safe | median ĉ safe | mean ĉ far-safe (>0.5) | mean ĉ in hazard | AUROC | 15-step sum | discounted sum | 1000-step sum |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for h, s in S.items():
        f = lambda k: f"{s[k][0]:.4f} ± {s[k][1]:.4f}"  # noqa: E731
        L.append(f"| {h} | {f('mean_pred_safe')} | {s['median_pred_safe'][0]:.4f} | {f('mean_pred_far_safe')} | "
                 f"{f('mean_pred_hazard')} | {s['auroc'][0]:.3f} ± {s['auroc'][1]:.3f} | {s['sum_H15'][0]:.2f} | "
                 f"{s['sum_discounted'][0]:.2f} | {s['sum_T1000'][0]:.1f} |")
    L.append("")

# ---------- exp2 ----------
_rows = []
for _p in ["exp2_uncertainty_A.json", "exp2_uncertainty_B.json"]:
    if os.path.exists(os.path.join(RES, _p)):
        _rows += json.load(open(os.path.join(RES, _p)))["rows"]
if _rows:
    R = pd.DataFrame(_rows)
    A = R[R.setup == "A"]
    if len(A):
        L.append("## Exp 2A: hazards never seen in training (K=5, 3 seeds) -- AUROC contact vs familiar states\n")
        L.append("| variant | AUROC μ | AUROC σ | AUROC μ+σ (current method) | AUROC σ (logit) | AUROC logit-pessimism ℓ̄+σ_ℓ (fix) | TPR@1%FPR μ+σ | TPR@1%FPR ℓ̄+σ_ℓ | mean σ (prob) at contact |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        for v, S in A.groupby("variant", sort=False):
            L.append(f"| {v} | {ms(S.auroc_mu_contact_vs_far)} | {ms(S.auroc_sigma_contact_vs_far)} | "
                     f"{ms(S.auroc_c_tilde_contact_vs_far)} | {ms(S.auroc_sigma_logit_contact_vs_far)} | "
                     f"{ms(S.auroc_c_tilde_logit_contact_vs_far)} | {ms(S.tpr1_c_tilde)} | {ms(S.tpr1_c_tilde_logit)} | {S.sigma_contact.mean():.2e} |")
        L.append("")
    B = R[R.setup == "B"]
    if len(B):
        L.append("## Exp 2B: data scarcity (train on first N transitions) -- detecting true hazard states on unseen layouts\n")
        L.append("| N | variant | contacts in train | AUROC κ=0 | AUROC κ=1 | AUROC κ=2 | AUROC logit-pess. κ=1 | TPR@5%FPR κ=0 | TPR@5%FPR κ=1 | TPR@5%FPR κ=2 | AUROC σ→error |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for (n, v), S in B.groupby(["n_train", "variant"], sort=True):
            L.append(f"| {n} | {v} | {S.train_contacts.mean():.0f} | {ms(S.auroc_k0)} | {ms(S.auroc_k1)} | {ms(S.auroc_k2)} | "
                     f"{ms(S.auroc_logit_k1)} | {ms(S.tpr5_k0)} | {ms(S.tpr5_k1)} | {ms(S.tpr5_k2)} | {ms(S.auroc_sigma_predicts_error)} |")
        L.append("")

# ---------- training runs ----------
rows = []
for p in sorted(glob.glob(f"{RES}/ppo_lag_seed*.csv")) + sorted(glob.glob(f"{RES}/mb_kappa*_seed*.csv")):
    d = pd.read_csv(p)
    if not len(d):
        continue
    name = "PPO-Lag" if "ppo" in p else f"MB kappa={re.search(r'kappa([\d.]+)_', p).group(1)}"
    learn = d[d.phase == "learn"] if "phase" in d else d
    last = learn.tail(5)
    rows.append(dict(method=name, seed=int(re.search(r"seed(\d+)", p).group(1)), steps=int(d.env_step.max()),
                     cum_viol=int(d.cum_violations.iloc[-1]),
                     viol_learning=int(d.cum_violations.iloc[-1] - (d[d.phase == "prefill"].cum_violations.max()
                                                                     if "phase" in d and (d.phase == "prefill").any() else 0)),
                     last5_return=last.ep_return.mean(), last5_cost=last.ep_cost.mean(), final_lambda=d["lambda"].iloc[-1]))
if rows:
    T = pd.DataFrame(rows)
    L.append("## Online training pilot (real env, SafetyPointGoal1-v0, b=25)\n")
    L.append("| method | seed | env steps | cumulative violations | violations after prefill | return (last 5 eps) | cost (last 5 eps) | final λ |")
    L.append("|---|---|---|---|---|---|---|---|")
    for r in T.itertuples():
        L.append(f"| {r.method} | {r.seed} | {r.steps} | {r.cum_viol} | {r.viol_learning} | {r.last5_return:.2f} | "
                 f"{r.last5_cost:.1f} | {r.final_lambda:.2f} |")
    L.append("")

for p in sorted(glob.glob(f"{RES}/eval_*.json")):
    if "eval_" in L[-1] and False:
        pass
    e = json.load(open(p))
    if not any(l.startswith("## Deterministic evaluation") for l in L):
        L.append("## Deterministic evaluation of final checkpoints (10 new layouts)\n")
        L.append("| checkpoint | mean return | mean cost | zero-violation episodes |")
        L.append("|---|---|---|---|")
    L.append(f"| {os.path.basename(e['checkpoint'])} | {e['mean_return']:.2f} ± {e['std_return']:.2f} | "
             f"{e['mean_cost']:.1f} ± {e['std_cost']:.1f} | {100 * e['zero_violation_episode_frac']:.0f}% |")

open(os.path.join(RES, "RESULTS.md"), "w").write("\n".join(L) + "\n")
print("\n".join(L))

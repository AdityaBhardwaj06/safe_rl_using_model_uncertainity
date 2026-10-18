"""
Plot ONLY logged results (replaces the old script, which plotted hard-coded arrays).

Reads from results/ and writes PNGs to artifacts/. Any figure whose inputs are missing is skipped.
  fig_training_curves.png   real env: cumulative violations + episode return (MB kappa=0/1, PPO-Lag)
  fig_integration_test.png  fixed vs buggy training loop; lambda at b=0 vs b=25
  fig_exp1_noise_floor.png  single-head predictions on safe vs hazard states (held-out layouts)
  fig_exp2_uncertainty.png  does sigma flag never-seen hazards (A) / help under data scarcity (B)

Usage:  python scripts/plot_results.py
"""
import glob
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

RES, ART = "results", "artifacts"
os.makedirs(ART, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.linestyle": ":", "grid.alpha": 0.6})
COL = {"PPO-Lag": "#dc2626", "MB, kappa=0 (mean only)": "#d97706", "MB, kappa=1 (pessimistic)": "#16a34a"}


def load_runs():
    runs = {}
    for p in sorted(glob.glob(f"{RES}/ppo_lag_seed*.csv")):
        runs.setdefault("PPO-Lag", []).append(pd.read_csv(p))
    for p in sorted(glob.glob(f"{RES}/mb_kappa*_seed*.csv")):
        k = float(re.search(r"kappa([\d.]+)_", p).group(1))
        name = "MB, kappa=0 (mean only)" if k == 0 else f"MB, kappa={k:g} (pessimistic)"
        runs.setdefault(name, []).append(pd.read_csv(p))
    return {k: [d for d in v if len(d)] for k, v in runs.items() if any(len(d) for d in v)}


def fig_training():
    runs = load_runs()
    if not runs:
        return print("skip training curves (no logs)")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8), dpi=180)
    max_step = 0
    for name, dfs in runs.items():
        c = COL.get(name, None)
        for i, d in enumerate(dfs):
            lab = f"{name} ({len(dfs)} seed{'s' if len(dfs) > 1 else ''})" if i == 0 else None
            a1.plot(d.env_step / 1000, d.cum_violations, "-", color=c, lw=1.6, alpha=0.9, label=lab)
            a2.plot(d.env_step / 1000, d.ep_return, "o-", color=c, lw=1.2, ms=2.5, alpha=0.9, label=lab)
            max_step = max(max_step, d.env_step.max())
    mb = [d for k, v in runs.items() if k.startswith("MB") for d in v]
    if mb and "phase" in mb[0]:
        pre = mb[0][mb[0].phase == "prefill"].env_step.max()
        for a in (a1, a2):
            a.axvspan(0, pre / 1000, color="grey", alpha=0.12, lw=0)
            a.text(pre / 2000, a.get_ylim()[1] * 0.97, "random\nprefill", ha="center", va="top", fontsize=7, color="grey")
    a1.set(xlabel="Real environment steps (x1000)", ylabel="Cumulative hazard steps (real env)",
           title="(A) Training-time safety violations")
    a2.set(xlabel="Real environment steps (x1000)", ylabel="Episode return", title="(B) Episode return")
    a1.legend(fontsize=7, frameon=False); a2.legend(fontsize=7, frameon=False)
    fig.suptitle(f"SafetyPointGoal1-v0, vector obs, budget b=25 -- CPU pilot ({max_step/1000:.0f}k steps)",
                 fontsize=9.5, y=1.02)
    plt.tight_layout(); plt.savefig(f"{ART}/fig_training_curves.png", bbox_inches="tight"); plt.close()
    print("saved fig_training_curves.png")


def fig_integration():
    fx, bg = f"{RES}/smoke_test_fixed.csv", f"{RES}/smoke_test_buggy.csv"
    if not (os.path.exists(fx) and os.path.exists(bg)):
        return print("skip integration figure")
    F, B = pd.read_csv(fx), pd.read_csv(bg)
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.3), dpi=180)
    ax[0].semilogy(B.iter, B.img_reward_H.clip(lower=1e-3), color="#dc2626", label="buggy: single optimiser")
    ax[0].semilogy(F.iter, F.img_reward_H.abs().clip(lower=1e-3), color="#16a34a", label="fixed (|value|)")
    ax[0].axhline(14.3, color="k", ls="--", lw=0.8, label="max achievable (~14.3)")
    ax[0].set(title="(A) Imagined 15-step reward", xlabel="iteration"); ax[0].legend(fontsize=7, frameon=False)
    ax[1].plot(B.iter, B.img_cost_H, color="#dc2626", label="buggy"); ax[1].plot(F.iter, F.img_cost_H, color="#16a34a", label="fixed")
    ax[1].set(title="(B) Imagined 15-step pessimistic cost", xlabel="iteration"); ax[1].legend(fontsize=7, frameon=False)
    ax[2].plot(F.iter, F.lam, color="#16a34a", label="fixed, b = 0")
    b25 = f"{RES}/smoke_test_fixed_b25.csv"
    if os.path.exists(b25):
        G = pd.read_csv(b25); ax[2].plot(G.iter, G.lam, color="#2563eb", label="fixed, b = 25")
    ax[2].set(title="(C) Lagrange multiplier", xlabel="iteration"); ax[2].legend(fontsize=7, frameon=False)
    plt.tight_layout(); plt.savefig(f"{ART}/fig_integration_test.png", bbox_inches="tight"); plt.close()
    print("saved fig_integration_test.png")


def fig_exp1():
    p = f"{RES}/exp1_noise_floor_hist.json"
    if not os.path.exists(p):
        return print("skip exp1 figure")
    H = json.load(open(p))
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2), dpi=180)
    for a, head in zip(ax, ["bce", "mse"]):
        bins = np.linspace(-0.1 if head == "mse" else 0, 0.6, 61)
        a.hist(H[head]["safe"], bins=bins, density=True, alpha=0.6, color="#2563eb", label="true cost = 0")
        a.hist(H[head]["hazard"], bins=bins, density=True, alpha=0.6, color="#dc2626", label="true cost = 1")
        a.set_yscale("log"); a.set(xlabel="predicted cost", title=f"single {head.upper()} head, unseen layouts")
        a.legend(fontsize=7, frameon=False)
    plt.tight_layout(); plt.savefig(f"{ART}/fig_exp1_noise_floor.png", bbox_inches="tight"); plt.close()
    print("saved fig_exp1_noise_floor.png")


def fig_exp2():
    rr = []
    for p in [f"{RES}/exp2_uncertainty_A.json", f"{RES}/exp2_uncertainty_B.json"]:
        if os.path.exists(p):
            rr += json.load(open(p))["rows"]
    if not rr:
        return print("skip exp2 figure")
    rows = pd.DataFrame(rr)
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 3.6), dpi=180)
    A = rows[rows.setup == "A"]
    if len(A):
        sigs = ["mu", "sigma", "c_tilde", "sigma_logit", "c_tilde_logit"]
        variants = list(dict.fromkeys(A.variant))
        x = np.arange(len(sigs)); wdt = 0.8 / len(variants)
        for i, v in enumerate(variants):
            S = A[A.variant == v]
            m = [S[f"auroc_{s}_contact_vs_far"].mean() for s in sigs]
            e = [S[f"auroc_{s}_contact_vs_far"].std() for s in sigs]
            ax[0].bar(x + i * wdt, m, wdt, yerr=e, capsize=2, label=v)
        ax[0].axhline(0.5, color="k", ls="--", lw=0.8)
        ax[0].set_xticks(x + wdt * (len(variants) - 1) / 2, ["mu", "sigma", "mu+sigma\n(current)", "sigma\n(logit)", "logit mean+sigma\n(fix)"])
        ax[0].set(ylim=(0, 1), ylabel="AUROC: contact vs familiar states",
                  title="(A) Hazards never seen in training"); ax[0].legend(fontsize=7, frameon=False)
    B = rows[rows.setup == "B"]
    if len(B):
        for v, ls in [("bootstrap", "-"), ("boot+prior", "--")]:
            S = B[B.variant == v]
            for k, c in [("0", "#d97706"), ("1", "#16a34a"), ("2", "#2563eb")]:
                g = S.groupby("n_train")[f"auroc_k{k}"]
                ax[1].errorbar(g.mean().index, g.mean(), yerr=g.std(), fmt="o" + ls, color=c, ms=3, lw=1.2,
                               capsize=2, label=f"mu+{k}sigma, {v}")
            g = S.groupby("n_train")["auroc_logit_k1"]
            ax[1].errorbar(g.mean().index, g.mean(), yerr=g.std(), fmt="s" + ls, color="#7c3aed", ms=3, lw=1.2,
                           capsize=2, label=f"logit pessimism k=1, {v}")
        ax[1].set_xscale("log")
        ax[1].set(xlabel="training transitions N", ylabel="AUROC on unseen layouts",
                  title="(B) Detecting true hazard states vs data size"); ax[1].legend(fontsize=6.5, frameon=False)
    plt.tight_layout(); plt.savefig(f"{ART}/fig_exp2_uncertainty.png", bbox_inches="tight"); plt.close()
    print("saved fig_exp2_uncertainty.png")


if __name__ == "__main__":
    fig_training(); fig_integration(); fig_exp1(); fig_exp2()

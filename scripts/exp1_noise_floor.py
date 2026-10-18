"""
Experiment 1 -- MEASURE the single-cost-head "noise floor" on real Safety-Gymnasium data.

Replaces the hard-coded 0.021 in the old eval_checkpoint.py.

Two common single-head designs are trained on the same data (3 seeds each):
  * bce : sigmoid classifier trained with binary cross-entropy (what our ensemble heads use)
  * mse : unbounded linear regression head trained with mean-squared error
Inputs are the 60-d vector observations (a proxy for the world-model latent z; see README).

For held-out SAFE states (true cost 0) we report the mean prediction m and what it would add up to:
  * over a 15-step imagined rollout:          15 * m
  * as a discounted return (gamma = 0.99):    m * (1 - gamma^T) / (1 - gamma)  ~= 100 * m
  * as an undiscounted 1000-step episode sum: 1000 * m   (the number the report used)

Usage:  python scripts/exp1_noise_floor.py --data data/pointgoal1_random.npz
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from cost_data import Normalizer, load_split, predict, train_ensemble


def train_mse_head(X, y, steps, seed, hidden=256, batch=256, lr=1e-3):
    torch.manual_seed(seed)
    net = nn.Sequential(nn.Linear(X.shape[1], hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                        nn.Linear(hidden, 1))
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    yt = torch.as_tensor(y, dtype=torch.float32).view(-1, 1)
    g = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        idx = torch.randint(0, len(X), (batch,), generator=g)
        loss = ((net(X[idx]) - yt[idx]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        return net


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/pointgoal1_random.npz")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--T", type=int, default=1000)
    ap.add_argument("--out", default="results/exp1_noise_floor.json")
    args = ap.parse_args()

    disc = (1 - args.gamma ** args.T) / (1 - args.gamma)
    rows, hist = [], {}
    for seed in range(args.seeds):
        S = load_split(args.data, seed=seed)
        norm = Normalizer(S["X_tr"])
        Xtr, Xte = norm(S["X_tr"]), norm(S["X_te"])
        safe, hz = S["y_te"] == 0, S["y_te"] == 1
        far = S["dist_te"] > 0.5
        for head in ["bce", "mse"]:
            if head == "bce":
                ens = train_ensemble(Xtr.numpy(), S["y_tr"], num_heads=1, steps=args.steps, seed=seed)
                pred = predict(ens, Xte)["mu"]
            else:
                net = train_mse_head(Xtr, S["y_tr"], args.steps, seed)
                with torch.no_grad():
                    pred = net(Xte).squeeze(-1).numpy()
            m = float(pred[safe].mean())
            rows.append(dict(
                head=head, seed=seed,
                n_train=int(len(Xtr)), n_test=int(len(Xte)), test_contact_rate=float(hz.mean()),
                mean_pred_safe=m,
                median_pred_safe=float(np.median(pred[safe])),
                p95_pred_safe=float(np.quantile(pred[safe], 0.95)),
                mean_abs_pred_safe=float(np.abs(pred[safe]).mean()),
                mean_pred_far_safe=float(pred[safe & far].mean()),
                mean_pred_hazard=float(pred[hz].mean()),
                auroc=float(roc_auc_score(S["y_te"], pred)),
                sum_H15=15 * m, sum_discounted=disc * m, sum_T1000=args.T * m,
                true_cost_per_episode_in_data=float(S["y_te"].sum() / (len(S["y_te"]) / args.T)),
            ))
            if seed == 0:
                hist[head] = dict(safe=pred[safe].tolist()[:20000], hazard=pred[hz].tolist())
            r = rows[-1]
            print(f"[seed {seed}] {head}: mean c_hat on safe = {m:.4f} | far-safe {r['mean_pred_far_safe']:.4f}"
                  f" | hazard {r['mean_pred_hazard']:.3f} | AUROC {r['auroc']:.3f}"
                  f" | 15-step {r['sum_H15']:.3f} | discounted {r['sum_discounted']:.2f} | 1000-step {r['sum_T1000']:.1f}",
                  flush=True)

    summary = {}
    for head in ["bce", "mse"]:
        R = [r for r in rows if r["head"] == head]
        summary[head] = {k: [float(np.mean([r[k] for r in R])), float(np.std([r[k] for r in R]))]
                         for k in R[0] if isinstance(R[0][k], float)}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(dict(config=vars(args), rows=rows, summary_mean_std=summary), open(args.out, "w"), indent=2)
    json.dump(hist, open(args.out.replace(".json", "_hist.json"), "w"))
    print("\nSUMMARY (mean +- std over seeds)")
    for head, s in summary.items():
        print(f"  {head}: safe-state mean c_hat {s['mean_pred_safe'][0]:.4f} +- {s['mean_pred_safe'][1]:.4f};"
              f" 15-step sum {s['sum_H15'][0]:.3f}; discounted {s['sum_discounted'][0]:.2f};"
              f" 1000-step {s['sum_T1000'][0]:.1f}; AUROC {s['auroc'][0]:.3f}")
    print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()

"""
Minimal PPO-Lagrangian baseline on Safety-Gymnasium (vector observations).

Follows the standard PPO-Lag recipe (as in OmniSafe / Ray et al. 2019):
  * separate reward and cost critics, GAE for both
  * clipped surrogate on the combined advantage  (A_R - lambda * A_C) / (1 + lambda)
  * lambda updated once per epoch from the REAL mean episode cost:  lambda <- max(0, lambda + lr*(Jc - b))
This compact version exists so the pipeline can be run and inspected end-to-end; for the final
report, also run OmniSafe's PPOLag (the teammate's Weeks 1-2 task) and compare.

Logs one CSV row per finished episode: env_step, episode, ep_return, ep_cost, cum_violations, lambda.

Usage:  python scripts/train_ppo_lag.py --steps 100000 --budget 25 --seed 0
"""
import argparse
import csv
import os
import time

import sg_compat  # noqa: F401
import numpy as np
import torch
import torch.nn as nn
from sg_compat import make_env


class RunningNorm:
    def __init__(self, dim):
        self.n, self.mean, self.m2 = 1e-4, np.zeros(dim), np.ones(dim)

    def update(self, x):
        self.n += 1
        d = x - self.mean
        self.mean += d / self.n
        self.m2 += d * (x - self.mean)

    def __call__(self, x):
        return np.clip((x - self.mean) / np.sqrt(self.m2 / self.n + 1e-8), -10, 10)


def mlp(i, o, h=64):
    return nn.Sequential(nn.Linear(i, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh(), nn.Linear(h, o))


def gae(r, v, last_v, gamma, lam):
    adv, g = np.zeros_like(r), 0.0
    for t in reversed(range(len(r))):
        nv = last_v if t == len(r) - 1 else v[t + 1]
        delta = r[t] + gamma * nv - v[t]
        g = delta + gamma * lam * g
        adv[t] = g
    return adv, adv + v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="SafetyPointGoal1-v0")
    ap.add_argument("--steps", type=int, default=100000)
    ap.add_argument("--rollout", type=int, default=2000)
    ap.add_argument("--budget", type=float, default=25.0)
    ap.add_argument("--lag_lr", type=float, default=0.035)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed); torch.set_num_threads(1)
    gamma, lam_gae, clip, epochs, mb = 0.99, 0.95, 0.2, 10, 500

    env = make_env(args.task)
    od, ad = env.observation_space.shape[0], env.action_space.shape[0]
    pi, vr, vc = mlp(od, ad), mlp(od, 1), mlp(od, 1)
    log_std = nn.Parameter(-0.5 * torch.ones(ad))
    opt_pi = torch.optim.Adam(list(pi.parameters()) + [log_std], lr=3e-4)
    opt_v = torch.optim.Adam(list(vr.parameters()) + list(vc.parameters()), lr=1e-3)
    norm, lagrange = RunningNorm(od), 0.0

    out = args.out or f"results/ppo_lag_seed{args.seed}.csv"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    f = open(out, "w", newline="")
    w = csv.writer(f); w.writerow(["env_step", "episode", "ep_return", "ep_cost", "cum_violations", "lambda"])

    obs, _ = env.reset(seed=args.seed)
    ep_ret = ep_cost = 0.0; ep = 0; cum_viol = 0; step = 0; t0 = time.time(); epoch_costs = []
    while step < args.steps:
        B = {k: [] for k in ["o", "a", "logp", "r", "c", "vr", "vc"]}
        for _ in range(args.rollout):
            norm.update(obs); o = norm(obs)
            ot = torch.as_tensor(o, dtype=torch.float32)
            with torch.no_grad():
                dist = torch.distributions.Normal(pi(ot), log_std.exp())
                a = dist.sample(); logp = dist.log_prob(a).sum()
                v_r, v_c = vr(ot).item(), vc(ot).item()
            nobs, r, c, term, trunc, _ = env.step(np.clip(a.numpy(), -1, 1))
            c = float(c > 0)
            for k, v in zip(B, [o, a.numpy(), logp.item(), r, c, v_r, v_c]):
                B[k].append(v)
            ep_ret += r; ep_cost += c; cum_viol += int(c); step += 1; obs = nobs
            if term or trunc:
                w.writerow([step, ep, ep_ret, ep_cost, cum_viol, lagrange]); f.flush()
                epoch_costs.append(ep_cost); ep += 1; ep_ret = ep_cost = 0.0
                obs, _ = env.reset()
        with torch.no_grad():
            lo = torch.as_tensor(norm(obs), dtype=torch.float32)
            last_r, last_c = vr(lo).item(), vc(lo).item()
        arr = {k: np.asarray(v, dtype=np.float32) for k, v in B.items()}
        adv_r, ret_r = gae(arr["r"], arr["vr"], last_r, gamma, lam_gae)
        adv_c, ret_c = gae(arr["c"], arr["vc"], last_c, gamma, lam_gae)
        adv_r = (adv_r - adv_r.mean()) / (adv_r.std() + 1e-8)
        adv_c = adv_c - adv_c.mean()

        # dual update from REAL episode costs (before the policy step, like OmniSafe)
        if epoch_costs:
            lagrange = max(0.0, lagrange + args.lag_lr * (np.mean(epoch_costs) - args.budget))
            epoch_costs = []
        adv = torch.as_tensor((adv_r - lagrange * adv_c) / (1 + lagrange))
        O = torch.as_tensor(arr["o"]); A = torch.as_tensor(arr["a"]); LP = torch.as_tensor(arr["logp"])
        RR, RC = torch.as_tensor(ret_r), torch.as_tensor(ret_c)
        for _ in range(epochs):
            for idx in torch.randperm(len(O)).split(mb):
                d = torch.distributions.Normal(pi(O[idx]), log_std.exp())
                ratio = (d.log_prob(A[idx]).sum(-1) - LP[idx]).exp()
                l_pi = -torch.min(ratio * adv[idx], ratio.clamp(1 - clip, 1 + clip) * adv[idx]).mean()
                opt_pi.zero_grad(); l_pi.backward(); nn.utils.clip_grad_norm_(pi.parameters(), 40); opt_pi.step()
                l_v = ((vr(O[idx]).squeeze(-1) - RR[idx]) ** 2).mean() + ((vc(O[idx]).squeeze(-1) - RC[idx]) ** 2).mean()
                opt_v.zero_grad(); l_v.backward(); opt_v.step()
        print(f"step {step:>7} | episodes {ep:>4} | cum violations {cum_viol:>6} | lambda {lagrange:.3f} | {time.time()-t0:.0f}s",
              flush=True)
    f.close()
    torch.save(dict(pi=pi.state_dict(), log_std=log_std.data, norm_mean=norm.mean, norm_m2=norm.m2, norm_n=norm.n),
               f"checkpoints/ppo_lag_seed{args.seed}.pt")
    print(f"Done. Log -> {out}")


if __name__ == "__main__":
    main()

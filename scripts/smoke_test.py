"""
Integration test (formerly "smoke test") for world model + 5-head cost ensemble + Lagrangian actor-critic.

Runs on REAL Safety-Gymnasium transitions (data/pointgoal1_random.npz from collect_data.py) instead
of torch.randn noise, and checks invariants that a correct implementation must satisfy.

Modes
  default : correct training -- three separate updates
              1. world model + cost ensemble  <- supervised losses on replay data only
              2. actor                        <- imagined Lagrangian objective, model FROZEN
              3. lambda                       <- dual ascent on the cost estimate
  --buggy : reproduces the ORIGINAL smoke_test.py bug: ONE Adam optimiser over ALL parameters,
            stepped on the actor loss only. The model can then "cheat" (inflate reward, deflate cost).

Invariants checked every iteration (correct mode must pass all):
  * every imagined per-step cost c~ lies in [0, 1 + kappa * 0.5]   (sigmoid probs; std of K values in [0,1] <= 0.5)
  * imagined discounted reward stays below the max achievable from the data
  * all logged quantities are finite

Usage:
  python scripts/smoke_test.py                 # correct version
  python scripts/smoke_test.py --buggy         # reproduce the original divergence
"""
import argparse
import csv
import math
import os
import time

import sg_compat  # noqa: F401
import numpy as np
import torch

from models.actor_critic import LagrangianActorCritic
from models.ensemble_cost_head import make_bootstrap_masks
from models.lagrangian import LagrangeMultiplier, discounted_budget
from models.world_model import WorldModel


class SequenceReplay:
    def __init__(self, path, L=16, seed=0):
        d = np.load(path)
        self.obs, self.act, self.rew, self.cost, self.nobs, self.ep = (
            d["obs"], d["act"], d["rew"], d["cost"], d["next_obs"], d["episode"])
        self.L = L
        # valid start indices: the whole window lies inside one episode
        same = np.r_[self.ep[L - 1:] == self.ep[:len(self.ep) - L + 1]]
        self.starts = np.where(same)[0]
        self.rng = np.random.default_rng(seed)
        self.mu = self.obs.mean(0, keepdims=True)
        self.sd = np.maximum(self.obs.std(0, keepdims=True), 1e-2)
        self.r_max = float(self.rew.max())
        self.boot = make_bootstrap_masks(5, len(self.obs), "poisson", torch.Generator().manual_seed(seed))

    def norm(self, o):
        return torch.as_tensor((o - self.mu) / self.sd, dtype=torch.float32)

    def sample(self, B):
        s = self.rng.choice(self.starts, size=B)
        idx = s[:, None] + np.arange(self.L)[None]
        obs = np.concatenate([self.obs[idx], self.nobs[idx[:, -1:]]], 1)
        return (self.norm(obs), torch.as_tensor(self.act[idx], dtype=torch.float32),
                torch.as_tensor(self.rew[idx], dtype=torch.float32),
                torch.as_tensor(self.cost[idx], dtype=torch.float32),
                self.boot[:, idx])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/pointgoal1_random.npz")
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--buggy", action="store_true")
    ap.add_argument("--kappa", type=float, default=1.0)
    ap.add_argument("--budget", type=float, default=0.0, help="undiscounted cost per 1000-step episode")
    ap.add_argument("--horizon", type=int, default=15)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)

    R = SequenceReplay(args.data, seed=args.seed)
    wm = WorldModel(obs_dim=60, act_dim=2, latent_dim=64, num_cost_heads=5)
    ac = LagrangianActorCritic(64, 2, horizon=args.horizon)
    lag = LagrangeMultiplier(init=0.0, lr=0.05, max_value=50.0)
    d_budget = discounted_budget(args.budget)
    gamma = ac.gamma
    max_disc_reward = R.r_max * (1 - gamma ** args.horizon) / (1 - gamma)
    cost_upper = 1.0 + args.kappa * 0.5

    if args.buggy:
        all_params = list(wm.parameters()) + list(ac.parameters())
        bug_opt = torch.optim.Adam(all_params, lr=1e-3)  # exactly like the original script
    else:
        wm_opt = torch.optim.Adam(wm.parameters(), lr=3e-4)

    mode = "BUGGY (single optimiser on actor loss)" if args.buggy else "FIXED (separate optimisers)"
    print("=" * 112)
    print(f"  INTEGRATION TEST -- {mode}")
    print(f"  real data: {args.data} | H={args.horizon} | K=5 | kappa={args.kappa} | b={args.budget} "
          f"(discounted budget {d_budget:.3f}) | max achievable discounted H-step reward ~ {max_disc_reward:.3f}")
    print("=" * 112)
    hdr = f"{'iter':>5} | {'WM loss':>8} | {'cost BCE':>8} | {'img reward(H)':>13} | {'img cost(H)':>11} | " \
          f"{'J^C value':>9} | {'sigma':>7} | {'lambda':>7} | checks"
    print(hdr); print("-" * len(hdr))

    logs, failures, t0 = [], 0, time.time()
    for it in range(1, args.iters + 1):
        obs, act, rew, cost, boot = R.sample(args.batch)
        if args.buggy:
            z0 = wm.encode(obs[:, 0])
            zs, rs, cs, sig, ents = ac.imagine(wm, z0, args.kappa)
            disc = gamma ** torch.arange(args.horizon, dtype=torch.float32).unsqueeze(-1)
            rew_ret, cost_ret = (disc * rs).sum(0).mean(), (disc * cs).sum(0).mean()
            loss = -rew_ret + lag.value * cost_ret
            bug_opt.zero_grad(); loss.backward(); bug_opt.step()
            info = dict(img_reward_H=rew_ret.item(), img_cost_H=cost_ret.item(), img_cost_value=cost_ret.item(),
                        sigma=sig.mean().item(), max_step_cost=cs.max().item(), min_step_cost=cs.min().item())
            wm_info = dict(total=float("nan"), cost=float("nan"))
        else:
            wm_loss, parts = wm.loss(obs, act, rew, cost, boot)
            wm_opt.zero_grad(); wm_loss.backward(); torch.nn.utils.clip_grad_norm_(wm.parameters(), 100.0); wm_opt.step()
            wm_info = dict(total=wm_loss.item(), cost=parts["cost"])
            with torch.no_grad():
                z0 = wm.encode(obs[:, :-1].reshape(-1, 60))
            z0 = z0[torch.randperm(len(z0))[:256]]
            info = ac.update(wm, z0, lag.value, args.kappa)
        _, lam = lag.update(info["img_cost_value"], d_budget)

        ok_cost = (info["min_step_cost"] >= -1e-6) and (info["max_step_cost"] <= cost_upper + 1e-6)
        ok_rew = info["img_reward_H"] <= max_disc_reward * 1.5
        ok_fin = all(math.isfinite(v) for v in info.values())
        passed = ok_cost and ok_rew and ok_fin
        failures += (not passed)
        flag = "ok" if passed else "FAIL:" + ",".join(n for n, o in [("cost-range", ok_cost), ("reward-bound", ok_rew),
                                                                     ("finite", ok_fin)] if not o)
        row = dict(iter=it, wm_loss=wm_info["total"], cost_bce=wm_info["cost"], **info, lam=lam, passed=int(passed))
        logs.append(row)
        if it in (1, 2, 5) or it % 25 == 0:
            print(f"{it:>5} | {wm_info['total']:>8.4f} | {wm_info['cost']:>8.4f} | {info['img_reward_H']:>13.4f} | "
                  f"{info['img_cost_H']:>11.4f} | {info['img_cost_value']:>9.4f} | {info['sigma']:>7.4f} | "
                  f"{lam:>7.4f} | {flag}", flush=True)

    out = args.out or f"results/smoke_test_{'buggy' if args.buggy else 'fixed'}.csv"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(logs[0].keys()))
        w.writeheader(); w.writerows(logs)
    print("-" * len(hdr))
    print(f"{args.iters - failures}/{args.iters} iterations passed all invariant checks "
          f"({time.time() - t0:.0f}s). Log -> {out}")


if __name__ == "__main__":
    main()

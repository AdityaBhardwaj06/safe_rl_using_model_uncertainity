"""
Online model-based safe RL with the K-head cost ensemble (small CPU-scale pilot of the method).

Loop (Dyna / Dreamer style):
  1. prefill: held-action random exploration
  2. act in the REAL env with the actor on the encoded observation
  3. every `train_every` env steps:
       world model + cost ensemble  <- supervised loss on replayed sequences
       actor/critics                <- imagined H-step Lagrangian objective with c~ = mu + kappa*sigma
  4. after every REAL episode: lambda <- clip(lambda + lr*(episode_cost - b), 0, lambda_max)

kappa = 0 is the "mean-only" ablation; kappa > 0 is the pessimistic (uncertainty-aware) version.
Everything logged is measured in the real environment (training-time violations are what the
project claims to reduce).

Usage:  python scripts/train_mb_safe.py --kappa 1.0 --steps 30000 --seed 0
"""
import argparse
import csv
import os
import time

import sg_compat  # noqa: F401
import numpy as np
import torch
from sg_compat import make_env

from models.actor_critic import LagrangianActorCritic
from models.lagrangian import LagrangeMultiplier
from models.world_model import WorldModel


class Replay:
    def __init__(self, cap, od, ad, K, seed):
        self.o = np.zeros((cap, od), np.float32); self.no = np.zeros((cap, od), np.float32)
        self.a = np.zeros((cap, ad), np.float32); self.r = np.zeros(cap, np.float32)
        self.c = np.zeros(cap, np.float32); self.ep = np.zeros(cap, np.int64); self.n = 0
        self.boot = torch.poisson(torch.ones(K, cap), generator=torch.Generator().manual_seed(seed))
        self.rng = np.random.default_rng(seed)

    def add(self, o, a, r, c, no, ep):
        i = self.n
        self.o[i], self.a[i], self.r[i], self.c[i], self.no[i], self.ep[i] = o, a, r, c, no, ep
        self.n += 1

    def sample(self, B, L, norm):
        n = self.n
        starts = []
        while len(starts) < B:
            s = self.rng.integers(0, n - L)
            if self.ep[s] == self.ep[s + L - 1]:
                starts.append(s)
        idx = np.asarray(starts)[:, None] + np.arange(L)[None]
        obs = np.concatenate([self.o[idx], self.no[idx[:, -1:]]], 1)
        T = lambda x: torch.as_tensor(x, dtype=torch.float32)  # noqa: E731
        return T(norm(obs)), T(self.a[idx]), T(self.r[idx]), T(self.c[idx]), self.boot[:, idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="SafetyPointGoal1-v0")
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--prefill", type=int, default=5000)
    ap.add_argument("--train_every", type=int, default=10)
    ap.add_argument("--kappa", type=float, default=1.0)
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--budget", type=float, default=25.0)
    ap.add_argument("--lag_lr", type=float, default=0.02)
    ap.add_argument("--horizon", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    torch.manual_seed(args.seed); torch.set_num_threads(1)
    rng = np.random.default_rng(args.seed)

    env = make_env(args.task)
    od, ad = env.observation_space.shape[0], env.action_space.shape[0]
    wm = WorldModel(od, ad, latent_dim=32, hidden=128, num_cost_heads=args.K, cost_hidden=128)
    ac = LagrangianActorCritic(32, ad, hidden=128, horizon=args.horizon)
    wm_opt = torch.optim.Adam(wm.parameters(), lr=3e-4)
    lag = LagrangeMultiplier(0.0, args.lag_lr, 50.0)
    R = Replay(args.steps + 10, od, ad, args.K, args.seed)

    tag = f"mb_kappa{args.kappa:g}_seed{args.seed}"
    out = args.out or f"results/{tag}.csv"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    f = open(out, "w", newline="")
    w = csv.writer(f)
    w.writerow(["env_step", "episode", "ep_return", "ep_cost", "cum_violations", "lambda", "mean_sigma", "phase"])

    mu = sd = None
    norm = lambda x: (x - mu) / sd  # noqa: E731
    obs, _ = env.reset(seed=args.seed)
    ep = 0; ep_ret = ep_cost = 0.0; cum_viol = 0; hold = 0; a = np.zeros(ad); sig_log = []; t0 = time.time()
    for step in range(1, args.steps + 1):
        if step <= args.prefill:
            if hold <= 0:
                a = rng.uniform(-1, 1, ad); hold = int(rng.integers(5, 26))
            hold -= 1
        else:
            with torch.no_grad():
                z = wm.encode(torch.as_tensor(norm(obs), dtype=torch.float32))
                a = ac.actor.sample(z)[0].numpy().reshape(-1)
        nobs, r, c, term, trunc, _ = env.step(a)
        c = float(c > 0)
        R.add(obs, a, r, c, nobs, ep)
        ep_ret += r; ep_cost += c; cum_viol += int(c); obs = nobs

        if step == args.prefill:  # fix the observation normaliser from prefill data
            mu = R.o[:R.n].mean(0, keepdims=True); sd = np.maximum(R.o[:R.n].std(0, keepdims=True), 1e-2)
            for _ in range(500):  # warm up the world model on prefill data
                batch = R.sample(16, 16, norm)
                loss, _ = wm.loss(*batch); wm_opt.zero_grad(); loss.backward(); wm_opt.step()
        if step > args.prefill and step % args.train_every == 0:
            o_, a_, r_, c_, b_ = R.sample(16, 16, norm)
            loss, _ = wm.loss(o_, a_, r_, c_, b_)
            wm_opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(wm.parameters(), 100.0); wm_opt.step()
            with torch.no_grad():
                z0 = wm.encode(o_[:, :-1].reshape(-1, od))[torch.randperm(16 * 16)[:128]]
            info = ac.update(wm, z0, lag.value, args.kappa)
            sig_log.append(info["sigma"])

        if term or trunc:
            if step > args.prefill:
                lag.update(ep_cost, args.budget)
            w.writerow([step, ep, ep_ret, ep_cost, cum_viol, lag.value,
                        float(np.mean(sig_log)) if sig_log else 0.0, "prefill" if step <= args.prefill else "learn"])
            f.flush()
            print(f"[{tag}] step {step:>6} | ep {ep:>3} | return {ep_ret:7.2f} | cost {ep_cost:5.0f} | "
                  f"cum viol {cum_viol:>5} | lambda {lag.value:6.3f} | sigma {np.mean(sig_log) if sig_log else 0:.4f} | "
                  f"{time.time()-t0:.0f}s", flush=True)
            ep += 1; ep_ret = ep_cost = 0.0; sig_log = []
            obs, _ = env.reset()
    f.close()
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(dict(wm=wm.state_dict(), actor=ac.actor.state_dict(), mu=mu, sd=sd, kappa=args.kappa,
                    K=args.K), f"checkpoints/{tag}.pt")
    print(f"Done -> {out}")


if __name__ == "__main__":
    main()

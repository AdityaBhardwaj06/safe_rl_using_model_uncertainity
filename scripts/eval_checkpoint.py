"""
Evaluate a TRAINED checkpoint in the REAL Safety-Gymnasium environment.

Replaces the old eval_checkpoint.py, which never loaded a model and printed hard-coded numbers.
Supports checkpoints written by this repo:
  * checkpoints/ppo_lag_seed*.pt      (scripts/train_ppo_lag.py)
  * checkpoints/mb_kappa*_seed*.pt    (scripts/train_mb_safe.py)
If the checkpoint file does not exist, the script STOPS with an error -- it never simulates numbers.

Note on the official SafeDreamer checkpoints: the official SafeDreamer code is built on DreamerV3's
JAX implementation, so its checkpoints cannot be loaded by this PyTorch code. To evaluate them, use
the evaluation entry point in the official repository.

Usage:  python scripts/eval_checkpoint.py checkpoints/mb_kappa1_seed0.pt --episodes 10
"""
import argparse
import json
import os
import sys

import sg_compat  # noqa: F401
import numpy as np
import torch
import torch.nn as nn
from sg_compat import make_env

from models.actor_critic import Actor
from models.world_model import WorldModel


def load_policy(path, od, ad):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if "pi" in ck:  # PPO-Lag
        pi = nn.Sequential(nn.Linear(od, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, ad))
        pi.load_state_dict(ck["pi"])
        mean, var = ck["norm_mean"], ck["norm_m2"] / ck["norm_n"]

        def act(o):
            x = torch.as_tensor(np.clip((o - mean) / np.sqrt(var + 1e-8), -10, 10), dtype=torch.float32)
            return np.clip(pi(x).numpy(), -1, 1)
        return act, "PPO-Lag"
    wm = WorldModel(od, ad, latent_dim=32, hidden=128, num_cost_heads=ck["K"], cost_hidden=128)
    wm.load_state_dict(ck["wm"])
    actor = Actor(32, ad, 128); actor.load_state_dict(ck["actor"])

    def act(o):
        z = wm.encode(torch.as_tensor((o - ck["mu"][0]) / ck["sd"][0], dtype=torch.float32))
        return actor.sample(z, deterministic=True)[0].numpy()
    return act, f"Model-based cost ensemble (kappa={ck['kappa']:g})"


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--task", default="SafetyPointGoal1-v0")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1000)
    args = ap.parse_args()
    if not os.path.exists(args.checkpoint):
        sys.exit(f"Checkpoint not found: {args.checkpoint}. Train one first (no simulated numbers are produced).")

    env = make_env(args.task)
    act, name = load_policy(args.checkpoint, env.observation_space.shape[0], env.action_space.shape[0])
    rets, costs = [], []
    for e in range(args.episodes):
        o, _ = env.reset(seed=args.seed + e)
        R = C = 0.0
        while True:
            o, r, c, term, trunc, _ = env.step(act(o))
            R += r; C += float(c > 0)
            if term or trunc:
                break
        rets.append(R); costs.append(C)
        print(f"  episode {e}: return {R:7.2f} | cost {C:5.0f}", flush=True)
    res = dict(checkpoint=args.checkpoint, policy=name, episodes=args.episodes,
               mean_return=float(np.mean(rets)), std_return=float(np.std(rets)),
               mean_cost=float(np.mean(costs)), std_cost=float(np.std(costs)),
               zero_violation_episode_frac=float(np.mean(np.array(costs) == 0)), returns=rets, costs=costs)
    print(f"{name}: return {res['mean_return']:.2f} +- {res['std_return']:.2f} | cost {res['mean_cost']:.1f} +- "
          f"{res['std_cost']:.1f} | zero-violation episodes {100 * res['zero_violation_episode_frac']:.0f}%")
    out = "results/eval_" + os.path.basename(args.checkpoint).replace(".pt", ".json")
    json.dump(res, open(out, "w"), indent=2)
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()

"""
Collect REAL transitions from Safety-Gymnasium for the cost-head experiments.

Policy: "held-action" random exploration (sample a uniform action, hold it for 5-25 steps).
Plain per-step random actions make the Point robot jitter in place (0 hazard contacts in
2,000 steps in our test); holding actions gives broader coverage (~3% contact rate).

Saved per transition: obs, action, reward, cost (binarised), next_obs, the signed distance from
the agent to the nearest hazard boundary AFTER the step (<= 0 means inside a hazard),
and the episode id (so train/test can be split by episode, avoiding leakage).

Usage:  python scripts/collect_data.py --steps 60000 --out data/pointgoal1_random.npz
"""
import argparse
import os
import time

import sg_compat  # noqa: F401  (NumPy-2 shim + path)
import numpy as np
from sg_compat import make_env, hazard_boundary_distance


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="SafetyPointGoal1-v0")
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/pointgoal1_random.npz")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    env = make_env(args.task)
    buf = {k: [] for k in ["obs", "act", "rew", "cost", "next_obs", "dist", "episode"]}
    ep, t0 = 0, time.time()
    obs, _ = env.reset(seed=args.seed * 10_000 + ep)
    hold, a = 0, np.zeros(2)
    for i in range(args.steps):
        if hold <= 0:
            a = rng.uniform(-1, 1, size=2)
            hold = int(rng.integers(5, 26))
        hold -= 1
        nobs, r, c, term, trunc, _ = env.step(a)
        buf["obs"].append(obs); buf["act"].append(a); buf["rew"].append(r)
        buf["cost"].append(float(c > 0)); buf["next_obs"].append(nobs)
        buf["dist"].append(hazard_boundary_distance(env)); buf["episode"].append(ep)
        obs = nobs
        if term or trunc:
            ep += 1
            obs, _ = env.reset(seed=args.seed * 10_000 + ep)
        if (i + 1) % 10000 == 0:
            print(f"  {i+1:>7} steps | episodes {ep} | contact rate so far "
                  f"{np.mean(buf['cost']):.3%} | {time.time()-t0:.0f}s", flush=True)

    out = {k: np.asarray(v, dtype=np.float32) for k, v in buf.items()}
    out["episode"] = out["episode"].astype(np.int32)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(args.out, **out)
    print(f"Saved {len(out['cost'])} transitions, {ep} complete episodes, "
          f"contact rate {out['cost'].mean():.3%}, mean episode cost "
          f"{out['cost'].sum() / max(ep, 1):.1f} -> {args.out}")


if __name__ == "__main__":
    main()

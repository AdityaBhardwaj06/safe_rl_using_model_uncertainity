"""
Verify the Safety-Gymnasium install and (optionally) save a REAL rendered frame.

Rendering needs an OpenGL backend. On Colab with a GPU use:   MUJOCO_GL=egl python scripts/verify_env.py
On a headless CPU machine try:                               MUJOCO_GL=osmesa python scripts/verify_env.py
If rendering fails, the script still verifies stepping and costs, and says so honestly --
do NOT replace the frame with a hand-drawn figure labelled as the environment.
"""
import os

import sg_compat  # noqa: F401
import numpy as np
from sg_compat import hazard_boundary_distance, make_env


def main():
    task = "SafetyPointGoal1-v0"
    print("=" * 65 + f"\n  Safe RL environment verification: {task}\n" + "=" * 65)
    env = make_env(task, render_mode="rgb_array")
    obs, _ = env.reset(seed=42)
    print(f"  observation shape : {obs.shape}\n  action space      : {env.action_space}")

    try:
        frame = env.render()
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        os.makedirs("artifacts", exist_ok=True)
        plt.figure(figsize=(5, 5), dpi=150); plt.imshow(frame); plt.axis("off")
        plt.title(f"{task} (rendered frame, seed 42)", fontsize=10)
        plt.savefig("artifacts/safety_point_goal1_frame.png", bbox_inches="tight"); plt.close()
        print(f"  rendered frame {frame.shape} -> artifacts/safety_point_goal1_frame.png")
    except Exception as e:  # noqa: BLE001
        print(f"  rendering unavailable here ({type(e).__name__}); set MUJOCO_GL=egl (GPU) or osmesa (CPU).")

    rng = np.random.default_rng(0)
    total, contacts, a = 0, 0, np.zeros(2)
    for t in range(3000):  # held-action exploration so the robot actually moves
        if t % 15 == 0:
            a = rng.uniform(-1, 1, 2)
        obs, r, c, term, trunc, _ = env.step(a)
        contacts += int(c > 0)
        assert (c > 0) == (hazard_boundary_distance(env) <= 0), "cost does not match hazard geometry"
        total += 1
        if term or trunc:
            env.reset()
    print(f"  {total} steps: {contacts} hazard-contact steps; cost always matched true hazard geometry")
    env.close()
    print("Environment verification PASSED.")


if __name__ == "__main__":
    main()

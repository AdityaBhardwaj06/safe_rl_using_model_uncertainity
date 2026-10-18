"""Small compatibility helpers for Safety-Gymnasium on Python 3.11+ / NumPy 2."""
import os
import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np

if not hasattr(np, "bool8"):          # gymnasium 0.28.1 still references np.bool8 (removed in NumPy 2)
    np.bool8 = np.bool_

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def make_env(task="SafetyPointGoal1-v0", **kw):
    import safety_gymnasium
    return safety_gymnasium.make(task, **kw)


def hazard_boundary_distance(env):
    """Signed distance from the agent to the nearest hazard boundary (<= 0 means inside a hazard)."""
    task = env.unwrapped.task
    agent = np.asarray(task.agent.pos)[:2]
    hz = np.asarray(task.hazards.pos)[:, :2]
    return float(np.min(np.linalg.norm(hz - agent, axis=1)) - task.hazards.size)

"""
Smoke Test: Standalone Training Loop Verification
Project: Safe Reinforcement Learning using Model Uncertainty (CS F317, BITS Pilani)

Tests the forward pass, imagined rollout simulation, 5-head cost ensemble uncertainty,
Lagrangian dual update, and backpropagation in PyTorch without requiring a GPU.
Run this script to generate terminal logs and verify code integrity before mid-sem.
"""

import sys
import os
import torch
import torch.optim as optim

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.world_model_wrapper import SimplifiedWorldModelPlanner


def run_smoke_test(num_iterations=20, batch_size=8, horizon=15, kappa=1.0, cost_limit=0.0):
    print("=" * 70)
    print("  SMOKE TEST: Epistemic Cost Ensemble SafeDreamer Training Loop")
    print("=" * 70)
    print(f"  Configuration:")
    print(f"  • Imagined Horizon (H) : {horizon} steps")
    print(f"  • Ensemble Size (K)    : 5 heads")
    print(f"  • Pessimism Weight (κ) : {kappa}")
    print(f"  • Safety Budget (b)    : {cost_limit}")
    print(f"  • Batch Size           : {batch_size}")
    print("-" * 70)

    # Initialize model
    planner = SimplifiedWorldModelPlanner(
        obs_dim=60,
        action_dim=2,
        latent_dim=128,
        horizon=horizon,
        kappa=kappa,
        cost_limit=cost_limit
    )

    optimizer = optim.Adam(planner.parameters(), lr=1e-3)

    print(f"{'Iter':>5} | {'Reward Return':>14} | {'Cost Return':>12} | {'Uncertainty (σ)':>16} | {'Lambda (λ)':>11} | {'Status'}")
    print("-" * 70)

    for it in range(1, num_iterations + 1):
        # Generate synthetic observation batch (representing robot lidar/velocities)
        synthetic_obs = torch.randn(batch_size, 60)

        # 1. Simulate H-step imagined rollouts inside world model
        rollouts = planner.imagine_rollouts(synthetic_obs)

        # 2. Compute actor loss under Lagrangian relaxation
        actor_loss, reward_ret, cost_ret = planner.compute_actor_lagrangian_loss(rollouts)

        # 3. Backpropagate policy gradients through imagined rollouts
        optimizer.zero_grad()
        actor_loss.backward()
        optimizer.step()

        # 4. Update Lagrangian dual multiplier lambda based on constraint violation
        violation, lam_val = planner.lagrangian.update(cost_ret)

        # Measure average epistemic disagreement (uncertainty) along the rollout
        avg_uncertainty = rollouts["uncertainties"].mean().item()

        status = "Safe" if cost_ret <= cost_limit else "VIOLATION"
        print(f"{it:>5} | {reward_ret:>14.3f} | {cost_ret:>12.3f} | {avg_uncertainty:>16.4f} | {lam_val:>11.4f} | {status}")

    print("-" * 70)
    print("✓ All 20 iterations completed successfully!")
    print("✓ World model dynamics, 5-head ensemble forward pass, backpropagation,")
    print("  and dual multiplier updates verified without error.")
    print("=" * 70)


if __name__ == "__main__":
    run_smoke_test()

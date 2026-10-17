"""
Environment Verification & Frame Rendering Script
Project: Safe Reinforcement Learning using Model Uncertainty (CS F317, BITS Pilani)

Checks Safety-Gymnasium installation, inspects action/observation spaces,
and renders sample camera frames for presentation slides.
"""

import os
import sys

def main():
    print("=" * 65)
    print("  Safe RL Environment Verification: SafetyPointGoal1-v0")
    print("=" * 65)

    try:
        import safety_gymnasium
        print("✓ safety_gymnasium is successfully imported!")
    except ImportError:
        print("✗ safety_gymnasium is NOT installed.")
        print("  Run: pip install safety-gymnasium")
        sys.exit(1)

    try:
        import matplotlib.pyplot as plt
        print("✓ matplotlib is available for frame export.")
    except ImportError:
        print("✗ matplotlib not found. Run: pip install matplotlib")
        sys.exit(1)

    task_name = "SafetyPointGoal1-v0"
    print(f"\n[1/3] Initializing task: {task_name} (rgb_array render mode)...")
    env = safety_gymnasium.make(task_name, render_mode="rgb_array")

    obs, info = env.reset(seed=42)
    print(f"✓ Environment reset successful!")
    print(f"  • Observation Shape : {obs.shape}")
    print(f"  • Action Space      : {env.action_space}")

    print("\n[2/3] Rendering RGB camera frame...")
    frame = env.render()
    print(f"✓ Rendered camera frame shape: {frame.shape}")

    out_dir = os.path.join(os.path.dirname(__file__), "..", "artifacts")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "safety_point_goal1_frame.png")

    plt.figure(figsize=(7, 7), dpi=150)
    plt.imshow(frame)
    plt.axis("off")
    plt.title("Safety-Gymnasium: SafetyPointGoal1-v0\n(Agent: Yellow | Goal: Green | Hazards: Purple Circles)",
              fontsize=11, weight='bold')
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved high-resolution camera frame to:\n  --> {os.path.abspath(out_path)}")

    print("\n[3/3] Simulating 10 random steps to verify cost calculation...")
    total_cost = 0.0
    for step in range(10):
        action = env.action_space.sample()
        obs, reward, cost, terminated, truncated, info = env.step(action)
        total_cost += cost
        if cost > 0:
            print(f"  [Step {step+1}] Hazard contact detected! Cost = {cost}")

    print(f"✓ Step verification complete. 10-step total cost: {total_cost}")
    env.close()
    print("\nEnvironment verification PASSED! Ready for SafeDreamer experiments.")
    print("=" * 65)

if __name__ == "__main__":
    main()

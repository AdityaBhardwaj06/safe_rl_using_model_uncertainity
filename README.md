# Safe Reinforcement Learning using Model Uncertainty — corrected codebase

**Course:** CS F317 Reinforcement Learning, BITS Pilani · **Instructor-in-charge:** Dr. Prashant Trivedi
**Team:** Aditya Bhardwaj & partner

Every number and figure produced by this repo comes from a script that runs the real
Safety-Gymnasium environment and writes a log in `results/`. No script contains hard-coded
"results". `results/RESULTS.md` is generated from those logs by `scripts/summarize_results.py`.

## 1. Idea (unchanged)

SafeDreamer uses a single cost head ĉ(z). We replace it with K bootstrapped heads and plan with the
pessimistic cost c̃(z) = μ_c(z) + κ·σ_c(z), where σ_c is the disagreement across heads.

## 2. What changed relative to the mid-sem draft, and why

| Old file | Problem | Now |
|---|---|---|
| `plot_results.py` | curves were hard-coded NumPy arrays | plots only logged CSV/JSON results |
| `eval_checkpoint.py` | never loaded a checkpoint; printed fixed numbers incl. the "0.021" | evaluates a real saved policy in the real env, errors if no checkpoint |
| `smoke_test.py` | one Adam optimiser over **all** parameters stepped on the actor loss → the model inflates reward / deflates cost (Colab log: reward 133,603, cost −58,081) | separate world-model / actor / critic / λ updates; `--buggy` reproduces the old failure; invariant checks every iteration |
| `ensemble_cost_head.py` | Sigmoid + BCELoss; bootstrap masks unused | logits + `binary_cross_entropy_with_logits`; Poisson bootstrap weights; optional randomized priors |
| `verify_env.py` / notebook cell 4 | notebook replaced the real frame with a hand-drawn matplotlib figure titled as the environment | real render (needs `MUJOCO_GL=egl`), plus a check that cost matches true hazard geometry |
| `world_model_wrapper.py`, `lagrangian.py` | not available for review | rewritten: `models/world_model.py`, `models/actor_critic.py`, `models/lagrangian.py` |

## 3. Installation (Python 3.12 / current Colab)

`pip install safety-gymnasium` does **not** work on Python ≥ 3.11: it pins `pygame==2.1.0` and
`mujoco==2.3.3` (no 3.12 wheels) and fails on import with
`ValueError: mutable default <class 'numpy.ndarray'> ... use default_factory`. Use:

```bash
pip install -r requirements.txt
git clone --depth 1 https://github.com/PKU-Alignment/safety-gymnasium.git
python scripts/patch_safety_gymnasium.py safety-gymnasium      # fixes 48 dataclass fields
pip install --no-deps -e safety-gymnasium
pip install --no-deps gymnasium==0.28.1 gymnasium-robotics==1.2.2
python scripts/verify_env.py                                   # MUJOCO_GL=egl on a Colab GPU for the frame
```
`scripts/sg_compat.py` also restores `np.bool8` (removed in NumPy 2, still used by gymnasium 0.28.1).

## 4. Reproducing every result (CPU is enough; ~1.5 h on one core)

```bash
python scripts/collect_data.py --steps 60000                    # 60 layouts, real transitions
python scripts/smoke_test.py --iters 300                        # fixed loop, b = 0
python scripts/smoke_test.py --iters 300 --buggy                # reproduces the original bug
python scripts/smoke_test.py --iters 150 --budget 25 --out results/smoke_test_fixed_b25.csv
cd scripts && python exp1_noise_floor.py --data ../data/pointgoal1_random.npz --out ../results/exp1_noise_floor.json
python exp2_uncertainty_detection.py --data ../data/pointgoal1_random.npz --steps 2500 --out ../results/exp2_uncertainty.json && cd ..
python scripts/train_ppo_lag.py --steps 60000 --budget 25 --seed 0
python scripts/train_mb_safe.py --kappa 0 --steps 30000 --seed 0   # also --kappa 1, and seed 1
python scripts/eval_checkpoint.py checkpoints/mb_kappa1_seed0.pt --episodes 10
python scripts/plot_results.py && python scripts/summarize_results.py
# or simply: bash run_all.sh
```

## 5. Experiments

* **Exp 1 — single-head noise floor.** Measures what a single cost head predicts in truly safe
  states of *unseen layouts* and what that adds up to over 15 imagined steps, a discounted return,
  and a 1000-step episode.
* **Exp 2A — never-seen hazards.** Ensemble trained only on states ≥ 0.25 from every hazard; does σ
  (or μ+κσ) rise at real contacts? Reported in probability space and in logit space.
* **Exp 2B — data scarcity.** Train on N ∈ {1k, 5k, 20k, all} transitions; does κ > 0 detect more true
  hazard states at a fixed false-alarm rate?
* **Online pilot.** Small model-based agent (κ = 0 vs κ = 1) and PPO-Lag in the real environment,
  logging *training-time* violations. This is a CPU-scale pilot (30k–60k steps, 1–2 seeds), **not**
  a SafeDreamer reproduction.

## 6. Scope and limitations (say these in the viva)

* The world model here is a small deterministic MLP latent model on vector observations, not DreamerV3's
  RSSM on pixels. Cost heads in Exp 1/2 take observations as a proxy for the latent z.
* The official SafeDreamer is built on DreamerV3's JAX code, so its checkpoints can't be loaded here.
  To test the idea inside SafeDreamer itself, the ensemble head must be added in JAX in that repo.
* Budget scaling: Safety-Gymnasium budgets are undiscounted per-episode costs. A discounted cost value
  is compared with d = (b/T)(1−γ^T)/(1−γ) ≈ b/10 (γ = 0.99, T = 1000). λ is capped at 50.
* Pilot training runs are short and few-seeded; treat them as sanity checks, not conclusions.

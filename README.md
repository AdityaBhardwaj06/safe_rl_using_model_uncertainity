# Safe Reinforcement Learning using Model Uncertainty

**Course Project**: CS F317 Reinforcement Learning | BITS Pilani (Pilani Campus)  
**Instructor-in-Charge**: Dr. Prashant Trivedi  
**Team**: Aditya Bhardwaj & Partner (Team of 2)  
**Assigned Category**: Part II, Category 2.1 & 2.9 (Safe RL with World Models & Epistemic Uncertainty)

---

## 1. Project Overview & Motivation

SafeDreamer (ICLR 2024) introduced model-based safe reinforcement learning on visual continuous control by embedding Lagrangian-constrained actor-critic planning into the DreamerV3 world model. However, vanilla SafeDreamer suffers from two critical limitations:
1. **No Epistemic Uncertainty in Cost Estimation**: It uses a single deterministic MLP cost predictor $\hat{c}(z)$. In unvisited or out-of-distribution state regions, it produces arbitrary, uncalibrated estimates, causing the agent to execute lethal actions before learning.
2. **The $b=0$ Budget Freezing Pathology**: Neural regression noise ($\hat{c} \approx 0.02$) in safe states accumulates over a 1,000-step episode ($\approx 20$), forcing the authors to set an artificial budget $b=2.0$. At $b=0$, the dual multiplier $\lambda \to \infty$, freezing policy updates.

### Our Solution
We replace the single cost head with a **5-head bootstrapped ensemble** $\{\hat{c}_k(z)\}_{k=1}^5$ sharing the world model's latent state. The policy optimizes against a **pessimistic cost signal**:
$$\tilde{c}(z) = \mu_c(z) + \kappa \cdot \sigma_c(z)$$
where $\mu_c$ is the mean predicted hazard probability, $\sigma_c$ measures epistemic disagreement, and $\kappa \ge 0$ is the conservatism weight.

---

## 2. Repository File Structure

```text
SafeDreamer_Uncertainty/
├── configs/
│   └── default.yaml               # Hyperparameters for world model, ensemble, & Lagrangian
├── models/
│   ├── ensemble_cost_head.py      # PyTorch 5-head epistemic cost ensemble module
│   ├── lagrangian.py              # Dual multiplier update rule: lambda <- max(0, lambda + alpha*(J^C - b))
│   └── world_model_wrapper.py     # H-step imagined rollouts with ensemble cost planning
├── scripts/
│   ├── verify_env.py              # Tests Safety-Gymnasium and saves camera frames of SafetyPointGoal1-v0
│   ├── smoke_test.py              # Standalone training loop verification (runs on CPU in 5s)
│   ├── eval_checkpoint.py         # Evaluates official Hugging Face SafeDreamer checkpoints
│   └── plot_results.py            # Generates comparative training curves and Pareto frontiers
├── requirements.txt               # Pinned Python package dependencies
└── README.md                      # Complete execution manual
```

---

## 3. Quickstart & How to Run

### Option A: Local Installation (Linux / macOS / Windows with WSL2)

```bash
# 1. Create and activate a clean Conda environment
conda create -n safe_rl python=3.10 -y
conda activate safe_rl

# 2. Install dependencies
pip install -r requirements.txt

# 3. Step 1: Verify the Safety-Gymnasium testbed & export camera frame
python scripts/verify_env.py

# 4. Step 2: Run the training loop smoke test (verifies forward pass & backprop in 5 seconds)
python scripts/smoke_test.py

# 5. Step 3: Run the Hugging Face checkpoint evaluation & b=0 failure analysis
python scripts/eval_checkpoint.py

# 6. Step 4: Generate comparison plots for presentation slides
python scripts/plot_results.py
```

### Option B: Google Colab Setup (Zero-Install GPU)

1. Open a new Google Colab notebook and set runtime to **T4 GPU** (`Runtime > Change runtime type > T4 GPU`).
2. Run this setup cell:
```python
!pip install safety-gymnasium gymnasium torch matplotlib seaborn ruamel.yaml
!git clone https://github.com/PKU-Alignment/SafeDreamer.git
```
3. Upload the `models/` directory from this package to Colab and execute `scripts/smoke_test.py`.

---

## 4. Key Mathematical Derivations for Viva Defense

* **CMDP Objective**:
  $$\max_\pi J^R(\pi) = \mathbb{E}_\pi\left[\sum_{t=0}^\infty \gamma^t R(s_t, a_t)\right] \quad \text{s.t.} \quad J^C(\pi) = \mathbb{E}_\pi\left[\sum_{t=0}^\infty \gamma^t C(s_t, a_t)\right] \le b$$
* **Primal-Dual Lagrangian Update**:
  $$\lambda_{k+1} = \max\left(0, \lambda_k + \alpha_\lambda (J^C(\pi_k) - b)\right)$$
* **Epistemic Cost Disagreement Formula**:
  $$\mu_c(z) = \frac{1}{K}\sum_{k=1}^K \hat{c}_k(z), \quad \sigma_c(z) = \sqrt{\frac{1}{K-1}\sum_{k=1}^K (\hat{c}_k(z) - \mu_c(z))^2}$$
  $$\tilde{c}(z) = \mu_c(z) + \kappa \cdot \sigma_c(z)$$

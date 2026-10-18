# Results summary (auto-generated from results/)

## Integration test (real data, H=15, K=5, kappa=1)

| run | iters passing checks | final img. reward (H) | max img. reward | final img. cost (H) | final lambda |
|---|---|---|---|---|---|
| fixed, b=0 | 300/300 | 0.0526 | 0.626 | 0.438 | 15.02 |
| buggy (original), b=0 | 3/300 | 9.34e+06 | 9.34e+06 | 9.01e-14 | 2.65 |
| fixed, b=25 | 150/150 | -0.123 | 0.627 | 0.661 | 0.00 |

## Exp 1: single cost head on held-out layouts (3 seeds, mean ± std)

| head | mean ĉ safe | median ĉ safe | mean ĉ far-safe (>0.5) | mean ĉ in hazard | AUROC | 15-step sum | discounted sum | 1000-step sum |
|---|---|---|---|---|---|---|---|---|
| bce | 0.0228 ± 0.0029 | 0.0000 | 0.0059 ± 0.0031 | 0.0668 ± 0.0402 | 0.779 ± 0.032 | 0.34 | 2.28 | 22.8 |
| mse | 0.0228 ± 0.0036 | 0.0007 | 0.0068 ± 0.0002 | 0.0716 ± 0.0383 | 0.570 ± 0.060 | 0.34 | 2.28 | 22.8 |

## Exp 2A: hazards never seen in training (K=5, 3 seeds) -- AUROC contact vs familiar states

| variant | AUROC μ | AUROC σ | AUROC μ+σ (current method) | AUROC σ (logit) | AUROC logit-pessimism ℓ̄+σ_ℓ (fix) | TPR@1%FPR μ+σ | TPR@1%FPR ℓ̄+σ_ℓ | mean σ (prob) at contact |
|---|---|---|---|---|---|---|---|---|
| plain | 0.283 ± 0.040 | 0.319 ± 0.026 | 0.295 ± 0.035 | 0.728 ± 0.020 | 0.298 ± 0.040 | 0.013 ± 0.009 | 0.014 ± 0.010 | 2.63e-07 |
| bootstrap | 0.282 ± 0.043 | 0.303 ± 0.025 | 0.290 ± 0.037 | 0.662 ± 0.129 | 0.293 ± 0.040 | 0.017 ± 0.012 | 0.018 ± 0.011 | 2.95e-07 |
| boot+prior | 0.283 ± 0.043 | 0.303 ± 0.024 | 0.290 ± 0.036 | 0.660 ± 0.132 | 0.293 ± 0.040 | 0.017 ± 0.012 | 0.016 ± 0.011 | 3.01e-07 |

## Exp 2B: data scarcity (train on first N transitions) -- detecting true hazard states on unseen layouts

| N | variant | contacts in train | AUROC κ=0 | AUROC κ=1 | AUROC κ=2 | AUROC logit-pess. κ=1 | TPR@5%FPR κ=0 | TPR@5%FPR κ=1 | TPR@5%FPR κ=2 | AUROC σ→error |
|---|---|---|---|---|---|---|---|---|---|---|
| 1000 | boot+prior | 35 | 0.416 ± 0.034 | 0.416 ± 0.035 | 0.417 ± 0.036 | 0.419 ± 0.032 | 0.107 ± 0.055 | 0.112 ± 0.055 | 0.135 ± 0.037 | 0.574 ± 0.160 |
| 1000 | bootstrap | 35 | 0.416 ± 0.035 | 0.416 ± 0.036 | 0.417 ± 0.037 | 0.418 ± 0.033 | 0.112 ± 0.060 | 0.113 ± 0.059 | 0.134 ± 0.038 | 0.574 ± 0.159 |
| 5000 | boot+prior | 86 | 0.518 ± 0.072 | 0.521 ± 0.071 | 0.521 ± 0.071 | 0.513 ± 0.071 | 0.149 ± 0.073 | 0.157 ± 0.080 | 0.157 ± 0.080 | 0.553 ± 0.095 |
| 5000 | bootstrap | 86 | 0.514 ± 0.078 | 0.515 ± 0.077 | 0.516 ± 0.076 | 0.506 ± 0.073 | 0.138 ± 0.069 | 0.138 ± 0.069 | 0.138 ± 0.069 | 0.544 ± 0.099 |
| 20000 | boot+prior | 538 | 0.752 ± 0.021 | 0.752 ± 0.021 | 0.752 ± 0.021 | 0.750 ± 0.023 | 0.116 ± 0.055 | 0.118 ± 0.054 | 0.118 ± 0.054 | 0.801 ± 0.008 |
| 20000 | bootstrap | 538 | 0.749 ± 0.015 | 0.749 ± 0.015 | 0.750 ± 0.015 | 0.746 ± 0.019 | 0.116 ± 0.051 | 0.119 ± 0.053 | 0.119 ± 0.053 | 0.799 ± 0.003 |
| 48000 | boot+prior | 1461 | 0.769 ± 0.069 | 0.770 ± 0.069 | 0.771 ± 0.070 | 0.769 ± 0.070 | 0.146 ± 0.096 | 0.156 ± 0.106 | 0.161 ± 0.111 | 0.857 ± 0.063 |
| 48000 | bootstrap | 1461 | 0.768 ± 0.071 | 0.768 ± 0.071 | 0.768 ± 0.071 | 0.766 ± 0.070 | 0.138 ± 0.084 | 0.130 ± 0.077 | 0.130 ± 0.077 | 0.855 ± 0.066 |

## Online training pilot (real env, SafetyPointGoal1-v0, b=25)

| method | seed | env steps | cumulative violations | violations after prefill | return (last 5 eps) | cost (last 5 eps) | final λ |
|---|---|---|---|---|---|---|---|
| PPO-Lag | 0 | 60000 | 6036 | 6036 | -2.47 | 28.8 | 79.66 |
| MB kappa=0 | 0 | 30000 | 1058 | 963 | -4.82 | 8.8 | 6.76 |
| MB kappa=0 | 1 | 30000 | 1377 | 1347 | 0.14 | 25.2 | 15.28 |
| MB kappa=1 | 0 | 30000 | 1779 | 1684 | -1.70 | 43.2 | 21.18 |
| MB kappa=1 | 1 | 30000 | 808 | 778 | -0.14 | 75.4 | 5.04 |

## Deterministic evaluation of final checkpoints (10 new layouts)

| checkpoint | mean return | mean cost | zero-violation episodes |
|---|---|---|---|
| mb_kappa0_seed0.pt | -4.89 ± 6.99 | 16.7 ± 17.3 | 40% |
| mb_kappa0_seed1.pt | -1.45 ± 0.79 | 1.9 ± 3.8 | 80% |
| mb_kappa1_seed0.pt | -1.88 ± 1.38 | 21.8 ± 34.4 | 40% |
| mb_kappa1_seed1.pt | 0.76 ± 1.94 | 76.3 ± 74.2 | 40% |
| ppo_lag_seed0.pt | -2.37 ± 1.89 | 36.2 ± 98.1 | 70% |

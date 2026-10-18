#!/usr/bin/env bash
# Reproduce every result in results/ and artifacts/ (about 1.5 h on one CPU core; faster on Colab).
set -e
python scripts/verify_env.py
python scripts/collect_data.py --steps 60000
python scripts/smoke_test.py --iters 300
python scripts/smoke_test.py --iters 300 --buggy
python scripts/smoke_test.py --iters 150 --budget 25 --out results/smoke_test_fixed_b25.csv
(cd scripts && python exp1_noise_floor.py --data ../data/pointgoal1_random.npz --out ../results/exp1_noise_floor.json)
(cd scripts && python exp2_uncertainty_detection.py --data ../data/pointgoal1_random.npz --steps 2500 --seeds 3 --only A --out ../results/exp2_uncertainty_A.json)
(cd scripts && python exp2_uncertainty_detection.py --data ../data/pointgoal1_random.npz --steps 2500 --seeds 2 --only B --out ../results/exp2_uncertainty_B.json)
python scripts/train_ppo_lag.py --steps 60000 --budget 25 --seed 0
for s in 0 1; do for k in 0 1; do python scripts/train_mb_safe.py --kappa $k --steps 30000 --seed $s; done; done
for c in checkpoints/*.pt; do python scripts/eval_checkpoint.py "$c" --episodes 10; done
python scripts/plot_results.py
python scripts/summarize_results.py

#!/usr/bin/env bash
# Fine-tune Kronos with VCA (arm A2: CE + ACF^2 loss) on the crypto-top20 universe.
# Reproduces the paper's main VCA recipe, averaged over seeds 42 44 46.
# The training data/window/loss hyperparameters live in finetune/config.py
# (Phase2Config + arm_preset), not in a yaml file.
#
# Usage: bash scripts/run_train_vca.sh [n_gpus]
set -euo pipefail

N_GPUS="${1:-4}"

for seed in 42 44 46; do
  torchrun --standalone --nproc_per_node="$N_GPUS" finetune/train_predictor.py \
    --arm A2 --seed "$seed" --run_name "vca_s${seed}"
done

echo "Checkpoints under \$SCRATCH/outputs/phase2/vca_s{42,44,46}/best_arval_agg_model"
echo "Baseline (arm A1, CE-only) is produced the same way with --arm A1."

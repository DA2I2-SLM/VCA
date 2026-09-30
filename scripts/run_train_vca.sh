#!/usr/bin/env bash
# Fine-tune a backbone with one of the paper's presets, over the paper's three
# seeds. All data/window/loss hyperparameters live in finetune/config.py
# (Phase2Config + arm_preset), not in a yaml -- the yamls configure EVALUATION.
#
# Usage: bash scripts/run_train_vca.sh [PRESET] [n_gpus]
#   PRESET defaults to CRYPTO_A2_L10_H32 (the paper's crypto VCA cell at H=32).
#
# Preset names are {DATASET}_{METHOD}_H{H}:
#   DATASET  CRYPTO | CSI300_PIT | CSI500PIT
#   METHOD   A1 = CE        MSE = MSE-AR        A2_L10 = VCA
#   H        8 | 16 | 32 | 48
# `python -c "from finetune.config import arm_preset; print(arm_preset())"` lists them.
#
# NOTE: the bare `A2` preset is NOT VCA (lr=1e-4, teacher-forced, k_train=3,
# per-window ACF target). Always use *_A2_L10_H*.
set -euo pipefail

cd "$(dirname "$0")/.."

PRESET="${1:-CRYPTO_A2_L10_H32}"
N_GPUS="${2:-4}"

for seed in 1 2 3; do
  RUN="$(echo "$PRESET" | tr '[:upper:]' '[:lower:]')_s${seed}"
  echo "== $PRESET  seed=$seed  ($N_GPUS GPUs) -> $RUN =="
  torchrun --standalone --nproc_per_node="$N_GPUS" finetune/train_predictor.py \
    --arm "$PRESET" --seed "$seed" --run_name "$RUN"
done

cat <<EOF

Checkpoints: \$SCRATCH/outputs/phase2/<run>/
  VCA and MSE-AR select best_arval_agg_model (lowest autoregressive-validation
  aggregate ACF^2 gap); CE selects best_model.

Evaluate one with:
  bash scripts/run_eval_baseline.sh configs/<dataset>_h<H>.yaml \\
       /scratch/\$USER/Kronos/outputs/phase2/<run>/best_arval_agg_model
EOF

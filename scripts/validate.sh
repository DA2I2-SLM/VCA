#!/usr/bin/env bash
# Quick end-to-end smoke test on a single GPU: download demo data, evaluate the
# frozen backbone, fine-tune for one epoch with VCA (arm A2), then run the
# inference-time control step on the resulting cache.
set -euo pipefail

cd "$(dirname "$0")/.."

CONFIG="configs/crypto_demo.yaml"

echo "== [1/4] Downloading demo crypto data =="
python data/download_binance.py --config "$CONFIG"

echo "== [2/4] Frozen backbone baseline (A0) =="
torchrun --standalone --nproc_per_node=1 eval/compute_baseline.py --config "$CONFIG"

echo "== [3/4] Fine-tuning one epoch with VCA (arm A2) =="
torchrun --standalone --nproc_per_node=1 finetune/train_predictor.py \
  --arm A2 --epochs 1 --run_name vca_validate

echo "== [4/4] Inference-time control over the cached rollouts =="
python eval/test_time_control.py "/scratch/$USER/Kronos/predictions/demo/forecast"

echo "validate.sh: all steps completed."

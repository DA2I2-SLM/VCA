#!/usr/bin/env bash
# Evaluate the frozen Kronos backbone (arm A0) on the crypto-top20 universe, and
# optionally a fine-tuned checkpoint from scripts/run_train_vca.sh.
#
# Usage:
#   bash scripts/run_eval_baseline.sh                       # frozen backbone only
#   bash scripts/run_eval_baseline.sh /path/to/best_arval_agg_model  # + fine-tuned checkpoint
set -euo pipefail

CONFIG="configs/crypto_top20.yaml"
N_GPUS="${N_GPUS:-8}"

python data/download_binance.py --config "$CONFIG"

echo "== Frozen backbone (A0) =="
torchrun --standalone --nproc_per_node="$N_GPUS" eval/compute_baseline.py --config "$CONFIG"

if [[ $# -ge 1 ]]; then
  CKPT="$1"
  echo "== Fine-tuned checkpoint: $CKPT =="
  torchrun --standalone --nproc_per_node="$N_GPUS" eval/compute_baseline.py \
    --config "$CONFIG" --model-id "$CKPT" \
    --predictions-cache-dir "/scratch/$USER/Kronos/predictions/vca_eval"
fi

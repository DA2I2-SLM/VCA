#!/usr/bin/env bash
# Evaluate the frozen backbone on one config's universe, and optionally a
# fine-tuned checkpoint from scripts/run_train_vca.sh. Both passes (FORE T=0.6
# N=10, VOL T=0.9 N=1) are defined in the config's `runs:` block.
#
# Usage:
#   bash scripts/run_eval_baseline.sh [CONFIG] [CHECKPOINT]
#     CONFIG      defaults to configs/crypto_top20_h32.yaml
#     CHECKPOINT  optional; a directory holding a saved predictor
#
# Crypto configs download their own data. CSI configs need a Qlib bundle exported
# first -- see data/export_csi300_qlib.py.
set -euo pipefail

cd "$(dirname "$0")/.."

CONFIG="${1:-configs/crypto_top20_h32.yaml}"
N_GPUS="${N_GPUS:-4}"
STEM="$(basename "$CONFIG" .yaml)"

if [[ "$STEM" == crypto* ]]; then
  python data/download_binance.py --config "$CONFIG"
fi

echo "== Frozen backbone: $CONFIG =="
torchrun --standalone --nproc_per_node="$N_GPUS" eval/compute_baseline.py --config "$CONFIG"

if [[ $# -ge 2 ]]; then
  CKPT="$2"
  TAG="$(basename "$(dirname "$CKPT")")"
  echo "== Fine-tuned checkpoint: $CKPT =="
  torchrun --standalone --nproc_per_node="$N_GPUS" eval/compute_baseline.py \
    --config "$CONFIG" --model-id "$CKPT" --run-tag "$TAG" \
    --predictions-cache-dir "/scratch/$USER/Kronos/predictions/${STEM}-${TAG}"
fi

echo "Metrics JSON: outputs/baseline_${STEM}.json"

#!/usr/bin/env bash
# End-to-end plumbing check on ONE GPU, crypto only (no Qlib bundle needed, so a
# fresh clone can run it). It exercises every stage the paper's pipeline uses:
# download -> frozen eval -> fine-tune -> eval the checkpoint -> test-time control
# -> classical baselines. Every step is idempotent, so a re-run resumes.
#
# This is NOT a paper-quality run: it fine-tunes for 1 epoch on 1 GPU, where the
# paper's cells are 10 epochs, 3 seeds and 4 GPUs -- see scripts/run_train_vca.sh.
# Everything else is the paper's crypto H=32 setting, so the download is the full
# 20-symbol universe and takes a while on a cold /scratch.
#
# Usage: bash scripts/validate.sh
set -euo pipefail

cd "$(dirname "$0")/.."

CONFIG="configs/crypto_top20_h32.yaml"
CACHE="/scratch/$USER/Kronos/predictions/crypto_top20_h32"
CKPT_DIR="/scratch/$USER/Kronos/outputs/phase2/vca_validate"

echo "== [0/6] Preflight: imports and CUDA =="
python -c "
import torch
from finetune.config import arm_preset
from finetune.train_predictor import acf2_loss
from eval.metrics import aggregate_qlike, acf_sq_returns
import eval.compute_baseline, eval.inference_chronos, eval.garch_har_baseline
print(f'torch {torch.__version__}  cuda={torch.cuda.is_available()}  '
      f'devices={torch.cuda.device_count()}  presets={len(arm_preset())}')
assert torch.cuda.is_available(), 'validate.sh needs a GPU'
"

echo "== [1/6] Downloading crypto data =="
python data/download_binance.py --config "$CONFIG"

echo "== [2/6] Frozen backbone baseline =="
torchrun --standalone --nproc_per_node=1 eval/compute_baseline.py --config "$CONFIG"

echo "== [3/6] Fine-tuning one epoch with VCA =="
torchrun --standalone --nproc_per_node=1 finetune/train_predictor.py \
  --arm CRYPTO_A2_L10_H32 --epochs 1 --run_name vca_validate

echo "== [4/6] Evaluating the fine-tuned checkpoint =="
torchrun --standalone --nproc_per_node=1 eval/compute_baseline.py \
  --config "$CONFIG" --model-id "$CKPT_DIR/best_arval_agg_model" \
  --run-tag vca_validate \
  --predictions-cache-dir "$CACHE-vca_validate"

echo "== [5/6] Inference-time control over the cached rollouts =="
python eval/test_time_control.py "$CACHE/forecast"

echo "== [6/6] Classical baselines (GARCH(1,1), HAR-RV; CPU) =="
python eval/garch_har_baseline.py --config "$CONFIG"

cat <<'EOF'

validate.sh: all steps completed.

This was a plumbing check, not a result. It fine-tuned for 1 epoch on 1 GPU;
the paper's cells are 10 epochs, 3 seeds, 4 GPUs.

Next steps:
  * full crypto run      -> bash scripts/run_train_vca.sh CRYPTO_A2_L10_H32
  * CSI300 / CSI500      -> install a Qlib cn_data bundle, then
                            python data/export_csi300_qlib.py --config configs/csi300_h32.yaml
                            (the exporter reads an existing bundle; it cannot download one)
EOF

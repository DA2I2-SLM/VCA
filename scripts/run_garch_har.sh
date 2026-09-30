#!/usr/bin/env bash
# Classical volatility baselines -- GARCH(1,1) (Bollerslev 1986) and HAR-RV
# (Corsi 2009) -- on exactly the test windows the neural models are scored on.
# CPU only: no GPU, no checkpoint. Each model is fit once per symbol on
# train+val, then forecasts cumulative variance over each window's horizon.
#
# Neither produces a price path, so only the volatility metrics are reported
# (sigma^2-MAE, QLIKE, calibration ratio); RankIC is not defined for them.
#
# Usage: bash scripts/run_garch_har.sh [CONFIG...]
#   Defaults to the four crypto horizons. Pass CSI configs once exported.
set -euo pipefail

cd "$(dirname "$0")/.."

CONFIGS=("$@")
if [[ ${#CONFIGS[@]} -eq 0 ]]; then
  CONFIGS=(configs/crypto_top20_h{8,16,32,48}.yaml)
fi

for cfg in "${CONFIGS[@]}"; do
  echo "== $cfg =="
  python eval/garch_har_baseline.py --config "$cfg"
done

echo "Results: outputs/garch_har_<config stem>.json"

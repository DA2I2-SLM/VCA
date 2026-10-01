<div align="center">
<img src="assets/vca_logo.svg" height=140 alt="VCA">
  <h1><b> VCA: Volatility-Clustering Adaptation </b></h1>
  <p><i>Fine-tuning financial time-series foundation models to match how volatility clusters, not just to predict the next bar.</i></p>
</div>

<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![Backbone](https://img.shields.io/badge/backbones-Kronos%20%7C%20Chronos--T5-orange)](#bench)
[![arXiv](https://img.shields.io/badge/arXiv-2609.37715-b31b1b.svg)](https://arxiv.org/abs/2609.37715)

</div>

<div align="center">

📖 [**Overview**](#overview) **|**
🚀 [**Install**](#install) **|**
🔧 [**Usage**](#usage) **|**
🎯 [**Benchmarks**](#bench) **|**
📊 [**Baselines**](#baselines) **|**
📂 [**Structure**](#structure)

</div>

---

## <a name="overview"></a> 📖 Overview

Returns are weakly predictable; their *squared* returns are not — volatility arrives in clusters.
Cross-entropy never asks a foundation model to reproduce that, so its rollouts flatten the temporal
dependence of volatility. **VCA** adds the missing term during fine-tuning, matching the ACF² of a
Gumbel-softmax differentiable rollout ($\bar\rho_k$) to that of the ground-truth window
($\bar\rho^{*}_k$), both batch-averaged before squaring:

$$\mathcal{L} = \mathcal{L}_{\mathrm{CE}} + \lambda \cdot \mathcal{L}_{\mathrm{ACF}^2},
\qquad
\mathcal{L}_{\mathrm{ACF}^2} = \sum_{k=1}^{K} w_k \left( \bar{\rho}_k - \bar{\rho}^{*}_k \right)^2,
\qquad
w_k = e^{-k/\gamma}$$

A training-free variant re-weights the `N` rollouts evaluation already draws. Details are in the
[paper](https://arxiv.org/abs/2609.37715); hyperparameters in [Benchmarks](#bench).

> [!IMPORTANT]
> If you use this code or the method, please cite the paper:
>
> ```bibtex
> @misc{nguyen2026vca,
>       title={Volatility-Clustering Adaptation for Financial Time Series},
>       author={Manh Nguyen and Minh Hoang Nguyen and Huu Hiep Nguyen and Van Dai Do and Hung Le},
>       year={2026},
>       eprint={2609.37715},
>       archivePrefix={arXiv},
>       primaryClass={cs.LG},
>       url={https://arxiv.org/abs/2609.37715},
> }
> ```

---

## <a name="install"></a> 🚀 Installation

### Cloning

```bash
git clone https://github.com/DA2I2-SLM/VCA.git VCA
cd VCA
```

### Installing dependencies

```bash
conda create -n vca python=3.10 -y
conda activate vca
pip install -r requirements.txt
```

---

## <a name="usage"></a> 🔧 Usage

Training hyperparameters live in `finetune/config.py` (`Phase2Config` + `arm_preset`); the YAMLs
in `configs/` configure **evaluation**. Preset names are `{DATASET}_{METHOD}_H{H}`, with
`DATASET ∈ {CRYPTO, CSI300_PIT, CSI500PIT}`, `METHOD ∈ {A1, MSE, A2_L10}` (see
[Baselines](#baselines) for which paper method each is) and `H ∈ {8, 16, 32, 48}`. List them with:

```bash
python -c "from finetune.config import arm_preset; print(arm_preset())"
```

**Fine-tune VCA** on crypto at H=32, 4 GPUs:

```bash
torchrun --standalone --nproc_per_node=4 finetune/train_predictor.py \
  --arm CRYPTO_A2_L10_H32 --seed 1
```

Swap the preset for the CE baseline (`CRYPTO_A1_H32`) or the MSE-AR ablation
(`CRYPTO_MSE_H32`). Ablations that only change one scalar are CLI overrides —
`--k_train`, `--lambda_acf`, `--tau_lag`, `--data_fraction`, `--data_select`, `--lr`, `--epochs`.

**Evaluate** the frozen backbone, then a fine-tuned checkpoint:

```bash
python data/download_binance.py --config configs/crypto_top20_h32.yaml
torchrun --standalone --nproc_per_node=4 eval/compute_baseline.py \
  --config configs/crypto_top20_h32.yaml

torchrun --standalone --nproc_per_node=4 eval/compute_baseline.py \
  --config configs/crypto_top20_h32.yaml \
  --model-id /scratch/$USER/Kronos/outputs/phase2/<run>/best_arval_agg_model \
  --run-tag vca
```

**Apply the inference-time control** to the rollouts cached by that run:

```bash
python eval/test_time_control.py /scratch/$USER/Kronos/predictions/crypto_top20_h32/forecast
```

### ⚡ Quick validation

One GPU, crypto only, no Qlib bundle required — exercises every stage of the pipeline
(download → frozen eval → fine-tune → eval checkpoint → test-time control → classical baselines):

```bash
bash scripts/validate.sh
```

---

## <a name="bench"></a> 🎯 Benchmarks

### ☝️ Tested backbones

| Backbone | Params | Architecture | Horizons | Adaptation |
|---|---|---|---|---|
| Kronos-base | 102M | decoder-only over BSQ-quantized OHLCV tokens | 8, 16, 32, 48 | full fine-tuning, no adapters |
| Chronos-T5-small | — | encoder-decoder over binned univariate values | 16, 32 | full fine-tuning, no adapters |

The checkpoints are `NeoQuasar/Kronos-base` + `NeoQuasar/Kronos-Tokenizer-base` and
`amazon/chronos-t5-small`. The Chronos path always runs in float32, ignoring a config's `fp16`.

### ✌️ Supported datasets

| Dataset | Source | Bars | Symbols | Train / Val / Test years | Windows (train/val/test) |
|---|---|---|---|---|---|
| CSI300 | Qlib `cn_data` | daily | 515 | 2015–17 / 2018 / 2019–20 | 3991 / 1836 / 3557 |
| CSI500 | Qlib `cn_data` | daily | 937 | 2015–17 / 2018 / 2019–20 | 6246 / 2991 / 5749 |
| Crypto top-20 | Binance public REST | 15-minute | 20 | 2021–23 / 2024-H1 / 2024-H2–25 | 63514 / 10860 / 19308 |

Window counts are at the main cell (H = 32, W = 160). Crypto uses disjoint windows
(`stride = H`); CSI overlaps them (`stride = 8`), since daily A-share history is far shorter.

Crypto data downloads itself. The CSI universes need a Qlib bundle installed first — the exporter
reads a local bundle and cannot fetch one:

```bash
python -m qlib.run.get_data qlib_data --target_dir ~/.qlib/qlib_data/cn_data --region cn
python data/export_csi300_qlib.py --config configs/csi300_h32.yaml
python data/export_csi300_qlib.py --config configs/csi500_h32.yaml --market csi500
```

Pass `--config` so the shipped YAML's `symbols:` list is used verbatim: bundle vintage otherwise
changes the universe, which is the most likely reason CSI numbers do not reproduce.

### 🔬 Running full benchmark experiments

```bash
# one method x dataset x horizon, over the paper's three seeds
bash scripts/run_train_vca.sh CRYPTO_A2_L10_H32 4
bash scripts/run_train_vca.sh CSI300_PIT_A2_L10_H16 4

# evaluate, then the classical baselines
bash scripts/run_eval_baseline.sh configs/csi300_h16.yaml /path/to/best_arval_agg_model
bash scripts/run_garch_har.sh configs/csi300_h16.yaml
```

Chronos uses a separate trainer with its own CLI (`--dataset`, and `--arm {A1,A2,MSE}` rather than
a preset name):

```bash
torchrun --standalone --nproc_per_node=4 finetune/train_predictor_chronos.py \
  --dataset csi300 --arm A2 --pred_len 32 --lambda_acf 10 --k_train 1 --seed 1
```

**Fixed hyperparameters** (identical across every dataset and horizon — no per-dataset tuning):

| | |
|---|---|
| Optimizer | AdamW, $\beta = (0.9, 0.95)$, weight decay 0.1, grad clip 3.0 |
| Learning rate / epochs | 5e-5 / 10 |
| Effective batch | 32 (the per-GPU size shrinks as H grows; the AR rollout holds the graph across all H steps) |
| Loss | $\lambda_{\mathrm{acf}} = 10$, $K = 1$, $\gamma = 2.0$, Gumbel $\tau$ 0.5 → 0.1 |
| Lookback | W = 160 |
| Seeds | 1, 2, 3 |
| Hardware | 4 × V100 |

**Evaluation** runs two passes per config, defined in each YAML's `runs:` block:

| Pass | Temperature | Rollouts | Reported |
|---|---|---|---|
| FORE | 0.6 | 10 | path RankIC, return IC, ACF² gap |
| VOL | 0.9 | 1 | $\sigma^2$-MAE, $\sigma^2$-MSE, QLIKE |

---

## <a name="baselines"></a> 📊 Baselines

| | Method | How to run | Notes |
|---|---|---|---|
| **Frozen** | pretrained backbone, no fine-tuning | `compute_baseline.py` with no `--model-id` | reference point for every Δ |
| **CE** | next-token cross-entropy only | `--arm {DATASET}_A1_H{H}` | the standard fine-tuning recipe |
| **MSE** | CE + squared relative price error on the AR rollout | `--arm {DATASET}_MSE_H{H}` | path-level loss without the ACF² term |
| **VCA** (ours) | CE + ACF² matching | `--arm {DATASET}_A2_L10_H{H}` | |
| **GARCH(1,1)** | Bollerslev (1986); Gaussian QMLE, Nelder–Mead | `garch_har_baseline.py --config ...` | variance only ⇒ RankIC / Score are `---` |
| **HAR-RV** | Corsi (2009); OLS on realized-variance lags {1, 5, 22} | same | same |

Both classical models are fit **once per symbol** on train+val log-returns and forecast cumulative
variance by iterated one-step-ahead prediction — CPU only, no GPU, no checkpoint. 

<details>
<summary><b>Ablations reported in the paper</b></summary>

| Ablation | Range | How to run |
|---|---|---|
| Lag count `K` | 1, 2, 4, 8 | `--k_train N` on a `*_A2_L10_H*` preset |
| Loss weight `λ` | 5, 10, 20, 40 | `--lambda_acf N` |
| Lookback `W` | 80, 160, 320 | set `lookback` in `finetune/config.py` — there is no CLI flag |
| Data efficiency | {1%, 10%, 100%} × {first, last} | `--data_fraction 0.1 --data_select first` |
| Checkpoint selection | teacher-forced vs. AR validation gap | `ar_val_select` in the preset |

`--data_select` accepts `first`, `last`, `random`, `high_acf2`, `low_acf2`.

</details>

---

## <a name="structure"></a> 📂 Project structure

```text
VCA/
├── scripts/
│   ├── validate.sh              # 1-GPU end-to-end plumbing check
│   ├── run_train_vca.sh         # one preset over seeds 1 2 3
│   ├── run_eval_baseline.sh     # frozen backbone, and optionally a checkpoint
│   └── run_garch_har.sh         # GARCH(1,1) + HAR-RV on a config's windows
├── configs/                     # EVALUATION configs: {dataset}_h{H}[_chronos].yaml
│   ├── crypto_top20_h{8,16,32,48}.yaml
│   ├── crypto_top20_h{16,32}_chronos.yaml
│   ├── csi300_h{8,16,32,48}.yaml
│   ├── csi300_h{16,32}_chronos.yaml
│   ├── csi500_h{8,16,32,48}.yaml
│   └── csi500_h{16,32}_chronos.yaml
├── data/
│   ├── download_binance.py      # public Binance REST downloader (no API key)
│   └── export_csi300_qlib.py    # CSI300/CSI500 exporter over a local Qlib bundle
├── model/                       # Kronos tokenizer + predictor architecture
│   ├── kronos.py
│   └── module.py
├── finetune/
│   ├── config.py                # Phase2Config, symbol universes, arm_preset()
│   ├── dataset.py               # strided-window dataset over the OHLCV CSVs
│   ├── train_predictor.py       # Kronos trainer: CE + ACF^2 / MSE-AR, DDP
│   ├── train_predictor_chronos.py   # Chronos-T5 trainer (same losses)
│   └── utils/training_utils.py
├── eval/
│   ├── inference.py             # autoregressive rollout sampling (Kronos)
│   ├── inference_chronos.py     # same, for the Chronos pipeline
│   ├── compute_baseline.py      # evaluation entrypoint, both backbones
│   ├── garch_har_baseline.py    # classical volatility baselines
│   ├── test_time_control.py     # inference-time re-weighting over cached rollouts
│   └── metrics.py               # RankIC, ACF^2, sigma^2-MAE/MSE, QLIKE
├── assets/vca_logo.svg
├── requirements.txt
├── README.md
└── LICENSE
```

---

## 🙏 Acknowledgements

* [Kronos](https://huggingface.co/NeoQuasar/Kronos-base) — pretrained OHLCV foundation model
* [Chronos](https://github.com/amazon-science/chronos-forecasting) — pretrained univariate
  forecaster used as the second backbone
* [Qlib](https://github.com/microsoft/qlib) — CSI300 / CSI500 daily data
* [Hugging Face Hub](https://huggingface.co/) — model hosting

---

## 👋 About us

VCA is developed by the authors of
[*Volatility-Clustering Adaptation for Financial Time Series*](https://arxiv.org/abs/2609.37715).
Released under the [MIT License](LICENSE).

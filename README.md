<div align="center">
<img src="assets/vca_logo.svg" height=140 alt="VCA">
  <h1><b> VCA: Volatility-Clustering Adaptation </b></h1>
  <p><i>Fine-tuning financial time-series foundation models to match how volatility clusters, not just to predict the next bar.</i></p>
</div>

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)

</div>

<div align="center">

🚀 [**Install**](#install) **|**
🔧 [**Usage**](#usage) **|**
🧪 [**Reproduce**](#reproduce) **|**
🎯 [**Benchmarks**](#bench) **|**
📂 [**Structure**](#structure)

</div>

Financial returns are weakly predictable, but their *squared* returns are not: volatility comes
in clusters, and how well a model's rollouts reproduce that clustering is a signal that
next-token cross-entropy never sees. **VCA** adds that signal in two independent, composable
places:

- **Training-time.** During fine-tuning, we replace token sampling with a Gumbel-softmax
  straight-through estimator to get a differentiable autoregressive rollout, then add a loss term
  matching the autocorrelation of squared returns (ACF²) of that rollout to the realized ACF² of
  the ground-truth future window — on top of, not instead of, the usual next-token cross-entropy.
- **Inference-time, training-free.** At test time the model already draws `N` autoregressive
  rollouts per window; instead of a plain mean over them, we re-weight the `N` rollouts toward a
  volatility-clustering target (e.g. penalizing over-clustering) with no retraining and no extra
  forward passes.

This repo contains the minimal code to run both on one representative dataset (20 liquid Binance
USDT pairs, 15-minute bars) against `NeoQuasar/Kronos-base`, a pretrained OHLCV foundation model.

---

## <a name="install"></a> 🚀 Installation

```bash
git clone https://github.com/DA2I2-SLM/VCA.git VCA
cd VCA

conda create -n vca python=3.10 -y
conda activate vca
pip install -r requirements.txt
```

The Kronos tokenizer and predictor weights (`NeoQuasar/Kronos-Tokenizer-base`,
`NeoQuasar/Kronos-base`) are downloaded automatically from Hugging Face on first use.

---

## <a name="usage"></a> 🔧 Usage

**Fine-tune with VCA** (Method `A2`: cross-entropy + ACF² loss), 4 GPUs:

```bash
torchrun --standalone --nproc_per_node=4 finetune/train_predictor.py --arm A2 --seed 42
```

**Fine-tune the cross-entropy-only baseline** (Method `A1`) the same way with `--arm A1`.

**Fine-tune the MSE-AR ablation** (Method `A1-MSE`: cross-entropy + relative squared error of
predicted vs. true close, `(p̂/p − 1)²`, on the same autoregressive rollout ACF² uses, instead of
ACF²-matching) with `--arm A1-MSE`.

**Evaluate the frozen backbone / a fine-tuned checkpoint** on the crypto universe:

```bash
python data/download_binance.py --config configs/crypto_top20.yaml
torchrun --standalone --nproc_per_node=8 eval/compute_baseline.py --config configs/crypto_top20.yaml
```

**Apply the inference-time control** to the cached rollouts from the run above:

```bash
python eval/test_time_control.py /scratch/$USER/Kronos/predictions/h32/forecast
```

#### ⚡ Quick validation

A single-GPU smoke test covering all four steps above on a 5-symbol slice:

```bash
bash scripts/validate.sh
```

---

## <a name="reproduce"></a> 🧪 Reproducing results

| Script | What it does |
|--------|--------------|
| `scripts/validate.sh` | End-to-end smoke test on 1 GPU (5 symbols) |
| `scripts/run_train_vca.sh` | Fine-tunes VCA (Method `A2`) over seeds `42 44 46` |
| `scripts/run_eval_baseline.sh` | Evaluates the frozen backbone, and optionally a fine-tuned checkpoint |

The full paper additionally sweeps hyperparameters (lag count `K`, lookback `W`, loss weight
`λ`, lag decay `γ`), evaluates on two more universes (CSI300, CSI500 via Qlib), and compares
against more baselines. That grid is out of scope for this minimal release; this repo covers
the core method end-to-end on one dataset.

---

## <a name="bench"></a> 🎯 Benchmarks

| | |
|---|---|
| **Model** | `NeoQuasar/Kronos-base` (predictor) + `NeoQuasar/Kronos-Tokenizer-base` |
| **Universe** | 20 liquid Binance USDT spot pairs, 15-minute bars (`configs/crypto_top20.yaml`) |
| **Methods** | `A1` (CE only), `A1-MSE` (CE + MSE-AR), `A2` (CE + ACF² = VCA) |
| **Metrics** | Path-wise RankIC, $\sigma^2$-MAE and $\sigma^2$-MSE of realized variance (`eval/metrics.py`) |

---

## <a name="structure"></a> 📂 Project structure

```text
VCA/
├── scripts/
│   ├── validate.sh          # 1-GPU smoke test
│   ├── run_train_vca.sh     # fine-tune Method A2 over 3 seeds
│   └── run_eval_baseline.sh # evaluate frozen backbone / a checkpoint
├── configs/
│   ├── crypto_top20.yaml    # full 20-symbol recipe
│   └── crypto_demo.yaml     # 5-symbol config for validate.sh
├── data/
│   └── download_binance.py  # public Binance REST downloader (no API key)
├── model/                    # Kronos tokenizer + predictor architecture
├── finetune/
│   ├── config.py             # Phase2Config dataclass + arm_preset('A1'|'A1-MSE'|'A2')
│   ├── dataset.py            # strided-window dataset over the OHLCV CSVs
│   ├── train_predictor.py    # training loop: CE + ACF² loss, DDP
│   └── utils/training_utils.py
└── eval/
    ├── inference.py           # autoregressive rollout sampling
    ├── compute_baseline.py    # cross-sectional eval entrypoint (RankIC, MAE, MSE)
    ├── test_time_control.py   # inference-time re-weighting over cached rollouts
    └── metrics.py             # RankIC, ACF², σ²-MAE, σ²-MSE definitions
```

---

## Acknowledgements

* [Kronos](https://huggingface.co/NeoQuasar/Kronos-base) — the pretrained OHLCV foundation
  model this work fine-tunes
* [Hugging Face Hub](https://huggingface.co/) — model hosting and downloads

This project is released under the [MIT License](LICENSE).

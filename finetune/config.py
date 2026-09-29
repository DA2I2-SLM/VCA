from dataclasses import dataclass, field
import os
from typing import List

_SCRATCH = f'/scratch/{os.getenv("USER", "user")}/Kronos'


CRYPTO_TOP20 = [
    'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT',
    'ADAUSDT', 'DOGEUSDT', 'AVAXUSDT', 'DOTUSDT', 'MATICUSDT',
    'LINKUSDT', 'LTCUSDT', 'BCHUSDT', 'ATOMUSDT', 'UNIUSDT',
    'ETCUSDT', 'XLMUSDT', 'FILUSDT', 'NEARUSDT', 'APTUSDT',
]


@dataclass
class Phase2Config:
    # ── data ─────────────────────────────────────────────────────────────────
    data_dir: str = field(default_factory=lambda: f'{_SCRATCH}/data')
    frequency: str = '15m'  # CSV suffix: '15m' crypto | '1d' CSI300
    symbols: List[str] = field(default_factory=lambda: list(CRYPTO_TOP20))
    lookback: int = 160
    pred_len: int = 32
    stride: int = 32
    max_context: int = 512
    clip: float = 5.0
    zero_vol_amount: bool = True  # paper Appendix D: exclude vol/amount for crypto

    train_start: str = '2021-01-01'
    train_end: str = '2023-12-31'
    val_start: str = '2024-01-01'
    val_end: str = '2024-06-30'
    # Thread A: optional 2nd val fold for cross-validation selection. When set,
    # checkpoint is chosen by the MEAN acf2_gap across both folds (regime-robust).
    val2_start: str = ''
    val2_end: str = ''

    # fraction of training windows to use, taken chronologically first per symbol.
    # A2=1.0, A5=0.5, A6=0.25, A7=0.10
    data_fraction: float = 1.0
    data_select: str = None       # None|'high_acf2'|'low_acf2': select data_fraction by future-window ACF²
    acf_select_lags: int = 3
    adaptive_k: bool = False      # per-batch discrepancy/SNR-weighted lags (self-selects K)
    k_max: int = 10
    adaptive_prior_tau: float = 0.0
    adaptive_mode: str = 'discrepancy'
    over_penalty: float = 1.0
    acf_pooled: bool = False

    # ── arm ──────────────────────────────────────────────────────────────────
    arm: str = 'A1'  # 'A1' (CE only) or 'A2' (CE + ACF²)

    # ACF² rollout mode: how the predicted trajectory feeding the ACF² loss is produced.
    #   'teacher_forced' — one-step soft-decode of teacher-forced logits (default, cheap)
    #   'full_ar'        — fully differentiable autoregressive rollout (faithful to test,
    #                      ~H× slower, large autograd graph; see ar_rollout_prices)
    rollout_mode: str = 'teacher_forced'

    # AR val selection: also compute an autoregressive (greedy) val acf2_gap each epoch and
    # save `best_arval_model` by it. Matches the test generation procedure (test is
    # autoregressive); the default teacher-forced val metric does not. Adds per-epoch eval cost.
    ar_val_select: bool = False

    # save every epoch's checkpoint (epoch_{n}/) — for oracle test-epoch selection studies
    save_all_epochs: bool = False

    # Gumbel straight-through in the AR training rollout: hard=True feeds the argmax token
    # forward (matches test's discrete tokens) but keeps soft gradients backward. Default
    # hard=False (fully soft mixture — smoother gradient, but fed-back tokens aren't real).
    gumbel_hard: bool = False

    # ── training ─────────────────────────────────────────────────────────────
    epochs: int = 10
    # per-GPU batch size; effective = batch_size × n_gpus.
    # ACF² loss requires effective ≥ 256 for stable gradient → 64 × 4 GPUs = 256.
    batch_size: int = 64
    # Micro-batches accumulated before each optimizer step; effective batch becomes
    # batch_size x n_gpus x grad_accum. The AR rollout holds the graph across all H
    # steps, so batch_size has to shrink as H grows (8 at H32, 4 at H48) and the ACF²
    # gradient ends up 8-16x below the >=256 this file documents as stable. Accumulation
    # buys that back without extra memory.
    # CAVEAT for acf_loss_agg=True: the aggregate ACF loss averages over the batch INSIDE
    # the loss, so accumulating gradients of per-micro-batch aggregates is NOT identical
    # to one true large-batch aggregate. It reduces optimiser-step noise, not the
    # within-loss estimator noise. Read the result with that in mind.
    grad_accum: int = 1
    # BEST CONFIG (2026-07-18): lr=5e-5 beats 1e-4 on all 4 metrics (RankIC 0.068 vs 0.058).
    # Every pre-2026-07-18 preset pins lr=1e-4 explicitly so its published numbers stay reproducible.
    lr: float = 5e-5
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    weight_decay: float = 0.1
    grad_clip: float = 3.0
    seed: int = 42
    num_workers: int = 2

    # ── ACF² loss (A2 only) ───────────────────────────────────────────────────
    lambda_acf: float = 10.0     # gradient-norm calibrated: ‖∇CE‖/‖∇ACF²‖ ≈ 137 → λ=10 ≈ 7%
    lambda_schedule: str = 'const'  # 'const' | 'warmup' (ramp λ_acf 0→λ over warmup_frac of steps)
    lambda_mse: float = 0.0
    lambda_lev: float = 0.0       # B: leverage-effect match
    lambda_kurt: float = 0.0      # B: kurtosis (fat-tail) match
    lambda_var: float = 0.0       # B: variance/dispersion-level match      # optional MSE(pred close, true close) term on the AR rollout (ablation)
    lambda_dir: float = 0.0      # directional (soft sign-agreement) term; 0 = off. ACF² is sign-blind
                                 # (measured ρ(r̂,r) ≈ 0) → this supplies the missing sign constraint.
    dir_mode: str = 'add'        # 'add':  CE + λ_acf·ACF² + λ_dir·L_dir  (independent sign pressure)
                                 # 'mult': CE + λ_acf·ACF²·(1 + λ_dir·L_dir)  (ACF credit gated by sign)
    acf_loss_agg: bool = False   # aggregate-targeting ACF loss: match batch-MEAN pred/true ACF (cancels
                                 # per-window noise) instead of per-window match. Default False = current.
    adaptive_lambda: bool = False  # gradient-norm auto-λ (removes per-dataset λ tuning)
    val_select_agg: bool = False # select best_arval_model by the AGGREGATE val ACF gap (lower-variance)
    # Both AR selection criteria are ACF gaps, i.e. bounded scale-invariant correlations, so neither
    # can penalise a validation rollout that diverges. With this on, an epoch whose val rollouts
    # produce a non-positive close is not eligible to be saved as a checkpoint. It changes the
    # method, so it must be declared in advance and applied to EVERY arm, never to one of them.
    val_stability_gate: bool = False
                                 # instead of per-window acf2_gap_ar. Default False = current behaviour.
    val_ar_max_batches: int = 100000  # #batches for the AR val metric — default now covers the FULL val
                                 # set (selection must be on full validation, not a subset).
    k_train: int = 3             # lags 1-3: strongest signal; lags 4+ too noisy at H=32
    tau_lag: float = 2.0         # lag-decay weight w_k = exp(-k / tau_lag)
    gumbel_tau_start: float = 0.5
    gumbel_tau_end: float = 0.1  # anneal toward hard during training
    p_max: float = 0.5           # max scheduled-sampling self-rollout fraction
    warmup_frac: float = 0.2     # fraction of total steps to ramp p_self 0 → p_max

    # ── A3: adaptive λ (gradient-norm balancing, Wang et al. 2021) ──────────────
    alpha_grad: float = 0.07         # target ACF² gradient fraction (7% = A2 at init)
    grad_norm_interval: int = 10     # steps between proxy gradient-norm updates
    ema_lambda_beta: float = 0.98    # EMA smoothing factor for λ

    # ── early stopping ────────────────────────────────────────────────────────
    rankic_baseline: float = 0.049   # phase 1 A0 baseline
    rankic_drop_tol: float = 0.10    # stop if RankIC drops >10% below baseline
    parkinson_r2_floor: float = 0.25 # warn if Parkinson R² drops below this

    # ── model ─────────────────────────────────────────────────────────────────
    tokenizer_path: str = 'NeoQuasar/Kronos-Tokenizer-base'
    predictor_path: str = 'NeoQuasar/Kronos-base'

    # ── output ────────────────────────────────────────────────────────────────
    use_wandb: bool = False      # opt-in W&B logging of the training curves (rank 0)
    save_dir: str = field(default_factory=lambda: f'{_SCRATCH}/outputs/phase2')
    run_name: str = 'A1_full'
    log_interval: int = 50


# Preset configs for the two core arms. Pass arm_preset(arm) to get the right config.
def arm_preset(arm: str = None, **overrides) -> Phase2Config:
    """Build the config for `arm`. Called with no argument it returns the
    sorted list of valid arm names instead -- that is what the CLI uses for
    its --arm choices, so the two can no longer drift apart."""
    presets = {
        'A1': dict(lr=1e-4, arm='A1', run_name='A1_full', data_fraction=1.0),
        'A2': dict(lr=1e-4, arm='A2', run_name='A2_full', data_fraction=1.0),
    }
    if arm is None:
        return sorted(presets)
    if arm not in presets:
        raise ValueError(f"Unknown arm '{arm}'. Choose from {list(presets)}")
    kwargs = {**presets[arm], **overrides}
    return Phase2Config(**kwargs)

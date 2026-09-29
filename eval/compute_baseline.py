#!/usr/bin/env python3
"""
Cross-sectional baseline computation for Kronos on top-N Binance USDT pairs.

Reproduces (as closely as possible) the paper's metrics from Tables 14-18 by:
  - Running inference across N symbols in parallel (one GPU per symbol)
  - Aligning windows by timestamp across symbols
  - Computing IC/RankIC cross-sectionally at each timestamp, then mean over time
  - Computing vol MAE with paper's Eq. 12 (σ², not σ)
  - Computing ACF² gap on forecast window only (K=20 lags)

Performs TWO inference runs per task family (paper Table 6):
  - "forecast"  : T=0.6  → IC/RankIC for price & return; ACF² gap
  - "volatility": T=0.9  → vol MAE

Per-symbol predictions are cached to disk; rerunning with --cache-only skips
inference and just recomputes metrics. This is essential for ablations.

Usage:
    # Full run (download data first, then run)
    python data/download_binance.py --config configs/crypto_top20.yaml
    torchrun --standalone --nproc_per_node=8 eval/compute_baseline.py \\
        --config configs/crypto_top20.yaml

    # Recompute metrics from cached predictions
    python eval/compute_baseline.py --config configs/crypto_top20.yaml --cache-only
"""

from __future__ import annotations
import argparse
import datetime
import getpass
import json
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import yaml

# Add project root to path so we can import Kronos `model` package
KRONOS_ROOT = Path(__file__).resolve().parent.parent
if str(KRONOS_ROOT) not in sys.path:
    sys.path.insert(0, str(KRONOS_ROOT))

from eval.inference import (
    InferConfig, infer_symbol, FEATURES, CLOSE_IDX,
)
from eval.metrics import (
    acf_sq_returns, realized_variance, end_of_window_simple_return,
    cross_sectional_ic, cross_sectional_price_ic,
    aggregate_vol_mae, aggregate_acf2_gap,
)


def _lookback_acf_targets(sym, common_ts, cfg_yaml, K):
    """Self-estimated target: ACF(r²) of the lookback window before each test window.
    Maps each aligned row's timestamp (context_end) → CSV index → close[idx-lookback:idx].
    Returns dict ts(int64) -> target ACF (K,), or None if CSV missing."""
    data_dir = Path(cfg_yaml['data_dir']); freq = cfg_yaml.get('frequency', '15m')
    lookback = cfg_yaml['lookback']
    csv = data_dir / f'{sym.lower()}_{freq}.csv'
    if not csv.exists():
        return None
    df = pd.read_csv(csv)
    ts = pd.to_datetime(df['timestamps']).astype('int64').to_numpy()
    close = df['close'].to_numpy()
    idx_of = {int(t): i for i, t in enumerate(ts)}
    out = {}
    for t in common_ts:
        i = idx_of.get(int(t))
        if i is None or i < lookback:
            continue
        out[int(t)] = acf_sq_returns(close[i - lookback:i], K)
    return out


def _apply_test_time_control(aligned: dict, control: str, K: int,
                             common_ts=None, cfg_yaml=None, tau: float = 2.0,
                             k_cost: int = None):
    """Test-time control: replace the uniform-over-rollouts aggregation with a soft
    ACF-weighted reweighting (weighted-over-N). `control`='none' (no-op),
    'soft_zero_b{beta}' (target rho*=0 — shrink toward no-clustering), or
    'soft_ctx_b{beta}' (target = ACF of the LOOKBACK window — self-estimated, a realizable
    proxy for the true future ACF since clustering persists). Sets per-symbol `pred_paths`
    = weighted path and `ctrl_acf` = weighted ensemble ACF (pipeline scale).

    `tau` is the lag-decay of the cost weights (w_k = exp(-k/tau)); `k_cost` is the number of
    near-lags the cost sums over. `k_cost` is DECOUPLED from the eval horizon `K`: the ensemble
    ACF `ctrl_acf` is always K-dim (so acf2_gap is still measured at K lags), while the rollout
    weights `pi` are driven only by the first `k_cost` lags. Defaults `k_cost=K, tau=2` reproduce
    the original single-K behaviour; the tuned operating point is `k_cost=5, tau=4` (see
    June26_inference.md §'Tuning the control')."""
    if not control or control == 'none':
        return
    k_cost = K if k_cost is None else int(k_cost)
    if not 1 <= k_cost <= K:
        raise ValueError(f"k_cost={k_cost} must be in [1, K={K}]")
    # mode: target kind {zero, ctx, oracle}; hard=True → best-of-N (argmin), else soft(beta)
    hard = False
    if control.startswith('soft_zero_b'):
        mode, beta = 'zero', float(control.split('_b')[1])
    elif control.startswith('soft_ctx_b'):
        mode, beta = 'ctx', float(control.split('_b')[1])
    elif control == 'bestn_zero':        # realizable best-of-N toward rho*=0
        mode, beta, hard = 'zero', None, True
    elif control == 'oracle':            # best-of-N toward the TRUE future ACF (unrealizable ceiling)
        mode, beta, hard = 'oracle', None, True
    else:
        raise ValueError(f"unknown control '{control}'")
    w = np.exp(-np.arange(1, k_cost + 1) / tau)      # cost weights over the first k_cost lags
    for sym, a in aligned.items():
        roll = a['pred_per_rollout']                 # (n_t, N, H, D)
        n_t, N = roll.shape[0], roll.shape[1]
        ctx_tgt = _lookback_acf_targets(sym, common_ts, cfg_yaml, K) if mode == 'ctx' else None
        ctrl_path = np.full_like(a['pred_paths'], np.nan)
        ctrl_acf = np.full((n_t, K), np.nan, dtype=np.float64)
        for t in range(n_t):
            if not np.isfinite(roll[t]).all():
                continue
            if mode == 'ctx':
                target = ctx_tgt.get(int(common_ts[t])) if ctx_tgt else None
                if target is None:
                    continue   # leave NaN → falls back to baseline aggregation downstream
            elif mode == 'oracle':
                target = acf_sq_returns(a['true_paths'][t, :, CLOSE_IDX], K)
            else:
                target = np.zeros(K)
            racf = np.stack([acf_sq_returns(roll[t, n, :, CLOSE_IDX], K) for n in range(N)])
            # cost over near-lags only; ensemble ACF (below) stays full K-dim for the metric
            C = (w * (racf[:, :k_cost] - target[:k_cost]) ** 2).sum(-1)
            if hard:
                pi = np.zeros(N); pi[int(np.argmin(C))] = 1.0   # best-of-N: pick single rollout
            else:
                pi = np.exp(-beta * (C - C.min())); pi /= pi.sum()
            ctrl_path[t] = (pi[:, None, None] * roll[t]).sum(0)
            ctrl_acf[t] = (pi[:, None] * racf).sum(0)
        # for ctx rows that fell back (NaN), keep the uniform mean path + mean-of-acf
        if mode == 'ctx':
            for t in range(n_t):
                if np.isnan(ctrl_path[t]).all() and np.isfinite(roll[t]).all():
                    ctrl_path[t] = roll[t].mean(0)
                    ctrl_acf[t] = np.stack([acf_sq_returns(roll[t, n, :, CLOSE_IDX], K)
                                            for n in range(N)]).mean(0)
        a['pred_paths'] = ctrl_path
        a['ctrl_acf'] = ctrl_acf


# ─────────────────────────────────────────────────────────────────────────────
# Distributed setup helpers
# ─────────────────────────────────────────────────────────────────────────────
def get_rank_world():
    """Get distributed rank and world size; falls back to single-process."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    rank = int(os.environ.get('LOCAL_RANK', os.environ.get('RANK', '0')))
    world = int(os.environ.get('WORLD_SIZE', '1'))
    return rank, world


def setup_distributed():
    """Initialise process group if running under torchrun."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        if not dist.is_initialized():
            # Default 10-min collective timeout is shorter than a single symbol's
            # inference time at large lookback (W320 ~20 min/symbol) → ranks that
            # finish their symbol list slightly later than others get killed while
            # still computing, not actually stuck. Give enough slack for the
            # slowest plausible single symbol plus a margin.
            dist.init_process_group(backend='nccl', timeout=datetime.timedelta(minutes=60))
        rank = dist.get_rank()
        torch.cuda.set_device(rank % torch.cuda.device_count())
        return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Predictions cache (per symbol × run × rank)
# ─────────────────────────────────────────────────────────────────────────────
def get_predictions_dir(cfg_yaml: dict) -> Path:
    """
    Return the root directory for cached pkl predictions.

    Config key `predictions_cache_dir` (optional): path where large pkl files
    are stored. Supports `{USER}` placeholder → expanded to getpass.getuser().
    Falls back to `{output_dir}/predictions` for backward compatibility.
    """
    raw = cfg_yaml.get('predictions_cache_dir', '')
    if raw:
        resolved = raw.replace('{USER}', getpass.getuser())
        p = Path(resolved)
    else:
        p = Path(cfg_yaml['output_dir']) / 'predictions'
    p.mkdir(parents=True, exist_ok=True)
    return p


def cache_path(pred_dir: Path, run_name: str, symbol: str) -> Path:
    p = pred_dir / run_name
    p.mkdir(parents=True, exist_ok=True)
    return p / f'{symbol}.pkl'


def save_predictions(out_path: Path, data: Dict):
    """Save predictions in a forward-compatible pickle format."""
    # Convert datetime64 to int64 ns for portability
    if 'timestamps' in data and data['timestamps'].dtype.kind == 'M':
        data = {**data, 'timestamps': data['timestamps'].astype('int64')}
    with open(out_path, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_predictions(in_path: Path) -> Dict:
    with open(in_path, 'rb') as f:
        data = pickle.load(f)
    if 'timestamps' in data and data['timestamps'].dtype == np.int64:
        data['timestamps'] = data['timestamps'].astype('datetime64[ns]')
    return data


# ─────────────────────────────────────────────────────────────────────────────
# Inference phase (distributed across GPUs, one symbol per rank-cycle)
# ─────────────────────────────────────────────────────────────────────────────
def run_inference_for_symbols(
    cfg_yaml: dict,
    symbols: List[str],
    run_name: str,
    temperature: float,
    n_rollouts: int,
):
    """
    Each rank takes a subset of symbols. With torchrun --nproc_per_node=8 and
    20 symbols, ranks 0-3 get 3 symbols each and ranks 4-7 get 2 each.
    """
    backbone = cfg_yaml.get('backbone', 'kronos')

    rank, world = get_rank_world()
    device = torch.device(f'cuda:{rank % torch.cuda.device_count()}')

    # Load models on this rank's GPU
    if rank == 0:
        print(f"[rank 0] loading {cfg_yaml['model_id']} ({backbone}) on {device}")

    if backbone == 'chronos':
        from chronos import BaseChronosPipeline
        # float32: T5 is known to NaN under fp16, and V100 (Volta) lacks bf16
        # tensor-core support; chronos-t5-small is tiny so the cost is negligible.
        pipeline = BaseChronosPipeline.from_pretrained(
            cfg_yaml['model_id'], device_map=str(device), torch_dtype=torch.float32)
        checkpoint = cfg_yaml.get('checkpoint')
        if checkpoint:
            # Phase-2 Chronos fine-tunes save a raw T5 state_dict (train_predictor_chronos.py
            # torch.save(t5.state_dict(), ...)), not a from_pretrained-loadable directory like
            # Kronos checkpoints, so it's loaded onto the base architecture instead of replacing
            # model_id.
            sd = torch.load(checkpoint, map_location=device)
            pipeline.model.model.load_state_dict(sd, strict=True)  # raises on any key mismatch
            if rank == 0:
                print(f"[rank 0] loaded fine-tuned checkpoint {checkpoint}")
        pipeline.model.model.eval()
    else:
        from model import Kronos, KronosTokenizer
        from model.kronos import sample_from_logits
        tokenizer = KronosTokenizer.from_pretrained(cfg_yaml['tokenizer_id']).to(device).eval()
        model = Kronos.from_pretrained(cfg_yaml['model_id']).to(device).eval()

    # Assign symbols to this rank (round-robin)
    my_symbols = [s for i, s in enumerate(symbols) if i % world == rank]
    print(f"[rank {rank}] device={device}  symbols={my_symbols}")

    pred_dir = get_predictions_dir(cfg_yaml)
    data_dir = Path(cfg_yaml['data_dir'])
    test_start, test_end = cfg_yaml['test_period']

    icfg = InferConfig(
        lookback=cfg_yaml['lookback'],
        pred_len=cfg_yaml['pred_len'],
        stride=cfg_yaml['stride'],
        n_rollouts=n_rollouts,
        max_context=cfg_yaml['max_context'],
        clip=cfg_yaml['clip'],
        top_p=cfg_yaml['top_p'],
        top_k=cfg_yaml['top_k'],
        temperature=temperature,
        batch_size=cfg_yaml['batch_size_per_gpu'],
        zero_vol_amount=cfg_yaml.get('zero_vol_amount', False),
    )

    for sym in my_symbols:
        cache_file = cache_path(pred_dir, run_name, sym)
        if cache_file.exists():
            print(f"[rank {rank}] {sym}: cached → skip")
            continue

        csv_path = data_dir / f'{sym.lower()}_{cfg_yaml["frequency"]}.csv'
        if not csv_path.exists():
            print(f"[rank {rank}] {sym}: missing data file {csv_path} — skipping")
            continue

        t0 = time.time()
        if backbone == 'chronos':
            from eval.inference_chronos import infer_symbol_chronos
            result = infer_symbol_chronos(
                symbol=sym,
                csv_path=csv_path,
                cfg=icfg,
                test_start=test_start, test_end=test_end,
                pipeline=pipeline,
                device=device,
            )
        else:
            result = infer_symbol(
                symbol=sym,
                csv_path=csv_path,
                cfg=icfg,
                test_start=test_start, test_end=test_end,
                tokenizer=tokenizer, model=model, sample_fn=sample_from_logits,
                device=device, use_fp16=cfg_yaml.get('fp16', True),
            )
        save_predictions(cache_file, result)
        print(f"[rank {rank}] {sym}: saved {cache_file.name} "
              f"({result['n_windows']} windows, {(time.time()-t0)/60:.1f} min)")

    # Barrier so metric aggregation only happens after ALL ranks done
    if dist.is_initialized():
        dist.barrier()


# ─────────────────────────────────────────────────────────────────────────────
# Metric aggregation (single-process; reads all cached predictions)
# ─────────────────────────────────────────────────────────────────────────────
def aggregate_metrics(cfg_yaml: dict) -> Dict:
    """
    Read all cached predictions and compute paper-aligned metrics.

    Two runs:
      forecast  (T=0.6) → return IC/RankIC, price IC/RankIC, ACF² gap
      volatility (T=0.9) → vol MAE, vol R²
    """
    pred_dir = get_predictions_dir(cfg_yaml)
    symbols = cfg_yaml['symbols']
    K = cfg_yaml['acf_lags']

    final = {
        'config': {
            'model': cfg_yaml['model_id'],
            'tokenizer': cfg_yaml.get('tokenizer_id', cfg_yaml['model_id']),
            'checkpoint': cfg_yaml.get('checkpoint'),
            'symbols': symbols,
            'test_period': cfg_yaml['test_period'],
            'lookback': cfg_yaml['lookback'],
            'pred_len': cfg_yaml['pred_len'],
            'n_rollouts': cfg_yaml['n_rollouts'],
            'acf_lags': K,
        }
    }

    for run in cfg_yaml['runs']:
        run_name = run['name']
        T = run['temperature']
        wanted = set(run['metrics'])
        print(f"\n=== Aggregating run '{run_name}' (T={T}, metrics={wanted}) ===")

        # Load all symbol predictions for this run
        per_sym: Dict[str, dict] = {}
        for sym in symbols:
            cf = cache_path(pred_dir, run_name, sym)
            if not cf.exists():
                print(f"  WARN: {sym} cache missing ({cf})")
                continue
            per_sym[sym] = load_predictions(cf)
            if per_sym[sym]['n_windows'] == 0:
                print(f"  WARN: {sym} has 0 windows — dropping")
                del per_sym[sym]

        if not per_sym:
            print(f"  No predictions found for run '{run_name}' — skipping")
            continue

        # Align windows by timestamp using the reference symbol's timeline.
        # Use the symbol with the most windows as the reference grid; all other
        # symbols participate only at timestamps where they have data.
        # Symbols delisted mid-period (e.g. MATICUSDT → POL Sep 2024) contribute
        # NaN for missing timestamps — cross_sectional_ic already masks NaN.
        ref_sym = max(per_sym, key=lambda s: per_sym[s]['n_windows'])
        ref_ts = per_sym[ref_sym]['timestamps'].astype('int64')
        common = sorted(ref_ts.tolist())
        n_common = len(common)
        n_partial = sum(1 for s in per_sym
                        if per_sym[s]['n_windows'] < n_common)
        print(f"  {len(per_sym)} symbols, {n_common} timestamps "
              f"(ref={ref_sym}, {n_partial} partial symbols)")

        # Build aligned dicts: NaN-fill missing windows for partial symbols
        aligned: Dict[str, Dict] = {}
        for sym, d in per_sym.items():
            ts_int = d['timestamps'].astype('int64')
            ts_to_idx = {int(t): i for i, t in enumerate(ts_int.tolist())}
            H, D = d['pred_paths'].shape[1], d['pred_paths'].shape[2]
            N = d['pred_paths_per_rollout'].shape[1]

            pred_out  = np.full((n_common, H, D), np.nan, dtype=np.float32)
            roll_out  = np.full((n_common, N, H, D), np.nan, dtype=np.float32)
            true_out  = np.full((n_common, H, D), np.nan, dtype=np.float32)
            anch_out  = np.full(n_common, np.nan, dtype=np.float32)

            for ti, t in enumerate(common):
                if t in ts_to_idx:
                    i = ts_to_idx[t]
                    pred_out[ti]  = d['pred_paths'][i]
                    roll_out[ti]  = d['pred_paths_per_rollout'][i]
                    true_out[ti]  = d['true_paths'][i]
                    anch_out[ti]  = d['anchor_close'][i]

            aligned[sym] = {
                'pred_paths':       pred_out,
                'pred_per_rollout': roll_out,
                'true_paths':       true_out,
                'anchor_close':     anch_out,
            }

        # test-time control (no-op unless cfg_yaml['control'] set, forecast run only)
        if run_name == 'forecast':
            _apply_test_time_control(aligned, cfg_yaml.get('control', 'none'), K,
                                     common_ts=common, cfg_yaml=cfg_yaml,
                                     tau=cfg_yaml.get('control_tau', 2.0),
                                     k_cost=cfg_yaml.get('control_kcost', None))

        run_results = {'n_symbols': len(aligned), 'n_common_timestamps': n_common,
                       'temperature': T, 'control': cfg_yaml.get('control', 'none'),
                       'control_tau': cfg_yaml.get('control_tau', 2.0),
                       'control_kcost': cfg_yaml.get('control_kcost', K)}

        # ── Return IC/RankIC ─────────────────────────────────────────────
        if 'return_ic' in wanted:
            pred_ret = {}
            true_ret = {}
            for sym, a in aligned.items():
                pred_close = a['pred_paths'][:, :, CLOSE_IDX]      # (n_t, H)
                true_close = a['true_paths'][:, :, CLOSE_IDX]
                anchors = a['anchor_close']
                pred_ret[sym] = np.array([
                    end_of_window_simple_return(pred_close[t], anchors[t])
                    for t in range(n_common)
                ])
                true_ret[sym] = np.array([
                    end_of_window_simple_return(true_close[t], anchors[t])
                    for t in range(n_common)
                ])
            ic = cross_sectional_ic(pred_ret, true_ret, rank=False)
            ric = cross_sectional_ic(pred_ret, true_ret, rank=True)
            run_results['return_IC'] = ic
            run_results['return_RankIC'] = ric
            print(f"  return  IC={ic['ic']:+.4f}  tstat={ic['ic_tstat']:.2f}  n_t={ic['n_t']}")
            print(f"  return RankIC={ric['ic']:+.4f}  tstat={ric['ic_tstat']:.2f}")

        # ── Price Series IC/RankIC (paper Table 14) ───────────────────────
        if 'price_ic' in wanted:
            pred_paths = {s: a['pred_paths'] for s, a in aligned.items()}
            true_paths = {s: a['true_paths'] for s, a in aligned.items()}
            pic = cross_sectional_price_ic(pred_paths, true_paths, rank=False)
            pric = cross_sectional_price_ic(pred_paths, true_paths, rank=True)
            run_results['price_IC'] = pic
            run_results['price_RankIC'] = pric
            print(f"  price   IC={pic['ic']:+.4f}  tstat={pic['ic_tstat']:.2f}  "
                  f"n_windows={pic['n_windows']}")
            print(f"  price RankIC={pric['ic']:+.4f}  tstat={pric['ic_tstat']:.2f}")

        # ── ACF² gap (forecast window only) ─────────────────────────────
        if 'acf2_gap' in wanted:
            pred_acf = {}
            true_acf = {}
            for sym, a in aligned.items():
                # Use per-rollout paths, not the mean — averaging paths before
                # ACF kills the autocorrelation structure in squared returns.
                # Compute ACF per rollout then average, consistent with vol MAE.
                # With test-time control, use the weighted ensemble ACF instead.
                true_close = a['true_paths'][:, :, CLOSE_IDX]
                if 'ctrl_acf' in a:
                    pred_acf[sym] = a['ctrl_acf']
                else:
                    rollouts = a['pred_per_rollout'][:, :, :, CLOSE_IDX]  # (n_t, N, H)
                    N_r = rollouts.shape[1]
                    pred_acf[sym] = np.stack([
                        np.mean([acf_sq_returns(rollouts[t, n], K)
                                 for n in range(N_r)], axis=0)
                        for t in range(n_common)
                    ])
                true_acf[sym] = np.stack([
                    acf_sq_returns(true_close[t], K) for t in range(n_common)
                ])
            acf_res = aggregate_acf2_gap(pred_acf, true_acf)
            run_results['acf2'] = acf_res
            print(f"  ACF² gap = {acf_res['acf2_gap']:.6e} "
                  f"± {acf_res['acf2_gap_std']:.6e}  n={acf_res['n']}")

        # ── Volatility MAE (paper Eq. 12 σ², not σ) ──────────────────────
        if 'vol_mae' in wanted:
            pred_rv = {}
            true_rv = {}
            for sym, a in aligned.items():
                # Per paper, vol is computed on PREDICTED close path only.
                # Average σ² across rollouts to get a single prediction per window.
                rollouts = a['pred_per_rollout'][:, :, :, CLOSE_IDX]  # (n_t, N, H)
                pred_rv_per_rollout = np.array([
                    [realized_variance(rollouts[t, n])
                     for n in range(rollouts.shape[1])]
                    for t in range(n_common)
                ])
                pred_rv[sym] = pred_rv_per_rollout.mean(axis=1)        # (n_t,)
                true_close = a['true_paths'][:, :, CLOSE_IDX]
                true_rv[sym] = np.array([
                    realized_variance(true_close[t]) for t in range(n_common)
                ])
            vol = aggregate_vol_mae(pred_rv, true_rv)
            run_results['vol'] = vol
            print(f"  vol MAE = {vol['vol_mae']:.6e}  R² = {vol['vol_r2']:.4f}  n={vol['n']}")

        final[run_name] = run_results

    return final


# ─────────────────────────────────────────────────────────────────────────────
# Entrypoint
# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--cache-only', action='store_true',
                    help='Skip inference; recompute metrics from cached predictions only')
    ap.add_argument('--run-tag', default='',
                    help='Optional tag appended to results filename, e.g. 20250528. '
                         'Does not affect the predictions cache path.')
    ap.add_argument('--model-id', default=None,
                    help='Override model_id from config (accepts local path or HF hub id).')
    ap.add_argument('--checkpoint', default=None,
                    help='Chronos backbone only: path to a fine-tuned T5 state_dict .pt file '
                         '(train_predictor_chronos.py output). Loaded onto the base architecture '
                         "named by model_id/config; leave model_id at the base HF id (e.g. "
                         'amazon/chronos-t5-small), do not point --model-id at the checkpoint.')
    ap.add_argument('--predictions-cache-dir', default=None,
                    help='Override predictions_cache_dir from config. Use separate dirs '
                         'per model to avoid overwriting cached predictions.')
    ap.add_argument('--n-rollouts', type=int, default=None,
                    help='Override n_rollouts for the FORECAST run (test-time scaling of N). '
                         'Use a separate cache dir to avoid clobbering the N=10 cache.')
    ap.add_argument('--control', default=None,
                    help="Test-time control over cached rollouts: 'none' (default) or "
                         "'soft_zero_b{beta}' e.g. soft_zero_b20. Aggregation-only (cache-reusable).")
    ap.add_argument('--control-tau', type=float, default=None,
                    help="Lag-decay of the control cost weights w_k=exp(-k/tau). Default 2.0; "
                         "tuned operating point is 4.0 (see June26_inference.md).")
    ap.add_argument('--control-kcost', type=int, default=None,
                    help="Number of near-lags the control cost sums over (decoupled from acf_lags, "
                         "the eval horizon). Default = acf_lags; tuned operating point is 5.")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg_yaml = yaml.safe_load(f)
    cfg_yaml = {k: os.path.expandvars(v) if isinstance(v, str) else v for k, v in cfg_yaml.items()}
    if args.model_id:
        cfg_yaml['model_id'] = args.model_id
    if args.checkpoint:
        cfg_yaml['checkpoint'] = args.checkpoint
    if args.predictions_cache_dir:
        cfg_yaml['predictions_cache_dir'] = args.predictions_cache_dir
    if args.n_rollouts:
        cfg_yaml['n_rollouts'] = args.n_rollouts
        for run in cfg_yaml.get('runs', []):
            if run['name'] == 'forecast':
                run['n_rollouts'] = args.n_rollouts
    if args.control:
        cfg_yaml['control'] = args.control
    if args.control_tau is not None:
        cfg_yaml['control_tau'] = args.control_tau
    if args.control_kcost is not None:
        cfg_yaml['control_kcost'] = args.control_kcost

    is_distributed = setup_distributed()
    rank, world = get_rank_world()
    print(f"[rank {rank}/{world}] config={args.config} "
          f"cache_only={args.cache_only} run_tag={args.run_tag or '(none)'}")

    out_dir = Path(cfg_yaml['output_dir'])
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_dir = get_predictions_dir(cfg_yaml)
    if rank == 0:
        print(f"Predictions cache: {pred_dir}")

    # ── Phase 1: inference (each run is a separate inference pass) ────────
    if not args.cache_only:
        for run in cfg_yaml['runs']:
            if rank == 0:
                print(f"\n{'='*70}\nInference run: {run['name']} (T={run['temperature']})\n{'='*70}")
            run_inference_for_symbols(
                cfg_yaml=cfg_yaml,
                symbols=cfg_yaml['symbols'],
                run_name=run['name'],
                temperature=run['temperature'],
                n_rollouts=run.get('n_rollouts', cfg_yaml['n_rollouts']),
            )

    # ── Phase 2: metric aggregation (rank 0 only) ────────────────────────
    if rank == 0:
        results = aggregate_metrics(cfg_yaml)
        base = cfg_yaml['results_file']
        if args.run_tag:
            stem, ext = base.rsplit('.', 1)
            base = f'{stem}_{args.run_tag}.{ext}'
        out_file = out_dir / base
        with open(out_file, 'w') as f:
            # Pickle/JSON-friendly: convert any nan to None? leave as 'NaN' string?
            # Custom encoder handles np types and NaN.
            json.dump(results, f, indent=2, default=_json_default)
        print(f"\nResults → {out_file}")

    if is_distributed:
        dist.barrier()
        dist.destroy_process_group()


def _json_default(obj):
    """JSON encoder helper for numpy types and NaN."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Unserializable {type(obj)}")


if __name__ == '__main__':
    main()
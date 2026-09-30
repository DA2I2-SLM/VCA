"""
Single-symbol Chronos (T5-based) inference utilities, mirroring eval/inference.py's
infer_symbol() so eval/compute_baseline.py's caching + aggregate_metrics need no changes.

Chronos is univariate: it only forecasts the `close` channel. All other FEATURES
channels (open/high/low/volume/amount) are filled with NaN in pred_paths /
pred_paths_per_rollout so any accidental read of a non-close channel fails loudly
via NaN propagation instead of silently scoring on fabricated data. true_paths keeps
all real channels since ground truth doesn't depend on the model.
"""

from __future__ import annotations
import time
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch

from eval.inference import FEATURES, CLOSE_IDX, InferConfig, build_test_windows


def infer_symbol_chronos(
    symbol: str,
    csv_path: Path,
    cfg: InferConfig,
    test_start: str,
    test_end: str,
    pipeline,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    """
    Run Chronos autoregressive generation on all test windows for one symbol.
    Same return schema as eval.inference.infer_symbol.
    """
    df = pd.read_csv(csv_path)
    df['timestamps'] = pd.to_datetime(df['timestamps'])
    df = df.sort_values('timestamps').reset_index(drop=True)

    vals = df[FEATURES].values.astype(np.float32)
    ts = df['timestamps']
    close = vals[:, CLOSE_IDX]

    windows = build_test_windows(
        ts, test_start, test_end, cfg.lookback, cfg.pred_len, cfg.stride,
    )
    n_t = len(windows)
    if n_t == 0:
        return {
            'symbol': symbol,
            'n_windows': 0,
            'pred_paths': np.empty((0, cfg.pred_len, 6)),
            'pred_paths_per_rollout': np.empty((0, cfg.n_rollouts, cfg.pred_len, 6)),
            'true_paths': np.empty((0, cfg.pred_len, 6)),
            'anchor_close': np.empty((0,)),
            'timestamps': np.empty((0,), dtype='datetime64[ns]'),
        }

    pred_paths = np.full((n_t, cfg.pred_len, 6), np.nan, dtype=np.float32)
    pred_paths_per_rollout = np.full(
        (n_t, cfg.n_rollouts, cfg.pred_len, 6), np.nan, dtype=np.float32)
    true_paths = np.zeros((n_t, cfg.pred_len, 6), dtype=np.float32)
    anchor_close = np.zeros(n_t, dtype=np.float32)
    window_timestamps = np.empty(n_t, dtype='datetime64[ns]')

    t0 = time.time()
    BS = cfg.batch_size
    n_batches = (n_t + BS - 1) // BS

    for b in range(n_batches):
        batch = windows[b * BS:(b + 1) * BS]
        B = len(batch)
        ctx_batch = []

        for j, (cs, ce, fe) in enumerate(batch):
            true_paths[b * BS + j] = vals[ce:fe]
            anchor_close[b * BS + j] = float(close[ce - 1])
            window_timestamps[b * BS + j] = ts.iloc[ce]
            ctx_batch.append(torch.from_numpy(close[cs:ce].copy()).float())

        with torch.no_grad():
            out = pipeline.predict(
                inputs=ctx_batch,
                prediction_length=cfg.pred_len,
                num_samples=cfg.n_rollouts,
                temperature=cfg.temperature,
                top_k=cfg.top_k,
                top_p=cfg.top_p,
            )
        out = out.float().cpu().numpy()

        pred_paths_per_rollout[b * BS:b * BS + B, :, :, CLOSE_IDX] = out
        pred_paths[b * BS:b * BS + B, :, CLOSE_IDX] = out.mean(axis=1)

    elapsed = time.time() - t0
    print(f"  [{symbol}] {n_t} windows in {elapsed/60:.1f} min "
          f"({elapsed/n_t:.2f} s/window)")

    return {
        'symbol': symbol,
        'n_windows': n_t,
        'pred_paths': pred_paths,
        'pred_paths_per_rollout': pred_paths_per_rollout,
        'true_paths': true_paths,
        'anchor_close': anchor_close,
        'timestamps': window_timestamps,
    }

"""
Single-symbol Kronos inference utilities.

Refactored from the user's compute_baseline.py. Key changes vs original:
  - Cleaner separation: prepare_window() / infer_batch() / decode_batch()
  - Returns BOTH raw predicted paths (for metrics) and predicted log-returns
  - Accepts run-specific temperature (paper Table 6: 0.6 for forecasting,
    0.9 for volatility, 1.0 for generation)

Each process owns ONE GPU and processes ONE symbol end-to-end.
"""

from __future__ import annotations
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch


FEATURES = ['open', 'high', 'low', 'close', 'volume', 'amount']
CLOSE_IDX = 3


@dataclass
class InferConfig:
    lookback: int
    pred_len: int
    stride: int
    n_rollouts: int
    max_context: int
    clip: float
    top_p: float
    top_k: int
    temperature: float
    batch_size: int
    zero_vol_amount: bool = False


def build_test_windows(
    timestamps: pd.Series,
    test_start_str: str,
    test_end_str: str,
    lookback: int,
    pred_len: int,
    stride: int,
) -> List[Tuple[int, int, int]]:
    """
    Build strided forecast windows entirely within the test period.

    Each window = (ctx_start, ctx_end, fut_end) such that:
      - context bars  vals[ctx_start : ctx_end]      length = lookback
      - target bars   vals[ctx_end   : fut_end]      length = pred_len

    The forecast START (ctx_end) must lie >= test_start. The forecast END
    (fut_end) must lie <= test_end. This guarantees both context and target
    do not leak from outside the test window if test_start is shifted by
    `lookback` bars (we DO allow context to extend before test_start since
    that's just historical context, not training data).

    Args:
        timestamps   : pd.Series of pd.Timestamp, sorted ascending
        test_start_str, test_end_str : ISO date strings
    """
    test_start = pd.Timestamp(test_start_str)
    test_end = pd.Timestamp(test_end_str)

    ts = pd.to_datetime(timestamps).reset_index(drop=True)
    n = len(ts)

    first_ctx_end = int(ts.searchsorted(test_start, side='left'))
    last_fut_end = int(ts.searchsorted(test_end, side='right'))

    windows = []
    i = first_ctx_end
    while i + pred_len <= last_fut_end and i - lookback >= 0:
        windows.append((i - lookback, i, i + pred_len))
        i += stride
    return windows


def calc_time_stamps(timestamps: pd.Series) -> pd.DataFrame:
    """
    Extract 5 time features: minute, hour, day_of_week, day_of_month, month.

    Matches model/kronos.py calc_time_stamps signature.
    """
    ts = pd.to_datetime(timestamps).reset_index(drop=True)
    return pd.DataFrame({
        'minute': ts.dt.minute.astype(float),
        'hour':   ts.dt.hour.astype(float),
        'dow':    ts.dt.dayofweek.astype(float),
        'dom':    ts.dt.day.astype(float),
        'month':  ts.dt.month.astype(float),
    })


def infer_rollouts(
    tokenizer, model, sample_from_logits_fn,
    x_norm: np.ndarray,
    x_stamp: np.ndarray,
    y_stamp: np.ndarray,
    cfg: InferConfig,
    device: torch.device,
) -> np.ndarray:
    """
    Run autoregressive generation. Returns (B, n_rollouts, L+H, 6) in
    normalized space — caller is responsible for denormalization.
    """
    B = x_norm.shape[0]
    N = cfg.n_rollouts
    L = cfg.lookback
    H = cfg.pred_len
    MAX = cfg.max_context

    x_t = torch.from_numpy(x_norm).float()
    xst_t = torch.from_numpy(x_stamp).float()
    yst_t = torch.from_numpy(y_stamp).float()
    x_t = torch.clip(x_t, -cfg.clip, cfg.clip)

    def rep(t):
        return (t.unsqueeze(1)
                  .repeat(1, N, 1, 1)
                  .reshape(-1, t.size(1), t.size(2))
                  .to(device))

    x_r, xst_r, yst_r = rep(x_t), rep(xst_t), rep(yst_t)

    with torch.no_grad():
        x_tok = tokenizer.encode(x_r, half=True)
        eff_batch = x_tok[0].size(0)
        total_len = L + H
        full_stamp = torch.cat([xst_r, yst_r], dim=1)

        gen_pre = x_tok[0].new_empty(eff_batch, H)
        gen_post = x_tok[1].new_empty(eff_batch, H)

        pre_buf = x_tok[0].new_zeros(eff_batch, MAX)
        post_buf = x_tok[1].new_zeros(eff_batch, MAX)
        buf_len = min(L, MAX)
        if buf_len > 0:
            sidx = max(0, L - MAX)
            pre_buf[:, :buf_len] = x_tok[0][:, sidx:sidx + buf_len]
            post_buf[:, :buf_len] = x_tok[1][:, sidx:sidx + buf_len]

        for i in range(H):
            cur_len = L + i
            win_len = min(cur_len, MAX)
            if cur_len <= MAX:
                inp = [pre_buf[:, :win_len], post_buf[:, :win_len]]
            else:
                inp = [pre_buf, post_buf]

            ctx_s = max(0, cur_len - MAX)
            cur_stamp = full_stamp[:, ctx_s:cur_len, :].contiguous()

            s1_logits, ctx = model.decode_s1(inp[0], inp[1], cur_stamp)
            s1 = sample_from_logits_fn(
                s1_logits[:, -1, :],
                temperature=cfg.temperature,
                top_k=cfg.top_k,
                top_p=cfg.top_p,
                sample_logits=True,
            )
            s2_logits = model.decode_s2(ctx, s1)
            s2 = sample_from_logits_fn(
                s2_logits[:, -1, :],
                temperature=cfg.temperature,
                top_k=cfg.top_k,
                top_p=cfg.top_p,
                sample_logits=True,
            )

            gen_pre[:, i] = s1.squeeze(-1)
            gen_post[:, i] = s2.squeeze(-1)

            if cur_len < MAX:
                pre_buf[:, cur_len] = s1.squeeze(-1)
                post_buf[:, cur_len] = s2.squeeze(-1)
            else:
                pre_buf = torch.roll(pre_buf, -1, dims=1)
                post_buf = torch.roll(post_buf, -1, dims=1)
                pre_buf[:, -1] = s1.squeeze(-1)
                post_buf[:, -1] = s2.squeeze(-1)

        full_pre = torch.cat([x_tok[0], gen_pre], dim=1)
        full_post = torch.cat([x_tok[1], gen_post], dim=1)
        ctx_s = max(0, total_len - MAX)
        inp_final = [
            full_pre[:, ctx_s:total_len].contiguous(),
            full_post[:, ctx_s:total_len].contiguous(),
        ]
        z = tokenizer.decode(inp_final, half=True)
        z = z.cpu().numpy()
        out_len = z.shape[1]
        z = z.reshape(B, N, out_len, z.shape[-1])

    return z


def infer_symbol(
    symbol: str,
    csv_path: Path,
    cfg: InferConfig,
    test_start: str,
    test_end: str,
    tokenizer, model, sample_fn,
    device: torch.device,
    use_fp16: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Run inference on all test windows for one symbol.

    Returns a dict with raw arrays needed by metrics:
      - pred_paths    : (n_t, H, 6)    averaged over rollouts (for path-level metrics)
      - pred_paths_per_rollout : (n_t, N, H, 6)  raw rollouts (for vol/ACF aggregation)
      - true_paths    : (n_t, H, 6)
      - anchor_close  : (n_t,)         last historical close per window
      - timestamps    : (n_t,)         context_end timestamps for cross-section alignment
      - test_start_idx, n_windows : metadata
    """
    df = pd.read_csv(csv_path)
    df['timestamps'] = pd.to_datetime(df['timestamps'])
    df = df.sort_values('timestamps').reset_index(drop=True)

    vals = df[FEATURES].values.astype(np.float32)
    if cfg.zero_vol_amount:
        vals = vals.copy()
        vals[:, 4] = 0.0
        vals[:, 5] = 0.0
    ts = df['timestamps']

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

    pred_paths = np.zeros((n_t, cfg.pred_len, 6), dtype=np.float32)
    pred_paths_per_rollout = np.zeros((n_t, cfg.n_rollouts, cfg.pred_len, 6),
                                       dtype=np.float32)
    true_paths = np.zeros((n_t, cfg.pred_len, 6), dtype=np.float32)
    anchor_close = np.zeros(n_t, dtype=np.float32)
    window_timestamps = np.empty(n_t, dtype='datetime64[ns]')

    t0 = time.time()
    BS = cfg.batch_size

    autocast_ctx = (
        torch.amp.autocast(device_type='cuda', dtype=torch.float16)
        if use_fp16 and device.type == 'cuda'
        else torch.amp.autocast(device_type='cpu', enabled=False)
    )

    n_batches = (n_t + BS - 1) // BS
    for b in range(n_batches):
        batch = windows[b * BS:(b + 1) * BS]
        B = len(batch)

        x_norm_b = np.zeros((B, cfg.lookback, 6), dtype=np.float32)
        x_stamp_b = np.zeros((B, cfg.lookback, 5), dtype=np.float32)
        y_stamp_b = np.zeros((B, cfg.pred_len, 5), dtype=np.float32)
        x_means = np.zeros((B, 6), dtype=np.float32)
        x_stds = np.zeros((B, 6), dtype=np.float32)

        for j, (cs, ce, fe) in enumerate(batch):
            x_raw = vals[cs:ce]
            y_raw = vals[ce:fe]

            mu = x_raw.mean(axis=0)
            sigma = x_raw.std(axis=0)
            x_norm = np.clip((x_raw - mu) / (sigma + 1e-5), -cfg.clip, cfg.clip)

            x_means[j] = mu
            x_stds[j] = sigma
            x_norm_b[j] = x_norm

            true_paths[b * BS + j] = y_raw
            anchor_close[b * BS + j] = float(vals[ce - 1, CLOSE_IDX])
            window_timestamps[b * BS + j] = ts.iloc[ce]

            ts_x = calc_time_stamps(ts.iloc[cs:ce])
            ts_y = calc_time_stamps(ts.iloc[ce:fe])
            x_stamp_b[j] = ts_x.values.astype(np.float32)
            y_stamp_b[j] = ts_y.values.astype(np.float32)

        with autocast_ctx:
            z_norm = infer_rollouts(
                tokenizer, model, sample_fn,
                x_norm_b, x_stamp_b, y_stamp_b, cfg, device,
            )

        mu4 = x_means[:, None, None, :]
        sig4 = x_stds[:, None, None, :]
        z_den = z_norm * (sig4 + 1e-5) + mu4

        pred_future_rollouts = z_den[:, :, cfg.lookback:, :]
        pred_future_mean = pred_future_rollouts.mean(axis=1)

        pred_paths[b * BS:b * BS + B] = pred_future_mean
        pred_paths_per_rollout[b * BS:b * BS + B] = pred_future_rollouts

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
"""
Crypto strided-window dataset for Phase 2 fine-tuning.

Mirrors the window construction in eval/inference.py but for train/val splits.
Each item is one (lookback + pred_len) window from a single symbol CSV.

data_fraction: take the chronologically first fraction * N windows per symbol
(used for data-efficiency arms A5/A6/A7).
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

import sys
sys.path.append(str(Path(__file__).parent.parent))
from eval.metrics import acf_sq_returns

FEATURES = ['open', 'high', 'low', 'close', 'volume', 'amount']
CLOSE_IDX = 3


def _calc_time_stamps(ts: pd.Series) -> np.ndarray:
    t = pd.to_datetime(ts).reset_index(drop=True)
    return np.stack([
        t.dt.minute.values,
        t.dt.hour.values,
        t.dt.dayofweek.values,
        t.dt.day.values,
        t.dt.month.values,
    ], axis=1).astype(np.float32)


def _build_windows(
    ts: pd.Series,
    period_start: str,
    period_end: str,
    lookback: int,
    pred_len: int,
    stride: int,
) -> List[Tuple[int, int, int]]:
    """Return list of (ctx_start, ctx_end, fut_end) indices."""
    start = pd.Timestamp(period_start)
    end = pd.Timestamp(period_end)
    t = pd.to_datetime(ts).reset_index(drop=True)
    n = len(t)

    first_ctx_end = max(int(t.searchsorted(start, side='left')), lookback)
    last_fut_end = int(t.searchsorted(end, side='right'))

    windows = []
    i = first_ctx_end
    while i + pred_len <= last_fut_end and i - lookback >= 0:
        windows.append((i - lookback, i, i + pred_len))
        i += stride
    return windows


class CryptoWindowDataset(Dataset):
    """
    Strided window dataset over crypto CSVs.

    Each item:
        x_norm   : (lookback, 6)  normalized context
        y_raw    : (pred_len, 6)  raw future prices (for ACF² target)
        x_stamp  : (lookback, 5) time features for context
        y_stamp  : (pred_len, 5) time features for prediction window
        x_mean   : (6,)          per-feature mean of context window
        x_std    : (6,)          per-feature std of context window

    x_norm is clipped to [-clip, clip]. zero_vol_amount zeroes channels 4-5.
    """

    def __init__(
        self,
        data_dir: str,
        symbols: List[str],
        period_start: str,
        period_end: str,
        lookback: int,
        pred_len: int,
        stride: int,
        clip: float = 5.0,
        zero_vol_amount: bool = True,
        data_fraction: float = 1.0,
        frequency: str = '15m',
        data_select: str = None,
        acf_select_lags: int = 3,
        data_select_seed: int = 0,
    ):
        self.lookback = lookback
        self.pred_len = pred_len
        self.clip = clip
        self.zero_vol_amount = zero_vol_amount

        self.windows: List[Tuple[np.ndarray, np.ndarray, int, int]] = []
        _scored: List[Tuple[float, np.ndarray, np.ndarray, int, int]] = []

        data_dir = Path(data_dir)
        for sym in symbols:
            csv_path = data_dir / f"{sym.lower()}_{frequency}.csv"
            if not csv_path.exists():
                print(f"[dataset] skip {sym}: {csv_path} not found")
                continue

            df = pd.read_csv(csv_path)
            df['timestamps'] = pd.to_datetime(df['timestamps'])
            df = df.sort_values('timestamps').reset_index(drop=True)

            vals = df[FEATURES].values.astype(np.float32)
            if zero_vol_amount:
                vals = vals.copy()
                vals[:, 4] = 0.0
                vals[:, 5] = 0.0

            windows = _build_windows(
                df['timestamps'], period_start, period_end,
                lookback, pred_len, stride,
            )

            if data_fraction < 1.0 and data_select in (None, 'first', 'last'):
                n_keep = max(1, int(len(windows) * data_fraction))
                windows = windows[-n_keep:] if data_select == 'last' else windows[:n_keep]

            ts_arr = df['timestamps'].values
            close_i = FEATURES.index('close')

            for cs, ce, fe in windows:
                if data_select in (None, 'first', 'last'):
                    self.windows.append((vals, ts_arr, cs, ce))
                elif data_select == 'random':
                    _scored.append((0.0, vals, ts_arr, cs, ce))
                else:
                    fut = vals[ce:fe, close_i]
                    a = acf_sq_returns(fut, acf_select_lags) if (fut > 0).all() else None
                    score = float(np.nanmean(np.abs(a))) if a is not None and np.isfinite(a).any() else -1.0
                    _scored.append((score, vals, ts_arr, cs, ce))

        if data_select is not None and data_select not in ('first', 'last'):
            valid = [w for w in _scored if w[0] >= 0.0]
            n_keep = max(1, int(len(valid) * data_fraction))
            if data_select == 'random':
                idx = np.random.default_rng(data_select_seed).permutation(len(valid))[:n_keep]
                keep = [valid[i] for i in idx]
                print(f"[dataset] data_select=random: kept {n_keep}/{len(valid)} windows (frac={data_fraction})")
            else:
                valid.sort(key=lambda w: w[0], reverse=(data_select == 'high_acf2'))
                keep = valid[:n_keep]
                print(f"[dataset] data_select={data_select}: kept {n_keep}/{len(valid)} "
                      f"(ACF² range {valid[-1][0]:.4f}..{valid[0][0]:.4f})")
            for _, vals, ts_arr, cs, ce in keep:
                self.windows.append((vals, ts_arr, cs, ce))
            print(f"[dataset] data_select={data_select}: kept {n_keep}/{len(valid)} windows "
                  f"(ACF² score range {valid[-1][0]:.4f}..{valid[0][0]:.4f})")

        if len(self.windows) == 0:
            raise RuntimeError(
                f"No windows found for period {period_start}–{period_end}. "
                f"Check data_dir={data_dir} and symbol CSV names."
            )

        print(f"[dataset] {period_start}–{period_end}: {len(self.windows)} windows "
              f"from {len(symbols)} symbols (fraction={data_fraction})")

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int):
        vals, ts_arr, cs, ce = self.windows[idx]
        fe = ce + self.pred_len

        x_raw = vals[cs:ce]
        y_raw = vals[ce:fe]

        mu = x_raw.mean(axis=0)
        sigma = x_raw.std(axis=0)

        x_norm = np.clip((x_raw - mu) / (sigma + 1e-5), -self.clip, self.clip)

        ts_ctx = pd.to_datetime(ts_arr[cs:ce])
        ts_fut = pd.to_datetime(ts_arr[ce:fe])

        x_stamp = _calc_time_stamps(pd.Series(ts_ctx))
        y_stamp = _calc_time_stamps(pd.Series(ts_fut))

        return {
            'x_norm':  torch.from_numpy(x_norm),
            'y_raw':   torch.from_numpy(y_raw),
            'x_stamp': torch.from_numpy(x_stamp),
            'y_stamp': torch.from_numpy(y_stamp),
            'x_mean':  torch.from_numpy(mu),
            'x_std':   torch.from_numpy(sigma),
        }

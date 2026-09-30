"""GARCH(1,1) and HAR-RV classical volatility baselines, evaluated on the EXACT
same test windows the foundation models are evaluated on (same lookback / stride
/ pred_len / test_period, replicating eval/inference.py's window definition), so
sigma^2-MAE and QLIKE are directly comparable to the paper's tables.

Both models are fit ONCE per symbol on train+val log-returns (CPU only, no GPU
or checkpoint dependency), then produce an H-bar-ahead cumulative variance
forecast (sum of squared bar returns, matching the paper's own sigma^2
definition) at every test window's context end via iterative one-step-ahead
forecasting. Neither model produces a price path, so no RankIC is reported for
them -- only the volatility side.

HAR-RV here uses BAR-LEVEL lags {1, 5, 22} rather than true calendar
day/week/month aggregation, applied to whatever bar frequency the dataset uses.
On daily CSI bars these coincide with Corsi (2009)'s intended structure; on
15-minute crypto bars they do not (they are 15 min / 75 min / ~5.5 h), so the
crypto row is a same-lag-structure control rather than a faithful calendar HAR.
The paper's own sigma^2 is defined at the bar level throughout.

Usage:
    python eval/garch_har_baseline.py --config configs/crypto_top20_h32.yaml

Output: {output_dir}/garch_har_{config stem}.json, holding the pooled metrics
per model plus every (symbol, window) row with 's2_pred' / 's2_true'.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from scipy.optimize import minimize

from eval.metrics import aggregate_vol_mae, aggregate_qlike


def strided_windows(ts, lookback, pred_len, stride, test_start, test_end):
    """Verbatim port of eval/inference.py's window definition."""
    test_start = pd.Timestamp(test_start)
    test_end = pd.Timestamp(test_end)
    first_ctx_end = int(ts.searchsorted(test_start, side='left'))
    last_fut_end = int(ts.searchsorted(test_end, side='right'))
    windows = []
    i = first_ctx_end
    while i + pred_len <= last_fut_end and i - lookback >= 0:
        windows.append((i - lookback, i, i + pred_len))
        i += stride
    return windows


def fit_garch11(r):
    r = r[np.isfinite(r)]
    var0 = float(np.var(r)) if np.var(r) > 0 else 1e-8

    def neg_ll(params):
        omega, alpha, beta = params
        if omega <= 1e-12 or alpha < 0 or beta < 0 or alpha + beta >= 0.999:
            return 1e10
        n = len(r)
        h = np.empty(n)
        h[0] = var0
        for t in range(1, n):
            h[t] = omega + alpha * r[t - 1] ** 2 + beta * h[t - 1]
        h = np.maximum(h, 1e-12)
        return float(0.5 * np.sum(np.log(2 * np.pi * h) + r ** 2 / h))

    x0 = [var0 * 0.05, 0.05, 0.90]
    res = minimize(neg_ll, x0, method='Nelder-Mead',
                   options={'maxiter': 3000, 'xatol': 1e-8, 'fatol': 1e-10})
    omega, alpha, beta = res.x
    if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1:
        omega, alpha, beta = var0 * 1e-6, 0.0, 0.0
    return float(omega), float(alpha), float(beta), var0


def garch_h_forecast(omega, alpha, beta, r_full, h_full, ctx_end, H):
    """Sum of the next H one-step-ahead variance forecasts, from origin
    ctx_end (0-indexed: r_full[ctx_end-1] is the last observed return)."""
    h_t = h_full[ctx_end - 1] if ctx_end - 1 < len(h_full) else h_full[-1]
    r_t = r_full[ctx_end - 1] if ctx_end - 1 < len(r_full) else 0.0
    total = 0.0
    h = omega + alpha * r_t ** 2 + beta * h_t
    total += h
    for _ in range(H - 1):
        h = omega + (alpha + beta) * h
        total += h
    return float(total)


def garch_h_series(omega, alpha, beta, r, var0):
    """Filtered conditional variance over the whole return series (train+val
    +test), fixed parameters -- standard for producing out-of-sample forecast
    origins without refitting per window."""
    n = len(r)
    h = np.empty(n)
    h[0] = var0
    for t in range(1, n):
        h[t] = omega + alpha * r[t - 1] ** 2 + beta * h[t - 1]
    return h


def fit_har(rv, lags=(1, 5, 22)):
    """OLS of RV_t on (RV_{t-1}, mean RV over past 5, mean RV over past 22).
    Bar-level lags (not calendar day/week/month); see module docstring."""
    l1, l5, l22 = lags
    n = len(rv)
    X, y = [], []
    for t in range(l22, n):
        X.append([1.0, rv[t - l1], np.mean(rv[t - l5:t]), np.mean(rv[t - l22:t])])
        y.append(rv[t])
    X, y = np.array(X), np.array(y)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return coef


def har_h_forecast(coef, rv_full, ctx_end, H, lags=(1, 5, 22)):
    """Iterative one-step-ahead HAR forecast, H steps, summed."""
    l1, l5, l22 = lags
    rv = list(rv_full[:ctx_end])
    total = 0.0
    for _ in range(H):
        x = np.array([1.0, rv[-l1], np.mean(rv[-l5:]), np.mean(rv[-l22:])])
        pred = max(float(coef @ x), 1e-12)
        total += pred
        rv.append(pred)
    return float(total)


def load_symbol(data_dir, sym, frequency):
    path = os.path.join(data_dir, f'{sym.lower()}_{frequency}.csv')
    if not os.path.exists(path):
        return None
    return pd.read_csv(path, parse_dates=['timestamps'])


def run_config(cfg_path: str) -> dict:
    """Fit both baselines on one config's universe and return pooled metrics."""
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    data_dir = os.path.expandvars(cfg['data_dir'])
    lookback, pred_len, stride = cfg['lookback'], cfg['pred_len'], cfg['stride']
    test_start, test_end = cfg['test_period']
    fit_end = pd.Timestamp(cfg['val_period'][1])
    frequency = cfg['frequency']

    rows = []
    pred = {'GARCH': {}, 'HAR-RV': {}}
    true = {'GARCH': {}, 'HAR-RV': {}}
    n_syms = 0

    for sym in cfg['symbols']:
        df = load_symbol(data_dir, sym, frequency)
        if df is None or len(df) < lookback + pred_len + 30:
            continue
        ts = df['timestamps']
        close = df['close'].values.astype(float)
        windows = strided_windows(ts, lookback, pred_len, stride, test_start, test_end)
        if not windows:
            continue
        r_full = np.diff(np.log(np.maximum(close, 1e-16)))
        rv_full = r_full ** 2
        fit_end_idx = min(int(ts.searchsorted(fit_end, side='right')), len(r_full))
        if fit_end_idx < 60:
            continue
        omega, alpha, beta, var0 = fit_garch11(r_full[:fit_end_idx])
        h_full = garch_h_series(omega, alpha, beta, r_full, var0)
        har_coef = fit_har(rv_full[:fit_end_idx]) if fit_end_idx > 30 else None
        n_syms += 1

        s2_g, s2_h, s2_t = [], [], []
        for (_, ctx_end, fut_end) in windows:
            if fut_end - 1 > len(rv_full):
                continue
            s2_true = float(np.sum(rv_full[ctx_end:fut_end - 1]))
            s2_garch = garch_h_forecast(omega, alpha, beta, r_full, h_full,
                                        ctx_end, pred_len - 1)
            s2_t.append(s2_true)
            s2_g.append(s2_garch)
            rows.append({'sym': sym, 'ctx_end': int(ctx_end), 'model': 'GARCH',
                         's2_pred': s2_garch, 's2_true': s2_true})
            if har_coef is not None:
                s2_har = har_h_forecast(har_coef, rv_full, ctx_end, pred_len - 1)
                s2_h.append(s2_har)
                rows.append({'sym': sym, 'ctx_end': int(ctx_end), 'model': 'HAR-RV',
                             's2_pred': s2_har, 's2_true': s2_true})
        if not s2_t:
            continue
        pred['GARCH'][sym] = np.array(s2_g)
        true['GARCH'][sym] = np.array(s2_t)
        if len(s2_h) == len(s2_t):
            pred['HAR-RV'][sym] = np.array(s2_h)
            true['HAR-RV'][sym] = np.array(s2_t)

    summary = {}
    for model in ('GARCH', 'HAR-RV'):
        if not pred[model]:
            continue
        m = aggregate_vol_mae(pred[model], true[model])
        ql = aggregate_qlike(pred[model], true[model])
        m['n_qlike'] = ql.pop('n')
        m.update(ql)
        summary[model] = m

    return {'config': cfg_path, 'pred_len': pred_len, 'n_symbols': n_syms,
            'summary': summary, 'rows': rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True,
                    help='Eval config; its universe, windows and test_period define the task')
    ap.add_argument('--out', default=None,
                    help='Output JSON path (default: {output_dir}/garch_har_{config stem}.json)')
    args = ap.parse_args()

    result = run_config(args.config)
    with open(args.config) as f:
        out_dir = Path(yaml.safe_load(f)['output_dir'])
    out = Path(args.out) if args.out else out_dir / f'garch_har_{Path(args.config).stem}.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w') as f:
        json.dump(result, f)

    print(f"{args.config}: {result['n_symbols']} symbols, H={result['pred_len']}")
    for model, m in result['summary'].items():
        print(f"  {model:<7} vol MAE = {m['vol_mae']:.6e}  QLIKE = {m['qlike']:.6f}  "
              f"calib = {m['calib_ratio']:.3f}  n={m['n']}")
    print(f"-> {out}  ({len(result['rows'])} rows)")


if __name__ == '__main__':
    main()

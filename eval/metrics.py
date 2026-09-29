"""
Evaluation metrics for Kronos baseline runs.

All metrics designed to match the Kronos paper (arXiv:2508.02739):
  - Price Series IC/RankIC : per-channel correlation between predicted and
    true OHLC paths, averaged over (Open, High, Low, Close). Cross-sectional
    aggregation across symbols at each timestamp, then mean over time.
    Reference: Appendix D, "Metric Calculation Details".
  - Return IC/RankIC : Pearson/Spearman between predicted and true
    end-of-window log return. Cross-sectional aggregation as above.
  - Realized Volatility MAE : σ² = Σ(log p_{i+1} − log p_i)² over the
    PREDICTED close path only (paper Eq. 12, NOT sqrt).
  - ACF² gap : ||ACF(r̂²) − ACF(r²)||² averaged over K lags, computed on
    the prediction window only (not history).
"""

from typing import Dict, List, Sequence
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Per-window primitives
# ─────────────────────────────────────────────────────────────────────────────

def acf_sq_returns(close: np.ndarray, max_lag: int, eps: float = 1e-16) -> np.ndarray:
    """
    ACF of squared log returns for lags 1..max_lag.

    Uses biased 1/T normalizer (Box-Jenkins convention). Returns zeros if
    variance of r² is negligible (e.g. flat sequence).

    Args:
        close   : (T,) close price array
        max_lag : largest lag K to compute
        eps     : floor for log inputs

    Returns:
        (max_lag,) ACF values in [-1, 1]
    """
    log_ret = np.diff(np.log(np.maximum(close, eps)))
    sq = log_ret ** 2
    sq_centered = sq - sq.mean()
    denom = np.dot(sq_centered, sq_centered)
    if denom < eps:
        return np.zeros(max_lag, dtype=np.float64)
    T = len(sq_centered)
    # Lags k >= T have no valid pairs; fill those with 0 to keep shape (max_lag,).
    # Without this guard, k > T would produce mismatched slice lengths → ValueError.
    valid = min(max_lag, T - 1)
    result = np.zeros(max_lag, dtype=np.float64)
    result[:valid] = [
        np.dot(sq_centered[k:], sq_centered[:T - k]) / denom
        for k in range(1, valid + 1)
    ]
    return result


def realized_variance(close: np.ndarray, eps: float = 1e-16) -> float:
    """
    Realized variance σ² = Σ_{i=1}^{H-1} (log p_{i+1} − log p_i)².

    Matches paper Equation 12 exactly. Does NOT take sqrt — returns
    variance, not standard deviation. Use only the predicted/true closes;
    do NOT prepend last historical close (paper formula starts from p_1).

    Args:
        close : (H,) predicted or true close path of length H

    Returns:
        scalar realized variance
    """
    log_r = np.diff(np.log(np.maximum(close, eps)))
    return float(np.sum(log_r ** 2))


def end_of_window_simple_return(close_path: np.ndarray, anchor_close: float,
                                eps: float = 1e-16) -> float:
    """
    Simple return forecasting signal: r̂ = p̂_{t+H} / p_t − 1  (paper Equation 11).

    Note: this is a simple return, not a log return, despite the historical
    function name. The paper explicitly uses this ratio form.

    Args:
        close_path   : (H,) predicted or true close path
        anchor_close : last historical close p_t
    """
    return float(max(close_path[-1], eps) / max(anchor_close, eps) - 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Price-channel IC (multi-channel correlation between paths)
# ─────────────────────────────────────────────────────────────────────────────

def price_channel_ic(pred_path: np.ndarray, true_path: np.ndarray,
                     rank: bool = False) -> float:
    """
    IC/RankIC between predicted and true OHLC paths, averaged over 4 channels.

    Per paper Appendix D: "IC and RankIC are calculated between predicted and
    true series for each of the four price channels (Open, High, Low, Close).
    Final reported metrics are the average across these four channels."

    Args:
        pred_path : (H, D) predicted path. D >= 4: columns 0..3 = OHLC.
        true_path : (H, D) ground truth path, same shape.
        rank      : if True compute Spearman (rank correlation), else Pearson.

    Returns:
        scalar in [-1, 1], the per-window OHLC-averaged IC. NaN if any
        predicted channel is constant (correlation undefined). Channels that
        are entirely NaN (backbone doesn't forecast that price, e.g. Chronos
        univariate close-only) are skipped rather than averaged in — NaN must
        be excluded before ranking, since _rankdata() treats NaN as a normal
        (non-tied) value and would otherwise fabricate a spurious IC.
    """
    n_ch = 4  # O, H, L, C
    ics = []
    for c in range(n_ch):
        x = pred_path[:, c]
        y = true_path[:, c]
        if np.isnan(x).any() or np.isnan(y).any():
            continue  # channel not forecast by this backbone
        if rank:
            x = _rankdata(x)
            y = _rankdata(y)
        ic = _pearson(x, y)
        if np.isnan(ic):
            return np.nan  # propagate: constant channel = degenerate window
        ics.append(ic)
    if not ics:
        return np.nan
    return float(np.mean(ics))


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    xm = x - x.mean()
    ym = y - y.mean()
    denom = np.linalg.norm(xm) * np.linalg.norm(ym)
    if denom < 1e-12:
        return np.nan
    return float(np.dot(xm, ym) / denom)


def _rankdata(x: np.ndarray) -> np.ndarray:
    """Average-rank tie-breaking, identical to scipy.stats.rankdata(method='average')."""
    n = len(x)
    order = np.argsort(x, kind='stable')
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i + 1
        while j < n and x[order[j]] == x[order[i]]:
            j += 1
        avg = (i + 1 + j) / 2.0
        ranks[order[i:j]] = avg
        i = j
    return ranks


# ─────────────────────────────────────────────────────────────────────────────
# Cross-sectional aggregation (the central fix)
# ─────────────────────────────────────────────────────────────────────────────

def cross_sectional_ic(
    pred_returns: Dict[str, np.ndarray],
    true_returns: Dict[str, np.ndarray],
    rank: bool = False,
) -> Dict[str, float]:
    """
    Cross-sectional IC across symbols at each timestamp, then mean over time.

    This is the paper's IC formulation: "for all samples within a given asset
    class and frequency" → samples = (symbol, timestamp) pairs grouped by
    timestamp into cross-sectional vectors.

    Args:
        pred_returns : {symbol: (n_windows,) array of predicted end-of-window
                       log returns}. All symbols must have the SAME n_windows
                       (aligned by timestamp).
        true_returns : same structure, ground truth.
        rank         : True = RankIC (Spearman), False = IC (Pearson).

    Returns:
        dict with:
            ic         : mean of per-timestamp cross-sectional ICs
            ic_std     : std across timestamps
            ic_tstat   : ic / (ic_std / sqrt(n_t)), Newey-West-lite
            n_t        : number of timestamps used
            n_symbols  : cross-section size
            ic_skipped : timestamps dropped due to NaN/insufficient symbols
    """
    symbols = sorted(pred_returns.keys())
    assert symbols == sorted(true_returns.keys()), \
        "pred and true must have same symbol set"

    # Stack: (n_t, n_symbols)
    pred_mat = np.stack([pred_returns[s] for s in symbols], axis=1)
    true_mat = np.stack([true_returns[s] for s in symbols], axis=1)
    assert pred_mat.shape == true_mat.shape

    n_t, n_symbols = pred_mat.shape
    ic_per_t = []
    skipped = 0
    for t in range(n_t):
        pred_vec = pred_mat[t]
        true_vec = true_mat[t]
        mask = np.isfinite(pred_vec) & np.isfinite(true_vec)
        # Need at least 5 symbols for meaningful cross-section
        if mask.sum() < 5:
            skipped += 1
            continue
        x = pred_vec[mask]
        y = true_vec[mask]
        if rank:
            x = _rankdata(x)
            y = _rankdata(y)
        ic_t = _pearson(x, y)
        if np.isnan(ic_t):
            skipped += 1
            continue
        ic_per_t.append(ic_t)

    if not ic_per_t:
        return {
            'ic': np.nan, 'ic_std': np.nan, 'ic_tstat': np.nan,
            'n_t': 0, 'n_symbols': n_symbols, 'ic_skipped': skipped,
        }

    arr = np.array(ic_per_t)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if len(arr) > 1 else np.nan
    tstat = mean / (std / np.sqrt(len(arr))) if (std and std > 0) else np.nan
    return {
        'ic': mean,
        'ic_std': std,
        'ic_tstat': float(tstat) if np.isfinite(tstat) else np.nan,
        'n_t': len(arr),
        'n_symbols': n_symbols,
        'ic_skipped': skipped,
    }


def cross_sectional_price_ic(
    pred_paths: Dict[str, np.ndarray],   # {sym: (n_t, H, D>=4)}
    true_paths: Dict[str, np.ndarray],
    rank: bool = False,
) -> Dict[str, float]:
    """
    Paper Table 14 metric: mean over all (symbol, timestamp) of per-window
    OHLC-channel-averaged path IC.

    Aggregation (per Appendix D):
      Step 1: per (symbol, timestamp) window → IC_window = mean over {O,H,L,C}
              of Pearson(pred_path[:,c], true_path[:,c]) across the H time steps.
      Step 2: mean of IC_window over all (symbol, timestamp) pairs.

    Note: despite the function name, this is NOT cross-sectional at each
    timestamp (no Pearson across symbols per t). The paper averages all
    per-window IC scalars directly.

    Returns dict with:
        ic        : mean over all (symbol, timestamp) per-window ICs
        ic_std    : std of those per-window ICs
        ic_tstat  : t-stat treating windows as independent samples
        n_windows : total (symbol, timestamp) count
    """
    symbols = sorted(pred_paths.keys())
    per_window_ics = []
    nan_windows = 0
    for sym in symbols:
        pp = pred_paths[sym]   # (n_t, H, D)
        tp = true_paths[sym]
        assert pp.shape == tp.shape, f"shape mismatch on {sym}: {pp.shape} vs {tp.shape}"
        n_t = pp.shape[0]
        for i in range(n_t):
            ic = price_channel_ic(pp[i], tp[i], rank=rank)
            if np.isnan(ic):
                nan_windows += 1
                continue
            per_window_ics.append(ic)

    if not per_window_ics:
        return {'ic': np.nan, 'ic_std': np.nan, 'ic_tstat': np.nan,
                'n_windows': 0, 'nan_windows': nan_windows}

    arr = np.array(per_window_ics)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if len(arr) > 1 else np.nan
    tstat = mean / (std / np.sqrt(len(arr))) if (std and std > 0) else np.nan
    return {
        'ic': mean,
        'ic_std': std,
        'ic_tstat': float(tstat) if np.isfinite(tstat) else np.nan,
        'n_windows': len(arr),
        'nan_windows': nan_windows,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Vol MAE aggregator
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_vol_mae(
    pred_rv_sq: Dict[str, np.ndarray],   # {sym: (n_t,)} predicted σ²
    true_rv_sq: Dict[str, np.ndarray],
) -> Dict[str, float]:
    """
    Mean absolute error between predicted and true realized variance σ²
    across all (symbol, timestamp) pairs.

    Paper Table 18 reports per-asset-class MAE. We pool all symbols within
    the same class (crypto here) to match.

    Args:
        pred_rv_sq : {sym: (n_windows,) σ² predictions, averaged over rollouts}
        true_rv_sq : same, ground truth
    """
    abs_errs = []
    for sym in sorted(pred_rv_sq.keys()):
        ae = np.abs(pred_rv_sq[sym] - true_rv_sq[sym])
        abs_errs.append(ae)
    arr = np.concatenate(abs_errs)
    finite = np.isfinite(arr)
    arr = arr[finite]
    sq = arr ** 2
    return {
        'vol_mae':       float(arr.mean()) if len(arr) else np.nan,
        'vol_mae_std':   float(arr.std(ddof=1)) if len(arr) > 1 else np.nan,
        'vol_mse':       float(sq.mean()) if len(sq) else np.nan,
        'vol_mse_std':   float(sq.std(ddof=1)) if len(sq) > 1 else np.nan,
        'vol_r2':        _vol_r2(pred_rv_sq, true_rv_sq),
        'n':             int(len(arr)),
    }


def _vol_r2(pred: Dict[str, np.ndarray], true: Dict[str, np.ndarray]) -> float:
    """
    R² for vol forecast across all (symbol, timestamp) — paper's other vol metric.
    """
    p_all = np.concatenate([pred[s] for s in sorted(pred.keys())])
    t_all = np.concatenate([true[s] for s in sorted(true.keys())])
    mask = np.isfinite(p_all) & np.isfinite(t_all)
    p, t = p_all[mask], t_all[mask]
    if len(t) < 2:
        return np.nan
    ss_res = np.sum((t - p) ** 2)
    ss_tot = np.sum((t - t.mean()) ** 2)
    if ss_tot < 1e-16:
        return np.nan
    return float(1.0 - ss_res / ss_tot)


# ─────────────────────────────────────────────────────────────────────────────
# ACF² gap aggregator
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_acf2_gap(
    pred_acf: Dict[str, np.ndarray],   # {sym: (n_t, K)}  ACF per window
    true_acf: Dict[str, np.ndarray],
) -> Dict[str, float]:
    """
    ||ACF(r̂²) − ACF(r²)||² averaged over K lags, then pooled across all
    (symbol, timestamp). This is the metric the fine-tuning loss targets.
    """
    gaps = []
    for sym in sorted(pred_acf.keys()):
        diff = pred_acf[sym] - true_acf[sym]                # (n_t, K)
        g = np.mean(diff ** 2, axis=1)                      # (n_t,)
        gaps.append(g)
    arr = np.concatenate(gaps)
    arr = arr[np.isfinite(arr)]
    return {
        'acf2_gap':     float(arr.mean()) if len(arr) else np.nan,
        'acf2_gap_std': float(arr.std(ddof=1)) if len(arr) > 1 else np.nan,
        'n':            int(len(arr)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    rng = np.random.default_rng(0)

    # Synthetic test: perfect predictions should give IC=1, vol_MAE=0
    n_t, H, n_sym = 100, 96, 10
    pred_paths, true_paths = {}, {}
    pred_rv, true_rv = {}, {}
    pred_acf, true_acf = {}, {}
    pred_ret, true_ret = {}, {}

    for i in range(n_sym):
        s = f"SYM{i}"
        # Each symbol gets its own RNG stream → cross-sectional variance is real
        rng_s = np.random.default_rng(seed=i)
        # Generate n_t × (1 anchor + H forecast bars) = n_t × (H+1) prices
        prices = 100 * np.exp(np.cumsum(rng_s.normal(0, 0.01, n_t * (H + 1))))
        prices = prices.reshape(n_t, H + 1)
        anchors = prices[:, 0]            # (n_t,)  bar BEFORE the window
        close_walk = prices[:, 1:]        # (n_t, H) forecast window
        # Build OHLC paths around the close walk
        paths = np.stack([
            close_walk * 0.9995,    # open ~ close
            close_walk * 1.0010,    # high
            close_walk * 0.9990,    # low
            close_walk,             # close
        ], axis=2)                  # (n_t, H, 4)
        true_paths[s] = paths
        pred_paths[s] = paths.copy()

        true_rv[s] = np.array([realized_variance(close_walk[t]) for t in range(n_t)])
        pred_rv[s] = true_rv[s].copy()

        true_acf[s] = np.stack([acf_sq_returns(close_walk[t], 20) for t in range(n_t)])
        pred_acf[s] = true_acf[s].copy()

        true_ret[s] = np.array([
            end_of_window_simple_return(close_walk[t], anchors[t]) for t in range(n_t)
        ])
        pred_ret[s] = true_ret[s].copy()

    # Run all metrics
    ic_ret = cross_sectional_ic(pred_ret, true_ret, rank=False)
    ric_ret = cross_sectional_ic(pred_ret, true_ret, rank=True)
    ic_price = cross_sectional_price_ic(pred_paths, true_paths, rank=False)
    vol = aggregate_vol_mae(pred_rv, true_rv)
    acf = aggregate_acf2_gap(pred_acf, true_acf)

    print("=== Self-test: perfect predictions ===")
    print(f"  return IC      : {ic_ret['ic']:.6f}  (expect ~1.0)")
    print(f"  return RankIC  : {ric_ret['ic']:.6f}  (expect ~1.0)")
    print(f"  price IC       : {ic_price['ic']:.6f}  (expect ~1.0)")
    print(f"  vol MAE        : {vol['vol_mae']:.6e}  (expect ~0)")
    print(f"  vol MSE        : {vol['vol_mse']:.6e}  (expect ~0)")
    print(f"  vol R²         : {vol['vol_r2']:.6f}  (expect ~1.0)")
    print(f"  acf2 gap       : {acf['acf2_gap']:.6e}  (expect ~0)")
    assert ic_ret['ic'] > 0.99, "return IC should be ~1 on perfect preds"
    assert vol['vol_mae'] < 1e-10
    assert vol['vol_mse'] < 1e-10
    assert acf['acf2_gap'] < 1e-10
    print("OK — perfect-prediction sanity checks pass.")

    # ── Test 2: noisy predictions should degrade smoothly ────────────────
    print("\n=== Self-test 2: noisy predictions ===")
    for noise in [0.1, 0.5, 1.0]:
        rng_n = np.random.default_rng(42)
        pred_ret_n = {s: true_ret[s] + rng_n.normal(0, noise * np.std(true_ret[s]),
                                                     len(true_ret[s]))
                      for s in true_ret}
        ic_n = cross_sectional_ic(pred_ret_n, true_ret, rank=False)
        print(f"  noise={noise}: IC={ic_n['ic']:.4f}  tstat={ic_n['ic_tstat']:.2f}  n_t={ic_n['n_t']}")
    print("OK — noisy IC degrades as expected.")
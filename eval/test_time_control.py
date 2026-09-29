"""
Test-time control over cached autoregressive rollouts (CPU post-processing, no GPU).

Replaces the mean-over-rollouts aggregation with SELECTION/RE-WEIGHTING toward an ACF-of-
squared-returns target (volatility-clustering stylized fact). See June26_inference.md.

Quick "verify potential" runner: for each arm's cached predictions, compares acf2_gap and
path-shape Price RankIC under several control modes vs the mean baseline, including the
ORACLE best-of-N (target = true ACF) which upper-bounds how much selection can help.

Usage:
    python eval/test_time_control.py <forecast_cache_dir> [<forecast_cache_dir> ...]
    e.g. python eval/test_time_control.py \
         /scratch/$USER/Kronos/predictions/phase2_A2_pathsel_s1/forecast
"""
from __future__ import annotations
import sys
import pickle
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).parent.parent))
from eval.metrics import acf_sq_returns, price_channel_ic

CLOSE_IDX = 3
K = 15
TAU = 2.0
W = np.exp(-np.arange(1, K + 1) / TAU)   # lag weights w_k = exp(-k/tau)


def control_objective(rho_hat: np.ndarray, rho_star: np.ndarray, weighted: bool = True) -> np.ndarray:
    """C = sum_k w_k (rho_hat_k - rho_star_k)^2.  rho_hat: (N, K), rho_star: (K,) -> (N,)."""
    w = W if weighted else 1.0
    return (w * (rho_hat - rho_star[None, :]) ** 2).sum(axis=-1)


def select_idx(roll_acf: np.ndarray, target: np.ndarray, weighted=True) -> int:
    """Best-of-N index: argmin_n C(rollout_n, target)."""
    return int(np.argmin(control_objective(roll_acf, target, weighted)))


def softmax_weights(C: np.ndarray, beta: float) -> np.ndarray:
    z = -beta * (C - C.min())
    w = np.exp(z)
    return w / w.sum()


def run_arm(cache_dir: Path) -> dict:
    pkls = sorted(cache_dir.glob("*.pkl"))
    # accumulators: squared ACF gap per window, path RankIC per window, per control mode.
    # SOFT modes = weighted-over-N (the main proposal): ensemble ACF = sum_n pi_n rho_n,
    # pi_n ∝ exp(-beta C_n). beta=0 == avg_roll_acf. Targets: zero (counter over-clustering),
    # consensus (denoise). best-of-N kept only as the beta->inf ceiling probe.
    modes = ["mean_path", "avg_roll_acf",
             "soft_zero_b5", "soft_zero_b20", "soft_cons_b10",
             "oracle_bestN", "zero_bestN"]
    acf_sq = {m: [] for m in modes}
    rankic = {m: [] for m in modes}

    for pk in pkls:
        d = pickle.load(open(pk, "rb"))
        roll = d["pred_paths_per_rollout"][:, :, :, CLOSE_IDX]   # (T, N, H)
        mean_close = d["pred_paths"][:, :, CLOSE_IDX]            # (T, H)
        roll_full = d["pred_paths_per_rollout"]                  # (T, N, H, 6)
        mean_full = d["pred_paths"]                              # (T, H, 6)
        true_full = d["true_paths"]                              # (T, H, 6)
        true_close = true_full[:, :, CLOSE_IDX]
        T, N, H = roll.shape

        for t in range(T):
            tr_acf = acf_sq_returns(true_close[t], K)
            roll_acf = np.stack([acf_sq_returns(roll[t, n], K) for n in range(N)])  # (N, K)

            # targets (no future / no context needed)
            consensus = np.median(roll_acf, axis=0)
            zero = np.zeros(K)

            # ── baselines ──
            acf_sq["mean_path"].append(((acf_sq_returns(mean_close[t], K) - tr_acf) ** 2).sum())
            rankic["mean_path"].append(_safe_pric(mean_full[t], true_full[t]))
            acf_sq["avg_roll_acf"].append(((roll_acf.mean(0) - tr_acf) ** 2).sum())  # == soft beta=0
            rankic["avg_roll_acf"].append(_safe_pric(mean_full[t], true_full[t]))

            # ── SOFT (weighted-over-N) — the main proposal ──
            # ensemble ACF = sum_n pi_n rho_n; ensemble path = sum_n pi_n path_n
            for name, target, beta in [("soft_zero_b5", zero, 5.0),
                                       ("soft_zero_b20", zero, 20.0),
                                       ("soft_cons_b10", consensus, 10.0)]:
                pi = softmax_weights(control_objective(roll_acf, target), beta)   # (N,)
                ens_acf = (pi[:, None] * roll_acf).sum(0)
                ens_path = (pi[:, None, None] * roll_full[t]).sum(0)
                acf_sq[name].append(((ens_acf - tr_acf) ** 2).sum())
                rankic[name].append(_safe_pric(ens_path, true_full[t]))

            # ── best-of-N ceiling probes ──
            for name, target in [("oracle_bestN", tr_acf), ("zero_bestN", zero)]:
                idx = select_idx(roll_acf, target)
                acf_sq[name].append(((roll_acf[idx] - tr_acf) ** 2).sum())
                rankic[name].append(_safe_pric(roll_full[t, idx], true_full[t]))

    out = {}
    for m in modes:
        out[m] = (float(np.nanmean(acf_sq[m])), float(np.nanmean(rankic[m])))
    return out


def _safe_pric(pred_path, true_path) -> float:
    v = price_channel_ic(pred_path, true_path, rank=True)
    return v if np.isfinite(v) else np.nan


def main():
    dirs = [Path(p) for p in sys.argv[1:]]
    if not dirs:
        print(__doc__)
        return
    print(f"{'arm':<26} {'control mode':<16} {'acf2_gap':>10} {'pathRankIC':>11}")
    print("-" * 66)
    for cd in dirs:
        arm = cd.parent.name
        res = run_arm(cd)
        base_acf = res["avg_roll_acf"][0]
        for m, (a, r) in res.items():
            delta = "" if m == "avg_roll_acf" else f"  ({(a-base_acf)/base_acf*100:+.1f}% vs avg_roll)"
            print(f"{arm:<26} {m:<16} {a:>10.6f} {r:>11.4f}{delta}")
        print("-" * 66)


if __name__ == "__main__":
    main()

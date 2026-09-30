"""
Phase 2 fine-tuning: CE loss (A1) or CE + ACF² loss (A2).

Launch:
    torchrun --standalone --nproc_per_node=NUM_GPUS finetune/train_predictor.py --arm A1
    torchrun --standalone --nproc_per_node=NUM_GPUS finetune/train_predictor.py --arm A2

Gradient path for ACF² (A2):
    model logits → Gumbel-softmax → soft bits (via frozen BSQ bit matrix)
    → frozen tokenizer decoder → denormalized prices → ACF²(close) → loss

Tokenizer weights are frozen; gradient flows through tokenizer computations
(not parameters) back to predictor logits.
"""

from __future__ import annotations
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

sys.path.append(str(Path(__file__).parent.parent))
from model.kronos import KronosTokenizer, Kronos
from finetune.config import Phase2Config, arm_preset
from finetune.dataset import CryptoWindowDataset
from eval.metrics import price_channel_ic
from finetune.utils.training_utils import (
    setup_ddp, cleanup_ddp, set_seed, get_model_size, format_time,
)



def _build_bit_matrix(n_bits: int, device: torch.device) -> torch.Tensor:
    """
    Returns (2^n_bits, n_bits) matrix in {-1, +1}, matching BSQuantizer bit ordering.

    bits_to_indices: sum(bit[k] * 2^k) = index, so bit[k] = (index >> k) & 1.
    """
    vocab = 2 ** n_bits
    indices = torch.arange(vocab, device=device)
    bit_pos = torch.arange(n_bits, device=device)
    bits_0_1 = ((indices.unsqueeze(-1) >> bit_pos.unsqueeze(0)) & 1).float()
    return bits_0_1 * 2 - 1


def soft_decode_prices(
    s1_logits: torch.Tensor,
    s2_logits: torch.Tensor,
    tokenizer: KronosTokenizer,
    tau: float,
    x_means: torch.Tensor,
    x_stds: torch.Tensor,
) -> torch.Tensor:
    """
    Differentiable price reconstruction via Gumbel-softmax relaxation.

    Gradient flows: ACF² loss → prices → soft_bits → Gumbel(logits) → model weights.
    Tokenizer parameters are frozen but gradient passes through their computations.
    """
    B, H, vocab_s1 = s1_logits.shape
    vocab_s2 = s2_logits.shape[-1]
    s1_bits = tokenizer.s1_bits
    s2_bits = tokenizer.s2_bits
    codebook_dim = s1_bits + s2_bits

    s1_bit_mat = _build_bit_matrix(s1_bits, s1_logits.device)
    s2_bit_mat = _build_bit_matrix(s2_bits, s2_logits.device)

    soft_s1 = F.gumbel_softmax(s1_logits.reshape(-1, vocab_s1), tau=tau, hard=False)
    soft_s2 = F.gumbel_softmax(s2_logits.reshape(-1, vocab_s2), tau=tau, hard=False)

    soft_bits_s1 = soft_s1 @ s1_bit_mat
    soft_bits_s2 = soft_s2 @ s2_bit_mat

    q_scale = 1.0 / math.sqrt(codebook_dim)
    soft_full = torch.cat([soft_bits_s1, soft_bits_s2], dim=-1) * q_scale

    soft_full = soft_full.view(B, H, codebook_dim)

    z = tokenizer.post_quant_embed(soft_full)
    for layer in tokenizer.decoder:
        z = layer(z)
    prices_norm = tokenizer.head(z)

    prices = prices_norm * (x_stds.unsqueeze(1) + 1e-5) + x_means.unsqueeze(1)
    return prices


def _soft_tokens_to_prices(
    soft_s1: torch.Tensor,
    soft_s2: torch.Tensor,
    tokenizer: KronosTokenizer,
    x_means: torch.Tensor,
    x_stds: torch.Tensor,
) -> torch.Tensor:
    """
    Decode already-soft tokens (no Gumbel here — the caller sampled them) through the
    frozen tokenizer decoder. Same bit-projection path as soft_decode_prices, but takes
    probabilities directly so the AR rollout can reuse the exact tokens it fed back.
    """
    s1_bits = tokenizer.s1_bits
    s2_bits = tokenizer.s2_bits
    codebook_dim = s1_bits + s2_bits
    s1_bit_mat = _build_bit_matrix(s1_bits, soft_s1.device)
    s2_bit_mat = _build_bit_matrix(s2_bits, soft_s2.device)

    soft_bits_s1 = soft_s1 @ s1_bit_mat
    soft_bits_s2 = soft_s2 @ s2_bit_mat

    q_scale = 1.0 / math.sqrt(codebook_dim)
    soft_full = torch.cat([soft_bits_s1, soft_bits_s2], dim=-1) * q_scale

    z = tokenizer.post_quant_embed(soft_full)
    for layer in tokenizer.decoder:
        z = layer(z)
    prices_norm = tokenizer.head(z)
    prices = prices_norm * (x_stds.unsqueeze(1) + 1e-5) + x_means.unsqueeze(1)
    return prices


def ar_rollout_prices(
    m,
    tokenizer: KronosTokenizer,
    x_norm: torch.Tensor,
    full_stamp: torch.Tensor,
    x_mean: torch.Tensor,
    x_std: torch.Tensor,
    H: int,
    tau: float,
    use_checkpoint: bool = True,
    hard: bool = False,
) -> torch.Tensor:
    """
    Fully differentiable autoregressive rollout.

    Generate the H prediction tokens one step at a time, feeding the model's OWN
    (Gumbel-soft) tokens back as input via soft embeddings, so the ACF² gradient flows
    through the entire generated trajectory — matching what the model actually does at
    test time, unlike the teacher-forced one-step soft-decode.

    Per step (mirrors Kronos.forward exactly, but on the growing sequence):
      transformer → s1_logits → Gumbel-soft s1 → dep_layer(sibling=soft s1) → s2_logits
      → Gumbel-soft s2 → fuse soft s1/s2 embeddings → append → repeat.
    The same Gumbel samples are reused to decode prices, so feedback and ACF are consistent.

    Cost: H sequential transformer passes over a growing sequence; the autograd graph
    spans all H steps (memory-heavy — lower batch_size or add grad checkpointing if OOM).
    """
    emb = m.embedding
    d = m.d_model

    with torch.no_grad():
        tok_s1_ctx, tok_s2_ctx = tokenizer.encode(x_norm, half=True)

    seq_emb = emb([tok_s1_ctx, tok_s2_ctx])

    def _trunk(x_in):
        for layer in m.transformer:
            x_in = layer(x_in)
        return m.norm(x_in)

    soft_s1_list, soft_s2_list = [], []
    for _ in range(H):
        cur = seq_emb.shape[1]
        x_in = m.token_drop(seq_emb + m.time_emb(full_stamp[:, :cur]))
        if use_checkpoint:
            x = checkpoint(_trunk, x_in, use_reentrant=False)
        else:
            x = _trunk(x_in)

        s1_logits_all = m.head(x)
        s1_logits_all = torch.nan_to_num(s1_logits_all, nan=0.0, posinf=30.0, neginf=-30.0)
        soft_s1_all = F.gumbel_softmax(s1_logits_all, tau=tau, hard=hard, dim=-1)
        sibling_all = soft_s1_all @ emb.emb_s1.weight
        x2 = m.dep_layer(x, sibling_all)
        s2_logits_last = m.head.cond_forward(x2)[:, -1]

        soft_s1_last = soft_s1_all[:, -1]
        s2_logits_last = torch.nan_to_num(s2_logits_last, nan=0.0, posinf=30.0, neginf=-30.0)
        soft_s2_last = F.gumbel_softmax(s2_logits_last, tau=tau, hard=hard, dim=-1)
        soft_s1_list.append(soft_s1_last)
        soft_s2_list.append(soft_s2_last)

        s1e = (soft_s1_last @ emb.emb_s1.weight) * math.sqrt(d)
        s2e = (soft_s2_last @ emb.emb_s2.weight) * math.sqrt(d)
        fused = emb.fusion_proj(torch.cat([s1e, s2e], dim=-1)).unsqueeze(1)
        seq_emb = torch.cat([seq_emb, fused], dim=1)

    soft_s1 = torch.stack(soft_s1_list, dim=1)
    soft_s2 = torch.stack(soft_s2_list, dim=1)
    return _soft_tokens_to_prices(soft_s1, soft_s2, tokenizer, x_mean, x_std)


@torch.no_grad()
def ar_argmax_prices(
    m,
    tokenizer: KronosTokenizer,
    x_norm: torch.Tensor,
    full_stamp: torch.Tensor,
    x_mean: torch.Tensor,
    x_std: torch.Tensor,
    H: int,
) -> torch.Tensor:
    """
    Non-differentiable GREEDY autoregressive rollout for VAL selection (no grad, argmax,
    hard tokens fed back). Matches the test-time generation procedure (autoregressive) so the
    val acf2_gap measures the *same object* as test acf2_gap — unlike the teacher-forced argmax
    in eval_val. Greedy (not sampled) keeps it deterministic and cheap for per-epoch selection.
    """
    emb = m.embedding
    seq_s1, seq_s2 = tokenizer.encode(x_norm, half=True)
    gen_s1, gen_s2 = [], []
    for _ in range(H):
        cur = seq_s1.shape[1]
        x = emb([seq_s1, seq_s2]) + m.time_emb(full_stamp[:, :cur])
        for layer in m.transformer:
            x = layer(x)
        x = m.norm(x)
        s1_ids_all = m.head(x).argmax(-1)
        sibling_all = emb.emb_s1(s1_ids_all)
        x2 = m.dep_layer(x, sibling_all)
        s2_last = m.head.cond_forward(x2)[:, -1].argmax(-1, keepdim=True)
        s1_last = s1_ids_all[:, -1:]
        gen_s1.append(s1_last)
        gen_s2.append(s2_last)
        seq_s1 = torch.cat([seq_s1, s1_last], dim=1)
        seq_s2 = torch.cat([seq_s2, s2_last], dim=1)

    g1 = torch.cat(gen_s1, dim=1)
    g2 = torch.cat(gen_s2, dim=1)
    prices_norm = tokenizer.decode([g1, g2], half=True)
    prices = prices_norm.float() * (x_std.unsqueeze(1) + 1e-5) + x_mean.unsqueeze(1)
    return prices



def acf2_loss(
    pred_prices: torch.Tensor,
    true_close: torch.Tensor,
    k_train: int,
    tau_lag: float,
    eps: float = 1e-8,
    aggregate: bool = False,
    adaptive_k: bool = False,
    k_max: int = 10,
    w_floor: float = 0.02,
    adaptive_prior_tau: float = 0.0,
    adaptive_mode: str = 'discrepancy',
    over_penalty: float = 1.0,
    pooled: bool = False,
) -> torch.Tensor:
    """
    Weighted MSE between predicted and true ACF of squared log-returns.

    w_k = exp(-k / tau_lag), summed over lags k=1..k_train.
    aggregate=False (default): per-window — mean over the batch of (pred_acf−true_acf)². Each window
      chases its OWN (noisy, 31-return) true ACF.
    aggregate=True: match the batch-MEAN pred ACF to the batch-MEAN true ACF — (E[pred_acf]−E[true_acf])².
      Averaging the ACF over the batch before squaring cancels the per-window estimation noise, so the
      gradient targets the ensemble clustering LEVEL (the aggregate metric) rather than per-path noise.
    """
    pred_close = pred_prices[:, :, 3].clamp(min=eps)
    true_close = true_close.clamp(min=eps)

    pred_lr = torch.diff(torch.log(pred_close), dim=-1)
    true_lr = torch.diff(torch.log(true_close), dim=-1)

    pred_sq = pred_lr ** 2
    true_sq = true_lr ** 2

    T = pred_sq.shape[-1]
    pred_sq_c = pred_sq - pred_sq.mean(dim=-1, keepdim=True)
    true_sq_c = true_sq - true_sq.mean(dim=-1, keepdim=True)

    pred_denom = (pred_sq_c ** 2).sum(dim=-1).clamp(min=eps)
    true_denom = (true_sq_c ** 2).sum(dim=-1).clamp(min=eps)

    if pooled and not adaptive_k:
        pgc = pred_sq - pred_sq.mean()
        tgc = true_sq - true_sq.mean()
        pden = (pgc ** 2).sum().clamp(min=eps)
        tden = (tgc ** 2).sum().clamp(min=eps)
        loss = pred_prices.new_zeros(())
        for k in range(1, k_train + 1):
            w_k = math.exp(-k / tau_lag)
            prho = (pgc[:, k:] * pgc[:, :T - k]).sum() / pden
            trho = (tgc[:, k:] * tgc[:, :T - k]).sum() / tden
            loss = loss + w_k * (prho - trho) ** 2
        return loss

    if adaptive_k:
        Kx = min(k_max, T - 1)
        pm, tm, sq_err, dsc, snr = [], [], [], [], []
        for k in range(1, Kx + 1):
            pak = (pred_sq_c[:, k:] * pred_sq_c[:, :T - k]).sum(dim=-1) / pred_denom
            tak = (true_sq_c[:, k:] * true_sq_c[:, :T - k]).sum(dim=-1) / true_denom
            if aggregate:
                d = pak.mean() - tak.mean()
                se = d ** 2
                if over_penalty != 1.0 and d.detach() > 0:
                    se = se * over_penalty
                sq_err.append(se)
                dsc.append(tak.detach().abs().mean() if adaptive_mode == 'target' else d.detach().abs())
                snr.append(tak.detach().std() + eps)
            else:
                diff = pak - tak
                se_vec = diff ** 2
                if over_penalty != 1.0:
                    se_vec = se_vec * torch.where(diff.detach() > 0, over_penalty, 1.0)
                sq_err.append(se_vec.mean())
                dsc.append(tak.detach().abs().mean() if adaptive_mode == 'target'
                           else diff.detach().abs().mean())
                snr.append(tak.detach().std() + eps)
        prior = [math.exp(-(k) / adaptive_prior_tau) if adaptive_prior_tau > 0 else 1.0
                 for k in range(1, Kx + 1)]
        raw = torch.stack([p * d / s for p, d, s in zip(prior, dsc, snr)])
        w = raw / (raw.sum() + eps)
        w = torch.clamp(w, min=w_floor); w = w / w.sum()
        return torch.stack(sq_err).mul(w).sum()

    loss = pred_prices.new_zeros(())
    for k in range(1, k_train + 1):
        w_k = math.exp(-k / tau_lag)
        pred_acf_k = (pred_sq_c[:, k:] * pred_sq_c[:, :T - k]).sum(dim=-1) / pred_denom
        true_acf_k = (true_sq_c[:, k:] * true_sq_c[:, :T - k]).sum(dim=-1) / true_denom
        if aggregate:
            loss = loss + w_k * (pred_acf_k.mean() - true_acf_k.mean()) ** 2
        else:
            loss = loss + w_k * ((pred_acf_k - true_acf_k) ** 2).mean()
    return loss


def directional_loss(
    pred_prices: torch.Tensor,
    true_close: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Soft sign-agreement between predicted and true per-step log-returns.

        L_dir = mean(1 - tanh(r̂/s) · tanh(r/s)),   s = std(r).detach()

    ACF(r²) is sign-blind, so ACF² alone buys volatility *magnitude* with no control over
    *direction* — measured ρ(r̂, r) ≈ 0, which is why the added amplitude lands wrong-way and
    inflates MAE. This term is the missing sign constraint.

    Range [0, 2]: 0 = confident agreement, 1 = no directional information, 2 = confident
    disagreement. tanh saturates, so a single outlier return cannot dominate the batch (a raw
    r̂·r product would). s makes it scale-free: returns are O(1e-3), so without it tanh is
    linear and the term degenerates into a plain correlation.
    """
    pred_close = pred_prices[:, :, 3].clamp(min=eps)
    true_close = true_close.clamp(min=eps)
    pred_lr = torch.diff(torch.log(pred_close), dim=-1)
    true_lr = torch.diff(torch.log(true_close), dim=-1)
    s = true_lr.std().detach().clamp(min=eps)
    return (1.0 - torch.tanh(pred_lr / s) * torch.tanh(true_lr / s)).mean()


def stylized_fact_loss(pred_prices, true_close, eps=1e-8):
    """
    B — extra econometric stylized facts beyond clustering, matched on the differentiable rollout
    (aggregate: batch-mean pred vs batch-mean true). Returns (leverage, kurtosis, variance) so the
    caller can weight each.
      leverage : lag-1 cross-moment corr(r_t, r²_{t+1}) — leverage effect
      kurtosis : E[r⁴]/E[r²]² − 3 — fat tails
      variance : E[r²] — dispersion LEVEL (VaR needs this; ACF² does not control it)
    """
    p = pred_prices[:, :, 3].clamp(min=eps)
    t = true_close.clamp(min=eps)
    pr = torch.diff(torch.log(p), dim=-1)
    tr = torch.diff(torch.log(t), dim=-1)
    psq, tsq = pr ** 2, tr ** 2
    l_var = (psq.mean() - tsq.mean()) ** 2
    pk = (psq ** 2).mean() / (psq.mean() ** 2 + eps) - 3.0
    tk = (tsq ** 2).mean() / (tsq.mean() ** 2 + eps) - 3.0
    l_kurt = (pk - tk) ** 2
    prc, psc = pr - pr.mean(), psq - psq.mean()
    trc, tsc = tr - tr.mean(), tsq - tsq.mean()
    p_lev = (prc[:, :-1] * psc[:, 1:]).mean() / (pr.std() * psq.std() + eps)
    t_lev = (trc[:, :-1] * tsc[:, 1:]).mean() / (tr.std() * tsq.std() + eps)
    l_lev = (p_lev - t_lev) ** 2
    return l_lev, l_kurt, l_var



def gumbel_tau(step: int, total_steps: int, tau_start: float, tau_end: float) -> float:
    frac = min(step / max(total_steps, 1), 1.0)
    return tau_start * (tau_end / tau_start) ** frac


def scheduled_p_self(step: int, total_steps: int, warmup_frac: float, p_max: float) -> float:
    warmup = int(total_steps * warmup_frac)
    return min(step / max(warmup, 1), 1.0) * p_max



def parkinson_rv(high: np.ndarray, low: np.ndarray, eps: float = 1e-16) -> float:
    """Parkinson realized variance: sum (log H/L)^2 / (4 ln 2)."""
    return float(np.sum((np.log(np.maximum(high, eps) / np.maximum(low, eps)) ** 2)
                        / (4 * math.log(2))))


def acf_sq_returns_np(close: np.ndarray, max_lag: int, eps: float = 1e-16) -> np.ndarray:
    lr = np.diff(np.log(np.maximum(close, eps)))
    sq = lr ** 2
    sq_c = sq - sq.mean()
    denom = np.dot(sq_c, sq_c)
    if denom < eps:
        return np.zeros(max_lag)
    T = len(sq_c)
    valid = min(max_lag, T - 1)
    result = np.zeros(max_lag)
    result[:valid] = [np.dot(sq_c[k:], sq_c[:T - k]) / denom for k in range(1, valid + 1)]
    return result


def _rankdata_np(x: np.ndarray) -> np.ndarray:
    n = len(x)
    order = np.argsort(x, kind='stable')
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i + 1
        while j < n and x[order[j]] == x[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + 1 + j) / 2.0
        i = j
    return ranks


def _pearson_np(x: np.ndarray, y: np.ndarray) -> float:
    xm, ym = x - x.mean(), y - y.mean()
    denom = np.linalg.norm(xm) * np.linalg.norm(ym)
    return float(np.dot(xm, ym) / denom) if denom > 1e-12 else float('nan')


@torch.no_grad()
def eval_val(
    model: torch.nn.Module,
    tokenizer: KronosTokenizer,
    val_loader: DataLoader,
    cfg: Phase2Config,
    device: torch.device,
    max_batches: int = 50,
) -> dict:
    """
    Quick per-epoch val eval on a subset of val windows.
    Returns: acf2_gap, price_rankic (proxy), parkinson_r2.

    ACF² gap: teacher-forced logits → argmax tokens → decode → ACF.
    Price RankIC proxy: correlation of predicted vs true last-close change.
    Parkinson R²: regression of predicted vs true Parkinson RV.
    """
    model.eval()
    tokenizer.eval()

    K_eval = 15

    pred_acf_all, true_acf_all = [], []
    pred_rv_all, true_rv_all = [], []
    pred_ret_all, true_ret_all = [], []
    price_ic_path_all, price_rankic_path_all = [], []

    autocast_ctx = torch.amp.autocast('cuda', dtype=torch.float16)

    n_batches = 0
    for batch in val_loader:
        if n_batches >= max_batches:
            break
        n_batches += 1

        x_norm = batch['x_norm'].to(device)
        y_raw  = batch['y_raw'].to(device)
        x_stamp = batch['x_stamp'].to(device)
        y_stamp = batch['y_stamp'].to(device)
        x_mean = batch['x_mean'].to(device)
        x_std  = batch['x_std'].to(device)
        B, H = x_norm.shape[0], y_raw.shape[1]

        y_norm = torch.clamp(
            (y_raw - x_mean.unsqueeze(1)) / (x_std.unsqueeze(1) + 1e-5),
            -cfg.clip, cfg.clip,
        )
        full_seq_norm = torch.cat([x_norm, y_norm], dim=1)
        tok_s1, tok_s2 = tokenizer.encode(full_seq_norm, half=True)

        full_stamp = torch.cat([x_stamp, y_stamp], dim=1)
        token_in_s1 = tok_s1[:, :-1]
        token_in_s2 = tok_s2[:, :-1]

        with autocast_ctx:
            s1_logits, s2_logits = model(token_in_s1, token_in_s2, full_stamp[:, :-1])
        H = cfg.pred_len
        with autocast_ctx:
            pred_s1_logits = s1_logits[:, -H:, :]
            pred_s2_logits = s2_logits[:, -H:, :]
            pred_tok_s1 = pred_s1_logits.argmax(-1)
            pred_tok_s2 = pred_s2_logits.argmax(-1)
            pred_prices_norm = tokenizer.decode([pred_tok_s1, pred_tok_s2], half=True)
            pred_prices = (pred_prices_norm.float() * (x_std.unsqueeze(1) + 1e-5)
                           + x_mean.unsqueeze(1))

        pred_np = pred_prices.cpu().numpy()
        true_np = y_raw.cpu().numpy()

        for i in range(B):
            pred_close = pred_np[i, :, 3]
            true_close = true_np[i, :, 3]
            pred_high = pred_np[i, :, 1]
            pred_low = pred_np[i, :, 2]
            true_high = true_np[i, :, 1]
            true_low = true_np[i, :, 2]

            pred_acf_all.append(acf_sq_returns_np(pred_close, K_eval))
            true_acf_all.append(acf_sq_returns_np(true_close, K_eval))

            pred_rv_all.append(parkinson_rv(pred_high, pred_low))
            true_rv_all.append(parkinson_rv(true_high, true_low))

            pred_ret_all.append(float(pred_close[-1] / max(pred_close[0], 1e-16) - 1))
            true_ret_all.append(float(true_close[-1] / max(true_close[0], 1e-16) - 1))

            pic = price_channel_ic(pred_np[i], true_np[i], rank=False)
            pric = price_channel_ic(pred_np[i], true_np[i], rank=True)
            if np.isfinite(pic):
                price_ic_path_all.append(pic)
            if np.isfinite(pric):
                price_rankic_path_all.append(pric)

    pred_acf = np.array(pred_acf_all)
    true_acf = np.array(true_acf_all)
    diff = pred_acf - true_acf
    acf2_gap = float(np.mean(diff ** 2))

    pred_rv = np.array(pred_rv_all)
    true_rv = np.array(true_rv_all)
    mask = np.isfinite(pred_rv) & np.isfinite(true_rv)
    pred_rv, true_rv = pred_rv[mask], true_rv[mask]
    if len(true_rv) > 1:
        ss_res = np.sum((true_rv - pred_rv) ** 2)
        ss_tot = np.sum((true_rv - true_rv.mean()) ** 2)
        park_r2 = float(1 - ss_res / max(ss_tot, 1e-16))
    else:
        park_r2 = float('nan')

    pred_ret = np.array(pred_ret_all)
    true_ret = np.array(true_ret_all)
    try:
        rank_ic = float(_pearson_np(_rankdata_np(pred_ret), _rankdata_np(true_ret)))
    except Exception:
        rank_ic = float('nan')
    try:
        price_ic = float(_pearson_np(pred_ret, true_ret))
    except Exception:
        price_ic = float('nan')

    price_ic_path = float(np.mean(price_ic_path_all)) if price_ic_path_all else float('nan')
    price_rankic_path = (float(np.mean(price_rankic_path_all))
                         if price_rankic_path_all else float('nan'))

    return {
        'acf2_gap': acf2_gap,
        'parkinson_r2': park_r2,
        'price_rankic': rank_ic,
        'price_ic': price_ic,
        'price_ic_path': price_ic_path,
        'price_rankic_path': price_rankic_path,
        'n_windows': len(pred_acf_all),
    }


@torch.no_grad()
def eval_val_ar(
    model: torch.nn.Module,
    tokenizer: KronosTokenizer,
    val_loader: DataLoader,
    cfg: Phase2Config,
    device: torch.device,
    max_batches: int = 15,
) -> dict:
    """
    Autoregressive val metrics (greedy rollout) — matches the test generation procedure,
    so acf2_gap_ar / price_rankic_path_ar measure the same object as test. Slower than the
    teacher-forced eval_val (H sequential passes), so it runs on fewer batches.
    """
    model.eval()
    tokenizer.eval()
    K_eval = 15
    pred_acf_all, true_acf_all = [], []
    pic_path_all, pric_path_all = [], []
    pred_ret_all, true_ret_all = [], []
    mse_rel_all = []
    n_roll_all, n_neg_all, n_s2gt1_all, s2_max_all = [0], [0], [0], [0.0]

    nb = 0
    for batch in val_loader:
        if nb >= max_batches:
            break
        nb += 1
        x_norm = batch['x_norm'].to(device)
        y_raw = batch['y_raw'].to(device)
        x_stamp = batch['x_stamp'].to(device)
        y_stamp = batch['y_stamp'].to(device)
        x_mean = batch['x_mean'].to(device)
        x_std = batch['x_std'].to(device)
        full_stamp = torch.cat([x_stamp, y_stamp], dim=1)

        prices = ar_argmax_prices(model, tokenizer, x_norm, full_stamp,
                                  x_mean, x_std, cfg.pred_len)
        pred_np = prices.cpu().numpy()
        true_np = y_raw.cpu().numpy()
        for i in range(pred_np.shape[0]):
            pc, tc = pred_np[i, :, 3], true_np[i, :, 3]
            n_roll_all[0] += 1
            if float(np.min(pc)) <= 0:
                n_neg_all[0] += 1
            s2 = float(np.sum(np.diff(np.log(np.maximum(pc, 1e-16))) ** 2))
            if np.isfinite(s2):
                s2_max_all[0] = max(s2_max_all[0], s2)
                if s2 > 1.0:
                    n_s2gt1_all[0] += 1
            pred_acf_all.append(acf_sq_returns_np(pc, K_eval))
            true_acf_all.append(acf_sq_returns_np(tc, K_eval))
            pic = price_channel_ic(pred_np[i], true_np[i], rank=False)
            pric = price_channel_ic(pred_np[i], true_np[i], rank=True)
            if np.isfinite(pic):
                pic_path_all.append(pic)
            if np.isfinite(pric):
                pric_path_all.append(pric)
            pred_ret_all.append(float(pc[-1] / max(pc[0], 1e-16) - 1))
            true_ret_all.append(float(tc[-1] / max(tc[0], 1e-16) - 1))
            mse_rel_all.append(float(np.mean((pc / np.maximum(tc, 1e-6) - 1.0) ** 2)))

    pred_acf_arr, true_acf_arr = np.array(pred_acf_all), np.array(true_acf_all)
    diff = pred_acf_arr - true_acf_arr
    w_agg = np.exp(-np.arange(1, K_eval + 1) / 2.0)
    acf2_gap_ar_agg = float((w_agg * (pred_acf_arr.mean(0) - true_acf_arr.mean(0)) ** 2).sum())
    pred_ret, true_ret = np.array(pred_ret_all), np.array(true_ret_all)
    try:
        ret_rankic = float(_pearson_np(_rankdata_np(pred_ret), _rankdata_np(true_ret)))
        ret_ic = float(_pearson_np(pred_ret, true_ret))
    except Exception:
        ret_rankic = ret_ic = float('nan')
    return {
        'acf2_gap_ar': float(np.mean(diff ** 2)),
        'acf2_gap_ar_agg': acf2_gap_ar_agg,
        'price_ic_path_ar': float(np.mean(pic_path_all)) if pic_path_all else float('nan'),
        'price_rankic_path_ar': float(np.mean(pric_path_all)) if pric_path_all else float('nan'),
        'return_ic_ar': ret_ic,
        'return_rankic_ar': ret_rankic,
        'n_windows_ar': len(pred_acf_all),
        'val_pct_neg_ar': 100.0 * n_neg_all[0] / max(n_roll_all[0], 1),
        'val_pct_s2gt1_ar': 100.0 * n_s2gt1_all[0] / max(n_roll_all[0], 1),
        'val_max_s2_ar': s2_max_all[0],
        'mse_rel_ar': float(np.mean(mse_rel_all)) if mse_rel_all else float('nan'),
    }



def train(
    model: DDP,
    tokenizer: KronosTokenizer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: Phase2Config,
    save_dir: Path,
    rank: int,
    device: torch.device,
    val_loader2: DataLoader = None,
) -> dict:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        betas=(cfg.adam_beta1, cfg.adam_beta2),
        weight_decay=cfg.weight_decay,
    )
    total_steps = len(train_loader) * cfg.epochs
    sched_steps = max(1, (len(train_loader) // cfg.grad_accum) * cfg.epochs)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=cfg.lr,
        total_steps=sched_steps,
        pct_start=0.03, div_factor=10,
    )

    scaler = torch.amp.GradScaler('cuda')
    autocast_ctx = torch.amp.autocast('cuda', dtype=torch.float16)

    wb = None
    if rank == 0 and getattr(cfg, 'use_wandb', False):
        try:
            import wandb as _wb
            _wb.init(project='vca', name=cfg.run_name,
                     config={k: v for k, v in vars(cfg).items() if isinstance(v, (int, float, str, bool))})
            wb = _wb
            print(f"[wandb] logging to project=vca run={cfg.run_name}")
        except Exception as e:
            print(f"[wandb] disabled ({e})")

    best_acf2_gap = float('inf')
    best_price_rankic = float('-inf')
    best_acf2_gap_ar = float('inf')
    best_arval_pw = float('inf')
    best_arval_agg = float('inf')
    history = []
    global_step = 0
    t0 = time.time()
    ema_lambda = cfg.lambda_acf

    for epoch in range(cfg.epochs):
        model.train()
        train_loader.sampler.set_epoch(epoch)

        ep_ce_loss = 0.0
        ep_acf_loss = 0.0
        ep_batches = 0
        accum_n = 0

        for batch in train_loader:
            x_norm = batch['x_norm'].to(device, non_blocking=True)
            y_raw = batch['y_raw'].to(device, non_blocking=True)
            x_stamp = batch['x_stamp'].to(device, non_blocking=True)
            y_stamp = batch['y_stamp'].to(device, non_blocking=True)
            x_mean = batch['x_mean'].to(device, non_blocking=True)
            x_std = batch['x_std'].to(device, non_blocking=True)

            B, L, _ = x_norm.shape
            H = cfg.pred_len
            full_stamp = torch.cat([x_stamp, y_stamp], dim=1)

            with torch.no_grad():
                y_norm = torch.clamp(
                    (y_raw - x_mean.unsqueeze(1)) / (x_std.unsqueeze(1) + 1e-5),
                    -cfg.clip, cfg.clip,
                )
                full_seq_norm = torch.cat([x_norm, y_norm], dim=1)
                tok_s1, tok_s2 = tokenizer.encode(full_seq_norm, half=True)

            token_in_s1 = tok_s1[:, :-1]
            token_in_s2 = tok_s2[:, :-1]
            token_out_s1 = tok_s1[:, 1:]
            token_out_s2 = tok_s2[:, 1:]

            if cfg.rollout_mode == 'full_ar':
                mod = model.module
                with autocast_ctx:
                    s1_logits, s2_logits = mod(token_in_s1, token_in_s2, full_stamp[:, :-1])
                    loss_ce, _, _ = mod.head.compute_loss(
                        s1_logits, s2_logits, token_out_s1, token_out_s2,
                    )
                tau = gumbel_tau(global_step, total_steps,
                                 cfg.gumbel_tau_start, cfg.gumbel_tau_end)
                with torch.amp.autocast('cuda', enabled=False):
                    pred_prices = ar_rollout_prices(
                        mod, tokenizer, x_norm.float(), full_stamp,
                        x_mean.float(), x_std.float(), H, tau, hard=cfg.gumbel_hard,
                    )
                    loss_acf = acf2_loss(pred_prices, y_raw[:, :, 3].float(),
                                         cfg.k_train, cfg.tau_lag, aggregate=cfg.acf_loss_agg,
                                         adaptive_k=getattr(cfg, 'adaptive_k', False),
                                         k_max=getattr(cfg, 'k_max', 10),
                                         adaptive_prior_tau=getattr(cfg, 'adaptive_prior_tau', 0.0),
                                         adaptive_mode=getattr(cfg, 'adaptive_mode', 'discrepancy'),
                                         over_penalty=getattr(cfg, 'over_penalty', 1.0),
                                         pooled=getattr(cfg, 'acf_pooled', False))
                    loss_dir = (directional_loss(pred_prices, y_raw[:, :, 3].float())
                                if cfg.lambda_dir > 0.0 else pred_prices.new_zeros(()))
                if cfg.adaptive_lambda and (global_step % cfg.grad_norm_interval == 0):
                    try:
                        proxy = [p for p in mod.head.parameters() if p.requires_grad]
                        gc = torch.autograd.grad(loss_ce.float(), proxy, retain_graph=True, allow_unused=True)
                        ga = torch.autograd.grad(loss_acf, proxy, retain_graph=True, allow_unused=True)
                        gcn = float(torch.sqrt(sum((g.detach()**2).sum() for g in gc if g is not None)))
                        gan = float(torch.sqrt(sum((g.detach()**2).sum() for g in ga if g is not None)))
                        if gan > 1e-12:
                            lam_inst = cfg.alpha_grad * gcn / gan
                            lam_inst = float(np.clip(lam_inst, 0.5, 50.0))
                            ema_lambda = cfg.ema_lambda_beta * ema_lambda + (1 - cfg.ema_lambda_beta) * lam_inst
                    except Exception as e:
                        if global_step == 0 and rank == 0:
                            print(f"[adaptive-λ] disabled ({e})")
                lam_acf = ema_lambda if cfg.adaptive_lambda else cfg.lambda_acf
                if cfg.lambda_schedule == 'warmup':
                    ramp = max(1, int(cfg.warmup_frac * total_steps))
                    lam_acf = cfg.lambda_acf * min(1.0, global_step / ramp)
                if cfg.lambda_dir > 0.0 and cfg.dir_mode == 'mult':
                    loss = loss_ce.float() + lam_acf * loss_acf * (1.0 + cfg.lambda_dir * loss_dir)
                else:
                    loss = loss_ce.float() + lam_acf * loss_acf + cfg.lambda_dir * loss_dir
                if cfg.lambda_mse > 0.0:
                    true_close = y_raw[:, :, 3].float().clamp_min(1e-6)
                    pred_close = pred_prices[:, :, 3].float()
                    loss_mse = ((pred_close / true_close - 1.0) ** 2).mean()
                    loss = loss + cfg.lambda_mse * loss_mse
                if (getattr(cfg, 'lambda_lev', 0.0) + getattr(cfg, 'lambda_kurt', 0.0)
                        + getattr(cfg, 'lambda_var', 0.0)) > 0.0:
                    l_lev, l_kurt, l_var = stylized_fact_loss(pred_prices, y_raw[:, :, 3].float())
                    loss = (loss + getattr(cfg, 'lambda_lev', 0.0) * l_lev
                            + getattr(cfg, 'lambda_kurt', 0.0) * l_kurt
                            + getattr(cfg, 'lambda_var', 0.0) * l_var)

                if accum_n == 0:
                    optimizer.zero_grad()
                finite = torch.tensor(float(torch.isfinite(loss)), device=loss.device)
                if dist.is_initialized() and dist.get_world_size() > 1:
                    dist.all_reduce(finite, op=dist.ReduceOp.MIN)
                if finite.item() < 0.5:
                    if rank == 0:
                        print(f"  [skip] non-finite loss at ep{epoch+1} step {global_step} — batch dropped")
                    continue
                scaler.scale(loss / cfg.grad_accum).backward()
                accum_n += 1
                if accum_n >= cfg.grad_accum:
                    accum_n = 0
                    scaler.unscale_(optimizer)
                    ws = dist.get_world_size()
                    if ws > 1:
                        for p in mod.parameters():
                            if p.grad is not None:
                                dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                                p.grad.div_(ws)
                    grad_norm = torch.nn.utils.clip_grad_norm_(mod.parameters(), cfg.grad_clip)
                    grad_finite = torch.tensor(float(torch.isfinite(grad_norm)), device=loss.device)
                    if ws > 1:
                        dist.all_reduce(grad_finite, op=dist.ReduceOp.MIN)
                    if grad_finite.item() < 0.5:
                        if rank == 0:
                            print(f"  [skip] non-finite grad at ep{epoch+1} step {global_step} — update dropped")
                        optimizer.zero_grad()
                    else:
                        scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
            else:
                with autocast_ctx:
                    s1_logits, s2_logits = model(token_in_s1, token_in_s2, full_stamp[:, :-1])
                    loss_ce, _, _ = model.module.head.compute_loss(
                        s1_logits, s2_logits, token_out_s1, token_out_s2,
                    )

                    loss_acf = x_norm.new_zeros(())
                    if cfg.arm in ('A2', 'A3'):
                        tau = gumbel_tau(global_step, total_steps,
                                         cfg.gumbel_tau_start, cfg.gumbel_tau_end)
                        pred_s1_logits = s1_logits[:, -H:, :]
                        pred_s2_logits = s2_logits[:, -H:, :]
                        pred_prices = soft_decode_prices(
                            pred_s1_logits, pred_s2_logits,
                            tokenizer, tau, x_mean, x_std,
                        )
                        true_close = y_raw[:, :, 3]
                        loss_acf = acf2_loss(pred_prices, true_close, cfg.k_train, cfg.tau_lag,
                                             aggregate=cfg.acf_loss_agg,
                                             adaptive_k=getattr(cfg, 'adaptive_k', False),
                                             k_max=getattr(cfg, 'k_max', 10),
                                             adaptive_prior_tau=getattr(cfg, 'adaptive_prior_tau', 0.0),
                                             adaptive_mode=getattr(cfg, 'adaptive_mode', 'discrepancy'),
                                             over_penalty=getattr(cfg, 'over_penalty', 1.0),
                                             pooled=getattr(cfg, 'acf_pooled', False))

                if cfg.arm == 'A3' and global_step % cfg.grad_norm_interval == 0:
                    proxy = [p for p in model.module.head.parameters() if p.requires_grad]
                    if proxy and loss_acf.item() > 1e-10:
                        g_ce = torch.autograd.grad(
                            loss_ce, proxy, retain_graph=True,
                            allow_unused=True, create_graph=False,
                        )
                        g_acf = torch.autograd.grad(
                            loss_acf, proxy, retain_graph=True,
                            allow_unused=True, create_graph=False,
                        )
                        g_ce_n  = sum(g.detach().norm()**2 for g in g_ce  if g is not None).sqrt().item()
                        g_acf_n = sum(g.detach().norm()**2 for g in g_acf if g is not None).sqrt().item()
                        if g_acf_n > 1e-8:
                            lam_inst = float(cfg.alpha_grad * g_ce_n / g_acf_n)
                            lam_inst = max(0.1, min(lam_inst, 500.0))
                            ema_lambda = (cfg.ema_lambda_beta * ema_lambda
                                          + (1 - cfg.ema_lambda_beta) * lam_inst)

                with autocast_ctx:
                    lam = ema_lambda if cfg.arm == 'A3' else cfg.lambda_acf
                    loss = loss_ce + lam * loss_acf

                if accum_n == 0:
                    optimizer.zero_grad()
                finite = torch.tensor(float(torch.isfinite(loss)), device=loss.device)
                if dist.is_initialized() and dist.get_world_size() > 1:
                    dist.all_reduce(finite, op=dist.ReduceOp.MIN)
                if finite.item() < 0.5:
                    if rank == 0:
                        print(f"  [skip] non-finite loss at ep{epoch+1} step {global_step} — batch dropped")
                    continue
                scaler.scale(loss / cfg.grad_accum).backward()
                accum_n += 1
                if accum_n >= cfg.grad_accum:
                    accum_n = 0
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()

            ep_ce_loss += loss_ce.item()
            ep_acf_loss += loss_acf.item() if cfg.arm in ('A2', 'A3') else 0.0
            ep_batches += 1
            global_step += 1

            if rank == 0 and global_step % cfg.log_interval == 0:
                lr = optimizer.param_groups[0]['lr']
                msg = (f"[{cfg.run_name}] ep {epoch+1} step {global_step}  "
                       f"CE={loss_ce.item():.4f}")
                if cfg.arm == 'A2':
                    msg += f"  ACF²={loss_acf.item():.5f}"
                elif cfg.arm == 'A3':
                    msg += f"  ACF²={loss_acf.item():.5f}  λ={ema_lambda:.2f}"
                msg += f"  lr={lr:.2e}"
                print(msg)
                if wb is not None:
                    wb.log({'train/CE': loss_ce.item(), 'train/ACF2': float(loss_acf),
                            'train/DIR': float(loss_dir) if cfg.lambda_dir > 0 else 0.0,
                            'train/loss': loss.item(), 'train/lr': lr, 'epoch': epoch + 1},
                           step=global_step)

        if rank == 0:
            avg_ce = ep_ce_loss / max(ep_batches, 1)
            avg_acf = ep_acf_loss / max(ep_batches, 1)
            elapsed = format_time(time.time() - t0)

            val_metrics = eval_val(model.module, tokenizer, val_loader, cfg, device,
                                   max_batches=cfg.val_ar_max_batches)
            acf2_gap = val_metrics['acf2_gap']
            if wb is not None:
                wb.log({f'val/{k}': v for k, v in val_metrics.items()
                        if isinstance(v, (int, float))}, step=global_step)
                wb.log({'train/epoch_CE': avg_ce, 'train/epoch_ACF2': avg_acf}, step=global_step)

            sel_acf2 = acf2_gap
            if val_loader2 is not None:
                val_metrics2 = eval_val(model.module, tokenizer, val_loader2, cfg, device)
                acf2_gap2 = val_metrics2['acf2_gap']
                sel_acf2 = 0.5 * (acf2_gap + acf2_gap2)
                val_metrics['acf2_gap_fold2'] = acf2_gap2
                val_metrics['acf2_gap_mean'] = sel_acf2
                val_metrics['acf2_gap_worst'] = max(acf2_gap, acf2_gap2)

            if cfg.ar_val_select:
                ar_metrics = eval_val_ar(model.module, tokenizer, val_loader, cfg, device,
                                         max_batches=cfg.val_ar_max_batches)
                val_metrics.update(ar_metrics)

            print(f"\n── Epoch {epoch+1}/{cfg.epochs} [{elapsed}] ──")
            acf2_str = f"  ACF²={avg_acf:.5f}" if cfg.arm in ('A2', 'A3') else ""
            lam_str  = f"  λ_ema={ema_lambda:.2f}" if cfg.arm == 'A3' else ""
            print(f"  train CE={avg_ce:.4f}{acf2_str}{lam_str}")
            cv_str = (f"  acf2_fold2={val_metrics['acf2_gap_fold2']:.5f}  "
                      f"acf2_mean={sel_acf2:.5f}" if val_loader2 is not None else "")
            print(f"  val   acf2_gap={acf2_gap:.5f}{cv_str}  "
                  f"parkinson_r2={val_metrics['parkinson_r2']:.3f}  "
                  f"price_rankic_path={val_metrics['price_rankic_path']:.4f}  "
                  f"price_ic_path={val_metrics['price_ic_path']:.4f}")
            if cfg.ar_val_select:
                print(f"  val(AR) acf2_gap_ar={val_metrics['acf2_gap_ar']:.5f}  "
                      f"acf2_gap_ar_agg={val_metrics['acf2_gap_ar_agg']:.6f}  "
                      f"price_ic_path_ar={val_metrics['price_ic_path_ar']:.4f}  "
                      f"price_rankic_path_ar={val_metrics['price_rankic_path_ar']:.4f}  "
                      f"return_ic_ar={val_metrics['return_ic_ar']:.4f}  "
                      f"return_rankic_ar={val_metrics['return_rankic_ar']:.4f}  "
                      f"mse_rel_ar={val_metrics['mse_rel_ar']:.6f}")
                print(f"  val(AR) stability: pct_close<=0={val_metrics['val_pct_neg_ar']:.3f}%  "
                      f"pct_sigma2>1={val_metrics['val_pct_s2gt1_ar']:.3f}%  "
                      f"max_sigma2={val_metrics['val_max_s2_ar']:.4g}")

            if val_metrics['price_rankic'] < cfg.rankic_baseline * (1 - cfg.rankic_drop_tol):
                print(f"  [WARN] RankIC {val_metrics['price_rankic']:.4f} dropped >10% below "
                      f"baseline {cfg.rankic_baseline:.4f} — consider stopping")
            if val_metrics['parkinson_r2'] < cfg.parkinson_r2_floor:
                print(f"  [WARN] Parkinson R² {val_metrics['parkinson_r2']:.3f} < "
                      f"{cfg.parkinson_r2_floor} — ACF² loss may be degrading H-L structure")

            ep_record = {
                'epoch': epoch + 1,
                'train_ce': avg_ce,
                'train_acf2': avg_acf,
                **val_metrics,
            }
            if cfg.arm == 'A3':
                ep_record['lambda_ema'] = ema_lambda
            history.append(ep_record)

            if sel_acf2 < best_acf2_gap:
                best_acf2_gap = sel_acf2
                ckpt = save_dir / 'best_model'
                model.module.save_pretrained(str(ckpt))
                tag = 'acf2_mean' if val_loader2 is not None else 'acf2_gap'
                print(f"  [save] best model → {ckpt}  ({tag}={best_acf2_gap:.5f})")

            price_rankic_path = val_metrics['price_rankic_path']
            if np.isfinite(price_rankic_path) and price_rankic_path > best_price_rankic:
                best_price_rankic = price_rankic_path
                ckpt_r = save_dir / 'best_pricerankic_model'
                model.module.save_pretrained(str(ckpt_r))
                print(f"  [save] best price_rankic model → {ckpt_r}  "
                      f"(price_rankic_path={best_price_rankic:.5f})")

            gated = cfg.val_stability_gate and val_metrics['val_pct_neg_ar'] > 0
            if gated:
                print(f"  [gate] epoch not eligible for checkpointing — val rollouts produced "
                      f"non-positive close ({val_metrics['val_pct_neg_ar']:.3f}%)")
            if cfg.ar_val_select and not gated:
                sel_key = 'acf2_gap_ar_agg' if cfg.val_select_agg else 'acf2_gap_ar'
                if val_metrics[sel_key] < best_acf2_gap_ar:
                    best_acf2_gap_ar = val_metrics[sel_key]
                    model.module.save_pretrained(str(save_dir / 'best_arval_model'))
                    print(f"  [save] best AR-val model  ({sel_key}={best_acf2_gap_ar:.5f})")
                if val_metrics['acf2_gap_ar'] < best_arval_pw:
                    best_arval_pw = val_metrics['acf2_gap_ar']
                    model.module.save_pretrained(str(save_dir / 'best_arval_pw_model'))
                if val_metrics['acf2_gap_ar_agg'] < best_arval_agg:
                    best_arval_agg = val_metrics['acf2_gap_ar_agg']
                    model.module.save_pretrained(str(save_dir / 'best_arval_agg_model'))

            if getattr(cfg, 'save_all_epochs', False):
                ckpt_e = save_dir / f'epoch_{epoch + 1}'
                model.module.save_pretrained(str(ckpt_e))

            with open(save_dir / 'history.json', 'w') as f:
                json.dump(history, f, indent=2)

            model.train()

        dist.barrier()

    return {'best_acf2_gap': best_acf2_gap, 'best_price_rankic': best_price_rankic,
            'best_acf2_gap_ar': best_acf2_gap_ar, 'history': history}



def main(cfg: Phase2Config):
    rank, world_size, local_rank = setup_ddp()
    device = torch.device(f'cuda:{local_rank}')
    set_seed(cfg.seed, rank)

    save_dir = Path(cfg.save_dir) / cfg.run_name
    if rank == 0:
        save_dir.mkdir(parents=True, exist_ok=True)
        import dataclasses, subprocess
        cfg_d = dataclasses.asdict(cfg)
        try:
            cfg_d['git_commit'] = subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'], cwd=Path(__file__).parent,
                stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            cfg_d['git_commit'] = 'unknown'
        with open(save_dir / 'config.json', 'w') as f:
            json.dump(cfg_d, f, indent=2)
        print(f"[RUN-CONFIG] run={cfg.run_name} arm={cfg.arm} pred_len={cfg.pred_len} "
              f"lookback={cfg.lookback} stride={cfg.stride} lambda_acf={cfg.lambda_acf} "
              f"k_train={cfg.k_train} lambda_mse={getattr(cfg,'lambda_mse',0.0)} "
              f"data_fraction={cfg.data_fraction} data_dir={cfg.data_dir} freq={cfg.frequency} "
              f"git={cfg_d['git_commit'][:8]}", flush=True)

    train_ds = CryptoWindowDataset(
        data_dir=cfg.data_dir, symbols=cfg.symbols,
        period_start=cfg.train_start, period_end=cfg.train_end,
        lookback=cfg.lookback, pred_len=cfg.pred_len, stride=cfg.stride,
        clip=cfg.clip, zero_vol_amount=cfg.zero_vol_amount,
        data_fraction=cfg.data_fraction, frequency=cfg.frequency,
        data_select=getattr(cfg, 'data_select', None),
        acf_select_lags=getattr(cfg, 'acf_select_lags', 3),
        data_select_seed=cfg.seed,
    )
    val_ds = CryptoWindowDataset(
        data_dir=cfg.data_dir, symbols=cfg.symbols,
        period_start=cfg.val_start, period_end=cfg.val_end,
        lookback=cfg.lookback, pred_len=cfg.pred_len, stride=cfg.stride,
        clip=cfg.clip, zero_vol_amount=cfg.zero_vol_amount,
        data_fraction=1.0, frequency=cfg.frequency,
    )

    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, sampler=train_sampler,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, sampler=val_sampler,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=False,
    )

    val_loader2 = None
    if cfg.val2_start and cfg.val2_end:
        val_ds2 = CryptoWindowDataset(
            data_dir=cfg.data_dir, symbols=cfg.symbols,
            period_start=cfg.val2_start, period_end=cfg.val2_end,
            lookback=cfg.lookback, pred_len=cfg.pred_len, stride=cfg.stride,
            clip=cfg.clip, zero_vol_amount=cfg.zero_vol_amount, frequency=cfg.frequency,
            data_fraction=1.0,
        )
        val_sampler2 = DistributedSampler(val_ds2, num_replicas=world_size, rank=rank, shuffle=False)
        val_loader2 = DataLoader(
            val_ds2, batch_size=cfg.batch_size, sampler=val_sampler2,
            num_workers=cfg.num_workers, pin_memory=True, drop_last=False,
        )

    tokenizer = KronosTokenizer.from_pretrained(cfg.tokenizer_path)
    tokenizer.eval().to(device)
    for p in tokenizer.parameters():
        p.requires_grad_(False)

    predictor = Kronos.from_pretrained(cfg.predictor_path)
    predictor.to(device)
    predictor = DDP(predictor, device_ids=[local_rank], find_unused_parameters=False)

    if rank == 0:
        print(f"[{cfg.run_name}] predictor: {get_model_size(predictor.module)}")
        print(f"[{cfg.run_name}] train windows: {len(train_ds)}, val windows: {len(val_ds)}")

    result = train(predictor, tokenizer, train_loader, val_loader,
                   cfg, save_dir, rank, device, val_loader2=val_loader2)

    if rank == 0:
        with open(save_dir / 'summary.json', 'w') as f:
            json.dump({'run_name': cfg.run_name, **result}, f, indent=2, default=str)
        print(f"[{cfg.run_name}] best_price_rankic={result.get('best_price_rankic', float('nan')):.5f}")
        print(f"\n[{cfg.run_name}] done. best_acf2_gap={result['best_acf2_gap']:.5f}")

    cleanup_ddp()


if __name__ == '__main__':
    if 'WORLD_SIZE' not in os.environ:
        raise RuntimeError("Launch with torchrun: "
                           "torchrun --standalone --nproc_per_node=N finetune/train_predictor.py --arm A1")

    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', default='A1', choices=arm_preset())
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--k_train', type=int, default=None)
    parser.add_argument('--lambda_acf', type=float, default=None)
    parser.add_argument('--tau_lag', type=float, default=None)
    parser.add_argument('--data_fraction', type=float, default=None)
    parser.add_argument('--data_select', default=None)
    parser.add_argument('--data_select_seed', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--save_dir', default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--run_name', default=None)
    parser.add_argument('--save_all_epochs', action='store_true', default=None)
    parser.add_argument('--wandb', dest='use_wandb', action='store_true', default=None,
                        help='log training curves to Weights & Biases (needs `wandb login`)')
    args = parser.parse_args()

    overrides = {k: v for k, v in vars(args).items() if v is not None and k != 'arm'}
    cfg = arm_preset(args.arm, **overrides)
    main(cfg)

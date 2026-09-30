"""
Phase 2 fine-tuning ported to the Chronos-t5-small backbone: CE loss (arm A1) or
CE + aggregate ACF^2 (arm A2), via a new differentiable AR soft-rollout written for
T5's flat bin-softmax tokenizer (MeanScaleUniformBins) -- Chronos has no two-level
BSQ head to hook into like Kronos, so this rollout is architecturally new, not a
port of finetune/train_predictor.py's ar_rollout_prices.

Single-GPU: Chronos-t5-small is ~40M dense params (vs Kronos-base ~100M with a
24-layer trunk), so the fixed recipe's effective batch (256 for CE, 32 for ACF^2)
is reached directly via --batch_size on one V100, no DDP/grad_accum needed.

Gradient path for ACF^2 (A2), the T5 analogue of Kronos's ar_rollout_prices:
    T5 decoder step logits -> Gumbel-softmax over the 4096-bin vocab
    -> soft embedding (probs @ shared.weight) fed back as next decoder input
    -> ... -> soft "close" value (probs @ bin_centers) * scale -> ACF^2 loss
Encoder + decoder weights are fine-tuned; the tokenizer (MeanScaleUniformBins) has
no learnable parameters, it only defines the fixed bin centers used to soft-decode.

Launch:
    python finetune/train_predictor_chronos.py --dataset crypto --arm A1
    python finetune/train_predictor_chronos.py --dataset crypto --arm A2
"""

from __future__ import annotations
import argparse
import json
import math
import os as _os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import sys
sys.path.append(str(Path(__file__).parent.parent))
from finetune.dataset import CryptoWindowDataset
from finetune.utils.training_utils import set_seed, format_time
from eval.metrics import price_channel_ic, acf_sq_returns, _rankdata, _pearson
from finetune.train_predictor import acf2_loss

CLOSE_IDX = 3


def label_transform(tokenizer, label: torch.Tensor, scale: torch.Tensor):
    """
    Same as tokenizer.label_input_transform, minus its `length == config.
    prediction_length` assert. That assert is a library sanity check, not an
    architectural constraint -- T5 is length-agnostic on decoder targets (per-
    position cross-entropy) -- but our pred_len (32) differs from Chronos's
    default prediction_length (64), so the public method would reject us.
    """
    token_ids, attention_mask, _ = tokenizer._input_transform(context=label, scale=scale)
    if tokenizer.config.use_eos_token:
        token_ids, attention_mask = tokenizer._append_eos_token(token_ids, attention_mask)
    return token_ids, attention_mask

_SCRATCH = _os.path.expandvars(_os.environ.get('VCA_SCRATCH', '/scratch/$USER/Kronos'))

DATASETS = {
    'crypto': dict(
        data_dir=f'{_SCRATCH}/data', frequency='15m',
        symbols=None,
        train_start='2021-01-01', train_end='2023-12-31',
        val_start='2024-01-01', val_end='2024-06-30',
        lookback=160, pred_len=32, stride=32,
    ),
    'csi300': dict(
        data_dir=f'{_SCRATCH}/data_csi300_piT', frequency='1d',
        symbols=None,
        train_start='2015-01-01', train_end='2017-12-31',
        val_start='2018-01-01', val_end='2018-12-31',
        lookback=160, pred_len=32, stride=8,
    ),
    'csi500': dict(
        data_dir=f'{_SCRATCH}/data_csi500_piT', frequency='1d',
        symbols=None,
        train_start='2015-01-01', train_end='2017-12-31',
        val_start='2018-01-01', val_end='2018-12-31',
        lookback=160, pred_len=32, stride=8,
    ),
}


def resolve_dataset_cfg(name: str) -> dict:
    from finetune.config import CRYPTO_TOP20, CSI300_PIT_SYMBOLS, CSI500_PIT_SYMBOLS
    cfg = dict(DATASETS[name])
    cfg['symbols'] = {
        'crypto': CRYPTO_TOP20, 'csi300': CSI300_PIT_SYMBOLS, 'csi500': CSI500_PIT_SYMBOLS,
    }[name]
    return cfg



def build_bin_value_vector(tokenizer, n_tokens: int, device: torch.device) -> torch.Tensor:
    """
    (n_tokens,) vector mapping EVERY token id (including pad/eos) to a scaled
    bin value, reproducing MeanScaleUniformBins.output_transform's index formula
    exactly: idx = clamp(token_id - n_special_tokens - 1, 0, len(centers)-1).
    Real decoding never special-cases pad/eos differently, so neither do we --
    this lets the soft mixture use the SAME formula for every vocab entry.
    """
    n_special = tokenizer.config.n_special_tokens
    centers = tokenizer.centers.to(device)
    ids = torch.arange(n_tokens, device=device)
    idx = torch.clamp(ids - n_special - 1, 0, len(centers) - 1)
    return centers[idx]


def soft_ar_rollout_close(
    t5,
    encoder_hidden: torch.Tensor,
    encoder_mask: torch.Tensor,
    scale: torch.Tensor,
    bin_value: torch.Tensor,
    H: int,
    tau: float,
    hard: bool,
) -> torch.Tensor:
    """
    Differentiable autoregressive rollout over T5's flat vocab, generating H steps
    one at a time and feeding the model's OWN Gumbel-soft token back as the next
    decoder input embedding -- the direct T5 analogue of Kronos's ar_rollout_prices,
    but simpler: one flat softmax per step instead of a two-level (s1, s2) head.
    """
    B = encoder_hidden.shape[0]
    device = encoder_hidden.device
    start_id = t5.config.decoder_start_token_id
    dec_embed = t5.shared(torch.full((B, 1), start_id, dtype=torch.long, device=device))

    soft_closes = []
    for _ in range(H):
        dec_out = t5.decoder(
            inputs_embeds=dec_embed,
            encoder_hidden_states=encoder_hidden,
            encoder_attention_mask=encoder_mask,
            use_cache=False,
        ).last_hidden_state
        last = dec_out[:, -1]
        if t5.config.tie_word_embeddings:
            last = last * (t5.model_dim ** -0.5)
        logits = t5.lm_head(last)
        logits = torch.nan_to_num(logits, nan=0.0, posinf=30.0, neginf=-30.0)
        soft = F.gumbel_softmax(logits, tau=tau, hard=hard, dim=-1)

        soft_closes.append((soft * bin_value.unsqueeze(0)).sum(dim=-1))
        next_embed = (soft @ t5.shared.weight).unsqueeze(1)
        dec_embed = torch.cat([dec_embed, next_embed], dim=1)

    soft_scaled = torch.stack(soft_closes, dim=1)
    return soft_scaled * scale.unsqueeze(1)


@torch.no_grad()
def greedy_ar_rollout_close(t5, bin_value, encoder_hidden, encoder_mask, scale, H: int) -> torch.Tensor:
    """Non-differentiable greedy (argmax) rollout for val checkpoint selection."""
    B = encoder_hidden.shape[0]
    device = encoder_hidden.device
    start_id = t5.config.decoder_start_token_id
    dec_embed = t5.shared(torch.full((B, 1), start_id, dtype=torch.long, device=device))
    gen_ids = []
    for _ in range(H):
        dec_out = t5.decoder(
            inputs_embeds=dec_embed, encoder_hidden_states=encoder_hidden,
            encoder_attention_mask=encoder_mask, use_cache=False,
        ).last_hidden_state
        last = dec_out[:, -1]
        if t5.config.tie_word_embeddings:
            last = last * (t5.model_dim ** -0.5)
        logits = t5.lm_head(last)
        ids = logits.argmax(-1)
        gen_ids.append(ids)
        next_embed = t5.shared(ids).unsqueeze(1)
        dec_embed = torch.cat([dec_embed, next_embed], dim=1)
    ids = torch.stack(gen_ids, dim=1)
    return bin_value[ids] * scale.unsqueeze(1)


def _sample_ids_from_logits(logits: torch.Tensor, top_k: int, top_p: float) -> torch.Tensor:
    """Top-k/top-p filtered multinomial sampling -- the same do_sample path HF's
    ChronosPipeline.predict uses at test time (eval/inference_chronos.py), so the
    val-time gate below samples from the SAME decoding regime that determines
    actual test-time stability instead of a deterministic argmax."""
    if top_k and top_k > 0:
        kth = torch.topk(logits, top_k, dim=-1).values[:, -1].unsqueeze(-1)
        logits = logits.masked_fill(logits < kth, float('-inf'))
    if top_p and top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_mask = (cum_probs - F.softmax(sorted_logits, dim=-1)) > top_p
        sorted_logits = sorted_logits.masked_fill(sorted_mask, float('-inf'))
        logits = torch.full_like(logits, float('-inf')).scatter(-1, sorted_idx, sorted_logits)
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


@torch.no_grad()
def sample_ar_rollout_close(t5, bin_value, encoder_hidden, encoder_mask, scale, H: int,
                             temperature: float, top_k: int, top_p: float) -> torch.Tensor:
    """Temperature/top-p sampled AR rollout matching eval/inference_chronos.py's test-time
    decoding (ChronosPipeline.predict(temperature=..., top_k=..., top_p=...)) -- used only
    for the val stability gate, NOT for acf2_gap_agg checkpoint selection (still greedy,
    unchanged), since the gate needs to predict test-time AR-rollout explosion and the
    test-time rollout is stochastic, not greedy (PLAN.md 2026-09-10 finding)."""
    B = encoder_hidden.shape[0]
    device = encoder_hidden.device
    start_id = t5.config.decoder_start_token_id
    dec_embed = t5.shared(torch.full((B, 1), start_id, dtype=torch.long, device=device))
    gen_ids = []
    for _ in range(H):
        dec_out = t5.decoder(
            inputs_embeds=dec_embed, encoder_hidden_states=encoder_hidden,
            encoder_attention_mask=encoder_mask, use_cache=False,
        ).last_hidden_state
        last = dec_out[:, -1]
        if t5.config.tie_word_embeddings:
            last = last * (t5.model_dim ** -0.5)
        logits = t5.lm_head(last) / temperature
        ids = _sample_ids_from_logits(logits, top_k, top_p)
        gen_ids.append(ids)
        next_embed = t5.shared(ids).unsqueeze(1)
        dec_embed = torch.cat([dec_embed, next_embed], dim=1)
    ids = torch.stack(gen_ids, dim=1)
    return bin_value[ids] * scale.unsqueeze(1)



def _rankic(pred: np.ndarray, true: np.ndarray) -> float:
    try:
        return float(_pearson(_rankdata(pred), _rankdata(true)))
    except Exception:
        return float('nan')


@torch.no_grad()
def eval_val(t5, tokenizer, bin_value, val_loader, device, K_eval: int, max_batches: int,
             gate_n_samples: int = 5, gate_temperature: float = 0.6,
             gate_top_k: int = 0, gate_top_p: float = 0.9) -> dict:
    t5.eval()
    pred_acf_all, true_acf_all = [], []
    pic_all, pric_all = [], []
    ret_pred, ret_true = [], []
    n_neg = n_roll = 0
    n_neg_sampled = n_roll_sampled = 0

    for nb, batch in enumerate(val_loader):
        if nb >= max_batches:
            break
        ctx_close = (batch['x_norm'][:, :, CLOSE_IDX] * batch['x_std'][:, CLOSE_IDX:CLOSE_IDX + 1]
                     + batch['x_mean'][:, CLOSE_IDX:CLOSE_IDX + 1])
        y_close = batch['y_raw'][:, :, CLOSE_IDX]
        H = y_close.shape[1]

        ids, mask, scale = tokenizer.context_input_transform(ctx_close)
        ids, mask, scale = ids.to(device), mask.to(device), scale.to(device)
        enc = t5.encoder(input_ids=ids, attention_mask=mask).last_hidden_state
        pred_close = greedy_ar_rollout_close(t5, bin_value, enc, mask, scale, H)

        for _ in range(gate_n_samples):
            sampled_close = sample_ar_rollout_close(
                t5, bin_value, enc, mask, scale, H,
                temperature=gate_temperature, top_k=gate_top_k, top_p=gate_top_p)
            mins = sampled_close.amin(dim=1).cpu().numpy()
            n_roll_sampled += mins.shape[0]
            n_neg_sampled += int((mins <= 0).sum())

        pred_np, true_np = pred_close.cpu().numpy(), y_close.numpy()
        for i in range(pred_np.shape[0]):
            pc, tc = pred_np[i], true_np[i]
            n_roll += 1
            if float(np.min(pc)) <= 0:
                n_neg += 1
            pred_acf_all.append(acf_sq_returns(pc, K_eval))
            true_acf_all.append(acf_sq_returns(tc, K_eval))
            pp = np.full((H, 4), np.nan, dtype=np.float32); pp[:, CLOSE_IDX] = pc
            tp = np.full((H, 4), np.nan, dtype=np.float32); tp[:, CLOSE_IDX] = tc
            pic, pric = price_channel_ic(pp, tp, rank=False), price_channel_ic(pp, tp, rank=True)
            if np.isfinite(pic):
                pic_all.append(pic)
            if np.isfinite(pric):
                pric_all.append(pric)
            ret_pred.append(float(pc[-1] / max(pc[0], 1e-16) - 1))
            ret_true.append(float(tc[-1] / max(tc[0], 1e-16) - 1))

    pred_acf, true_acf = np.array(pred_acf_all), np.array(true_acf_all)
    w = np.exp(-np.arange(1, K_eval + 1) / 2.0)
    acf2_gap_agg = float((w * (pred_acf.mean(0) - true_acf.mean(0)) ** 2).sum())
    acf2_gap_pw = float(np.mean((pred_acf - true_acf) ** 2))
    return {
        'acf2_gap_agg': acf2_gap_agg,
        'acf2_gap_pw': acf2_gap_pw,
        'price_ic_path': float(np.mean(pic_all)) if pic_all else float('nan'),
        'price_rankic_path': float(np.mean(pric_all)) if pric_all else float('nan'),
        'return_rankic': _rankic(np.array(ret_pred), np.array(ret_true)),
        'val_pct_neg': 100.0 * n_neg / max(n_roll, 1),
        'val_pct_neg_sampled': 100.0 * n_neg_sampled / max(n_roll_sampled, 1),
        'n_windows': n_roll,
    }



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', choices=list(DATASETS), required=True)
    ap.add_argument('--arm', choices=['A1', 'A2', 'MSE'], required=True,
                     help='A1: CE only. A2: CE + aggregate ACF^2 (Ours). MSE: CE + relative-MSE '
                          'on the SAME AR rollout (the AR-rollout ablation baseline -- isolates '
                          'whether the effect comes from the ACF^2 term or just the rollout).')
    ap.add_argument('--model_id', default='amazon/chronos-t5-small')
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--batch_size', type=int, default=None,
                     help='default: 256 for A1 (CE), 32 for A2/MSE (AR rollout) -- matches the '
                          'fixed recipe effective batch (Kronos reaches this via 4 GPUs; Chronos-'
                          't5-small is small enough to hit it directly on one GPU).')
    ap.add_argument('--lr', type=float, default=5e-5)
    ap.add_argument('--lambda_acf', type=float, default=10.0)
    ap.add_argument('--lambda_mse', type=float, default=200.0,
                     help='MSE arm only. Matches Kronos preset (e.g. CRYPTO_MSE_H8): lambda_acf=0, '
                          'lambda_mse=200 on the SAME full-AR rollout.')
    ap.add_argument('--pred_len', type=int, default=None,
                     help='override dataset default pred_len=32 (e.g. 16 for the H16 scaling row). '
                          'stride is left at the dataset default unless --stride is also given.')
    ap.add_argument('--stride', type=int, default=None,
                     help='override dataset default stride; e.g. crypto ties stride=pred_len '
                          '(16 at H16, matching the native-Kronos H16 recipe), equity keeps stride=8.')
    ap.add_argument('--k_train', type=int, default=1)
    ap.add_argument('--tau_lag', type=float, default=2.0)
    ap.add_argument('--gumbel_tau_start', type=float, default=0.5)
    ap.add_argument('--gumbel_tau_end', type=float, default=0.1)
    ap.add_argument('--gumbel_hard', action='store_true', default=True)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--val_max_batches', type=int, default=20,
                     help='AR val rollout is O(H) sequential decoder passes per batch -- capped '
                          'for per-epoch selection speed (Kronos default covers the full val set; '
                          'this is a scope simplification, see PLAN.md Phase B entry).')
    ap.add_argument('--log_interval', type=int, default=50)
    ap.add_argument('--val_stability_gate', action='store_true', default=False,
                     help='skip checkpointing an epoch whose val AR rollout produced any '
                          'non-positive close (val_pct_neg_sampled > 0) -- port of the native-'
                          'Kronos gate (finetune/train_predictor.py), applied identically to '
                          'all arms per the ex-ante rule (PLAN.md 2026-08-27b): if enabled, '
                          'enable for every arm in the comparison, not just the one that '
                          'explodes. Gates on the TEMPERATURE-SAMPLED rollout (val_pct_neg_'
                          'sampled), not the deterministic greedy one (val_pct_neg) -- the '
                          'greedy pilot (2026-09-10) found the two regimes disagree badly '
                          '(greedy flags 41-90% negative on epochs that are clean under '
                          'test-time sampling), so gating on greedy mis-selects checkpoints.')
    ap.add_argument('--gate_n_samples', type=int, default=5,
                     help='stochastic rollouts averaged per val batch for the gate signal '
                          '(val_pct_neg_sampled) -- reduces single-draw sampling noise.')
    ap.add_argument('--gate_temperature', type=float, default=0.6,
                     help='matches the "forecast" run temperature in the eval config YAML '
                          '(e.g. configs/csi300_piT_h16_chronos.yaml) so the gate reflects '
                          'the same decode regime test-time eval actually uses.')
    ap.add_argument('--gate_top_k', type=int, default=0)
    ap.add_argument('--gate_top_p', type=float, default=0.9)
    ap.add_argument('--save_all_epochs', action='store_true', default=False,
                     help='also save every epoch checkpoint (epoch_N.pt), for oracle test-epoch '
                          'selection studies -- comparing selection rules after the fact.')
    ap.add_argument('--run_name', default=None)
    ap.add_argument('--save_dir', default=None)
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(args.seed)

    if args.batch_size is None:
        args.batch_size = 256 if args.arm == 'A1' else 32
    if args.run_name is None:
        args.run_name = f'chronos_{args.dataset}_{args.arm}_s{args.seed}'
    if args.save_dir is None:
        args.save_dir = f'{_SCRATCH}/outputs/phase2_chronos/{args.run_name}'
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    for stale in list(save_dir.glob('best_*.pt')) + list(save_dir.glob('epoch_*.pt')):
        stale.unlink()
    history_path = save_dir / 'history.json'
    if history_path.exists():
        history_path.unlink()

    dcfg = resolve_dataset_cfg(args.dataset)
    if args.pred_len is not None:
        dcfg['pred_len'] = args.pred_len
    if args.stride is not None:
        dcfg['stride'] = args.stride
    print(f"[chronos-p2] dataset={args.dataset} arm={args.arm} run={args.run_name} "
          f"pred_len={dcfg['pred_len']} stride={dcfg['stride']} batch_size={args.batch_size} device={device}")

    from chronos import BaseChronosPipeline
    pipe = BaseChronosPipeline.from_pretrained(
        args.model_id, device_map=str(device), torch_dtype=torch.float32)
    tokenizer = pipe.tokenizer
    t5 = pipe.model.model.to(device)
    t5.train()
    n_params = sum(p.numel() for p in t5.parameters() if p.requires_grad)
    print(f"[chronos-p2] model params (trainable): {n_params/1e6:.1f}M")

    bin_value = build_bin_value_vector(tokenizer, t5.config.vocab_size, device)

    train_ds = CryptoWindowDataset(
        data_dir=dcfg['data_dir'], symbols=dcfg['symbols'],
        period_start=dcfg['train_start'], period_end=dcfg['train_end'],
        lookback=dcfg['lookback'], pred_len=dcfg['pred_len'], stride=dcfg['stride'],
        frequency=dcfg['frequency'],
    )
    val_ds = CryptoWindowDataset(
        data_dir=dcfg['data_dir'], symbols=dcfg['symbols'],
        period_start=dcfg['val_start'], period_end=dcfg['val_end'],
        lookback=dcfg['lookback'], pred_len=dcfg['pred_len'], stride=dcfg['stride'],
        frequency=dcfg['frequency'],
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=2, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    optimizer = torch.optim.AdamW(t5.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)
    total_steps = max(1, len(train_loader) * args.epochs)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=total_steps, pct_start=0.03, div_factor=10)

    best_val_ce = float('inf')
    best_acf2_gap_agg = float('inf')
    arval_agg_saved = False
    model_saved = False
    fallback_epoch, fallback_pct_neg, fallback_acf2_gap = None, float('inf'), float('inf')
    fallback_epoch_a1, fallback_pct_neg_a1, fallback_ce_a1 = None, float('inf'), float('inf')
    history = []
    global_step = 0
    t0 = time.time()

    for epoch in range(args.epochs):
        t5.train()
        ep_ce = ep_acf = 0.0
        ep_batches = 0

        for batch in train_loader:
            ctx_close_cpu = (batch['x_norm'][:, :, CLOSE_IDX] * batch['x_std'][:, CLOSE_IDX:CLOSE_IDX + 1]
                              + batch['x_mean'][:, CLOSE_IDX:CLOSE_IDX + 1])
            y_close_cpu = batch['y_raw'][:, :, CLOSE_IDX]
            H = y_close_cpu.shape[1]

            ctx_ids, ctx_mask, scale_cpu = tokenizer.context_input_transform(ctx_close_cpu)
            label_ids, label_mask = label_transform(tokenizer, y_close_cpu, scale_cpu)
            ctx_ids, ctx_mask = ctx_ids.to(device), ctx_mask.to(device)
            scale = scale_cpu.to(device)
            label_ids, label_mask = label_ids.to(device), label_mask.to(device)
            y_close = y_close_cpu.to(device)
            labels = label_ids.masked_fill(~label_mask, -100)

            enc_hidden = t5.encoder(input_ids=ctx_ids, attention_mask=ctx_mask).last_hidden_state
            out = t5(encoder_outputs=(enc_hidden,), attention_mask=ctx_mask, labels=labels)
            loss_ce = out.loss

            if args.arm in ('A2', 'MSE'):
                tau = args.gumbel_tau_start * (
                    (args.gumbel_tau_end / args.gumbel_tau_start) ** min(global_step / total_steps, 1.0))
                pred_close = soft_ar_rollout_close(
                    t5, enc_hidden, ctx_mask, scale, bin_value, H, tau, args.gumbel_hard)
                if args.arm == 'A2':
                    pred_prices = torch.zeros(pred_close.shape[0], H, 4, device=device)
                    pred_prices[:, :, CLOSE_IDX] = pred_close
                    loss_acf = acf2_loss(pred_prices, y_close, args.k_train, args.tau_lag, aggregate=True)
                    loss = loss_ce + args.lambda_acf * loss_acf
                    ep_acf += float(loss_acf.detach())
                else:
                    loss_mse = ((pred_close / y_close.clamp_min(1e-6) - 1.0) ** 2).mean()
                    loss = loss_ce + args.lambda_mse * loss_mse
                    ep_acf += float(loss_mse.detach())
            else:
                loss = loss_ce

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(t5.parameters(), 3.0)
            optimizer.step()
            scheduler.step()

            ep_ce += float(loss_ce.detach())
            ep_batches += 1
            global_step += 1
            if global_step % args.log_interval == 0:
                msg = f"[e{epoch} s{global_step}] ce={ep_ce/ep_batches:.4f}"
                if args.arm in ('A2', 'MSE'):
                    tag = 'acf' if args.arm == 'A2' else 'mse'
                    msg += f" {tag}={ep_acf/ep_batches:.5f}"
                msg += f" lr={scheduler.get_last_lr()[0]:.2e} elapsed={format_time(time.time()-t0)}"
                print(msg, flush=True)

        val = eval_val(t5, tokenizer, bin_value, val_loader, device, K_eval=15, max_batches=args.val_max_batches,
                       gate_n_samples=args.gate_n_samples, gate_temperature=args.gate_temperature,
                       gate_top_k=args.gate_top_k, gate_top_p=args.gate_top_p)
        rec = {'epoch': epoch, 'train_ce': ep_ce / max(ep_batches, 1),
               'train_acf': ep_acf / max(ep_batches, 1), **val}
        history.append(rec)
        print(f"[epoch {epoch}] {json.dumps(rec)}", flush=True)

        gated = args.val_stability_gate and val['val_pct_neg_sampled'] > 0
        if gated:
            print(f"  [gate] epoch {epoch} not eligible for checkpointing -- sampled val "
                  f"rollouts produced non-positive close "
                  f"({val['val_pct_neg_sampled']:.3f}%, greedy was {val['val_pct_neg']:.3f}%)")
        best_val_ce = min(best_val_ce, rec['train_ce'])
        if args.arm in ('A2', 'MSE') and not gated and val['acf2_gap_agg'] < best_acf2_gap_agg:
            best_acf2_gap_agg = val['acf2_gap_agg']
            torch.save(t5.state_dict(), save_dir / 'best_arval_agg_model.pt')
            arval_agg_saved = True
            print(f"[epoch {epoch}] new best acf2_gap_agg={best_acf2_gap_agg:.5f} -> saved")
        if args.arm in ('A2', 'MSE') and gated and (
                val['val_pct_neg_sampled'] < fallback_pct_neg or (
                    val['val_pct_neg_sampled'] == fallback_pct_neg and val['acf2_gap_agg'] < fallback_acf2_gap)):
            fallback_epoch = epoch
            fallback_pct_neg = val['val_pct_neg_sampled']
            fallback_acf2_gap = val['acf2_gap_agg']
            torch.save(t5.state_dict(), save_dir / '_fallback_arval_agg_model.pt')
        if args.arm == 'A1' and not gated and epoch == int(np.argmin(
                [h['train_ce'] for h in history])):
            torch.save(t5.state_dict(), save_dir / 'best_model.pt')
            model_saved = True
            print(f"[epoch {epoch}] new best val_ce={best_val_ce:.5f} -> saved")
        if args.arm == 'A1' and gated and (
                val['val_pct_neg_sampled'] < fallback_pct_neg_a1 or (
                    val['val_pct_neg_sampled'] == fallback_pct_neg_a1 and rec['train_ce'] < fallback_ce_a1)):
            fallback_epoch_a1 = epoch
            fallback_pct_neg_a1 = val['val_pct_neg_sampled']
            fallback_ce_a1 = rec['train_ce']
            torch.save(t5.state_dict(), save_dir / '_fallback_best_model.pt')
        if args.save_all_epochs:
            torch.save(t5.state_dict(), save_dir / f'epoch_{epoch}.pt')

    fallback_path = save_dir / '_fallback_arval_agg_model.pt'
    if args.arm in ('A2', 'MSE') and not arval_agg_saved:
        if fallback_epoch is not None:
            fallback_path.rename(save_dir / 'best_arval_agg_model.pt')
            print(f"[fallback] --val_stability_gate rejected every epoch this run -- saving "
                  f"least-unstable epoch {fallback_epoch} (val_pct_neg_sampled="
                  f"{fallback_pct_neg:.3f}%, acf2_gap_agg={fallback_acf2_gap:.5f}) as "
                  f"best_arval_agg_model instead of leaving no checkpoint")
        else:
            print("[fallback] no epoch produced a val_pct_neg_sampled reading -- "
                  "best_arval_agg_model was NOT written this run")
    if fallback_path.exists():
        fallback_path.unlink()

    fallback_path_a1 = save_dir / '_fallback_best_model.pt'
    if args.arm == 'A1' and not model_saved:
        if fallback_epoch_a1 is not None:
            fallback_path_a1.rename(save_dir / 'best_model.pt')
            print(f"[fallback] --val_stability_gate rejected every epoch this run -- saving "
                  f"least-unstable epoch {fallback_epoch_a1} (val_pct_neg_sampled="
                  f"{fallback_pct_neg_a1:.3f}%, train_ce={fallback_ce_a1:.5f}) as "
                  f"best_model instead of leaving no checkpoint")
        else:
            print("[fallback] no epoch produced a val_pct_neg_sampled reading -- "
                  "best_model was NOT written this run")
    if fallback_path_a1.exists():
        fallback_path_a1.unlink()

    (save_dir / 'history.json').write_text(json.dumps(history, indent=2))
    print(f"===== done in {format_time(time.time()-t0)} =====")


if __name__ == '__main__':
    main()

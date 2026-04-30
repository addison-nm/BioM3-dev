"""GDPO single-GPU trainer for BioM3 Stage 3.

Group Diffusion Policy Optimization (Rojas et al., ICLR 2026 — arXiv
2510.08554v3): a sibling of GRPO that replaces diffu-GRPO's biased
one-step token-level mean-field log-prob with a sequence-level ELBO
estimated by a Semi-deterministic Monte Carlo (SDMC) scheme.

For ProteoScribe (an order-agnostic absorbing-state diffusion model)
the SDMC integrand is exactly the per-time conditional log-prob over
currently-masked positions — the same quantity the existing
``Stage3.transformer_training_helper`` utilities already compute for
training. We reuse those helpers verbatim and only build the
quadrature scaffolding + sequence-level PPO-clip loss on top.

See docs/.claude_prompts/PROMPT_grpo_integration.md and
plan-this-out-in-serene-riddle.md for the design doc.
"""

import copy
import json
import os
import time
from argparse import Namespace
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import biom3.Stage1.preprocess as S1prep
import biom3.Stage3.animation_tools as S3ani
import biom3.Stage3.sampling_analysis as S3sample
import biom3.Stage3.transformer_training_helper as S3train
from biom3.Stage3.run_ProteoScribe_sample import (
    _build_initial_mask_state,
    _resolve_fill_token_id,
    load_pre_unmask_config,
)
from biom3.backend.device import setup_logger
from biom3.rl.rollout import RolloutPool
from biom3.rl.grpo import (
    NUM_CLASSES,
    PAD_ID,
    START_ID,
    END_ID,
    TOK2ID,
    TOKENS,
    VALID_AA,
    _PromptEncoder,
    decode_tokens,
    diffusion_rollout,
    load_prompts,
)
from biom3.rl.io import (
    load_facilitator_frozen,
    load_pencl_frozen,
    load_proteoscribe_trainable,
)

logger = setup_logger(__name__)


# Mask token id for the absorbing-state diffusion. NOT the PAD id.
# transformer_training_helper.mask_realizations writes 0 at masked
# positions; sampling_analysis.batch_generate_denoised_sampled starts
# the rollout from torch.zeros(...) ⇒ all-mask ⇒ id 0.
MASK_ID = 0


@dataclass
class GDPOConfig:
    output_dir: str = "./gdpo_output"
    learning_rate: float = 1e-5
    weight_decay: float = 1e-6
    beta: float = 0.01           # KL coefficient
    eps: float = 0.20            # PPO clip
    num_generations: int = 4     # G
    batch_size: int = 1          # prompts per gradient step
    steps: int = 200
    save_steps: int = 50
    max_grad_norm: float = 1.0
    seed: int = 42

    # SDMC quadrature for the ELBO integral over t ∈ (0, 1].
    # paper_t (fraction masked) maps to model idx = round((1 - paper_t) * L).
    n_quadrature: int = 3
    quadrature_grid: str = "uniform"   # "uniform" midpoints | "explicit"
    # If quadrature_grid == "explicit", use these t_n values (and weights if
    # provided; else uniform 1/N). Floats in (0, 1].
    quadrature_points: Optional[List[float]] = None
    quadrature_weights: Optional[List[float]] = None
    inner_mc: int = 1                  # K_inner: MC mask samples per t_n
    eps_t: float = 1e-3                # clamp t_n away from 0 in 1/t

    # KL term: "tokenwise_k3" reuses the cheap GRPO-style k3 estimator on a
    # single fully-masked forward; "sdmc" computes (ELBO_ref − ELBO_new) at
    # the same SDMC grid (one extra ELBO pass through π_ref).
    kl_estimator: str = "tokenwise_k3"

    # Paper Alg. 1 line 3: snapshot π_old per outer iteration, separate
    # from the frozen π_ref. False ⇒ collapse to GRPO behavior.
    use_old_policy_snapshot: bool = True

    # Paper eq. (6) uses unnormalized advantage R - mean(R). Switch on if
    # you want classic GRPO normalization for parity testing.
    advantage_normalize: bool = False

    # Per-step verbose dump (sequences, rewards, advantages, ELBOs,
    # SDMC mask visualizations) appended to {output_dir}/debug.out.
    # Useful for sanity-checking corruption sampling and reward signal;
    # cheap to write but produces ~tens of MB for long runs.
    debug_log: bool = True

    # Wrap each per-corruption forward in torch.utils.checkpoint so
    # activations are recomputed in backward. With L=1024 and N≥7 the
    # cumulative autograd graph from the trainable-policy ELBO will OOM
    # otherwise. Roughly halves runtime for the trainable ELBO; no-op
    # for the no-grad ELBOs (π_old, π_ref).
    gradient_checkpoint: bool = True

    # Pre-unmask: keep the architectural sequence_length L (the value the
    # model was trained at, e.g. 1024) but only diffuse over the first D
    # positions (the "diffusion budget"). Positions [D, L) are pre-filled
    # with PAD and never touched. Useful when the target proteins (e.g.
    # SH3 ≈ 60 AA) are much shorter than L; collapses both rollout and
    # ELBO compute by a factor ≈ L/D. Mirrors the existing flag wired
    # into biom3.Stage3.run_ProteoScribe_sample.
    pre_unmask: bool = False
    pre_unmask_config: Optional[str] = None  # path to JSON with
                                             # strategy/fill_with/diffusion_budget

    # Multi-device rollout: replicate the trainable policy onto each
    # device in this list, dispatch rollout chunks in parallel via
    # threads, gather to devices[0]. ``None`` (default) preserves the
    # single-device behavior. ``["auto"]`` selects all visible XPU
    # tiles via torch.xpu.device_count(). ELBO and gradient updates
    # remain single-device on devices[0].
    rollout_devices: Optional[List[str]] = None


# ─────────────────────────────────────────────────────────────────────────────
# Quadrature
# ─────────────────────────────────────────────────────────────────────────────


def _build_grid(
    cfg: GDPOConfig,
    L: int,
    device: torch.device,
):
    """Build the SDMC time grid.

    ``L`` here is the **active diffusion budget** D (= ``cfg3.diffusion_steps``).
    Under pre_unmask the architectural sequence length may be larger;
    only the first D positions are actually diffused over, so the time
    embedding's domain is [0, D-1].

    Returns:
        idx_grid: LongTensor (N,) — model-time indices in [0, L-1] where
            ``idx`` = number of revealed positions. idx = 0 ↔ fully masked
            ↔ paper_t = 1; idx = L-1 ↔ paper_t ≈ 1/L.
        t_floats: FloatTensor (N,) — paper_t ∈ (0, 1] used to scale the
            ``1/t`` factor in the SDMC integrand.
        weights: FloatTensor (N,) — quadrature weights summing to 1.
    """
    if cfg.quadrature_grid == "uniform":
        N = max(1, int(cfg.n_quadrature))
        # Midpoint rule on (0, 1].
        t_floats = torch.tensor(
            [(n - 0.5) / N for n in range(1, N + 1)],
            dtype=torch.float32,
            device=device,
        )
        weights = torch.full((N,), 1.0 / N, dtype=torch.float32, device=device)
    elif cfg.quadrature_grid == "explicit":
        if not cfg.quadrature_points:
            raise ValueError(
                "quadrature_grid='explicit' requires non-empty quadrature_points"
            )
        t_floats = torch.tensor(
            list(cfg.quadrature_points), dtype=torch.float32, device=device
        )
        N = t_floats.numel()
        if cfg.quadrature_weights:
            if len(cfg.quadrature_weights) != N:
                raise ValueError(
                    f"quadrature_weights length {len(cfg.quadrature_weights)} "
                    f"!= quadrature_points length {N}"
                )
            weights = torch.tensor(
                list(cfg.quadrature_weights), dtype=torch.float32, device=device
            )
        else:
            weights = torch.full((N,), 1.0 / N, dtype=torch.float32, device=device)
    else:
        raise ValueError(
            f"quadrature_grid must be 'uniform' or 'explicit', got {cfg.quadrature_grid!r}"
        )

    if (t_floats <= 0).any() or (t_floats > 1).any():
        raise ValueError(f"quadrature t values must be in (0, 1]; got {t_floats.tolist()}")

    # paper_t (fraction masked) → number of revealed positions = round((1 - t) * L).
    # Clamp to [0, L-1] so the model time-embedding stays in range.
    idx_grid = torch.clamp(
        torch.round((1.0 - t_floats) * L).long(), min=0, max=L - 1
    )
    return idx_grid, t_floats, weights


# ─────────────────────────────────────────────────────────────────────────────
# Shared SDMC corruptions
# ─────────────────────────────────────────────────────────────────────────────


def _build_shared_corruptions(
    ids: torch.Tensor,             # (BG, L_total) — rolled-out token ids
    idx_grid: torch.Tensor,        # (N,) — values in [0, D-1]
    t_floats: torch.Tensor,        # (N,)
    weights: torch.Tensor,         # (N,)
    inner_mc: int,
    device: torch.device,
    diffusion_budget: Optional[int] = None,
):
    """Sample masked corruptions ``y_t ~ π_t(·|y)`` once per (n, k).

    Reusing the same corruptions across π_old, π_new, π_ref keeps the
    importance ratio dominated by policy difference rather than
    mask-sampling noise. Each corruption mirrors what
    ``cond_elbo_objective`` does at a fixed ``idx_n``: sample a random
    permutation, reveal exactly ``idx_n`` positions, mask the rest with
    token id 0 (MASK).

    Pre-unmask: if ``diffusion_budget`` (= D) is given and < L_total,
    only positions [0, D) participate in masking — positions [D, L_total)
    keep the rolled-out token (PAD by construction) and are treated as
    permanently revealed.
    """
    BG, L_total = ids.shape
    D = diffusion_budget if diffusion_budget is not None else L_total
    if D > L_total:
        raise ValueError(f"diffusion_budget={D} cannot exceed L_total={L_total}")

    corruptions = []
    for n in range(idx_grid.numel()):
        idx_n_scalar = int(idx_grid[n].item())
        idx_n_tensor = torch.full(
            (BG, 1), idx_n_scalar, dtype=torch.long, device=device
        )
        for _ in range(max(1, inner_mc)):
            # Permute only the active diffusion region [0, D).
            path_first_D = torch.stack(
                [torch.randperm(D, device=device) for _ in range(BG)]
            )                                                          # (BG, D)
            random_path_mask = torch.ones(
                BG, L_total, dtype=torch.bool, device=device
            )
            random_path_mask[:, :D] = path_first_D < idx_n_scalar
            # Positions ≥ D stay at the rolled-out value (PAD); always "revealed".
            x_t = torch.where(
                random_path_mask, ids, torch.full_like(ids, MASK_ID)
            )
            corruptions.append(
                {
                    "x_t": x_t,                      # (BG, L_total) long; 0 at masked
                    "idx": idx_n_tensor,             # (BG, 1) long
                    "t": t_floats[n].item(),         # python float
                    "w": weights[n].item(),          # python float
                    "n_idx": n,
                }
            )
    return corruptions


def _per_corruption_logprob_sum(
    model: torch.nn.Module,
    ids: torch.Tensor,             # (BG, L) — rolled-out token ids
    z_c_rep: torch.Tensor,         # (BG, emb_dim)
    x_t: torch.Tensor,             # (BG, L) — corrupted (mask=0 at masked positions)
    idx_t: torch.Tensor,           # (BG, 1) — model-time index for this corruption
) -> torch.Tensor:
    """Per-sequence sum of `log π_θ(y^i | x_t, q)` over currently-masked positions.

    Returns ``(BG,)``. Bypasses the ``OneHotCategorical`` indirection in
    ``transformer_training_helper`` — uses ``F.log_softmax + gather``
    directly so the function can be wrapped in ``checkpoint``.
    """
    logits = model(x=x_t, t=idx_t.view(-1), y_c=z_c_rep)         # (BG, V, L)
    log_probs = F.log_softmax(logits, dim=1)                     # (BG, V, L)
    lp = log_probs.gather(1, ids.unsqueeze(1)).squeeze(1)        # (BG, L)
    mask_indicator = (x_t == MASK_ID).to(lp.dtype)               # (BG, L)
    return (lp * mask_indicator).sum(dim=1)                      # (BG,)


def _elbo_sdmc(
    model: torch.nn.Module,
    ids: torch.Tensor,             # (BG, L)
    z_c_rep: torch.Tensor,         # (BG, emb_dim)
    corruptions: list,
    args_namespace: Namespace,     # kept for signature compat; unused
    eps_t: float,
    inner_mc: int,
    gradient_checkpoint: bool = False,
) -> torch.Tensor:
    """Sequence-level ELBO under SDMC. Returns (BG,).

    Implements Eq. (5) of the paper:
        L_ELBO ≈ Σ_n w_n · (1/inner_mc) · Σ_k (1/t_n) · Σ_i 1[y_{t_n}^i = M]
                 · log π_θ(y^i | y_{t_n}, q)

    With ``gradient_checkpoint=True`` and grad enabled, each
    per-corruption forward is wrapped in
    ``torch.utils.checkpoint.checkpoint(use_reentrant=False)`` to keep
    peak activation memory at one forward's worth — necessary for L≥1024
    and N≥7 on a single Aurora tile.
    """
    elbo = None
    n_per_idx = max(1, inner_mc)
    do_ckpt = (
        gradient_checkpoint
        and torch.is_grad_enabled()
        and any(p.requires_grad for p in model.parameters())
    )

    for c in corruptions:
        x_t = c["x_t"]
        idx_n = c["idx"]
        t_n = max(c["t"], eps_t)
        w_n = c["w"]
        if do_ckpt:
            # ``model`` captured by closure; checkpoint only sees tensor
            # args, so its parameter grads still flow correctly under
            # use_reentrant=False.
            def _ckpt_fn(x_t_, idx_t_, ids_, z_):
                return _per_corruption_logprob_sum(model, ids_, z_, x_t_, idx_t_)
            lp_sum = torch.utils.checkpoint.checkpoint(
                _ckpt_fn, x_t, idx_n, ids, z_c_rep, use_reentrant=False
            )
        else:
            lp_sum = _per_corruption_logprob_sum(model, ids, z_c_rep, x_t, idx_n)
        contrib = (w_n / (n_per_idx * t_n)) * lp_sum
        elbo = contrib if elbo is None else elbo + contrib
    return elbo


def _tokenwise_k3_kl(
    s3: torch.nn.Module,
    ref_s3: torch.nn.Module,
    ids_all: torch.Tensor,         # (BG, L_total)
    z_c_rep: torch.Tensor,         # (BG, emb_dim)
    diffusion_budget: Optional[int] = None,
    fill_id: int = PAD_ID,
) -> torch.Tensor:
    """Cheap k3 KL estimator on a single fully-masked forward pass.

    Mirrors the GRPO KL term in ``biom3.rl.grpo._policy_logprobs`` —
    except we feed token id 0 (mask) at every position, not PAD.

    Pre-unmask: when ``diffusion_budget`` (= D) is given and < L_total,
    positions [0, D) are MASK and positions [D, L_total) hold ``fill_id``
    (PAD). The KL average then naturally restricts to non-PAD positions
    via ``valid = (ids_all != PAD_ID)``.
    """
    BG, L_total = ids_all.shape
    D = diffusion_budget if diffusion_budget is not None else L_total
    x_masked = torch.full_like(ids_all, fill_id)
    x_masked[:, :D] = MASK_ID
    t_steps = torch.zeros(BG, dtype=torch.long, device=ids_all.device)
    logits_new = s3(x_masked, t_steps, z_c_rep).float().permute(0, 2, 1)
    with torch.no_grad():
        logits_ref = ref_s3(x_masked, t_steps, z_c_rep).float().permute(0, 2, 1)
    lp_new = F.log_softmax(logits_new, dim=-1).gather(-1, ids_all.unsqueeze(-1)).squeeze(-1)
    lp_ref = F.log_softmax(logits_ref, dim=-1).gather(-1, ids_all.unsqueeze(-1)).squeeze(-1)
    valid = (ids_all != PAD_ID).float()
    delta = lp_ref - lp_new
    kl = (torch.exp(delta) - delta - 1.0)
    return (kl * valid).sum() / (valid.sum() + 1e-8)


def _resolve_rollout_devices(
    requested: Optional[List[str]],
    master: torch.device,
) -> List[torch.device]:
    """Resolve ``rollout_devices`` config to concrete torch.device objects.

    - ``None`` → ``[master]`` (single-device, current behavior).
    - ``["auto"]`` → all visible XPU/CUDA tiles, with the master device
      placed first. Falls back to ``[master]`` if no multi-device
      runtime is available.
    - explicit list → parsed as-is, with the master moved to position 0
      if present (so devices[0] is always the master tile).
    """
    if not requested:
        return [master]
    if len(requested) == 1 and str(requested[0]).lower() == "auto":
        n = 0
        if master.type == "xpu" and hasattr(torch, "xpu"):
            try:
                n = int(torch.xpu.device_count())
            except Exception:
                n = 0
        elif master.type == "cuda":
            try:
                n = int(torch.cuda.device_count())
            except Exception:
                n = 0
        if n <= 1:
            logger.warning(
                "rollout_devices='auto' but device_count=%d; falling back to single device", n,
            )
            return [master]
        # Master at index 0; rest in ascending tile order, skipping master.
        master_idx = master.index if master.index is not None else 0
        out = [master] + [
            torch.device(f"{master.type}:{i}") for i in range(n) if i != master_idx
        ]
        return out
    parsed = [torch.device(s) for s in requested]
    # Ensure devices[0] is the master.
    if master not in parsed:
        return [master] + parsed
    parsed = [master] + [d for d in parsed if d != master]
    return parsed


def _gdpo_rollout(
    s3: torch.nn.Module,
    cfg3: Namespace,
    z_c: torch.Tensor,             # (1, emb_dim)
    K: int,
    device: torch.device,
) -> torch.Tensor:
    """Rollout ``K`` sequences honoring ``cfg3.pre_unmask``.

    Returns ``(K, L_total)`` int64 token IDs. Equivalent to
    ``biom3.rl.grpo.diffusion_rollout`` when pre_unmask is False;
    when True, only positions [0, D) are diffused over and positions
    [D, L_total) are pre-filled with the configured fill token.
    """
    L_total = getattr(cfg3, 'sequence_length', cfg3.diffusion_steps)
    z_rep = z_c.repeat(K, 1)
    init, sampling_path = _build_initial_mask_state(cfg3, batch_size=K, tokens=TOKENS)
    init = init.to(device)
    sampling_path = sampling_path.to(device)

    mask_list, _, _ = S3sample.batch_generate_denoised_sampled(
        args=cfg3,
        model=s3,
        extract_digit_samples=init.float(),
        extract_time=torch.zeros(K, dtype=torch.long, device=device),
        extract_digit_label=z_rep,
        sampling_path=sampling_path,
    )
    final = mask_list[-1]   # (K, 1, L_total)
    out = torch.empty(K, L_total, dtype=torch.long, device=device)
    for k in range(K):
        out[k] = torch.as_tensor(final[k, 0], dtype=torch.long, device=device)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Debug dump
# ─────────────────────────────────────────────────────────────────────────────


def _visualize_corruption(x_t_row: torch.Tensor) -> str:
    """Render one corrupted sequence as a same-length string.

    Revealed positions show the actual amino-acid / special-token glyph
    from ``TOKENS``. Masked positions (token id 0) render as ``_`` so
    the reveal pattern is visible at a glance.
    """
    out = []
    for tid in x_t_row.tolist():
        if tid == MASK_ID:
            out.append("_")
        else:
            out.append(TOKENS[int(tid)])
    return "".join(out)


def _write_debug_step(
    debug_path: str,
    step: int,
    batch_prompts: List[str],
    ids_all: torch.Tensor,           # (BG, L)
    seqs: List[str],                 # decoded, special-stripped
    rewards_raw,                     # list[float] or array, len BG
    adv: torch.Tensor,               # (BG,)
    elbo_old: torch.Tensor,          # (BG,)
    elbo_new: torch.Tensor,          # (BG,) — detached for logging
    log_ratio_seq: torch.Tensor,     # (BG,) — detached
    ratio_seq: torch.Tensor,         # (BG,) — detached
    seq_lengths: torch.Tensor,       # (BG,)
    components_per_replica,          # dict[str, list[float]] | None
    corruptions: list,
    G: int,
) -> None:
    """Append one stanza per training step to ``debug.out``.

    Always opens in append mode so a crash mid-run leaves the prior
    steps' diagnostics intact on disk.
    """
    BG, L = ids_all.shape
    rewards_list = [float(x) for x in rewards_raw]
    adv_list = adv.detach().cpu().tolist()
    elbo_old_list = elbo_old.detach().cpu().tolist()
    elbo_new_list = elbo_new.detach().cpu().tolist()
    lr_list = log_ratio_seq.detach().cpu().tolist()
    r_list = ratio_seq.detach().cpu().tolist()
    seqlen_list = [int(x) for x in seq_lengths.detach().cpu().tolist()]

    bar = "=" * 100
    sub = "-" * 100
    lines: List[str] = []
    lines.append(bar)
    lines.append(
        f"step={step:>5d}   B*G={BG}   L={L}   N_quadrature={len(corruptions) // max(1, BG // BG) if False else len(corruptions)}"
    )
    # corruptions has N * inner_mc entries; report unique idx values.
    unique_idx = []
    for c in corruptions:
        v = int(c["idx"][0, 0].item())
        if v not in unique_idx:
            unique_idx.append(v)
    lines.append(
        f"distinct_quadrature_idx={unique_idx}   total_corruptions={len(corruptions)}"
    )
    lines.append(bar)
    lines.append("prompts:")
    for i, p in enumerate(batch_prompts):
        lines.append(f"  [{i}] {p}")
    lines.append("")

    # Per-replica scalar table
    lines.append("per-replica:")
    header = (
        f"  {'g':>3}  {'p':>3}  {'len':>4}  {'reward':>10}  {'advantage':>10}  "
        f"{'elbo_old':>11}  {'elbo_new':>11}  {'log_ratio':>10}  {'ratio':>8}"
    )
    lines.append(header)
    for i in range(BG):
        p_idx = i // G
        g_idx = i % G
        lines.append(
            f"  {g_idx:>3d}  {p_idx:>3d}  {seqlen_list[i]:>4d}  "
            f"{rewards_list[i]:>10.4f}  {adv_list[i]:>+10.4f}  "
            f"{elbo_old_list[i]:>11.4f}  {elbo_new_list[i]:>11.4f}  "
            f"{lr_list[i]:>+10.4f}  {r_list[i]:>8.4f}"
        )
    if components_per_replica:
        lines.append("")
        lines.append("components per replica:")
        for name, vals in components_per_replica.items():
            vals_s = ", ".join(f"{float(v):.3f}" for v in vals)
            lines.append(f"  {name}: [{vals_s}]")

    lines.append("")
    lines.append("generated sequences (decoded, special tokens stripped):")
    for i in range(BG):
        p_idx = i // G
        g_idx = i % G
        lines.append(f"  [g={g_idx} p={p_idx}] len={len(seqs[i])} {seqs[i]}")

    lines.append("")
    lines.append("raw token-id sequences (full L, before decoding):")
    for i in range(BG):
        ids_str = "".join(TOKENS[int(t)] if int(t) != MASK_ID else "_"
                          for t in ids_all[i].tolist())
        lines.append(f"  [g={i % G} p={i // G}] {ids_str}")

    lines.append("")
    lines.append("SDMC corruptions (mask visualization: '_' = masked, otherwise = revealed token):")
    for n_c, c in enumerate(corruptions):
        idx_n = int(c["idx"][0, 0].item())
        t_n = float(c["t"])
        w_n = float(c["w"])
        n_idx = int(c["n_idx"])
        revealed = idx_n
        masked = L - idx_n
        lines.append(sub)
        lines.append(
            f"corruption[{n_c}]  n_idx={n_idx}  idx={idx_n}  t={t_n:.4f}  w={w_n:.4f}  "
            f"revealed={revealed}/{L}  masked={masked}/{L}"
        )
        x_t = c["x_t"]
        for i in range(BG):
            mask_str = _visualize_corruption(x_t[i])
            lines.append(f"  [g={i % G} p={i // G}] {mask_str}")

    lines.append("")
    with open(debug_path, "a") as f:
        f.write("\n".join(lines) + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────────────────────


def gdpo_train(
    gdpo_cfg: GDPOConfig,
    cfg1: Namespace,
    cfg2: Namespace,
    cfg3: Namespace,
    prompts: List[str],
    reward_fn,
    device: torch.device,
    stage1_weights: Optional[str] = None,
    stage2_weights: Optional[str] = None,
    stage3_init_weights: Optional[str] = None,
):
    cfg3.device = str(device)

    torch.manual_seed(gdpo_cfg.seed)
    np.random.seed(gdpo_cfg.seed)

    logger.info("Loading Stage 1 (PenCL)...")
    s1 = load_pencl_frozen(cfg1, stage1_weights, device=str(device))
    logger.info("Loading Stage 2 (Facilitator)...")
    s2 = load_facilitator_frozen(cfg2, stage2_weights, device=str(device))
    logger.info("Loading Stage 3 (ProteoScribe, trainable)...")
    s3 = load_proteoscribe_trainable(cfg3, stage3_init_weights, device=str(device))
    s3.train()

    logger.info("Snapshotting frozen reference policy (π_ref)...")
    ref_s3 = copy.deepcopy(s3).eval()
    for p in ref_s3.parameters():
        p.requires_grad_(False)

    encode_prompt = _PromptEncoder(s1, s2, cfg1, device)

    optimizer = torch.optim.AdamW(
        s3.parameters(),
        lr=gdpo_cfg.learning_rate,
        weight_decay=gdpo_cfg.weight_decay,
    )

    # Resolve pre-unmask state on cfg3 (mirrors run_ProteoScribe_sample.main).
    # cfg3.diffusion_steps is treated as the architectural sequence length
    # the model was trained at; sequence_length captures it before we
    # potentially override diffusion_steps with the budget D.
    if getattr(cfg3, 'sequence_length', None) is None:
        cfg3.sequence_length = cfg3.diffusion_steps
    cfg3.pre_unmask = bool(gdpo_cfg.pre_unmask)
    fill_id = PAD_ID
    if cfg3.pre_unmask:
        if not gdpo_cfg.pre_unmask_config:
            raise ValueError("pre_unmask=True requires pre_unmask_config (path to JSON)")
        pre_cfg = load_pre_unmask_config(gdpo_cfg.pre_unmask_config)
        if pre_cfg["diffusion_budget"] > cfg3.sequence_length:
            raise ValueError(
                f"pre_unmask diffusion_budget ({pre_cfg['diffusion_budget']}) "
                f"must be <= sequence_length ({cfg3.sequence_length})"
            )
        cfg3.diffusion_steps = pre_cfg["diffusion_budget"]
        cfg3.pre_unmask_strategy = pre_cfg["strategy"]
        cfg3.pre_unmask_fill_with = pre_cfg["fill_with"]
        fill_id = _resolve_fill_token_id(cfg3.pre_unmask_fill_with, TOKENS)
        logger.info(
            "Pre-unmask enabled: strategy=%s fill_with=%s D=%d L_total=%d",
            cfg3.pre_unmask_strategy, cfg3.pre_unmask_fill_with,
            cfg3.diffusion_steps, cfg3.sequence_length,
        )

    os.makedirs(gdpo_cfg.output_dir, exist_ok=True)
    log_rows: list = []
    log_path = os.path.join(gdpo_cfg.output_dir, "train_log.json")
    debug_path = os.path.join(gdpo_cfg.output_dir, "debug.out")
    if gdpo_cfg.debug_log and os.path.exists(debug_path):
        # Truncate any previous run's debug log so we don't accumulate
        # across re-runs of the same output_dir.
        open(debug_path, "w").close()
    D = cfg3.diffusion_steps                   # active diffusion budget
    L_total = cfg3.sequence_length             # full tensor length (may be > D under pre_unmask)
    eps = gdpo_cfg.eps
    beta = gdpo_cfg.beta
    B = gdpo_cfg.batch_size
    G = gdpo_cfg.num_generations
    BG = B * G

    idx_grid, t_floats, weights = _build_grid(gdpo_cfg, D, device)
    logger.info(
        "SDMC grid: N=%d t=%s w=%s idx=%s (D=%d, L_total=%d, inner_mc=%d, kl=%s)",
        idx_grid.numel(), t_floats.tolist(), weights.tolist(),
        idx_grid.tolist(), D, L_total, gdpo_cfg.inner_mc, gdpo_cfg.kl_estimator,
    )

    # Multi-device rollout pool. devices[0] is the master (gradient
    # updates, ELBO under π_new, KL all live here); the rest hold
    # frozen replicas used for parallel rollout only.
    rollout_devs = _resolve_rollout_devices(gdpo_cfg.rollout_devices, device)
    if len(rollout_devs) > 1:
        rollout_pool = RolloutPool(
            s3_master=s3,
            cfg3=cfg3,
            rollout_fn=_gdpo_rollout,
            devices=rollout_devs,
        )
        logger.info(
            "Multi-device rollout enabled: %d tiles → %s",
            len(rollout_devs), [str(d) for d in rollout_devs],
        )
    else:
        rollout_pool = None

    trainable_params = [p for p in s3.parameters() if p.requires_grad]
    initial_params = [p.detach().clone() for p in trainable_params]
    reward_sum = 0.0

    logger.info(
        "GDPO start: steps=%d G=%d batch=%d beta=%.4f eps=%.2f | %d trainable params",
        gdpo_cfg.steps, G, B, beta, eps,
        sum(p.numel() for p in trainable_params),
    )

    # Snapshot the algorithm config as the first row of the log so that
    # downstream analysis can recover the SDMC grid used.
    log_rows.append({
        "_meta": True,
        "n_quadrature": int(idx_grid.numel()),
        "quadrature_grid": gdpo_cfg.quadrature_grid,
        "quadrature_t": [round(float(x), 6) for x in t_floats.tolist()],
        "quadrature_w": [round(float(x), 6) for x in weights.tolist()],
        "quadrature_idx": idx_grid.tolist(),
        "inner_mc": gdpo_cfg.inner_mc,
        "kl_estimator": gdpo_cfg.kl_estimator,
        "use_old_policy_snapshot": gdpo_cfg.use_old_policy_snapshot,
        "advantage_normalize": gdpo_cfg.advantage_normalize,
        "diffusion_budget": int(D),
        "sequence_length": int(L_total),
        "pre_unmask": bool(cfg3.pre_unmask),
        "pre_unmask_fill_with": getattr(cfg3, 'pre_unmask_fill_with', None),
    })
    # Flush the meta-only log so that crashes during weight loading or
    # the very first step still leave a parseable train_log.json behind.
    from biom3.rl.plotting import write_train_log_atomic
    write_train_log_atomic(log_path, log_rows)

    for step in range(1, gdpo_cfg.steps + 1):
        t0 = time.time()
        _stamp_t = time.perf_counter()

        def _stamp(label):
            nonlocal _stamp_t
            now = time.perf_counter()
            logger.info("  [t] %-14s %5.2fs", label, now - _stamp_t)
            _stamp_t = now

        batch_prompts = [
            prompts[np.random.randint(len(prompts))] for _ in range(B)
        ]
        logger.info("step=%d batch_prompts:", step)
        for i, p in enumerate(batch_prompts):
            logger.info("  [%d] %s", i, p)

        # Paper Alg. 1 line 3: π_old ← π_θ each outer iteration.
        if gdpo_cfg.use_old_policy_snapshot:
            old_s3 = copy.deepcopy(s3).eval()
            for p in old_s3.parameters():
                p.requires_grad_(False)
        else:
            old_s3 = ref_s3
        _stamp("snapshot_old")

        s3.eval()
        with torch.no_grad():
            z_cs = torch.cat([encode_prompt(p) for p in batch_prompts], dim=0)
            z_cs_rep = z_cs.repeat_interleave(G, dim=0)            # (BG, emb)
            _stamp("encode_z_c")

            # Rollout under π_old (honors pre_unmask via _build_initial_mask_state).
            if rollout_pool is not None:
                # Broadcast π_old's weights to all replica tiles, then
                # dispatch G rollouts per prompt across the pool.
                rollout_pool.sync_from(old_s3)
                ids_per_prompt = [
                    rollout_pool.rollout(z_cs[i:i + 1], G)
                    for i in range(B)
                ]
            else:
                ids_per_prompt = [
                    _gdpo_rollout(old_s3, cfg3, z_cs[i:i + 1], G, device)
                    for i in range(B)
                ]
            ids_all = torch.cat(ids_per_prompt, dim=0)              # (BG, L_total)
            _stamp("rollout")

            seqs = [decode_tokens(ids_all[i]) for i in range(BG)]
            logger.info("step=%d generated sequences (B*G=%d):", step, BG)
            for i, seq in enumerate(seqs):
                p_idx = i // G
                logger.info(
                    "  [prompt %d / replica %d] len=%d %s",
                    p_idx, i % G, len(seq), seq,
                )
            _stamp("decode")

            # Shared SDMC corruptions — used by π_old, π_new, optionally π_ref.
            corruptions = _build_shared_corruptions(
                ids=ids_all,
                idx_grid=idx_grid,
                t_floats=t_floats,
                weights=weights,
                inner_mc=gdpo_cfg.inner_mc,
                device=device,
                diffusion_budget=D,
            )
            _stamp("corruptions")

            elbo_old = _elbo_sdmc(
                model=old_s3,
                ids=ids_all,
                z_c_rep=z_cs_rep,
                corruptions=corruptions,
                args_namespace=cfg3,
                eps_t=gdpo_cfg.eps_t,
                inner_mc=gdpo_cfg.inner_mc,
                gradient_checkpoint=False,  # no_grad block; checkpoint not needed
            )
            _stamp("elbo_old")

            if gdpo_cfg.kl_estimator == "sdmc":
                elbo_ref = _elbo_sdmc(
                    model=ref_s3,
                    ids=ids_all,
                    z_c_rep=z_cs_rep,
                    corruptions=corruptions,
                    args_namespace=cfg3,
                    eps_t=gdpo_cfg.eps_t,
                    inner_mc=gdpo_cfg.inner_mc,
                    gradient_checkpoint=False,
                )
                _stamp("elbo_ref")
            else:
                elbo_ref = None

        rewards_raw = reward_fn(seqs)
        _stamp("reward")
        R = torch.tensor(rewards_raw, dtype=torch.float32, device=device)
        Rg = R.view(B, G)
        if gdpo_cfg.advantage_normalize:
            adv = (
                (Rg - Rg.mean(dim=-1, keepdim=True))
                / (Rg.std(dim=-1, keepdim=True).clamp(min=1e-8))
            ).view(BG)
        else:
            # Paper eq. (6): unnormalized A_g = R_g - mean(R) (Liu et al. 2025b).
            adv = (Rg - Rg.mean(dim=-1, keepdim=True)).view(BG)

        # ELBO under π_new — autograd flows here. Gradient checkpointing
        # keeps peak activation memory at one forward's worth instead of
        # N · inner_mc, which is what makes L=1024, N≥7 fit on one tile.
        s3.train()
        elbo_new = _elbo_sdmc(
            model=s3,
            ids=ids_all,
            z_c_rep=z_cs_rep,
            corruptions=corruptions,
            args_namespace=cfg3,
            eps_t=gdpo_cfg.eps_t,
            inner_mc=gdpo_cfg.inner_mc,
            gradient_checkpoint=gdpo_cfg.gradient_checkpoint,
        )
        _stamp("elbo_new")

        # Sequence-level PPO-clip (paper eq. 6). r_g and A_g are scalars per sequence.
        log_ratio_seq = elbo_new - elbo_old.detach()                 # (BG,)
        ratio_seq = torch.exp(log_ratio_seq)
        seq_lengths = (ids_all != PAD_ID).float().sum(dim=1).clamp(min=1.0)
        pg1 = -adv * ratio_seq
        pg2 = -adv * ratio_seq.clamp(1 - eps, 1 + eps)
        pg_loss = (torch.max(pg1, pg2) / seq_lengths).mean()

        if gdpo_cfg.kl_estimator == "tokenwise_k3":
            kl_loss = _tokenwise_k3_kl(
                s3, ref_s3, ids_all, z_cs_rep,
                diffusion_budget=D, fill_id=fill_id,
            )
        elif gdpo_cfg.kl_estimator == "sdmc":
            # Forward-KL surrogate via SDMC; elbo_ref is a no-grad estimate.
            kl_loss = (elbo_ref.detach() - elbo_new).mean()
        else:
            raise ValueError(
                f"kl_estimator must be 'tokenwise_k3' or 'sdmc', got {gdpo_cfg.kl_estimator!r}"
            )

        loss = pg_loss + beta * kl_loss
        _stamp("loss")

        pre_step_params = [p.detach().clone() for p in trainable_params]

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(s3.parameters(), gdpo_cfg.max_grad_norm)
        optimizer.step()
        _stamp("backward+opt")

        with torch.no_grad():
            dw_step = torch.sqrt(sum(
                ((p - pre) ** 2).sum() for p, pre in zip(trainable_params, pre_step_params)
            )).item()
            dw_total = torch.sqrt(sum(
                ((p - init) ** 2).sum() for p, init in zip(trainable_params, initial_params)
            )).item()
        del pre_step_params
        _stamp("dw_norms")

        if gdpo_cfg.use_old_policy_snapshot:
            del old_s3  # free GPU snapshot before next step

        dt = time.time() - t0
        clip_frac = ((ratio_seq - 1.0).abs() > eps).float().mean().item()
        mean_reward = R.mean().item()
        reward_sum += mean_reward
        reward_avg = reward_sum / step
        avg_len = float(np.mean([len(s) for s in seqs]))
        all_lengths = [len(s) for s in seqs]

        elbo_new_mean = float(elbo_new.detach().mean().item())
        elbo_old_mean = float(elbo_old.mean().item())
        log_ratio_mean = float(log_ratio_seq.detach().mean().item())
        log_ratio_max_abs = float(log_ratio_seq.detach().abs().max().item())

        logger.info(
            "step=%4d | reward=%5.2f (avg=%5.2f) | loss=%.4f pg=%.4f kl=%.4f clip=%.2f "
            "| elbo_new=%.3f elbo_old=%.3f lr_seq=%.3f (|max|=%.3f) "
            "| dw=%.2e (tot=%.2e) | len=%.2f | %.1fs | replicas=%d | all_lengths=%s",
            step, mean_reward, reward_avg,
            loss.item(), pg_loss.item(), kl_loss.item(), clip_frac,
            elbo_new_mean, elbo_old_mean, log_ratio_mean, log_ratio_max_abs,
            dw_step, dw_total, avg_len, dt, len(seqs), str(all_lengths),
        )
        row = {
            "step": step,
            "reward": round(mean_reward, 3),
            "reward_avg": round(reward_avg, 3),
            # Raw per-replica reward values for downstream plotting
            # (length BG = batch_size * num_generations).
            "rewards_per_replica": [float(x) for x in rewards_raw],
            "loss": round(loss.item(), 5),
            "pg": round(pg_loss.item(), 5),
            "kl": round(kl_loss.item(), 5),
            "clip_frac": round(clip_frac, 4),
            "elbo_new": round(elbo_new_mean, 4),
            "elbo_old": round(elbo_old_mean, 4),
            "log_ratio_seq": round(log_ratio_mean, 4),
            "log_ratio_seq_max_abs": round(log_ratio_max_abs, 4),
            "dw_step": dw_step,
            "dw_total": dw_total,
            "avg_len": round(avg_len, 1),
        }
        components_per_replica = None
        last_components = getattr(reward_fn, "last_components", None)
        if callable(last_components):
            comps = last_components()
            if comps:
                row["components"] = {
                    name: round(float(np.mean(vals)), 3) for name, vals in comps.items()
                }
                components_per_replica = {
                    name: [float(x) for x in vals] for name, vals in comps.items()
                }
                row["components_per_replica"] = components_per_replica
                logger.info(
                    "  components: %s",
                    ", ".join(f"{n}={row['components'][n]:.3f}" for n in row["components"]),
                )
        log_rows.append(row)
        # Atomic per-step flush so a job hitting walltime or crashing
        # leaves behind a fully-parseable train_log.json with every
        # step recorded up to that point.
        try:
            write_train_log_atomic(log_path, log_rows)
        except Exception as e:  # pragma: no cover
            logger.warning("train_log.json flush failed (non-fatal): %s", e)

        if gdpo_cfg.debug_log:
            try:
                _write_debug_step(
                    debug_path=debug_path,
                    step=step,
                    batch_prompts=batch_prompts,
                    ids_all=ids_all,
                    seqs=seqs,
                    rewards_raw=rewards_raw,
                    adv=adv,
                    elbo_old=elbo_old,
                    elbo_new=elbo_new.detach(),
                    log_ratio_seq=log_ratio_seq.detach(),
                    ratio_seq=ratio_seq.detach(),
                    seq_lengths=seq_lengths,
                    components_per_replica=components_per_replica,
                    corruptions=corruptions,
                    G=G,
                )
            except Exception as e:  # pragma: no cover
                logger.warning("debug.out write failed (non-fatal): %s", e)

        if step % gdpo_cfg.save_steps == 0:
            ckpt_path = os.path.join(gdpo_cfg.output_dir, f"step{step}.pt")
            torch.save({"step": step, "model_state": s3.state_dict()}, ckpt_path)
            logger.info("Saved checkpoint: %s", ckpt_path)

    final_path = os.path.join(gdpo_cfg.output_dir, "final.pt")
    torch.save({"step": gdpo_cfg.steps, "model_state": s3.state_dict()}, final_path)
    # train_log.json was already flushed after every step; this final
    # rewrite is a no-op safety net in case any exception path skipped
    # the in-loop flush.
    write_train_log_atomic(log_path, log_rows)
    logger.info("GDPO done. Final checkpoint: %s", final_path)

    # Best-effort end-of-run diagnostic plots.
    try:
        from biom3.rl.plotting import plot_train_log
        plot_train_log(log_path, gdpo_cfg.output_dir, algo="gdpo")
    except Exception as e:  # pragma: no cover
        logger.warning("plot_train_log failed (non-fatal): %s", e)

    if rollout_pool is not None:
        rollout_pool.shutdown()

    return log_rows

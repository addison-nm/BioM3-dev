"""Direct Preference Optimization (Diffusion-DPO) for BioM3 Stage 3.

ProteoScribe is an order-agnostic absorbing-state (masked) diffusion model, so
the exact sequence likelihood ``log pi(y | z_c)`` is intractable. Following
Diffusion-DPO (Wallace et al. 2024) and ProteinDPO (Widatalla, Rafailov & Hie
2024), we substitute the **ELBO** as the log-likelihood surrogate, estimated
with the same Semi-deterministic Monte-Carlo (SDMC) quadrature the GDPO trainer
already uses (Rojas et al. 2026). Concretely we reuse
``biom3.rl.gdpo._elbo_sdmc`` verbatim for the log-prob and only add the DPO
loss head + an offline preference-data loop on top.

Two objectives (ProteinDPO):

  * **paired**   — Bradley-Terry logistic loss on an ordered ``(y_w, y_l)`` pair
    (eq. 10):  ``-log sigma(beta * ((E_w - E_w^ref) - (E_l - E_l^ref)))``.
  * **weighted** — scalar-label objective on a set of ``K`` scored candidates
    (eqs. 15-17): match the model's implicit-reward softmax to a Boltzmann
    target ``softmax(score / T)``; consumes scalar scores without binarization.

``E`` is the (optionally length-normalized) SDMC ELBO under the trainable policy
and ``E^ref`` the same under a frozen reference snapshot. Reference and policy
share the same sampled corruptions per sequence to cancel mask-sampling noise.
"""

import copy
import os
import time
from argparse import Namespace
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from biom3.backend.device import setup_logger
from biom3.rl.grpo import PAD_ID, TOKENS, _PromptEncoder
from biom3.rl.gdpo import (
    _build_grid,
    _build_shared_corruptions,
    _elbo_sdmc,
)
from biom3.rl.io import (
    load_facilitator_frozen,
    load_pencl_frozen,
    load_proteoscribe_trainable,
)
from biom3.rl.preference_data import PreferenceSampler, valid_length
from biom3.Stage3.run_ProteoScribe_sample import (
    _resolve_fill_token_id,
    load_pre_unmask_config,
)

logger = setup_logger(__name__)


@dataclass
class DPOConfig:
    output_dir: str = "./dpo_output"
    learning_rate: float = 1e-6
    weight_decay: float = 1e-6
    lr_scheduler: str = "constant"   # "constant" | "linear" (warmup then decay to 0)
    warmup_steps: int = 0
    beta: float = 0.1            # DPO temperature (implicit-reward scale)
    steps: int = 200
    batch_size: int = 4         # pairs (paired) or groups (weighted) per step
    save_steps: int = 50
    max_grad_norm: float = 1.0
    seed: int = 42

    loss_type: str = "paired"   # "paired" | "weighted"

    # paired
    pairing: str = "margin"     # "margin" (rank-gap) | "label" (functional vs not)
    gap_level: float = 0.5      # ProteinDPO rank-gap heuristic, in [0, 1]
    min_margin: float = 0.0     # minimum score gap for a valid margin pair
    label_smoothing: float = 0.0  # cDPO smoothing on the logistic target

    # weighted
    num_candidates: int = 4     # K candidates per group
    temperature: float = 0.1    # T for the Boltzmann target softmax(score / T)

    # Divide the ELBO log-ratio by the non-PAD token count so beta is
    # comparable across designs of different length (ours vary; ProteinDPO's
    # same-length mutants used the raw sum).
    length_normalize: bool = True

    # SDMC quadrature (names match biom3.rl.gdpo._build_grid).
    n_quadrature: int = 3
    quadrature_grid: str = "uniform"
    quadrature_points: Optional[List[float]] = None
    quadrature_weights: Optional[List[float]] = None
    inner_mc: int = 1
    eps_t: float = 1e-3
    gradient_checkpoint: bool = True

    # Pre-unmask: diffuse only the first D positions (must exceed the longest
    # encoded design). Off by default -> diffuse the full architectural length.
    pre_unmask: bool = False
    pre_unmask_config: Optional[str] = None

    # Caption used to build z_c for prompt-less (Case C) data.
    default_caption: str = "SH3 domain protein."


def _resolve_pre_unmask(cfg3: Namespace, dpo_cfg: DPOConfig):
    """Mirror gdpo_train's pre-unmask resolution. Returns (D, L_total, fill_id)."""
    if getattr(cfg3, "sequence_length", None) is None:
        cfg3.sequence_length = cfg3.diffusion_steps
    cfg3.pre_unmask = bool(dpo_cfg.pre_unmask)
    fill_id = PAD_ID
    if cfg3.pre_unmask:
        if not dpo_cfg.pre_unmask_config:
            raise ValueError("pre_unmask=True requires pre_unmask_config (path to JSON)")
        pre_cfg = load_pre_unmask_config(dpo_cfg.pre_unmask_config)
        if pre_cfg["diffusion_budget"] > cfg3.sequence_length:
            raise ValueError(
                f"pre_unmask diffusion_budget ({pre_cfg['diffusion_budget']}) "
                f"must be <= sequence_length ({cfg3.sequence_length})"
            )
        cfg3.diffusion_steps = pre_cfg["diffusion_budget"]
        cfg3.pre_unmask_strategy = pre_cfg["strategy"]
        cfg3.pre_unmask_fill_with = pre_cfg["fill_with"]
        fill_id = _resolve_fill_token_id(cfg3.pre_unmask_fill_with, TOKENS)
    return cfg3.diffusion_steps, cfg3.sequence_length, fill_id


def _paired_elbos(s3, ref_s3, w_ids, l_ids, z_c, grid, cfg3, dpo_cfg, D):
    """Return length-normalized ELBO log-ratios (rho_w, rho_l), each (B,)."""
    idx_grid, t_floats, weights = grid
    B = w_ids.shape[0]
    ids = torch.cat([w_ids, l_ids], dim=0)                 # (2B, L)
    z_rep = torch.cat([z_c, z_c], dim=0)                   # (2B, emb)
    corruptions = _build_shared_corruptions(
        ids=ids, idx_grid=idx_grid, t_floats=t_floats, weights=weights,
        inner_mc=dpo_cfg.inner_mc, device=ids.device, diffusion_budget=D,
    )
    elbo_new = _elbo_sdmc(s3, ids, z_rep, corruptions, cfg3,
                          dpo_cfg.eps_t, dpo_cfg.inner_mc,
                          gradient_checkpoint=dpo_cfg.gradient_checkpoint)
    with torch.no_grad():
        elbo_ref = _elbo_sdmc(ref_s3, ids, z_rep, corruptions, cfg3,
                              dpo_cfg.eps_t, dpo_cfg.inner_mc,
                              gradient_checkpoint=False)
    logratio = elbo_new - elbo_ref                         # (2B,)
    if dpo_cfg.length_normalize:
        logratio = logratio / valid_length(ids)
    rho = dpo_cfg.beta * logratio
    return rho[:B], rho[B:]


def _weighted_elbos(s3, ref_s3, ids_bk, z_c_bk, grid, cfg3, dpo_cfg, D):
    """Return rho (B, K) length-normalized ELBO log-ratios for a candidate set."""
    idx_grid, t_floats, weights = grid
    B, K, L = ids_bk.shape
    ids = ids_bk.reshape(B * K, L)
    z_rep = z_c_bk.reshape(B * K, -1)
    corruptions = _build_shared_corruptions(
        ids=ids, idx_grid=idx_grid, t_floats=t_floats, weights=weights,
        inner_mc=dpo_cfg.inner_mc, device=ids.device, diffusion_budget=D,
    )
    elbo_new = _elbo_sdmc(s3, ids, z_rep, corruptions, cfg3,
                          dpo_cfg.eps_t, dpo_cfg.inner_mc,
                          gradient_checkpoint=dpo_cfg.gradient_checkpoint)
    with torch.no_grad():
        elbo_ref = _elbo_sdmc(ref_s3, ids, z_rep, corruptions, cfg3,
                              dpo_cfg.eps_t, dpo_cfg.inner_mc,
                              gradient_checkpoint=False)
    logratio = elbo_new - elbo_ref
    if dpo_cfg.length_normalize:
        logratio = logratio / valid_length(ids)
    rho = dpo_cfg.beta * logratio
    return rho.reshape(B, K)


def _build_scheduler(optimizer, dpo_cfg: DPOConfig):
    """Linear warmup then linear decay to 0 (ProtRL uses linspace(peak->0)).

    ``lr_scheduler='constant'`` keeps the base LR flat (a no-op LambdaLR).
    """
    total = max(1, dpo_cfg.steps)
    warm = max(0, dpo_cfg.warmup_steps)

    def lr_lambda(cur):  # cur = 0, 1, ..., total-1
        if warm and cur < warm:
            return (cur + 1) / warm
        if dpo_cfg.lr_scheduler == "linear":
            return max(0.0, (total - cur) / max(1, total - warm))
        return 1.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def dpo_train(
    dpo_cfg: DPOConfig,
    cfg1: Namespace,
    cfg2: Namespace,
    cfg3: Namespace,
    sampler: PreferenceSampler,
    device: torch.device,
    stage1_weights: Optional[str] = None,
    stage2_weights: Optional[str] = None,
    stage3_init_weights: Optional[str] = None,
):
    cfg3.device = str(device)
    cfg3._silent_tqdm = True
    torch.manual_seed(dpo_cfg.seed)
    np.random.seed(dpo_cfg.seed)

    logger.info("Loading Stage 1 (PenCL)...")
    s1 = load_pencl_frozen(cfg1, stage1_weights, device=str(device))
    logger.info("Loading Stage 2 (Facilitator)...")
    s2 = load_facilitator_frozen(cfg2, stage2_weights, device=str(device))
    logger.info("Loading Stage 3 (ProteoScribe, trainable)...")
    s3 = load_proteoscribe_trainable(cfg3, stage3_init_weights, device=str(device))
    s3.train()

    logger.info("Snapshotting frozen reference policy (pi_ref)...")
    ref_s3 = copy.deepcopy(s3).eval()
    for p in ref_s3.parameters():
        p.requires_grad_(False)

    D, L_total, _ = _resolve_pre_unmask(cfg3, dpo_cfg)
    grid = _build_grid(dpo_cfg, D, device)
    logger.info("SDMC grid: N=%d idx=%s (D=%d, L_total=%d)",
                grid[0].numel(), grid[0].tolist(), D, L_total)

    # Pre-encode z_c once per unique caption (Stage 1/2 are frozen).
    encode_prompt = _PromptEncoder(s1, s2, cfg1, device)
    z_c_by_key: Dict[str, torch.Tensor] = {}
    for key, caption in sampler.captions:
        z_c_by_key[key] = encode_prompt(caption).detach()   # (1, emb)
    logger.info("Encoded z_c for %d unique caption(s).", len(z_c_by_key))

    optimizer = torch.optim.AdamW(
        s3.parameters(), lr=dpo_cfg.learning_rate, weight_decay=dpo_cfg.weight_decay,
    )
    scheduler = _build_scheduler(optimizer, dpo_cfg)

    os.makedirs(dpo_cfg.output_dir, exist_ok=True)
    log_path = os.path.join(dpo_cfg.output_dir, "train_log.json")
    from biom3.rl.plotting import write_train_log_atomic
    log_rows: list = [{
        "_meta": True, "algo": "dpo", "loss_type": dpo_cfg.loss_type,
        "beta": dpo_cfg.beta, "length_normalize": dpo_cfg.length_normalize,
        "n_quadrature": int(grid[0].numel()), "diffusion_budget": int(D),
        "sequence_length": int(L_total), "pre_unmask": bool(cfg3.pre_unmask),
    }]
    write_train_log_atomic(log_path, log_rows)

    logger.info("DPO start: loss=%s steps=%d batch=%d beta=%.3f",
                dpo_cfg.loss_type, dpo_cfg.steps, dpo_cfg.batch_size, dpo_cfg.beta)

    B = dpo_cfg.batch_size
    for step in range(1, dpo_cfg.steps + 1):
        t0 = time.time()
        s3.train()

        if dpo_cfg.loss_type == "paired":
            batch = sampler.sample_paired_batch(
                B, pairing=dpo_cfg.pairing, gap_level=dpo_cfg.gap_level,
                min_margin=dpo_cfg.min_margin)
            z_c = torch.cat([z_c_by_key[k] for k in batch["caption_keys"]], dim=0)
            rho_w, rho_l = _paired_elbos(
                s3, ref_s3, batch["w_ids"].to(device), batch["l_ids"].to(device),
                z_c, grid, cfg3, dpo_cfg, D)
            margin = rho_w - rho_l
            ls = dpo_cfg.label_smoothing
            loss = -((1 - ls) * F.logsigmoid(margin)
                     + ls * F.logsigmoid(-margin)).mean()
            acc = (margin.detach() > 0).float().mean().item()
            metric = {"pref_acc": round(acc, 4),
                      "margin": round(margin.detach().mean().item(), 4)}
        elif dpo_cfg.loss_type == "weighted":
            batch = sampler.sample_weighted_batch(B, dpo_cfg.num_candidates)
            keys = batch["caption_keys"]
            K = dpo_cfg.num_candidates
            z_c_bk = torch.stack([z_c_by_key[k].expand(K, -1) for k in keys], dim=0)
            scores = batch["scores"].to(device)
            rho = _weighted_elbos(s3, ref_s3, batch["ids"].to(device),
                                  z_c_bk, grid, cfg3, dpo_cfg, D)
            target = F.softmax(scores / dpo_cfg.temperature, dim=1)
            loss = -(target * F.log_softmax(rho, dim=1)).sum(dim=1).mean()
            top1 = (rho.argmax(dim=1) == scores.argmax(dim=1)).float().mean().item()
            metric = {"top1_agree": round(top1, 4)}
        else:
            raise ValueError(f"loss_type must be 'paired' or 'weighted', got {dpo_cfg.loss_type!r}")

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(s3.parameters(), dpo_cfg.max_grad_norm)
        optimizer.step()
        cur_lr = scheduler.get_last_lr()[0]
        scheduler.step()

        dt = time.time() - t0
        row = {"step": step, "loss": round(loss.item(), 5), **metric,
               "lr": cur_lr, "time_s": round(dt, 2)}
        log_rows.append(row)
        logger.info("step=%4d | loss=%.4f | %s | lr=%.2e | %.1fs", step, loss.item(),
                    ", ".join(f"{k}={v}" for k, v in metric.items()), cur_lr, dt)
        try:
            write_train_log_atomic(log_path, log_rows)
        except Exception as e:  # pragma: no cover
            logger.warning("train_log.json flush failed (non-fatal): %s", e)

        if step % dpo_cfg.save_steps == 0:
            ckpt = os.path.join(dpo_cfg.output_dir, f"step{step}.pt")
            torch.save({"step": step, "model_state": s3.state_dict()}, ckpt)
            logger.info("Saved checkpoint: %s", ckpt)

    final_path = os.path.join(dpo_cfg.output_dir, "final.pt")
    torch.save({"step": dpo_cfg.steps, "model_state": s3.state_dict()}, final_path)
    write_train_log_atomic(log_path, log_rows)
    logger.info("DPO done. Final checkpoint: %s", final_path)
    return log_rows

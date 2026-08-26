"""GDPO multi-node trainer for BioM3 Stage 3 — distributed gradient version.

Every rank holds the full per-step state (trainable π_new, frozen π_ref,
rollout π_old, optimizer, Stage 1, Stage 2, ESMFold). Per step, each
rank rolls out its own shard of (B*G) replicas, scores them locally
with ESMFold, contributes its shard to the gradient computation, then
all-reduces gradients across ranks before the optimizer step. Rank-0
work no longer grows with N — this is what lets the trainer scale to
many nodes.

Replaces the earlier rank-0-only training design (which OOM'd at
BG≥96 because rank 0 had to forward+backward the full batch on a
single 64 GB Aurora tile). See docs/.claude_plans/ for the architecture
notes; the failure profile that drove the redesign is captured in
logs/example_gdpo_scaling_v01_n8_*.o.

One rank per node by convention (NGPU_PER_NODE=1 in the launcher).
``RolloutPool`` still fans rollouts across the local Aurora tiles
inside each rank. Cross-rank comm per step:

  - one all-reduce on a (BG,) reward vector (a few KB).
  - two all-reduce on small scalar tensors (KL normalizer, logging).
  - one all-reduce on the trainable model's gradients (~350 MB at
    Stage 3 size; ~seconds over CCL/fabric).
  - one all-gather on per-replica logging fields (only on rank 0's
    write path; small payload).

Reuses the SDMC quadrature, corruptions, ELBO, debug dump, and rollout
helpers verbatim from ``biom3.rl.gdpo``; only the orchestration is new.
"""

import copy
import json
import os
import time
from argparse import Namespace
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist

from biom3.backend.device import setup_logger
from biom3.core.distributed import (
    barrier,
    gather_object_to_main,
    init_distributed_if_launched,
)
from biom3.rl.diversity import diversity_stats
from biom3.rl.gdpo import (
    GDPOConfig,
    MASK_ID,
    _build_grid,
    _build_shared_corruptions,
    _elbo_sdmc,
    _gdpo_rollout,
    _resolve_rollout_devices,
    _time_offset,
    _tokenwise_k3_kl,
    _write_debug_step,
)
from biom3.rl.grpo import (
    PAD_ID,
    TOKENS,
    _PromptEncoder,
    decode_tokens,
)
from biom3.rl.rewards.manifold import bind_pencl_rewards
from biom3.rl.io import (
    load_facilitator_frozen,
    load_pencl_frozen,
    load_proteoscribe_trainable,
)
from biom3.rl.rollout import RolloutPool
from biom3.Stage3.run_ProteoScribe_sample import (
    _resolve_fill_token_id,
    load_pre_unmask_config,
)
import torch.nn.functional as F

logger = setup_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Distributed helpers
# ─────────────────────────────────────────────────────────────────────────────


def _shard_replicas(B: int, G: int, rank: int, world_size: int) -> List[Tuple[int, int, int]]:
    """Round-robin shard of ``B*G`` replicas across ``world_size`` ranks.

    Returns this rank's owned replicas as a list of
    ``(global_idx, prompt_idx, replica_idx)`` tuples, where
    ``global_idx = prompt_idx * G + replica_idx``. Replica ``k`` lives
    on rank ``k % world_size`` — every rank including rank 0 owns a
    share of the batch (rank 0 is no longer reserved for training-only).
    """
    if world_size <= 0:
        raise ValueError(f"world_size must be positive, got {world_size}")
    BG = B * G
    return [
        (k, k // G, k % G)
        for k in range(BG)
        if k % world_size == rank
    ]


def _rollout_local_shard(
    rollout_s3: torch.nn.Module,
    rollout_pool: Optional[RolloutPool],
    cfg3: Namespace,
    z_cs: torch.Tensor,
    shard: List[Tuple[int, int, int]],
    L_total: int,
    device: torch.device,
) -> Tuple[torch.Tensor, List[int]]:
    """Roll out this rank's owned replicas in a single batched call.

    Builds a per-replica ``(K, emb)`` z_c tensor (each row = the
    conditioning for that replica's prompt) and dispatches one rollout
    call. The pool then distributes the K rows evenly across all local
    tiles, so every tile gets work in parallel.

    Earlier versions grouped by prompt and dispatched one call per
    prompt. At scale this left most tiles idle: e.g. at N=8 (B=4,
    K=12) each rank had 3 replicas per prompt → ``_split_evenly(3, 12)``
    only fed tiles 0–2, leaving 9 of 12 idle on every call (and 4
    such serialized calls per step). The single-call dispatch removes
    that serialization.

    Returns the local ids tensor ``(K, L_total)`` and the list of
    global indices in row order — caller uses it to index advantages,
    conditioning vectors, and the gather buffer.
    """
    K = len(shard)
    if K == 0:
        return torch.empty(0, L_total, dtype=torch.long, device=device), []

    p_indices = torch.tensor(
        [p_idx for _, p_idx, _ in shard], dtype=torch.long, device=device
    )
    z_local = z_cs[p_indices]                                   # (K, emb)
    global_idx_order = [g for g, _, _ in shard]

    if rollout_pool is not None:
        ids = rollout_pool.rollout(z_local, K)
    else:
        ids = _gdpo_rollout(rollout_s3, cfg3, z_local, K, device)
    return ids, global_idx_order


def _local_copy_state(src: torch.nn.Module, dst: torch.nn.Module) -> None:
    """``dst`` ← ``src`` in place via per-tensor copy. Avoids any allocation."""
    with torch.no_grad():
        for sp, dp in zip(src.parameters(), dst.parameters()):
            dp.data.copy_(sp.data)
        for sb, db in zip(src.buffers(), dst.buffers()):
            db.data.copy_(sb.data)


def _model_fingerprint(model: torch.nn.Module) -> str:
    """Compute a quick fingerprint of a model's params and buffers.

    Stats: number of params, sum of params (float), L2 norm of params,
    first-layer first-element value. Cheap (no full hash) but sensitive
    enough to catch any cross-rank divergence or unexpected drift.

    Returns a compact human-readable string suitable for log lines.
    """
    with torch.no_grad():
        total_count = 0
        total_sum = 0.0
        total_sq = 0.0
        first_val = None
        for name, p in sorted(model.named_parameters(), key=lambda kv: kv[0]):
            total_count += p.numel()
            total_sum += p.detach().to(torch.float64).sum().item()
            total_sq += (p.detach().to(torch.float64) ** 2).sum().item()
            if first_val is None and p.numel() > 0:
                first_val = float(p.detach().view(-1)[0].to(torch.float64).item())
        l2_norm = float(total_sq ** 0.5)
    return (
        f"n_params={total_count} sum={total_sum:.6e} l2={l2_norm:.6e} "
        f"first={first_val:.6e}" if first_val is not None
        else f"n_params={total_count} sum={total_sum:.6e} l2={l2_norm:.6e}"
    )


def _diag_cross_rank_fingerprint(
    model: torch.nn.Module,
    label: str,
    rank: int,
    world_size: int,
) -> None:
    """Verify ``model`` is bit-identical across all ranks.

    Each rank computes a fingerprint, gathers to rank 0, and rank 0
    logs the result + flags any mismatch. Cheap (~milliseconds + tiny
    gather payload). Useful to confirm the rank-0 broadcast actually
    synchronized weights and to catch any later drift.
    """
    fp = _model_fingerprint(model)
    if world_size <= 1:
        if rank == 0:
            logger.info("[DIAG] %s rank=%d: %s", label, rank, fp)
        return
    fps = gather_object_to_main({"rank": rank, "fp": fp})
    if rank == 0:
        unique = {entry["fp"] for entry in fps if entry is not None}
        if len(unique) == 1:
            logger.info(
                "[DIAG] %s consistent across %d ranks: %s",
                label, world_size, fps[0]["fp"],
            )
        else:
            logger.error(
                "[DIAG] %s DIVERGES across ranks (%d distinct fingerprints):",
                label, len(unique),
            )
            for entry in fps:
                if entry is not None:
                    logger.error(
                        "[DIAG]   rank=%d: %s", entry["rank"], entry["fp"],
                    )


def _broadcast_state(model: torch.nn.Module, src: int = 0) -> None:
    """In-place broadcast of ``model``'s params + buffers from ``src`` rank.

    Used once at init to defend against any nondeterminism in model
    construction (e.g., uninitialized params using per-rank torch RNG)
    so every rank starts from a bit-identical state. Without this,
    accumulated FP error could drift the ranks apart over many steps,
    leaving their rollouts and KL terms slightly inconsistent.
    """
    if not (dist.is_available() and dist.is_initialized()):
        return
    for p in model.parameters():
        dist.broadcast(p.data, src=src)
    for b in model.buffers():
        dist.broadcast(b.data, src=src)


def _all_reduce_grads(model: torch.nn.Module, world_size: int) -> None:
    """Sum-reduce all parameter gradients across ranks. No-op for world_size==1.

    Combined with the per-rank loss scaling (each rank divides its
    local sum by BG before backward), the all-reduced gradient equals
    the gradient of the global mean loss.
    """
    if world_size <= 1 or not (dist.is_available() and dist.is_initialized()):
        return
    for p in model.parameters():
        if p.grad is not None:
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)


def _all_reduce_per_replica(
    local_values: List[float],
    local_global_idx: List[int],
    BG: int,
    device: torch.device,
    world_size: int,
) -> torch.Tensor:
    """Scatter local per-replica scalars into a (BG,) tensor and all-reduce.

    Every rank fills in the entries it owns; non-owners contribute zero.
    A sum-reduce then gives every rank the full (BG,) vector. Used for
    rewards (so each rank can compute its shard's advantages from the
    global reward matrix) and for any per-replica scalar we want to log.
    """
    out = torch.zeros(BG, dtype=torch.float32, device=device)
    for i, g_idx in enumerate(local_global_idx):
        out[g_idx] = float(local_values[i])
    if world_size > 1 and dist.is_available() and dist.is_initialized():
        dist.all_reduce(out, op=dist.ReduceOp.SUM)
    return out


def _write_step_sequences(
    path: str,
    step: int,
    seqs: List[str],
    rewards: List[float],
    advantages: List[float],
    elbo_new: List[float],
    elbo_old: List[float],
    log_ratio_seq: List[float],
    ratio_seq: List[float],
    G: int,
    world_size: int,
) -> None:
    """Append BG rows to ``sequences.jsonl`` — one per replica this step.

    Schema per row:
      step, global_idx, prompt_idx, replica_idx, rank,
      sequence, length, reward, advantage,
      elbo_new, elbo_old, log_ratio_seq, ratio_seq.

    Append-only — the trainer truncates the file once at startup. The
    file is the post-run authority for *what* each replica generated;
    train_log.json captures aggregates and ``batch_prompts_idx`` joins
    rows back to the underlying prompts.json entries.
    """
    BG = len(seqs)
    with open(path, "a") as f:
        for gi in range(BG):
            rec = {
                "step": step,
                "global_idx": gi,
                "prompt_idx": gi // G,
                "replica_idx": gi % G,
                "rank": gi % world_size,
                "sequence": seqs[gi],
                "length": len(seqs[gi]),
                "reward": float(rewards[gi]),
                "advantage": float(advantages[gi]),
                "elbo_new": float(elbo_new[gi]),
                "elbo_old": float(elbo_old[gi]),
                "log_ratio_seq": float(log_ratio_seq[gi]),
                "ratio_seq": float(ratio_seq[gi]),
            }
            f.write(json.dumps(rec) + "\n")


def _all_reduce_per_replica_t(
    local_values: torch.Tensor,        # (K,) any dtype on local device
    local_idx_t: torch.Tensor,         # (K,) long, on local device
    BG: int,
    device: torch.device,
    world_size: int,
) -> torch.Tensor:
    """Tensor analog of ``_all_reduce_per_replica``. Returns ``(BG,)`` float32.

    Saves the host→device roundtrip when the local scalars are already
    tensors (e.g. the detached log-ratio per replica). Each replica is
    owned by exactly one rank so non-owners' zeros don't collide with
    owners' contributions under a sum-reduce.
    """
    out = torch.zeros(BG, dtype=torch.float32, device=device)
    if local_values.numel() > 0:
        out[local_idx_t] = local_values.detach().float()
    if world_size > 1 and dist.is_available() and dist.is_initialized():
        dist.all_reduce(out, op=dist.ReduceOp.SUM)
    return out


def _gather_ids_to_rank0(
    local_ids: torch.Tensor,           # (K, L_total) on local device
    local_global_idx: List[int],
    BG: int,
    L_total: int,
    rank: int,
    world_size: int,
    device: torch.device,
) -> Optional[torch.Tensor]:
    """All-gather rolled-out ids into a (BG, L_total) tensor on rank 0.

    Implementation: every rank scatters its rows into a zero-initialized
    (BG, L_total) tensor on its device, then an all-reduce(sum) merges
    them. PAD_ID is the natural absence marker (every row gets exactly
    one rank's contribution, so non-owner zeros don't collide with
    owner values). Cheap for moderate BG: at BG=1536 and L=1024,
    that's 6 MB per all-reduce.

    Returns the (BG, L_total) tensor on every rank (callers use it on
    rank 0 only for logging — the cost is the same regardless).
    """
    out = torch.zeros(BG, L_total, dtype=torch.long, device=device)
    for i, g_idx in enumerate(local_global_idx):
        out[g_idx] = local_ids[i]
    if world_size > 1 and dist.is_available() and dist.is_initialized():
        dist.all_reduce(out, op=dist.ReduceOp.SUM)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────────────────────


def gdpo_train_multinode(
    gdpo_cfg: GDPOConfig,
    cfg1: Namespace,
    cfg2: Namespace,
    cfg3: Namespace,
    prompts: List[str],
    base_reward_fn: Callable[[List[str]], List[float]],
    device: torch.device,
    stage1_weights: Optional[str] = None,
    stage2_weights: Optional[str] = None,
    stage3_init_weights: Optional[str] = None,
    base_reward_name: str = "base",
    diversity_weight: float = 0.0,         # not honored in distributed path; see README
    diversity_mode: str = "monotone",
    diversity_target: float = 0.5,
    diversity_scale: float = 100.0,
):
    """Distributed-gradient GDPO trainer.

    ``base_reward_fn`` is the per-sequence base reward (e.g. ESMFold pLDDT)
    that each rank runs on its local shard. ``diversity_weight > 0`` is
    NOT supported by this trainer — the diversity term needs within-group
    seqs that span ranks; bringing them all to rank 0 defeats the point
    of distributing. If you need diversity composition, run the single-
    node trainer or land the gather-broadcast pattern as a follow-up.
    """
    # ----- Distributed init (no-op when not launched) -----
    rank, local_rank, world_size, resolved_device_str = init_distributed_if_launched(str(device))
    if world_size > 1:
        device = torch.device(resolved_device_str)

    cfg3.device = str(device)
    cfg3._rank = rank
    cfg3._world_size = world_size
    cfg3._base_seed = int(gdpo_cfg.seed)
    # Silence Stage 3's per-rollout tqdm bar — RL invokes the diffusion
    # loop hundreds of times per run; the progress bar is pure log bloat
    # (~90% of the .o file at BG=96, growing with N). Per-replica diags
    # still go to debug.out.
    cfg3._silent_tqdm = True

    if diversity_weight > 0:
        if rank == 0:
            logger.warning(
                "diversity_weight=%.3f passed to distributed trainer; ignored. "
                "Set diversity_weight=0 in your config; see jobs/aurora/gdpo/scaling/README.md.",
                diversity_weight,
            )

    np.random.seed(gdpo_cfg.seed)
    torch.manual_seed(gdpo_cfg.seed)  # identical RNG state for deterministic model init

    if rank == 0:
        logger.info(
            "GDPO distributed-multinode start: world_size=%d device=%s rank=%d",
            world_size, device, rank,
        )

    # ----- Load models on every rank -----
    if rank == 0:
        logger.info("Loading Stage 1 (PenCL)...")
    s1 = load_pencl_frozen(cfg1, stage1_weights, device=str(device))
    # Every rank scores its own shard, so every rank binds its own PenCL.
    n_bound = bind_pencl_rewards(base_reward_fn, s1)
    if rank == 0 and n_bound:
        logger.info("Bound PenCL into z_p-based reward(s)")
    if rank == 0:
        logger.info("Loading Stage 2 (Facilitator)...")
    s2 = load_facilitator_frozen(cfg2, stage2_weights, device=str(device))

    if rank == 0:
        logger.info("Loading Stage 3 (ProteoScribe) trainable + ref + rollout on every rank...")
    s3 = load_proteoscribe_trainable(cfg3, stage3_init_weights, device=str(device))
    s3.train()

    # Defensive init broadcast: guarantee every rank starts bit-identical.
    # Without this, any per-rank torch RNG drift during model construction
    # could leave the ranks ever-so-slightly out of sync.
    _broadcast_state(s3, src=0)
    _diag_cross_rank_fingerprint(s3, "s3 (trainable, post-broadcast)", rank, world_size)

    # Load ref_s3 and rollout_s3 directly from the same file rather than
    # ``copy.deepcopy(s3)``. The deepcopy path was the only change between
    # the working V20260608_162650 smoke and the broken V20260608_164843
    # smoke that touched generation, and Stage 3 uses
    # ``linear_attention_transformer`` whose reversible blocks have
    # internal hooks / context-manager state that don't always survive
    # deepcopy cleanly. Three independent file loads is ~1 extra second
    # of startup and removes deepcopy as a suspect.
    ref_s3 = load_proteoscribe_trainable(cfg3, stage3_init_weights, device=str(device))
    ref_s3.eval()
    for p in ref_s3.parameters():
        p.requires_grad_(False)
    _broadcast_state(ref_s3, src=0)
    _diag_cross_rank_fingerprint(ref_s3, "ref_s3 (frozen, post-broadcast)", rank, world_size)

    rollout_s3 = load_proteoscribe_trainable(cfg3, stage3_init_weights, device=str(device))
    rollout_s3.eval()
    for p in rollout_s3.parameters():
        p.requires_grad_(False)
    _broadcast_state(rollout_s3, src=0)
    _diag_cross_rank_fingerprint(rollout_s3, "rollout_s3 (frozen, post-broadcast)", rank, world_size)

    optimizer = torch.optim.AdamW(
        s3.parameters(),
        lr=gdpo_cfg.learning_rate,
        weight_decay=gdpo_cfg.weight_decay,
    )

    encode_prompt = _PromptEncoder(s1, s2, cfg1, device)

    # ----- Pre-unmask resolution (mirrors gdpo.gdpo_train) -----
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
        if rank == 0:
            logger.info(
                "Pre-unmask enabled: strategy=%s fill_with=%s D=%d L_total=%d",
                cfg3.pre_unmask_strategy, cfg3.pre_unmask_fill_with,
                cfg3.diffusion_steps, cfg3.sequence_length,
            )

    # ----- Output paths (rank 0 only writes) -----
    log_rows: list = []
    log_path = os.path.join(gdpo_cfg.output_dir, "train_log.json")
    debug_path = os.path.join(gdpo_cfg.output_dir, "debug.out")
    sequences_path = os.path.join(gdpo_cfg.output_dir, "sequences.jsonl")
    prompts_path = os.path.join(gdpo_cfg.output_dir, "prompts.json")
    if rank == 0:
        os.makedirs(gdpo_cfg.output_dir, exist_ok=True)
        if gdpo_cfg.debug_log and os.path.exists(debug_path):
            open(debug_path, "w").close()
        # Truncate sequences.jsonl on (re)start so a re-run in the same
        # output_dir doesn't accumulate stale entries from a prior run.
        open(sequences_path, "w").close()
        # Snapshot the prompts pool — sequences.jsonl + train_log.json's
        # batch_prompts_idx join through this file. Single small write.
        with open(prompts_path, "w") as f:
            json.dump(list(prompts), f, indent=2)

    D = cfg3.diffusion_steps
    L_total = cfg3.sequence_length
    t_offset = _time_offset(cfg3)
    eps = gdpo_cfg.eps
    beta = gdpo_cfg.beta
    B = gdpo_cfg.batch_size
    G = gdpo_cfg.num_generations
    BG = B * G

    idx_grid, t_floats, weights = _build_grid(gdpo_cfg, D, device)
    if rank == 0:
        logger.info(
            "SDMC grid: N=%d t=%s w=%s idx=%s (+offset %d → model t=%s) "
            "(D=%d, L_total=%d, inner_mc=%d, kl=%s)",
            idx_grid.numel(), t_floats.tolist(), weights.tolist(),
            idx_grid.tolist(), t_offset, (idx_grid + t_offset).tolist(),
            D, L_total, gdpo_cfg.inner_mc, gdpo_cfg.kl_estimator,
        )

    # ----- RolloutPool (per-rank, across local tiles) -----
    rollout_devs = _resolve_rollout_devices(gdpo_cfg.rollout_devices, device)
    if len(rollout_devs) > 1:
        rollout_pool = RolloutPool(
            s3_master=rollout_s3,
            cfg3=cfg3,
            rollout_fn=_gdpo_rollout,
            devices=rollout_devs,
        )
        if rank == 0:
            logger.info(
                "Multi-device rollout enabled on each rank: %d tiles → %s",
                len(rollout_devs), [str(d) for d in rollout_devs],
            )
    else:
        rollout_pool = None

    # ----- Promote ESMFold to multi-tile dispatch -----
    # ESMFold scoring is sequential per-sequence and runs on the master
    # tile by default. At K=12 sequences × ~3s each this is ~36s on a
    # single tile while every other tile sits idle. Replace the
    # single-tile reward with one that spreads sequences across all
    # rollout_devs in parallel. The original reward is GC'd unloaded
    # because the trainer hasn't called it yet at this point — first
    # invocation is inside the main loop below.
    from biom3.rl.rewards import ESMFoldReward
    if isinstance(base_reward_fn, ESMFoldReward) and len(rollout_devs) > 1:
        if rank == 0:
            logger.info(
                "Promoting ESMFold to multi-tile: %d tiles → %s",
                len(rollout_devs), [str(d) for d in rollout_devs],
            )
        base_reward_fn = ESMFoldReward(
            devices=[str(d) for d in rollout_devs],
            max_length=base_reward_fn.max_length,
            min_length=base_reward_fn.min_length,
            model_name=base_reward_fn.model_name,
        )

    trainable_params = [p for p in s3.parameters() if p.requires_grad]
    initial_params = [p.detach().clone() for p in trainable_params]
    write_train_log_atomic = None
    reward_sum = 0.0
    if rank == 0:
        logger.info(
            "GDPO start: steps=%d G=%d batch=%d beta=%.4f eps=%.2f | %d trainable params",
            gdpo_cfg.steps, G, B, beta, eps,
            sum(p.numel() for p in trainable_params),
        )

        log_rows.append({
            "_meta": True,
            "algo": "gdpo_multinode_distributed",
            "world_size": int(world_size),
            "batch_size": int(B),
            "num_generations": int(G),
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
            "time_offset": int(t_offset),
            "pre_unmask": bool(cfg3.pre_unmask),
            "pre_unmask_fill_with": getattr(cfg3, 'pre_unmask_fill_with', None),
            "base_reward": str(base_reward_name),
        })
        from biom3.rl.plotting import write_train_log_atomic
        write_train_log_atomic(log_path, log_rows)

    # ─────────────────────────────────────────────────────────────────────────
    # Main loop
    # ─────────────────────────────────────────────────────────────────────────
    try:
        for step in range(1, gdpo_cfg.steps + 1):
            t0 = time.time()

            # ----- 1. Identical batch selection across ranks -----
            prompt_rng = np.random.RandomState(gdpo_cfg.seed + step)
            batch_prompts_idx = [prompt_rng.randint(len(prompts)) for _ in range(B)]
            batch_prompts = [prompts[i] for i in batch_prompts_idx]
            # Per-rank torch RNG so each rank's rollout sampling differs.
            torch.manual_seed(gdpo_cfg.seed + rank * 1_000_003 + step)

            if rank == 0:
                logger.info("step=%d batch_prompts:", step)
                for i, p in enumerate(batch_prompts):
                    logger.info("  [%d] %s", i, p)

            # ----- 2. rollout_s3 ← s3 (LOCAL copy on each rank) -----
            # All ranks' s3 stay bit-identical via the per-step gradient
            # all-reduce + identical optimizer step. No cross-rank weight
            # broadcast needed.
            if gdpo_cfg.use_old_policy_snapshot:
                _local_copy_state(src=s3, dst=rollout_s3)
                if rollout_pool is not None:
                    rollout_pool.sync_from(rollout_s3)
            # else: rollout_s3 stays at initial weights (π_ref equivalent),
            # set once at construction; no sync needed.

            # Per-step cross-rank fingerprint on the first step ONLY.
            # Confirms (a) the all-reduced gradient + optimizer step kept
            # every rank's s3 bit-identical and (b) the local copy to
            # rollout_s3 didn't introduce divergence. Skip subsequent
            # steps to keep log volume bounded.
            if step == 1:
                _diag_cross_rank_fingerprint(
                    s3, "s3 (trainable, step 1 pre-rollout)", rank, world_size,
                )
                _diag_cross_rank_fingerprint(
                    rollout_s3, "rollout_s3 (step 1 pre-rollout)", rank, world_size,
                )

            # ----- 3. Encode z_cs locally (deterministic, frozen S1+S2) -----
            with torch.no_grad():
                z_cs = torch.cat([encode_prompt(p) for p in batch_prompts], dim=0)  # (B, emb)
            z_cs_rep_global = z_cs.repeat_interleave(G, dim=0)                        # (BG, emb)

            # ----- 4. Roll out + score this rank's shard locally -----
            shard = _shard_replicas(B, G, rank, world_size)
            local_global_idx_list = []  # filled in by _rollout_local_shard
            with torch.no_grad():
                local_ids, local_global_idx_list = _rollout_local_shard(
                    rollout_s3=rollout_s3,
                    rollout_pool=rollout_pool,
                    cfg3=cfg3,
                    z_cs=z_cs,
                    shard=shard,
                    L_total=L_total,
                    device=device,
                )
            K = local_ids.shape[0]
            local_seqs = [decode_tokens(local_ids[i]) for i in range(K)]
            local_base_rewards = base_reward_fn(local_seqs) if K else []

            # ----- 5. All-gather rewards into a global (BG,) vector -----
            global_rewards = _all_reduce_per_replica(
                local_values=[float(r) for r in local_base_rewards],
                local_global_idx=local_global_idx_list,
                BG=BG, device=device, world_size=world_size,
            )

            # ----- 6. Compute global advantages, take local slice -----
            R = global_rewards.detach()                                      # (BG,)
            Rg = R.view(B, G)
            if gdpo_cfg.advantage_normalize:
                adv_full = (
                    (Rg - Rg.mean(dim=-1, keepdim=True))
                    / (Rg.std(dim=-1, keepdim=True).clamp(min=1e-8))
                ).view(BG)
            else:
                adv_full = (Rg - Rg.mean(dim=-1, keepdim=True)).view(BG)

            if K > 0:
                local_idx_t = torch.tensor(
                    local_global_idx_list, dtype=torch.long, device=device
                )
                adv_local = adv_full[local_idx_t]                            # (K,)
                z_c_rep_local = z_cs_rep_global[local_idx_t]                  # (K, emb)
            else:
                # Pathological — should not happen if world_size <= BG.
                local_idx_t = torch.empty(0, dtype=torch.long, device=device)
                adv_local = torch.zeros(0, device=device)
                z_c_rep_local = torch.zeros(0, z_cs.shape[1], device=device)

            # ----- 7. Local-shard ELBO compute -----
            with torch.no_grad():
                local_corruptions = _build_shared_corruptions(
                    ids=local_ids,
                    idx_grid=idx_grid,
                    t_floats=t_floats,
                    weights=weights,
                    inner_mc=gdpo_cfg.inner_mc,
                    device=device,
                    diffusion_budget=D,
                    time_offset=t_offset,
                )
                elbo_old_local = _elbo_sdmc(
                    model=rollout_s3,
                    ids=local_ids,
                    z_c_rep=z_c_rep_local,
                    corruptions=local_corruptions,
                    args_namespace=cfg3,
                    eps_t=gdpo_cfg.eps_t,
                    inner_mc=gdpo_cfg.inner_mc,
                    gradient_checkpoint=False,
                )
                if gdpo_cfg.kl_estimator == "sdmc":
                    elbo_ref_local = _elbo_sdmc(
                        model=ref_s3,
                        ids=local_ids,
                        z_c_rep=z_c_rep_local,
                        corruptions=local_corruptions,
                        args_namespace=cfg3,
                        eps_t=gdpo_cfg.eps_t,
                        inner_mc=gdpo_cfg.inner_mc,
                        gradient_checkpoint=False,
                    )
                else:
                    elbo_ref_local = None

            s3.train()
            elbo_new_local = _elbo_sdmc(
                model=s3,
                ids=local_ids,
                z_c_rep=z_c_rep_local,
                corruptions=local_corruptions,
                args_namespace=cfg3,
                eps_t=gdpo_cfg.eps_t,
                inner_mc=gdpo_cfg.inner_mc,
                gradient_checkpoint=gdpo_cfg.gradient_checkpoint,
            )

            # ----- 8. Compose local loss contributions -----
            #
            # Each rank computes its shard's contribution scaled so that
            # summing across ranks (via the gradient all-reduce) equals
            # the original mean-over-BG loss:
            #   pg_loss_total = sum_R pg_local_R / BG  =  mean over BG
            #   kl_loss_total = sum_R kl_local_R / valid_global_sum
            #
            if K > 0:
                log_ratio_seq_local = elbo_new_local - elbo_old_local.detach()
                ratio_seq_local = torch.exp(log_ratio_seq_local)
                seq_lengths_local = (local_ids != PAD_ID).float().sum(dim=1).clamp(min=1.0)
                pg1 = -adv_local * ratio_seq_local
                pg2 = -adv_local * ratio_seq_local.clamp(1 - eps, 1 + eps)
                pg_local_sum = (torch.max(pg1, pg2) / seq_lengths_local).sum()
                pg_loss_local = pg_local_sum / BG
            else:
                log_ratio_seq_local = torch.zeros(0, device=device)
                ratio_seq_local = torch.zeros(0, device=device)
                seq_lengths_local = torch.zeros(0, device=device)
                pg_loss_local = torch.zeros((), device=device)

            if gdpo_cfg.kl_estimator == "tokenwise_k3":
                # Each rank computes its shard's (kl * valid).sum() with
                # grad; the denominator (valid.sum()) we reduce as a
                # detached scalar so divisions across ranks line up.
                if K > 0:
                    BG_local, _ = local_ids.shape
                    x_masked = torch.full_like(local_ids, fill_id)
                    x_masked[:, :D] = MASK_ID
                    t_steps = torch.full(
                        (BG_local,), t_offset, dtype=torch.long, device=device
                    )
                    do_ckpt = (
                        gdpo_cfg.gradient_checkpoint
                        and any(p.requires_grad for p in s3.parameters())
                    )
                    if do_ckpt:
                        def _kl_new_fwd(x_, t_, z_):
                            return s3(x_, t_, z_).float().permute(0, 2, 1)
                        logits_new = torch.utils.checkpoint.checkpoint(
                            _kl_new_fwd, x_masked, t_steps, z_c_rep_local, use_reentrant=False
                        )
                    else:
                        logits_new = s3(x_masked, t_steps, z_c_rep_local).float().permute(0, 2, 1)
                    with torch.no_grad():
                        logits_ref = ref_s3(x_masked, t_steps, z_c_rep_local).float().permute(0, 2, 1)
                    lp_new = F.log_softmax(logits_new, dim=-1).gather(
                        -1, local_ids.unsqueeze(-1)
                    ).squeeze(-1)
                    lp_ref = F.log_softmax(logits_ref, dim=-1).gather(
                        -1, local_ids.unsqueeze(-1)
                    ).squeeze(-1)
                    valid_local = (local_ids != PAD_ID).float()
                    delta = lp_ref - lp_new
                    kl_tokens = torch.exp(delta) - delta - 1.0
                    kl_local_sum = (kl_tokens * valid_local).sum()
                    valid_local_sum = valid_local.sum()
                else:
                    kl_local_sum = torch.zeros((), device=device)
                    valid_local_sum = torch.zeros((), device=device)

                # All-reduce the (detached) denominator
                valid_global_sum_t = valid_local_sum.detach().clone()
                if world_size > 1 and dist.is_available() and dist.is_initialized():
                    dist.all_reduce(valid_global_sum_t, op=dist.ReduceOp.SUM)
                kl_loss_local = kl_local_sum / (valid_global_sum_t + 1e-8)
            elif gdpo_cfg.kl_estimator == "sdmc":
                if K > 0:
                    kl_loss_local = (elbo_ref_local.detach() - elbo_new_local).sum() / BG
                else:
                    kl_loss_local = torch.zeros((), device=device)
            else:
                raise ValueError(
                    f"kl_estimator must be 'tokenwise_k3' or 'sdmc', got {gdpo_cfg.kl_estimator!r}"
                )

            loss_local = pg_loss_local + beta * kl_loss_local

            # ----- 9. Backward + gradient all-reduce + optimizer step -----
            pre_step_params = [p.detach().clone() for p in trainable_params]
            optimizer.zero_grad()
            loss_local.backward()
            _all_reduce_grads(s3, world_size)
            torch.nn.utils.clip_grad_norm_(s3.parameters(), gdpo_cfg.max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                dw_step = torch.sqrt(sum(
                    ((p - pre) ** 2).sum() for p, pre in zip(trainable_params, pre_step_params)
                )).item()
                dw_total = torch.sqrt(sum(
                    ((p - init) ** 2).sum() for p, init in zip(trainable_params, initial_params)
                )).item()
            del pre_step_params

            # ----- 10. Logging aggregation -----
            #
            # Build a small per-rank metrics tensor and sum-reduce it.
            # We aggregate sums (not means) so we can normalize correctly
            # by BG or valid_count on rank 0.
            #
            with torch.no_grad():
                if K > 0:
                    elbo_new_sum_local = elbo_new_local.detach().sum()
                    elbo_old_sum_local = elbo_old_local.sum()
                    log_ratio_seq_sum_local = log_ratio_seq_local.detach().sum()
                    log_ratio_seq_absmax_local = log_ratio_seq_local.detach().abs().max()
                    seq_lengths_sum_local = (local_ids != PAD_ID).float().sum()
                    clip_count_local = (
                        (ratio_seq_local.detach() - 1.0).abs() > eps
                    ).float().sum()
                    pg_sum_local = pg_local_sum.detach() if K > 0 else torch.zeros((), device=device)
                    n_local = torch.tensor(float(K), device=device)
                else:
                    elbo_new_sum_local = torch.zeros((), device=device)
                    elbo_old_sum_local = torch.zeros((), device=device)
                    log_ratio_seq_sum_local = torch.zeros((), device=device)
                    log_ratio_seq_absmax_local = torch.zeros((), device=device)
                    seq_lengths_sum_local = torch.zeros((), device=device)
                    clip_count_local = torch.zeros((), device=device)
                    pg_sum_local = torch.zeros((), device=device)
                    n_local = torch.zeros((), device=device)

                kl_loss_local_scalar = kl_loss_local.detach()
                # kl_loss_local is already normalized to be a per-rank
                # contribution; summing across ranks gives the global KL.

                pack = torch.stack([
                    elbo_new_sum_local,
                    elbo_old_sum_local,
                    log_ratio_seq_sum_local,
                    seq_lengths_sum_local,
                    clip_count_local,
                    pg_sum_local,
                    kl_loss_local_scalar,
                    n_local,
                ])
                if world_size > 1 and dist.is_available() and dist.is_initialized():
                    dist.all_reduce(pack, op=dist.ReduceOp.SUM)
                # For absmax we'd need a separate MAX-reduce.
                if world_size > 1 and dist.is_available() and dist.is_initialized():
                    dist.all_reduce(log_ratio_seq_absmax_local, op=dist.ReduceOp.MAX)

                elbo_new_global_sum = pack[0].item()
                elbo_old_global_sum = pack[1].item()
                log_ratio_seq_global_sum = pack[2].item()
                seq_lengths_global_sum = pack[3].item()
                clip_count_global = pack[4].item()
                pg_global_sum = pack[5].item()
                kl_global_scalar = pack[6].item()
                n_global = pack[7].item()
                log_ratio_seq_global_absmax = log_ratio_seq_absmax_local.item()

                R_mean = R.mean().item()

            # ----- 11. Gather per-replica fields for rank-0 outputs -----
            #
            # Per-replica ELBO + log-ratio + ratio are gathered every step
            # (regardless of debug_log) because they're written to
            # sequences.jsonl — the post-run authority for what each
            # replica generated. The 4 all-reduces are on (BG,) float32 →
            # 6 KB each at BG=1536 → microseconds over CCL.
            #
            # full_ids (BG × L_total long tensor) is bigger and only
            # needed by debug.out, so it stays conditional.
            #
            zero_K_f = torch.zeros(0, dtype=torch.float32, device=device)
            elbo_old_global_t = _all_reduce_per_replica_t(
                elbo_old_local if K > 0 else zero_K_f,
                local_idx_t, BG, device, world_size,
            )
            elbo_new_global_t = _all_reduce_per_replica_t(
                elbo_new_local.detach() if K > 0 else zero_K_f,
                local_idx_t, BG, device, world_size,
            )
            log_ratio_global_t = _all_reduce_per_replica_t(
                log_ratio_seq_local.detach() if K > 0 else zero_K_f,
                local_idx_t, BG, device, world_size,
            )
            ratio_global_t = _all_reduce_per_replica_t(
                ratio_seq_local.detach() if K > 0 else zero_K_f,
                local_idx_t, BG, device, world_size,
            )
            full_ids = None
            if gdpo_cfg.debug_log:
                full_ids = _gather_ids_to_rank0(
                    local_ids=local_ids,
                    local_global_idx=local_global_idx_list,
                    BG=BG, L_total=L_total,
                    rank=rank, world_size=world_size, device=device,
                )

            local_seqs_payload = {
                "global_idx": list(local_global_idx_list),
                "seqs": list(local_seqs),
            }
            if world_size > 1:
                shards = gather_object_to_main(local_seqs_payload)
            else:
                shards = [local_seqs_payload]

            # ----- 12. Rank-0 logging -----
            if rank != 0:
                barrier()
                continue

            # Assemble full seqs in global order.
            seqs: List[Optional[str]] = [None] * BG
            for sh in shards:
                if sh is None:
                    continue
                for i, g_idx in enumerate(sh["global_idx"]):
                    seqs[g_idx] = sh["seqs"][i]
            assert all(s is not None for s in seqs)
            seqs = [str(s) for s in seqs]

            # Per-replica sequence dump goes to debug.out (when debug_log=True).
            # Don't repeat it on stdout — at BG=1536 it would be 1536 lines per step.
            div_stats = diversity_stats(seqs, group_size=G)

            dt = time.time() - t0

            # Global aggregates (already reduced)
            pg_loss_global = pg_global_sum / max(BG, 1)
            loss_global = pg_loss_global + beta * kl_global_scalar
            clip_frac = clip_count_global / max(BG, 1)
            elbo_new_mean = elbo_new_global_sum / max(BG, 1)
            elbo_old_mean = elbo_old_global_sum / max(BG, 1)
            log_ratio_mean = log_ratio_seq_global_sum / max(BG, 1)
            log_ratio_max_abs = float(log_ratio_seq_global_absmax)
            avg_len = float(np.mean([len(s) for s in seqs]))
            all_lengths = [len(s) for s in seqs]
            mean_reward = float(R_mean)
            reward_sum += mean_reward
            reward_avg = reward_sum / step

            rewards_per_replica = [float(x) for x in global_rewards.tolist()]

            logger.info(
                "step=%4d | reward=%5.2f (avg=%5.2f) | loss=%.4f pg=%.4f kl=%.4f clip=%.2f "
                "| elbo_new=%.3f elbo_old=%.3f lr_seq=%.3f (|max|=%.3f) "
                "| div=%.3f (worst_pair=%.3f, uniq=%d) "
                "| dw=%.2e (tot=%.2e) | len=%.2f | %.1fs | ws=%d | all_lengths=%s",
                step, mean_reward, reward_avg,
                loss_global, pg_loss_global, kl_global_scalar, clip_frac,
                elbo_new_mean, elbo_old_mean, log_ratio_mean, log_ratio_max_abs,
                float(div_stats["diversity_mean"]),
                float(div_stats["diversity_min_pair"]),
                int(div_stats["unique_count"]),
                dw_step, dw_total, avg_len, dt, world_size, str(all_lengths),
            )
            row = {
                "step": step,
                "step_time_s": round(dt, 2),
                "reward": round(mean_reward, 3),
                "reward_avg": round(reward_avg, 3),
                "rewards_per_replica": rewards_per_replica,
                "loss": round(loss_global, 5),
                "pg": round(pg_loss_global, 5),
                "kl": round(kl_global_scalar, 5),
                "clip_frac": round(clip_frac, 4),
                "elbo_new": round(elbo_new_mean, 4),
                "elbo_old": round(elbo_old_mean, 4),
                "log_ratio_seq": round(log_ratio_mean, 4),
                "log_ratio_seq_max_abs": round(log_ratio_max_abs, 4),
                "dw_step": dw_step,
                "dw_total": dw_total,
                "avg_len": round(avg_len, 1),
                "lengths_per_replica": all_lengths,
                "batch_prompts_idx": [int(i) for i in batch_prompts_idx],
                "diversity_mean": round(float(div_stats["diversity_mean"]), 4),
                "diversity_min_pair": round(float(div_stats["diversity_min_pair"]), 4),
                "unique_count": int(div_stats["unique_count"]),
                "per_replica_diversity": [float(x) for x in div_stats["per_replica_diversity"]],
            }
            log_rows.append(row)
            try:
                write_train_log_atomic(log_path, log_rows)
            except Exception as e:  # pragma: no cover
                logger.warning("train_log.json flush failed (non-fatal): %s", e)

            # Append this step's BG sequences (one row per replica) to
            # sequences.jsonl — the post-run artifact for what each
            # replica generated. Join back to prompts.json via
            # batch_prompts_idx in this step's train_log.json row.
            try:
                _write_step_sequences(
                    path=sequences_path,
                    step=step,
                    seqs=seqs,
                    rewards=rewards_per_replica,
                    advantages=adv_full.detach().cpu().tolist(),
                    elbo_new=elbo_new_global_t.cpu().tolist(),
                    elbo_old=elbo_old_global_t.cpu().tolist(),
                    log_ratio_seq=log_ratio_global_t.cpu().tolist(),
                    ratio_seq=ratio_global_t.cpu().tolist(),
                    G=G,
                    world_size=world_size,
                )
            except Exception as e:  # pragma: no cover
                logger.warning("sequences.jsonl write failed (non-fatal): %s", e)

            if gdpo_cfg.debug_log and full_ids is not None:
                # Per-corruption "mask visualization" section of debug.out is
                # omitted under distributed training — rank 0 doesn't have
                # all the other ranks' SDMC corruptions (and gathering them
                # is pointless: they're random masks generated per rank).
                # All per-replica scalars ARE gathered properly via
                # per-replica all-reduces above so the per-replica table
                # in debug.out is faithful.
                try:
                    seq_lengths_t = (full_ids != PAD_ID).float().sum(dim=1).clamp(min=1.0)
                    _write_debug_step(
                        debug_path=debug_path,
                        step=step,
                        batch_prompts=batch_prompts,
                        ids_all=full_ids,
                        seqs=seqs,
                        rewards_raw=rewards_per_replica,
                        adv=adv_full.detach(),
                        elbo_old=elbo_old_global_t,
                        elbo_new=elbo_new_global_t,
                        log_ratio_seq=log_ratio_global_t,
                        ratio_seq=ratio_global_t,
                        seq_lengths=seq_lengths_t,
                        components_per_replica=None,
                        corruptions=[],
                        G=G,
                        per_replica_diversity=div_stats["per_replica_diversity"],
                    )
                except Exception as e:  # pragma: no cover
                    logger.warning("debug.out write failed (non-fatal): %s", e)

            if step % gdpo_cfg.save_steps == 0:
                ckpt_path = os.path.join(gdpo_cfg.output_dir, f"step{step}.pt")
                torch.save({"step": step, "model_state": s3.state_dict()}, ckpt_path)
                logger.info("Saved checkpoint: %s", ckpt_path)

            # Hold all ranks at end-of-step.
            barrier()

        # ----- End of training: rank 0 finalizes, others exit clean -----
        if rank == 0:
            final_path = os.path.join(gdpo_cfg.output_dir, "final.pt")
            torch.save({"step": gdpo_cfg.steps, "model_state": s3.state_dict()}, final_path)
            write_train_log_atomic(log_path, log_rows)
            logger.info("GDPO multinode done. Final checkpoint: %s", final_path)
            try:
                from biom3.rl.plotting import plot_train_log
                plot_train_log(log_path, gdpo_cfg.output_dir, algo="gdpo")
            except Exception as e:  # pragma: no cover
                logger.warning("plot_train_log failed (non-fatal): %s", e)

        if rollout_pool is not None:
            rollout_pool.shutdown()

        barrier()
        return log_rows if rank == 0 else None
    except BaseException as e:
        import traceback
        # Failure-fast: log, best-effort cleanup of the rollout pool +
        # process group, then ``os._exit`` so we bypass Python's atexit
        # / destructor unwinding (which can hang on XPU under OOM and
        # leave peers waiting for the default 30-min collective timeout).
        # mpiexec sees the non-zero exit and kills the remaining ranks.
        logger.error(
            "rank %d caught fatal error in training: %s: %s\n%s",
            rank, type(e).__name__, str(e), traceback.format_exc(),
        )
        try:
            if rollout_pool is not None:
                rollout_pool.shutdown()
        except Exception:
            pass
        try:
            if dist.is_available() and dist.is_initialized():
                dist.destroy_process_group()
        except Exception:
            pass
        os._exit(1)

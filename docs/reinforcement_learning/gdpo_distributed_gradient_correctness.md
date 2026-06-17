# GDPO multinode trainer — distributed gradient correctness

## TL;DR

The multinode GDPO trainer (`src/biom3/rl/gdpo_multinode.py`) shards the BG replicas across `world_size` ranks. Each rank computes the gradient of its shard's contribution to the loss; gradients are all-reduced before the optimizer step. The all-reduced gradient is **mathematically identical** to what a single process would compute on the full BG batch — modulo bit-level FP non-determinism in CCL's all-reduce, which is negligible in practice.

This note walks through why each component of the GDPO loss survives the shard-and-all-reduce treatment.

## The structural property that makes this work

The single-process GDPO loss has the shape:

```
L(θ) = (1/BG) · Σₖ pg_term_k(θ)   +   β · (Σₖ kl_token_term_k(θ) · valid_k) / Σₖ valid_k
```

Every per-replica term `pg_term_k(θ)` and `kl_token_term_k(θ)` depends only on `θ` and replica `k`'s data — there is no cross-replica coupling inside any single term. That's what makes the shard-then-reduce pattern work.

Differentiating:

```
∂L/∂θ = (1/BG) · Σₖ ∂pg_term_k/∂θ
      + β · (Σₖ ∂kl_token_term_k/∂θ · valid_k) / Σₖ valid_k
```

A sum-of-per-replica-gradients with a constant scalar denominator. Sharding the sum and all-reducing the per-shard gradients recovers the exact full-batch gradient.

## Term-by-term

### PG-clip loss

[`src/biom3/rl/gdpo_multinode.py`](../../src/biom3/rl/gdpo_multinode.py) — step 8 in the per-step body.

Each rank `R` owns a replica set `S_R ⊂ {1..BG}` (round-robin) and computes:

```
pg_loss_local_R = (1/BG) · Σ_{k ∈ S_R} max(-adv_k · r_k, -adv_k · clip(r_k, 1±ε)) / seq_len_k
```

The division by `BG` (not by `|S_R|`) is the key — it makes the per-rank contributions sum to the global mean:

```
Σ_R pg_loss_local_R = (1/BG) · Σ_{k=1..BG} max(...) / seq_len_k = pg_loss_single_process
```

Therefore:

```
∂(Σ_R pg_loss_local_R)/∂θ = Σ_R ∂pg_loss_local_R/∂θ
```

The gradient all-reduce sums the per-rank gradients, recovering `∂pg_loss_single_process/∂θ` exactly. **No approximation, identity.**

### KL `tokenwise_k3` term

Same file, same step.

Each rank `R` computes:

- `kl_local_sum_R = Σ_{k ∈ S_R, j ∈ [0,L)} kl_tokens_{k,j} · valid_{k,j}` — has grad through `s3`.
- `valid_local_sum_R = Σ valid_{k,j}` — pure scalar, no grad.

The denominator is all-reduced first, **detached** so no gradient flows through it:

```
valid_global = Σ_R valid_local_sum_R   (detached)
```

Each rank's loss contribution:

```
kl_loss_local_R = kl_local_sum_R / valid_global
```

Sum across ranks:

```
Σ_R kl_loss_local_R = (Σ_R kl_local_sum_R) / valid_global
                    = kl_total_sum / valid_global
                    = kl_loss_single_process
```

Same gradient-identity argument as PG: the gradient all-reduce on parameters gives the exact single-process gradient.

**Why the KL forward is inlined in `gdpo_multinode.py`, not calling `_tokenwise_k3_kl`:** the original helper in `gdpo.py` returns the already-normalized scalar `(kl_sum / valid_sum)`. The distributed version needs the un-normalized `kl_local_sum` and a separately-reduced `valid_global` so the division across the shard boundary works out. If you change the KL math in one file, mirror it in the other (or refactor to a shared helper that returns both pieces).

### Group-relative advantage

GDPO advantages are `A_k = R_k − mean(R_{g(k)})` where `g(k)` is the prompt of replica `k`. The mean is over the `G` replicas of that prompt.

This is **not** sharded computation. The trainer:

1. Each rank fills its owned rewards into a zero-initialized `(BG,)` tensor.
2. `dist.all_reduce(SUM)` → every rank has the full `R[BG]` vector.
3. Every rank computes `adv_full = (Rg − Rg.mean(-1, keepdim=True)).view(BG)` from the global rewards — identically.
4. Each rank takes `adv_local = adv_full[local_idx_t]` for its own pg_loss.

So `adv_k` for any replica is computed from **all G** of its group's rewards — identical to single-process. No information loss from the sharding.

The all-reduce on rewards is the one fundamental piece of cross-rank coordination the algorithm requires. The reward payload is `BG · float32` (≤ a few KB), so this is essentially free.

### Per-replica policy ratio and the "shared corruptions" trick

`ratio_seq_k = exp(elbo_new_k − elbo_old_k.detach())`.

The variance-reduction trick in [`_build_shared_corruptions`](../../src/biom3/rl/gdpo.py) uses the **same** mask realizations for `elbo_old_k` and `elbo_new_k` on the same replica. The mask-sampling noise then cancels in the ratio, leaving only the policy difference — much lower variance than independent draws.

In the distributed trainer, each rank generates corruptions for its own shard using its local torch RNG (`torch.manual_seed(base_seed + rank * 1_000_003 + step)` makes them differ across ranks per step).

What's preserved:

- **Within a rank**, the same corruptions are used for `elbo_old` and `elbo_new` on that rank's replicas → per-replica ratio noise cancels.

What's different from single-process:

- **Across ranks**, the corruption realizations differ. But each replica is owned by exactly one rank, so its ratio uses one corruption set. There is no cross-replica corruption sharing in the single-process math either.

So the variance-reduction property is preserved at the per-replica level, which is exactly what the algorithm depends on. The distributed estimator has the same statistical properties as single-process; the per-replica random seeds just happen to be drawn on different ranks.

## Why every rank stays bit-synced

For the gradient all-reduce to give every rank the same updated weights, every rank must hold identical `s3` weights and identical optimizer state going into each step. The trainer guarantees this:

1. **Init:** every rank loads `s3` from the same `stage3_init_weights` file. As a defense against any model-construction RNG drift (e.g., unloaded params picking up per-rank torch seeds), the trainer calls `_broadcast_state(s3, src=0)` immediately after init. After this, all ranks are bit-identical.
2. **Optimizer state:** AdamW starts at zeros on every rank.
3. **Per step:** the gradient all-reduce gives every rank the same gradient. AdamW is deterministic given identical state and identical input, so all ranks produce identical updated weights. Identical `s3` going into the next step.

**Theoretical risk:** floating-point sum order in `dist.all_reduce(SUM)` could differ across ranks for non-associative FP. In practice CCL/xccl ring-reduce or tree-reduce produces bit-identical results across ranks on the same topology. If drift were ever a concern, a diagnostic `dist.all_reduce(MAX) - dist.all_reduce(MIN)` on any parameter would catch it directly — every parameter should reduce to itself.

## What would actually break correctness

For posterity, the failure modes I specifically watched out for during implementation:

- **All-reducing a non-detached denominator** — would let gradient flow through `valid_global` and double-count it. Fix: `valid_local_sum.detach().clone()` before the all-reduce.
- **Normalizing pg_loss by `len(local_shard)` instead of `BG`** — would give the gradient of `sum of per-rank means`, not the gradient of the mean. The trainer divides by `BG`.
- **Skipping the gradient all-reduce when `world_size == 1`** — the helper handles this as a no-op so single-process invocations still work.
- **All-reducing on a condition that diverges across ranks** — guaranteed deadlock. Per-replica collectives in the trainer are only gated on `gdpo_cfg.debug_log` (rank-uniform), never on `rank`.
- **Forgetting that `s3` could drift across ranks** — defended by the init broadcast and the bit-deterministic per-step path. Diagnosable via any-param all-reduce-min/max check.

## Sanity-check recipe

If you ever want to verify correctness empirically:

1. Single-process run: `biom3_gdpo_train --steps 1 --batch_size 1 --num_generations 24 --seed 42 ...` (no mpiexec).
2. 2-rank distributed run: `WORLD_SIZE=2 mpiexec -n 2 ... biom3_gdpo_train --steps 1 --batch_size 1 --num_generations 24 --seed 42 ...`.

The updated `s3` weights after step 1 should be bit-identical between the two runs (modulo any genuine CCL FP non-determinism, which on Aurora xccl I expect to be zero). Compare via `torch.allclose(sd_a, sd_b, atol=0)` parameter-by-parameter.

Note that the per-rank torch RNG seed (`base_seed + rank * 1_000_003 + step`) makes the rollout sampling differ across ranks in the distributed run, so the *rolled-out sequences* will be different than the single-process run. But for a parity check at fixed BG, what matters is that the gradient on a fixed batch is identical, not that the rollout is identical. If you want full bit-for-bit reproducibility, force every rank to use the same torch seed (defeats the point of distributing).

# 2026-04-29 — GDPO trainer for Stage 3 (sibling of GRPO)

Implemented Group Diffusion Policy Optimization (Rojas et al., ICLR 2026, arXiv [2510.08554v3](https://arxiv.org/abs/2510.08554)) as a sibling RL fine-tuner to the existing GRPO module. GDPO replaces diffu-GRPO's biased one-step token-level mean-field log-prob with a sequence-level ELBO estimated via Semi-deterministic Monte Carlo (SDMC). For ProteoScribe's order-agnostic absorbing-state diffusion, the SDMC integrand is exactly the per-time conditional log-prob over currently-masked positions — the quantity `Stage3.transformer_training_helper.cond_elbo_objective` already computes for training, so the implementation is mostly composition over existing utilities.

Branch: `addison-hack-grpo`. No worktree this session — small, scoped addition next to existing GRPO code.

## What landed

### Core trainer

```
src/biom3/rl/gdpo.py                 # GDPOConfig + _build_grid + _build_shared_corruptions
                                     # + _elbo_sdmc + _tokenwise_k3_kl + gdpo_train
src/biom3/rl/run_gdpo_train.py       # CLI runner (mirror of run_grpo_train.py)
src/biom3/rl/__main__.py             # +run_gdpo_train wrapper
src/biom3/rl/plotting.py             # plot_train_log: per-run diagnostic figures
                                     # (shared by both GRPO and GDPO)
pyproject.toml                       # +biom3_gdpo_train entry point
configs/grpo/example_gdpo.json       # smoke config (G=4, N=3)
configs/grpo/production_gdpo.json    # production config (G=8, N=7)
scripts/gdpo_train_singlenode.sh
scripts/run_gdpo_smoke_5steps.sh
tests/rl_tests/test_gdpo_smoke.py    # 8 CPU smoke tests on the mini Stage 3 fixture
docs/gdpo_finetuning.md              # user-facing how-to and algorithm explainer
```

### Algorithm details

- **Sequence-level ELBO via SDMC** (paper Eq. 5): for each rolled-out sequence, evaluate the integrand at `N` deterministic time-points `t_n ∈ (0, 1]` with weights `w_n` summing to 1; at each `t_n` draw `inner_mc` masked corruptions `y_t ~ π_t(·|y)` and compute `(1/t_n) · Σ_i 1[y_t^i = M] log π_θ(y^i | y_t, q)`. The deterministic time grid kills the dominant variance source (paper Fig. 2a: ~96% of ELBO variance comes from random `t`).
- **Quadrature:** uniform midpoint rule on `(0, 1]` by default — `t_n = (n - 0.5)/N`, `w_n = 1/N`. Explicit `t_n`/`w_n` lists also supported via CLI/config. We don't use Gauss-Legendre or other fancy schemes — paper says Riemann is fine for `N ≥ 2`.
- **paper-`t` ↔ model-`idx` mapping:** ProteoScribe's `idx ∈ [0, L-1]` is the count of revealed positions. Paper-`t` is the fraction *masked*, so `idx_n = clamp(round((1 - t_n) · L), 0, L-1)`. With `N=3, L=128`: `idx_grid = [107, 64, 21]` (≈83%, 50%, 17% revealed).
- **Sequence-level PPO-clip** (paper Eq. 6): `r_g = exp(ELBO_new(y_g) − ELBO_old(y_g))` is a scalar per sequence (not per-token). Loss is `(1/G) Σ_g (1/|y_g|) min(r_g A_g, clip(r_g, 1±ε) A_g) − β·KL`.
- **Unnormalized advantage** `A_g = R_g − mean(R)` (paper-faithful, Liu et al. 2025b). `--advantage_normalize` switch to recover GRPO's `(R-μ)/σ` for parity testing.
- **`π_old` snapshot per outer iteration** (paper Alg. 1 line 3) — distinct from frozen `π_ref` (taken once at init). `--no_old_policy_snapshot` collapses to the GRPO behavior of using `π_ref` as `π_old`.
- **Shared corruptions across `π_old`/`π_new`/`π_ref`** within a step. Without this, the importance ratio is dominated by mask-sampling noise instead of policy difference — was the single most subtle correctness point.
- **KL term:** `tokenwise_k3` (cheap, single fully-masked forward, mirrors GRPO's KL) is the default; `sdmc` (uses `ELBO_ref - ELBO_new` at the same grid, one extra ELBO pass) is the principled option.

### Side finding (existing GRPO bug, not fixed in this commit)

`src/biom3/rl/grpo.py:178` `_policy_logprobs` builds the fully-masked input as `torch.full_like(ids, pad_id)` with `pad_id = 23`, but the model's mask/absorbing-state token is **0**, not PAD. This means the diffu-GRPO port has been feeding the model an all-PAD prefix at `t=0` rather than all-MASK whenever it computes per-token log-probs (used for both the PPO ratio and the KL term in `grpo_train`). Did a deep audit of the rest of the codebase — confirmed the bug is **isolated** to that one helper. Stage 3 training, sampling, the pre-unmask feature, and the new GDPO module all correctly use 0 (MASK) for absorbing-state input. PAD-as-PAD usages elsewhere (length masks, decode stripping, ESM-2 mean-pool) are legitimate.

The fix is one line at `grpo.py:178` — flagged separately, not bundled into this commit so the GDPO change is reviewable on its own.

### Logging and diagnostics

- **Per-step `train_log.json` rows** now include `rewards_per_replica` (raw scalar reward per replica) and `components_per_replica` (when reward is `CompositeReward`). Both GRPO and GDPO. Backward-compatible — existing fields untouched.
- **End-of-run auto-plot** via `biom3.rl.plotting.plot_train_log`: 6-panel grid (reward + per-replica scatter, loss components, clip frac, weight drift, sequence length, reward histogram); GDPO adds two more panels (per-sequence ELBO, sequence-level log-ratio). Composite rewards get a second figure with one panel per component. Best-effort — wrapped in `try/except`, never aborts training.
- **`debug.out` per-step plaintext dump** (`debug_log: true` by default). Per step: prompt, per-replica scalar table (reward / advantage / `elbo_old` / `elbo_new` / `log_ratio` / `ratio` / valid-length), decoded sequences, raw token-id sequences with `_` for any leftover mask, and for each SDMC corruption a same-length string visualization (`_` = masked, AA letter = revealed) for every replica. File is truncated at trainer start and opened in append mode each step so a crash leaves prior steps' diagnostics intact. Disable with `--no_debug_log`.

### Tests

`tests/rl_tests/test_gdpo_smoke.py` — 8 CPU tests against the mini Stage 3 fixture (`tests/_data/models/stage3/`):

- `test_build_grid_uniform_midpoints` — validates `t_n = (n - 0.5)/N`, `w_n = 1/N`, and `idx_n` rounding.
- `test_build_grid_explicit_with_weights` and `_default_weights` — explicit grid path.
- `test_build_grid_rejects_out_of_range` — guards `t ∈ (0, 1]`.
- `test_elbo_sdmc_shape_grad_finiteness` — ELBO is `(BG,)`, `≤ 0`, finite, gradient flows to Stage 3 params.
- `test_shared_corruptions_use_mask_token` — masked positions equal `MASK_ID = 0`; revealed equals `ids`; mask count = `L − idx_n`.
- `test_seq_log_ratio_zero_at_init` — when `π_new == π_old`, the per-sequence log-ratio is identically zero (sanity check on first iteration).
- `test_tokenwise_k3_kl_zero_when_policies_match` — same idea for the KL term.
- `test_gdpo_inner_step_updates_params` — full mock GDPO step; loss is finite, params move under AdamW.

User confirmed all 8 tests pass.

## Design decisions worth recording

- **Algorithm 1 in the paper has a notation typo**: lines 7 and 10 both bind a counter named `n`, with different bounds (`μ` for inner gradient updates vs. `N` for quadrature points). Resolved: the inner `n` (line 10, used in `(t_n, w_n)`) is the gridpoint; the outer should be `m` or unindexed. Our implementation follows the inner meaning (`n` is the quadrature index in `_build_shared_corruptions` and `_elbo_sdmc`). We currently have **no `μ`-loop** — one gradient step per rollout (`μ=1` implicitly). Adding `--mu_inner` for PPO-style amortization is straightforward but deferred until we want it.
- **No backtracking / no hyperparameter `θ_old.detach()` outside the loss**: the snapshot is a `copy.deepcopy(s3)` taken at the top of each outer step, so `elbo_old` is computed under no-grad against a fully-detached parameter set — no need for autograd tricks.
- **Quadrature `t_n` clamped to `(0, 1]`**, not `[0, 1]`. The integrand has a `1/t` factor which blows up at `t=0`. `eps_t = 1e-3` clamp is a safety net; uniform midpoint on `(0, 1]` never gets close.
- **Production config** (`G=8`, `N=7`, `L=128`): per-step forwards ≈ `L (rollout) + N·G·2 (elbo_old + elbo_new)` = 128 + 112 = ~240 per step. Reasonable on one Aurora tile.

## Reference

- Rojas, Lin, Rasul, Schneider, Nevmyvaka, Tao, Deng. *Improving Reasoning for Diffusion Language Models via Group Diffusion Policy Optimization.* ICLR 2026. arXiv [2510.08554v3](https://arxiv.org/abs/2510.08554). Paper PDF kept at repo root: `2510.08554v3.pdf`.
- Plan file: `~/.claude/plans/plan-this-out-in-serene-riddle.md`.
- Companion docs: [docs/gdpo_finetuning.md](../gdpo_finetuning.md) (user-facing how-to + algorithm explainer), [docs/grpo_finetuning.md](../grpo_finetuning.md) (sibling reference).

## Late-session additions (after the initial smoke run)

A first production attempt against `production_gdpo.json` (G=8, N=7, L=1024, no pre_unmask) OOM'd on the Aurora tile inside `_tokenwise_k3_kl` after `elbo_old` and `elbo_new` had already completed. Diagnosis: cumulative autograd activations from 56 grad-retaining ELBO forwards plus one more full-sequence forward in the KL term exceeded the tile's HBM. Two compounding fixes landed:

### Gradient checkpointing on the trainable ELBO

[src/biom3/rl/gdpo.py](../../src/biom3/rl/gdpo.py) — `_elbo_sdmc` now wraps each per-corruption forward in `torch.utils.checkpoint.checkpoint(use_reentrant=False)` when `gradient_checkpoint=True` and grad is enabled. Replaced the `cond_predict_conditional_prob` + `OneHotCategorical` path with an inline `F.log_softmax + gather`-based `_per_corruption_logprob_sum` so the checkpointed function returns plain tensors. Peak activation memory drops from `N · inner_mc` forwards' worth to one forward's worth at the cost of a recompute pass in backward.

`GDPOConfig.gradient_checkpoint` defaults to `True`. CLI: `--no_gradient_checkpoint` to disable.

### Pre-unmask integration

The user pointed out that SH3 sequences are ~60 AA and the architectural `L = 1024` is hugely wasteful. Wired the existing `Stage3.run_ProteoScribe_sample` pre-unmask flag into GDPO:

- `GDPOConfig.pre_unmask: bool` and `pre_unmask_config: Optional[str]`.
- New `_gdpo_rollout` that calls `_build_initial_mask_state` (imported from `run_ProteoScribe_sample`) so positions `[D, L_total)` start at PAD and stay there; sampling path permutes only `[0, D)`.
- `_build_shared_corruptions(diffusion_budget=D, ...)` now permutes only `[0, D)` and treats positions `[D, L_total)` as permanently revealed (PAD stays PAD).
- `_tokenwise_k3_kl(diffusion_budget=D, fill_id=...)` builds an all-mask prefix only over `[0, D)` with PAD at the tail.
- `gdpo_train` resolves pre-unmask state on `cfg3` exactly like `run_ProteoScribe_sample.main` (overriding `cfg3.diffusion_steps` with the budget D, capturing the original value as `cfg3.sequence_length`).

New file: [configs/grpo/pre_unmask_sh3.json](../../configs/grpo/pre_unmask_sh3.json) — `{strategy: last_k, fill_with: PAD, diffusion_budget: 128}`. Both `example_gdpo.json` and `production_gdpo.json` now point at it.

Compute impact at production settings (G=8, N=4, D=128, L_total=1024): rollout drops from 1024 to 128 iterations (8×); ELBO forwards still operate on tensors of shape `(BG, 1024)` but the `1/t` factor in the integrand divides over D=128 masked positions instead of L_total=1024, so ELBO values are better-conditioned.

### Reward + plotting + debug logging (mid-session, before the production OOM)

Three additions to make runs interpretable:

- **`rewards_per_replica`** and **`components_per_replica`** added to every per-step row of `train_log.json`. Backward-compat — existing fields (`reward`, `reward_avg`, `components` mean) untouched. Same change applied to `grpo.py` so the two trainers share schema.
- **End-of-run diagnostic plots** via `biom3.rl.plotting.plot_train_log` — 6-panel (reward + per-replica scatter, loss, clip frac, weight drift, length, reward histogram), GDPO adds 2 more (per-sequence ELBO, sequence-level log-ratio). Composite rewards get a second figure with one panel per component. Wrapped in try/except — never aborts a run.
- **Per-step `debug.out`** (`debug_log: true` default) — prompts, per-replica scalar table (reward / advantage / `elbo_old` / `elbo_new` / `log_ratio` / `ratio` / valid-length), decoded sequences, raw token-id sequences with `_` for any leftover mask, and for each SDMC corruption a same-length string visualization (`_` = masked, AA letter = revealed) for every replica. File truncated at trainer start; opened in append mode each step so a crash leaves prior steps intact. CLI: `--no_debug_log` to disable.

### Plan-file reuse

The user's `/loop`-style review of Algorithm 1 caught a real notation typo in the paper: lines 7 and 10 both bind a counter `n` (μ inner gradient updates vs. N quadrature points). Resolved that the inner `n` is the gridpoint; the outer should be `m` or unindexed. Documented inline in `gdpo.py` near the SDMC loop. We currently implement μ=1 implicitly — see follow-ups below.

### Production observation: log_ratio_seq ≡ 0

User noticed during production that `log_ratio_seq` stays at 0. Confirmed this is **expected** at μ=1: `π_old` is snapshotted at the start of each outer iteration and only one gradient step is taken before the next snapshot, so `π_new == π_old` *in value* at the moment `elbo_new` is computed (autograd graph is still attached, gradient still flows correctly). At μ=1 the algorithm is mathematically equivalent to "REINFORCE on the SDMC-estimated sequence-level log-likelihood with group-advantage centering" — the PPO-clip is dormant. `dw_step` and `reward` are the load-bearing signals to watch, not `clip_frac` or `log_ratio_seq`.

## Follow-up items and potential improvements

Roughly ranked by expected ROI:

### Correctness / faithfulness

- **Implement `mu_inner` loop** (paper Alg. 1 line 7). With μ > 1, `π_old` stays fixed across multiple gradient steps while `π_θ` moves, giving non-degenerate `log_ratio_seq` and activating the PPO-clip. Also amortizes the rollout cost (the most expensive part of a step under L=1024 / D=128). Estimated ~10 LOC: wrap the loss / backward / step block in `for _ in range(gdpo_cfg.mu_inner)` and pull rollout + `elbo_old` outside the inner loop. Safest default: `mu_inner=4` (typical PPO).
- **Fix the standing GRPO bug at `grpo.py:178`** — `_policy_logprobs` builds the fully-masked input as PAD (`pad_id=23`) instead of MASK (`0`). One-line fix, deferred so it lands in its own commit. Worth testing the GRPO baseline before/after to see how much this changes diffu-GRPO behavior.
- **`kl_estimator="sdmc"` validation.** We support it but haven't run a comparison against `tokenwise_k3`. The SDMC KL is more principled (uses the same SDMC integral as the ratio) but doubles the no-grad ELBO budget. One ablation run would tell us whether it stabilizes training.
- **Prompt sampling.** Currently picks one prompt per step uniformly at random with replacement; for `B=1` and a single prompt file, every step uses the same prompt. For multi-prompt training, the random choice without prompt-bucketing means the running reward average is noisy across prompt difficulty. Consider epoch-style iteration over prompts.

### Performance

- **Pre-unmask numerics.** Setting `cfg3.diffusion_steps = D = 128` while the model was trained with `num_timesteps = 1024` changes the absolute time embedding (`t / num_timesteps · rescale_steps`). The user reports `run_ProteoScribe_sample.py` produces sane sequences in this configuration, but worth confirming the same holds for ELBO numerics — particularly that the `1/t_n` factor at small `t_n` doesn't blow up. Add a sanity test that `_elbo_sdmc` under pre-unmask matches direct `cond_elbo_objective` evaluation on the same sequences.
- **Larger inner_mc for bias measurement.** Paper Fig. 3 shows N=2–3 with K=1 already beats double-MC. We default to `inner_mc=1`. A quick ablation at `inner_mc=2,4,8` would tell us how stable the gradient is in our setting (long SH3 sequences may behave differently from natural-language).
- **Mixed precision on the trainable ELBO.** `_inference_autocast` already routes the rollout through bf16; the trainable ELBO doesn't. With L=1024 forwards under autograd, bf16 + GradScaler-equivalent could halve memory and speed up 2×. Needs care around the `1/t` factor and the log-softmax stability.

### Diagnostics

- **Plot the SDMC time grid in `train_diagnostics.png`.** A sub-panel showing where the quadrature points land on `[0, 1]` would make `t_n` choices auditable from the figure alone.
- **Per-quadrature-point ELBO contributions.** Currently we report the aggregated per-sequence ELBO. Surfacing each `t_n`'s contribution would let us see whether one timestep dominates the gradient (a known SDMC failure mode).
- **`debug.out` at this granularity is large** (~50 KB/step at production settings). Add a config knob for "verbosity": full dump every K steps and only the per-replica table on intermediate steps.

### Tests

- **Pre-unmask test.** `tests/rl_tests/test_gdpo_smoke.py` doesn't currently exercise pre-unmask. Add a test that flips `pre_unmask=True` with a tiny budget and confirms (a) tail positions stay at PAD throughout the rollout, (b) corruptions never mask the tail, (c) ELBO matches a hand-computed value on the trimmed prefix.
- **Sequence-level log-ratio is exactly 0 at μ=1**: covered structurally by `test_seq_log_ratio_zero_at_init` but worth adding a dedicated test that asserts the property holds across a few outer iterations as well.

### Documentation

- **`docs/gdpo_finetuning.md` doesn't yet document `pre_unmask`, `gradient_checkpoint`, or the μ=1 caveat.** Should be updated when the μ_inner loop lands.
- **Side-by-side GRPO vs GDPO benchmark.** Once both algorithms are stable (post the GRPO mask-id fix and μ_inner addition), run them on the same prompts/seed and report reward trajectories for the README.

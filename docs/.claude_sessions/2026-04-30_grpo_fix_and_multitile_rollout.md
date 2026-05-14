# 2026-04-30 — GRPO PAD-vs-MASK fix, multi-tile rollout pool, atomic train_log

Continuation of [2026-04-29_gdpo_implementation.md](2026-04-29_gdpo_implementation.md). After the v01 GDPO production run we identified mode collapse + a single-tile compute ceiling as the two open issues. Today landed Phase B of the [diversity + multi-tile plan](../../home/.claude/plans/plan-this-out-in-serene-riddle.md) (multi-tile rollout) plus a separate cluster of GRPO observability and correctness changes that bring GRPO to feature-parity with GDPO.

Branch: `addison-hack-grpo`. Single commit.

## Headline changes

### 1. Multi-tile rollout pool for GDPO (Phase B of the plan)

New [src/biom3/rl/rollout.py](../../src/biom3/rl/rollout.py): `RolloutPool` class. Single Python process, one trainable master on `xpu:0` with optimizer + autograd, frozen replicas on `xpu:1..N-1` for parallel rollout, gather to master, ELBO/backward stay single-device. Persistent `ThreadPoolExecutor`, deep-copied per-tile cfg3 (avoids the [gdpo.py:560 cfg3.device mutation race](../../src/biom3/rl/gdpo.py#L560) the explorer flagged). Single-device path short-circuits to direct `_gdpo_rollout` call — zero behavior change for existing users.

[gdpo.py](../../src/biom3/rl/gdpo.py) wires `GDPOConfig.rollout_devices`, `_resolve_rollout_devices` helper (`None | "auto" | explicit list`), and `pool.sync_from(old_s3) → pool.rollout(z_c, G)` per outer iter. `pool.shutdown()` on exit.

[run_gdpo_train.py](../../src/biom3/rl/run_gdpo_train.py) gets `--rollout_devices` (comma-separated or `auto`).

New tests in [tests/rl_tests/test_rollout_pool.py](../../tests/rl_tests/test_rollout_pool.py) — 8 CPU smoke tests covering even-split, device-resolver, single-device parity (bit-identical to direct call with same seed), 2-thread dispatch, replica replication after `sync_from`, master-cfg-not-mutated invariant. All pass.

New [scripts/run_gdpo_smoke_multixpu.sh](../../scripts/run_gdpo_smoke_multixpu.sh) and [jobs/aurora/gdpo/job_gdpo_production_v02_multixpu.pbs](../../jobs/aurora/gdpo/job_gdpo_production_v02_multixpu.pbs).

**Gotcha caught during planning**: Aurora is **12 tiles per node** (6 GPUs × 2 tiles per GPU), not 6. `--rollout_devices auto` calls `torch.xpu.device_count()` which returns 12 on a full-node allocation, so the pool spins up 12 worker threads automatically; the PBS `unset ZE_AFFINITY_MASK` is the load-bearing bit. Updated comments and G sizing to reflect 12 tiles.

**GIL ceiling discussion**: realistic speedup at 12 tiles is 3-5×, not 12×, because PyTorch holds the GIL during kernel-launch dispatch. The `mpiexec`-with-one-process-per-tile pattern (matching `run_PL_training.py`) is the standard escape hatch and is on the follow-up list, but we deferred it pending v02 measurement: if we observe the GIL ceiling biting at 12 tiles, we'll add an MPI-backed rollout pool as Phase B.5 with the same external API.

### 2. GRPO PAD-vs-MASK bug fix + observability parity

Long-standing bug at [grpo.py:178](../../src/biom3/rl/grpo.py#L178): `_policy_logprobs` constructed the fully-masked input as `torch.full_like(ids, pad_id)` with `pad_id = 23` (`<PAD>`), but the model's mask / absorbing-state token is **0**. The diffu-GRPO port has been feeding the model an all-`<PAD>` prefix at `t=0` rather than all-MASK whenever it computes per-token log-probs (used for both the PPO ratio and the KL term in `grpo_train`).

Confirmed via the explorer audit on 2026-04-29 that the bug was **isolated** to that one helper — Stage 3 training, sampling, the pre-unmask feature, and GDPO all use 0 (MASK) correctly.

Fix:
- Added `MASK_ID = 0` constant at [grpo.py](../../src/biom3/rl/grpo.py).
- Renamed `_policy_logprobs(..., pad_id)` → `_policy_logprobs(..., mask_id: int = MASK_ID)`. Default to MASK so future callers can't reintroduce the bug.
- Both call sites at the previous `lp_ref_tok = ...` and `lp_new_tok = ...` lines now pass `MASK_ID` instead of `PAD_ID`. The `valid = (ids_all != PAD_ID).float()` line below stays unchanged — that one correctly uses PAD as a length mask, not as a model fill.
- New test `test_policy_logprobs_default_uses_mask_id` in [tests/rl_tests/test_grpo_smoke.py](../../tests/rl_tests/test_grpo_smoke.py) — load-bearing assertion that the default fill is MASK and that a PAD fill produces a *different* tensor (proves the fix isn't cosmetic).

**Reward trajectories from old GRPO logs are NOT directly comparable to post-fix runs.** The model is being fed the right prefix now. The new `_meta` row carries `diffu_grpo_style: true` so consumers can tell the two apart.

GRPO was missing the GDPO-side observability that landed yesterday; brought to parity:

- **`_meta` row** at the top of `train_log.json` with `algo: grpo`, `diffu_grpo_style: true`, K/B/β/ε/L, `advantage_normalize: true`.
- **Per-step token-level log-ratio metrics**: `log_ratio_tok` (mean over valid positions of `lp_new − lp_ref`), `log_ratio_tok_max_abs`, `ratio_tok` (mean of `exp(log_ratio)`). Both per-replica summaries (`log_ratio_per_replica`, `ratio_per_replica`, `log_ratio_max_per_replica`) and aggregates flow into the row + debug.out.
- **`debug.out` per-step plaintext dump** matching GDPO format: prompts, per-replica scalar table (k, p, len, reward, advantage, log_ratio, ratio, |lr|max), composite-reward components per replica, decoded sequences, raw token-id sequences with `_` for any leftover MASK. No per-corruption mask block (GRPO is single-step, all-MASK).
- **`debug_log` flag** on `GRPOConfig` (default `True`) and `--no_debug_log` CLI flag.

**Plotting** ([plotting.py](../../src/biom3/rl/plotting.py)) extended:
- 6-panel layout grows to **7 panels for GRPO** when log-ratio fields are present (back-compat with old logs that lack them).
- New panel 7: `log_ratio_tok` (mean) and `log_ratio_tok_max_abs` (`|max|`) over steps with a zero baseline.
- GDPO 8-panel layout unchanged.

### 3. Atomic per-step `train_log.json` flush

Previously both trainers wrote `train_log.json` only at the end of the run. Walltime overruns or crashes left an empty output dir. Now:

- New `write_train_log_atomic(log_path, log_rows)` in [plotting.py](../../src/biom3/rl/plotting.py) — writes to `train_log.json.tmp` then `os.replace` (atomic on POSIX).
- Both trainers flush after the `_meta` row, after each per-step `log_rows.append(row)`, and as a safety net after `final.pt` is written. Per-step flush wrapped in `try/except` and logged as a warning if it fails (non-fatal — training continues).
- File size at peak ~50 KB; per-step write is tens of microseconds — invisible against the multi-minute step time.
- Live `tail`-able log: `watch -n 30 'jq ".[-1]" outputs/grpo/<run_id>/train_log.json'` shows the most recent step's metrics during training.

### 4. New configs and PBS jobs

| File | Purpose |
|---|---|
| [configs/grpo/production_grpo.json](../../configs/grpo/production_grpo.json) | Production GRPO config. K=8, steps=100, save_steps=5, β/ε from `_base_grpo.json`. |
| [jobs/aurora/grpo/job_grpo_production_v01.pbs](../../jobs/aurora/grpo/job_grpo_production_v01.pbs) | Production GRPO PBS, single-tile, walltime 8h, queue `gpu_hack_large`. Sized at K=8, steps=50 to fit walltime (no pre_unmask in GRPO yet ⇒ full L=1024 rollout per step ⇒ ~7 min/step). |
| [jobs/aurora/gdpo/job_gdpo_production_v02_multixpu.pbs](../../jobs/aurora/gdpo/job_gdpo_production_v02_multixpu.pbs) | Multi-tile GDPO production, G=24, `--rollout_devices auto`, `unset ZE_AFFINITY_MASK`. |
| [scripts/run_gdpo_smoke_multixpu.sh](../../scripts/run_gdpo_smoke_multixpu.sh) | 5-step interactive smoke for the rollout pool. |

## Open follow-ups

Carrying forward from the 2026-04-29 session note plus new ones surfaced today:

### Still open from yesterday

- **`mu_inner` PPO loop** — required to make the PPO-clip non-degenerate (today `log_ratio_seq ≡ 0` for GDPO; same applies to GRPO once we add a snapshot). ~10 LOC.
- **`kl_estimator="sdmc"` validation** — supported but not run.
- **Multi-prompt training** — currently overfitting one SH3-Sho1 prompt.
- **GRPO bug fix** — done today. Strike from the open list.

### New today

- **MPI-backed rollout pool (Phase B.5)**. The threaded `RolloutPool` is the simplest design but caps at ~3-5× speedup at 12 tiles due to Python's GIL holding during kernel-launch dispatch. If the v02 multi-tile production run shows we're hitting that ceiling and the marginal cost is too high, fall back to the standard `mpiexec` pattern with one process per tile (rank 0 trains, rank 1..11 are pure rollout workers using `dist.broadcast` for state-dict sync and `dist.gather` for ids return). The XCCL hang documented in `2026-04-27_xccl_hang_recap.md` is a known hazard for this stack.
- **Diversity metric (Phase A)**. Deferred behind multi-tile per the plan (Phase B → A → C). The mode-collapse failure mode in v01 (pairwise identity 65.6% → 91.9%) makes diversity a high-priority diagnostic; ideally surface it in the v02 multi-tile run.
- **Diversity reward (Phase C)**. Same.
- **GRPO `pre_unmask` support**. GRPO's `diffusion_rollout` doesn't honor the pre_unmask flag (it always rolls out the full L=1024). With L=1024 each step's rollout is ~7 min — production runs need 14h for 100 steps. Adding pre_unmask to GRPO is a small refactor (use `_build_initial_mask_state` in `diffusion_rollout`, mirror the GDPO pre_unmask resolution in `grpo_train`). Today's production_grpo PBS works around this by truncating to steps=50 + 8h walltime.
- **Comparison-aware plotting**. With both GRPO and GDPO now writing the same schema (per-replica rewards, components, log-ratio diagnostics) we could land a side-by-side plot helper for A/B comparison runs. Not urgent.
- **GDPO log_ratio_seq ≡ 0 expectations documented**. Confirmed structural at μ=1; documented in the trainer comments. When mu_inner > 1 lands, this changes; the per-step log will surface non-degenerate ratios.

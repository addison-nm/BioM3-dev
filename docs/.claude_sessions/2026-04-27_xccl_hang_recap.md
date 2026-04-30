# 2026-04-27 — DDP-on-xccl phantom hang: recap

Aurora Stage 3 finetune was deterministically deadlocking at the train→val
boundary. After 23 rounds of trials we determined this was *not* an xccl bug —
it was a `DistributedSampler(drop_last=False)` + leftover-sample race in our
own data pipeline. xccl was correctly hanging on a genuinely mismatched DDP
gradient `all_reduce`.

Fix shipped to `addison-dev` as commit `52807fe` (after a small `7123b1e`
deletion-only commit and a `4384ffd` snapshot commit on the same branch).

## What shipped

| Change | File | Why |
|---|---|---|
| `_make_distributed_sampler(drop_last=True)` helper, used by all 4 train/val DataLoaders | `src/biom3/Stage3/PL_wrapper.py` | The actual fix — drops the leftover samples at shard level so every rank sees the same sample count |
| `'use_distributed_sampler': False` in `trainer_params` | `src/biom3/Stage3/run_PL_training.py` | Prevents Lightning from auto-wrapping our DataLoader with its own `DistributedSampler(drop_last=False)`, which would override our explicit sampler |
| `os.path.isdir(ckpt)` gating around `convert_zero_checkpoint_to_fp32_state_dict` | `src/biom3/Stage3/run_PL_training.py` (`save_model`) | Single-file `.ckpt` paths (DDP / single-device DeepSpeed) are copied; only DeepSpeed sharded dirs go through the ZeRO→fp32 converter. Fixes a separate, pre-existing bug surfaced by the same investigation |
| `limit_val_batches` plumbed unconditionally | `src/biom3/Stage3/run_PL_training.py` | Was ignored in `primary_only` training_strategy; real bug found in round 5 |
| Post-mortem | `docs/bug_reports/2026-04-27_ddp_xccl_phantom_hang/postmortem.md` | Full analytical write-up of the bug, the failed hypotheses, and the lessons |
| Session log | `docs/.claude_sessions/2026-04-27_ddp_xccl_phantom_hang.md` | Chronological round-by-round play-by-play |
| Snapshot tag | `debug/2026-04-27-xccl-hang-investigation` (both BioM3 and Lightning fork) | Pins the full debug-time state for future profiling; commit `4384ffd` includes the BATCH-print instrumentation, Fix-A `device_sync()`, full bug-doc dir with `pip_freeze.txt`, `.patch` files, etc. |
| Removed | `docs/.claude_prompts/2026-04-26_alcf-ticket-draft.md` (commit `7123b1e`) | The draft ticket would have been wrong — the bug wasn't xccl. Historical version preserved in the snapshot commit. |

## Verification

| Round | Strategy | Trials | Result |
|---|---|---|---|
| 21 | DDP + `DistributedSampler(drop_last=True)` + val on, no cap | 3 | 3/3 EXIT_0 |
| 22 | DDP + sampler fix + `DeviceStatsMonitor` + val on | 3 | 3/3 EXIT_0 (training fully clean; underlying `EXIT_1` was the orthogonal `save_model` bug, now also fixed) |
| 23 | DeepSpeed + sampler fix + val on | 3 | 3/3 EXIT_0 (everything clean — DeepSpeed's sharded ckpt format makes the converter call work) |
| (test suite) | Stage 3 entrypoint tests, multiple device parametrizations | 49 collected | All passing after the `os.path.isdir` save_model gating fix |

The fix is strategy-independent (works under both DDP and DeepSpeed), but the
project's production default and test suite both expect DeepSpeed historically,
so the merged code stays on `'strategy': deepspeed_strategy`.

## Quick technical summary

**Bug:**
- Dataset has 17,165 samples → after `train_test_split(test_size=0.2)` →
  ~13,732 train samples → after `min_seq_length` filtering → effective ~13,633.
- With `world_size=12` and `batch_size=16`, the canonical "fits cleanly"
  count is `12 × 71 × 16 = 13,632`. The actual filtered count is
  ~13,632 + (1–11 leftover samples).
- `DistributedSampler(drop_last=False)` (PyTorch's default) handles the
  leftover by *padding* with duplicate samples, scattering 1–4 extra
  samples across random ranks. Those ranks run a 72nd batch (often
  `bs=1`) that the others don't.
- The 72nd batch's DDP gradient `all_reduce` has no peer collective from
  the other ranks → mismatched-collective deadlock. xccl correctly hangs.

**Fix:** explicit `DistributedSampler(drop_last=True)` truncates the dataset
to `world_size * floor(N / world_size)` samples *before* sharding. Every
rank gets exactly the same count. No leftover, no extra batch, no
mismatched collective.

## Why it took 23 rounds

The deadlock site is in xccl C++ code. py-spy stacks pointed there. Most CCL
env-var profiles, Lightning patches, gloo/MPI tests, and ALCF-doc-recommended
fixes were tried before we noticed (in round 6 trial 1) that the *single
successful trial* had every rank running exactly 71 batches, while the failing
trials had 1–4 ranks running 72. Once we compared per-rank batch counts, the
mismatch was unambiguous and the fix was straightforward.

## Open follow-ups

1. **Audit Stage 1 (PenCL) and Stage 2 (Facilitator) DataLoaders** for the
   same `DistributedSampler(drop_last=False)` issue. They likely have it but
   haven't bitten yet because their dataset shard sizes happen to align.
   The same `_make_distributed_sampler` helper from Stage 3 can be reused.
2. **Stale `_jobscript.sh`** in repo root — predates this session, untracked.
   Delete if you want a clean tree.
3. **`debug/rounds/round_*` artifacts** are still on disk (~GBs). Can be
   deleted at your convenience; nothing references them anymore. Round-level
   index is preserved in the bug-doc dir's `rounds_index.txt`.
4. **Reverse `--no-ff` merge note:** the snapshot commit `4384ffd` carries
   instrumentation (BATCH prints, Fix A `device_sync()`) that's *not* needed
   in production. It exists only as the historical debug-state snapshot and
   was reverted in `52807fe`. The two commits compose to "production fix";
   the snapshot is preserved purely via the `debug/2026-04-27-xccl-hang-investigation`
   tag for future profiling. Worth knowing if anyone walks `git log --first-parent`.
5. **Cosmetic glitch:** commits `7123b1e` and `52807fe` share the same title
   (the first add command short-circuited on the staged-deletion file, so
   the deletion landed alone before the actual fix). Not breaking; mention
   if it confuses anyone reading `git log`.

## Memory artifacts

Saved to claude memory for future sessions:
- Aurora `environment.sh` must be sourced for tests/training (TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD)
- DistributedSampler drop_last footgun and the `_make_distributed_sampler` pattern
- Stage 3 test suite expects DeepSpeed strategy
- Branch push policy (scratch branches only; user merges)

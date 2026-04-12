# Session: ModelCheckpoint reduce_boolean_decision bug on Aurora

**Date:** 2026-04-12
**Branch:** addison-dev

## Problem

Multi-node ProteoScribe finetuning on Aurora (8 nodes × 12 XPU tiles = 96
ranks) produced continuously improving `val_loss` (0.179 → 0.102 over 100
epochs), but `ModelCheckpoint` only saved checkpoints at epochs 0 and 1.
Every subsequent epoch logged `'val_loss' was not in top 2`.  The saved
"best" model was from epoch 1 — effectively discarding 98 epochs of
improvement.

## Investigation

### Phase 1 — Reproduce and narrow scope

Confirmed the bug from `runlog.txt` of the original 8-node run.  The first
two saves correspond exactly to `save_top_k=2`: Lightning's
`check_monitor_top_k` short-circuits through `less_than_k_models=True` for
the first K saves, then switches to a consensus path
(`reduce_boolean_decision`) from epoch K onward.

### Phase 2 — Rule out sync_dist

Added `all_ranks_val_loss` mode to `MetricsHistoryCallback` to dump
per-rank `val_loss` at every epoch end.  Ran a 5-epoch 2-node diagnostic
on Aurora.  Result: **all 24 ranks had identical `val_loss` values at every
epoch** (zero spread).  `sync_dist=True` is working correctly via CCL.

### Phase 3 — Instrument ModelCheckpoint

Created `DiagnosticModelCheckpoint` subclass that logs `current`,
`kth_best`, the local `torch.lt` result, and the
`reduce_boolean_decision` outcome.  Results:

```
epoch=2: current=0.166156  kth_best=0.178507  torch.lt=True  reduce_boolean_decision=False
epoch=3: current=0.159461  kth_best=0.178507  torch.lt=True  reduce_boolean_decision=False
epoch=4: current=0.155291  kth_best=0.178507  torch.lt=True  reduce_boolean_decision=False
```

Every rank locally agrees the metric improved (`torch.lt=True`), but the
all-reduce consensus returns False.

### Phase 4 — Root cause in custom Lightning fork

`reduce_boolean_decision` calls `self.reduce(tensor(1), reduce_op=SUM)`.
In the custom Lightning fork (`../lightning/`), `DDPStrategy.reduce()` had
been patched on 2026-03-18 to work around `ReduceOp.AVG` not being
supported on XPU/CCL.  The patch **unconditionally** did:

```python
reduced = _sync_ddp_if_available(tensor, group, reduce_op="sum")
reduced = reduced / world_size   # ← always divides, even for SUM requests
```

This turned every reduce into a mean.  For `reduce_boolean_decision`:
`24 / 24 = 1.0`, then `1.0 == 24` → `False`.  Always.

## Fix

### Primary fix (custom Lightning fork)

Only divide by `world_size` when the caller explicitly requests
`"mean"` or `"avg"`.  All other reduce ops pass through unchanged.

**Commit:** `e8b4be3` in `../lightning/` —
`fix: only apply AVG→SUM workaround when caller requests mean reduction`

### Fallback (BioM3-dev)

`SyncSafeModelCheckpoint` in `biom3.Stage3.callbacks` overrides
`check_monitor_top_k` to skip `reduce_boolean_decision` entirely.
Enabled via `--use_sync_safe_checkpoint True`.  Safe because
`sync_dist=True` guarantees identical values across ranks.

**Commit:** `7638038` on addison-dev —
`feat: add SyncSafeModelCheckpoint and per-rank val_loss diagnostics`

### Other changes

- `a60c8f5` — Removed `set -euo pipefail` from all Aurora/Polaris PBS
  scripts (22 files) to prevent premature exits on non-fatal return codes.
- Added `docs/aurora_distributed_training.md` covering tile/device
  topology, oneCCL limitations, and the reduce bug.

## Key takeaways

1. **`reduce_boolean_decision` is fragile on non-NCCL backends.**  It
   relies on integer all-reduce correctness AND on `self.reduce()` not
   altering the semantics of the requested op.  Any middleware that
   normalizes reduce ops (like our AVG workaround) can silently break it.

2. **The first `save_top_k` saves always work** because they short-circuit
   before the consensus check.  This made the bug appear to be about
   "improvement detection" when it was actually about the consensus
   mechanism.

3. **Per-rank metric dumps are valuable diagnostics.**  The
   `all_ranks_val_loss` mode quickly ruled out sync_dist issues and
   focused the investigation on the checkpoint callback itself.

4. **Aurora oneCCL has additional quirks:** `ReduceOp.AVG` unsupported,
   integer-dtype all-reduce potentially unreliable.  Both are documented
   in `docs/aurora_distributed_training.md`.

## Training infrastructure improvements

After the checkpoint bug was resolved, additional improvements were made
to the training pipeline in the same session.

### Per-epoch and checkpoint logging (`dd49bf2`)

The file logger (`run.log`) previously had no per-epoch output — all
epoch metrics only appeared in the tqdm progress bar (stdout), which is
noisy and hard to parse from PBS log files.

- `MetricsHistoryCallback.on_validation_epoch_end` now emits a one-line
  epoch summary: train_loss, val_loss, gap, accuracy metrics, and lr.
- `_CheckpointLogMixin` (shared by `LoggingModelCheckpoint` and
  `SyncSafeModelCheckpoint`) logs each checkpoint save with epoch,
  score, best score, and filename.
- `LoggingModelCheckpoint` is now used by default instead of bare
  `ModelCheckpoint`.

### Disabled per-batch GPU memory logging (`a18db10`)

Commented out the `print_gpu_initialization()` call in
`PL_wrapper.common_step` that ran at `realization_idx==0` for every
stage. This was producing noisy device/vendor/memory lines on every
first validation and training batch.

### PBS job script cleanup (`a60c8f5`)

Removed `set -euo pipefail` from all 22 Aurora and Polaris `.pbs` files.
The strict error handling caused jobs to exit prematurely on non-fatal
return codes from environment setup commands.

### Dataset split persistence and testing (`459fe36`)

- `HDF5DataModule.setup()` now stores `split_info` (list of dicts with
  path, train_indices, val_indices) on the data module instance.
- `train_model()` saves `split_info` to `artifacts/dataset_splits.pt`
  on rank 0 after training completes.
- `log_every_n_steps` is now clamped to the number of training batches
  when it exceeds them (avoids Lightning warning).
- New `tests/stage3_tests/test_data_splitting.py` with 8 tests:
  - Deterministic splits given the same seed
  - Different seeds produce different splits
  - Train/val indices are disjoint and cover all filtered data
  - Identical splits across simulated ranks (same seed → same result)
  - `DistributedSampler` assigns non-overlapping samples per rank
  - All samples covered across ranks
  - Different epochs produce different batch orderings

## Commits

| Hash | Description |
|------|-------------|
| `e8b4be3` | (lightning/) fix: only apply AVG→SUM workaround for mean reduction |
| `7638038` | feat: SyncSafeModelCheckpoint + per-rank val_loss diagnostics |
| `a60c8f5` | chore: remove set -euo pipefail from Aurora/Polaris PBS scripts |
| `dd49bf2` | feat: per-epoch and checkpoint-save logging to run.log |
| `a18db10` | chore: disable per-batch GPU memory logging in common_step |
| `459fe36` | feat: save dataset split indices and add splitting tests |

## Files changed

- `src/biom3/Stage3/callbacks.py` — `MetricsHistoryCallback` all-ranks
  mode + epoch summary logging, `_CheckpointLogMixin`,
  `LoggingModelCheckpoint`, `SyncSafeModelCheckpoint`
- `src/biom3/Stage3/run_PL_training.py` — `--use_sync_safe_checkpoint`,
  `--metrics_history_all_ranks_val_loss` flags, dataset split saving,
  `log_every_n_steps` clamping, `LoggingModelCheckpoint` as default
- `src/biom3/Stage3/PL_wrapper.py` — `split_info` storage, GPU logging
  commented out
- `configs/stage3_training/finetune_v1.json` — cleaned diagnostic flags
- `docs/aurora_distributed_training.md` — new Aurora reference doc
- `jobs/aurora/*.pbs`, `jobs/polaris/*.pbs` — removed `set -euo pipefail`
- `tests/stage3_tests/test_data_splitting.py` — new test file (8 tests)
- `../lightning/src/lightning/pytorch/strategies/ddp.py` — DDP reduce fix

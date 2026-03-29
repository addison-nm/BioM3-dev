# Session: Multinode Logging Fixes and Training Stability

**Date:** 2026-03-29
**Branch:** addison-main
**Status:** Complete, verified on Aurora 2-node/24-GPU run

## Context

Follow-up to the 2026-03-28 rank-aware logging session. The `setup_logger` / `get_rank` utilities were deployed but not working correctly on Aurora.

## Problems Found and Fixed

### 1. Rank gating not working on Aurora (device.py)

`get_rank()` only checked `RANK` and `LOCAL_RANK` env vars (torchrun/deepspeed convention). Aurora's PALS launcher sets `PALS_RANKID` / `PALS_LOCAL_RANKID` instead. All 24 ranks returned 0, producing 24x duplicated output.

**Fix:** Extended `get_rank()` to check: `RANK` → `PALS_RANKID` → `PMI_RANK` → `OMPI_COMM_WORLD_RANK` → `LOCAL_RANK` → `PALS_LOCAL_RANKID` → default 0.

### 2. Duplicate log lines per message (device.py)

Every message printed twice — once from our handler (short format) and once from the root logger (long format set by DeepSpeed/Lightning). Default `propagate=True` caused messages to bubble up.

**Fix:** Added `logger.propagate = False` in `setup_logger()`.

### 3. Distributed deadlock after sanity check (PL_wrapper.py)

The 3/28 session changed `_shared_step` to guard `print_gpu_initialization()` with `self.global_rank == 0`, but `self.log(..., sync_dist=True)` was inside the same guard. `sync_dist=True` triggers `all_reduce` — with only rank 0 entering, the other 23 ranks never participated → deadlock.

**Fix:** Moved `self.log(sync_dist=True)` outside the rank guard. Only `print_gpu_initialization()` is rank-0-gated; non-zero ranks log `0.0` for `gpu_memory_usage`. Also added `None` guard for the return value.

### 4. NaN/Inf in perplexity metrics (eval_metrics.py)

`compute_ppl()` computed `p * log(p)` without handling `p=0`, producing NaN via `0 * -inf`. Also, when `time_split_on_seq` returned empty tensors (zero future/prev positions), `compute_ppl` produced NaN from `torch.tensor([]).mean()`, and `batch_compute_ppl` / `batch_hard_acc` hit division by zero on empty lists.

**Fix:**
- `compute_ppl`: Guard `log(p)` with `torch.where(p > 0, ...)` to handle 0*log(0)
- `compute_ppl`: Return `float('nan')` for zero-length sequences (semantically correct: "no data")
- `batch_compute_ppl` / `batch_hard_acc`: Return `float('nan')` for empty input lists
- `compute_pos_entropy` in `transformer_training_helper.py`: Same `p*log(p)` fix (user applied manually)
- Silenced `tensorboardX.x2num` logger to ERROR level (NaN from empty splits is expected, not actionable; TensorBoard skips NaN points in plots)

### 5. Checkpoint race condition on Lustre (run_PL_training.py)

With `save_last=True`, two concurrent DeepSpeed sharded saves at the same step caused rank 13 (second node) to write its shard to a `-v1` directory due to Lustre metadata propagation delay.

**Fix:**
- `save_last="link"` — creates symlink instead of full second save
- `enable_version_counter=False` — disables `-v1` suffixing so all ranks always use the same path

### 6. Missing `logging` import (run_PL_training.py)

Added `import logging` to support `logging.getLogger("tensorboardX.x2num").setLevel(logging.ERROR)`.

## Files Modified

- `src/biom3/backend/device.py` — `get_rank()` PALS/PMI/OMPI vars, `propagate=False`
- `src/biom3/Stage3/PL_wrapper.py` — `sync_dist` deadlock fix, `gpu_memory_usage` None guard
- `src/biom3/Stage3/eval_metrics.py` — NaN fixes in `compute_ppl`, `batch_compute_ppl`, `batch_hard_acc`
- `src/biom3/Stage3/transformer_training_helper.py` — `compute_pos_entropy` log(0) fix (user applied)
- `src/biom3/Stage3/run_PL_training.py` — `save_last="link"`, `enable_version_counter=False`, tensorboardX silence, `import logging`

## Remaining Warnings (not addressed)

- Lightning `sync_dist=True` recommendation for epoch-level train metrics — suppressing or enabling is a tradeoff (overhead vs accuracy)
- DeepSpeed `initializing deepspeed distributed: GLOBAL_RANK: X` lines from all ranks — internal to DeepSpeed, cannot be silenced via our logger
- `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD` warnings from Lightning checkpoint loading

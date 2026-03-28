# Session: Rank-Aware Logging for Multinode Training

**Date:** 2026-03-28
**Branch:** addison-local
**Status:** Implementation complete, tests pending

## Problem

Log files generated during multinode training on Aurora (2 nodes, 12 GPUs each = 24 ranks) were extremely cluttered:

- Every `print()` statement executed on all 24 ranks, producing 24x duplicated output
- The startup block alone (~30 lines) repeated 24 times = ~730 lines before training began
- `print_gpu_initialization()` in `PL_wrapper.py` fired on every rank at every first batch, producing ~72 lines of "Running on device: Intel..." per step boundary
- Lightning `LeafSpec` deprecation warnings repeated 24x during sanity check
- No rank-gating existed anywhere in the codebase
- No `logging` module usage existed -- everything was raw `print()`

Reference log: `logs/run_logs/pretraining/pretrain_scratch_v1_n2_d12_e5_V20260326_193215.*.o` (3915 lines for ~2 epochs)

## Solution

Replaced all `print()` calls with Python's `logging` module, using a rank-aware `setup_logger()` utility that silences non-rank-0 processes.

### New utility: `biom3/backend/device.py`

Added two functions:

- `get_rank()` — reads `RANK` / `LOCAL_RANK` env vars (set by launcher before Python starts)
- `setup_logger(name)` — returns a `logging.Logger` that only emits on rank 0 (non-zero ranks set to `CRITICAL`)

Usage pattern in each module:
```python
from biom3.backend.device import setup_logger
logger = setup_logger(__name__)
logger.info("Only printed on rank 0")
```

### Key fix: `PL_wrapper.py` global_rank guard

The `print_gpu_initialization()` call in `_shared_step()` was guarded only by `realization_idx == 0`, which is true on every rank. Changed to:
```python
if realization_idx == 0 and self.global_rank == 0:
```

### Warnings filter

Added to `run_PL_training.py:main()`:
```python
warnings.filterwarnings("ignore", message=".*LeafSpec.*is deprecated.*")
warnings.filterwarnings("ignore", message=".*isinstance.*treespec.*")
```

### Log level choices

- `logger.info()` — normal operational messages (model size, training config, results)
- `logger.debug()` — noisy per-item messages (data paths, shape dumps, per-block freeze details)
- `logger.warning()` — actual warnings (missing checkpoints, size mismatches)
- `logger.error()` — fatal errors before exit

## Files Modified

### Backend (3 files)
- `src/biom3/backend/device.py` — added `get_rank()`, `setup_logger()`
- `src/biom3/backend/xpu.py` — `print()` -> `logger.info()`/`logger.warning()`
- `src/biom3/backend/cpu.py` — `print()` -> `logger.info()`
- `src/biom3/backend/cuda.py` — `print()` -> `logger.info()`

### Core (2 files)
- `src/biom3/core/io.py` — `print()` -> `logger.info()`
- `src/biom3/core/helpers.py` — `print()` -> `logger.info()`/`logger.warning()`/`logger.debug()`

### Stage1 (3 files)
- `src/biom3/Stage1/run_PenCL_inference.py` — all prints -> logger (inference entrypoint)
- `src/biom3/Stage1/preprocess.py` — data loading prints -> logger
- `src/biom3/Stage1/helper_funcs.py` — summary prints -> logger

### Stage2 (1 file)
- `src/biom3/Stage2/run_Facilitator_sample.py` — all prints -> logger (inference entrypoint)

### Stage3 (9 files)
- `src/biom3/Stage3/run_PL_training.py` — ~35 prints -> logger, removed `if verbosity:` guards, added warnings filter
- `src/biom3/Stage3/PL_wrapper.py` — prints -> logger, added `self.global_rank == 0` guard
- `src/biom3/Stage3/io.py` — checkpoint loading prints -> logger
- `src/biom3/Stage3/diff_transformer_layer.py` — architecture prints -> logger.debug()
- `src/biom3/Stage3/cond_diff_transformer_layer.py` — architecture prints -> logger.debug()
- `src/biom3/Stage3/preprocess.py` — data prep prints -> logger
- `src/biom3/Stage3/eval_metrics.py` — warning prints -> logger.warning()
- `src/biom3/Stage3/sampling_analysis.py` — debug prints -> logger.debug()
- `src/biom3/Stage3/helper_funcs.py` — summary prints -> logger
- `src/biom3/Stage3/run_ProteoScribe_sample.py` — sampling prints -> logger

## Not Changed

- `__main__.py` files in Stage1/2/3 — `print("Hello world...")` left as-is (module test, not training/inference path)
- `verbosity` parameter still exists in function signatures of `core/helpers.py` and `core/io.py` to avoid breaking the API. Could be cleaned up in a follow-up.

## Expected Impact

On the same 2-node, 24-GPU run:
- Startup block: ~30 lines (down from ~730)
- Per-step GPU info: 3 lines (down from ~72)
- LeafSpec warnings: 0 lines (down from 24)
- Estimated total log reduction: ~90%+

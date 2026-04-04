# Session: Fix XPU TensorBoardLogger Import

**Date:** 2026-04-04
**Branch:** addison-main

## Summary

Fixed a pre-existing bug where Stage 3 training on Aurora (XPU) failed with
`TypeError: TensorBoardLogger.__init__() missing 1 required positional argument: 'root_dir'`.
The XPU code path imported `TensorBoardLogger` from `lightning.fabric.loggers`
(fabric API, expects `root_dir` positional) instead of `lightning.pytorch.loggers`
(PyTorch Lightning API, accepts `save_dir` keyword).

## Change

**File:** `src/biom3/Stage3/run_PL_training.py`, line 39

```diff
-    from lightning.fabric.loggers import TensorBoardLogger
+    from lightning.pytorch.loggers import TensorBoardLogger
```

## Bug report

See `docs/bug_reports/xpu_tensorboard_logger_import.md`.

## Testing

- All 17 previously failing XPU training tests now pass on Aurora.
- `pytest tests/test_imports.py` — 5/5 passed on Spark (CUDA).

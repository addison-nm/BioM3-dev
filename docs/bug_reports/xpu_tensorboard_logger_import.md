# Bug: TensorBoardLogger Import From Wrong lightning Subpackage on XPU

## Summary

Stage 3 training failed on Aurora (XPU) with:

```
TypeError: TensorBoardLogger.__init__() missing 1 required positional argument: 'root_dir'
```

The XPU code path imported `TensorBoardLogger` from the wrong `lightning`
subpackage, causing an API mismatch.

---

## Root Cause

The `lightning` package (used on XPU/Aurora) has two `TensorBoardLogger` classes
with different constructor signatures:

| Import path | First argument | Used by |
|---|---|---|
| `lightning.fabric.loggers.TensorBoardLogger` | `root_dir` (positional, required) | Fabric (low-level API) |
| `lightning.pytorch.loggers.TensorBoardLogger` | `save_dir` (keyword) | PyTorch Lightning Trainer |

The XPU import block in `run_PL_training.py` used the **fabric** version:

```python
# WRONG — fabric API, requires positional root_dir
from lightning.fabric.loggers import TensorBoardLogger
```

The CUDA/CPU path correctly used:

```python
from pytorch_lightning.loggers import TensorBoardLogger
```

The training code instantiates the logger with `save_dir=` as a keyword:

```python
tb_logger = TensorBoardLogger(
    save_dir=logs_dir,
    version="",
)
```

This worked on CUDA/CPU (where `save_dir` is accepted), but failed on XPU
because the fabric version expects `root_dir` as the first positional argument.

## Fix

Changed the XPU import from `lightning.fabric.loggers` to
`lightning.pytorch.loggers`, which provides the same `save_dir`-based API as
`pytorch_lightning.loggers`:

```python
# CORRECT — pytorch API, accepts save_dir keyword
from lightning.pytorch.loggers import TensorBoardLogger
```

## Impact

- All Stage 3 training tests on XPU were failing (`test_train_from_scratch`,
  `test_train_from_pretrained_weights`, `test_resume_training`,
  `test_finetuning`, `test_start_phase2_training`).
- Stage 3 training on Aurora was broken for any run that reached the
  `TensorBoardLogger` initialization in `train_model()`.
- Inference (sampling) was unaffected — `run_ProteoScribe_sample.py` does not
  use `TensorBoardLogger`.

## Timeline

- **Introduced:** When the XPU import block was originally written. The fabric
  import may have worked with an older version of `lightning` that accepted
  `save_dir`, or this code path was never tested on XPU until now.
- **Discovered:** 2026-04-04, running the full test suite on Aurora.
- **Fixed:** commit `14437de`.

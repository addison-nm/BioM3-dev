# Session: Restructure Stage 3 training output directories

**Date:** 2026-04-04
**Branch:** addison-main

## Motivation

The Stage 3 training pipeline dumped all outputs (Lightning checkpoints, derived
weights, manifests, logs) into a single flat directory per run. This made it hard
to distinguish raw checkpoints from derived artifacts and logs. The previous
session fixed a `checkpoints/checkpoints/` nesting bug, renamed
`--tb_logger_folder` to `--checkpoint_folder`, added `backup_if_exists`, and
co-located wandb logs. This session completes the restructuring with a clean
separation of concerns.

## New output structure

```
{output_root}/
├── {checkpoints_folder}/{run_id}/        ← Lightning/DeepSpeed .ckpt dirs
│   ├── last.ckpt/                          + derived weights (state_dict.*.pth,
│   ├── epoch=X-step=Y.ckpt/                 single_model.*.pth, params.csv)
│   └── ...
├── {runs_folder}/{run_id}/
│   ├── logs/
│   │   ├── lightning_logs/               ← TensorBoardLogger
│   │   └── wandb/                        ← WandBLogger
│   └── artifacts/
│       ├── state_dict.best.pth           ← copy of best from checkpoints
│       ├── args.json                     ← full argparse namespace
│       ├── build_manifest.json           ← run metadata + env
│       └── run.log                       ← Python file logging
```

## CLI arg renames

| Old | New | Notes |
|-----|-----|-------|
| `--tb_logger_path` | `--output_root` | Base output directory |
| `--checkpoint_folder` | `--checkpoints_folder` | Plural for consistency |
| `--version_name` | `--run_id` | Unique per-run identifier |
| *(new)* | `--runs_folder` | Default `"runs"` |
| `--wandb_logging_dir` | *(removed)* | Now derived from runs path |
| `--output_hist_folder` | *(removed)* | Dead code |
| `--output_folder` | *(removed)* | Dead code |
| `--save_hist_path` | *(removed)* | Dead code |

## Changes by file category

### Core Python
- **`src/biom3/Stage3/run_PL_training.py`**
  - Added `_LOGS_SUBDIR`, `_ARTIFACTS_SUBDIR` module constants
  - Renamed args in `get_args()`, `get_path_args()`, `get_wrapper_args()`
  - Added `--runs_folder` arg
  - Removed dead args and dead `save_history_log()` function
  - `train_model()`: new path construction; TensorBoardLogger → `logs_dir`;
    WandBLogger → `logs_dir`; ModelCheckpoint → `checkpoint_dir`
  - `save_model()`: added `artifacts_path` param; copies `state_dict.best.pth`
    to artifacts after writing derived files to checkpoints (using `shutil.copy2`)
  - `main()`: creates output dirs (rank 0); sets up `setup_file_logging` to
    `artifacts_dir`; writes `args.json`; writes manifest to `artifacts_dir`;
    calls `teardown_file_logging` on exit
  - Added imports: `json`, `shutil`, `setup_file_logging`, `teardown_file_logging`

### Shell scripts (6 files)
- `scripts/stage3_{pretraining,finetuning}.sh`: renamed variables and CLI args;
  removed dead arg passthrough (`output_hist_folder`, `output_folder`, `save_hist_path`)
- `scripts/{pretraining,finetuning}/{pretrain,finetune}_{multinode,singlenode}.sh`:
  renamed `version_name` → `run_id` in positional params

### Arglists (7 files)
- All `arglists/config_*.sh`: `version_name` → `run_id`, `tb_logger_path` →
  `output_root`, `checkpoint_folder` → `checkpoints_folder`; removed dead exports

### HPC job templates (15 files)
- All `jobs/{polaris,aurora,spark}/_template_stage3_*`: `version_name` → `run_id`

### Tests
- 8 test arg `.txt` files: renamed args, removed dead args
- `tests/stage3_tests/test_stage3_run_PL_training.py`: updated `prefix_paths()`
  to use `args.output_root`

### Documentation
- `README.md`: "version name" → "run ID" in Stage 3 pretraining docs
- `CLAUDE.md`: added "Training output structure" section

## Dead code removed
- `save_history_log()` function (defined but never called)
- `--output_hist_folder`, `--output_folder`, `--save_hist_path` args
- `--wandb_logging_dir` arg (wandb dir now derived from run path)
- `export output_hist_folder`, `export save_hist_path`, `export output_folder`
  from arglists

## Resume compatibility
No breakage. `--resume_from_checkpoint` takes an explicit file path passed to
`trainer.fit(ckpt_path=...)`. Checkpoint `.ckpt` dirs still live in
`{checkpoints_folder}/{run_id}/`.

## Testing
- `pytest tests/test_imports.py` — 5/5 passed
- `pytest tests/core_tests/test_run_utils.py` — 14/14 passed
- GPU-dependent tests (`-k cuda`) not run (require GPU device)
- `grep` sweep: zero remaining references to old arg names in `src/`, `scripts/`,
  `arglists/`, `jobs/`, `tests/_data/`
- JSON configs for inference/sampling still have vestigial `tb_logger_path` key;
  these are loaded by non-training entrypoints and are harmless

## Total files modified
~47 files across Python, shell scripts, arglists, HPC templates, test data, and docs.

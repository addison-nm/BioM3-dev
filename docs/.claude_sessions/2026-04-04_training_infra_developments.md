# Session: Training Infrastructure Developments

**Date:** 2026-04-04
**Branch:** addison-main

## Summary

Major development session implementing planned improvements across Stage 3 training, sampling, config system, and documentation. All changes were planned upfront and executed in phases.

## Changes

### 1. Metrics and Training

- **1.1 — XPU cache bug fix** (`PL_wrapper.py`): `torch.xpu.memory.empty_cache()` was hardcoded at line 307 and would fail on CUDA. Wrapped in a `BACKEND_NAME` guard to call the correct backend's cache clear.

- **1.2 — Gradient norm metric** (`PL_wrapper.py`): Added `on_before_optimizer_step()` hook to `PL_ProtARDM` that computes and logs the global L2 gradient norm as `train_grad_norm`.

- **1.3 — Multi-metric checkpoint saving** (`run_PL_training.py`): New `checkpoint_monitors` config key (list of `{metric, mode}` dicts). The first entry is the primary monitor (top-2, with `last.ckpt` symlink). Additional entries each save their single best checkpoint and produce `state_dict.best_{metric}.pth` in both checkpoints and artifacts directories. A `checkpoint_summary.json` is written to artifacts.

- **1.4 — Periodic checkpoint saving** (`run_PL_training.py`): New `checkpoint_every_n_steps` and `checkpoint_every_n_epochs` config keys. For step-based training (`combine` strategy), periodic saving defaults to `log_every_n_steps` if not explicitly set.

- **1.5 — Early stopping** (`run_PL_training.py`): Added `EarlyStopping` callback controlled by `early_stopping_metric` (null = disabled), `early_stopping_patience`, `early_stopping_min_delta`, and `early_stopping_mode`. Patience counts validation checks, not raw steps or epochs.

- **1.6 — Metrics history callback** (new `callbacks.py`): `MetricsHistoryCallback` records per-step training metrics and per-epoch validation metrics (including `loss_gap` = val_loss - train_loss). Saves to `artifacts/metrics_history.pt` as a dict of numpy arrays. Configurable via `save_metrics_history`, `metrics_history_every_n_steps`, `metrics_history_ranks`.

### 2. Documentation

- **2.1 — `docs/stage3_training.md`**: Comprehensive training documentation covering pretraining workflows, finetuning, config composition, per-machine instructions (Polaris/Aurora/Spark), output directory structure, checkpoint formats, metrics reference, early stopping, and troubleshooting.

### 3. Nested Training Configs

- **3.1 — `_base_configs` and `_overwrite_configs`** (`core/helpers.py`): Enhanced `load_json_config()` to support two composition keys:
  - `_base_configs`: loaded before the current file (current file overrides)
  - `_overwrite_configs`: loaded after the current file (they override it)
  - Priority: `_base_configs` < current file < `_overwrite_configs` < CLI
  - Paths resolve relative to the JSON file's directory. Circular references detected.

- **Base config directory structure:**
  - `configs/training/models/_base_ProteoScribe_1block.json` — 1-block transformer architecture
  - `configs/training/models/_base_ProteoScribe_16blocks.json` — 16-block transformer architecture
  - `configs/training/machines/_aurora.json` — xpu, 12 devices
  - `configs/training/machines/_polaris.json` — cuda, 4 devices
  - `configs/training/machines/_spark.json` — cuda, 1 device

- All 7 training configs refactored to use `_base_configs` for model architecture and include empty `_overwrite_configs` placeholder.

### 4. Stage 3 Sequence Generation

- **4.1 — FASTA output** (`run_ProteoScribe_sample.py`): New `--fasta` flag writes one FASTA per prompt to `<outdir>/fasta/`. New `--fasta_merge` flag also writes a single `all_sequences.fasta` with all prompt/replica entries. Plain string I/O, no BioPython dependency.

- **4.2 — Output .pt restructure** (`run_ProteoScribe_sample.py`): Changed output dictionary from `{replica_N: [prompt_seqs]}` to `{prompt_N: [replica_seqs], _metadata: {format_version: 2, ...}}`. Tests updated accordingly.

### 5. Training Dataset Generalization

- **5.2 — Renamed dataset args**: `swissprot_data_root` → `primary_data_path`, `pfam_data_root` → `secondary_data_paths` (nargs='+'), `start_pfam_trainer` → `start_secondary`, `data_strategy` → `training_strategy`. Old names kept as deprecated aliases with warnings.

- **`HDF5_PFamDataModule` → `HDF5DataModule`** (`PL_wrapper.py`): Generalized data module accepts `primary_path` and `secondary_paths` (list). Iterates all paths uniformly instead of separate swissprot/pfam branches. Old class name kept as alias.

- All 7 training configs updated with new key names. `training_strategy: "auto"` added (resolves to `primary_only` or `combine` based on secondary data presence).

### Config completeness

All new CLI args from this session are now represented in the training JSON configs with their default values: `runs_folder`, `checkpoint_monitors`, `checkpoint_every_n_steps`, `checkpoint_every_n_epochs`, `early_stopping_*`, `save_metrics_history`, `metrics_history_*`, `training_strategy`, `finetune_output_layers`.

### Finetuning config corrections

- `finetune_v1.json`: `finetune_last_n_layers` changed from `1` to `-1` (all layers of last block)
- `finetune_v2.json`: `finetune_last_n_layers` explicitly set to `1` (last layer only)
- Descriptions updated to reflect the difference

## Files Modified

| File | Changes |
|------|---------|
| `src/biom3/core/helpers.py` | `load_json_config()` with `_base_configs` + `_overwrite_configs` |
| `src/biom3/Stage3/PL_wrapper.py` | XPU cache fix, grad norm hook, `HDF5DataModule` rename |
| `src/biom3/Stage3/run_PL_training.py` | Multi-metric checkpoints, early stopping, metrics history, periodic saving, dataset generalization |
| `src/biom3/Stage3/run_ProteoScribe_sample.py` | FASTA output, .pt output restructure |
| `src/biom3/Stage3/callbacks.py` | New file: `MetricsHistoryCallback` |
| `tests/stage3_tests/test_stage3_run_ProteoScribe_sample.py` | Updated for prompt-indexed dict |
| `configs/training/*.json` (7 files) | New keys, `_base_configs`, dataset renames |
| `configs/training/models/*.json` (2 files) | New base model architecture configs |
| `configs/training/machines/*.json` (3 files) | New per-machine device configs |
| `docs/stage3_training.md` | New comprehensive training documentation |
| `README.md` | Config composition note, FASTA args, finetune example fix |
| `CLAUDE.md` | Config composition docs, training/ in repo layout |

## Verification

- `pytest tests/test_imports.py` — 5/5 passed
- Config composition verified: `_base_configs` merge order, circular reference detection, `_overwrite_configs` priority
- Training arg parsing verified end-to-end for all 7 configs

# 2026-04-05 — Job Scripts & Template Hardening

**Pre-session state:** `git checkout 37abe75`

## Summary

Compared the new aurora-specific job scripts against the templates, identified divergences, and backported improvements to all templates across all three machines (Aurora, Polaris, Spark). Created equivalent job scripts for Polaris and Spark. Hardened all shell scripts against a silent argument-shifting bug when `WANDB_API_KEY` is unset. Renamed base model configs for clarity and removed deprecated flat inference configs.

## Detailed Changes

### 1. Template improvements (all 15 templates: 5 per machine)

Five changes backported from the working aurora job scripts:

- **Finetune `run_id` now includes block/layer counts:** `_ft${finetune_last_n_blocks}-${finetune_last_n_layers}_` added to distinguish finetune runs
- **`primary_data_path` placeholder** added to finetune templates with `--primary_data_path` CLI passthrough
- **`--wandb True` passed as CLI arg:** removed the dead `wandb=True` shell variable that was never passed to the training script
- **Pretrain-from-scratch walltime bumped** from `00:30:00` to `01:00:00` (aurora + polaris)
- **Flat log paths:** `logs/run_logs/{pretraining,finetuning}/` simplified to `logs/`

### 2. Wandb API key safety (all 32 job/template files + 2 training wrappers)

Discovered that an empty `$WANDB_API_KEY` (unquoted) would be silently dropped by the shell, shifting all subsequent positional arguments and causing the training script to misinterpret run_id, device, etc.

Fixes:
- **Quoted `"${wandb_api_key}"`** in all script invocations so an empty value passes as an empty string rather than being dropped
- **`wandb_api_key=${WANDB_API_KEY:-}`** for `set -u` compatibility
- **Auto-disable wandb in training wrappers** (`stage3_train_multinode.sh`, `stage3_train_singlenode.sh`): if the key is empty, inject `--wandb False` and print a warning instead of crashing

### 3. Shell script consistency (all 32 job/template files)

Three conventions unified across all machines:
- **`set -euo pipefail`** added to all PBS and shell scripts
- **`mkdir -p logs`** added before `log_fpath` in all scripts
- **`source environment.sh`** standardized to appear right before the training script call

### 4. Job scripts for Polaris and Spark

Created machine-adapted copies of the aurora job scripts:

**Polaris** (6 files): direct PBS equivalents with polaris conventions (`home:grand`, conda module loading, `num_devices=4`, `device=cuda`)
- `job_pretrain_from_scratch_v1_{n2,n8,n512}.pbs`
- `job_stage3_finetune_v1_{ft1,ft8,ft16}_n8.pbs`

**Spark** (5 files): single-node `.sh` scripts (`singlenode.sh` wrapper, `num_devices=1`). Aurora's n2/n8 debug jobs collapsed into one since Spark is always single-node.
- `job_pretrain_from_scratch_v1_{debug,prod}.sh`
- `job_stage3_finetune_v1_{ft1,ft8,ft16}.sh`

### 5. Base model config rename

- `configs/training/models/_base_model_1block.json` -> `_base_ProteoScribe_1block.json`
- `configs/training/models/_base_model_16blocks.json` -> `_base_ProteoScribe_16blocks.json`
- Updated all `_base_configs` references in 8 JSON configs, CLAUDE.md, and docs

### 6. Removed deprecated flat inference configs

Deleted the old top-level configs that were superseded by `configs/inference/`:
- `configs/stage1_config_PenCL_inference.json`
- `configs/stage2_config_Facilitator_sample.json`
- `configs/stage3_config_ProteoScribe_sample.json`

Updated all references in demos, scripts, docs, and README.

### 7. App settings

Added `data/` directory to `configs/app_settings.json` data_dirs.

## Config path resolution — decision documented

Evaluated switching `_base_configs` path resolution from relative-to-file to relative-to-project-root. Decided to keep relative-to-file (the current approach) because it's the standard pattern (CSS, TypeScript, etc.) and is more robust — configs remain self-contained regardless of CWD. Already documented in `helpers.py` docstring, CLAUDE.md, and `docs/stage3_training.md`.

## Files changed

- `scripts/stage3_train_multinode.sh`, `scripts/stage3_train_singlenode.sh` — wandb auto-disable
- `scripts/embedding_pipeline.sh`, `scripts/seqgen_wrapper.sh` — config path updates
- 15 templates across `jobs/{aurora,polaris,spark}/` — all 5 improvements
- 17 new job scripts across `jobs/{aurora,polaris,spark}/`
- 8 JSON configs — base model config rename
- `CLAUDE.md`, `README.md`, 6 docs — reference updates
- `configs/app_settings.json` — added `data/` dir
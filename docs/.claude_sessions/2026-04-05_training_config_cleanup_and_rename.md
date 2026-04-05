# 2026-04-05 — Training config cleanup and rename

## Summary

Removed `scripts/seqgen_wrapper.sh`, updated training config defaults, and
renamed `configs/training/` to `configs/stage3_training/` with all references
updated across the codebase.

**Pre-session state:** `git checkout 0d198a8`

## Changes

### seqgen_wrapper.sh removal (commit 07385c4)

- Deleted `scripts/seqgen_wrapper.sh` — thin wrapper around `biom3_ProteoScribe_sample`
- Updated `demos/SH3/run_gen_seqs_SH3.sh` and `run_gen_seqs_SH3_prompts.sh` to
  call the entry point directly with long-form flags
- Updated `docs/sequence_generation_strategies.md` CLI examples to use long-form flags

### Training config field updates

Applied across all 7 training configs (`configs/stage3_training/*.json`):

- Added `"model_name": "BioM3-v0"` between `description` and `tags` — for future
  model architecture versioning
- Set all `seed` values to `0` (was 123 in finetune configs, 162 in pfam_v2)
- Set all `wandb` to `false` (was `true`)
- Set all `wandb_project` to `"BioM3-dev"` (was `"BioM3"`)

Pretrain-specific:
- Removed `finetune_output_layers` from all 5 pretrain configs (not relevant to pretraining)

Finetune-specific:
- Set `pretrained_weights` to `"weights/ProteoScribe/ProteoScribe_epoch200.pth"`
- Set `primary_data_path` to `null`

### Directory rename: configs/training → configs/stage3_training

Renamed via `git mv` and updated all references:
- `configs/inference/stage3_ProteoScribe_sample.json` — `_base_configs` path
- `tests/stage3_tests/test_stage3_run_PL_training.py` — `CONFIGS_DIR`
- `CLAUDE.md` — configuration docs
- `README.md` — examples and docs
- `docs/stage3_training.md` — all path references
- 32 job scripts in `jobs/{aurora,polaris,spark}/`

Session notes (historical) were intentionally left unchanged.

### Ecosystem settings

Added `Glob` and `Grep` permissions for the full BioM3-ecosystem path to
`.claude/settings.local.json` in all 4 repos (BioM3-dev, BioM3-data-share,
BioM3-workflow-demo, BioM3-workspace-template).

Created top-level `BioM3-ecosystem/CLAUDE.md` covering the multi-repo ecosystem.

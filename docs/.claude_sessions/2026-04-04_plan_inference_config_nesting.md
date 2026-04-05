# Plan: Config Composition for Stage 1/2/3 Inference Configs

**Date:** 2026-04-04
**Status:** Ready to implement

## Problem

Stage 1, 2, and 3 inference scripts each define their own local `load_json_config()` (simple `json.load`) and `convert_to_namespace()`. These don't support `_base_configs` / `_overwrite_configs` composition — that only works in the training pipeline, which uses `core.helpers.load_json_config()`.

The inference configs in `configs/` are flat JSON files with no composition. The Stage 3 sampling config (`stage3_config_ProteoScribe_sample.json`) duplicates all model architecture fields that already exist in the training base configs.

## Goal

1. Unify all config loading to use `core.helpers.load_json_config()` (with composition support)
2. Factor inference configs into base + leaf structure similar to training
3. Eliminate duplication of model architecture params between training and inference configs

## Current State

### Config loading per stage

| Stage | Script | Config loader | Uses argparse? |
|-------|--------|--------------|----------------|
| Stage 1 inference | `Stage1/run_PenCL_inference.py:106` | Local `load_json_config()` + `convert_to_namespace()` | CLI args parsed separately, then JSON loaded into namespace |
| Stage 2 inference | `Stage2/run_Facilitator_sample.py:80` | Local `load_json_config()` + `convert_to_namespace()` | Same pattern |
| Stage 3 sampling | `Stage3/run_ProteoScribe_sample.py:125` | Local `load_json_config()` + `convert_to_namespace()` | Same pattern |
| Stage 3 training | `Stage3/run_PL_training.py:763` | `core.helpers.load_json_config()` + `parser.set_defaults()` | Full argparse integration |

Key difference: inference scripts load JSON → namespace directly (no argparse merge). Training uses JSON → `set_defaults()` → argparse.

### Config field analysis

**Stage 1** (`stage1_config_PenCL_inference.json`) — 47 keys:
- **Model architecture** (could be shared): `seq_model_path`, `pretrained_seq`, `trainable_seq`, `rep_layer`, `protein_encoder_embedding`, `text_model_path`, `pretrained_text`, `trainable_text`, `text_encoder_embedding`, `text_max_length`, `proj_embedding_dim`, `dropout`
- **Training params** (unused in inference but present): `epochs`, `lr`, `base_lr`, `weight_decay`, `patience`, `factor`, `acc_grad_batches`, `valid_size`
- **Runtime/paths** (per-job): `data_path`, `model_checkpoint_path`, `output_dict_path`, `num_gpus`, `precision`, `seed`, `batch_size`, `num_workers`, `device` (passed via CLI)
- **Data config**: `sequence_keyword`, `id_keyword`, `dataset_source`, `dataset_type`, `model_type`, `pfam_data_split_label`

**Stage 2** (`stage2_config_Facilitator_sample.json`) — 15 keys:
- **Model architecture**: `emb_dim`, `hid_dim`, `loss_type`, `dropout`
- **Runtime/paths**: `model_checkpoint_path`, `stage1_dataset_path`, `stage2_output_path`, `seed`, `batch_size`, `num_workers`, `precision`
- **Data config**: `model_type`, `dataset_type`, `fast_dev_run`

**Stage 3 sampling** (`stage3_config_ProteoScribe_sample.json`) — 62 keys:
- **Model architecture** (overlaps with training base configs): `transformer_dim`, `transformer_heads`, `transformer_depth`, `transformer_blocks`, `transformer_dropout`, `transformer_reversible`, `transformer_local_heads`, `transformer_local_size`, `num_classes`, `num_y_class_labels`, `text_emb_dim`, `facilitator`, `diffusion_steps`, `image_size`, `model_option`, `num_steps`, `actnorm`, `perm_channel`, `perm_length`, `input_dp_rate`
- **Sampling-specific**: `num_replicas`, `unmasking_order`, `token_strategy`, `batch_size_sample`
- **Training params** (unused in sampling): `epochs`, `learning_rate`, `weight_decay`, `warmup_steps`, `choose_optim`, `scheduler_gamma`, `acc_grad_batches`, `enter_eval`, `valid_size`
- **Runtime/paths**: `device`, `seed`, many path fields set to `"None"`

## Implementation Plan

### Step 1: Replace local loaders with `core.helpers` import

In each of these three files, replace the local `load_json_config` and `convert_to_namespace` with imports from `core.helpers`:

- `src/biom3/Stage1/run_PenCL_inference.py` — delete lines 106-119, import from `biom3.core.helpers`
- `src/biom3/Stage2/run_Facilitator_sample.py` — delete lines 80-93, import from `biom3.core.helpers`
- `src/biom3/Stage3/run_ProteoScribe_sample.py` — delete lines 125-138, import from `biom3.core.helpers`

This is a mechanical change. The `core.helpers` versions are functionally identical for flat JSON (no `_base_configs` = same behavior as before), so it's backward compatible. The only difference is that composition is now available.

**Test:** existing tests should pass unchanged.

### Step 2: Factor Stage 3 sampling config

The Stage 3 sampling config has the most duplication — its model architecture fields are identical to the training base configs. Factor it:

**Create `configs/inference/stage3_sample.json`:**
```json
{
  "_base_configs": ["../training/models/_base_model_1block.json"],
  "_overwrite_configs": [],

  "num_replicas": 5,
  "unmasking_order": "random",
  "token_strategy": "sample",
  "batch_size_sample": 32,
  "seed": 42,
  ...remaining sampling-specific and runtime keys...
}
```

This eliminates ~20 duplicated model architecture keys. The existing `configs/stage3_config_ProteoScribe_sample.json` can be kept as-is for backward compatibility or replaced with a symlink/redirect.

**Decision needed:** should inference configs move to `configs/inference/` or stay flat in `configs/`? Moving them would be cleaner:
```
configs/
├── inference/
│   ├── stage1_PenCL.json
│   ├── stage2_Facilitator.json
│   └── stage3_ProteoScribe_sample.json
├── training/
│   ├── models/
│   ├── machines/
│   └── *.json
└── dbio_config.json
```

### Step 3: Factor Stage 1 inference config

Stage 1 has model architecture fields (ESM-2 and BioBERT encoder specs) that could become a base config, plus training params that are unused during inference but still present.

**Create `configs/inference/models/_base_PenCL.json`:**
```json
{
  "seq_model_path": "./weights/LLMs/esm2_t33_650M_UR50D.pt",
  "pretrained_seq": true,
  "rep_layer": 33,
  "protein_encoder_embedding": 1280,
  "text_model_path": "./weights/LLMs/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
  "pretrained_text": true,
  "text_encoder_embedding": 768,
  "text_max_length": 512,
  "proj_embedding_dim": 512
}
```

**Decision needed:** should we also clean up the unused training params from the inference config, or leave them for now? They don't cause harm (the inference script ignores them) but they add confusion.

### Step 4: Factor Stage 2 inference config

Stage 2 is small (15 keys). A base model config may be overkill, but for consistency:

**Create `configs/inference/models/_base_Facilitator.json`:**
```json
{
  "emb_dim": 512,
  "hid_dim": 1024,
  "loss_type": "MMD",
  "dropout": 0.0
}
```

### Step 5: Add machine overwrite support to inference configs

Inference configs currently hardcode `device` (or get it from CLI). With `_overwrite_configs`, a user could reference a machine config for device selection:

```json
{
  "_base_configs": ["./models/_base_PenCL.json"],
  "_overwrite_configs": ["../training/machines/_polaris.json"],
  ...
}
```

The machine configs currently only have `device` and `gpu_devices`. Stage 1 uses `num_gpus` instead of `gpu_devices` — this naming inconsistency should be noted but not necessarily resolved in this task (it's a deeper refactor).

### Step 6: Update tests and docs

- Verify existing tests pass with the import change (Step 1)
- Update `docs/stage3_training.md` config composition section to note it applies to all stages
- Update `README.md` if inference config paths change
- Update `CLAUDE.md` configuration section

## Suggested implementation order

1. **Step 1** first — it's mechanical, low-risk, and enables everything else
2. **Step 2** next — biggest win (eliminates ~20 duplicated keys)
3. **Steps 3-4** — smaller wins, can be done together
4. **Step 5** — optional, depends on how much machine config reuse is desired
5. **Step 6** — docs

## Open questions to resolve during implementation

1. Move inference configs to `configs/inference/` or keep them flat in `configs/`?
2. Clean up unused training params from inference configs, or leave them?
3. Should the `updated_stage1_config_PenCL_inference.json` and `updated_stage2_config_Facilitator_sample.json` (currently untracked) be folded into the new structure or deleted?
4. Stage 1 uses `num_gpus` while Stage 3 uses `gpu_devices` — harmonize naming now or defer?

# Session: Replace Shell Arglists with JSON Training Configs

**Date:** 2026-04-04
**Branch:** addison-main

## Motivation

The Stage 3 training configuration passed through 4 layers of shell indirection
before reaching Python: HPC template → multinode/singlenode wrapper →
core training script (sources `.sh` arglist, enumerates 40+ CLI flags) → Python
argparse. Adding or renaming a single parameter required touching the arglist,
the core shell script, and the Python argparse — three places for one change.

Inference entrypoints already used JSON configs via `--json_path`. This session
extends that pattern to training, collapsing the shell chain from 3 layers to 1
and replacing all bash arglist files with JSON configs.

## Changes

### 1. Rename `--json_path` → `--config_path` across all entrypoints

All 3 inference scripts (`run_PenCL_inference.py`, `run_Facilitator_sample.py`,
`run_ProteoScribe_sample.py`) renamed `--json_path` to `--config_path` for
format-agnostic naming. Updated all attribute references (`.json_path` →
`.config_path`), README, demos, and docs.

### 2. Add `--config_path` to training entrypoint

**`src/biom3/Stage3/run_PL_training.py`** — modified `retrieve_all_args()`:
- Pre-parser extracts `--config_path` via `parse_known_args`
- If provided, loads JSON via `load_json_config()` from `biom3.core.helpers`
- Calls `parser.set_defaults(**json_dict)` so JSON values become defaults
- Priority: CLI > JSON > argparse defaults
- Added import of `load_json_config` from `biom3.core.helpers`

**Bug fix:** Added missing `str_to_bool(args.finetune_output_layers)` to the
type conversions block. Previously never converted; test masked it by setting
the value directly as a bool.

### 3. New metadata arguments

Added three new argparse arguments to `get_args()`:
- `--_description` (str, default `""`) — loaded from JSON configs
- `--tags` (str, nargs='+', default `[]`) — for categorizing runs
- `--notes` (str, nargs='+', default `[]`) — free-form run notes

All three are stored in the `args.json` artifact automatically.

### 4. JSON training configs

Created `configs/training/` with 7 JSON files converted from the shell arglists:

| JSON config | Key differences |
|---|---|
| `pretrain_scratch_v1.json` | 1 transformer block, paper architecture |
| `pretrain_scratch_v2.json` | 16 transformer blocks |
| `pretrain_scratch_v3.json` | 16 transformer blocks, lr=1e-2 |
| `pretrain_start_pfam_v1.json` | Phase 2 with Pfam, 1 block |
| `pretrain_start_pfam_v2.json` | Phase 2 with Pfam, 16 blocks, seed=162 |
| `finetune_v1.json` | Finetuning, 1 block, with pretrained weights |
| `finetune_v2.json` | Same as v1, no `finetune_last_n_layers` |

Conversion: `"None"` → `null`, `True/False` → `true/false`, proper JSON types,
`wandb_tags` as arrays, dropped `run_id` and `traindata_len` (runtime values).

### 5. Simplified shell scripts

**New unified wrappers (2 files):**
- `scripts/stage3_train_multinode.sh` — 6 positional args + `"$@"` passthrough
- `scripts/stage3_train_singlenode.sh` — 5 positional args + `"$@"` passthrough

**Updated 15 HPC job templates** (`jobs/{polaris,aurora,spark}/`) to use new
wrappers with `--config_path` and named `--key value` overrides instead of
positional args.

**Deleted 8 old files:**
- `scripts/stage3_pretraining.sh`, `scripts/stage3_finetuning.sh`
- `scripts/pretraining/{pretrain_multinode,pretrain_singlenode}.sh`
- `scripts/finetuning/{finetune_multinode,finetune_singlenode}.sh`
- Empty directories `scripts/pretraining/`, `scripts/finetuning/`

**Deleted 7 arglist files + directory:**
- All files in `arglists/` + the directory itself

### 6. Tests

Added 9 new tests to `tests/stage3_tests/test_stage3_run_PL_training.py`:
- `test_parse_from_json_config` (parametrized × 4 configs)
- `test_cli_overrides_json`
- `test_json_overrides_argparse_defaults`
- `test_json_native_types`
- `test_description_tags_notes`
- `test_no_config_path_uses_defaults`

### 7. Documentation

- `CLAUDE.md`: updated Configuration section, Repository layout, Distributed training
- `README.md`: rewrote Stage 3 Pretraining/Finetuning sections with examples

## Before → After

**Before (4 layers):**
```
HPC template
  → pretrain_multinode.sh (positional args, mpiexec)
    → stage3_pretraining.sh (source arglist.sh, enumerate 40+ --flag ${var})
      → biom3_pretrain_stage3 (argparse)
```

**After (2 layers):**
```
HPC template
  → stage3_train_multinode.sh (6 positional + "$@", mpiexec)
    → biom3_pretrain_stage3 --config_path *.json --overrides... (argparse)
```

## Testing

- `pytest tests/test_imports.py` — 5/5 passed
- `pytest tests/core_tests/test_run_utils.py` — 14/14 passed
- `pytest tests/stage3_tests/test_stage3_run_PL_training.py -k "json or config"` — 9/9 passed
- `grep` sweep: zero references to `arglists`, old script names, or `--json_path` in source

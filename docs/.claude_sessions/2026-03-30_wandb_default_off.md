# Session: Disable wandb by default, add explicit opt-in

**Date:** 2026-03-30

## Context

A new user following the BioM3-workflow-demo README was hitting issues because wandb
logging was enabled by default in `run_PL_training.py` (`--wandb` defaulted to `"True"`).
Users without a wandb setup shouldn't need to configure it just to run training.

## Changes

1. **`src/biom3/Stage3/run_PL_training.py`** — Changed `--wandb` argparse default from
   `"True"` to `"False"`.

2. **`scripts/stage3_pretraining.sh` and `scripts/stage3_finetuning.sh`** — Added
   `--wandb ${wandb}` to the argument list passed to `biom3_pretrain_stage3`.

3. **All 7 arglist configs** (`arglists/config_*.sh`) — Added `export wandb=True` so
   existing training configs continue to use wandb.

4. **All 15 job templates** (`jobs/{spark,polaris,aurora}/_template_*.{sh,pbs}`) — Added
   `wandb=True` variable declaration alongside `wandb_api_key`.

## Side notes

- Also resolved a pip caching issue where a user installing via
  `pip install git+...` was only seeing 3 of 6 entrypoints. A
  `--no-cache-dir --force-reinstall` fixed it.
- Tab completion for entrypoints after `pip install git+...` requires `hash -r` or a
  new shell session.

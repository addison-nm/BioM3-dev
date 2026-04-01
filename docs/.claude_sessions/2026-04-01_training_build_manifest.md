# 2026-04-01 — Training Build Manifest

## Summary

Added a `build_manifest.json` to the Stage 3 training entrypoint
(`run_PL_training.py`) for reproducibility and versioning, following the
same pattern already used by the inference and dataset-building entrypoints.
Also extended the shared `write_manifest()` utility with an optional
`environment` parameter, and added filtering to prevent sensitive environment
variables (API keys, tokens, etc.) from being written to the manifest.

## Pre-session state

```bash
git checkout 44f09f3   # feat: add dual logging, build manifests, and fix taxonomy filter
```

## Changes

### `src/biom3/core/run_utils.py`

- Added optional `environment` keyword argument to `write_manifest()`.
  When provided, it is written as an `"environment"` key in the manifest
  JSON. Fully backwards-compatible with all existing callers.

### `src/biom3/Stage3/run_PL_training.py`

- **New imports**: `datetime`, `write_manifest`, `get_rank`.
- **`_TRAINING_ENV_PREFIXES`** (module constant): tuple of environment
  variable prefixes to capture — covers CUDA, NCCL, Torch, DeepSpeed,
  WandB, distributed launcher vars (MASTER, RANK, SLURM, PBS, COBALT,
  PALS, PMI, OMPI), and threading vars (OMP, MKL).
- **`_SENSITIVE_ENV_SUBSTRINGS`** (module constant): tuple of substrings
  (`KEY`, `SECRET`, `TOKEN`, `PASSWORD`, `CREDENTIAL`, `AUTH`) used to
  filter out sensitive variables like `WANDB_API_KEY`.
- **`_collect_training_env()`**: collects hostname and filtered env vars.
- **`main()` changes**:
  - Records `start_time = datetime.now()` at the top.
  - After `train_model()` returns, on rank 0 only, computes elapsed time
    and writes `build_manifest.json` to the checkpoint directory with:
    - **outputs**: seed, total/trainable params, batch size, effective LR,
      precision, GPU devices, nodes, gradient accumulation steps,
      DeepSpeed stage, epochs or max_steps, and finetuning config when
      applicable.
    - **resolved_paths**: checkpoint dir, data roots, pretrained weights,
      resume checkpoint (all absolute paths, included conditionally).
    - **environment**: hostname + captured env vars.
    - Standard versioning fields from `write_manifest` (biom3
      version/build, git hash/branch/dirty/remote, python version,
      timestamp, elapsed time, full command line, all args).

## Testing

- Verified clean import of the modified module.
- Verified `_collect_training_env()` correctly excludes `WANDB_API_KEY`
  and `TORCH_AUTH_TOKEN` while passing through safe variables like
  `WANDB_PROJECT` and `CUDA_VISIBLE_DEVICES`.

## Notes

- No new tests were added; the manifest write is a thin wrapper around
  `write_manifest()` which is already exercised by the existing
  entrypoint tests.
- The `environment` parameter is available to all entrypoints now but
  only used by the training script so far.

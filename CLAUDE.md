# CLAUDE.md

## Practices

Store session notes in docs/.claude_sessions/

## Parallel development with worktrees

For large features, prefer working in a git worktree branched off `addison-dev`, so multiple Claude agents can work in parallel without stepping on each other.

- Worktrees live under `.claude/worktrees/<feature-name>/` (gitignored).
- Create with: `git worktree add .claude/worktrees/<feature> -b addison-<feature> addison-dev`
- When a feature is ready, merge its branch back into `addison-dev`. Don't commit directly to `addison-dev` from the main checkout while worktrees are active.
- Remove finished worktrees with `git worktree remove .claude/worktrees/<feature>`.

While working in a worktree:
- Edit only the files the feature requires. If bugs are spotted in unrelated areas, note them (e.g., in the session log or a TODO) but do not fix them in this worktree — they belong to their own branch.
- Each new worktree starts without populated `data/databases/` or `data/datasets/`. Repopulate them via the appropriate `scripts/sync_*.sh` script before running anything that reads from them.

Small fixes and docs edits can still be made directly on `addison-dev`.

## Project overview

BioM3 is a multi-stage framework for generating novel protein sequences guided by natural language prompts (NeurIPS 2024). It combines protein language models (ESM-2), biomedical text encoders (BioBERT), and diffusion-based sequence generation.

**Pipeline:**
```
Input (CSV: sequences + text)
  → Stage 1: PenCL inference (ESM-2 + BioBERT → joint embeddings z_t, z_p)
  → Stage 2: Facilitator (z_t → z_c, aligned to protein space)
  → Stage 3: ProteoScribe (diffusion model, z_c → generated sequences)
```

## Ecosystem context

BioM3-dev is the core library in a multi-repo ecosystem. See [docs/biom3_ecosystem.md](docs/biom3_ecosystem.md) for full details.

Related repositories:
- **BioM3-data-share** — shared model weights, datasets, and reference databases
- **BioM3-workflow-demo** — end-to-end demo of finetuning and generation
- **BioM3-workspace-template** — *(planned)* workspace configuration template

Machine-specific repo paths are in `.claude/repo_paths.json` (gitignored, not version controlled). This file maps repo names to absolute paths on the current machine.

## Repository layout

```
src/biom3/
  backend/          # Device abstraction (CPU / CUDA / XPU)
  core/             # Shared utilities: model I/O (io.py), state-dict helpers (helpers.py)
  dbio/             # Database I/O: readers for SwissProt/Pfam/NCBI taxonomy, enrichment, dataset building
  Stage1/           # PenCL: model.py (encoders + projection), preprocess.py, PL_wrapper.py
  Stage2/           # Facilitator: run_Facilitator_sample.py
  Stage3/           # ProteoScribe: diffusion model, PL training, sampling
configs/            # JSON configs for each stage's inference, training, and dbio
  inference/        #   Inference configs with models/ base configs (uses _base_configs composition)
  training/         #   Training configs with models/ and machines/ base configs
scripts/            # Bash wrappers (embedding_pipeline, training, generation, sync)
demos/              # End-to-end demos (dbio dataset building, SH3 embedding pipeline)
data/databases/     # Symlinked reference databases (gitignored, see docs/setup_databases.md)
tests/              # pytest suite (conftest.py, per-stage tests, test data in tests/_data/)
weights/            # Pre-trained model weights (gitignored, see weights/README.md)
docs/               # Per-machine setup guides (Polaris, Aurora, DGX Spark)
jobs/               # HPC job submission scripts
```

## Entry points

Defined in `pyproject.toml`:
- `biom3_PenCL_inference` → `biom3.Stage1.__main__:run_PenCL_inference`
- `biom3_Facilitator_sample` → `biom3.Stage2.__main__:run_Facilitator_sample`
- `biom3_pretrain_stage3` → `biom3.Stage3.__main__:run_stage3_pretraining`
- `biom3_ProteoScribe_sample` → `biom3.Stage3.__main__:run_ProteoScribe_sample`
- `biom3_build_dataset` → `biom3.dbio.__main__:run_build_dataset`
- `biom3_build_taxid_index` → `biom3.dbio.__main__:run_build_taxid_index`
- `biom3_convert_to_parquet` → `biom3.dbio.__main__:run_convert_to_parquet`

## Building and running

```bash
# Install (editable, from repo root)
pip install -e .

# Smoke only (imports)
pytest tests/test_imports.py

# Fast dev loop — skips entrypoint/training tests (~3 min, CPU, no weights required)
pytest tests/ --quick

# Full test suite (default — includes entrypoint + training tests; weight-gated tests skip if weights missing)
pytest tests/

# Include GPU-only tests
pytest tests/ --use_gpu
```

## Testing conventions

- Test files live in `tests/` with subdirectories per stage.
- Test data is in `tests/_data/`; test outputs go to `tests/_tmp/`.
- CLI arguments for entrypoint tests are stored in `tests/_data/entrypoint_args/*.txt`.
- Custom pytest markers:
  - `@pytest.mark.benchmark` (needs `--benchmark` to run)
  - `@pytest.mark.use_gpu` (needs `--use_gpu` to run)
  - `@pytest.mark.network` (needs `--network` to run)
  - `@pytest.mark.database_files` (needs `--database_files` to run)
  - `@pytest.mark.slow` (skipped under `--quick`; applied module-wide to entrypoint + training + pipeline tests)
- Tests that require downloaded weights skip gracefully with a message if files are missing.
- For the fast dev loop, prefer `pytest tests/ --quick` over `pytest tests/test_imports.py` — the former covers dbio, viz, Stage 3 sampling/data-splitting/model-IO, and core utilities without needing weights or GPU.

## Code style

### Python conventions
- **Classes**: PascalCase (`ProteinEncoder`, `PEN_CL`, `PL_ProtARDM`)
- **Functions / variables**: snake_case (`load_json_config`, `prepare_model`)
- **Constants**: UPPER_SNAKE_CASE (`DATDIR`, `TMPDIR`, `BACKEND_NAME`)
- **Private helpers**: leading underscore (`_load_state_dict_from_file`)
- Keep type hints lightweight — use them on public function signatures but don't over-annotate internals.
- Avoid adding docstrings, comments, or type annotations to code you didn't change.

### File organization pattern
Each stage follows a consistent layout:
- `model.py` — nn.Module definitions
- `preprocess.py` — datasets and collate functions
- `PL_wrapper.py` — PyTorch Lightning modules
- `run_*.py` — end-to-end scripts (arg parsing, loading, inference/training loop)
- `io.py` — model building and checkpoint loading
- `__main__.py` — thin wrappers that call into `run_*.py`

### Imports
- Group: stdlib → third-party → project (`biom3.*`)
- Device-conditional imports (e.g., `lightning` vs `pytorch_lightning`) go behind `if BACKEND_NAME == _XPU:` guards in `backend/device.py`.

## Commit style

Use [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>: <short summary>

<optional body with context>
```

Types: `feat`, `fix`, `refactor`, `test`, `docs`, `chore`, `perf`

Examples:
- `feat: add --load_from_checkpoint flag to Stage 1 inference`
- `fix: correct BERT padding to match training config`
- `refactor: extract raw-weight and checkpoint loaders in run_PenCL_inference`

Keep the summary under 72 characters. Use the body for "why", not "what".

## Key architectural details

### Device abstraction
`biom3.backend.device` detects the available backend (CUDA → XPU → CPU) and exposes `get_device()`, `get_backend_name()`. All device-specific code lives in `backend/{cuda,xpu,cpu}.py`. When writing new code, import from `backend.device` rather than hardcoding `torch.device("cuda")`.

### Checkpoint formats
The codebase handles three weight formats:
1. **Raw weights** (`.bin`, `.pt`) — plain `state_dict`, loaded via `core.io.load_and_prepare_model`
2. **Lightning checkpoints** (`.ckpt`) — loaded via `PL_wrapper.load_from_checkpoint`, then unwrap `.model`
3. **DeepSpeed sharded** (directory) — merged via `Stage3.io._load_state_dict_from_sharded_dir`

When loading models, use `core.io.load_and_prepare_model` for raw weights. For Lightning checkpoints, use the stage-specific `prepare_model_from_checkpoint` functions which handle PL wrapper construction and unwrapping.

### Configuration
- **Inference**: JSON files in `configs/inference/` → loaded via `--config_path` with `load_json_config()` → converted to `argparse.Namespace`. Old flat configs in `configs/` still work for backward compatibility.
- **Training**: JSON files in `configs/stage3_training/` → loaded via `--config_path` into argparse defaults. CLI args override JSON values; JSON overrides argparse defaults.
- **Config composition**: All stages (inference and training) use `core.helpers.load_json_config()`, which supports two special keys:
  - `_base_configs`: list of paths loaded *before* the current file (current file overrides them)
  - `_overwrite_configs`: list of paths loaded *after* the current file (they override it)
  - Priority (low → high): `_base_configs` < current file < `_overwrite_configs` < CLI
  - Paths resolve relative to the JSON file's directory. Both keys are stripped from the result.
- **Base configs**: `configs/inference/models/` has shared encoder/model configs (`_base_PenCL.json`, `_base_Facilitator.json`). `configs/stage3_training/models/` has shared model architecture configs (`_base_ProteoScribe_1block.json`, `_base_ProteoScribe_16blocks.json`). `configs/stage3_training/machines/` has per-machine device configs (`_aurora.json`, `_polaris.json`, `_spark.json`). Stage 3 inference reuses the training model base configs via `_base_configs`.

### Training output structure
Stage 3 training (`biom3_pretrain_stage3`) organizes outputs under `--output_root` with three key CLI args: `--checkpoints_folder` (default `checkpoints`), `--runs_folder` (default `runs`), and `--run_id` (unique per run, constructed automatically by HPC job templates).

```
{output_root}/
├── {checkpoints_folder}/{run_id}/    ← Lightning/DeepSpeed .ckpt dirs + derived weights
├── {runs_folder}/{run_id}/
│   ├── logs/                         ← lightning_logs/, wandb/
│   └── artifacts/                    ← state_dict.best.pth copy, args.json,
│                                       build_manifest.json, run.log
```

### Distributed training
Stage 3 supports multi-node training via DeepSpeed + PyTorch Lightning. The `scripts/stage3_train_multinode.sh` wrapper uses `mpiexec`. Environment variable `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=true` is set in training scripts.

## Things to watch out for

- **Padding mismatch bug** (documented in `docs/bug_reports/bert_embedding_mismatch.md`): BERT text encoder must use `padding="max_length"` with `max_length=512` to match training. Dynamic padding produces different embeddings because no `attention_mask` is passed to the model.
- **strict=False** is used when loading PenCL weights because some checkpoint keys may not match the current model graph. This is intentional but means missing/extra keys are silently ignored — be careful when changing model architecture.
- **PL wrapper vs nn.Module**: PL wrappers store the model as `.model`. Inference scripts must unwrap to get the raw `nn.Module` for the forward pass.

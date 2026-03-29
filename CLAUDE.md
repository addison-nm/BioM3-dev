# CLAUDE.md

## Practices

Store session notes in docs/claude_sessions/

## Project overview

BioM3 is a multi-stage framework for generating novel protein sequences guided by natural language prompts (NeurIPS 2024). It combines protein language models (ESM-2), biomedical text encoders (BioBERT), and diffusion-based sequence generation.

**Pipeline:**
```
Input (CSV: sequences + text)
  → Stage 1: PenCL inference (ESM-2 + BioBERT → joint embeddings z_t, z_p)
  → Stage 2: Facilitator (z_t → z_c, aligned to protein space)
  → Stage 3: ProteoScribe (diffusion model, z_c → generated sequences)
```

## Repository layout

```
src/biom3/
  backend/          # Device abstraction (CPU / CUDA / XPU)
  core/             # Shared utilities: model I/O (io.py), state-dict helpers (helpers.py)
  Stage1/           # PenCL: model.py (encoders + projection), preprocess.py, PL_wrapper.py
  Stage2/           # Facilitator: run_Facilitator_sample.py
  Stage3/           # ProteoScribe: diffusion model, PL training, sampling
configs/            # JSON configs for each stage's inference
arglists/           # Shell-variable configs for Stage 3 training
scripts/            # Bash wrappers (embedding_pipeline, pretraining, finetuning, generation)
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

## Building and running

```bash
# Install (editable, from repo root)
pip install -e .

# Run tests (CPU-only, no downloaded weights needed for import tests)
pytest tests/test_imports.py

# Full test suite (requires weights/ to be populated)
pytest tests/

# GPU-specific tests
pytest tests/ --use_gpu
```

## Testing conventions

- Test files live in `tests/` with subdirectories per stage.
- Test data is in `tests/_data/`; test outputs go to `tests/_tmp/`.
- CLI arguments for entrypoint tests are stored in `tests/_data/entrypoint_args/*.txt`.
- Custom pytest markers: `@pytest.mark.benchmark` (needs `--benchmark`), `@pytest.mark.use_gpu` (needs `--use_gpu`).
- Tests that require downloaded weights skip gracefully with a message if files are missing.

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
- **Inference**: JSON files in `configs/` → loaded with `load_json_config()` → converted to `argparse.Namespace`
- **Training**: Shell-variable files in `arglists/` → sourced by wrapper scripts in `scripts/`

### Distributed training
Stage 3 supports multi-node training via DeepSpeed + PyTorch Lightning. The `scripts/pretraining/pretrain_multinode.sh` wrapper uses `mpiexec`. Environment variable `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=true` is set in training scripts.

## Things to watch out for

- **Padding mismatch bug** (documented in `docs/BUG_bert_embedding_mismatch.md`): BERT text encoder must use `padding="max_length"` with `max_length=512` to match training. Dynamic padding produces different embeddings because no `attention_mask` is passed to the model.
- **strict=False** is used when loading PenCL weights because some checkpoint keys may not match the current model graph. This is intentional but means missing/extra keys are silently ignored — be careful when changing model architecture.
- **PL wrapper vs nn.Module**: PL wrappers store the model as `.model`. Inference scripts must unwrap to get the raw `nn.Module` for the forward pass.

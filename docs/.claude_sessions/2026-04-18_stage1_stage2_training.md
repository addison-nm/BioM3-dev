# Session: Stage 1 & Stage 2 Training Entrypoints

**Date:** 2026-04-18
**Branch:** `addison-stage1-stage2-training`
**Worktree:** `.claude/worktrees/stage1-stage2-training/`
**Plan:** `~/.claude/plans/i-need-to-implement-peppy-pebble.md`

## Context

BioM3-dev had shipped Stage 3 (ProteoScribe) training but not Stage 1 (PenCL)
or Stage 2 (Facilitator). Collaborator Rama (`~/Projects/BioM3-dev-space/rama-code/`)
has production-tested Stage 1 and Stage 2 training code, exercised at
256-node scale on Aurora XPU. The goal of this session was to port that
training into the modern BioM3-dev package under the Stage 3 conventions —
`run_PL_training.py` pattern, `_base_configs` JSON composition, per-run
`{output_root}/{checkpoints_folder|runs_folder}/{run_id}/` output layout,
`build_manifest.json`, `args.json`, `run.log` — and expose two new
entrypoints `biom3_pretrain_stage1` and `biom3_pretrain_stage2`.

## Source-of-truth verification

Before porting, traced every `.pbs` invocation in `rama-code/jobs/` against
executed runs in `rama-code/logs/` to identify what's actually used in
production:

- **Stage 1 entry:** `rama-code/code/PL_training.py` (not `pencl_sweep_*.py`
  — those are hyperparameter sweep scripts). Imports `Stage1_source.*`.
  All production PenCL jobs (`pencl_run1_prod.pbs`, `pencl_256node_prod.pbs`,
  `pencl_16node_prod.pbs`, etc.) invoke this script.
- **Stage 2 entry:** `rama-code/code/Stage2_PL_training.py`. Also imports
  `Stage1_source.*` for the `Facilitator_DataModule` and `PL_Facilitator`.
- **Production mode:** `dataset_type=pfam` (SwissProt + Pfam) — uses
  `Pfam_DataModule` + `pfam_PEN_CL` + `pfam_PL_PEN_CL`, not default-only.
- **Ignored:** `rama-code/code/source/` (older snapshot, superseded by
  `Stage1_source/` with `.bak` siblings), `pencl_sweep_*.py`, Stage 3
  finetuning files, XPU-specific `train_condition.py`.

## Parity delta vs Rama

Training configs follow the **inference base configs** (not Rama's
production values), so any weights produced are drop-in compatible with the
shipped inference stack:

| Parameter | Rama production | Training config choice |
|---|---|---|
| `proj_embedding_dim` | 256 | 512 |
| `temperature` | 1.0 | 0.8 |
| `pLM_n_layers_to_finetune` | 4 | 1 |
| `bLM_n_layers_to_finetune` | 4 | 1 |
| `emb_dim` (Facilitator) | 256 | 512 |
| Strategy | `auto` (XPU) / `ddp` (CUDA) | `ddp` on CUDA, `auto` on single-device |
| Scheduler | none | none |

Per-group LRs (`head_lr=1e-3`, `protein_encoder_lr=1e-4`,
`text_encoder_lr=1e-6`, `weight_decay=1e-6`) follow Rama since inference
base doesn't specify these.

## What changed (Stage 1 & Stage 2 source only)

### Phase A — preprocess.py bug-fix + class ports

`src/biom3/Stage1/preprocess.py`:
- Ported from `rama-code/code/Stage1_source/preprocess.py`:
  - `MaskTextSeqPairing_Dataset` (165 lines) — 15% token masking for MLM
    auxiliary objective.
  - `Pfam_TextSeqPairing_Dataset` (~270 lines) — SwissProt + Pfam joint
    sampling with stochastic field dropout (`apply_field_dropout` +
    `_FIELD_PREFIXES` + `_RETENTION_PROBS`).
  - `Pfam_DataModule` (~270 lines) — rank-0 shard prep with `dist.barrier()`
    sync, per-rank shard loading, 10 OOD Pfam families held out for
    zero-shot eval.
  - `check_available_memory()` helper.
- Fixed two pre-existing bugs that prevented `Default_DataModule` from
  instantiating with non-default `dataset_type`:
  - Unresolved references to `MaskTextSeqPairing_Dataset` /
    `Pfam_TextSeqPairing_Dataset` at lines 205–207 (now resolved).
  - Undefined `check_available_memory()` call at line 231 (now defined).
- **Deviation from Rama:** `Pfam_DataModule._resolve_splits_dir()` prefers
  `args.pfam_splits_dir` over the Rama default (a directory next to the
  input CSV). The Stage 1 entrypoint points this at
  `{output_root}/{runs_folder}/{run_id}/pfam_splits/` so shard writes land
  inside the run directory.
- **Minor fix from Rama:** `MaskTextSeqPairing_Dataset` referenced a bare
  `sequence_keyword` (undefined module-level name) at its line 191;
  ported as `args.sequence_keyword`. Same fix for the hardcoded
  `'primary_Accession'` → `args.id_keyword`.

### Phase B — Stage 1 training entrypoint

`src/biom3/Stage1/run_PL_training.py` (new, ~580 lines). Mirrors
`src/biom3/Stage3/run_PL_training.py` structure:

- `retrieve_all_args()` — pre-parses `--config_path`, loads JSON via
  `core.helpers.load_json_config`, builds full parser with grouped
  helpers (`get_args`/`get_model_args`/`get_path_args`/`get_wrapper_args`),
  runs type-conversion pass.
- `get_dataloaders_models()` — dispatches on `dataset_type`:
  - `default` → `Default_DataModule` + `PEN_CL` + `PL_PEN_CL`
  - `masked` → `Default_DataModule` + `PEN_CL` + `mask_PL_PEN_CL`
  - `pfam` / `pfam_ablated` → `Pfam_DataModule` + `pfam_PEN_CL` +
    `pfam_PL_PEN_CL`
- `train_model()` — plain-DDP Trainer on CUDA, `auto` strategy on
  single-device / XPU. Reuses
  `biom3.Stage3.callbacks.{LoggingModelCheckpoint, MetricsHistoryCallback,
  TrainingBenchmarkCallback}`, `biom3.core.run_utils.{write_manifest,
  setup_file_logging, backup_if_exists}`.
- Machine-agnostic precision handling: CUDA uses Trainer native AMP for
  bf16; XPU wraps `training_step`/`validation_step` with manual
  `torch.autocast(device_type='xpu', dtype=torch.bfloat16)` and forces
  Trainer precision to `'32'` (Rama's pattern, since PL's bf16 plugin is
  broken on Aurora).
- `save_model()` — plain-DDP save path: load best `.ckpt`, unwrap `model.`
  prefix, save as `state_dict.best.pth` under both checkpoint_dir and
  artifacts_dir.

`src/biom3/Stage1/__main__.py`: added `run_stage1_pretraining()`
dispatcher (mirrors `run_stage3_pretraining()` pattern).

### Phase B.5 — PL_wrapper.py: `_safe_barrier()` for machine agnosticism

`src/biom3/Stage1/PL_wrapper.py` had 18 unguarded `dist.barrier()` calls
in `PL_PEN_CL.training_step`/`validation_step`, `mask_PL_PEN_CL`,
`pfam_PL_PEN_CL`. These failed with
`ValueError: Default process group has not been initialized` on
single-GPU or CPU runs (including the new smoke tests).

Added a module-level `_safe_barrier()` helper:
```python
def _safe_barrier():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
```
Replaced all 18 calls via `Edit(replace_all=True)`. Behaviour is unchanged
in distributed mode; single-device and CPU now work. This is the
`_safe_barrier` pattern Rama's production code uses.

### Phase C — Stage 2 training entrypoint

`src/biom3/Stage2/run_PL_training.py` (new, ~470 lines). Simpler than
Stage 1 (no tokenizers, no per-group LRs):

- Imports `PL_Facilitator` from `biom3.Stage1.PL_wrapper` and
  `Facilitator_DataModule` from `biom3.Stage1.preprocess` via thin
  Stage 2 re-export modules.
- Arg surface matches Rama's `Stage2_PL_training.py`:
  `--swissprot_data_path`, `--pfam_data_path` (string `"None"` for absent),
  `--output_swissprot_dict_path`, `--output_pfam_dict_path`, `--emb_dim`,
  `--hid_dim`, `--loss_type` ∈ `{MSE, MMD}`, `--lr`, plus common training
  flags.
- After `trainer.fit()`, runs `trainer.predict()` on the full
  SwissProt / Pfam dataloaders and writes the transformed embedding dict
  with `text_to_protein_embedding` keys for Stage 3 consumption
  (`get_final_embeddings()` + `save_embeddings()`).

Thin re-exports so Stage 2 consumers don't reach across stages:
- `src/biom3/Stage2/PL_wrapper.py` — re-exports `PL_Facilitator`.
- `src/biom3/Stage2/preprocess.py` — re-exports `Facilitator_Dataset`,
  `Facilitator_DataModule`.

`src/biom3/Stage2/__main__.py`: added `run_stage2_pretraining()`
dispatcher.

### Phase D — Training configs

- `configs/stage1_training/models/_base_PenCL_train.json` — inference-base
  parity (512/0.8/1/1) plus `sequence_keyword`, `id_keyword`,
  `model_type="default"`, `dataset_type="default"`.
- `configs/stage1_training/machines/_aurora.json`, `_polaris.json`,
  `_spark.json` — copies of Stage 3 machine configs
  (`{device, gpu_devices}`).
- `configs/stage1_training/pretrain_scratch_v1.json` — default-mode training.
- `configs/stage1_training/pretrain_pfam_v1.json` — pfam-mode training
  (production path).
- `configs/stage2_training/models/_base_Facilitator_train.json` —
  `emb_dim=512`, `hid_dim=1024`, `dropout=0.1`, `loss_type="MSE"`.
- `configs/stage2_training/machines/{_aurora,_polaris,_spark}.json`.
- `configs/stage2_training/pretrain_scratch_v1.json`.

### Phase E — Infrastructure

- `pyproject.toml` — registered two new entry points:
  ```
  biom3_pretrain_stage1 = "biom3.Stage1.__main__:run_stage1_pretraining"
  biom3_pretrain_stage2 = "biom3.Stage2.__main__:run_stage2_pretraining"
  ```
- `scripts/stage1_train_singlenode.sh`, `scripts/stage2_train_singlenode.sh`
  — copies of `stage3_train_singlenode.sh` with swapped entrypoint.
- `jobs/spark/_template_stage1_pretrain.sh`,
  `jobs/spark/_template_stage2_pretrain.sh` — single-node templates.
  Polaris/Aurora deferred.
- `tests/stage1_tests/test_stage1_run_PL_training.py`,
  `tests/stage2_tests/test_stage2_run_PL_training.py` — smoke tests
  modeled on `test_stage3_run_PL_training.py`. Each runs one epoch on
  tiny input and asserts `state_dict.best.pth`, `args.json`, and
  `build_manifest.json` land in the right spots. Stage 2 also asserts the
  transformed embedding dict has `text_to_protein_embedding`.
- `tests/_data/entrypoint_args/training/stage{1,2}_training_args_scratch_v1.txt`.

## Environment setup (Phase 0)

Done once at start of session:

```
git -C <main-checkout> worktree add -b addison-stage1-stage2-training \
    .claude/worktrees/stage1-stage2-training addison-dev

cd .claude/worktrees/stage1-stage2-training
./scripts/sync_weights.sh   /data/data-share/BioM3-data-share/data/weights weights
./scripts/sync_databases.sh /data/data-share/BioM3-data-share/databases   data/databases
conda run -n biom3-env pip install -e .
```

Sync scripts created 15 weight symlinks and 1544 database symlinks.

## Verification

From the worktree:

```
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=true conda run -n biom3-env \
    pytest tests/test_imports.py tests/stage1_tests/ tests/stage2_tests/
```

Result: **15 passed, 6 skipped, 2 warnings in 98.35s** (6 skipped are XPU
and CPU-device variants on a CUDA-only host; pre-existing Stage 1/2
inference tests still pass).

Stage 1 smoke: one epoch of default-mode PenCL on
`tests/_data/stage1_inputs/sample_text_seqs1.csv` (5 rows, batch_size=2)
with the full 762M-param PenCL model (28.3M trainable with
`pLM_n_layers_to_finetune=1`, `bLM_n_layers_to_finetune=1`). `train_loss`
logged, `valid_loss` checkpoint written.

Stage 2 smoke: one epoch on an 8-row synthetic embeddings fixture,
Facilitator model (~1M params) in MSE-loss mode, followed by a predict
pass that produces a `text_to_protein_embedding` tensor of matching
length.

## Not done / deferred

- **pfam-mode smoke test.** Requires a tiny hand-crafted
  `sample_pfam.csv` fixture with `id`, `pfam_label`, `sequence`,
  `[final]text_caption` columns. Skipped; `pretrain_pfam_v1.json` is
  untested end-to-end.
- **XPU pathway.** Code is wired (manual bf16 autocast, `_safe_barrier`)
  but hasn't been exercised on Aurora. `dist.init_process_group(backend='xccl')`
  setup is expected to come from the multi-node wrapper, which also
  remains deferred.
- **Multi-node job templates.** Polaris/Aurora templates not written;
  Spark single-node only.
- **LR schedulers, EMA, `scale_learning_rate=True`.** Arg surface exists
  but schedulers aren't wired in the PL wrappers (Rama uses none); EMA
  not ported.
- **Inference entry point for Stage 1 training output.** The existing
  `run_PenCL_inference.py` consumes the output checkpoint format, so no
  changes needed there.

## Files touched

**New:**
- `src/biom3/Stage1/run_PL_training.py`
- `src/biom3/Stage2/run_PL_training.py`
- `src/biom3/Stage2/PL_wrapper.py` (re-export)
- `src/biom3/Stage2/preprocess.py` (re-export)
- `configs/stage1_training/` (7 files)
- `configs/stage2_training/` (5 files)
- `scripts/stage1_train_singlenode.sh`, `scripts/stage2_train_singlenode.sh`
- `jobs/spark/_template_stage{1,2}_pretrain.sh`
- `tests/stage{1,2}_tests/test_stage{1,2}_run_PL_training.py`
- `tests/_data/entrypoint_args/training/stage{1,2}_training_args_scratch_v1.txt`

**Modified:**
- `src/biom3/Stage1/preprocess.py` (+~480 lines: three dataset/module ports)
- `src/biom3/Stage1/PL_wrapper.py` (added `_safe_barrier()`; replaced 18 calls)
- `src/biom3/Stage1/__main__.py` (+5: `run_stage1_pretraining` dispatcher)
- `src/biom3/Stage2/__main__.py` (+5: `run_stage2_pretraining` dispatcher)
- `pyproject.toml` (+2 lines: entry-point registrations)

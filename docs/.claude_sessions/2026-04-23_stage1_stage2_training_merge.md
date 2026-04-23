# Session: Stage 1 & Stage 2 Training — Aurora bring-up, pfam CI, merge

**Date:** 2026-04-23
**Branch:** `addison-stage1-stage2-training` → merged into `addison-dev` at `9182c61`
**Worktree:** `.claude/worktrees/stage1-stage2-training/` (removed post-merge)
**Predecessor:** [2026-04-18_stage1_stage2_training.md](2026-04-18_stage1_stage2_training.md)

## Context

The 2026-04-18 session ported Rama's Stage 1 (PenCL) and Stage 2
(Facilitator) training into the BioM3-dev package under Stage 3
conventions. It paused with three items before merge:
Stage 2 XPU device hardcode, pfam-mode CI coverage, and Polaris/Aurora
job templates. The user also deferred the explicit merge gate: merge
only after pfam-mode was exercised on both CUDA and XPU.

Today's session closed all three items in sequence, debugged four
Aurora XPU failures that only surfaced on real hardware, and merged
into `addison-dev`.

## Aurora bring-up — four bugs surfaced across four runs

Each Aurora `python -m pytest tests` round exposed one root cause and
was fixed before the next run. All diagnosed from stack traces alone
(no reproducer on local Spark CUDA, because each bug required the
custom Lightning install and/or an initialized XPU process group).

### Run 1 → `TypeError: model must be a LightningModule, got PL_PEN_CL`

Aurora's custom PyTorch Lightning install exposes the module as
`lightning`; stock `pytorch_lightning` on other machines. Our
`Stage1/PL_wrapper.py` and `Stage1/preprocess.py` imported
`pytorch_lightning.LightningModule` / `LightningDataModule`
unconditionally. `Trainer.fit()` from `lightning` checks
`isinstance(model, lightning.LightningModule)` — fails against
wrappers rooted at `pytorch_lightning.LightningModule`.

**Fix (`27405fe`):** mirror Stage 3's `BACKEND_NAME == _XPU` guard in
both files so inheritance resolves to the right PL class per machine.

### Run 2 → `DeviceStatsMonitor: Expected a xpu device, but got: cpu`

Got past the isinstance error and into the fit loop; failed on
`on_train_batch_start` trying to read memory stats. Tracing into
Lightning's `_choose_strategy()` at
[accelerator_connector.py:450-458](../../../Projects/BioM3-dev-space/lightning/src/lightning/pytorch/trainer/connectors/accelerator_connector.py#L450-L458):
it only recognizes CUDA/MPS/GPU as GPU accelerators for single-device;
XPU falls through to `SingleDeviceStrategy(device="cpu")`. Result:
accelerator is XPU but `trainer.strategy.root_device` reports CPU,
which any callback reading `root_device` will choke on.

Stage 3 dodged this entirely by passing
`strategy='deepspeed_stage_2'` (bypasses the auto picker). Using
DeepSpeed on the 66K-param Facilitator or the 28M-trainable PenCL
would be overkill.

**Fix (`22fbbcf`):** pass
`SingleDeviceStrategy(device=torch.device('xpu'))` explicitly on the
single-XPU path in both Stage 1 and Stage 2 `run_PL_training.py`.
Multi-XPU path (`gpu_devices > 1`) still falls through to `'auto'`
and is flagged in the Aurora Stage 1 PBS template as untested.

### Run 3 → `RecursionError: maximum recursion depth exceeded` in `_safe_barrier`

Simple copy-paste typo in [Stage1/PL_wrapper.py:38](src/biom3/Stage1/PL_wrapper.py#L38):
the guarded branch of `_safe_barrier()` called `_safe_barrier()`
instead of `dist.barrier()`. Invisible on Spark CUDA single-device
(`dist.is_initialized()` is False so the function silently no-ops);
on Aurora XPU distributed is initialized, the predicate flips True,
and the stack blows.

Called from 16 sites — one definition fix covers them all.

**Fix (`aeaaac7`):** `_safe_barrier() → dist.barrier()`.

### Run 4 → green. Aurora bring-up cleared.

Default + pfam × xpu all pass.

## Pfam-mode CI coverage

Previously no test exercised `pfam_PEN_CL` / `Pfam_DataModule` /
`pfam_PL_PEN_CL` — the production code path. Built the minimum
fixture:

- `tests/_data/stage1_inputs/sample_pfam.csv` — 10 rows (cols
  `id, pfam_label, sequence, [final]text_caption`).
- `tests/_data/stage1_inputs/sample_swiss_for_pfam.csv` — 6 rows.
  **Uses ASCII apostrophes** in `pfam_label` (`['PF00001']`); the
  existing `sample_text_seqs1.csv` uses a typographic right-single-quote
  (`’`, U+2019) that would fail `ast.literal_eval`. Default mode never
  hits `literal_eval` so the typo is latent; pfam mode would trip on it.
- `tests/_data/entrypoint_args/training/stage1_training_args_pfam_v1.txt` —
  CLI overrides for the scratch config, flipping
  `dataset_type`/`model_type` to `pfam`.
- Test parametrized over `(default, pfam) × (cuda, xpu)`;
  `prefix_paths()` also prefixes `pfam_data_path` now.

**Fixture surfaced a second pre-existing bug.** The pfam training_step
calls `print_memory_usage()` (default training_step doesn't). That
function lives in [backend/cuda.py:27](src/biom3/backend/cuda.py#L27) and used
`psutil` + `os.getpid()` without importing either. Shipped as a
separate fix commit so the cause is easy to bisect later.

Commits: `f7e6d87` (import fix), `277e066` (fixture).

Green on Spark CUDA (11 passed, 7 skipped full Stage 1+2 matrix) and
Aurora XPU (2 passed, 2 skipped on the targeted rerun).

## Job templates

Polaris and Aurora each got Stage 1 and Stage 2 PBS templates under
`jobs/<machine>/`. Stage 1 calls a new `scripts/stage1_train_multinode.sh`
(mpiexec wrapper mirroring Stage 3's); Stage 2 reuses the existing
`stage2_train_singlenode.sh` since the Facilitator is ~66K trainable
params and never benefits from multi-device.

Aurora Stage 1 defaults to `num_devices=12` with a comment flagging
that multi-tile XPU exercises an untested strategy path — the
single-XPU SingleDeviceStrategy fix from run 2 doesn't cover
`gpu_devices > 1`, which still falls to `'auto'`.

Commit: `9182c61`.

## Merge

1. `git pull --ff-only` on `addison-dev` (main checkout) picked up 15
   unpushed local commits plus origin's one-commit webapp fix
   (`ef062e9`). Zero overlap with Stage 1/2 files.
2. Rebased the feature branch in the worktree onto the updated
   `addison-dev` tip (`d600ef1`) — all 8 commits replayed cleanly.
3. Force-pushed the rebased feature branch; ran a Stage 1+2 regression
   on Spark (3 passed, 3 skipped).
4. Fast-forward merge: `git merge --ff-only addison-stage1-stage2-training`
   → `addison-dev` at `9182c61`.
5. Push (`ef062e9..9182c61`, 23 commits including the earlier unpushed
   local work).
6. Cleanup: removed the worktree, deleted the local branch. Origin
   branch kept for now — user can delete when convenient.

## Final branch state

8 atomic commits on top of `addison-dev`, bisect-friendly:

| Commit | Summary |
|--------|---------|
| `c758ed0` | feat(stage1,stage2): add PenCL and Facilitator training entrypoints |
| `cecb421` | fix(stage2): route Facilitator_Dataset device through backend.device |
| `0e28e05` | fix(stage1): guard pytorch_lightning imports behind XPU backend check |
| `aeaeb6e` | fix(stage1,stage2): pin XPU single-device strategy explicitly |
| `aeaaac7` | fix(stage1): stop _safe_barrier from recursing into itself |
| `f7e6d87` | fix(backend): add missing psutil and os imports in cuda.print_memory_usage |
| `277e066` | test(stage1): add pfam-mode smoke fixture |
| `9182c61` | feat(jobs): add Polaris + Aurora PBS templates for Stage 1/2 training |

(Pre-rebase hashes: `b5eef97, 6b6b818, 27405fe, 22fbbcf, 00417d3, e174b15, 99b65a5, 6cb49c4`.)

## Follow-ups

- **Multi-tile XPU on Aurora.** Stage 1's `gpu_devices > 1` path uses
  `strategy='auto'`. Lightning's auto picker falls through to
  `SingleDeviceStrategy(device='cpu')` for XPU — same class of bug as
  run 2, just guarded by a different condition. First real Aurora
  pretraining run at 12 tiles will surface whatever happens next;
  the current single-XPU fix is a template to extend.
- **Typographic apostrophe in `sample_text_seqs1.csv`.** Not touched;
  the existing default-mode test doesn't hit `ast.literal_eval`. Worth
  cleaning up whenever someone's in that file for other reasons.
- **`origin/addison-stage1-stage2-training`** still exists on GitHub.
  Delete with `git push origin --delete addison-stage1-stage2-training`
  whenever convenient.

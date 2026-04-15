# Session: Fix Stage 3 finetune resume when pretrained_weights is also set

**Date:** 2026-04-14
**Branch:** fix-ft-resume-from-checkpoint

## Problem

Resuming a Stage 3 finetuning job from a Lightning checkpoint was silently
broken when `--pretrained_weights` was also supplied on the command line
(the natural scenario when re-running the same finetune configuration
with a `--resume_from_checkpoint` pointer added).

Two coupled bugs in [src/biom3/Stage3/run_PL_training.py](../../src/biom3/Stage3/run_PL_training.py):

1. **`train_model` dropped `ckpt_path` in the finetune branch.** The
   `if args.finetune:` block around line 1384 called
   `trainer.fit(PL_model, data_module)` with no `ckpt_path`, so
   `--resume_from_checkpoint` was silently ignored for any finetune run.
2. **`run_stage3_pretraining` clobbered the resume with
   `pretrained_weights`.** The finetune setup block unconditionally
   called `load_pretrained_weights(...)` when the flag was non-null,
   overwriting the weights that Lightning would (or should) have
   restored from the checkpoint.

Combined effect: a user intending to resume finetuning from a checkpoint
would instead restart from the pretrained weights, losing the optimizer
state and all training progress, with no error or warning.

## Fix

In [src/biom3/Stage3/run_PL_training.py](../../src/biom3/Stage3/run_PL_training.py):

- **`train_model`** — split the finetune branch. When
  `resume_from_checkpoint is None`, call `trainer.fit(PL_model, data_module)`
  as before. Otherwise call `trainer.fit(PL_model, data_module, ckpt_path=resume_from_checkpoint)`
  so Lightning restores weights and optimizer state.
- **`run_stage3_pretraining`** — in the finetune setup block, if
  `resume_from_checkpoint` is set, skip `load_pretrained_weights` (with an
  info log explaining that pretrained_weights is being ignored because
  the checkpoint will supply the weights) but **still** call
  `freeze_except_last_n_blocks_and_layers`. This is necessary because
  Lightning checkpoints don't persist `requires_grad` flags; the freeze
  logic has to run on every invocation to rebuild the trainable subset
  before the optimizer is constructed.

The stale `# TODO: handle case where resume_from_checkpoint is specified`
was removed.

**Commit:** `ef74486` — `fix: resume finetune from checkpoint when pretrained_weights is set`

## Regression test

Added an entrypoint-level test in
[tests/stage3_tests/test_stage3_run_PL_training.py](../../tests/stage3_tests/test_stage3_run_PL_training.py)
following the existing `test_resume_training` pattern (two-stage run
with `lr=0` on the resume to neutralize momentum).

### Design

- **v1a** — fresh finetune, `pretrained_weights=minimodel1_weights1.pth`,
  `lr=1e-3`, `epochs=1`, `finetune_last_n_blocks=1`, `finetune_last_n_layers=1`.
  The unfrozen subset drifts away from the pretrained file, making the
  v1a checkpoint distinguishable from `minimodel1_weights1.pth`.
- **v1b** — finetune resume, `pretrained_weights=minimodel1_weights1.pth`
  (same file — this is what triggers the bug condition),
  `resume_from_checkpoint=<v1a last.ckpt>`, `lr=0`, `epochs=1`.

### Expected behavior

| Code path | v1b final state_dict |
|-----------|----------------------|
| **Fixed** | pretrained load skipped → freeze applied → `trainer.fit(ckpt_path=v1a)` restores weights → `lr=0` → **matches v1a checkpoint** |
| **Buggy** | pretrained reloaded → freeze applied → `trainer.fit` called without `ckpt_path` → `lr=0` → **matches pretrained file, not v1a** |

The test asserts `torch.allclose(v1a_state_dict, v1b_state_dict)` for
every parameter. Because v1a's unfrozen layers drifted, this equality
is only achievable via the fix — the buggy path would leave unfrozen
layers at `minimodel1_weights1.pth`.

### Why end-to-end, not mocks

The existing test file is 100% entrypoint-style. A single entrypoint
test covers **both** fixes at once: if `trainer.fit` hadn't received
`ckpt_path`, or if `pretrained_weights` had been reloaded over the
checkpoint, the final weights would mismatch. Mocking `train_model` /
`Trainer` would have required stubbing loggers, callbacks, DeepSpeed,
and filesystem setup — fragile and out of character for this file.

### Files added

- [tests/_data/entrypoint_args/training/finetune_resume_pretrained_v1a.txt](../../tests/_data/entrypoint_args/training/finetune_resume_pretrained_v1a.txt)
- [tests/_data/entrypoint_args/training/finetune_resume_pretrained_v1b.txt](../../tests/_data/entrypoint_args/training/finetune_resume_pretrained_v1b.txt)

**Commit:** `689bd9d` — `test: finetune resume ignores pretrained_weights override`

## Verification

Not yet run on GPU hardware — the test is parametrized over
`["cuda", "xpu"]` and skips on Mac. Next step:

```bash
pytest tests/stage3_tests/test_stage3_run_PL_training.py::test_resume_finetune_ignores_pretrained_weights -v
```

As a confidence check, revert each of the two fixes one at a time and
re-run the test to confirm both failure modes are detected.

## Key takeaways

1. **`requires_grad` is not persisted in Lightning checkpoints.** When
   resuming any finetune-style run, the freeze logic must re-run before
   the optimizer is built, even though the model weights come from the
   checkpoint. Skipping the freeze call on resume would break
   finetune scope.

2. **The finetune branch of `train_model` was missing a resume path
   entirely.** This was a straight control-flow omission, not a subtle
   Lightning quirk — `trainer.fit` silently accepts `ckpt_path=None` as
   "don't resume", so the bug produced no error and no warning.

3. **For regression tests of control-flow bugs in training orchestration,
   `lr=0` + weight equality is a cleaner assertion than mocks when the
   test file is already entrypoint-style.** The existing
   `test_resume_training` was a direct template.

## Commits

| Hash | Description |
|------|-------------|
| `ef74486` | fix: resume finetune from checkpoint when pretrained_weights is set |
| `689bd9d` | test: finetune resume ignores pretrained_weights override |

## Files changed

- `src/biom3/Stage3/run_PL_training.py` — finetune branch of
  `train_model` now passes `ckpt_path`; `run_stage3_pretraining` finetune
  setup skips `load_pretrained_weights` on resume but still applies
  freeze
- `tests/stage3_tests/test_stage3_run_PL_training.py` — new
  `test_resume_finetune_ignores_pretrained_weights`
- `tests/_data/entrypoint_args/training/finetune_resume_pretrained_v1a.txt` — new argfile
- `tests/_data/entrypoint_args/training/finetune_resume_pretrained_v1b.txt` — new argfile

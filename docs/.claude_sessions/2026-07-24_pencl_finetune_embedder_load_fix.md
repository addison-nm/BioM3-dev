# Session: PenCL Finetune Embedder Silent Load Failure

**Date:** 2026-07-24
**Branch:** addison-dev

## Goal

Confirm, fix, and add regression coverage for a suspected bug in the Stage 1 PenCL
generalized finetuning path, documented in `docs/.claude_prompts/FINETUNE_EMBEDDER_REVIEW.md`
by a prior agent that lacked local weights. This session had `weights/PenCL/run1_base_pencl.ckpt`
available, so the claim could be checked empirically rather than by inspection.

## Summary

The bug is real and severe. `build_text_to_zc_embedder` loaded PenCL weights with
`strict=False` and no key substitution, so a Lightning `.ckpt` — whose keys are all
`model.`-prefixed because Stage 1's PL wrappers do `self.model = model` — matched nothing
and was silently discarded. Generalized finetuning conditioned ProteoScribe on a fixed
random projection of stock BioBERT, then pushed that through a Facilitator trained for a
completely different input distribution. No error, no warning.

**Every Stage 3 generalized finetune run predating commit `28b9af7` is invalid and must be
rerun.** This covers anything driven by `configs/stage3_training/finetune_generalized_v1.json`
or `finetune_generalized_aurora.json`.

Two secondary defects from the same review were also fixed, and the duplicated checkpoint
loading logic was consolidated.

## Confirmation

Loading `weights/PenCL/run1_base_pencl.ckpt` (symlink into
`/data/data-share/BioM3-data-share/.../Run1_TrackC_step_187000_3b7e39f9.ckpt`) into
`TextToZcEmbedder` via the pre-fix path:

- All 788 checkpoint keys `model.`-prefixed; `load_state_dict(strict=False)` reported
  **216 missing / 788 unexpected**
- `text_projection.projection.weight` maxdiff vs checkpoint **3.31**; `.fc.weight` **5.28**;
  `.layer_norm.weight` **1.96** — i.e. still at random init
- `text_encoder.model.bert.encoder.layer.11.output.dense.weight` maxdiff **0.0149** — the
  one BERT layer PenCL finetunes (`bLM_n_layers_to_finetune=1`) stayed at stock BioBERT
- `bert.embeddings.word_embeddings.weight` matched exactly, which is expected: PenCL never
  trains it, so stock and checkpoint agree. This also confirmed the checkpoint was a genuine
  PenCL run and not corrupt.

After stripping the prefix: **0 text-branch keys missing**, all 578 unexpected keys from the
protein branch. So the prefix strip is both necessary and sufficient.

The Facilitator load was never broken — it used `attempt_correction=True` with
`substitutions={"model.main.": "main."}`, which handled its `model.main.*` keys correctly.
The two loads in one function used opposite policies.

## Changes

### `fix: strip PL prefix when loading PenCL weights for finetuning` (`28b9af7`)

- **`src/biom3/core/io.py`** — added `strip_pl_model_prefix`, `load_state_dict_unwrap_pl`,
  and `_resolve_lightning_ckpt_in_dir`. Handles raw files, Lightning `.ckpt`, checkpoint
  directories, and DeepSpeed ZeRO directories. DeepSpeed import is lazy so `core.io` stays
  light. Existing functions untouched — purely additive.
- **`src/biom3/Stage3/finetune_embedder.py`** — routes both loads through the shared loader.
  `strict=False` is unavoidable (a PenCL checkpoint carries the protein branch this module
  deliberately does not instantiate), so `_load_frozen_weights` scopes strictness instead:
  a missing *parameter* under `text_encoder.`/`text_projection.` raises; missing *buffers*
  warn, which keeps it robust across transformers versions that add or drop `position_ids`.
  Null weight paths now raise instead of silently no-op'ing.
- **`tests/stage3_tests/test_finetune_embedder.py`** — new, 19 tests.

### `fix: apply --float32_matmul_precision in the finetune pipeline` (`eb92274`)

The arg was inherited from `run_PL_training` but applied inside *that* module's `main()`,
which `run_ProteoScribe_finetuning.main()` never calls — live in the CLI and configs while
having no effect. Now called at `run_ProteoScribe_finetuning.py:338`, placed to mirror
`run_PL_training.py:1909` (after `setup_file_logging` so the value lands in `run.log`,
before `clear_gpu_cache` and any matmul).

**Behaviour change:** effective default moves from torch's `"highest"` to the inherited
`'medium'`. Neither finetune config sets the key, so runs before and after are not
numerically comparable on that axis. Left implicit deliberately — see open items.

### `refactor: route Stage 3 and rl checkpoint loading through core.io` (`583c69f`)

Three implementations of the same PL-prefix-stripping load existed.

- **`src/biom3/rl/io.py`** — deleted `_load_state_dict_unwrap_pl`, `_strip_pl_prefix`,
  `_resolve_lightning_ckpt_in_dir` (−78 lines); imports the core function under the old
  private name so internal call sites are unchanged.
- **`src/biom3/Stage3/io.py`** — `_strip_pl_model_prefix`, `_load_state_dict_from_file`, and
  `_load_state_dict_from_sharded_dir` kept as thin wrappers delegating to core, so
  `prepare_model_ProteoScribe` and its call sites are untouched. This wrapper approach was
  the user's suggestion and is more conservative than the inline replacement first proposed.

Two behaviour changes fall out, both improvements:
- Stage 3 previously treated *any* directory as DeepSpeed-sharded, so a plain Lightning
  checkpoint directory (e.g. `Facilitator_MMD15.ckpt/` in `configs/weights/run0_nm_base.json`)
  would fail. The shared loader checks for the `latest` marker and otherwise resolves
  `last.ckpt`.
- `map_location` becomes `device or "cpu"`, matching the Aurora fix in
  `core.io.load_state_dict`. No functional effect — `load_state_dict` copies into the
  model's existing params either way.

## Testing

Each new test was verified to **fail against pre-fix code**, not merely pass against the fix:

- `test_finetune_embedder.py` — 7 of 19 failed pre-fix, including
  `test_pl_prefixed_checkpoint_populates_text_branch` and `test_zc_matches_reference`
- `TestFloat32MatmulPrecision` — all 3 failed pre-fix

A stub `TextEncoder` stands in for BioBERT so the embedder tests need no downloaded weights
and run under `pytest --quick`. One opt-in test (`slow`, skipped if weights absent) pins the
premise by asserting the shipped `run1_base_pencl.ckpt` really is `model.`-prefixed.

Results: `tests/stage3_tests` + `tests/stage2_tests` → **403 passed, 83 skipped** (17m19s).
Full suite run by the user → clean.

Note: an intermediate `tests/ --quick` run reported 2 `test_stage1_run_PL_training[cuda-...]`
failures. These were collision artifacts from overlapping with the user's concurrent suite
run, not regressions — the same invocation passed cleanly before and after, and nothing in
these changes touches Stage 1.

## Open items

1. **Rerun invalidated finetunes.** Audit `outputs/Stage3/finetuning_generalized/` for runs
   predating `28b9af7`. Not done here — scope was the fix, not the retraining.
2. **Pin `float32_matmul_precision` explicitly?** Currently inherits `'medium'`; Stage 1/2
   inference uses `'high'` (`configs/inference/_base_runtime.json`). Since
   `torch.set_float32_matmul_precision` is process-global, the finetune pipeline cannot match
   both its own training lineage and the embedder's inference lineage. Left as the inherited
   default rather than decided unilaterally, because it changes numerics for every run.
3. **`docs/.claude_prompts/FINETUNE_EMBEDDER_REVIEW.md` is uncommitted.** It is the review doc
   that prompted this work; left untracked pending a call on whether `.claude_prompts/`
   belongs in version control.

`docs/APPTAINER.md` and `docs/rama_ft_corpus_construction.md` were already modified/untracked
before this session and were deliberately left out of these commits.

## Reverting

State immediately before this session:

```bash
git checkout 568e403
```

To drop only this session's commits while keeping the branch:

```bash
git revert 583c69f eb92274 28b9af7
```

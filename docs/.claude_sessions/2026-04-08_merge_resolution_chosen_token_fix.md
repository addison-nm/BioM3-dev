# 2026-04-08 — Merge resolution, chosen-token fix, and results-dict refactor

## Summary

Resolved a merge between `addison-main` (general "gauge" animation style + new
token colors) and `addison-spark` (PAD probability gauge + dynamics plot), then
fixed several bugs uncovered during resolution and refactored
`batch_stage3_generate_sequences` to return an extensible results dict.

## Context

`addison-main` and `addison-spark` had developed two orthogonal but
neighbouring features in `animation_tools.py`:

- `addison-main` (commit `32dc916`) added a generic `prob_style="gauge"` mode
  that draws a max-probability fill bar inside AA cells, plus distinct
  START/END/PAD colors.
- `addison-spark` (commit `8c8e23c`) added a PAD-specific two-tone overlay
  whose fill height is `P(PAD)`, plus the new `biom3.viz.dynamics` module.

The merge had been left in an `accept-both` state which compiled but contained
three latent issues:

1. `show_confidence` was redefined with `is_pad`, causing the AA gauge to fire
   on PAD cells using the wrong probability source.
2. A stray duplicate `draw.rectangle(..., fill=bg)` overdrew the AA gauge
   immediately after it was painted, silently disabling the gauge feature.
3. `_SPECIAL_COLOR` was referenced by the PAD overlay but had been *deleted*
   in `32dc916` (replaced with per-token colors). Would have crashed on the
   first PAD cell with stored probabilities.

## Changes

### 1. Unified gauge rendering (`animation_tools.py`)

The AA gauge and PAD overlay branches are now mutually exclusive paths gated
on `is_aa` / `is_pad`, with no `show_confidence` flag and no duplicate draws.
Every cell with stored probabilities renders as a bottom-up fill meter:

- **AA cells**: faint AA-color background (`_alpha_blend` against `_BG_COLOR`),
  full AA-color fill from the bottom, height proportional to the probability
  the model assigned to the *chosen* token at that step.
- **PAD cells**: grey background (`_MASK_COLOR`), pure black fill
  (`_PAD_COLOR = (0, 0, 0)`), height proportional to `step_probs[j][_PAD_IDX]`.
  The existing text branch already drew the central dot.
- **Other tokens (MASK / START / END) or no probs**: plain `_cell_color()`
  flat fill.

The `_confidence_modulate()` helper became unreachable and was removed.

### 2. Chosen-token vs argmax bug fix

Both the AA gauge and the dynamics plot were using `step_probs[j].max()` /
`probs.max(axis=-1)` — the model's confidence in its top-1 prediction at each
step. For sampled (non-argmax) tokens this diverges from the probability of
the token actually chosen at that position.

**`animation_tools.py`**: AA gauge fill now uses
`step_probs[j][token_indices[j]]`. The PAD branch was already correct because
`is_pad` implies `token_indices[j] == _PAD_IDX`.

**`viz/dynamics.py`**: heatmap and aggregate line plot now compute
`np.take_along_axis(probs, last_frame[None, :, None], axis=-1)` so every
(step, position) cell shows the probability the model assigned to the
eventually-chosen token. Variable renamed `max_probs` → `chosen_probs`
throughout. Title and y-label switch between `P(chosen-token)` and `P(top-1)`
depending on whether final-frame information is available, so a fallback plot
doesn't lie about what it's displaying.

### 3. `final_frame` storage in `.npz`

`plot_probability_dynamics_from_file()` previously had no way to compute
chosen-token probabilities because the `.npz` only stored `probs` and
`tokens`. Now:

- `run_ProteoScribe_sample.py` captures the final-step token indices into a
  new `stored_final_frames` dict alongside `stored_probs` (one
  `np.ndarray[seq_len]` per pair — much smaller than per-step frames).
- `np.savez_compressed` writes a `final_frame` key.
- `plot_probability_dynamics()` gains an optional `final_frame` kwarg as a
  lightweight alternative to per-step `frames`. Sufficient for chosen-token
  indexing and `hide_pad`. `blank_unmasked` still requires per-step `frames`
  because it needs unmasking timing.
- `plot_probability_dynamics_from_file()` loads `final_frame` from the `.npz`
  when present and forwards it. Backward-compatible: older `.npz` files
  without the key fall back to top-1.

### 4. `batch_stage3_generate_sequences` results dict

The function's return signature was growing each time a new auxiliary output
was added (most recently to a 5-tuple). Refactored to return
`(design_sequence_dict, results)` where `results` is a dict with keys
`animation_frames`, `tokens`, `stored_probs`, `stored_final_frames`. Future
additions just go in the dict — no more growing tuple, no more call-site
churn.

The single in-tree call site unpacks `results` into the same locals, so the
rest of the script needs no changes. Audited all four ecosystem repos for
external callers — none exist.

The function's docstring was rewritten to document all six parameters, the
batching strategy, and the new return shape.

### 5. Test file fix

`tests/stage3_tests/test_animation_pad_gauge.py` imported `_SPECIAL_COLOR`
which was removed in `32dc916`. Without the merge fix the entire test class
failed at module load. Swapped to `_PAD_COLOR` and updated a stale docstring.
The tests themselves only assert `isinstance(img, Image.Image)`, so no logic
changes were needed.

## Files changed

- `src/biom3/Stage3/animation_tools.py` — merge resolution, unified gauge,
  chosen-token fix, removed `_confidence_modulate`
- `src/biom3/Stage3/run_ProteoScribe_sample.py` — `stored_final_frames`
  capture, `final_frame` in `.npz`, results-dict refactor, docstring rewrite
- `src/biom3/viz/dynamics.py` — chosen-token heatmap, `final_frame` kwarg,
  `_from_file` loading, backward-compatible fallbacks
- `tests/stage3_tests/test_animation_pad_gauge.py` — `_SPECIAL_COLOR` →
  `_PAD_COLOR` import fix

## Pre-session state

```bash
git checkout addison-main
git merge addison-spark   # left in accept-both state
```

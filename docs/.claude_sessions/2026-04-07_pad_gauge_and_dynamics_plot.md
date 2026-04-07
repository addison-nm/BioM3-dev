# PAD Probability Gauge and Probability Dynamics Plot

**Date:** 2026-04-07
**Branch:** `addison-spark`
**Pre-session state:** `git checkout 9ea7f22`

## Summary

Two additions to the animation / visualization system:

1. **PAD probability gauge** — `_render_frame()` in `animation_tools.py` now renders `<PAD>` cells as a two-tone fill gauge (grey background with dark fill from the bottom, height proportional to PAD probability) instead of a flat dark cell, so the model's confidence in PAD is visible at a glance.

2. **Probability dynamics plot** — new `biom3.viz.dynamics` module with `plot_probability_dynamics()` and `plot_probability_dynamics_from_file()`. Produces a two-panel matplotlib figure: a heatmap of top-1 predicted probability (positions × steps) and an aggregate confidence line plot with min–max envelope. Options:
   - `hide_pad=True` — exclude positions that resolve to `<PAD>`
   - `blank_unmasked=True` — show white for (step, position) cells after a position has been unmasked

## Files changed

### `src/biom3/Stage3/animation_tools.py`
- Added `_PAD_IDX = 23` constant
- Added `is_pad` flag per position in the render loop
- After drawing the standard cell background, PAD positions with available `step_probs` now draw a two-tone gauge: `_MASK_COLOR` base → `_SPECIAL_COLOR` fill from bottom proportional to `step_probs[j][_PAD_IDX]`

### `src/biom3/viz/dynamics.py` *(new)*
- `plot_probability_dynamics(probs, tokens, frames, ...)` — heatmap + aggregate line plot
- `plot_probability_dynamics_from_file(npz_path, ...)` — convenience wrapper that loads the `.npz` files saved by `run_ProteoScribe_sample` (keys: `probs`, `tokens`)
- `hide_pad` option filters out PAD-terminal positions
- `blank_unmasked` option masks heatmap cells after unmasking via `np.ma.masked_array`

### `src/biom3/viz/__init__.py`
- Exports `plot_probability_dynamics` and `plot_probability_dynamics_from_file`

### `tests/stage3_tests/test_animation_pad_gauge.py` *(new)*
- `TestPadProbabilityGauge`: two-tone high/low prob, flat without probs, AA regression, full GIF integration

### `tests/viz_tests/test_dynamics.py` *(new)*
- `TestPlotProbabilityDynamics`: returns figure, with frames, save to file, custom figsize, hide_pad, blank_unmasked, combined options, no-frames fallback
- `TestPlotProbabilityDynamicsFromFile`: load-and-plot, load-and-save

## Design notes

- The PAD gauge reuses existing colour constants (`_MASK_COLOR` as grey background, `_SPECIAL_COLOR` as dark fill) so it is visually consistent with the rest of the animation palette.
- `blank_unmasked` uses `np.ma.masked_array` with `cmap.set_bad(color="white")` so matplotlib renders masked cells as white without extra drawing logic.
- `hide_pad` and `blank_unmasked` both require `frames` to determine token identity and unmasking timing; they silently no-op if `frames` is not provided.
- The `_from_file` wrapper matches the `.npz` format written at `run_ProteoScribe_sample.py:498–499`.

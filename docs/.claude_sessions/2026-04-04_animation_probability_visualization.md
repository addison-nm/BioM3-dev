# Animation Probability Visualization and Metric Annotation System

**Date:** 2026-04-04
**Branch:** `addison-main`
**Continues from:** `2026-04-04_animation_probabilities_and_unmasking.md` (which left probability visualization as a next step)

## Summary

Wired stored per-step probabilities into the GIF animation system, then generalized the annotation framework so arbitrary per-position metrics can be rendered as colored boxes above or below each residue cell.

Three visualization modes for the full probability distribution (`--animation_style`):

1. **`brightness`** (default) — cell color brightness scales with model confidence
2. **`colorbar`** — compact stacked amino-acid bars above each cell (heights proportional to probability, no letters)
3. **`logo`** — sequence-logo style stacked bars with letters above each cell

Plus a general-purpose **metric annotation system** (`MetricAnnotation` dataclass) for scalar per-position values:
- Dynamic metrics (`[steps, seq_len]`) that change each denoising step
- Static metrics (`[seq_len]`) that are constant across frames (e.g., conservation scores)
- Configurable position (`"above"` / `"below"` the residue cell), height, and colormap
- Multiple metrics stack independently; layout is computed automatically

## Files changed

### `src/biom3/Stage3/animation_tools.py`

- Added `MetricAnnotation` dataclass with `name`, `values`, `colormap`, `height`, `position`, `static` fields and a `value_at(step, pos)` accessor
- Added built-in colormaps: `red_yellow_green()`, `blue_white_red()`
- Added `confidence_metric(probs)` convenience factory
- Extended `_render_frame()` layout to compute `above_total` / `below_total` heights from metrics + logo, stack them in the grid loop
- Extended `generate_sequence_animation()` with `metrics` parameter
- Added `_draw_logo()` helper with `draw_letters` flag (shared by colorbar and logo modes)
- Added `_confidence_modulate()` for brightness mode

### `src/biom3/Stage3/run_ProteoScribe_sample.py`

- Added `--animation_style` CLI arg (choices: `brightness`, `colorbar`, `logo`)
- Added `--animation_metrics` CLI arg (currently supports `confidence`; extensible)
- GIF generation block builds `MetricAnnotation` list from requested metrics and passes to animation
- Warning logged if metric/style features requested without `--store_probabilities`

### `docs/sequence_generation_strategies.md`

- Documented `--animation_style` and `--animation_metrics` flags with table and usage example

## Public API additions in `animation_tools.py`

```python
# Dataclass
MetricAnnotation(name, values, colormap, height=6, position="above", static=False)

# Built-in colormaps
red_yellow_green(value)   # 0=red, 0.5=yellow, 1=green
blue_white_red(value)     # 0=blue, 0.5=white, 1=red

# Factory
confidence_metric(probs, position="above", height=6)
```

## Adding a new metric (recipe)

```python
from biom3.Stage3.animation_tools import MetricAnnotation, blue_white_red

conservation = MetricAnnotation(
    name="conservation",
    values=conservation_scores,        # [seq_len] array, floats in [0, 1]
    colormap=blue_white_red,
    height=6,
    position="below",
    static=True,
)

generate_sequence_animation(frames, tokens, path, metrics=[conservation])
```

To add a new CLI-derivable metric, add an `elif` branch in the metric-building loop in `run_ProteoScribe_sample.py` (~line 440).

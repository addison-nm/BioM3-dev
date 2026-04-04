# Animation Overhaul, Probability Storage, and confidence_no_pad

**Date:** 2026-04-04
**Branch:** `addison-main`
**Pre-session checkout:** `git checkout 5e988b6` (session began after the animation commit from the prior session)

## Summary

Three features added to Stage 3 ProteoScribe sampling, plus documentation:

1. **`--store_probabilities` flag** — optional per-step conditional probability storage during diffusion sampling
2. **`--unmasking_order confidence_no_pad`** — confidence-based unmasking that deprioritises positions predicted as `<PAD>`
3. **Sequence generation strategies doc** — comprehensive guide to all sampling options
4. Memory updated for the next planned feature: probability visualization in GIF animations

## Commits

- `5f34ba8` — `feat: add --store_probabilities flag for Stage 3 diffusion sampling`
- `846dc15` — `feat: add confidence_no_pad unmasking order for Stage 3 sampling`
- `d310388` — `docs: add Stage 3 sequence generation strategies guide`

## Detailed changes

### --store_probabilities (`5f34ba8`)

**Files:** `sampling_analysis.py`, `run_ProteoScribe_sample.py`, `test_batch_generate_denoised_sampled.py`

- Added `store_probabilities=False` parameter to both `batch_generate_denoised_sampled` and `batch_generate_denoised_sampled_confidence`
- When enabled, pre-allocates a `[steps, batch, seq_len, num_classes]` float32 tensor on GPU, fills at each diffusion step, single CPU transfer at the end
- Returns `np.ndarray` as third return value (or `None` when disabled)
- `run_ProteoScribe_sample.py` saves per-(prompt, replica) `.npz` files in `<output_dir>/probabilities/` with keys `probs` and `tokens`
- All existing tests updated to unpack the new third return value

### confidence_no_pad (`846dc15`)

**Files:** `sampling_analysis.py`, `run_ProteoScribe_sample.py`

- Added `skip_pad=False` parameter to `batch_generate_denoised_sampled_confidence`
- When True, positions whose argmax prediction is `<PAD>` have their confidence set to `-inf` so non-PAD masked positions are selected first
- Falls back gracefully: once only PAD-predicted positions remain masked, they are unmasked normally (no deadlock)
- `pad_token_id` is derived from the token vocabulary via `.index('<PAD>')` with a `ValueError` if missing — not hardcoded
- New CLI choice: `--unmasking_order confidence_no_pad`

### Documentation (`d310388`)

**File:** `docs/sequence_generation_strategies.md`

Covers all three unmasking orders, both token strategies, their six combinations, diagnostic options (`--store_probabilities`, animation), CLI examples, and JSON config defaults.

## Next steps (not implemented)

### Probability visualization in GIF animation

The storage infrastructure is in place. The next step is wiring probability data into `generate_sequence_animation()` in `animation_tools.py`. See the briefing document below for a starting point.

---

## Briefing: Probability Visualization in GIF Animation

### Goal

Extend the Stage 3 diffusion animation (colored amino acid grid GIF) to visually encode per-position model confidence at each denoising step.

### What exists today

- **`animation_tools.py`** — `generate_sequence_animation(frames, tokens, output_path, ...)` renders a GIF from a list of 1-D token-index arrays (one per diffusion step). Each frame draws a colored grid: amino acid cells colored by physicochemical property, masked positions as dark cells, newly-unmasked positions highlighted in yellow. Helper `_render_frame()` does the per-frame drawing.

- **`sampling_analysis.py`** — both `batch_generate_denoised_sampled` and `batch_generate_denoised_sampled_confidence` can return a `[steps, batch, seq_len, num_classes]` numpy array of conditional probabilities when `store_probabilities=True`.

- **`run_ProteoScribe_sample.py`** — the `stored_probs` dict (keyed by `(p_idx, r_idx)`, values are `[steps, seq_len, num_classes]` arrays) is available at the same call site where GIF generation happens (line ~410). The `animation_frames` dict (keyed the same way) holds the token-index arrays.

### Suggested approach

1. **Extend `generate_sequence_animation` signature** to accept an optional `probs` parameter — a `[steps, seq_len, num_classes]` numpy array matching the frames list. When `None`, render as today.

2. **Derive per-position confidence** in `_render_frame`: for each position, compute `max(probs[step, pos, :])` — the model's confidence in its top prediction. This is a float in `[0, 1]`.

3. **Visual encoding options** (pick one or make configurable):
   - **Cell brightness/saturation**: scale the existing AA color's brightness by confidence (low confidence → dim, high → vivid)
   - **Border color**: use a confidence-based colormap for the cell border (e.g., red→yellow→green)
   - **Underlay bar**: small bar below each cell whose fill reflects confidence
   - **Opacity overlay**: blend cell color with a neutral gray proportional to `(1 - confidence)`

4. **Wire in `run_ProteoScribe_sample.py`**: at the GIF generation block (~line 410), if `stored_probs` has data for a given `(p_idx, r_idx)`, pass `probs=stored_probs[(p_idx, r_idx)]` to `generate_sequence_animation`.

### Key shapes

```
frames:  list of np.ndarray, each shape [seq_len]       (int token indices)
probs:   np.ndarray, shape [steps, seq_len, num_classes] (float32, sums to ~1 over last dim)
tokens:  list of str, length num_classes                 (vocabulary)
```

### Files to modify

- `src/biom3/Stage3/animation_tools.py` — extend `generate_sequence_animation` and `_render_frame`
- `src/biom3/Stage3/run_ProteoScribe_sample.py` — pass `probs` when available

### Things to watch

- Probability arrays are only populated when `--store_probabilities` is passed. The animation should still work without them (current behavior as fallback).
- For masked positions, the probability distribution may still be informative (shows what the model *would* predict) — consider whether to visualize confidence for masked positions too, or only for the newly-unmasked one.
- The `--store_probabilities` flag and `--animate_prompts` flag are currently independent. They need to both be active for probability-augmented animations. Could add a log warning if animation is requested without probabilities.

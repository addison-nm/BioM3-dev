# Stage 3 Generation Animation

**Date:** 2026-04-03
**Branch:** addison-local

## Summary

Added a GIF animation feature to the Stage 3 ProteoScribe sampling script.
Users can now pass `--animate_prompts` to produce per-step animations of the
diffusion denoising process. The feature was already partially scaffolded in
`animation_tools.py`; this session wired it into the sampling pipeline.

## Changes

### `src/biom3/Stage3/run_ProteoScribe_sample.py`

**Three new CLI flags:**

| Flag | Default | Description |
| ---- | ------- | ----------- |
| `--animate_prompts` | *(omit to disable)* | Prompt indices to animate: integers, `all`, or `none` |
| `--animate_replicas` | `1` | `i` → `range(0, i)` replicas; `all`; `none` |
| `--animation_dir` | `<output_dir>/animations/` | GIF output directory |

**Four new helper functions** (parse + resolve for each axis):
- `parse_animate_prompts` / `resolve_animate_prompts` — parse raw CLI values; raise `ValueError` with valid range on bad indices; also catches negative indices
- `parse_animate_replicas` / `resolve_animate_replicas` — parse raw CLI values; clamp with a logged warning if `N > num_replicas` (rather than raising)

**`batch_stage3_generate_sequences`** gains two new parameters (`animate_prompts: set`, `animate_replicas: set`) and returns a tuple `(design_sequence_dict, animation_frames)`. Inside the batch loop, for each (prompt, replica) pair that falls in both sets, the full `mask_realization_list` (one array per diffusion step) is decoded to strings and stored in `animation_frames[(p_idx, r_idx)]`.

**`main`** resolves animation targets before sampling, unpacks the new return value, and calls `Stage3_ani_tools.generate_text_animation` once per (prompt, replica) pair, writing `prompt_<P>_replica_<R>.gif`.

### `demos/animate_generation.sh` *(new)*

Demonstrates four usage patterns:
1. Single prompt, default replica count (replica 0)
2. Specific prompt subset (`0 1 2`)
3. All prompts, first 3 replicas
4. All prompts, all replicas, custom animation directory

### `demos/README.md`

Added entry for `animate_generation.sh` with output table and runtime note.

### `README.md`

- Added `--animate_prompts`, `--animate_replicas`, `--animation_dir` to the Stage 3 arguments table
- Added animation example with pointer to the new docs page

### `docs/sequence_generation_animation.md` *(new)*

Full feature reference: how frames map to diffusion steps, flag semantics, output naming convention, worked examples, and a performance note.

## Design decisions

- **No animation by default.** Omitting `--animate_prompts` leaves `animate_prompts=None`, which short-circuits all animation logic. Zero runtime cost on existing workflows.
- **Clamping vs raising for replicas.** `--animate_replicas N > num_replicas` clamps silently with a warning. This is more forgiving than raising because replica count is an approximate "how many" rather than a precise index.
- **Raising for bad prompt indices.** Specific out-of-range indices are always an error (unlike `all` which self-limits). The error message includes the valid range.
- **GIF frames include `<START>`/`<END>` tokens.** Unlike the final saved sequence (which strips them), animation frames preserve the raw token output so the boundaries are visible during denoising.
- **Frame collection is cheap.** The `mask_realization_list` is already produced by `batch_generate_denoised_sampled`; we just decode the targeted subset. GIF encoding runs after sampling completes.

## State before this session

```bash
git checkout 83297a3
```

## Not implemented / future work

- Frame rate / duration control for GIFs (currently hardcoded to 0.2 s/frame in `animation_tools.generate_text_animation`)
- Selecting non-contiguous replica indices (currently `--animate_replicas` only supports `range(0, N)` or `all`)
- Colorised output (e.g. highlighting newly-placed residues per frame)

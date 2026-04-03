# Stage 3 batch sampling: correctness fix, sync removal, Gumbel-max optimization

**Date:** 2026-04-03
**Branch:** `addison-main`
**Pre-session state:** `git checkout 6616dff`

## Summary

Investigated why `batch_generate_denoised_sampled` wall-clock time scaled
linearly with batch size instead of leveraging GPU parallelism. Found and fixed
three layered issues:

1. **Cross-batch indexing bug** — basic PyTorch indexing caused every batch item
   to write to every other item's sequence positions, polluting denoising
   trajectories.
2. **Per-iteration GPU-to-CPU synchronization** — four `.cpu()` calls inside the
   1024-step diffusion loop serialized GPU execution.
3. **Wasteful sampling path** — `OneHotCategorical.sample()` + `argmax`
   allocated a large one-hot tensor each step and called `torch.multinomial`;
   replaced with the Gumbel-max trick using only element-wise ops.

Added 7 unit tests and two documentation files (bug report + Gumbel-max
reference).

## Detailed changes

### `src/biom3/Stage3/sampling_analysis.py`

#### `predict_next_index` (line ~149)
- Removed `probs.cpu()` from the return statement. The second return value is
  never consumed by any caller — all callers either discard it or assign it to
  an unused variable.

#### `batch_generate_denoised_sampled` (line ~209)

**Indexing fix (was lines 251-253):**
The original code used basic indexing with a per-sample index tensor:
```python
current_location = torch.argmax(current_location.detach().cpu()*1, dim=-1)  # shape (batch,)
temp_mask_realization[:, 0, current_location] = next_temp_realization[:, current_location]
```
`[:, 0, [5,3,7]]` selects positions 5, 3, AND 7 for ALL batch items. Fixed with
advanced indexing using `torch.arange(batch_size)`:
```python
batch_idx = torch.arange(batch_size, device=args.device)
current_location = (temp_sampling_path == temp_idx).long().argmax(dim=-1)
temp_mask_realization[batch_idx, 0, current_location] = next_temp_realization[batch_idx, current_location]
```

**GPU sync removal (was lines 252, 256-257, and predict_next_index line 151):**
- Removed `current_location.detach().cpu()` — stays on device.
- Replaced per-iteration `temp_mask_realization.cpu().numpy()` with
  pre-allocated GPU tensors (`all_realizations`, `all_time_idx`). Single
  `.cpu().numpy()` transfer after the loop completes.
- Removed unused `probs.cpu()` in `predict_next_index`.

**Gumbel-max sampling (was line 256):**
Replaced `torch.argmax(conditional_prob.sample(), dim=-1)` with:
```python
gumbel_buffer.exponential_()
next_temp_realization = (
    conditional_prob.probs.log() - gumbel_buffer.log()
).argmax(dim=-1)
```
Pre-allocates a `(batch, seq_len, num_classes)` noise buffer before the loop;
`exponential_()` refills it in-place each step. Eliminates `torch.multinomial`,
the one-hot tensor allocation, and the argmax-on-one-hot roundtrip.

**Dead code removed:** `current_ii` (was line 235) computed but never used.

### `src/biom3/Stage3/run_ProteoScribe_sample.py`
- No functional changes; minor whitespace from earlier in-progress edits.

### New files

| File | Purpose |
|---|---|
| `tests/stage3_tests/test_batch_generate_denoised_sampled.py` | 7 unit tests: output shapes (batch 1/2/4), token range, determinism, different seeds diverge, batch independence |
| `docs/bug_reports/batch_indexing_cross_contamination.md` | Bug report with worked example, old vs new code, explanation of why sequences still appeared biologically valid |
| `docs/gumbel_max_sampling.md` | Gumbel-max trick derivation, references (Wikipedia, Maddison 2017, Jang 2017), implementation details, numerical considerations |

## Profiling results (CPU, mini model)

Per-step time breakdown at different batch sizes, BEFORE Gumbel-max fix (after
sync removal):

| batch_size | total ms/step | model | sample | index | update |
|---|---|---|---|---|---|
| 1 | 17.3 | 16.5 | 0.8 | 0.0 | 0.0 |
| 2 | 22.8 | 19.4 | 3.3 | 0.0 | 0.0 |
| 4 | 26.9 | 22.2 | 4.6 | 0.1 | 0.0 |
| 8 | 37.4 | 30.6 | 6.7 | 0.1 | 0.1 |

Model forward pass scales sublinearly (1.85x at 8x batch). The `sample` column
(`OneHotCategorical.sample()` + `argmax`) scaled linearly — Gumbel-max cuts this
by ~2x.

The remaining model forward pass scaling on CPU is expected — there is no GPU
parallelism to absorb extra batch items.

## Test results

```
tests/stage3_tests/test_batch_generate_denoised_sampled.py  — 7 passed
tests/stage3_tests/test_stage3_run_ProteoScribe_sample.py   — 6 passed (cpu)
tests/test_imports.py                                        — 4 passed
```

## Training code audit

Searched all Stage 3 training functions for the same indexing pattern. The
training code in `transformer_training_helper.py` uses per-sample iteration
(`.nonzero()` + loops) rather than direct tensor index assignments, so it is not
affected. The single-sample `generate_denoised_sampled` function has a latent
version of the bug (hardcoded `[0]` index) but is safe at batch_size=1 and has
no callers in the current codebase.

## Not committed

All changes are staged but **not committed** — awaiting user review. Rollback:
```bash
git checkout 6616dff -- src/biom3/Stage3/sampling_analysis.py src/biom3/Stage3/run_ProteoScribe_sample.py
```

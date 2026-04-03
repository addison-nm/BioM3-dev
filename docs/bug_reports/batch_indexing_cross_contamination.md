# Bug: Cross-Batch Indexing Contamination in Stage 3 Sampling

## Summary

`batch_generate_denoised_sampled` in
[src/biom3/Stage3/sampling_analysis.py](../../src/biom3/Stage3/sampling_analysis.py)
used basic (broadcast) indexing to update per-sample token positions during the
diffusion loop. This caused every batch item to write to positions belonging to
*all other* batch items, polluting the denoising trajectory with premature token
predictions. The bug only manifests when `batch_size > 1`.

Fixed in the same commit that added unit tests in
`tests/stage3_tests/test_batch_generate_denoised_sampled.py`.

---

## Root Cause

Each batch item follows its own random sampling path — a permutation of sequence
positions that determines the order in which tokens are unmasked. At each
diffusion step, the code identifies which position to fill for each item
(`current_location`, shape `(batch_size,)`) and writes the model's prediction
there.

The original code used PyTorch basic indexing:

```python
temp_mask_realization[:, 0, current_location] = next_temp_realization[:, current_location]
```

When `current_location` is a 1-D tensor (e.g. `[2, 4, 1]`), PyTorch interprets
`[:, 0, [2, 4, 1]]` as: select positions 2, 4, **and** 1 for **every row** in
the batch dimension. The intended behavior — item 0 writes only to position 2,
item 1 only to position 4, item 2 only to position 1 — requires advanced
indexing with an explicit batch index.

---

## Worked Example

Consider `batch_size=3`, `diffusion_steps=5`, at step `ii=0`:

```
sampling_path[0] = [2, 0, 4, 1, 3]   -> at time 0, item 0 should fill position 2
sampling_path[1] = [4, 3, 1, 0, 2]   -> at time 0, item 1 should fill position 4
sampling_path[2] = [1, 2, 0, 3, 4]   -> at time 0, item 2 should fill position 1
```

So `current_location = [2, 4, 1]`.

**Old behavior** (`[:, 0, [2, 4, 1]]` — basic indexing):

| Item | Should write to | Actually writes to | Extra positions |
|------|-----------------|--------------------|-----------------|
| 0    | position 2      | positions 2, 4, 1  | 4, 1 (from items 1, 2) |
| 1    | position 4      | positions 2, 4, 1  | 2, 1 (from items 0, 2) |
| 2    | position 1      | positions 2, 4, 1  | 2, 4 (from items 0, 1) |

Item 0 gets positions 4 and 1 filled prematurely using item 0's own predictions
for those positions (`next_temp_realization[0, 4]` and
`next_temp_realization[0, 1]`). These are predictions conditioned on item 0's
partially-revealed sequence, not item 1's or item 2's — so they are likely
wrong.

**New behavior** (`[batch_idx, 0, current_location]` — advanced indexing):

```python
batch_idx = torch.arange(3)   # [0, 1, 2]
```

PyTorch pairs the indices element-wise: `(0, 2), (1, 4), (2, 1)`. Each item
writes only to its own position.

---

## Why Sequences Still Appear Valid

The bug is **self-correcting** at the individual position level. The premature
writes are overwritten when each position's actual turn in the sampling path
arrives. For a 1024-position sequence:

1. At step `k`, the model conditions on `k` correctly-filled positions plus up
   to `batch_size - 1` prematurely-filled positions.
2. The premature tokens are wrong but get overwritten later in the loop.
3. The damage is indirect: between the premature write and the correct overwrite
   (on average ~512 steps later), the model sees incorrect context when
   predicting other positions.

The net effect is a slightly degraded denoising trajectory — the model makes
marginally worse predictions at intermediate steps because its context includes
tokens that shouldn't be there yet. The final sequences are biologically
plausible because:

- The model is robust to small amounts of noise in its context.
- Each position ultimately gets a prediction conditioned on the correct set of
  previously-revealed tokens at that position's own step, plus some noise from
  the premature tokens at *other* positions.
- With `batch_size=1` the bug never triggers (only one element in
  `current_location`), so single-sample generation was always correct.

The impact would be measurable statistically (e.g., slightly lower sequence
quality or reduced diversity) rather than visible as obviously broken output.

---

## Old Code vs New Code

### Old (buggy)

```python
# sampling_analysis.py, inside batch_generate_denoised_sampled loop

# Dead code — never used
current_ii = torch.full((batch_size,), ii, dtype=torch.long, device=args.device)

conditional_prob, prob = predict_next_index(...)

next_temp_realization = torch.argmax(conditional_prob.sample(), dim=-1)

# BUG: basic indexing — writes to all items' positions for every item
current_location = temp_sampling_path == temp_idx
current_location = torch.argmax(current_location.detach().cpu()*1, dim=-1)
temp_mask_realization[:, 0, current_location] = next_temp_realization[:, current_location]

# PERF: GPU->CPU sync every iteration (1024x)
mask_realization_list.append(temp_mask_realization.cpu().numpy())
time_idx_list.append(temp_idx.cpu().numpy())
```

### New (fixed)

```python
# sampling_analysis.py, inside batch_generate_denoised_sampled loop
# batch_idx = torch.arange(batch_size, device=args.device)  [pre-computed before loop]

conditional_prob, prob = predict_next_index(...)

next_temp_realization = torch.argmax(conditional_prob.sample(), dim=-1)

# FIX: advanced indexing — each item writes only to its own position
current_location = (temp_sampling_path == temp_idx).long().argmax(dim=-1)
temp_mask_realization[batch_idx, 0, current_location] = next_temp_realization[batch_idx, current_location]

# PERF: accumulate on GPU, single transfer after loop
all_realizations[ii] = temp_mask_realization
all_time_idx[ii] = temp_idx
```

---

## Additional Performance Fixes in the Same Commit

The original code also had three unnecessary GPU-to-CPU synchronization points
per diffusion step, which serialized the GPU pipeline and caused wall-clock time
to scale linearly with batch size instead of leveraging GPU parallelism:

| Location | Sync call | Fix |
|---|---|---|
| `current_location` computation | `.detach().cpu()` | Keep on GPU |
| Per-step result accumulation | `temp_mask_realization.cpu().numpy()` | Pre-allocate GPU tensor, single `.cpu()` after loop |
| Per-step time accumulation | `temp_idx.cpu().numpy()` | Same |
| `predict_next_index` return | `probs.cpu()` | Removed — return value unused by all callers |

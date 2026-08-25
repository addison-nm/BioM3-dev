# Pre-unmask / `diffusion_budget` time-conditioning offset bug

**Date:** 2026-06-17
**Status:** Fixed for standalone pre_unmask + in-painting (commit `4b13045`; see
"Implementation" below). **GDPO fixed 2026-08-25** — deferring it turned out to
be unsafe, see "GDPO follow-up" at the end.
**Affects (primary):** Stage 3 ProteoScribe sampling with `pre_unmask` and the
in-painting feature.
**Affects (secondary):** GDPO RL training, which reuses the same rollout/time
machinery. Originally out of scope here; the deferral was later found to break
GDPO outright — see the follow-up section.
**Severity:** High for standalone pre-unmask sampling (degraded output).
High for GDPO under `pre_unmask` between `4b13045` and the follow-up fix
(rollout produced single-residue sequences).

---

## TL;DR

When `diffusion_budget D < sequence_length L_total`, the model is conditioned on
a time index that **undercounts the true number of revealed positions by
`(L_total − D)`**. With the production SH3 config (`D = 128`, `L_total = 1024`)
the offset is **896** — the time embedding is wrong for the entire trajectory,
not just an edge case. The feature only behaves correctly in the degenerate case
`D == L_total`, where the offset is zero.

This was not caught because the only committed `pre_unmask` config sets
`diffusion_budget = 128`, and it was implicitly assumed `128 == sequence_length`.
It is not — every ProteoScribe model is built at `sequence_length = 1024`.

---

## Background

### How `pre_unmask` works

ProteoScribe is an order-agnostic absorbing-state discrete diffusion model
(ARDM). Normal generation starts from an all-MASK tensor of length `L_total`
(= 1024) and unmasks one position per step over `L_total` steps.

`pre_unmask` ("post-padding") caps output length cheaply: pre-fill the tail
`[D, L_total)` with PAD (treated as already-revealed) and only diffuse over the
first `D` positions, running `D` steps instead of `L_total`. Config example
([configs/grpo/pre_unmask_sh3.json](../../configs/grpo/pre_unmask_sh3.json)):

```json
{ "strategy": "last_k", "fill_with": "PAD", "diffusion_budget": 128 }
```

At load time ([run_ProteoScribe_sample.py:626-635](../../src/biom3/Stage3/run_ProteoScribe_sample.py#L626-L635),
mirrored in [gdpo.py:687-703](../../src/biom3/rl/gdpo.py#L687-L703)):
- `sequence_length` is snapshotted to the trained architectural length
  (`= diffusion_steps` from the model config = **1024**).
- `diffusion_steps` is then overwritten with the budget `D` (= 128).

### Time-index semantics during training

The ARDM training contract
([transformer_training_helper.py:88](../../src/biom3/Stage3/transformer_training_helper.py#L88),
[:257-265](../../src/biom3/Stage3/transformer_training_helper.py#L257-L265)):

```python
mask = (sample_random_path < idx)      # revealed positions
idx  ∈ [0, seq_length]                  # sampled uniformly
```

Because `sample_random_path` is a permutation of `[0, seq_length)`, exactly
`idx` positions are revealed at time `idx`. Therefore **`t == idx == the number
of revealed positions out of `L_total`**, and revealed positions are uniformly
random across the full 1024-length tensor (including PAD positions). The time
embedding is a `SinusoidalPosEmb` injected at *every* transformer block
([cond_diff_transformer_layer.py:203](../../src/biom3/Stage3/cond_diff_transformer_layer.py#L203)),
so a wrong `t` is pervasive, not cosmetic.

---

## The bug

In the sampler, `temp_idx` does **double duty**:

1. **Path-step counter** — selects which position to unmask this step:
   `current_location = (temp_sampling_path == temp_idx).argmax(dim=-1)`
   ([sampling_analysis.py:296](../../src/biom3/Stage3/sampling_analysis.py#L296)).
   Must range over `[0, D)` to match `randperm(D)`. ✅ correct.
2. **Model time conditioning** — `t = temp_idx`
   ([sampling_analysis.py:166](../../src/biom3/Stage3/sampling_analysis.py#L166)).
   Must equal the true revealed-count. ❌ wrong.

`extract_time` is passed as **zeros** at every call site, so `temp_idx` starts
at 0 and runs `0 → D-1`:
- [run_ProteoScribe_sample.py:557](../../src/biom3/Stage3/run_ProteoScribe_sample.py#L557),
  [:568](../../src/biom3/Stage3/run_ProteoScribe_sample.py#L568)
- [gdpo.py:491](../../src/biom3/rl/gdpo.py#L491) (rollout)

At step `s`, the true revealed count is `(L_total − D) + s`, but the model is
told `t = s`. **The time signal is shifted low by exactly `(L_total − D)` for
the whole trajectory.** With `D=128, L_total=1024` the model is told `t ∈ [0,128)`
when it should be told `t ∈ [896, 1024)`.

Why it stays hidden at `D == L_total`: offset is 0, and the two roles of
`temp_idx` coincide — exactly the only case that gets exercised by the default
config and tests.

### In-painting (same root cause)

The in-painting feature ([inpaint.py](../../src/biom3/Stage3/inpaint.py)) freezes
arbitrary template residues as context — these are also "pre-revealed", so a
template that freezes `M` positions has the same `t`-undercount of `M`. The
inpaint path likewise starts `t` at 0.

### GDPO (secondary — for completeness)

GDPO reuses the same rollout and shares the same root cause: `_build_grid` is
called with `L = D` ([gdpo.py:729](../../src/biom3/rl/gdpo.py#L729)) and the
corruption sampler conditions on `t = idx_n` while the PAD tail is also revealed.
Not the focus here; see the impact note below.

---

## Impact: are past results wrong?

### Standalone `pre_unmask` sampling — yes, degraded
The model receives a time signal ~896 steps too early for the entire run: it
behaves as if at the very start (almost all masked) while looking at a sequence
that is ~90% filled. Nothing cancels the offset. Output is off-distribution.
Any `run_ProteoScribe_sample.py` generation with `--pre_unmask` is affected.

### GDPO training (secondary) — mostly self-cancelling
The offset hits `π_new`, `π_old`, and `π_ref` identically, so it cancels in the
log-ratio and KL, and rewards are on real decoded sequences; the objective stays
meaningful. The only cost is that rollouts were generated off-manifold. Prior
runs are not invalidated; re-running after the fix is optional.

### Runs with `pre_unmask` disabled — unaffected
GRPO (no pre_unmask, full `L=1024`) and any default sampling are fine.

---

## Critical fact: SH3 architectural length is 1024, not 128

This is what flips the bug from "dormant edge case" to "active in production".

- `diffusion_steps = 1024` is fixed in the shared base config
  [_base_ProteoScribe_1block.json](../../configs/stage3_training/models/_base_ProteoScribe_1block.json)
  (and `_base_ProteoScribe_16blocks.json`).
- All 38 Stage 3 training `args.json` under `outputs/Stage3/**` show
  `diffusion_steps: 1024` (pretraining and every finetune).
- The inpainting session note states generation "starts from an all-MASK tensor
  of length 1024".
- The SH3 model (`ProteoScribe_SH3_epoch52`) is a finetune → `L_total = 1024`.
- The GDPO config chain resolves with no override:
  `production_gdpo.json` → `_base_grpo.json` → `stage3_ProteoScribe_sample.json`
  → `_base_ProteoScribe_1block.json` (1024). `pre_unmask_sh3.json` sets `D=128`.
  ⇒ `L_total = 1024`, `D = 128`, **offset = 896**.

(Optional belt-and-suspenders check: inspect the axial positional-embedding
shape in `weights/ProteoScribe/ProteoScribe_SH3_epoch52.ckpt/single_model.pth` —
expect a 1024-length factorization. Requires the Aurora frameworks torch env.)

---

## The fix

**Principle:** decouple the path-step counter (`[0, D)`) from the model time
index (`step + (L_total − D)`), so the model always sees the true revealed
count. Equivalent framing: treat the `(L_total − D)` pre-filled positions as
occupying the *first* `(L_total − D)` slots of the sampling path, and run the
content positions over path-slots `[L_total − D, L_total)`.

### 1. `_build_initial_mask_state` + sampler ([run_ProteoScribe_sample.py](../../src/biom3/Stage3/run_ProteoScribe_sample.py), [sampling_analysis.py](../../src/biom3/Stage3/sampling_analysis.py))
- Compute `offset = sequence_length − diffusion_steps` (0 when not pre_unmask).
- Offset the sampling-path values by `offset` so `randperm(D)` maps to
  path-values `[offset, offset + D)`.
- Pass `extract_time = full(offset)` instead of `zeros` for pre_unmask batches.
- Then `temp_idx` runs `offset → offset + D − 1` (still `D` steps); the path
  match `(temp_sampling_path == temp_idx)` resolves to columns `[0, D)` →
  writes positions `[0, D)`; the model receives `t ∈ [offset, L_total)`. ✅
- Buffer sizing in `_init_sampling_state` indexes by the loop counter `ii ∈
  [0, D)`, not `temp_idx`, so no buffer-shape change is needed.
- The confidence path ([sampling_analysis.py:352](../../src/biom3/Stage3/sampling_analysis.py#L352))
  selects positions via `is_masked` (already correct) but needs the same
  `extract_time` offset for its `t`.

### 2. In-painting ([inpaint.py](../../src/biom3/Stage3/inpaint.py))
- `offset` = number of frozen (pre-revealed) positions, generally per-row.
  Set `extract_time` per-row accordingly and offset the per-row sampling path so
  frozen positions occupy the low path-slots. (Per-row offset is more involved
  than the uniform pre_unmask case — the inpaint driver already buckets by mask
  count `D`, so within a bucket the offset `L_total − D` is uniform.)

### 3. Single source of truth
Consider computing the offset once where `sequence_length` / `diffusion_steps`
are reconciled and threading it via the cfg, rather than recomputing in each
sampler, to avoid the two-roles-of-`temp_idx` trap recurring.

### GDPO (out of scope here, handle separately)
GDPO reuses the same sampler and would inherit fix #1 for rollouts. The ELBO
grid (`_build_grid` with `L=D`) and corruption `idx` would need the matching
offset too, but since the offset cancels in the ratio/KL the numerical effect is
nil — treat this as a follow-up for consistency, not a correctness blocker.

---

## Implementation (2026-06-17, commit `4b13045`)

Landed the decoupled path-step / time-index fix:

- **`run_ProteoScribe_sample.py`** — added `_pre_revealed_offset(args, D)`
  (`sequence_length − D`, the single source of truth). `_build_initial_mask_state`
  shifts the pre_unmask sampling-path values up by `offset`, and the per-batch
  body passes `extract_time = full(offset)` (was `zeros`) to both the random-path
  and confidence samplers.
- **`inpaint.py`** — `build_sampling_path_row` offsets the masked-position path
  values by `seq_len − D`; the inpaint driver inherits the matching
  `extract_time`. Frozen positions stay at sentinel `-1`.
- **GDPO** — untouched (out of scope; offset cancels in the log-ratio/KL).
- **Tests** — `tests/stage3_tests/test_pre_unmask.py` gains a stub-model suite
  (time-index span, written-region/tail preservation, default-path invariance,
  confidence path, and a regression guard asserting the old `zeros` behaviour
  collapses every step onto column 0). `test_inpaint.py` updated for the offset
  path values.

Result: `temp_idx` runs `offset → L−1` (true revealed count), the path match
still resolves to columns `[0, D)`, and the offset-0 (default) path is
byte-for-byte unchanged. The samplers in `sampling_analysis.py` were not
modified — they consume the offset-aware `extract_time` / `sampling_path` from
their callers.

## Testing plan

- **Unit (no weights):** assert that for a pre_unmask batch, the time index fed
  to the model reaches `L_total − 1` on the final step and starts at
  `L_total − D` on the first step. Assert non-pre_unmask is unchanged (offset 0,
  `t ∈ [0, L_total)`). Assert only positions `[0, D)` are ever written and the
  tail stays at the fill token.
- **Path consistency:** with the offset path, every step still selects a unique
  content position; all `D` positions written exactly once.
- **Confidence path:** same `t` start/end assertions.
- **Regression guard:** a test that fails if `extract_time` is ever passed as
  `zeros` while `pre_unmask`/`inpaint` is active and `D < L_total`.
- Run `pytest tests/stage3_tests --quick` + `tests/stage3_tests/test_inpaint.py`
  (35 tests) to confirm no regression in the offset-0 path.
- **Empirical (with weights):** generate at a few `D` values vs `D == L_total`
  and compare sequence quality / pLDDT; confirm `D=128` output is no longer
  off-manifold. Note even after the fix, pre_unmask forces a *deterministic*
  reveal order (all PAD first) that training saw only as a random-order special
  case — sanity-check this is in-distribution.

---

## Open questions / risks

- **Per-row offsets in inpaint** add complexity; confirm the bucket-by-`D`
  batching keeps the offset uniform within a batch (it should).
- **`SinusoidalPosEmb` range:** `num_timesteps` was built at 1024; feeding
  `t` up to 1023 is in range (it already was, during training).
- **User's original question — "is this worse than just limiting sequence
  lengths?"** You can't simply shrink `sequence_length`: positional embeddings
  and axial-attention windows are baked to 1024, so tail-PAD is the only way to
  keep the architecture fixed while spending fewer steps. `pre_unmask` is the
  right mechanism; it was just missing the time offset.

---

## GDPO follow-up (2026-08-25)

Deferring GDPO was not safe. The reasoning above — "the offset hits π_new, π_old and
π_ref identically, so it cancels" — held for the *pre-fix* code. The `4b13045` fix
shifted `_build_initial_mask_state`'s sampling-path values up by `offset` but GDPO
kept passing `extract_time=zeros`, so from that commit onward the path values lived
in `[896, 1024)` while `temp_idx` ran `[0, 128)`. The two never matched,
`(temp_sampling_path == temp_idx).long().argmax(-1)` fell through to 0 on every step,
and **all 128 unmask writes landed on position 0** — the exact behaviour the new
regression guard in `test_pre_unmask.py` was written to catch, reintroduced through a
call site that test does not cover.

Measured before the fix (D=128, L_total=1024, K=4): 1 distinct position written per
row instead of 128; positions 1..127 left MASK.

No run was affected — every Aurora GDPO log predates `4b13045` — but GDPO was
unrunnable from 2026-06-17 to 2026-08-25.

### What changed

The root cause reached three call sites, all of which assumed the diffusion clock
starts at 0. A single `_time_offset(cfg3)` helper (`gdpo.py`) now feeds all three:

| site | was | now |
|---|---|---|
| `_gdpo_rollout` `extract_time` | `zeros` | `full(K, offset)` |
| `_build_shared_corruptions` model-facing `idx` | `idx_n` | `idx_n + offset` |
| `_tokenwise_k3_kl` `t_steps` (+ the multinode trainer's inlined copy) | `zeros` | `full(BG, offset)` |

`idx_grid` itself is deliberately **not** shifted: it does double duty as the count of
content positions to reveal inside the corruption sampler. Shifting it would have
swapped this bug for its mirror image — the same two-roles-of-one-counter trap that
caused the original. `_build_grid`'s docstring now says so.

The corruption/KL sites go beyond the minimum. Fixing only the rollout would leave the
model generating at `t ∈ [896, 1024)` while the ELBO evaluated it at `t ∈ [0, 128)`;
that still cancels in the ratio, but computes the gradient at conditioning the model
never generates under.

### Verification

- `tests/rl_tests/test_gdpo_pre_unmask.py` — 7 tests. The rollout guard replays the
  sampler's position-selection arithmetic rather than inspecting decoded output,
  because the untrained mini model can emit MASK as a token and is therefore an
  unreliable witness. Confirmed that 4 of the 7 fail when the offset is neutered.
- `tests/rl_tests tests/geometry_tests tests/stage3_tests` — 564 passed, 158 skipped.
  The offset-0 path is unchanged.
- Aurora, one tile: `idx=992  revealed=992/1024  masked=32/1024` — the model-facing
  time index now equals the true revealed count. Sequences 62-67 aa.
- Aurora, 12 tiles via `RolloutPool`: 24/24 unique, lengths 62-84. This mattered
  independently — the pool deep-copies `cfg3` per tile, so had those copies been taken
  before the pre_unmask block, only the master would have been correct.

# Stage 3 sampling: confidence-based unmasking order and token strategy

**Date:** 2026-04-03
**Branch:** `addison-main`
**Pre-session state:** `git checkout 83297a3`

## Summary

Added two orthogonal sampling parameters for Stage 3 (ProteoScribe) generation,
making both the position unmasking order and token selection strategy
independently configurable.

## New parameters

| Parameter | Values | Default | Controls |
|-----------|--------|---------|----------|
| `unmasking_order` | `"random"`, `"confidence"` | `"random"` | Which position to unmask next |
| `token_strategy` | `"sample"`, `"argmax"` | `"sample"` | How the token at that position is chosen |

Four valid combinations:

- `random` + `sample` — original behavior (random permutation + Gumbel-max)
- `random` + `argmax` — random order, deterministic token selection
- `confidence` + `sample` — greedy position (highest max-class prob), stochastic token
- `confidence` + `argmax` — fully deterministic greedy

Note: `confidence` + `argmax` with identical conditioning produces identical
sequences across all replicas.

## Changes

### New function: `batch_generate_denoised_sampled_confidence`
- Added to `src/biom3/Stage3/sampling_analysis.py`
- At each diffusion step: computes per-position confidence (max class probability),
  masks out already-unmasked positions, selects the most confident masked position,
  then assigns a token via argmax or Gumbel-max depending on `token_strategy`
- No `sampling_path` parameter (order is determined dynamically)

### Updated: `batch_generate_denoised_sampled`
- Now respects `args.token_strategy` — branches between Gumbel-max (`"sample"`)
  and argmax (`"argmax"`)
- Gumbel buffer only allocated when `token_strategy == "sample"`

### Wiring in `run_ProteoScribe_sample.py`
- Added `--unmasking_order` and `--token_strategy` CLI arguments
- Config merge: CLI overrides > JSON config > defaults (`"random"`, `"sample"`)
- Dispatch in `batch_stage3_generate_sequences` based on `unmasking_order`
- Both parameters logged and included in manifest

### Config files updated
- `configs/stage3_config_ProteoScribe_sample.json`
- `tests/_data/configs/test_stage3_config_v1.json`
- `tests/_data/configs/test_stage3_config_v2.json`

### Tests added
12 new tests in `tests/stage3_tests/test_batch_generate_denoised_sampled.py`:
- Confidence output shapes (parametrized over batch_size x token_strategy)
- Confidence token range validation
- Confidence + argmax determinism
- Confidence batch independence
- Random + argmax determinism and token range

All 19 tests pass (7 existing + 12 new).

## torch.compile disabled

Discovered that `torch.compile` (added in `83297a3`) fails on the DGX Spark's
NVIDIA GB10 GPU due to Triton not supporting compute capability `sm_121a`.
Commented out the `torch.compile` block with a TODO to re-enable after a
PyTorch/Triton update.

## Commits

- `198aff0` — `feat: add confidence-based unmasking order and argmax token strategy for Stage 3`

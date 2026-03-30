# Session: Parallelize Stage 3 Generation Across Prompts

**Date**: 2026-03-30

## Goal

Speed up Stage 3 (ProteoScribe) sequence generation by parallelizing across prompts, not just replicates.

## Problem

`batch_stage3_generate_sequences()` had a sequential outer loop over prompts and only batched replicates of a single prompt. With 10 prompts and 5 replicates at `batch_size_sample=32`, only 5 slots were used per batch -- wasting ~84% of GPU capacity.

## Solution

Rewrote `batch_stage3_generate_sequences()` in `run_ProteoScribe_sample.py` to:

1. Flatten all (prompt_idx, replica_idx) pairs into a single work queue
2. Pack batches from the queue using `z_t[batch_prompt_indices]` to gather different prompts' conditioning vectors
3. Pre-allocate output as `design_sequences[replica_idx][prompt_idx]` and fill via direct indexing (order-independent)
4. Assert completeness after generation

## Key Insight

`batch_generate_denoised_sampled()` in `sampling_analysis.py` already supports heterogeneous conditioning vectors across the batch dimension -- no downstream changes needed.

## Files Changed

- `src/biom3/Stage3/run_ProteoScribe_sample.py` -- rewrote `batch_stage3_generate_sequences()`

## Notes

- Seed reproducibility: results with the same seed will differ from old code due to changed `torch.randperm` call order, but statistical properties are identical
- Backward compatible: single-prompt case degenerates naturally (z_t[[0,0,...]] == old .repeat())

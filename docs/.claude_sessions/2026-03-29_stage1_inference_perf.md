# Stage 1 Inference Performance Optimization

**Date:** 2026-03-29
**Branch:** addison-spark

## Problem

Stage 1 PenCL inference time per batch scaled linearly with batch size, indicating zero GPU parallelism benefit from batching.

## Root cause

All protein sequences were padded to a fixed length of 1024 tokens in `collate_fn`, regardless of actual sequence length. ESM-2's self-attention is O(n^2), so fixed 1024 padding meant each sample consumed substantial GPU resources — enough to saturate the GPU at batch_size=1. Additional samples in a batch added proportional compute with no parallelism headroom.

Combined factors:
- Fixed 1024 padding (dominant cause of linear scaling)
- fp32 precision during inference (config `"precision": "16"` was ignored)
- `BertForMaskedLM` computing the full MLM head (768 -> 28K vocab) for all 512 text positions, when only the CLS embedding was needed
- Embeddings accumulating on GPU across all batches

## Changes

### 1. Dynamic padding (`src/biom3/Stage1/preprocess.py`)

Removed the fixed 1024 padding block in `collate_fn`. ESM's `batch_converter` already pads to the longest sequence in the batch; we now only truncate sequences that exceed 1024. For a batch of ~200-token sequences, attention compute drops ~26x.

### 2. `inference_mode` + fp16 autocast (`src/biom3/Stage1/run_PenCL_inference.py`)

- Replaced `torch.no_grad()` with `torch.inference_mode()` (stricter, avoids view-tracking overhead).
- Wrapped the forward pass in `torch.autocast` (fp16 on CUDA, bf16 on XPU). Halves memory bandwidth and compute.

### 3. Bypass MLM head (`src/biom3/Stage1/model.py`)

In `TextEncoder.forward` when `compute_logits=False`, changed from:
```python
outputs = self.model(inputs, output_hidden_states=True)
last_hidden_state = outputs.hidden_states[-1]
```
to:
```python
outputs = self.model.bert(inputs)
return outputs.last_hidden_state[:, self.target_token_idx, :]
```
This calls the underlying BERT model directly, skipping the MLM head projection. The `compute_logits=True` training path is unchanged.

### 4. Move embeddings to CPU (`src/biom3/Stage1/run_PenCL_inference.py`)

Each batch's embeddings are moved to CPU immediately via `.detach().float().cpu()`, freeing GPU memory for subsequent forward passes.

## Files modified

- `src/biom3/Stage1/preprocess.py` — collate_fn dynamic padding
- `src/biom3/Stage1/run_PenCL_inference.py` — inference_mode, autocast, CPU offload
- `src/biom3/Stage1/model.py` — TextEncoder bypass MLM head

## Testing

All Stage 1 tests pass.

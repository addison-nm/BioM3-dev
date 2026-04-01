# Bug: BERT Text Embedding Mismatch Between Repos

## Summary

Text embeddings produced by `BioM3-dev` differ from those produced by `BioM3-SH3-workflow`
for the same inputs and model checkpoint. The root cause is a padding strategy mismatch
that interacts with a missing `attention_mask` in the BERT forward pass.

---

## Two Interacting Issues

### 1. `attention_mask` is never passed to BERT

In `TextEncoder.forward()` ([src/biom3/Stage1/model.py](../src/biom3/Stage1/model.py)):

```python
outputs = self.model(inputs, output_hidden_states=True)  # attention_mask absent
```

Without an `attention_mask`, BERT treats every token — including `[PAD]` tokens — as real
input and attends to them. This is true in both repos and was consistent during training.

### 2. Padding strategy differs between the two repos

The tokenizer call in `collate_fn`
([src/biom3/Stage1/preprocess.py](../src/biom3/Stage1/preprocess.py)):

| Repo | `padding=` argument | Effect |
|---|---|---|
| `BioM3-SH3-workflow` (training) | `'max_length'` | Every text padded to `text_max_length` (512) |
| `BioM3-dev` (inference) | `True` (dynamic) | Texts padded to longest sequence in the batch |

Because no `attention_mask` is passed, the number of `[PAD]` tokens BERT attends to
varies between the two settings. The `[CLS]` embedding shifts as a result.

The model was trained with `padding='max_length'`, so it implicitly learned to expect
512-length inputs. Running inference with dynamic padding presents a different input
distribution, producing different — and arguably worse — embeddings.

---

## Affected code

- `src/biom3/Stage1/preprocess.py` — `collate_fn`, line with `padding=`
- `src/biom3/Stage1/model.py` — `TextEncoder.forward()`, the `self.model(...)` call

---

## Fix Options

### Quick fix (match training distribution)

Change `collate_fn` in `preprocess.py`:

```python
# Before
padding=True,

# After
padding="max_length",
```

This makes inference inputs identical in shape to training inputs, restoring consistent
embeddings without any other changes.

### Proper fix (thread `attention_mask` through the pipeline)

This removes the dependency on a fixed padding length and is the standard way to use BERT.

1. **`collate_fn`** — return the attention mask alongside `input_ids`:
   ```python
   return text_inputs["input_ids"], text_inputs["attention_mask"], batch_tokens
   ```

2. **`PEN_CL.forward()`** — accept and forward `attn_mask`:
   ```python
   z_t = self.text_encoder(x_t, attn_mask=attn_mask, compute_logits=False)
   ```

3. **`TextEncoder.forward()`** — pass it to BERT:
   ```python
   outputs = self.model(inputs, attention_mask=attn_mask, output_hidden_states=True)
   ```

With the proper fix in place, `padding=True` (dynamic) can be used again, which reduces
memory and compute for short-text batches.

> **Note:** The proper fix changes the effective input to BERT relative to how the model
> was trained (which never used an attention mask). This may shift embeddings slightly
> even for `BioM3-SH3-workflow`-style inputs. If strict checkpoint compatibility is
> required, use the quick fix for now and apply the proper fix at the next retraining.
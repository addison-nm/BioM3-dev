# Stage 3 ProteoScribe — Training Loss

This note describes the loss function used by `biom3_train_stage3` (the conditional
diffusion model `ProteoScribe`). It answers a specific question — *do we sum the
per-position cross-entropy values over the masked tokens in a sequence?* — and
documents the full ELBO objective so the answer can be put in context.

**Short answer:** Yes. The per-position log-probabilities are summed over the
**masked (unsampled) positions only** in each sequence, then weighted by
`1 / (L − t + 1)`, then mean-reduced across the batch.
The summation is done in
[`log_prob_of_unsampled_locations`](../src/biom3/Stage3/transformer_training_helper.py#L298-L323)
(`return log_prob_unsampled.sum(1)`), and revealed positions are excluded by the
mask `(real_token_masked == 0)`. We do **not** divide by the number of masked
tokens — the `1 / (L − t + 1)` factor is the ARDM ELBO weight, not a normaliser.

## Where the loss is computed

| File | Function | Lines |
|---|---|---|
| `src/biom3/Stage3/PL_wrapper.py` | `PL_ProtARDM.cond_elbo_objective` | 417–500 |
| `src/biom3/Stage3/transformer_training_helper.py` | `log_prob_of_realization` | 272–296 |
| `src/biom3/Stage3/transformer_training_helper.py` | `log_prob_of_unsampled_locations` | 298–323 |
| `src/biom3/Stage3/transformer_training_helper.py` | `weight_log_prob` | 325–348 |
| `src/biom3/Stage3/transformer_training_helper.py` | `compute_average_loss_for_batch` | 350–366 |
| `src/biom3/Stage3/transformer_training_helper.py` | `mask_realizations` | 415–454 |
| `src/biom3/Stage3/transformer_training_helper.py` | `cond_elbo_objective` (legacy, mirror) | 852–971 |

The Lightning module method `cond_elbo_objective` is the live training path; the
free function in `transformer_training_helper.py` of the same name is the older
unconditional/legacy mirror and is no longer used by the `PL_ProtARDM` training
loop.

## Notation

| Symbol | Code | Shape | Meaning |
|---|---|---|---|
| `B` | `bs` | scalar | batch size |
| `L` | `seq_length` | scalar | sequence length (e.g. 1024 for ProteoScribe-1024) |
| `V` | — | scalar | vocabulary size (e.g. 29 amino-acid tokens) |
| `x` | `real_tokens` | `[B, L]` | ground-truth token ids |
| `σ` | `sampled_random_path` | `[B, L]` | per-row random permutation of `0..L-1` |
| `t` | `idx` | `[B, 1]` | per-row timestep, sampled uniformly from `{0, 1, …, L}` |
| `M_t` | mask `(σ ≥ t)` ⇔ `(real_token_masked == 0)` | `[B, L]` | set of **masked / unsampled** positions |
| `\bar M_t` | `random_path_mask = (σ < t)` | `[B, L]` | set of **revealed** positions |
| `x̃` | `real_token_masked` | `[B, L]` | ground truth with masked positions replaced by mask-token id `0` |
| `y_c` | `y_c` | `[B, d_text]` | conditional text embedding (Stage 2 output) |
| `ℓ` | `logits` | `[B, V, L]` | model output |

The mask token id `0` is reserved (the absorbing state), so
`real_token_masked == 0` uniquely identifies masked positions for the protein
task.

## Per-step loss derivation

For one batch element `b`, `cond_elbo_objective` runs the following pipeline.

### 1. Sample a diffusion state

```python
sampled_random_path = sample_random_path(bs, seq_length)         # [B, L], permutations
idx                 = sample_random_index_for_sampling(...)      # [B, 1], uniform in {0,…,L}
random_path_mask    = (sampled_random_path < idx)                # [B, L], True = revealed
```

### 2. Mask the ground-truth tokens

`mask_realizations` sets every unrevealed position to mask-token id `0`:

```python
real_token_masked = real_tokens.clone()
real_token_masked[~random_path_mask] = 0      # 0 is the mask / absorbing-state id
```

### 3. Forward pass and per-position log-prob

```python
logits           = self(x=real_token_masked, t=idx, y_c=y_c)        # [B, V, L]
conditional_prob = OneHotCategorical(logits=logits.permute(0,2,1))  # batch of L per-position categoricals
log_prob         = conditional_prob._categorical.log_prob(real_tokens)   # [B, L]
```

`log_prob[b, i]` is `log p_θ(x_i | x̃, t, y_c)` — i.e. the log-probability the
model assigns to the ground-truth token at position `i`, given the partially
masked input.

This is *equivalent to per-position cross-entropy* against one-hot targets:
`log_prob[b, i] = -F.cross_entropy(logits[b, :, i:i+1], real_tokens[b, i:i+1], reduction='none')`.
The implementation goes through `OneHotCategorical` rather than `F.cross_entropy`
because the same `conditional_prob` object is reused for sampling and metric
computation in `performance_step`.

### 4. Sum per-position log-probs over the masked positions (the question's focus)

```python
# log_prob_of_unsampled_locations(log_prob, real_token_masked)
log_prob_unsampled = ((real_token_masked == 0) * log_prob).sum(1)    # [B]
```

This is the **sum, not the mean**, over masked tokens of one sequence. The
revealed positions contribute zero because the mask zeroes them out before the
sum.

### 5. Apply the ARDM ELBO weighting

```python
# weight_log_prob(log_prob_unsampled, idx, seq_length)
log_prob_weighted = (1.0 / (seq_length - idx.squeeze(1) + 1)) * log_prob_unsampled    # [B]
```

The factor `1 / (L − t + 1)` is the standard
[Autoregressive Diffusion Model](https://arxiv.org/abs/2110.02037) ELBO weight
(Hoogeboom et al., 2021, eq. 6). Combined with uniform sampling of `t`, the
expectation of the per-step loss equals the ELBO on log-likelihood.

Note this is **not** the same as dividing by the number of masked positions
(which would be `L − t`); it is `L − t + 1` because `t ∈ {0, …, L}` is sampled
inclusively.

### 6. Mean over the batch and negate

```python
# compute_average_loss_for_batch(log_prob_weighted)
loss = -log_prob_weighted.mean()      # scalar
```

The `.mean()` is over the batch dimension. The minus sign converts maximisation
of the log-likelihood into a minimisation objective.

## Closed form

Putting the pieces together, the optimised loss for one batch is

```
                    1    B          1
loss  =  -  -----------  ∑   ───────────────  ∑   log p_θ ( x_{b,i} | x̃_b, t_b, y_{c,b} )
                    B   b=1   L − t_b + 1    i ∈ M_{t_b}
```

with `t_b` and `σ_b` sampled independently per batch element, and `M_{t_b}` the
set of positions still masked at step `t_b` for sequence `b`.

In expectation over `(t, σ)` this is the ARDM ELBO on `log p_θ(x | y_c)`.

## Relationship to "cross-entropy loss"

The loss is a weighted, masked, **summed** per-position cross-entropy:

* per-position cross-entropy ✅ — `_categorical.log_prob` is `log_softmax(logits)[target]`, the same value `F.cross_entropy(..., reduction='none')` would compute (with opposite sign);
* masked ✅ — only the still-masked positions contribute;
* summed (not averaged) over positions ✅ — `.sum(1)` in `log_prob_of_unsampled_locations`;
* reweighted by `1/(L − t + 1)` ✅ — the ARDM ELBO factor;
* averaged over the batch ✅ — final `.mean()`.

It is **not** equivalent to calling `F.cross_entropy(logits, targets, ignore_index=…, reduction='mean')`, which would average per *unmasked* token. The ARDM weighting and the per-sequence sum are essential — replacing them with a per-token mean would change the gradient and break the ELBO interpretation.

## Implementation caveats

* **Mask detection by token id.** `log_prob_of_unsampled_locations` identifies masked positions via `real_token_masked == 0` rather than via the boolean `random_path_mask`. This is correct only because token id `0` is reserved for the mask / absorbing state in the protein vocabulary. If a protein-token id of `0` were ever introduced for a real residue, every sequence position holding that residue would silently be treated as masked. A more defensive form would be `(~random_path_mask) * log_prob`.
* **`idx` range.** `sample_random_index_for_sampling` draws `idx ∈ {0, 1, …, L}` (inclusive), matching the `1/(L − t + 1)` weighting. At `idx = L` every position is revealed and `log_prob_unsampled` is 0; at `idx = 0` no position is revealed and the sum spans all `L` positions.
* **`logits.float()` cast.** `cond_elbo_objective` casts logits to `float32` before constructing `OneHotCategorical` to avoid a `Simplex()` constraint failure under bf16/fp16 — see the inline comment at `PL_wrapper.py:473`.
* **Padding.** ProteoScribe operates on fixed-length token tensors; padding (if any) is upstream of this loss. Within the loss there is no separate padding mask — every position contributes to either the revealed or the masked set.

# The Transformer in ProteoScribe (Stage 3)

This note documents the transformer backbone used by ProteoScribe, BioM3's
Stage 3 order-agnostic discrete diffusion model, and what it does (and does
not) assume about sequence position. It is meant as orientation for anyone
reading or modifying `Stage3/cond_diff_transformer_layer.py`.

## Where the model lives

- `src/biom3/Stage3/cond_diff_transformer_layer.py` — the BioM3-side wrapper
  (`LinearAttentionTransformerEmbedding`, `get_model`, `SinusoidalPosEmb`).
- `linear_attention_transformer` (third-party, lucidrains) — the actual
  attention/transformer blocks (`LinearAttentionTransformer`, `SelfAttention`,
  `linear_attn`).

ProteoScribe uses the **bare** `LinearAttentionTransformer`, *not* the
`LinearAttentionTransformerLM` wrapper. This distinction matters (see
[Positional information](#positional-information)).

## What the diffusion model computes

At each diffusion step ProteoScribe predicts token logits for a (partially
masked) amino-acid sequence, conditioned on:

- `x`  — current token sequence (masked positions included),
- `t`  — diffusion timestep,
- `y_c` — the embedded text condition `z_c` from Stage 2.

`generation` is order-agnostic (OA-ARDM, Hoogeboom et al.): masked positions are
unmasked in a randomized order. "Order-agnostic" refers to the *generation
order*, not to any positional invariance of the network — each token still
carries its absolute index.

## Architecture, top to bottom

Defined in `LinearAttentionTransformerEmbedding`:

1. **Token embedding** — `nn.Embedding(input_dim, dim)` maps each amino-acid /
   mask token to a vector.
2. **Absolute positional embedding** — `AxialPositionalEmbedding`, added once to
   the token embeddings (`x_embed_axial = x + x_pos`). "Axial" is just a
   parameter-saving factorization of an absolute lookup table; index `i` always
   maps to the same learned vector.
3. **Timestep conditioning** — `SinusoidalPosEmb(t)` → MLP, reshaped and added
   inside each block. This encodes the diffusion timestep, *not* sequence
   position.
4. **Text conditioning** — `y_mlp(y_c)`, reshaped and added inside each block
   alongside the timestep embedding.
5. **Transformer blocks** — `n_blocks × depth` stacked `LinearAttentionTransformer`
   layers (config: `transformer_blocks`, `transformer_depth`). The input token+
   positional embedding is re-injected at the start of every block.
6. **Output head** — `LayerNorm` → `nn.Linear(dim, output_dim)` → permuted to
   `(batch, classes, length)`.

### Attention configuration

`get_model` instantiates the transformer with `causal=False` (no
autoregression), `heads=16`, `n_local_attn_heads=8`. Each `SelfAttention` layer
therefore splits into:

- **8 global heads** running `linear_attn` — linear-complexity attention that
  summarizes the whole sequence into a single `context = Σ_n k⊗v`, then queries
  it. This is a **permutation-equivariant, position-agnostic** operation: it has
  no index term, so it cannot by itself tell position 5 from position 200.
- **8 local heads** running `LocalAttention(window_size=128, causal=False)` with
  default settings — attention restricted to fixed windows bucketed by
  **absolute** sequence index.

`attend_axially`, `shift_tokens`, and linformer settings are all off.

## Positional information

This is the part most worth understanding clearly.

**Vanilla self-attention does not encode relative position.** The attention
weight between two residues depends only on their content (the q·k interaction),
never on the distance `|i−j|`. Positional information must be injected
separately. There are three plausible places it could enter, and only one is
active here:

| Mechanism | Type | Active in ProteoScribe? |
|---|---|---|
| `linear_attn` global heads | none (set operation) | yes — but carries no position |
| Rotary embedding (`apply_rotory_pos_emb`) | relative | **no** — requires a `pos_emb` arg that ProteoScribe never passes |
| `AxialPositionalEmbedding` (input) | **absolute** | yes |
| Local-attention windows | absolute bucketing | yes |

The rotary path exists in the library (`use_rotary_emb` in
`LinearAttentionTransformerLM`) and *would* make attention depend on `i−j`, but
ProteoScribe calls the bare transformer with no `pos_emb`, so
`apply_rotory_pos_emb` never runs.

### Consequence: the model is position-aware, not translation-invariant

Because the only positional signal is an **absolute** learned embedding (plus
absolute local-window bucketing), ProteoScribe is **not** shift/translation
invariant. A motif near the START token produces different internal
representations — and different unmasking logits — than the same motif at the
C-terminus.

This is the desired behavior for proteins, whose biology is position-specific:
N-terminal methionine, signal peptides near the start, terminal localization
motifs, etc. A translation-invariant model could not learn these.

If true translation equivariance were ever a goal, the minimal change would be
to drive the existing rotary path and drop the additive axial embedding — but
the absolute local-window bucketing would still leave a residual position
dependence, and the model would lose its ability to learn terminal-specific
biology.

## Quick reference: config knobs

From `add_model_args` (defaults shown; overridden by training/inference configs):

| Arg | Default | Meaning |
|---|---|---|
| `transformer_dim` | 512 | model embedding dim |
| `transformer_heads` | 16 | total attention heads |
| `transformer_local_heads` | 8 | of which, local-attention heads |
| `transformer_local_size` | 128 | local attention window size |
| `transformer_depth` | 16 | layers per block |
| `transformer_blocks` | 1 | number of blocks |
| `transformer_dropout` | 0.1 | dropout |
| `transformer_reversible` | False | reversible-net memory trick |

`max_seq_len` must be divisible by `transformer_local_size` (asserted at
construction).

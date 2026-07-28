# Stage 3 Sequence Generation Strategies

ProteoScribe generates protein sequences through an absorbing-state discrete diffusion process. Starting from a fully masked sequence, positions are unmasked one at a time over `diffusion_steps` iterations (default: 1024). At each step the model predicts a probability distribution over all token classes at every position, one position is selected for unmasking, and a token is placed there.

Two independent axes control how generation proceeds:

1. **Unmasking order** — which position to reveal next
2. **Token strategy** — how to pick the token for that position

These can be set via CLI flags or in the JSON config (`configs/inference/stage3_ProteoScribe_sample.json`). CLI flags override config values when both are present.

## Unmasking order (`--unmasking_order`)

Controls the order in which masked positions are revealed.

### `random` (default)

Each sequence in the batch gets a random permutation of positions at the start of generation. Positions are unmasked in that fixed order regardless of model confidence.

```
Step 1: unmask position 47   (random)
Step 2: unmask position 3    (random)
Step 3: unmask position 112  (random)
...
```

**When to use:** General-purpose generation. The random order means the model must make predictions without relying on a particular reveal pattern, which produces diverse outputs across seeds.

### `confidence`

At each step, the position with the highest max-class probability among all still-masked positions is unmasked first. The model effectively chooses where it is most certain.

```
Step 1: unmask position 3    (confidence 0.94)
Step 2: unmask position 112  (confidence 0.91)
Step 3: unmask position 47   (confidence 0.87)
...
```

**When to use:** When you want the model to "anchor" on its most confident predictions first, building the sequence outward from high-certainty regions. This can produce tighter, more self-consistent sequences but may reduce diversity.

### `confidence_no_pad`

Same as `confidence`, but positions whose most-likely token is `<PAD>` are deprioritised. Sequence-content positions are unmasked first; padding positions are filled only after all non-PAD predictions have been placed.

```
Step 1: unmask position 3    (confidence 0.94, token: L)
Step 2: unmask position 112  (confidence 0.91, token: A)
...                          (PAD-predicted positions deferred)
Step N: unmask position 500  (confidence 0.99, token: <PAD>)
```

**When to use:** When generating sequences shorter than `diffusion_steps`, the model predicts `<PAD>` at many positions with high confidence. Plain `confidence` mode would unmask those first since the model is very sure they're padding. `confidence_no_pad` forces the model to commit to actual residue positions first, which can improve sequence quality by letting content predictions inform each other before padding is placed.

**Fallback:** If all remaining masked positions are predicted as `<PAD>`, they are unmasked normally (no deadlock).

## Token strategy (`--token_strategy`)

Controls how the token is selected at the chosen position, given the model's predicted probability distribution.

### `sample` (default)

Stochastic sampling via the Gumbel-max trick (equivalent to drawing from the categorical distribution). See [gumbel_max_sampling.md](gumbel_max_sampling.md) for the mathematical derivation.

```python
# Equivalent to: token = Categorical(probs).sample()
# Implemented as:
gumbel_noise.exponential_()
token = (probs.log() - gumbel_noise.log()).argmax()
```

**When to use:** Default for most generation tasks. Introduces controlled randomness so different seeds produce different sequences. Multiple replicas (`--num_replicas`) with sampling will explore the distribution.

### `argmax`

Deterministic: always picks the token with the highest probability.

```python
token = probs.argmax()
```

**When to use:** When you want fully deterministic output (same config + seed = identical sequence). Useful for debugging, ablation studies, or when you specifically want the mode of the distribution. Note that `argmax` combined with `confidence` is fully deterministic regardless of seed, since both position selection and token selection are greedy.

## Strategy combinations

| Unmasking | Token | Deterministic? | Typical use |
|-----------|-------|----------------|-------------|
| `random` | `sample` | No | General-purpose generation |
| `random` | `argmax` | Per-seed | Debugging, comparing seeds |
| `confidence` | `sample` | No | Anchored generation with diversity |
| `confidence` | `argmax` | Yes | Fully greedy, reproducible baseline |
| `confidence_no_pad` | `sample` | No | Content-first generation with diversity |
| `confidence_no_pad` | `argmax` | Yes | Content-first greedy baseline |

## Diagnostic options

### `--store_probabilities`

Saves the full `[steps, seq_len, num_classes]` probability tensor for each generated sequence as a compressed `.npz` file in `<output_dir>/probabilities/`. Useful for analysing model confidence over the course of generation, but memory-intensive for long sequences.

### `--animate_prompts` / `--animate_replicas`

Generates GIF animations showing the denoising process as a colored amino acid grid. Each frame is one diffusion step; residues are colored by physicochemical property and newly unmasked positions are highlighted.

### `--animation_style`

Controls how the full probability distribution is visualised in animations (requires `--store_probabilities`):

| Value        | Effect                                                                 |
|--------------|------------------------------------------------------------------------|
| `brightness` | Scale residue cell brightness by confidence (default)                  |
| `colorbar`   | Compact stacked amino-acid bars above each cell (no letters)           |
| `logo`       | Sequence-logo style stacked bars with letters above each cell          |

### `--animation_metrics`

Adds per-position scalar metric boxes above or below each residue cell. Multiple metrics can be stacked. Currently supported:

- `confidence` — red→yellow→green box derived from `max(probs)` at each position (requires `--store_probabilities`)

The metric annotation system (`MetricAnnotation` in `animation_tools.py`) is extensible: new metrics need only a name, a `[steps, seq_len]` or `[seq_len]` value array, and a colormap function. Static (per-position constant) and dynamic (per-step) metrics are both supported.

## CLI examples

```bash
# Default: random unmasking + stochastic sampling
biom3_ProteoScribe_sample \
    --input_path embeddings.pt --config_path config.json \
    --model_path weights.bin --output_path output.pt

# Confidence-based, content-first, with animation
biom3_ProteoScribe_sample \
    --input_path embeddings.pt --config_path config.json \
    --model_path weights.bin --output_path output.pt \
    --unmasking_order confidence_no_pad \
    --animate_prompts all

# Fully deterministic greedy generation with probability logging
biom3_ProteoScribe_sample \
    --input_path embeddings.pt --config_path config.json \
    --model_path weights.bin --output_path output.pt \
    --unmasking_order confidence \
    --token_strategy argmax \
    --store_probabilities

# Animated sequence logo with confidence metric boxes
biom3_ProteoScribe_sample \
    --input_path embeddings.pt --config_path config.json \
    --model_path weights.bin --output_path output.pt \
    --animate_prompts all \
    --store_probabilities \
    --animation_style logo \
    --animation_metrics confidence

# Override config defaults via CLI
biom3_ProteoScribe_sample \
    --input_path embeddings.pt --config_path config.json \
    --model_path weights.bin --output_path output.pt \
    --unmasking_order random \
    --token_strategy argmax \
    --seed 7
```

## JSON config defaults

The strategy defaults can also be set in the Stage 3 JSON config:

```json
{
  "unmasking_order": "random",
  "token_strategy": "sample",
  "num_replicas": 5,
  "diffusion_steps": 1024,
  "batch_size_sample": 32
}
```

CLI flags take precedence over config values when both are specified.

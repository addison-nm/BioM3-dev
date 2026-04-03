# Gumbel-Max Sampling in Stage 3

## Where it's used

`batch_generate_denoised_sampled` in
[src/biom3/Stage3/sampling_analysis.py](../src/biom3/Stage3/sampling_analysis.py)
samples one token per sequence position at each diffusion step. This happens
inside a tight loop that runs `diffusion_steps` (typically 1024) iterations, so
sampling efficiency directly affects wall-clock time.

## Background

Given a categorical distribution with probabilities `p = [p_0, p_1, ..., p_K]`,
we need to draw a sample (an index `k` with probability `p_k`). The standard
approach is `torch.multinomial`, which scans the cumulative distribution function
(CDF). An alternative is the **Gumbel-Max trick**.

### References

- Wikipedia: [Gumbel distribution - Application](https://en.wikipedia.org/wiki/Gumbel_distribution#Application)
- Gumbel, E. J. (1954). *Statistical Theory of Extreme Values and Some
  Practical Applications.* National Bureau of Standards Applied Mathematics
  Series, 33.
- Maddison, C. J., Mnih, A., & Teh, Y. W. (2017). "The Concrete Distribution:
  A Continuous Relaxation of Discrete Random Variables." *ICLR 2017.*
  [arXiv:1611.00712](https://arxiv.org/abs/1611.00712)
- Jang, E., Gu, S., & Poole, B. (2017). "Categorical Reparameterization with
  Gumbel-Softmax." *ICLR 2017.*
  [arXiv:1611.01144](https://arxiv.org/abs/1611.01144)

## The trick

**Claim:** if `G_k ~ Gumbel(0,1)` are independent, then

```
argmax_k( log(p_k) + G_k )
```

returns index `k` with probability exactly `p_k`.

### Intuition

Think of it as a noisy auction. Each token class places a bid equal to
`log(p_k)` plus random Gumbel noise. High-probability classes have higher base
bids (`log(p_k)` closer to 0), so they win more often — but the noise gives
every class a chance, and the Gumbel distribution is the unique noise family
where the winning probabilities exactly match the original categorical
distribution.

This property follows from the fact that the maximum of independent
Gumbel-shifted variables has a closed-form softmax distribution (see the
Wikipedia link above).

### Why it's faster than `torch.multinomial`

| | `torch.multinomial` | Gumbel-Max |
|---|---|---|
| Algorithm | CDF construction + binary search per row | Element-wise ops + argmax |
| Parallelism | Limited (sequential CDF scan per sample) | Fully parallel (all ops are element-wise) |
| Extra allocations | `OneHotCategorical` creates a `(batch, seq, classes)` one-hot tensor | None beyond the noise buffer |
| CUDA sync risk | Some PyTorch versions sync inside `multinomial` | No sync points |

For the BioM3 diffusion loop with `batch=32, seq_len=1024, classes=29`, the
`OneHotCategorical` path allocated and discarded a 3.6 MB one-hot tensor every
step (1024 steps = 3.7 GB transient memory). The Gumbel-Max path reuses a single
pre-allocated noise buffer.

## Implementation in BioM3

### Generating Gumbel noise efficiently

Drawing `G = -log(-log(U))` where `U ~ Uniform(0,1)` requires two `log` calls.
PyTorch's `Tensor.exponential_()` samples from `Exp(1)`, and since
`Exp(1) = -log(Uniform(0,1))`, we get:

```
G = -log(Exp(1) sample) = -log(-log(U))
```

So the implementation becomes:

```python
gumbel_buffer.exponential_()          # E ~ Exp(1), in-place
# G = -log(E), so log(p) + G = log(p) - log(E)
sampled = (probs.log() - gumbel_buffer.log()).argmax(dim=-1)
```

### Full code (from sampling_analysis.py)

```python
# Before the loop: pre-allocate buffer matching (batch, seq_len, num_classes)
gumbel_buffer = torch.empty(
    batch_size, seq_len, args.num_classes,
    dtype=temp_y_c.dtype, device=args.device
)

# Inside the loop:
conditional_prob, prob = predict_next_index(...)

gumbel_buffer.exponential_()
next_temp_realization = (
    conditional_prob.probs.log() - gumbel_buffer.log()
).argmax(dim=-1)
```

### Previous implementation (replaced)

```python
# Created OneHotCategorical, called torch.multinomial internally,
# produced a one-hot tensor, then argmax'd back to indices:
next_temp_realization = torch.argmax(conditional_prob.sample(), dim=-1)
```

## Numerical considerations

- `probs.log()` produces `-inf` for zero-probability classes. This is correct:
  `-inf + G = -inf`, so argmax never selects a zero-probability class.
- `exponential_()` produces strictly positive values, so `gumbel_buffer.log()`
  is always finite.
- The argmax only requires correct relative ordering, which `log` (monotonic)
  preserves. Absolute magnitude doesn't matter.

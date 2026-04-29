# GDPO fine-tuning of Stage 3

GDPO (Group Diffusion Policy Optimization) is a sibling RL fine-tuner
to [GRPO](grpo_finetuning.md) for Stage 3 ProteoScribe. Where GRPO ports
*diffu-GRPO* (a token-level mean-field RL adaptation for diffusion
language models), GDPO replaces that biased estimator with a
**sequence-level ELBO estimated via Semi-deterministic Monte Carlo
(SDMC)** following Rojas et al., ICLR 2026
([arXiv:2510.08554v3](https://arxiv.org/abs/2510.08554)). Both modules
sit side by side under `src/biom3/rl/` and share rollout, prompt
encoding, reward, and I/O plumbing.

If you're new to RL fine-tuning of diffusion language models, read this
section first.

## Background: why GDPO

The reverse process of an absorbing-state diffusion model (which
ProteoScribe is) defines `π_θ(y | q)` as a marginal over masked
intermediates `y_t`. Unlike autoregressive LMs, the *order-agnostic*
generation paradigm makes both **token-level** likelihoods and
**sequence-level** likelihoods intractable in closed form. Existing RL
methods diverge here:

- **diffu-GRPO** (Zhao et al. 2025, what `biom3.rl.grpo` ports)
  approximates per-token log-probs by feeding the model the *fully
  masked* prefix at `t=0` and reading off `log π_θ(y^i | M, …, M, q)`
  in a single forward pass — fast but mean-field-biased; sequential
  token correlations are discarded.
- **GDPO** (this module) replaces token log-probs with a
  **sequence-level ELBO**:
  $$
  \mathcal{L}_{\text{ELBO}}(y \mid q) \;=\; \int_0^1 \mathbb{E}_{y_t \sim \pi_t(\cdot \mid y)} \!\!\left[\frac{1}{t} \sum_{i=1}^L \mathbf{1}[y_t^i = M] \log \pi_\theta(y^i \mid y_t, q)\right] dt.
  $$
  The integrand is exactly what `Stage3.transformer_training_helper.cond_elbo_objective` already computes during training (sum of log-probs at masked positions, weighted by `1/(L − idx + 1)` ≈ `1/t`).

The hard part is making the integral cheap. Naive double-Monte-Carlo
(sample `t`, then sample `y_t`) explodes in variance — Fig. 2a of the
paper attributes ~96% of variance to random `t`. **SDMC** kills that
variance by replacing the outer sampling with a deterministic
quadrature on `(0, 1]`; only the inner mask sampling remains
stochastic. The paper shows `N = 2–3` quadrature points suffice.

The downstream payoffs:

- **Sequence-level importance ratio** in PPO-clip:
  `r_g = exp(ELBO_new(y_g) − ELBO_old(y_g))` — a single scalar per
  generated sequence rather than a vector of per-token ratios. Less
  brittle, less prone to clipping artifacts.
- **Unnormalized advantage** `A_g = R_g − mean(R)` (paper Eq. 6,
  following Liu et al. 2025b) instead of GRPO's `(R − μ)/σ`. The std
  normalization is a known bias source.
- **Distinct `π_old` per outer iteration**, separate from a frozen
  `π_ref` used only for KL.

## What GDPO does, end to end

```
for step in 1..S:
    π_old ← deepcopy(π_θ)                               # paper Alg.1 line 3
    sample G completions y_g ~ π_old(·|q)               # diffusion rollout (= L forwards)
    R_g = reward(y_g) for g in 1..G
    A_g = R_g − mean(R)                                 # unnormalized

    # Build SDMC corruptions ONCE; reuse across π_old, π_new, (π_ref).
    for each (t_n, w_n) in the quadrature grid:         # default: midpoint, N=3
        for k in 1..inner_mc:                            # default: 1
            sample y_t ~ π_t(·|y_g)                      # mask exactly idx_n positions

    elbo_old_g = Σ_n w_n · (1/inner_mc) · Σ_k (1/t_n) · Σ_i 1[masked] · log π_old(y_g^i | y_t, q)
    elbo_new_g = same, under π_θ                         # autograd flows here
    r_g        = exp(elbo_new_g − elbo_old_g)            # scalar per sequence
    pg_loss    = (1/G) Σ_g (1/|y_g|) · max(−A_g·r_g, −A_g·clip(r_g, 1±ε))
    kl_loss    = k3 estimator (cheap)  OR  (ELBO_ref − ELBO_new).mean()  (sdmc)
    loss       = pg_loss + β · kl_loss
    AdamW step on π_θ
```

Compute budget per step (`B=1`, `G=8`, `N=7`, `inner_mc=1`, `L=128`):
≈ `L (rollout) + N·G·2 (elbo_old + elbo_new)` = 128 + 112 = ~240 model
forwards. Doubles to ~352 if you switch `kl_estimator` from
`tokenwise_k3` to `sdmc`.

## Layout

```
src/biom3/rl/
  gdpo.py                # GDPOConfig, gdpo_train,
                         #   _build_grid, _build_shared_corruptions,
                         #   _elbo_sdmc, _tokenwise_k3_kl
  run_gdpo_train.py      # CLI runner (argparse + load_json_config composition)
  __main__.py            # +run_gdpo_train wrapper for the entry point
  plotting.py            # plot_train_log: shared diagnostic figures (GRPO + GDPO)

configs/grpo/
  example_gdpo.json      # smoke config
  production_gdpo.json   # production config
  _base_grpo.json        # shared base (Stage 1/2/3 inference configs)

scripts/
  gdpo_train_singlenode.sh      # thin wrapper for biom3_gdpo_train
  run_gdpo_smoke_5steps.sh      # 5-step / G=4 / N=2 / inner_mc=1 interactive smoke

tests/rl_tests/test_gdpo_smoke.py
```

## Configuration

GDPO inherits all GRPO fields via `_base_grpo.json` (Stage 1/2/3 paths,
weights, learning rate, AdamW, prompts, reward) and adds GDPO-specific
fields:

| Field | Default | Notes |
|---|---|---|
| `n_quadrature` | 3 | `N` — number of quadrature time-points |
| `quadrature_grid` | `"uniform"` | midpoint rule on `(0, 1]`. `"explicit"` lets you pass `quadrature_points` (and optionally `quadrature_weights`) verbatim |
| `quadrature_points` | `null` | list of `t_n ∈ (0, 1]`; required when `quadrature_grid="explicit"` |
| `quadrature_weights` | `null` | list of `w_n` summing to 1; defaults to `1/N` |
| `inner_mc` | 1 | MC mask samples per `t_n`. Paper says 1 is sufficient |
| `eps_t` | 1e-3 | clamp `t_n` from 0 to avoid blow-up in the `1/t` factor |
| `kl_estimator` | `"tokenwise_k3"` | `"tokenwise_k3"` (cheap, GRPO-style) or `"sdmc"` (one extra ELBO pass) |
| `use_old_policy_snapshot` | `true` | snapshot `π_old` per outer step (paper-faithful). `false` collapses to GRPO behavior of using `π_ref` as `π_old` |
| `advantage_normalize` | `false` | paper-faithful unnormalized `R − mean(R)`. `true` recovers GRPO's `(R − μ)/σ` for parity |
| `debug_log` | `true` | append a per-step plaintext dump (sequences, masks, ELBOs, advantages) to `{output_dir}/debug.out` |

**`paper_t` ↔ `idx` mapping.** ProteoScribe's `idx ∈ [0, L-1]` is the
*number of revealed positions*. The paper's `t ∈ (0, 1]` is the
*fraction masked*. Internally:
`idx_n = clamp(round((1 − t_n) · L), 0, L-1)`.

With the defaults (`N=3, L=128`):

| `t_n` | `idx_n` | reveal % |
|---|---|---|
| 1/6 ≈ 0.167 | 107 | ≈83% |
| 1/2 | 64 | 50% |
| 5/6 ≈ 0.833 | 21 | ≈17% |

These are the reveal fractions the model is conditioned on when
computing each contribution to the ELBO sum.

## Running

Single-tile interactive (Aurora):

```bash
ZE_AFFINITY_MASK=0 ./scripts/gdpo_train_singlenode.sh \
    ./configs/grpo/production_gdpo.json \
    production_v01 \
    xpu \
    --stage1_weights ./weights/PenCL/PenCL_V09152023_last.ckpt \
    --stage2_weights ./weights/Facilitator/Facilitator_MMD15.ckpt/last.ckpt \
    --stage3_init_weights ./weights/ProteoScribe/ProteoScribe_SH3_epoch52.ckpt/single_model.pth
```

5-step smoke (`G=4`, `N=2`, `inner_mc=1`):

```bash
./scripts/run_gdpo_smoke_5steps.sh
```

## Outputs

```
{output_root}/{run_id}/
├── final.pt                       # last Stage 3 state_dict
├── step{N}.pt                     # periodic checkpoints
├── train_log.json                 # one row per step + a leading `_meta` row
├── train_diagnostics.png          # 8-panel summary (reward, loss, clip, dw,
│                                  #   length, reward histogram, ELBOs, log-ratio)
├── train_reward_components.png    # one panel per CompositeReward component
└── debug.out                      # per-step plaintext dump (when debug_log=true):
                                   #   prompts, per-replica reward/advantage/ELBO/
                                   #   ratio table, decoded sequences, raw token
                                   #   ids, mask visualization for each SDMC
                                   #   corruption (`_` = masked, AA letter = revealed)
```

`train_log.json`'s leading row carries the SDMC grid metadata (`_meta:
true`, `quadrature_t`, `quadrature_w`, `quadrature_idx`,
`kl_estimator`, …) so downstream analysis can recover what was
actually run. The remaining rows include `rewards_per_replica`
(length `B*G`) and `components_per_replica` (when reward is composite)
for plotting per-replica trajectories.

## When to choose GRPO vs. GDPO

| Situation | Pick |
|---|---|
| First-pass RL fine-tune; want fast iteration; reward already gives strong gradient signal | **GRPO** — fewer NFEs/step (`L + 2`), simpler to reason about |
| Reward is noisy, group-normalized advantage feels weak, you suspect token-level mean-field is biased | **GDPO** — sequence-level ratio, paper-faithful unnormalized advantage |
| Long sequences, the paper-reported "GDPO improves performance beyond training sequence length" matters | **GDPO** |
| Smoke test on CPU / no ESMFold | either — both have `--reward stub` |

The two trainers share `train_log.json` schema (modulo GDPO's extra
ELBO/log-ratio columns) and `plot_train_log` produces consistent
diagnostic figures for both, so back-to-back A/B is straightforward.

## Verification

CPU smoke tests run against the mini Stage 3 fixture:

```bash
pytest tests/rl_tests/test_gdpo_smoke.py -v
```

Eight checks: quadrature midpoint correctness, explicit grid, range
guards, ELBO shape + grad flow + finiteness, mask-token correctness in
shared corruptions, log-ratio ≡ 0 when `π_new == π_old`, k3 KL ≡ 0
under matched policies, full inner GDPO step modifying parameters
under AdamW.

## Side note: known GRPO bug

[`src/biom3/rl/grpo.py:178`](../src/biom3/rl/grpo.py#L178) builds the
fully-masked input for `_policy_logprobs` as `torch.full_like(ids,
pad_id)` with `pad_id = 23`, but the mask/absorbing-state token id is
**0**. GRPO has been feeding the model an all-`<PAD>` prefix at `t=0`
rather than all-MASK whenever it computes per-token log-probs (used for
both the PPO ratio and the KL term). The new GDPO module uses
`MASK_ID = 0` correctly via the helpers in
[`Stage3/transformer_training_helper.py`](../src/biom3/Stage3/transformer_training_helper.py).
A deep audit confirmed the bug is isolated to that single helper —
Stage 3 training, sampling, the pre-unmask feature, and the new GDPO
module all use 0 (MASK) correctly. Fix is one line; tracked separately.

## Reference

- Rojas, Lin, Rasul, Schneider, Nevmyvaka, Tao, Deng. *Improving
  Reasoning for Diffusion Language Models via Group Diffusion Policy
  Optimization.* ICLR 2026. arXiv
  [2510.08554v3](https://arxiv.org/abs/2510.08554). The paper PDF is
  kept at the repo root (`2510.08554v3.pdf`).
- Zhao et al. 2025 — diffu-GRPO (the prior token-level approach
  ported in [grpo_finetuning.md](grpo_finetuning.md)).
- Liu et al. 2025b — unnormalized advantage estimate (paper Eq. 6
  citation).
- Companion docs:
  [docs/grpo_finetuning.md](grpo_finetuning.md),
  [docs/.claude_sessions/2026-04-29_gdpo_implementation.md](.claude_sessions/2026-04-29_gdpo_implementation.md).

# BioM3 RL alignment (DPO / GDPO / GRPO) — file map, commands, settings

Reference for `src/biom3/rl`, the module that fine-tunes BioM3's Stage-3 model
(ProteoScribe, a masked-diffusion transformer) with preference/reward-based
alignment. For Perlmutter-specific environment/Slurm setup, see
[README_perlmutter.md](README_perlmutter.md) — this file is about the RL code
itself: what each algorithm does, what files matter, what commands run them,
and what each setting means.

## The three algorithms

| | DPO | GRPO | GDPO |
|---|---|---|---|
| File | `src/biom3/rl/dpo.py` | `src/biom3/rl/grpo.py` | `src/biom3/rl/gdpo.py` (+ `gdpo_multinode.py`) |
| Type | Offline, contrastive (Diffusion-DPO) | Online RL, PPO-clip | Online RL, PPO-clip |
| Needs at train time | A pre-scored CSV of (prompt, sequence, score) | A live reward function | A live reward function |
| Log-prob handling | Exact likelihood is intractable for an order-agnostic absorbing-state diffusion model — approximated | diffu-GRPO's one-step token-level mean-field log-prob (biased) | Sequence-level ELBO via Semi-deterministic Monte Carlo (SDMC) quadrature — GRPO's sibling, unbiased where GRPO is biased |
| Multi-node | n/a (offline) | **Single-GPU only by design** (`docs/reinforcement_learning/grpo_finetuning.md`: "Multi-GPU support is deferred to Phase 4") | Has a real multi-node path (`gdpo_multinode.py`), raw `torch.distributed`, not DeepSpeed/Accelerate |
| Paper | Diffusion-DPO | GRPO (DeepSeek-style) applied to diffusion | Rojas et al., ICLR 2026, arXiv 2510.08554v3 — "Group Diffusion Policy Optimization" |

Practical implication: **DPO needs sequences you already generated and scored
somehow** (by GRPO/GDPO rollouts scored with a reward, or any other pipeline).
**GRPO/GDPO generate and score on the fly** — the reward function *is* the
data; no CSV needed.

## File map

```
src/biom3/rl/
  __main__.py          console-script entry points (see below)
  dpo.py                Diffusion-DPO trainer
  grpo.py               GRPO trainer (single-GPU)
  gdpo.py               GDPO trainer, single-process/single-GPU
  gdpo_multinode.py     GDPO trainer, multi-node (torch.distributed)
  run_dpo_train.py      argparse CLI + config loading for DPO
  run_grpo_train.py     argparse CLI + config loading for GRPO
  run_gdpo_train.py     argparse CLI + config loading for GDPO; auto-dispatches
                         to gdpo.py (single-process) or gdpo_multinode.py
                         (via is_launched(), i.e. whether torchrun/torch.distributed
                         env vars are present)
  io.py                 checkpoint loading: load_pencl_frozen, load_facilitator_frozen,
                         load_proteoscribe_trainable — handles PyTorch-Lightning-
                         wrapped state dicts
  preference_data.py    DPO's CSV -> PreferenceGroup/PreferenceSampler pipeline
  rollout.py            sequence generation from the diffusion policy
  featurizers.py        one-hot / ESM2 featurizers (used by SurrogateReward)
  diversity.py          pairwise-identity / diversity-bonus math
  plotting.py           train_diagnostics.png / train_reward_components.png
  rewards/
    registry.py          build_reward(name, device, **kwargs) dispatcher
    base.py               reward interface
    esmfold.py             "esmfold_plddt" — ESMFold mean pLDDT (default reward)
    stub.py                 "stub" — random/constant, for smoke tests
    tsv_lookup.py           "tsv_lookup" — exact-match reward from a TSV table
    aa_fraction.py           "aa_fraction" — peak reward at a target amino-acid fraction
    diversity.py              "diversity" — rewards within-group sequence diversity
    composite.py            CompositeReward — weighted-sum/product of others (build in code)
    surrogate.py             SurrogateReward — learned predictor + featurizer (build in code)

docs/reinforcement_learning/
  gdpo_finetuning.md                        GDPO settings, checkpoint paths, outputs
  grpo_finetuning.md                        GRPO settings, "single-GPU only" note
  gdpo_distributed_gradient_correctness.md  proof of the multi-node math (global-batch
                                             normalization, all-reduce on rewards/grads,
                                             detached KL denominators)

configs/dpo/
  _base_dpo.json                    all DPO defaults (below)
  example_dpo_paired.json / example_dpo_weighted.json
  production_dpo_sh3_paired.json / production_dpo_sh3_ftbase.json

configs/grpo/                       (holds both GRPO and GDPO configs)
  _base_grpo.json                   all GRPO/GDPO shared defaults (below)
  example_grpo.json / example_gdpo.json
  production_grpo.json / production_gdpo.json
  pre_unmask_sh3.json / pre_unmask_sh3_d70.json
  prompts/*.txt                     prompt lists for GRPO/GDPO

scripts/
  dpo_train_singlenode.sh(?)... actually: gdpo_train_singlenode.sh, grpo_train_singlenode.sh
  gdpo_train_multinode.sh           dispatches to scripts/launchers/${BIOM3_LAUNCHER:-${BIOM3_MACHINE}}_multinode_rl.sh
  run_gdpo_smoke_5steps.sh, run_grpo_smoke_5steps.sh   canned smoke-test wrappers
  eval_dpo_functional_ranking.py, eval_grpo_checkpoint.py
  train_grpo_surrogate.py, make_grpo_synthetic_eval.py

tests/rl_tests/    CPU-only unit + smoke tests (no GPU/network needed):
                   test_dpo.py, test_gdpo_smoke.py, test_grpo_smoke.py,
                   test_rewards.py, test_diversity.py, test_featurizers.py,
                   test_rollout_pool.py
```

## Entry points

Console scripts (installed by `pip install -e .`, resolve via `__main__.py`):
```
biom3_dpo_train   --config_path ... [--key value ...]
biom3_grpo_train  --config_path ... [--key value ...]
biom3_gdpo_train  --config_path ... [--key value ...]
```
Plain `argparse` — not Hydra/click. `--config_path` points at a JSON config;
any `--key value` flag on the command line overrides the JSON, which overrides
argparse defaults. Configs support inheritance via `"_base_configs": [...]`
and `"_overwrite_configs": [...]` (a simple layered-merge, not omegaconf
composition, even though `omegaconf` is a dependency used elsewhere in the repo).

Normally invoked through the wrapper shell scripts, not directly:
```bash
./scripts/gdpo_train_singlenode.sh CONFIG_PATH RUN_ID DEVICE [--key value ...]
./scripts/grpo_train_singlenode.sh CONFIG_PATH RUN_ID DEVICE [--key value ...]
```
These just `export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=true` and `exec` the
console script with `--config_path/--run_id/--device` filled in from the first
three positional args.

## DPO settings (`configs/dpo/_base_dpo.json`)

```jsonc
{
  "stage1_config": "./configs/inference/stage1_PenCL.json",
  "stage2_config": "./configs/inference/stage2_Facilitator.json",
  "stage3_config": "./configs/inference/stage3_ProteoScribe_sample.json",
  "stage1_weights": null, "stage2_weights": null, "stage3_init_weights": null,

  "data_csv": "./data/rl/processed/biom3_designs.csv",
  "dataset_filter": null,
  "group_by": "prompt_text",       // groups rows sharing a caption; falls back to
                                    // a single group under default_caption if absent
  "default_caption": "SH3 domain protein.",
  "max_len": null,
  "min_group_size": 2,

  "beta": 0.1,                     // DPO temperature
  "length_normalize": true,
  "learning_rate": 1e-6, "weight_decay": 1e-6,
  "steps": 200, "batch_size": 4, "save_steps": 50, "max_grad_norm": 1.0, "seed": 42,

  "n_quadrature": 3, "quadrature_grid": "uniform",  // SDMC ELBO estimator settings —
  "inner_mc": 1, "eps_t": 1e-3,                     // same knobs GDPO uses (see below)

  "pre_unmask": false, "pre_unmask_config": null,
  "output_root": "./outputs/dpo"
}
```

Two DPO loss modes (`--loss_type`, `--pairing`):
- **`paired`** (Bradley-Terry): rank sequences within a group by score, pick a
  chosen/rejected pair either by rank-gap (`--pairing margin`, tune with
  `--gap_level`/`--min_margin`) or by a binary `functional` label
  (`--pairing label`).
- **`weighted`** (ProteinDPO-style): keeps `K` scored candidates per group
  without binarizing (`--num_candidates`, `--temperature`).

**Data format** — `--data_csv` must have columns
`dataset, source, prompt, prompt_text, functional, sequence, score`
(produced upstream by `data/rl/convert_rl_data_to_csv.py`). Sequences are
filtered to valid amino acids for the Stage-3 tokenizer
(`biom3.Stage3.preprocess.encode_protein_sequence`). This is the **input you
need to build yourself** — e.g. by running GDPO/GRPO rollouts and scoring them,
or any other generation+scoring pipeline — before DPO has anything to train on.

## GRPO / GDPO settings (`configs/grpo/_base_grpo.json`)

```jsonc
{
  "stage1_config": "./configs/inference/stage1_PenCL.json",
  "stage2_config": "./configs/inference/stage2_Facilitator.json",
  "stage3_config": "./configs/inference/stage3_ProteoScribe_sample.json",
  "stage1_weights": null, "stage2_weights": null, "stage3_init_weights": null,

  "prompts_path": "./configs/grpo/prompts/example_prompts.txt",
  "steps": 200,
  "num_generations": 4,            // group size G: sequences sampled per prompt per step
  "batch_size": 1,
  "learning_rate": 1e-5, "weight_decay": 1e-6,
  "beta": 0.01,                    // KL coefficient vs. frozen reference policy
  "eps": 0.20,                     // PPO clip range
  "max_grad_norm": 1.0, "save_steps": 50, "seed": 42,

  "reward": "esmfold_plddt",
  "output_root": "./outputs/grpo"
}
```

GDPO-specific overrides on top of the above (see `configs/grpo/example_gdpo.json`
/ `production_gdpo.json`):
```jsonc
{
  "_base_configs": ["./_base_grpo.json"],
  "n_quadrature": 3,               // number of SDMC quadrature points (production uses 4)
  "quadrature_grid": "uniform",    // "uniform" | "explicit" (pair with quadrature_points/weights)
  "inner_mc": 1,                   // inner Monte Carlo samples per quadrature point
  "eps_t": 1e-3,
  "kl_estimator": "tokenwise_k3",  // "tokenwise_k3" | "sdmc"
  "use_old_policy_snapshot": true, // snapshot π_old at step start (--no_old_policy_snapshot to disable)
  "advantage_normalize": false,
  "pre_unmask": true,              // production_gdpo.json only; example_gdpo.json leaves this off
  "pre_unmask_config": "./configs/grpo/pre_unmask_sh3.json",
  "output_root": "./outputs/gdpo"
}
```
`pre_unmask_sh3.json`: `{"strategy": "last_k", "fill_with": "PAD", "diffusion_budget": 128}`
— pre-unmasks part of the sequence before rollout, per the SH3 domain length budget.

Per-step compute cost (from `docs/reinforcement_learning/gdpo_finetuning.md`,
example B=1,G=8,N=7,inner_mc=1,L=128): rollout (L) + N·G·2 for elbo_old+elbo_new
≈ 128+112 = **~240 model forwards/step** (~352 if `kl_estimator=sdmc`).

**Data format** — `--prompts_path` is a plain text file, one prompt per line,
`#`-prefixed lines ignored (`configs/grpo/prompts/example_prompts.txt`,
`sh3_v01_256.txt` … `sh3_v04_mixed_4096.txt`). No CSV, no pre-scoring —
generation and reward scoring both happen live, every step.

## Reward functions (`--reward name`, dispatched via `rewards/registry.py`)

| name | class | notes |
|---|---|---|
| `esmfold_plddt` | `ESMFoldReward` | default; mean pLDDT from `facebook/esmfold_v1` (lazy-loaded on first call — see README_perlmutter.md's known-issues section for the Perlmutter-specific loading failure) |
| `stub` | `StubReward` | random/constant — for smoke tests, no model/network needed |
| `tsv_lookup` | `TsvLookupReward` | exact sequence match against a TSV table |
| `aa_fraction` | `AAFractionReward` | peaks at a target fraction of one amino acid |
| `diversity` | `DiversityReward` | rewards within-group sequence diversity; needs explicit `group_size` kwarg, not available through the plain `--reward` string dispatch |

`CompositeReward` (weighted sum/product of other rewards) and `SurrogateReward`
(learned predictor + featurizer, e.g. from `scripts/train_grpo_surrogate.py`)
are built directly in code, not through `--reward`.

## Weight/config paths currently staged on Perlmutter

The default `stage1_config`/`stage2_config`/`stage3_config` above
(`configs/inference/stage{1,2,3}_*.json`) are architecturally compatible with
the `run1_base` weights bundle staged here — verified by diffing their
`_base_configs`-referenced model configs
(`configs/inference/models/_base_{PenCL,Facilitator}.json`,
`configs/inference/../stage3_training/models/_base_ProteoScribe_1block.json`)
against the bundle's own `configs/bundles/run1_base/_base_*.json`: identical
architecture fields (`protein_encoder_embedding=1280`,
`proj_embedding_dim=512`, `transformer_blocks=1`, etc.). So **no
`--stage{1,2,3}_config` override is needed** with `run1_base` — only the
weight paths:
```
--stage1_weights      ./weights/PenCL/run1_base_pencl.bin
--stage2_weights      ./weights/Facilitator/run1_base_facilitator.bin
--stage3_init_weights ./weights/ProteoScribe/run1_base_proteoscribe.bin
```
Note these differ from the filenames hardcoded in the upstream
`scripts/run_gdpo_smoke_5steps.sh` (`PenCL_V09152023_last.ckpt`,
`Facilitator_MMD15.ckpt/last.ckpt`, `ProteoScribe_SH3_epoch52.ckpt/single_model.pth`)
— that script assumes a different, SH3-finetuned checkpoint set we did not pull.

## Example commands (as actually run/tested here)

GDPO, single GPU, `run1_base` weights, `esmfold_plddt` reward (this is what
`jobs/perlmutter/job_gdpo_smoke.sbatch` runs):
```bash
./scripts/gdpo_train_singlenode.sh \
    ./configs/grpo/example_gdpo.json gdpo_test_${SLURM_JOB_ID} cuda \
    --steps 5 --num_generations 4 --batch_size 1 --save_steps 5 \
    --reward esmfold_plddt \
    --prompts_path ./configs/grpo/prompts/example_prompts.txt \
    --stage1_weights ./weights/PenCL/run1_base_pencl.bin \
    --stage2_weights ./weights/Facilitator/run1_base_facilitator.bin \
    --stage3_init_weights ./weights/ProteoScribe/run1_base_proteoscribe.bin \
    --output_root /pscratch/sd/t/tnnandi/biom3-rl-outputs/gdpo
```
Confirmed working through Stage 1/2/3 loading, rollout, and ELBO computation
(see log timings: `snapshot_old 0.23s`, `encode_z_c 8.81s`, `rollout 3.03s`,
`elbo_old 0.13s`) before failing at the ESMFold reward load step — a
filesystem issue (see README_perlmutter.md), not an RL-code issue.

DPO and GRPO have **not** been run end-to-end on Perlmutter yet (only unit
tests via `pytest tests/rl_tests/`). To try DPO once a preference CSV exists:
```bash
biom3_dpo_train \
    --config_path ./configs/dpo/example_dpo_paired.json \
    --run_id dpo_test --device cuda \
    --data_csv /global/cfs/cdirs/m5415/tnnandi/biom3/data/rl/<your>.csv \
    --stage1_weights ./weights/PenCL/run1_base_pencl.bin \
    --stage2_weights ./weights/Facilitator/run1_base_facilitator.bin \
    --stage3_init_weights ./weights/ProteoScribe/run1_base_proteoscribe.bin \
    --output_root /pscratch/sd/t/tnnandi/biom3-rl-outputs/dpo
```

## Outputs

Each run writes to `<output_root>/<run_id>/`:
```
final.pt                       final model weights
step{N}.pt                     checkpoint every --save_steps
train_log.json                 per-step metrics
train_diagnostics.png          loss/reward curves (src/biom3/rl/plotting.py)
train_reward_components.png    per-component reward breakdown (composite rewards)
debug.out                      if debug_log/--debug_log enabled
```

## CPU-only test suite (no GPU, no network, no weights needed)

```bash
PYTHONPATH=src pytest tests/rl_tests/ -v
```
89 passed / 1 skipped (`test_esm2_featurizer_shape_and_finite`, skips without
staged ESM weights) as of this port. `CLAUDE.md` also documents
`pytest tests/ --quick` for a faster dev loop that skips
`@pytest.mark.benchmark`/`@pytest.mark.requires_gpu`-marked tests repo-wide.

## Math and parameters

The actual equations implemented in `src/biom3/rl/{grpo,gdpo,dpo}.py`, with
parameter names matching the `*Config` dataclasses / JSON configs exactly so
they're greppable against the code.

### Shared setup

ProteoScribe is an order-agnostic **absorbing-state diffusion model**:
`π_θ(y | z_c)` generates a sequence `y = (y^1, …, y^L)` conditioned on a
frozen embedding `z_c` (Stage 1 PenCL → Stage 2 Facilitator, both frozen
throughout RL). A forward step reveals a random subset of `idx` positions
with their true tokens and masks the rest with the absorbing token `M`
(id 0, `MASK_ID` in code — **not** `<PAD>`, see the "known GRPO bug" note in
`docs/reinforcement_learning/gdpo_finetuning.md`); the model predicts a
distribution over the vocabulary at each masked position, conditioned on
what's revealed and on `z_c`.

The exact sequence likelihood `log π_θ(y | z_c)` requires marginalizing
over all `L!` reveal orders, which is intractable. **This is the one
problem all three algorithms solve differently** — each substitutes its own
tractable proxy for "how likely is the policy to produce this sequence":

| Algorithm | Log-prob proxy | Bias |
|---|---|---|
| GRPO | one-step, token-level, mean-field (`_policy_logprobs`) | biased |
| GDPO | sequence-level ELBO via SDMC quadrature (`_elbo_sdmc`) | unbiased-in-`t`, MC noise only |
| DPO | same SDMC ELBO as GDPO (reused verbatim) | same as GDPO |

All three also keep a frozen reference policy `π_ref`, a `copy.deepcopy` of
the policy taken once before training starts and never updated
(`ref_s3 = copy.deepcopy(s3).eval()`), used to keep the fine-tuned policy
from drifting too far from the pretrained one.

### GRPO — `src/biom3/rl/grpo.py`

**Log-prob proxy (diffu-GRPO, one-step mean-field).** Feed the model the
*fully-masked* input (`idx = 0`, every position = `M`) and read off each
position's log-prob in a single forward pass:

$$
\log \hat{\pi}_\theta(y^i \mid z_c) := \log \pi_\theta(y^i \mid x = M^L, z_c)
$$

(`_policy_logprobs`, `grpo.py:183-205`). This ignores inter-position
correlations the true order-agnostic marginal would capture — fast (one
forward per policy) but biased.

**Group-normalized advantage.** For prompt `q`, sample `K = num_generations`
sequences and normalize their rewards within the group (no critic network):

$$
A_{q,i} = \frac{R_{q,i} - \mu_q}{\sigma_q + \varepsilon}, \qquad
\mu_q = \operatorname{mean}_j R_{q,j}, \qquad
\sigma_q = \operatorname{std}_j R_{q,j}
$$

(`grpo.py:436-439`). Broadcast across every valid (non-`<PAD>`) token
position of sample `i`.

**Probability ratio and PPO-clip loss**, per token, averaged over valid positions:

$$
\rho_{i,t} = \exp\!\big(\log \pi_\theta(y^i_t) - \log \pi_{\mathrm{ref}}(y^i_t)\big)
$$

$$
L_{PG} = \operatorname{mean}_{i,t\ \mathrm{valid}} \Big[-\min\big(\rho_{i,t} A_i,\ \operatorname{clip}(\rho_{i,t},\ 1-\varepsilon,\ 1+\varepsilon)\, A_i\big)\Big]
$$

(`grpo.py:444-450`). **Code-verified detail:** the ratio's denominator is
`π_ref`, not a freshly-resnapshotted `π_old` — GRPO here keeps only one
frozen snapshot and uses it both as the PPO-ratio baseline and the KL
anchor (`lp_ref_tok` at `grpo.py:429`, reused at `grpo.py:445` and `453`).
This differs from GDPO, which explicitly resnapshots `π_old` every step
and keeps a separate, permanently-frozen `π_ref` for KL.

**KL penalty**, Schulman's k3 estimator (non-negative, lower-variance than
the naive `Δ`):

$$
\Delta_{i,t} = \log \pi_{\mathrm{ref}}(y^i_t) - \log \pi_\theta(y^i_t)
$$

$$
L_{KL} = \operatorname{mean}_{i,t\ \mathrm{valid}} \big[\exp(\Delta_{i,t}) - \Delta_{i,t} - 1\big]
$$

(`grpo.py:452-457`).

**Total loss**: $L = L_{PG} + \beta \cdot L_{KL}$ (`grpo.py:459`), gradient-clipped to
$\|g\|_2 \le \texttt{max\_grad\_norm}$ before the AdamW step.

| Parameter | Default | Meaning |
|---|---|---|
| `num_generations` (`K`) | 4 | sequences sampled per prompt per step |
| `batch_size` (`B`) | 1 | prompts per gradient step |
| `learning_rate` | 1e-5 | AdamW |
| `weight_decay` | 1e-6 | AdamW |
| `beta` (`β`) | 0.01 | KL coefficient |
| `eps` (`ε`) | 0.20 | PPO clip range |
| `max_grad_norm` | 1.0 | grad-norm clip |
| `steps` / `save_steps` | 200 / 50 | total steps / checkpoint cadence |
| `reward` | `esmfold_plddt` | reward function name (see main README) |

### GDPO — `src/biom3/rl/gdpo.py` (+ `gdpo_multinode.py`)

**Log-prob proxy: sequence-level ELBO.** The true ELBO (Rojas et al.,
ICLR 2026, eq. 5) integrates over the diffusion time axis:

$$
L_{\mathrm{ELBO}}(y \mid z_c) = \int_0^1 \mathbb{E}_{y_t \sim \pi_t(\cdot \mid y)}\left[\frac{1}{t} \sum_i \mathbb{1}[y_t^i = M] \cdot \log \pi_\theta(y^i \mid y_t, z_c)\right] dt \;\le\; \log \pi_\theta(y \mid z_c)
$$

a proper lower bound, unlike GRPO's single-step proxy. Naive double-MC
(sample `t`, then sample `y_t`) has ~96% of its variance coming from
random `t` (paper Fig. 2a); **SDMC** replaces the outer integral with a
deterministic quadrature so only the inner mask-sampling stays stochastic:

1. Pick $N = \texttt{n\_quadrature}$ points $t_n \in (0, 1]$ with weights $w_n$
   ($\sum_n w_n = 1$) — `quadrature_grid = "uniform"` uses midpoints
   $t_n = (n - 0.5)/N$; `"explicit"` takes `quadrature_points`/
   `quadrature_weights` verbatim (`_build_grid`).
2. Map each $t_n$ (fraction masked) to a model time-index
   $\mathrm{idx}_n = \operatorname{clamp}(\operatorname{round}((1-t_n)\cdot L),\ 0,\ L-1)$
   (revealed-position count).
3. At each $\mathrm{idx}_n$, draw $K_{\mathrm{inner}} = \texttt{inner\_mc}$ random reveal-order
   corruptions $y_{t_n,k}$ of $y$, each revealing exactly $\mathrm{idx}_n$ positions
   (`_build_shared_corruptions`) — **shared across `π_old`, `π_new`, and
   (optionally) `π_ref`**, so the importance ratio isn't polluted by extra
   mask-sampling noise.
4. Estimate:

$$
\hat{L}_{\mathrm{ELBO}}(y \mid z_c) = \sum_n w_n \cdot \frac{1}{K_{\mathrm{inner}}} \sum_k \frac{1}{\max(t_n, \varepsilon_t)} \sum_i \mathbb{1}[y_{t_n,k}^i = M] \cdot \log \pi_\theta(y^i \mid y_{t_n,k}, z_c)
$$

(`_elbo_sdmc`, `gdpo.py:298-346`; $\varepsilon_t = \texttt{eps\_t}$ clamps the $1/t$ blow-up
near $t \to 0$).

**Per-step loop** (Alg. 1 of the paper):
```
π_old ← deepcopy(π_θ)                         # every outer step (use_old_policy_snapshot)
y_g ~ π_old(· | z_c), g = 1..G                 # diffusion rollout, optionally across
                                                #   multiple devices via rollout.RolloutPool
R_g = reward(y_g);  A_g = R_g − mean(R)        # unnormalized (paper eq. 6) by default
elbo_old_g = L̂_ELBO under π_old (no grad)
elbo_new_g = L̂_ELBO under π_θ (autograd flows here)
r_g        = exp(elbo_new_g − elbo_old_g)      # sequence-level ratio, one scalar per sequence
```
(`gdpo.py:787-916`). Advantage normalization `(R−μ)/σ` is available via
`advantage_normalize=True` for parity testing against GRPO, but the
paper-faithful default is the unnormalized `R − mean(R)`.

**Sequence-level PPO-clip loss**, normalized per sequence by its valid
(non-`<PAD>`) length:

$$
L_{PG} = \operatorname{mean}_g\left[\frac{1}{|y_g|} \cdot \max\big(-A_g r_g,\ -A_g \operatorname{clip}(r_g,\ 1-\varepsilon,\ 1+\varepsilon)\big)\right]
$$

(`gdpo.py:918-924`).

**KL term**, `kl_estimator` selects one of two:
- `"tokenwise_k3"` (default) — the same cheap k3 estimator as GRPO, one
  extra fully-masked forward through `π_θ` and `π_ref` (`_tokenwise_k3_kl`).
- `"sdmc"` — a forward-KL surrogate reusing the same SDMC grid:
  $L_{KL} = \operatorname{mean}_g(\mathrm{elbo}_{\mathrm{ref},g} - \mathrm{elbo}_{\mathrm{new},g})$,
  i.e. one extra no-grad ELBO pass through `π_ref` per step (`gdpo.py:932-934`).

**Total loss**: $L = L_{PG} + \beta \cdot L_{KL}$ (`gdpo.py:940`).

**Pre-unmask** (`pre_unmask=True`): keeps the architectural length `L`
fixed but only diffuses over the first `D = diffusion_budget` positions;
`[D, L)` is pre-filled with a fixed token and always counted as "revealed."
Collapses rollout + ELBO compute by roughly `L/D`. Per-step cost with
`B=1, G=8, N=7, inner_mc=1, L=128`: rollout (`L`) + `N·G·2` (elbo_old +
elbo_new) ≈ 128 + 112 = **~240 model forwards**; ~352 if
`kl_estimator="sdmc"` (one more `N·G` pass for `elbo_ref`).

| Parameter | Default | Meaning |
|---|---|---|
| `num_generations` (`G`) | 4 | sequences per prompt per step |
| `batch_size` (`B`) | 1 | prompts per gradient step |
| `learning_rate` / `weight_decay` | 1e-5 / 1e-6 | AdamW |
| `beta` (`β`) | 0.01 | KL coefficient |
| `eps` (`ε`) | 0.20 | PPO clip range |
| `n_quadrature` (`N`) | 3 (production configs use 4) | SDMC quadrature points |
| `quadrature_grid` | `"uniform"` | midpoint rule, or `"explicit"` with `quadrature_points`/`quadrature_weights` |
| `inner_mc` (`K_inner`) | 1 | MC mask samples per quadrature point (paper: 1 suffices) |
| `eps_t` (`ε_t`) | 1e-3 | clamps `t_n` away from 0 |
| `kl_estimator` | `"tokenwise_k3"` | `"tokenwise_k3"` (cheap) or `"sdmc"` (paper-faithful, pricier) |
| `use_old_policy_snapshot` | `true` | resnapshot `π_old` each step; `false` collapses to GRPO-style behavior |
| `advantage_normalize` | `false` | `false` = paper eq. 6 unnormalized; `true` = GRPO's `(R−μ)/σ` |
| `pre_unmask` / `pre_unmask_config` | `false` / `null` | diffusion-budget truncation, see above |
| `gradient_checkpoint` | `true` | recompute activations in backward; needed to fit `L≥1024, N≥7` on one GPU |

### DPO — `src/biom3/rl/dpo.py`

Offline and **rollout-free**: no `π_old`, no live sampling. It reuses
GDPO's exact SDMC ELBO (`_build_grid`, `_build_shared_corruptions`,
`_elbo_sdmc`, imported directly from `gdpo.py`) against the single frozen
`π_ref` snapshotted once before training, applied to sequences that were
already generated and scored upstream. Both objectives use a
length-normalized, `β`-scaled ELBO log-ratio as the implicit reward:

$$
\rho_\theta(y) = \beta \cdot \frac{\mathrm{ELBO}_\theta(y \mid z_c) - \mathrm{ELBO}_{\mathrm{ref}}(y \mid z_c)}{|y|} \quad (\text{if length\_normalize; else no } /|y|)
$$

(`_paired_elbos` / `_weighted_elbos`, `dpo.py:128-188`).

**`paired`** — Bradley-Terry logistic loss (ProteinDPO eq. 10) on a
chosen/rejected pair `(y_w, y_l)` per prompt group, picked either by
score-rank gap (`pairing="margin"`, tuned by `gap_level`/`min_margin`) or
by a binary functional label (`pairing="label"`):

$$
\mathrm{margin} = \rho_\theta(y_w) - \rho_\theta(y_l)
$$

$$
L = -\big[(1 - \mathrm{ls}) \cdot \log \sigma(\mathrm{margin}) + \mathrm{ls} \cdot \log \sigma(-\mathrm{margin})\big] \qquad (\mathrm{ls} = \texttt{label\_smoothing},\ \text{cDPO})
$$

(`dpo.py:279-293`) — the standard DPO loss with the SDMC ELBO substituted
for the exact log-likelihood ratio (Diffusion-DPO's substitution).
$\mathrm{pref\_acc} = \operatorname{mean}(\mathrm{margin} > 0)$ (reference-relative accuracy) and
$\mathrm{abs\_acc} = \operatorname{mean}(\mathrm{ELBO}_\theta(y_w) > \mathrm{ELBO}_\theta(y_l))$ (reference-free, does the
policy's own ELBO already rank them right) are logged as diagnostics.

**`weighted`** — scalar-label objective (ProteinDPO eq. 15-17) on `K =
num_candidates` scored candidates per group, no binarization: match the
model's implicit-reward softmax to a Boltzmann target built from the raw
scores at temperature `T`:

$$
\mathrm{target} = \operatorname{softmax}(\mathrm{score} / T) \qquad (T = \texttt{temperature})
$$

$$
L = -\sum_k \mathrm{target}_k \cdot \log \operatorname{softmax}(\rho_\theta)_k
$$

(`dpo.py:294-309`). `top1_agree` / `abs_top1_agree` are the reference-
relative / reference-free argmax-agreement diagnostics.

| Parameter | Default | Meaning |
|---|---|---|
| `beta` (`β`) | 0.1 | DPO temperature (implicit-reward scale) |
| `length_normalize` | `true` | divide the ELBO log-ratio by non-`<PAD>` token count |
| `learning_rate` / `weight_decay` | 1e-6 / 1e-6 | AdamW (note: 10× smaller LR than GRPO/GDPO) |
| `lr_scheduler` / `warmup_steps` | `"constant"` / 0 | `"linear"` decays LR to 0 after warmup |
| `loss_type` | `"paired"` | `"paired"` or `"weighted"` |
| `pairing` / `gap_level` / `min_margin` | `"margin"` / 0.5 / 0.0 | paired-mode pair selection |
| `label_smoothing` | 0.0 | cDPO smoothing on the logistic target |
| `num_candidates` (`K`) / `temperature` (`T`) | 4 / 0.1 | weighted-mode group size / Boltzmann temperature |
| `n_quadrature` / `quadrature_grid` / `inner_mc` / `eps_t` | 3 / `"uniform"` / 1 / 1e-3 | same SDMC grid knobs as GDPO |
| `batch_size` | 4 | pairs (paired) or groups (weighted) per step |

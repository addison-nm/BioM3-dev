# Prompt: integrate GRPO reinforcement-learning fine-tuning into BioM3

Use this prompt to bootstrap a session focused on porting a collaborator's
GRPO (Group Relative Policy Optimization) implementation of BioM3 Stage 3
fine-tuning into our packaged BioM3-dev repo. Load context first, then wait
for the user's specific request.

## Mission

A collaborator built a GRPO RL fine-tuner on top of the upstream HuggingFace
distribution of BioM3 (`hugging-face-biom3`, a flat directory layout with
`Stage{1,3}_source/` modules). We want the same capability in our refactored
package (`biom3.*`) so it composes with our config system, checkpoint
formats, and HPC scaffolding.

The donor work is at `../biom3-grpo-rl/` and consists of one self-contained
script (`biom3_grpo.py`, ~600 lines), a launcher (`run_grpo.sh`), a
`prompts/` directory of plain-text prompts, and a small `requirements.txt`.

## Read these first (in order)

1. **`../biom3-grpo-rl/README.md`** and **`../biom3-grpo-rl/biom3_grpo.py`** —
   the entire algorithm lives in one file; read it end-to-end before planning.
2. **`../hugging-face-biom3/Stage3_source/sampling_analysis.py`** — the donor
   sampler that GRPO calls. Compare against our
   `src/biom3/Stage3/sampling_analysis.py` for signature drift.
3. **Project conventions** — `CLAUDE.md` at the repo root: layout, worktree
   workflow, commit style, distributed-training pitfalls.
4. **Stage 3 surface in our repo** —
   `src/biom3/Stage3/{model.py,cond_diff_transformer_layer.py,sampling_analysis.py,io.py,run_PL_training.py,__main__.py}`
   and `src/biom3/core/io.py`.

---

## Algorithm: what GRPO is doing here

**GRPO** (Group Relative Policy Optimization) is a variant of PPO that
estimates the baseline by group-normalizing rewards across multiple samples
drawn from the *same* prompt, rather than learning a separate value/critic
network. It's the algorithm DeepSeek used for R1-style RL on LLMs.

In this BioM3 instantiation:

- **Policy** — Stage 3 ProteoScribe (absorbing-state discrete diffusion
  transformer, ~86M params). Generates protein sequences conditioned on a
  text-derived embedding `z_c`.
- **Reference** — `copy.deepcopy(policy)` frozen at step 0, used for KL.
  (`biom3_grpo.py:431-433`)
- **Conditioning pipeline** — frozen Stage 1 PenCL + frozen Stage 2
  Facilitator turn each text prompt into `z_c`. Only Stage 3 is updated.
- **Rollout** — for each of B prompts, sample K sequences via
  `batch_generate_denoised_sampled` (`biom3_grpo.py:302-309`).
- **Reward** — ESMFold mean pLDDT on the decoded amino-acid sequence,
  clipped to 500 residues. (`ESMFoldReward.__call__`,
  `biom3_grpo.py:376-397`).
- **Advantage** — group-normalized per prompt: reshape `R` to `(B, K)`,
  subtract the per-prompt mean, divide by per-prompt std.
  (`biom3_grpo.py:503-507`)
- **Log-probabilities** — for both policy and reference, run the diffusion
  model on a *fully-masked* input at `t=0` and gather the log-probability of
  each generated token at its own position. (`biom3_grpo.py:338-343, 496`).
  Uses a valid-position mask so padding doesn't contribute.
- **Loss** — PPO-clip surrogate (default `ε = 0.20`) plus a KL penalty
  (default `β = 0.01`) using the Schulman estimator
  `KL ≈ exp(Δ) − Δ − 1` where `Δ = logp_ref − logp_new`.
  (`biom3_grpo.py:520-539`)
- **Optimizer** — AdamW, fixed LR 1e-5, weight decay 1e-6, grad clip 1.0.
  No scheduler. (`biom3_grpo.py:448-452, 543`)

Key hyperparameters and their defaults: `--steps 200`, `--K 4`,
`--batch_size 1`, `--lr 1e-5`, `--beta 0.01`, `--eps 0.20`,
`--save_every 50`.

### Equations

Notation: a *prompt* is indexed by $q$, a *sample* within a group by $i$,
and a token *position* within a sequence by $t$. For each prompt $q$ we
draw $K$ sequences $o_{q,1}, \dots, o_{q,K}$ from the policy
$\pi_{\theta_{\text{old}}}$. The policy is the diffusion model; for GRPO
purposes we treat its per-token log-probability
$\log \pi_\theta(o_{q,i,t} \mid q)$ as the value returned by running the
diffusion forward on a fully-masked input at $t = 0$ and gathering the
chosen token's log-probability at position $t$.

**Per-prompt group-normalized advantage.** No critic; the baseline is
the empirical mean of the group's rewards, scaled by their std:

$$
\hat{A}_{q,i,t} \;=\; \frac{r(o_{q,i}) \;-\; \mu_q}{\sigma_q + \varepsilon_{\text{std}}},
\qquad
\mu_q = \frac{1}{K}\sum_{j=1}^{K} r(o_{q,j}),
\qquad
\sigma_q = \sqrt{\tfrac{1}{K}\sum_{j=1}^{K}\bigl(r(o_{q,j}) - \mu_q\bigr)^2}.
$$

In this codebase the advantage is broadcast across all valid positions
$t$ of sample $i$ (i.e. $\hat{A}$ does not depend on $t$ beyond masking).
The reward $r(o)$ is ESMFold mean pLDDT.

**Probability ratio.**

$$
\rho_{q,i,t}(\theta) \;=\; \exp\!\Bigl(
  \log \pi_\theta(o_{q,i,t} \mid q) \;-\; \log \pi_{\theta_{\text{old}}}(o_{q,i,t} \mid q)
\Bigr).
$$

**PPO-clip surrogate (per-token, then averaged over valid positions).**

$$
\mathcal{L}_{\text{PG}}(\theta)
\;=\; -\,\mathbb{E}_{q,\,i,\,t \in \text{valid}}\!\Bigl[
  \min\!\bigl(
    \rho_{q,i,t}\,\hat{A}_{q,i,t},\;
    \operatorname{clip}(\rho_{q,i,t},\, 1-\epsilon,\, 1+\epsilon)\,\hat{A}_{q,i,t}
  \bigr)
\Bigr].
$$

**KL penalty against the frozen reference (Schulman's $k_3$ estimator).**
Let $\Delta_{q,i,t} = \log \pi_{\text{ref}}(o_{q,i,t} \mid q) - \log \pi_\theta(o_{q,i,t} \mid q)$. Then

$$
\widehat{\mathrm{KL}}\bigl(\pi_\theta \,\|\, \pi_{\text{ref}}\bigr)_{q,i,t}
\;=\; \exp(\Delta_{q,i,t}) \;-\; \Delta_{q,i,t} \;-\; 1.
$$

This is non-negative, unbiased in expectation under $\pi_\theta$, and lower
variance than the naive $\Delta$ estimator. Averaged over valid positions
to get $\mathcal{L}_{\text{KL}}(\theta)$.

**Total GRPO loss.**

$$
\mathcal{L}_{\text{GRPO}}(\theta) \;=\; \mathcal{L}_{\text{PG}}(\theta) \;+\; \beta \cdot \mathcal{L}_{\text{KL}}(\theta),
$$

with defaults $\epsilon = 0.20$, $\beta = 0.01$. Gradients are clipped to
$\lVert g \rVert_2 \le 1.0$ before the AdamW step
($\text{lr} = 10^{-5}$, weight decay $= 10^{-6}$).

**On-policy approximation in this implementation.** Strict GRPO/PPO
requires storing $\log \pi_{\theta_{\text{old}}}$ (the snapshot used for
sampling) so that $\rho$ can drift across multiple update epochs over the
same rollouts. The donor's loop performs one update per rollout batch, so
$\theta_{\text{old}} \equiv \theta$ at the moment of the update and
$\rho \equiv 1$ exactly. This still yields the correct gradient,

$$
\nabla_\theta \mathcal{L}_{\text{PG}} \;=\; -\,\mathbb{E}\bigl[\hat{A}\,\nabla_\theta \log \pi_\theta\bigr],
$$

but the PPO clip becomes a no-op for the first inner step and only
matters if/when we add multiple inner epochs per rollout. KL against the
*frozen* reference is the only thing keeping the policy from collapsing.

---

## What has been done (donor repo `../biom3-grpo-rl/`)

| Component | File | Status |
|---|---|---|
| GRPO trainer | `biom3_grpo.py` (single ~600-line script) | Working on HF BioM3 |
| Wrapped policy | `BioM3ForGRPO` (`biom3_grpo.py:237-343`) — `PreTrainedModel` subclass that owns Stage1/2/3 and exposes `forward(z_c, x_masked) → logp` | Done |
| Reward | `ESMFoldReward` (`biom3_grpo.py:350-397`) — loads `facebook/esmfold_v1`, returns per-sequence mean pLDDT | Done |
| Prompt loader | `_load_prompts()` (`biom3_grpo.py:409-419`) — txt file, `#` comments, optional `name|text` | Done |
| Launcher | `run_grpo.sh` — single GPU, env-var overrides, no torchrun/DeepSpeed | Single-GPU only |
| Checkpoints | Saved as `{"step": N, "model_state": ...}` to `step{N}.pt` and `final.pt` | Raw `.pt`, no Lightning shape |
| Metrics | `train_log.json` (one row per step: pLDDT, loss, pg, kl, clip, len) | Done |
| Distributed | None | Not implemented |
| Lightning integration | None | Not implemented |
| Config files | None — pure CLI args | Not implemented |

Donor's deps (`requirements.txt`): `transformers==4.49.0`, `fair-esm==2.0.0`,
`pytorch-lightning==1.9.5`, `axial-positional-embedding`,
`linear-attention-transformer`, `einops`, plus numpy/pandas/scipy. PyTorch
2.5.1 + CU124 installed separately.

The donor expects `BIOM3_DIR` to point at the HF clone (which contains
`Stage1_source/`, `Stage3_source/`, top-level `stage{1,2,3}_config.json`,
and `weights/{PenCL,Facilitator,ProteoScribe,LLMs}/`).

---

## Conflicts & things to watch out for

These are the concrete deltas the integration must resolve. Each is anchored
to specific files/lines so a session can verify before designing around them.

### 1. Import surface mismatch — biggest single issue
- Donor does `sys.path.insert(0, biom3_dir)` and then
  `from Stage3_source.cond_diff_transformer_layer import get_model` (and
  similar for `Stage1_source.model`, `Stage1_source.preprocess`,
  `Stage3_source.sampling_analysis`, `Stage3_source.animation_tools`).
- Our equivalents live under `biom3.Stage1.*` and `biom3.Stage3.*`. The path
  insert + flat-module import pattern won't work against an installed
  package.
- **Decision needed**: rewrite imports to `biom3.*` (preferred — clean,
  permanent) vs. add a one-time shim that aliases `Stage{1,3}_source` →
  `biom3.Stage{1,3}` (faster, but masks the dependency).

### 2. `batch_generate_denoised_sampled` signature drift
- HF version (donor): returns `(mask_realization_list, time_idx_list)`.
- Our version (`src/biom3/Stage3/sampling_analysis.py`): adds a
  `store_probabilities` kwarg and a third return (`np.ndarray | None`),
  pre-allocates GPU buffers, supports Gumbel-max vs. argmax via
  `token_strategy`. (See CLAUDE.md memory on Stage 3 sampling perf.)
- GRPO only consumes `mask_list[-1]` (`biom3_grpo.py:310`), so the third
  return is harmless — but the *kwargs* the donor passes may not all exist
  in our function. Diff the signatures before swapping.

### 3. Checkpoint format expectations
- Donor loads raw `.bin`/`.pt` from `${BIOM3_DIR}/weights/ProteoScribe/`
  (`biom3_grpo.py:154-176`), with a small Lightning unwrap path (strips
  `model.` prefix from `state_dict`).
- Our Stage 3 saves three formats: raw `.pt`, Lightning `.ckpt`, and
  DeepSpeed ZeRO sharded directories. Loading goes through
  `biom3.Stage3.io.prepare_model_ProteoScribe`, which dispatches by path
  type and calls `convert_zero_checkpoint_to_fp32_state_dict` for shards.
- **Action**: rewire the donor's loader to use
  `prepare_model_ProteoScribe` so any of our checkpoint flavors works as
  the GRPO init.

### 4. Padding / `attention_mask` for the Stage 1 text encoder
- Documented landmine — `docs/bug_reports/bert_embedding_mismatch.md`.
- Whatever code path GRPO uses to obtain `z_c` must call BERT with
  `padding="max_length", max_length=512` and pass the resulting
  `attention_mask`. The donor uses `Stage1_source.preprocess.TextSeqPairing_Dataset`;
  whichever side we end up calling must match training.

### 5. Single-GPU vs. our Lightning + DeepSpeed scaffolding
- `run_grpo.sh` uses plain `python` with `CUDA_VISIBLE_DEVICES`. No
  `torchrun`, no `mpiexec`, no DeepSpeed strategy.
- Our Stage 3 training stack is Lightning + DeepSpeed via
  `scripts/stage3_train_multinode.sh` and machine-specific launchers.
- **Decision needed**: keep GRPO as a standalone single-GPU loop for v1
  (simplest, ships fastest), then later refactor into a `PL_GRPO` Lightning
  module to inherit DeepSpeed/Aurora support. Multi-GPU GRPO is non-trivial
  because the rollout and reward are sequential bottlenecks; per-prompt
  parallelism is the natural axis if/when we scale.
- See memory note `distributed_drop_last.md` — any DataLoader we add for
  prompts must pair `DistributedSampler` with
  `Trainer(use_distributed_sampler=False)` if we go the Lightning route.

### 6. Reference model memory cost
- `copy.deepcopy(model.s3)` doubles Stage 3 GPU memory. ProteoScribe is
  ~86M params, so on a single GPU this is fine, but if we ever scale K or
  the model size we'll need to (a) keep ref on CPU and ship batches,
  (b) recompute ref logp under `torch.no_grad()` from the same weights with
  a periodic snapshot, or (c) use LoRA so the ref *is* the base.

### 7. Reward dependency — ESMFold
- Donor needs `fair-esm==2.0.0` and downloads `facebook/esmfold_v1` on first
  call (~600 MB, plus the ESM-2 backbone). Not currently in our env.
- **Action**: add as an *optional* extras_require (e.g.
  `pip install -e ".[grpo]"`) rather than a hard dep. Gate the import in
  the reward module so users without ESMFold can still smoke-test the
  trainer with a stub reward.

### 8. Config system absence
- Donor is pure CLI. Our convention (CLAUDE.md, "Configuration") is JSON
  configs under `configs/<stage>/` with `_base_configs` composition,
  loaded via `core.helpers.load_json_config()`.
- **Action**: add `configs/grpo/` with a base RL config and a per-experiment
  config; the entry point should call `load_json_config` and let CLI
  override.

### 9. Logging
- Donor writes `train_log.json` by hand. Our stack uses Lightning loggers
  (W&B, TB) with rank-aware filtering (see `2026-03-29_rank-aware-logging.md`).
- **Action**: optional W&B + structured logger from day one; keep the JSON
  log as a fallback.

### 10. PyTorch-Lightning version
- Donor pins `pytorch-lightning==1.9.5`. Our project uses Lightning 2.x in
  most paths (and on Aurora a custom `lightning` symlink — see
  `aurora_env.md` memory). If we keep the donor's training loop, fine; if
  we Lightning-ify it, write against 2.x and DeepSpeed strategy.

### 11. HF repo presence is intermittent
- During this prompt's authoring, `../hugging-face-biom3/` was observed
  both empty (only `.git`) and fully populated with `Stage{1,3}_source/`
  and `weights/`. Confirm state before relying on it; if empty, the
  collaborator's checkout may need re-cloning before any donor-vs-ours
  comparison.

---

## Plan for how to proceed

Phased; each phase ends in a green test/commit so we can stop or hand off.

### Phase 0 — Reconnaissance (no code changes)

1. Confirm `../hugging-face-biom3/` is populated; if not, ask the user to
   re-clone before continuing.
2. Diff `Stage3_source/sampling_analysis.py` (donor) against
   `src/biom3/Stage3/sampling_analysis.py` (ours) — note kwargs and return
   shape deltas.
3. Run the donor's `run_grpo.sh` once on a single Aurora/Polaris GPU
   against `../hugging-face-biom3/` to confirm the *donor* baseline works
   and to capture a reference `train_log.json` for later comparison.

### Phase 1 — Land GRPO as a standalone module under `biom3.rl`

Worktree: `git worktree add .claude/worktrees/grpo -b addison-grpo addison-dev`

1. Add `src/biom3/rl/__init__.py`, `src/biom3/rl/grpo.py` (port of
   `biom3_grpo.py`), `src/biom3/rl/rewards.py` (port of `ESMFoldReward`,
   plus a stub reward for testing without ESMFold).
2. Rewrite all imports to `biom3.*`. Use `prepare_model_ProteoScribe` to
   load the Stage 3 init checkpoint (any format).
3. Wire a console entry point `biom3_grpo_train` in `pyproject.toml` →
   `biom3.rl.__main__:run_grpo_train`. Mirror the existing Stage 3 entry
   pattern.
4. Add `configs/grpo/_base_grpo.json` and an example experiment config.
   Use `core.helpers.load_json_config` for composition + CLI overrides.
5. Make ESMFold an optional dep: add a `[project.optional-dependencies]`
   section in `pyproject.toml` with `grpo = ["fair-esm==2.0.0"]`.
6. Add a tiny test under `tests/rl/test_grpo_smoke.py` that constructs the
   trainer with a stub reward, runs 2 steps with K=2 batch=1 on a tiny
   model, and asserts loss is finite. Mark `@pytest.mark.use_gpu` if
   needed (or write a CPU path with a dummy diffusion forward).

### Phase 2 — Run scaffolding

1. Add `scripts/grpo_train_singlenode.sh` patterned on the existing
   single-node Stage 3 wrapper. Source `environment.sh` (Aurora memory:
   `aurora_env.md`).
2. Add a job template under `jobs/aurora/` (and Polaris if relevant).
   Single-GPU for v1.
3. Document in a new `docs/grpo_finetuning.md`: prerequisites, config
   schema, prompt-file format, output layout, expected runtimes, reward
   semantics.

### Phase 3 — Ergonomics & integration

1. Optional W&B logger plumbed through the GRPO loop, gated by a config
   flag (default off — see `2026-03-30_wandb_default_off.md`).
2. Periodic checkpoint export in our preferred raw `.pt` format so GRPO
   outputs feed straight back into `prepare_model_ProteoScribe` for
   downstream sampling/evaluation.
3. A short eval script that loads a GRPO checkpoint, samples N sequences
   per prompt, and reports pLDDT distribution — sanity-check that the
   reward actually improved.

### Phase 4 (optional, scope creep) — Lightning-ify

Only if multi-GPU GRPO is requested. Wrap `grpo.py` in a `PL_GRPO`
LightningModule, use `DeepSpeedStrategy`, and switch the rollout to be
done per-rank with reduce-scattered group statistics. Non-trivial; defer
until v1 is in use.

---

## Operational workflow

### Worktree (this is a substantive feature)

```
git worktree add .claude/worktrees/grpo -b addison-grpo addison-dev
```

After creating the worktree, populate symlinks if any task needs them:

```
ls data/databases/ data/datasets/ weights/
# If empty:
scripts/sync_weights.sh
```

GRPO needs `weights/Stage3/` populated (and `weights/Stage1/`,
`weights/Stage2/` for the conditioning pipeline). It does *not* need
`data/datasets/` since prompts come from a flat text file.

### Branch / push policy

Per memory `branch_push_policy.md`: push to scratch branches like
`addison-grpo` (or further-scoped `addison-grpo-<topic>`); never push to
`addison-dev` or `main`. The user merges.

### Commits

- Conventional Commits: `feat(rl):`, `feat(grpo):`, `fix(rl):`, `docs:`,
  `test(rl):`.
- One commit per logical step (port → entry point → configs → tests →
  scripts → docs).
- Co-author trailer on every commit.
- Never commit without the user asking.

### Testing

- `pytest tests/ --quick` should still pass (GRPO tests should be
  GPU-gated or stub-reward CPU tests).
- Smoke test: 2 steps, K=2, batch=1, stub reward, on whatever device is
  available. Fast enough to run in CI later.
- Aurora/Polaris real-reward smoke: 5 steps, K=4, batch=1 with ESMFold
  reward — verify pLDDT logs are non-degenerate.

### Session end

Write a session note to `docs/.claude_sessions/<YYYY-MM-DD>_grpo_<slug>.md`
matching the format used by other session notes:

- Summary (1-2 paragraphs).
- Commits list.
- Files changed (one short paragraph each).
- Design notes (especially anything that diverged from this prompt's plan).
- Follow-ups deferred.

---

## First action

After loading the above, **confirm to the user that context is loaded and
ask which phase to start with** (default: Phase 0 reconnaissance, since the
HF repo state is unstable and the donor baseline hasn't been reproduced on
our hardware yet). Do not start porting code until the user confirms which
phase and how aggressive to be on the import-rewrite vs. shim decision.

# 2026-04-28 — GRPO Phase 1 + 2: port collaborator's RL fine-tuner

Working under [docs/.claude_prompts/PROMPT_grpo_integration.md](../.claude_prompts/PROMPT_grpo_integration.md). Single-GPU GRPO RL fine-tuning of Stage 3 ProteoScribe, ported from a collaborator's `../biom3-grpo-rl/biom3_grpo.py` (built against the upstream HF BioM3 layout) into our packaged `biom3.*` module.

Worktree: `.claude/worktrees/grpo`. Branch: `addison-hack-grpo` (renamed mid-session from `addison-grpo` after the user asked to base the work off `addison-hack` — a no-op rename since `addison-hack` and `addison-dev` are at the same commit).

## What landed

### Phase 0 — recon (no code)

Confirmed the donor's algorithm shape (PPO-clip + Schulman $k_3$ KL + group-normalized advantages over $K$ samples per prompt; reward = ESMFold mean pLDDT). Three integration pain points identified before porting:

- **Sampler signature drift.** Donor unpacks `mask_list, _ = batch_generate_denoised_sampled(...)` (HF returns 2-tuple). Our [src/biom3/Stage3/sampling_analysis.py:223](../src/biom3/Stage3/sampling_analysis.py#L223) returns a 3-tuple — would have raised `ValueError: too many values to unpack`.
- **Import surface.** Donor does `sys.path.insert(0, biom3_dir)` then `from Stage{1,3}_source.X import Y`. Rewritten to `from biom3.Stage{1,3}.X import Y`.
- **Checkpoint format assumptions.** Donor only handles raw `.pt` + Lightning unwrap. Routed through [src/biom3/Stage3/io.py:78](../src/biom3/Stage3/io.py#L78)'s `prepare_model_ProteoScribe` so all three formats (raw, Lightning, DeepSpeed sharded) work as Stage 3 init.

`../hugging-face-biom3/` was confirmed populated for source code but the `weights/{ProteoScribe,PenCL,Facilitator}/` directories only have READMEs — actual checkpoints must be `gdown`-ed from Google Drive (READMEs document the IDs). Did **not** reproduce the donor's baseline since that would have required pulling several hundred MB.

### Phase 1 — port (commit `33468bf`)

`feat(rl): add GRPO single-GPU fine-tuner for Stage 3` — 12 files, +876 lines.

```
src/biom3/rl/__init__.py
src/biom3/rl/__main__.py            # biom3_grpo_train wrapper
src/biom3/rl/grpo.py                # GRPOConfig, grpo_train, helpers
src/biom3/rl/rewards.py             # ESMFoldReward, StubReward
src/biom3/rl/io.py                  # frozen S1/S2 + trainable S3 loaders
src/biom3/rl/run_grpo_train.py      # argparse + load_json_config
configs/grpo/_base_grpo.json
configs/grpo/example_grpo.json
configs/grpo/prompts/example_prompts.txt
tests/rl_tests/__init__.py
tests/rl_tests/test_grpo_smoke.py
pyproject.toml                      # +biom3_grpo_train script, +[grpo] extras
```

Design decisions vs. the donor:

- **Dropped the `BioM3ForGRPO(PreTrainedModel)` wrapper.** It pulled `transformers.PreTrainedModel` for no actual benefit (no save/load via HF, no generation API, no config registration that we use). Replaced with plain functions over the existing nn.Modules.
- **Reused our prompt-encoding stack.** `_PromptEncoder` in [src/biom3/rl/grpo.py](../src/biom3/rl/grpo.py) instantiates `Stage1.preprocess.TextSeqPairing_Dataset` once and mutates its lists per call. Keeps `padding="max_length"` / `max_length=512` correct (the documented landmine in `docs/bug_reports/bert_embedding_mismatch.md`) without us reimplementing the BERT tokenization path.
- **Optional ESMFold.** `fair-esm` lives in `[project.optional-dependencies].grpo`, not in the base requires. `ESMFoldReward._load` raises a friendly ImportError pointing at `pip install -e '.[grpo]'`. `StubReward` (deterministic length+diversity score in `[0, 100]`) lets the loop run without ESMFold for tests/debugging.
- **Config composition.** Mirrored the Stage 3 training pattern: `argparse → set_defaults(json) → parse_args` so CLI > JSON > defaults. `_base_grpo.json` includes Stage 1/2/3 inference configs by reference; experiments overlay via `_base_configs`.

### Phase 2 — run scaffolding + docs (this commit)

```
scripts/grpo_train_singlenode.sh           # thin wrapper over biom3_grpo_train
jobs/aurora/_template_grpo_singlenode.pbs  # 1-tile Aurora PBS template
docs/grpo_finetuning.md                    # user-facing how-to
```

The single-node script deliberately does **not** use `mpiexec` — GRPO is single-GPU by design (Phase 4 would Lightning/DeepSpeed-ify it for multi-GPU rollouts). On Aurora, the PBS template pins to one tile via `ZE_AFFINITY_MASK=0`.

`docs/grpo_finetuning.md` covers prerequisites, config schema, prompt format, output layout, both local and PBS invocation, the testing recipe, and the running list of "watch out for" gotchas (editable-install drift, BERT padding, `strict=False` weight load, reference-model memory cost, single-GPU-only constraint).

## Verification

CPU smoke ran clean in the user's `biom3-env` after exporting `PYTHONPATH=src` (the editable install was pointing at a sibling worktree — same gotcha called out in `PROMPT_webapp_development.md`):

```
PYTHONPATH=src python -m pytest tests/rl_tests/ -v
```

Tests cover: token vocab constants, `load_prompts` parsing (comments + `name|text`), `decode_tokens` special-token stripping, `StubReward` range, `build_reward` dispatch, `diffusion_rollout` shape/dtype on the mini Stage 3 fixture, `_policy_logprobs` shape + non-positivity, and an end-to-end inner GRPO update that exercises rollout → reward → group-norm advantage → PPO-clip + Schulman KL → AdamW step and asserts at least one parameter changed.

**Did NOT run** the full real-reward smoke (5 steps, K=4, B=1, ESMFold) — the user has not yet confirmed which Stage 3 init checkpoint to use, and ESMFold pulls hundreds of MB.

## Things to look back into

1. **Real-reward smoke.** Run `biom3_grpo_train` with `--reward esmfold_plddt` on Aurora (1 tile, 5 steps) once the user picks a Stage 3 init — likely `weights/ProteoScribe/state_dict.best.pth` or whatever the most recent finetune produced. Validates the import rewrite end-to-end and gives a baseline pLDDT trajectory.
2. **Phase 3 — ergonomics.** Optional W&B logger (gated, default-off per `2026-03-30_wandb_default_off.md`). The current trainer writes JSON-only.
3. **Phase 3 — eval script.** Load a GRPO checkpoint, sample N sequences per prompt, plot pLDDT distribution. Sanity check that the reward actually improved.
4. **Phase 4 — Lightning-ify (deferred).** Multi-GPU GRPO is non-trivial because rollout + reward are sequential bottlenecks; per-prompt parallelism is the natural axis. Wait for v1 to be in use before scoping.
5. **`copy.deepcopy` reference.** Doubles Stage 3 GPU memory. Fine at 86M params on a single Aurora tile, revisit if the model grows. LoRA on the policy with the base as reference is the cleaner long-term option.
6. **Polaris launcher.** The Aurora job template is the only one written. Polaris-specific PBS template would be a paste-and-edit job (no `ZE_AFFINITY_MASK`; use `CUDA_VISIBLE_DEVICES`).
7. **Donor baseline reproduction.** Skipped in this session because HF weights are gdown-only. Worth doing once the weights are local — establishes a known-good train_log to diff against ours.

## Operational notes

- The worktree's editable install points at a sibling, so `biom3.rl` won't import from there. `pip install -e .` from this worktree to make it permanent (affects all worktrees), or `PYTHONPATH=src` per-invocation.
- Branch policy: stays on `addison-hack-grpo` — user merges to `addison-dev` themselves per `branch_push_policy.md` memory.

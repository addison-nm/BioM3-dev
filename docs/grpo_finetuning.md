# GRPO fine-tuning of Stage 3

GRPO (Group Relative Policy Optimization) RL fine-tuning of the Stage 3
ProteoScribe diffusion model. Single-GPU only in this revision; multi-GPU
is deferred (Phase 4 in
[docs/.claude_prompts/PROMPT_grpo_integration.md](.claude_prompts/PROMPT_grpo_integration.md)).

## What it does

For each text prompt, sample K candidate protein sequences from the
diffusion policy, score each with ESMFold mean pLDDT, and update the
policy with PPO-clip + Schulman $k_3$ KL against a frozen reference
snapshot, using group-normalized advantages.

The math is laid out in
[PROMPT_grpo_integration.md](.claude_prompts/PROMPT_grpo_integration.md);
see "Equations".

## Layout

```
src/biom3/rl/
  __init__.py
  __main__.py            # biom3_grpo_train entry-point wrapper
  grpo.py                # GRPOConfig, grpo_train, sampling + log-prob helpers
  rewards.py             # ESMFoldReward, StubReward, build_reward
  io.py                  # frozen Stage 1/2 loaders + trainable Stage 3
  run_grpo_train.py      # argparse + load_json_config composition

configs/grpo/
  _base_grpo.json        # base config (no weights set)
  example_grpo.json      # 100-step example overlay
  prompts/example_prompts.txt

scripts/grpo_train_singlenode.sh
jobs/aurora/_template_grpo_singlenode.pbs

tests/rl_tests/test_grpo_smoke.py
```

## Prerequisites

- Stage 1 (PenCL) weights — for the frozen text encoder.
- Stage 2 (Facilitator) weights — for the frozen text→protein bridge.
- Stage 3 (ProteoScribe) init weights — any of: raw `.pt`/`.bin`,
  Lightning `.ckpt`, or a DeepSpeed sharded directory. Loaded through
  `biom3.Stage3.io.prepare_model_ProteoScribe`, which handles all three.
- ESMFold (`facebook/esmfold_v1`) — only if `reward = "esmfold_plddt"`.
  Install the optional extras:

  ```bash
  pip install -e '.[grpo]'
  ```

  Use `reward = "stub"` (the deterministic length+diversity reward in
  `biom3.rl.rewards.StubReward`) if you want to drive the loop without
  ESMFold (smoke testing, debugging).

## Configuration

GRPO uses the standard `core.helpers.load_json_config` composition.
`configs/grpo/_base_grpo.json` wires Stage 1/2/3 inference configs as
defaults; experiment configs overlay it via `_base_configs`. Precedence
(low → high) is `_base_configs` < current file < `_overwrite_configs` < CLI.

Required fields (must end up resolved by the time the trainer starts):

| Field | What |
|---|---|
| `stage1_config` | Path to PenCL JSON config |
| `stage2_config` | Path to Facilitator JSON config |
| `stage3_config` | Path to ProteoScribe JSON config |
| `stage1_weights` | PenCL `.bin`/`.pt` |
| `stage2_weights` | Facilitator `.bin`/`.pt` |
| `stage3_init_weights` | ProteoScribe init (any supported format) |
| `prompts_path` | UTF-8 text file, one prompt per line |

GRPO hyperparameters (defaults shown):

| Field | Default | Notes |
|---|---|---|
| `steps` | 200 | total gradient updates |
| `num_generations` | 4 | $K$ — sequences per prompt |
| `batch_size` | 1 | prompts per gradient update |
| `learning_rate` | 1e-5 | AdamW |
| `weight_decay` | 1e-6 | AdamW |
| `beta` | 0.01 | KL coefficient $\beta$ |
| `eps` | 0.20 | PPO clip $\epsilon$ |
| `max_grad_norm` | 1.0 | grad clip |
| `save_steps` | 50 | checkpoint cadence |
| `seed` | 42 | torch + numpy |
| `reward` | `esmfold_plddt` | `esmfold_plddt` \| `stub` |

## Prompt file format

Plain UTF-8, one prompt per line. Empty lines and lines starting with
`#` (after strip) are ignored. The `name|text` form is accepted; only
the `text` part is used. See `configs/grpo/prompts/example_prompts.txt`.

## Running locally (single GPU)

```bash
source environment.sh
./scripts/grpo_train_singlenode.sh \
    configs/grpo/example_grpo.json \
    my_grpo_run_001 \
    xpu \
    --stage1_weights weights/PenCL/BioM3_PenCL_epoch20.bin \
    --stage2_weights weights/Facilitator/BioM3_Facilitator_epoch20.bin \
    --stage3_init_weights weights/ProteoScribe/state_dict.best.pth
```

`xpu` for Aurora; use `cuda` on Polaris/Spark and `cpu` only for very
small smoke runs (the diffusion rollout is the dominant cost).

On Aurora, pin to a specific tile:

```bash
ZE_AFFINITY_MASK=0 ./scripts/grpo_train_singlenode.sh ...
```

CLI args override JSON config; JSON config overrides argparse defaults.

## Submitting on Aurora (PBS)

Copy the template, replace the placeholders, submit:

```bash
cp jobs/aurora/_template_grpo_singlenode.pbs jobs/aurora/job_grpo_<run>.pbs
# Edit <JOB_NAME>, <QUEUE>, <CONFIG_NAME>, <STAGE3_WEIGHTS_PATH>.
qsub jobs/aurora/job_grpo_<run>.pbs
```

The template requests `select=1`, pins to tile 0 via `ZE_AFFINITY_MASK`,
and tees output to `logs/${run_id}.${PBS_JOBID}.o`.

## Output layout

```
{output_root}/{run_id}/
├── step50.pt            # {"step": 50, "model_state": ...}
├── step100.pt
├── ...
├── final.pt
└── train_log.json       # one row per step: reward, loss, pg, kl, clip_frac, avg_len
```

Resulting `final.pt` is a raw Stage 3 state-dict and feeds straight back
into `biom3.Stage3.io.prepare_model_ProteoScribe(model_fpath=...)` for
downstream sampling/evaluation via `biom3_ProteoScribe_sample`.

## Testing

```bash
# CPU smoke (no weights, no ESMFold, no GPU):
PYTHONPATH=src pytest tests/rl_tests/ -v
```

Covers token vocabulary constants, prompt parsing, decoding, stub
reward range, sampler 3-tuple unpacking, policy log-prob shape, and an
end-to-end inner GRPO update on the mini Stage 3 fixture
(`tests/_data/models/stage3/weights/minimodel1_ds128_weights1.pth`).

## Watch out for

- **Editable install drift.** `biom3.__file__` may resolve to a sibling
  worktree. If `import biom3.rl` fails after a clean install, either
  `PYTHONPATH=src` for the run, or `pip install -e .` from this tree
  (note: re-installing affects sibling worktrees too).
- **BERT padding.** Stage 1 must use `padding="max_length"`,
  `max_length=512`. The `_PromptEncoder` in
  [src/biom3/rl/grpo.py](../src/biom3/rl/grpo.py) routes through
  `Stage1.preprocess.TextSeqPairing_Dataset` which already does this —
  don't bypass it.
- **`strict=False` on weight load.** Inherited from the donor and the
  rest of the codebase. Missing/extra keys will be silently ignored;
  if you change ProteoScribe architecture, check the loader's missing
  keys list before trusting a run.
- **Reference model memory.** GRPO snapshots a frozen ProteoScribe via
  `copy.deepcopy`, doubling Stage 3 GPU memory. Fine for ~86M params on
  a single Aurora tile / H100; revisit if the model grows.
- **Single-GPU only.** No DDP, no DeepSpeed in the trainer. Don't try
  to launch via `mpiexec` — `grpo_train_singlenode.sh` deliberately
  doesn't.

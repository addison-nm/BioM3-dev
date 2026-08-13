# Session: Multidomain training on Aurora multinode

**Date:** 2026-08-13
**Branch:** `dev` (commits `3a12168`, `6aafef5`)

First real multinode runs of `biom3_finetune_multidomain` on Aurora. Three fixes,
two of them in code. Training now runs on 2 nodes x 12 tiles (24 ranks) under
DeepSpeed ZeRO-2 at bf16.

## What was broken

**1. fp32 conditioning into bf16 weights** (`model.py`, fixed in `6aafef5`).
`_embed_domain` cast the time embedding to the parameter dtype and the axial
positional embedding to match `x_e`, but fed `y_d` to `y_mlp` uncast. Under
DeepSpeed the parameters are bf16 while `y_c` arrives fp32 from the dataloader,
so `F.linear` raised `mat1 and mat2 must have the same dtype`. Invisible on a
single device, where the strategy resolves to `auto` and autocast is then
disabled on XPU, leaving everything fp32.

**2. Stock `ModelCheckpoint`s** (`trainer.py`, fixed in `6aafef5`).
`build_callbacks` built its own, reintroducing two defects Stage 3's callbacks
already solve: Lightning's `reduce_boolean_decision` is an integer SUM
all-reduce that returns wrong values on XPU/CCL, and a periodic checkpoint
sharing the monitored callback's `dirpath` is silently pruned by `save_top_k`.
Now delegates to `build_checkpoint_callbacks(..., use_sync_safe=BACKEND_NAME == _XPU)`.
Its `enable_version_counter=False` also stops the `-v1` duplicate that a
colliding `last.ckpt` write produces. **Stage 3 source is untouched.**

**3. `num_workers: 4` OOM-kills the DataLoader workers** (config, not fixed in
code). This is the one that cost the most time: it presented three different
ways — a truncated checkpoint, a bare SIGKILL, and a stall — before a run
survived long enough to surface `RuntimeError: DataLoader worker (pid ...) is
killed by signal: Killed`. `lazy_records` defaults `False`, so every worker
holds the whole materialised corpus; at 12 ranks x 4 workers that is 48 copies
per node. Every working Stage 3 multinode run uses `num_workers: 0`, and passing
that fixes it. `--lazy_records True` is the fix that would make workers usable
again; it has not been tried.

Also merged from `addison-dev` (`3a12168`): Stage 2 batched inference
(`--batch_size` / `--stage2_batch_size`) and a `TimeLimitCallback` that checks
the deadline on the epoch's last batch. The latter matters here — multidomain at
a 3072 global batch is ~10 steps/epoch, well under `check_every_n_steps=50`, so
`--time_limit` was never evaluated at all.

## What is verified

- `--audit_only` passes: `Additive-null gate passed (bit-exact over 2 domains)`,
  `Freeze audit passed: coupling=33603584 experts=0 other=0`.
- 1 node / 12 ranks: 19 epochs in 1 h, no worker kills.
- 2 nodes / 24 ranks: past the failure point, checkpoints and `periodic/` clean.
- Corpus: 30,563 two-domain records; stratified cluster split reproduces the
  reference exactly at 21,393 / 6,113 / 3,057.

## Known-good PBS script

```bash
#!/bin/bash -l
#PBS -A NLDesignProtein
#PBS -N md_train_luc_v1_n2
#PBS -l walltime=01:00:00
#PBS -l select=2
#PBS -l place=scatter
#PBS -l filesystems=home:flare
#PBS -q debug-scaling
#PBS -j oe
#PBS -o /flare/NLDesignProtein/ahowe/biom3-workspaces/luciferase_v1/logs/

repo_root=/lus/flare/projects/NLDesignProtein/ahowe/BioM3-dev-space/BioM3-dev
workspace=/flare/NLDesignProtein/ahowe/biom3-workspaces/luciferase_v1

cd ${repo_root}

module load frameworks
source ./venvs/biom3-env/bin/activate

# Configurations to edit
config_path=configs/stage3_multidomain/finetune_multidomain_luciferase_v1.json
run_id=luciferase_v1_md_n2
num_nodes=2                     # Must match #PBS -l select above
batch_size=4                    # PER-RANK batch. Global = 4*2*12 = 96.
epochs=200
checkpoint_every_n_epochs=5
num_workers=0                   # 0, matching every working Stage 3 multinode run.
                                # The config's 4 OOM-killed the DataLoader workers:
                                # lazy_records defaults False, so each of 12 ranks x 4
                                # workers holds the whole materialised corpus.
output_root=${workspace}/outputs/multidomain

# Constant configurations
num_devices=12                  # tiles per Aurora node

mkdir -p ${workspace}/logs ${output_root}
log_fpath=${workspace}/logs/md_train_n${num_nodes}.${PBS_JOBID}.o

source environment.sh

# Read by scripts/launchers/aurora_multinode.sh. PBS sets PBS_NODEFILE.
# WORLD_SIZE is deliberately NOT exported: the Stage 3 multinode wrapper does not
# set it either, and Lightning does its own cluster-environment detection.
export NGPU_PER_NODE=${num_devices}
export NGPU_TOTAL=$((num_nodes * num_devices))

./scripts/launchers/aurora_multinode.sh \
    biom3_finetune_multidomain \
    --config_path ${config_path} \
    --run_id ${run_id} \
    --output_root ${output_root} \
    --num_nodes ${num_nodes} \
    --devices_per_node ${num_devices} \
    --batch_size ${batch_size} \
    --epochs ${epochs} \
    --checkpoint_every_n_epochs ${checkpoint_every_n_epochs} \
    --num_workers ${num_workers} \
> ${log_fpath} 2>&1
```

Notes on the launch, since Aurora is unforgiving here: there is no multidomain
equivalent of `stage3_train_multinode.sh` (it hardcodes `biom3_train_stage3` and
passes `--device`, which this runner does not accept), so
`scripts/launchers/aurora_multinode.sh` is wrapped directly — the same pattern
the embedding pipeline uses. That launcher is mpiexec, not torchrun; torchrun is
the container path only. Everything Aurora-specific (`CCL_ATL_SYNC_COLL`,
`CCL_WORKER_AFFINITY` pinned to the launcher's `--cpu-bind` ranges, `TMPDIR`,
`CCL_ZE_CACHE_GET_IPC_HANDLES_THRESHOLD`) comes from `source environment.sh`,
sourced before the launcher exactly as the Stage 3 jobs do.

## Open for whoever builds generation

- **Generation (Phase 4) does not exist.** No composed sampler, so a trained
  multidomain checkpoint cannot produce sequences. Per the phase-1 design that
  is a fixed-length two-grid OA-ARDM decode with the protein being the string
  concatenation of the per-canvas decodes.
- **Coupling loss is flat.** 19 epochs single-node moved val_loss between 0.633
  and 0.643 with no trend. Two candidates before concluding the coupling does
  not help: `sampler.set_epoch()` is never called (inherited from Stage 3's
  `PL_wrapper`), so every epoch replays an identical batch order and per-rank
  shard; and `lr=2.5e-05` may be small for parameters starting at an exact zero
  init. `expert_delta_norms` would show whether the coupling is moving at all.
- **`find_resume_checkpoint` only looks at `last.ckpt`**, which tracks save
  events and so lags the newest epoch whenever val_loss stops improving. With
  periodic snapshots now in `periodic/`, a resumed run can silently restart
  several epochs back.
- **Experts are still not converged.** The pair in use scored 0.7021 (PF00501,
  72 epochs) and 0.1089 (PF13193, 100 epochs); 1000-epoch runs at an unscaled lr
  actually scored *worse* (0.7121 / 0.1109), and PF13193 overfit from epoch 292.
  Nothing here says whether coupling helps until the experts are right.

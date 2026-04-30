#!/usr/bin/env bash
#=============================================================================
#
# FILE: run_gdpo_smoke_multixpu.sh
#
# Multi-XPU smoke for the GDPO trainer. Replicates the trainable Stage 3
# policy onto every visible Aurora tile and runs G rollouts in parallel
# via a thread pool (gradient updates stay on xpu:0). Use this to
# validate the RolloutPool plumbing end-to-end before submitting the
# full production_v02 multi-tile job.
#
# Differs from run_gdpo_smoke_5steps.sh in two places:
#   - ZE_AFFINITY_MASK is unset so torch.xpu.device_count() returns 6
#     (or however many tiles your interactive shell was allocated).
#   - --rollout_devices auto is passed to the trainer.
#
# Assumes: cwd = BioM3-dev root; environment.sh has been sourced;
# biom3-env active; pip install -e '.[grpo]' done; ESMFold reachable.
#
# USAGE:
#   ./scripts/run_gdpo_smoke_multixpu.sh                 # logs to logs/<run_id>.o
#   ./scripts/run_gdpo_smoke_multixpu.sh --no-log        # stream to stdout
#
#=============================================================================
set -euo pipefail

# Configurations to edit
config_path=./configs/grpo/example_gdpo.json
steps=5
num_generations=24              # G — bumped from 4 so 6 tiles each get 2 sequences
batch_size=1
beta=0.01
eps=0.20
n_quadrature=2
inner_mc=1
kl_estimator=tokenwise_k3
reward=esmfold_plddt
prompts_path=./configs/grpo/prompts/example_prompts.txt
stage1_weights=./weights/PenCL/PenCL_V09152023_last.ckpt
stage2_weights=./weights/Facilitator/Facilitator_MMD15.ckpt/last.ckpt
stage3_init_weights=./weights/ProteoScribe/ProteoScribe_SH3_epoch52.ckpt/single_model.pth

device=xpu

# IMPORTANT: do NOT pin ZE_AFFINITY_MASK here — we want the process to
# see all tiles via torch.xpu.device_count().
unset ZE_AFFINITY_MASK || true

# Run ID
datetime=$(date +%Y%m%d_%H%M%S)
config_name=$(basename "${config_path}" .json)
run_id=${config_name}_smoke_multixpu_G${num_generations}_b${batch_size}_N${n_quadrature}_s${steps}_V${datetime}

cmd=(
    ./scripts/gdpo_train_singlenode.sh
    "${config_path}"
    "${run_id}"
    "${device}"
    --steps "${steps}"
    --num_generations "${num_generations}"
    --batch_size "${batch_size}"
    --beta "${beta}"
    --eps "${eps}"
    --n_quadrature "${n_quadrature}"
    --inner_mc "${inner_mc}"
    --kl_estimator "${kl_estimator}"
    --rollout_devices auto
    --save_steps "${steps}"
    --reward "${reward}"
    --prompts_path "${prompts_path}"
    --stage1_weights "${stage1_weights}"
    --stage2_weights "${stage2_weights}"
    --stage3_init_weights "${stage3_init_weights}"
)

if [ "${1:-}" = "--no-log" ]; then
    echo "Running interactively (stdout). run_id=${run_id}"
    exec "${cmd[@]}"
else
    mkdir -p logs
    log_fpath=logs/${run_id}.o
    echo "Logging to ${log_fpath} (run_id=${run_id})"
    exec "${cmd[@]}" #> "${log_fpath}" 2>&1
fi

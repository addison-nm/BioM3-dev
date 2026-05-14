#!/usr/bin/env bash
#=============================================================================
#
# FILE: run_gdpo_smoke_5steps.sh
#
# Interactive smoke run for the GDPO trainer (sibling of GRPO).
# 5-step / G=4 / B=1 / N_quadrature=2 / inner_mc=1 against the matched
# .ckpt set:
#   Stage 1  PenCL_V09152023_last.ckpt
#   Stage 2  Facilitator_MMD15.ckpt
#   Stage 3  ProteoScribe_SH3_epoch52.ckpt
#
# Assumes:
#   - cwd is the BioM3-dev root
#   - environment.sh has been sourced (BIOM3_MACHINE set,
#     TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1, ONEAPI_DEVICE_SELECTOR, etc.)
#   - biom3-env is active and `pip install -e '.[grpo]'` has been done
#   - facebook/esmfold_v1 is reachable (cached or downloadable)
#
# USAGE:
#   ./scripts/run_gdpo_smoke_5steps.sh                 # logs to logs/<run_id>.o
#   ./scripts/run_gdpo_smoke_5steps.sh --no-log        # stream to stdout
#
#=============================================================================
set -euo pipefail

# Configurations to edit
config_path=./configs/grpo/example_gdpo.json
steps=5
num_generations=4
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

# Pin to one Aurora tile
export ZE_AFFINITY_MASK=0

# Run ID
datetime=$(date +%Y%m%d_%H%M%S)
config_name=$(basename "${config_path}" .json)
run_id=${config_name}_smoke_G${num_generations}_b${batch_size}_N${n_quadrature}_s${steps}_V${datetime}

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

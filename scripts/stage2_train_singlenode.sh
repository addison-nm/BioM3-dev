#!/usr/bin/env bash
#=============================================================================
#
# FILE: stage2_train_singlenode.sh
#
# USAGE: stage2_train_singlenode.sh CONFIG_PATH NGPU DEVICE \
#        WANDB_API_KEY RUN_ID [additional --key value overrides]
#
# DESCRIPTION: Single-node wrapper for Stage 2 Facilitator training. Launches
#   biom3_pretrain_stage2 directly (no mpiexec). The JSON config file provides
#   model/training hyperparameters; per-job overrides (epochs, data paths, etc.)
#   are passed as additional CLI arguments via "$@".
#
#=============================================================================

if [ "$#" -lt 5 ]; then
  echo "Usage: $0 CONFIG_PATH NGPU DEVICE WANDB_API_KEY RUN_ID [--key value ...]"
  exit 1
fi

config_path=$1
NGPU=$2
device=$3
wandb_api_key=$4
run_id=$5
shift 5

echo "Single-node: NGPU=${NGPU} (${device})"

export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=true

if [ -z "$wandb_api_key" ]; then
    echo "WARNING: WANDB_API_KEY is empty — disabling wandb logging"
    wandb_override="--wandb False"
else
    export WANDB_API_KEY=$wandb_api_key
    wandb_override=""
fi

biom3_pretrain_stage2 \
    --config_path ${config_path} \
    --run_id ${run_id} \
    --device ${device} \
    --num_nodes 1 \
    --gpu_devices ${NGPU} \
    ${wandb_override} \
    "$@"

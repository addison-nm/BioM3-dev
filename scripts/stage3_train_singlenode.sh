#!/usr/bin/env bash
#=============================================================================
#
# FILE: stage3_train_singlenode.sh
#
# USAGE: stage3_train_singlenode.sh CONFIG_PATH NGPU DEVICE \
#        RUN_ID [additional --key value overrides]
#
# DESCRIPTION: Single-node wrapper for Stage 3 training (pretraining and
#   finetuning). Launches biom3_pretrain_stage3 directly (no mpiexec).
#   The JSON config file provides model/training hyperparameters; per-job
#   overrides (epochs, resume, finetune flags, etc.) are passed as additional
#   CLI arguments via "$@".
#
#   WANDB_API_KEY is read from the environment. If unset/empty, wandb
#   logging is disabled.
#
#=============================================================================

if [ "$#" -lt 4 ]; then
  echo "Usage: $0 CONFIG_PATH NGPU DEVICE RUN_ID [--key value ...]"
  echo "WANDB_API_KEY is read from the environment (wandb disabled if unset)."
  exit 1
fi

config_path=$1
NGPU=$2
device=$3
run_id=$4
shift 4

echo "Single-node: NGPU=${NGPU} ($device)"

export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=true

if [ -z "${WANDB_API_KEY:-}" ]; then
    echo "WARNING: WANDB_API_KEY is empty — disabling wandb logging"
    wandb_override="--wandb False"
else
    wandb_override=""
fi

biom3_pretrain_stage3 \
    --config_path ${config_path} \
    --run_id ${run_id} \
    --device ${device} \
    --num_nodes 1 \
    --gpu_devices ${NGPU} \
    ${wandb_override} \
    "$@"

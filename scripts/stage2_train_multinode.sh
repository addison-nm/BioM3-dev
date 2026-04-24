#!/usr/bin/env bash
#=============================================================================
#
# FILE: stage2_train_multinode.sh
#
# USAGE: stage2_train_multinode.sh CONFIG_PATH NRANKS NGPU_PER_RANK \
#        DEVICE RUN_ID [additional --key value overrides]
#
# DESCRIPTION: Multi-node wrapper for Stage 2 Facilitator training. Launches
#   biom3_pretrain_stage2 across all ranks via mpiexec. The JSON config
#   provides model/training hyperparameters; per-job overrides (epochs,
#   data paths, etc.) are passed as additional CLI arguments via "$@".
#   Mirrors scripts/stage1_train_multinode.sh.
#
#   WANDB_API_KEY is read from the environment. If unset/empty, wandb
#   logging is disabled.
#
#=============================================================================

if [ "$#" -lt 5 ]; then
  echo "Usage: $0 CONFIG_PATH NRANKS NGPU_PER_RANK DEVICE RUN_ID [--key value ...]"
  echo "WANDB_API_KEY is read from the environment (wandb disabled if unset)."
  exit 1
fi

config_path=$1
NRANKS=$2
NGPU_PER_RANK=$3
device=$4
run_id=$5
shift 5

NGPUS="$((NRANKS * NGPU_PER_RANK))"

echo "NRANKS: ${NRANKS}, NGPU_PER_RANK: ${NGPU_PER_RANK}, NGPUS: ${NGPUS} ($device)"

export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=true

if [ -z "${WANDB_API_KEY:-}" ]; then
    echo "WARNING: WANDB_API_KEY is empty — disabling wandb logging"
    wandb_override="--wandb False"
else
    wandb_override=""
fi

mpiexec \
    --verbose \
    --envall \
    -n "${NGPUS}" \
    --ppn "${NGPU_PER_RANK}" \
    --hostfile="${PBS_NODEFILE}" \
    biom3_pretrain_stage2 \
        --config_path ${config_path} \
        --run_id ${run_id} \
        --device ${device} \
        --num_nodes ${NRANKS} \
        --gpu_devices ${NGPU_PER_RANK} \
        ${wandb_override} \
        "$@"

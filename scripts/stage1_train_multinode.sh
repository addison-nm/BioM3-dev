#!/usr/bin/env bash
#=============================================================================
#
# FILE: stage1_train_multinode.sh
#
# USAGE: stage1_train_multinode.sh CONFIG_PATH NRANKS NGPU_PER_RANK \
#        DEVICE WANDB_API_KEY RUN_ID [additional --key value overrides]
#
# DESCRIPTION: Multi-node wrapper for Stage 1 PenCL training. Launches
#   biom3_pretrain_stage1 across all ranks via mpiexec. The JSON config
#   provides model/training hyperparameters; per-job overrides (epochs,
#   dataset_type, pfam_data_path, etc.) are passed as additional CLI
#   arguments via "$@". Mirrors scripts/stage3_train_multinode.sh.
#
#=============================================================================

if [ "$#" -lt 6 ]; then
  echo "Usage: $0 CONFIG_PATH NRANKS NGPU_PER_RANK DEVICE WANDB_API_KEY RUN_ID [--key value ...]"
  exit 1
fi

config_path=$1
NRANKS=$2
NGPU_PER_RANK=$3
device=$4
wandb_api_key=$5
run_id=$6
shift 6

NGPUS="$((NRANKS * NGPU_PER_RANK))"

echo "NRANKS: ${NRANKS}, NGPU_PER_RANK: ${NGPU_PER_RANK}, NGPUS: ${NGPUS} ($device)"

export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=true

if [ -z "$wandb_api_key" ]; then
    echo "WARNING: WANDB_API_KEY is empty — disabling wandb logging"
    wandb_override="--wandb False"
else
    export WANDB_API_KEY=$wandb_api_key
    wandb_override=""
fi

mpiexec \
    --verbose \
    --envall \
    -n "${NGPUS}" \
    --ppn "${NGPU_PER_RANK}" \
    --hostfile="${PBS_NODEFILE}" \
    biom3_pretrain_stage1 \
        --config_path ${config_path} \
        --run_id ${run_id} \
        --device ${device} \
        --num_nodes ${NRANKS} \
        --gpu_devices ${NGPU_PER_RANK} \
        ${wandb_override} \
        "$@"

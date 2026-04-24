#!/usr/bin/env bash
#=============================================================================
#
# FILE: stage1_train_singlenode.sh
#
# USAGE: stage1_train_singlenode.sh CONFIG_PATH NGPU DEVICE \
#        RUN_ID [additional --key value overrides]
#
# DESCRIPTION: Single-node wrapper for Stage 1 PenCL training. Launches
#   biom3_pretrain_stage1 across NGPU ranks via mpiexec. The JSON config
#   file provides model/training hyperparameters; per-job overrides (epochs,
#   dataset_type, pfam_data_path, etc.) are passed as additional CLI arguments
#   via "$@".
#
#   WANDB_API_KEY is read from the environment. If unset/empty, wandb
#   logging is disabled.
#
#   On Aurora, launching via mpiexec is required so oneCCL uses the MPI
#   transport + pmix bootstrap that ALCF documents and tests. Launching
#   directly (no mpiexec) forces CCL to fall back to OFI/ATL, which is
#   untested at any scale and has produced collective deadlocks.
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

echo "Single-node: NGPU=${NGPU} (${device})"

export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=true

if [ -z "${WANDB_API_KEY:-}" ]; then
    echo "WARNING: WANDB_API_KEY is empty — disabling wandb logging"
    wandb_override="--wandb False"
else
    wandb_override=""
fi

# ALCF-recommended per-rank CPU binding for a 12-tile Aurora node. Each rank
# gets 4 framework cores, disjoint from the CCL worker cores pinned by
# CCL_WORKER_AFFINITY in environment.sh. Tile-topology-specific; if NGPU != 12
# the list below must be adjusted (see ALCF oneCCL.md §worker affinity).
CPU_BIND="verbose,list:4-7:8-11:12-15:16-19:20-23:24-27:56-59:60-63:64-67:68-71:72-75:76-79"

mpiexec \
    --verbose \
    --envall \
    -n "${NGPU}" \
    --ppn "${NGPU}" \
    --cpu-bind ${CPU_BIND} \
    biom3_pretrain_stage1 \
        --config_path ${config_path} \
        --run_id ${run_id} \
        --device ${device} \
        --num_nodes 1 \
        --gpu_devices ${NGPU} \
        ${wandb_override} \
        "$@"

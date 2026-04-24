#!/usr/bin/env bash
#=============================================================================
#
# FILE: stage2_train_singlenode.sh
#
# USAGE: stage2_train_singlenode.sh CONFIG_PATH NGPU DEVICE \
#        RUN_ID [additional --key value overrides]
#
# DESCRIPTION: Single-node wrapper for Stage 2 Facilitator training. Launches
#   biom3_pretrain_stage2 across NGPU ranks via mpiexec. The JSON config
#   file provides model/training hyperparameters; per-job overrides (epochs,
#   data paths, etc.) are passed as additional CLI arguments via "$@".
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

# ALCF-canonical per-rank CPU binding for a 12-tile Aurora node. Each rank
# gets an 8-core slot adjacent to the GPU it drives:
#   ranks  0,1 -> GPU 0 (cores  1-8 / 9-16)   socket 0
#   ranks  2,3 -> GPU 1 (cores 17-24 / 25-32) socket 0
#   ranks  4,5 -> GPU 2 (cores 33-40 / 41-48) socket 0
#   ranks  6,7 -> GPU 3 (cores 53-60 / 61-68) socket 1
#   ranks  8,9 -> GPU 4 (cores 69-76 / 77-84) socket 1
#   ranks 10,11-> GPU 5 (cores 85-92 / 93-100) socket 1
# Cores 0/52 are reserved for the OS; the gap 49-52 covers core-52 + buffer.
# CCL progress threads land on the last core of each rank's range
# (8,16,...,100) via CCL_WORKER_AFFINITY in environment.sh. NGPU must = 12.
CPU_BIND="verbose,list:1-8:9-16:17-24:25-32:33-40:41-48:53-60:61-68:69-76:77-84:85-92:93-100"

mpiexec \
    --verbose \
    --envall \
    -n "${NGPU}" \
    --ppn "${NGPU}" \
    --cpu-bind ${CPU_BIND} \
    biom3_pretrain_stage2 \
        --config_path ${config_path} \
        --run_id ${run_id} \
        --device ${device} \
        --num_nodes 1 \
        --gpu_devices ${NGPU} \
        ${wandb_override} \
        "$@"

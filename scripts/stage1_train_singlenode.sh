#!/usr/bin/env bash
#=============================================================================
#
# FILE: stage1_train_singlenode.sh
#
# USAGE: stage1_train_singlenode.sh CONFIG_PATH NGPU DEVICE \
#        RUN_ID [additional --key value overrides]
#
# DESCRIPTION: Single-node wrapper for Stage 1 PenCL training. Dispatches
#   to the machine-specific launcher under
#   scripts/launchers/${BIOM3_MACHINE}_singlenode.sh.
#
#   The JSON config provides model/training hyperparameters; per-job
#   overrides (epochs, dataset_type, pfam_data_path, etc.) are passed via
#   "$@". WANDB_API_KEY is read from the environment (wandb disabled if unset).
#
#   Requires: source environment.sh first so BIOM3_MACHINE is set.
#
#=============================================================================
set -euo pipefail

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

# Resolve machine-specific launcher
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MACHINE="${BIOM3_MACHINE:?BIOM3_MACHINE not set; source environment.sh first}"
LAUNCHER="${SCRIPT_DIR}/launchers/${MACHINE}_singlenode.sh"

if [ ! -x "${LAUNCHER}" ]; then
    echo "ERROR: no launcher for ${MACHINE} singlenode at ${LAUNCHER}"
    exit 1
fi

# Export args the launcher reads from env
export NGPU

exec "${LAUNCHER}" \
    biom3_pretrain_stage1 \
        --config_path "${config_path}" \
        --run_id "${run_id}" \
        --device "${device}" \
        --num_nodes 1 \
        --gpu_devices "${NGPU}" \
        ${wandb_override} \
        "$@"

#!/usr/bin/env bash
#=============================================================================
#
# FILE: scripts/cloud/pretrain.sh
#
# Stage-3 ProteoScribe pretraining FROM SCRATCH in the BioM3 container.
# Env-var driven; baked into the image. Trains directly on a precompiled
# Stage-2 embedding HDF5 — no embedding step, no init weights.
#
# INPUTS:
#   PRIMARY_HDF5        precompiled Stage-2 embedding dataset (required)
#   PRETRAIN_CONFIG     Stage-3 training config
#                       (default configs/stage3_training/pretrain_scratch_v1.json)
#
# RUN / OUTPUT:
#   RUN_ID              run identifier (default: auto, HPC-style naming)
#   DEVICE              cpu | cuda | xpu (default: auto-detected backend)
#   NGPU                GPUs per node for the trainer (default 1; >1 -> torchrun)
#   NNODES              nodes (default 1 or $SKYPILOT_NUM_NODES; >1 -> multi-node DDP)
#   DISTRIBUTED_STRATEGY  trainer strategy (default ddp when NNODES>1)
#   EPOCHS              passed to the trainer + used in the auto RUN_ID (optional)
#
# Any extra args are forwarded to the trainer.
#
#=============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=scripts/cloud/lib.sh
source "${SCRIPT_DIR}/lib.sh"

PRETRAIN_CONFIG="${PRETRAIN_CONFIG:-configs/stage3_training/pretrain_scratch_v1.json}"
DEVICE="${DEVICE:-$(default_device)}"
NGPU="${NGPU:-1}"
NNODES="${NNODES:-${SKYPILOT_NUM_NODES:-1}}"

[[ -n "${PRIMARY_HDF5:-}" ]] || die "set PRIMARY_HDF5 (precompiled Stage-2 embedding dataset)"

# .ckpt / weights loaded via torch.load without weights_only.
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=true

RUN_ID="${RUN_ID:-$(cloud_run_id "${PRETRAIN_CONFIG}" "${NGPU}" "${EPOCHS:-na}" pt "${NNODES}")}"
log "RUN_ID=${RUN_ID} device=${DEVICE} nnodes=${NNODES} ngpu=${NGPU}"

EPOCH_ARGS=()
[[ -n "${EPOCHS:-}" ]] && EPOCH_ARGS=(--epochs "${EPOCHS}")

log "Pretraining ProteoScribe from scratch on ${PRIMARY_HDF5} (nnodes=${NNODES})"
if [[ "${NNODES}" -le 1 ]]; then
    exec "${SCRIPT_DIR}/../stage3_train_singlenode.sh" \
        "${PRETRAIN_CONFIG}" "${NGPU}" "${DEVICE}" "${RUN_ID}" \
        --primary_data_path "${PRIMARY_HDF5}" \
        "${EPOCH_ARGS[@]}" \
        "$@"
else
    # Cloud multi-node has no shared filesystem -> DDP (single .ckpt from rank 0).
    exec "${SCRIPT_DIR}/../stage3_train_multinode.sh" \
        "${PRETRAIN_CONFIG}" "${NNODES}" "${NGPU}" "${DEVICE}" "${RUN_ID}" \
        --primary_data_path "${PRIMARY_HDF5}" \
        --distributed_strategy "${DISTRIBUTED_STRATEGY:-ddp}" \
        "${EPOCH_ARGS[@]}" \
        "$@"
fi

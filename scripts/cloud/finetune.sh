#!/usr/bin/env bash
#=============================================================================
#
# FILE: scripts/cloud/finetune.sh
#
# Cloud job: Stage-3 (ProteoScribe) finetuning. The provider-agnostic,
# env-driven generalization of scripts/sh3_run1_cluster05_finetune.sh — runs
# unchanged under a Mithril/AWS SkyPilot task, a bare `docker run`, or
# docker/run.sh. See docs/setup/cloud_jobs.md.
#
# Pipeline (steps auto-skip based on which inputs are set):
#   1. Embed    DATASET_CSV --(PenCL + Facilitator from WEIGHT_SET)--> HDF5
#               (skipped when PRIMARY_HDF5 is given instead)
#   2. Split    optional cluster-aware train/val/test manifest (SPLIT=cluster)
#   3. Finetune ProteoScribe from PROTEOSCRIBE_INIT on the data
#
# Outputs follow the trainer's own layout under the config's --output_root
# (default ./outputs/Stage3/finetuning/{checkpoints,runs}/<RUN_ID>/). The
# mounted/synced /app/outputs persists them; set BIOM3_OUTPUTS_PUSH_URI to also
# push to S3 on exit (handled by the container entrypoint).
#
# INPUT (set exactly one):
#   DATASET_CSV         input CSV (sequences + prompts) -> embedded to HDF5
#   PRIMARY_HDF5        a precompiled Stage-2 embedding HDF5 (skips embedding)
#
# WEIGHTS / CONFIG:
#   WEIGHT_SET          weight bundle JSON (default configs/weights/run1_base.json);
#                       supplies pencl/facilitator (embedding) + proteoscribe (init)
#   PROTEOSCRIBE_INIT   finetune init weights (default: proteoscribe_weights from WEIGHT_SET)
#   FINETUNE_CONFIG     Stage-3 training config (default configs/stage3_training/finetune_v1.json)
#   PENCL_CONFIG        Stage-1 config for embedding (default configs/inference/stage1_PenCL.json)
#   FACILITATOR_CONFIG  Stage-2 config for embedding (default configs/inference/stage2_Facilitator.json)
#
# RUN / OUTPUT:
#   RUN_ID              run identifier (default: auto, HPC-style naming)
#   PREFIX              filename prefix for embedding artifacts (default: RUN_ID)
#   DEVICE              cpu | cuda | xpu (default cuda)
#   NGPU                GPUs for the trainer (default 1; >1 -> torchrun)
#   EPOCHS              passed to the trainer + used in the auto RUN_ID (optional)
#
# SPLIT (optional cluster-aware split; default off):
#   SPLIT               none (default) | cluster
#                       none    -> no manifest; trainer uses its config valid_size
#                       cluster -> biom3_cluster_split (REQUIRES `mmseqs` on PATH,
#                                  NOT in the base image; add it or pass CLUSTERS_TSV)
#   SPLIT_MANIFEST      use this prebuilt manifest directly (skips clustering)
#   MIN_SEQ_ID/COVERAGE/TRAIN_FRAC/VAL_FRAC/TEST_FRAC/DIFFUSION_STEPS/FACILITATOR/CLUSTERS_TSV
#                       cluster-split knobs (see biom3_cluster_split --help)
#
# Any extra args are forwarded to the trainer, e.g.:
#   docker/run.sh scripts/cloud/finetune.sh --epochs 1 --batch_size 8
#
#=============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=scripts/cloud/lib.sh
source "${SCRIPT_DIR}/lib.sh"

# ── Inputs / config (with defaults) ──────────────────────────────────────────
WEIGHT_SET="${WEIGHT_SET:-configs/weights/run1_base.json}"
FINETUNE_CONFIG="${FINETUNE_CONFIG:-configs/stage3_training/finetune_v1.json}"
PENCL_CONFIG="${PENCL_CONFIG:-configs/inference/stage1_PenCL.json}"
FACILITATOR_CONFIG="${FACILITATOR_CONFIG:-configs/inference/stage2_Facilitator.json}"
DEVICE="${DEVICE:-cuda}"
NGPU="${NGPU:-1}"
SPLIT="${SPLIT:-none}"

require_one DATASET_CSV PRIMARY_HDF5

# .ckpt init weights are loaded via torch.load without weights_only.
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=true

RUN_ID="${RUN_ID:-$(cloud_run_id "${FINETUNE_CONFIG}" "${NGPU}" "${EPOCHS:-na}" ft)}"
PREFIX="${PREFIX:-${RUN_ID}}"
EMB_DIR="outputs/${RUN_ID}/embeddings"
log "RUN_ID=${RUN_ID} device=${DEVICE} ngpu=${NGPU} split=${SPLIT}"

# ── 1. Embeddings (Stage 1 PenCL + Stage 2 Facilitator -> HDF5) ──────────────
if [[ -n "${PRIMARY_HDF5:-}" ]]; then
    HDF5="${PRIMARY_HDF5}"
    log "[1/3] Using provided PRIMARY_HDF5=${HDF5} (skipping embedding)"
else
    HDF5="${EMB_DIR}/${PREFIX}.compiled_emb.hdf5"
    if [[ -f "${HDF5}" ]]; then
        log "[1/3] Embeddings already present: ${HDF5} (skipping Stage 1+2)"
    else
        log "[1/3] Embedding ${DATASET_CSV} -> ${HDF5}"
        biom3_embedding_pipeline \
            -i "${DATASET_CSV}" \
            -o "${EMB_DIR}" \
            --prefix "${PREFIX}" \
            --weight_set "${WEIGHT_SET}" \
            --pencl_config "${PENCL_CONFIG}" \
            --facilitator_config "${FACILITATOR_CONFIG}" \
            --device "${DEVICE}"
    fi
fi

# ── 2. Optional cluster-aware train/val/test split ───────────────────────────
SPLIT_ARGS=()
if [[ -n "${SPLIT_MANIFEST:-}" ]]; then
    log "[2/3] Using provided SPLIT_MANIFEST=${SPLIT_MANIFEST}"
    SPLIT_ARGS=(--split_manifest_path "${SPLIT_MANIFEST}")
elif [[ "${SPLIT}" == "cluster" ]]; then
    MANIFEST="${EMB_DIR}/${PREFIX}.split_manifest.json"
    log "[2/3] Cluster-aware split -> ${MANIFEST} (needs mmseqs on PATH)"
    CLUST_EXTRA=()
    [[ -n "${CLUSTERS_TSV:-}" ]] && CLUST_EXTRA+=(--clusters_tsv "${CLUSTERS_TSV}")
    biom3_cluster_split \
        --primary_data_path "${HDF5}" \
        --facilitator "${FACILITATOR:-MMD}" \
        --train_frac "${TRAIN_FRAC:-0.8}" \
        --val_frac "${VAL_FRAC:-0.1}" \
        --test_frac "${TEST_FRAC:-0.1}" \
        --min_seq_id "${MIN_SEQ_ID:-0.5}" \
        --coverage "${COVERAGE:-0.8}" \
        --diffusion_steps "${DIFFUSION_STEPS:-1024}" \
        "${CLUST_EXTRA[@]}" \
        -o "${MANIFEST}"
    SPLIT_ARGS=(--split_manifest_path "${MANIFEST}")
else
    log "[2/3] SPLIT=none; trainer will use its config valid_size for validation"
fi

# ── 3. Finetune ProteoScribe ─────────────────────────────────────────────────
PROTEOSCRIBE_INIT="${PROTEOSCRIBE_INIT:-$(weight_from_set "${WEIGHT_SET}" proteoscribe_weights)}"
[[ -n "${PROTEOSCRIBE_INIT}" ]] || die "no PROTEOSCRIBE_INIT (set it or add proteoscribe_weights to ${WEIGHT_SET})"

EPOCH_ARGS=()
[[ -n "${EPOCHS:-}" ]] && EPOCH_ARGS=(--epochs "${EPOCHS}")

log "[3/3] Finetuning ProteoScribe from ${PROTEOSCRIBE_INIT}"
exec "${SCRIPT_DIR}/../stage3_train_singlenode.sh" \
    "${FINETUNE_CONFIG}" "${NGPU}" "${DEVICE}" "${RUN_ID}" \
    --finetune True \
    --pretrained_weights "${PROTEOSCRIBE_INIT}" \
    --primary_data_path "${HDF5}" \
    "${SPLIT_ARGS[@]}" \
    "${EPOCH_ARGS[@]}" \
    "$@"

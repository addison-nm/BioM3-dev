#!/usr/bin/env bash
#=============================================================================
#
# FILE: scripts/cloud/generate.sh
#
# Cloud job: sequence generation (Stage-3 inference). Provider-agnostic,
# env-driven; runs unchanged under a Mithril/AWS SkyPilot task, a bare
# `docker run`, or docker/run.sh. See docs/setup/cloud_jobs.md.
#
# Chain (prompts CSV -> protein sequences):
#   1. biom3_PenCL_inference     PROMPTS_CSV          -> {prefix}.PenCL_emb.pt
#   2. biom3_Facilitator_sample  {prefix}.PenCL_emb   -> {prefix}.Facilitator_emb.pt
#   3. biom3_ProteoScribe_sample {prefix}.Facilitator -> {prefix}.generated.pt (+ .fasta)
#
# Note: this uses the Facilitator .pt directly as ProteoScribe input — it does
# NOT compile to HDF5 (that step is training-only).
#
# Outputs land under OUTPUT_DIR (default outputs/<OUTPUT_PREFIX>/), persisted via
# the mounted/synced /app/outputs; set BIOM3_OUTPUTS_PUSH_URI to also push to S3.
#
# INPUT:
#   PROMPTS_CSV         input CSV of prompts (required)
#
# WEIGHTS / CONFIG:
#   WEIGHT_SET          weight bundle JSON (default configs/weights/run1_base.json);
#                       supplies pencl/facilitator/proteoscribe weights
#   PENCL_WEIGHTS / FACILITATOR_WEIGHTS / PROTEOSCRIBE_WEIGHTS
#                       per-stage overrides (win over WEIGHT_SET)
#   PENCL_CONFIG        default configs/inference/stage1_PenCL.json
#   FACILITATOR_CONFIG  default configs/inference/stage2_Facilitator.json
#   PROTEOSCRIBE_CONFIG default configs/inference/stage3_ProteoScribe_sample.json
#
# RUN / OUTPUT:
#   OUTPUT_PREFIX       filename prefix (default: gen_<YYYYMMDD_HHMMSS>)
#   OUTPUT_DIR          output directory (default: outputs/<OUTPUT_PREFIX>)
#   DEVICE              cpu | cuda | xpu (default: auto-detected backend)
#   NGPU                GPUs per node for Stage-3 sampling (default 1; >1 -> torchrun)
#   NNODES              nodes (default 1 or $SKYPILOT_NUM_NODES; >1 -> multi-node sampling)
#   BATCH_SIZE          Stage-1 batch size (default 256)
#   MMD_SAMPLE_LIMIT    Stage-2 MMD limit (default 1000)
#   SEED                ProteoScribe sampling seed (default 0)
#   TOKEN_STRATEGY      sample | argmax (optional)
#   UNMASKING_ORDER     random | confidence | confidence_no_pad (optional)
#   FASTA               true (default) | false — also write FASTA alongside .pt
#
# Any extra args are forwarded to biom3_ProteoScribe_sample.
#
#=============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=scripts/cloud/lib.sh
source "${SCRIPT_DIR}/lib.sh"

WEIGHT_SET="${WEIGHT_SET:-configs/weights/run1_base.json}"
PENCL_CONFIG="${PENCL_CONFIG:-configs/inference/stage1_PenCL.json}"
FACILITATOR_CONFIG="${FACILITATOR_CONFIG:-configs/inference/stage2_Facilitator.json}"
PROTEOSCRIBE_CONFIG="${PROTEOSCRIBE_CONFIG:-configs/inference/stage3_ProteoScribe_sample.json}"
DEVICE="${DEVICE:-$(default_device)}"
NGPU="${NGPU:-1}"
NNODES="${NNODES:-${SKYPILOT_NUM_NODES:-1}}"

[[ -n "${PROMPTS_CSV:-}" ]] || die "set PROMPTS_CSV (input CSV of prompts)"

# .ckpt weights are loaded via torch.load without weights_only.
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=true

# Resolve per-stage weights: explicit env overrides win over the bundle.
PENCL_WEIGHTS="${PENCL_WEIGHTS:-$(weight_from_set "${WEIGHT_SET}" pencl_weights)}"
FACILITATOR_WEIGHTS="${FACILITATOR_WEIGHTS:-$(weight_from_set "${WEIGHT_SET}" facilitator_weights)}"
PROTEOSCRIBE_WEIGHTS="${PROTEOSCRIBE_WEIGHTS:-$(weight_from_set "${WEIGHT_SET}" proteoscribe_weights)}"
[[ -n "${PENCL_WEIGHTS}" ]]       || die "no PenCL weights (set PENCL_WEIGHTS or pencl_weights in ${WEIGHT_SET})"
[[ -n "${FACILITATOR_WEIGHTS}" ]] || die "no Facilitator weights (set FACILITATOR_WEIGHTS or facilitator_weights in ${WEIGHT_SET})"
[[ -n "${PROTEOSCRIBE_WEIGHTS}" ]] || die "no ProteoScribe weights (set PROTEOSCRIBE_WEIGHTS or proteoscribe_weights in ${WEIGHT_SET})"

OUTPUT_PREFIX="${OUTPUT_PREFIX:-gen_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/${OUTPUT_PREFIX}}"
mkdir -p "${OUTPUT_DIR}"
PENCL_PT="${OUTPUT_DIR}/${OUTPUT_PREFIX}.PenCL_emb.pt"
FAC_PT="${OUTPUT_DIR}/${OUTPUT_PREFIX}.Facilitator_emb.pt"
GEN_PT="${OUTPUT_DIR}/${OUTPUT_PREFIX}.generated.pt"
log "OUTPUT_PREFIX=${OUTPUT_PREFIX} device=${DEVICE} -> ${OUTPUT_DIR}"

# ── 1. Stage 1: PenCL ────────────────────────────────────────────────────────
log "[1/3] PenCL inference ${PROMPTS_CSV} -> ${PENCL_PT}"
biom3_PenCL_inference \
    -i "${PROMPTS_CSV}" -c "${PENCL_CONFIG}" -m "${PENCL_WEIGHTS}" \
    -o "${PENCL_PT}" --device "${DEVICE}" --batch_size "${BATCH_SIZE:-256}"

# ── 2. Stage 2: Facilitator ──────────────────────────────────────────────────
log "[2/3] Facilitator sample -> ${FAC_PT}"
biom3_Facilitator_sample \
    -i "${PENCL_PT}" -c "${FACILITATOR_CONFIG}" -m "${FACILITATOR_WEIGHTS}" \
    -o "${FAC_PT}" --device "${DEVICE}" --mmd_sample_limit "${MMD_SAMPLE_LIMIT:-1000}"

# ── 3. Stage 3: ProteoScribe sampling ────────────────────────────────────────
GEN_ARGS=(-i "${FAC_PT}" -c "${PROTEOSCRIBE_CONFIG}" -m "${PROTEOSCRIBE_WEIGHTS}"
          -o "${GEN_PT}" --device "${DEVICE}" --seed "${SEED:-0}")
[[ "${FASTA:-true}" == "true" ]] && GEN_ARGS+=(--fasta)
[[ -n "${TOKEN_STRATEGY:-}" ]] && GEN_ARGS+=(--token_strategy "${TOKEN_STRATEGY}")
[[ -n "${UNMASKING_ORDER:-}" ]] && GEN_ARGS+=(--unmasking_order "${UNMASKING_ORDER}")

log "[3/3] ProteoScribe sample -> ${GEN_PT} (nnodes=${NNODES} ngpu=${NGPU})"
# Only Stage 3 is parallelizable; the sampler shards work across ranks and only
# rank 0 writes. Stages 1-2 above ran once per node — Facilitator sampling is
# stochastic, so for deterministic multi-node output stage FAC_PT from rank 0
# (see docker/README.md); pass a fixed --seed and verify determinism first.
if [[ "${NNODES}" -gt 1 ]]; then
    NGPU_PER_NODE="${NGPU}" NUM_NODES="${NNODES}" \
      exec "${SCRIPT_DIR}/../launchers/container_multinode.sh" \
        biom3_ProteoScribe_sample "${GEN_ARGS[@]}" "$@"
elif [[ "${NGPU}" -gt 1 ]]; then
    NGPU="${NGPU}" exec "${SCRIPT_DIR}/../launchers/container_singlenode.sh" \
        biom3_ProteoScribe_sample "${GEN_ARGS[@]}" "$@"
else
    exec biom3_ProteoScribe_sample "${GEN_ARGS[@]}" "$@"
fi

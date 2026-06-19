#!/usr/bin/env bash
#=============================================================================
#
# FILE: docker/run.sh
#
# Convenience wrapper for `docker run` on a GPU host (AWS / Mithril): mounts
# the conventional BioM3 directories and forwards the useful env vars, then
# passes through whatever command you give it.
#
# USAGE:
#   docker/run.sh <command...>
#
# EXAMPLES:
#   # interactive shell
#   docker/run.sh bash
#
#   # single-GPU Stage 3 training
#   docker/run.sh scripts/stage3_train_singlenode.sh \
#       configs/stage3_training/pretrain_scratch_v1.json 1 cuda run001 --epochs 1
#
#   # 4-GPU training (NGPU drives torchrun inside the container)
#   BIOM3_GPUS=all NGPU=4 docker/run.sh scripts/stage3_train_singlenode.sh \
#       configs/stage3_training/pretrain_scratch_v1.json 4 cuda run001 --epochs 1
#
#   # generation
#   docker/run.sh biom3_ProteoScribe_sample --input_path outputs/facilitator.pt \
#       --config_path configs/inference/stage3_ProteoScribe_sample.json \
#       --model_path weights/ProteoScribe/BioM3_ProteoScribe_pfam_epoch20_v1.bin \
#       --output_path outputs/generated.pt --device cuda
#
# ENV (all optional):
#   BIOM3_IMAGE        image tag (default: biom3:gpu)
#   BIOM3_GPUS         value for --gpus (default: all; "none" omits --gpus)
#   BIOM3_WEIGHTS_DIR  host weights dir  (default: ./weights, mounted ro)
#   BIOM3_DATA_DIR     host data dir     (default: ./data,    mounted ro)
#   BIOM3_OUTPUTS_DIR  host outputs dir  (default: ./outputs, mounted rw)
#   BIOM3_CONFIGS_DIR  host configs dir  (optional; overrides baked-in configs)
#   Forwarded if set:  WANDB_API_KEY, NGPU, AWS_* and BIOM3_*_URI / sync vars.
#
#=============================================================================
set -euo pipefail

IMAGE="${BIOM3_IMAGE:-biom3:gpu}"
GPUS="${BIOM3_GPUS:-all}"
W="${BIOM3_WEIGHTS_DIR:-$PWD/weights}"
D="${BIOM3_DATA_DIR:-$PWD/data}"
O="${BIOM3_OUTPUTS_DIR:-$PWD/outputs}"
C="${BIOM3_CONFIGS_DIR:-}"

mkdir -p "${O}"

ARGS=(run --rm)
[[ -t 0 && -t 1 ]] && ARGS+=(-it)
[[ "${GPUS}" != "none" ]] && ARGS+=(--gpus "${GPUS}")

# Bind-mount weights/data unless an S3 (or other) sync URI is set for them —
# in sync mode the entrypoint writes into the container's own dir, so a
# read-only host mount would shadow it and the sync would fail.
[[ -d "${W}" && -z "${BIOM3_WEIGHTS_URI:-}" ]] && ARGS+=(-v "${W}:/app/weights:ro")
[[ -d "${D}" && -z "${BIOM3_DATA_URI:-}" ]] && ARGS+=(-v "${D}:/app/data:ro")
ARGS+=(-v "${O}:/app/outputs")
[[ -n "${C}" ]] && ARGS+=(-v "${C}:/app/configs:ro")
[[ -d "${HOME}/.aws" ]] && ARGS+=(-v "${HOME}/.aws:/root/.aws:ro")

# Forward env vars that are set in the caller's environment.
for v in WANDB_API_KEY NGPU \
         BIOM3_WEIGHTS_URI BIOM3_DATA_URI BIOM3_WEIGHTS_INCLUDES \
         BIOM3_SYNC_MODE BIOM3_SYNC_CMD BIOM3_SYNC_CMD_OUT BIOM3_OUTPUTS_PUSH_URI \
         AWS_ENDPOINT_URL AWS_PROFILE AWS_REGION AWS_DEFAULT_REGION \
         AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN; do
    [[ -n "${!v:-}" ]] && ARGS+=(-e "${v}=${!v}")
done

exec docker "${ARGS[@]}" "${IMAGE}" "$@"

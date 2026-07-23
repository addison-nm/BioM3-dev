#!/usr/bin/env bash
#=============================================================================
#
# FILE: scripts/polaris/apptainer_run.sh
#
# Run a command inside the BioM3 CUDA .sif on Polaris, single node. Exposes the
# NVIDIA GPUs (apptainer --nv), binds /grand, sources environment.sh inside the
# container (so it auto-detects `polaris`), then runs whatever command you pass.
#
# The CUDA image is the SAME one used for AWS/Mithril (docker/Dockerfile.cuda) —
# Polaris is just NVIDIA, so no device-specific image is needed, only this run
# glue. This is the container replacement for the bare-metal `module load conda +
# source venv + source environment.sh` prelude (docs/setup/setup_polaris.md).
#
# SINGLE NODE only — multi-node NCCL over Polaris's Slingshot fabric needs the
# host MPI bound in and is not handled here (see setup_polaris_container.md).
#
# USAGE:
#   scripts/polaris/apptainer_run.sh <command...>
#
# EXAMPLES:
#   # interactive shell (GPUs visible; try `nvidia-smi` or torch.cuda.device_count())
#   scripts/polaris/apptainer_run.sh bash
#
#   # single-node Stage 3 finetune on 4 A100s
#   BIOM3_WEIGHTS_DIR=/grand/NLDesignProtein/.../weights \
#   BIOM3_DATA_DIR=/grand/NLDesignProtein/.../data \
#   scripts/polaris/apptainer_run.sh scripts/stage3_train_singlenode.sh \
#       configs/stage3_training/pretrain_scratch_v1.json 4 cuda run001 --epochs 1
#
# ENV (all optional):
#   BIOM3_SIF          path to the .sif (default: ./biom3_cuda.sif)
#   BIOM3_WEIGHTS_DIR  host weights dir bound to /app/weights (ro)
#   BIOM3_DATA_DIR     host data dir    bound to /app/data    (ro)
#   BIOM3_OUTPUTS_DIR  host outputs dir bound to /app/outputs (rw; default ./outputs)
#   BIOM3_CONFIGS_DIR  host configs dir bound to /app/configs (ro; overrides baked-in)
#   BIOM3_BIND_EXTRA   extra colon/comma paths to --bind (e.g. an /eagle root)
#   WANDB_API_KEY      forwarded into the container if set
#
#=============================================================================
set -euo pipefail

[[ $# -ge 1 ]] || { echo "USAGE: $0 <command...>   (see --help header)" >&2; exit 1; }
[[ "$1" == "-h" || "$1" == "--help" ]] && { sed -n '3,40p' "$0"; exit 0; }

SIF="${BIOM3_SIF:-./biom3_cuda.sif}"
[[ -f "${SIF}" ]] || { echo "ERROR: sif '${SIF}' not found. Build it (on a compute node):" >&2
    echo "         apptainer build --fakeroot biom3_cuda.sif docker://ghcr.io/natural-machine/biom3:cuda-dev" >&2
    echo "       see docs/setup/setup_polaris_container.md, or set BIOM3_SIF." >&2; exit 1; }

command -v apptainer >/dev/null 2>&1 || { echo "ERROR: apptainer not found ('module load' it on the compute node)." >&2; exit 1; }

O="${BIOM3_OUTPUTS_DIR:-$PWD/outputs}"
mkdir -p "${O}"

# --- Binds ---------------------------------------------------------------
# /grand : ALCF Lustre; also what environment.sh fingerprints to pick `polaris`.
BINDS=("${O}:/app/outputs")
[[ -d /grand ]] && BINDS+=("/grand")
[[ -n "${BIOM3_WEIGHTS_DIR:-}" ]] && BINDS+=("${BIOM3_WEIGHTS_DIR}:/app/weights:ro")
[[ -n "${BIOM3_DATA_DIR:-}"    ]] && BINDS+=("${BIOM3_DATA_DIR}:/app/data:ro")
[[ -n "${BIOM3_CONFIGS_DIR:-}" ]] && BINDS+=("${BIOM3_CONFIGS_DIR}:/app/configs:ro")
[[ -n "${BIOM3_BIND_EXTRA:-}"  ]] && BINDS+=("${BIOM3_BIND_EXTRA}")

BIND_ARG="$(IFS=,; echo "${BINDS[*]}")"

# --- apptainer invocation ------------------------------------------------
# `--nv` exposes the NVIDIA driver/libs (the CUDA analog of the XPU path's
# /dev/dri bind). Built as one array so an unset WANDB_API_KEY doesn't leave an
# empty "${ENVS[@]}" to expand under set -u.
ARGS=(exec --nv --bind "${BIND_ARG}")
[[ -n "${WANDB_API_KEY:-}" ]] && ARGS+=(--env "WANDB_API_KEY=${WANDB_API_KEY}")
ARGS+=("${SIF}")

# The CUDA image bakes `BIOM3_MACHINE=container` (for AWS/Mithril). Unset it so
# environment.sh's /grand fingerprint selects `polaris` instead. `exec` (not
# `run`) bypasses the S3-sync entrypoint; we source environment.sh ourselves.
# The passed command runs from /app with "$@" preserved.
set -- bash -lc 'cd /app && unset BIOM3_MACHINE && source environment.sh >&2 && exec "$@"' _ "$@"

echo "+ apptainer exec --nv --bind ${BIND_ARG} ${SIF} <cmd>" >&2
exec apptainer "${ARGS[@]}" "$@"

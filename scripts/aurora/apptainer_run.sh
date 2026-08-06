#!/usr/bin/env bash
#=============================================================================
#
# FILE: scripts/aurora/apptainer_run.sh
#
# Run a command inside the BioM3 Intel-XPU .sif on Aurora, single node. Binds
# /flare, sources environment.sh inside the container (so it auto-detects
# `aurora` and applies the oneCCL/xccl/NUMEXPR settings), then runs whatever
# command you pass.
#
# This is the container replacement for the bare-metal `module load frameworks +
# source venv + source environment.sh` prelude. SINGLE NODE only — multi-node
# oneCCL over Aurora's CXI fabric needs the host libfabric/MPICH bound in and is
# not handled here (see docs/setup/setup_aurora_container.md).
#
# USAGE:
#   scripts/aurora/apptainer_run.sh <command...>
#
# EXAMPLES:
#   # interactive shell (GPUs visible; try `python -c "import torch;print(torch.xpu.device_count())"`)
#   scripts/aurora/apptainer_run.sh bash
#
#   # single-node Stage 3 finetune on 12 tiles
#   BIOM3_WEIGHTS_DIR=/flare/NLDesignProtein/.../weights \
#   BIOM3_DATA_DIR=/flare/NLDesignProtein/.../data \
#   scripts/aurora/apptainer_run.sh scripts/stage3_train_singlenode.sh \
#       configs/stage3_training/pretrain_scratch_v1.json 12 xpu run001 --epochs 1
#
# ENV (all optional):
#   BIOM3_SIF          path to the .sif (default: ./biom3_xpu.sif)
#   BIOM3_WEIGHTS_DIR  host weights dir bound to /app/weights (ro)
#   BIOM3_DATA_DIR     host data dir    bound to /app/data    (ro)
#   BIOM3_OUTPUTS_DIR  host outputs dir bound to /app/outputs (rw; default ./outputs)
#   BIOM3_CONFIGS_DIR  host configs dir bound to /app/configs (ro; overrides baked-in)
#   BIOM3_BIND_EXTRA   extra colon/comma paths to --bind (e.g. a checkpoints root)
#   BIOM3_FI_PROVIDER  libfabric provider (default tcp; cxi needs host binds)
#   BIOM3_CCL_LAUNCHER CCL_PROCESS_LAUNCHER (default torchrun; hydra|pmix|none)
#   WANDB_API_KEY      forwarded into the container if set
#
#=============================================================================
set -euo pipefail

[[ $# -ge 1 ]] || { echo "USAGE: $0 <command...>   (see --help header)" >&2; exit 1; }
[[ "$1" == "-h" || "$1" == "--help" ]] && { sed -n '3,40p' "$0"; exit 0; }

SIF="${BIOM3_SIF:-./biom3_xpu.sif}"
[[ -f "${SIF}" ]] || { echo "ERROR: sif '${SIF}' not found. Build it (on a login node):" >&2
    echo "         apptainer build biom3_xpu.sif docker://ghcr.io/natural-machine/biom3:xpu-dev" >&2
    echo "       see docs/setup/setup_aurora_container.md, or set BIOM3_SIF." >&2; exit 1; }

command -v apptainer >/dev/null 2>&1 || { echo "ERROR: apptainer not found." >&2; exit 1; }

O="${BIOM3_OUTPUTS_DIR:-$PWD/outputs}"
mkdir -p "${O}"

# --- Binds ---------------------------------------------------------------
# /flare   : ALCF Lustre; also what environment.sh fingerprints to pick `aurora`.
#
# Do NOT bind /dev/dri. Apptainer mounts /dev by default; adding /dev/dri as a
# user bind remounts it `nodev`, so the GPU character devices become unusable
# and torch.xpu.device_count() returns 0 (clinfo -l also comes back empty).
BINDS=("${O}:/app/outputs")
# /lus alongside /flare: on Aurora /flare IS /lus/flare/projects, and the
# weights/ and data/ trees are symlinks whose targets are spelled /lus/...
# Binding only /flare leaves every one of them dangling inside the container --
# which surfaces far from the cause, e.g. transformers reporting a local model
# directory as a malformed Hub repo id.
[[ -d /flare ]] && BINDS+=("/flare")
[[ -d /lus ]] && BINDS+=("/lus")
[[ -n "${BIOM3_WEIGHTS_DIR:-}" ]] && BINDS+=("${BIOM3_WEIGHTS_DIR}:/app/weights:ro")
[[ -n "${BIOM3_DATA_DIR:-}"    ]] && BINDS+=("${BIOM3_DATA_DIR}:/app/data:ro")
[[ -n "${BIOM3_CONFIGS_DIR:-}" ]] && BINDS+=("${BIOM3_CONFIGS_DIR}:/app/configs:ro")
[[ -n "${BIOM3_BIND_EXTRA:-}"  ]] && BINDS+=("${BIOM3_BIND_EXTRA}")

BIND_ARG="$(IFS=,; echo "${BINDS[*]}")"

# --- Env into the container ----------------------------------------------
# ZE_FLAT_DEVICE_HIERARCHY=FLAT exposes each of Aurora's 12 tiles as its own
# device (matches num_devices=12 in the PBS templates). On bare metal the
# frameworks module sets this; the container must set it itself.
ENVS=(--env "ZE_FLAT_DEVICE_HIERARCHY=FLAT")

# oneCCL settings that MUST differ from bare metal, because a shell with
# `module load frameworks` exports values that are wrong inside the container
# and apptainer forwards them:
#
#   FI_PROVIDER — the host asks for `cxi,tcp;ofi_rxm`, but the container's
#     libfabric has no cxi provider, so libfabric matches nothing and oneCCL
#     fails with "fi_getinfo error: ret -61, providers 0". tcp is correct for
#     single node (GPU transfers go over Level-Zero IPC, not the fabric).
#     Multi-node over CXI needs the host libfabric bound in; see the runbook.
#
#   CCL_PROCESS_LAUNCHER — the host sets `pmix`, but no PMIx server is
#     reachable in the container, so oneCCL cannot resolve local rank/size.
#     `torchrun` reads LOCAL_RANK/LOCAL_WORLD_SIZE, which covers both a single
#     process and a multi-rank launch.
#   CCL_ROOT — the host points at /opt/aurora/<ver>/oneapi/ccl/latest, which
#     does not exist in the container, so oneCCL cannot find its SPIR-V kernels
#     ("failed to load file containing oneCCL SPIR-V kernels"). The pip-installed
#     oneCCL keeps them under /opt/venv/lib/ccl/kernels.
ENVS+=(--env "FI_PROVIDER=${BIOM3_FI_PROVIDER:-tcp}")
ENVS+=(--env "CCL_PROCESS_LAUNCHER=${BIOM3_CCL_LAUNCHER:-torchrun}")
ENVS+=(--env "CCL_ROOT=/opt/venv")

# Spawn ranks with torchrun (launchers/container_singlenode.sh), not the host's
# mpiexec: the container has no access to PBS's hostfile, so Hydra fails with
# "unable to find an RMK and the node list". BIOM3_MACHINE stays `aurora`, so
# the Aurora oneCCL/NUMEXPR settings still apply.
ENVS+=(--env "BIOM3_LAUNCHER=${BIOM3_LAUNCHER:-container}")

[[ -n "${WANDB_API_KEY:-}" ]] && ENVS+=(--env "WANDB_API_KEY=${WANDB_API_KEY}")

# `exec` (not `run`) so we bypass the S3-sync entrypoint and instead source
# environment.sh ourselves — that is what applies the Aurora oneCCL/xccl vars.
# The passed command runs from /app with "$@" preserved.
set -- bash -lc 'cd /app && source environment.sh >&2 && exec "$@"' _ "$@"

# `--writable-tmpfs`: ephemeral RAM-backed overlay so incidental writes to the
# read-only image (caches, tests/_tmp) succeed; real outputs go to /app/outputs.
echo "+ apptainer exec --writable-tmpfs --bind ${BIND_ARG} ${SIF} <cmd>" >&2
exec apptainer exec --writable-tmpfs --bind "${BIND_ARG}" "${ENVS[@]}" "${SIF}" "$@"

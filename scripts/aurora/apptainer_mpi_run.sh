#!/usr/bin/env bash
#=============================================================================
#
# FILE: scripts/aurora/apptainer_mpi_run.sh
#
# Run a command inside the BioM3 Intel-XPU .sif on Aurora under the HOST's
# mpiexec: one container per rank, spawned by PALS, spanning one or more nodes.
#
# This is the multi-node counterpart to apptainer_run.sh. The two differ in
# which side of the container boundary the launcher lives on:
#
#   apptainer_run.sh      apptainer exec -> torchrun -> N ranks   (single node)
#   apptainer_mpi_run.sh  mpiexec -> N x (apptainer exec -> rank) (any node count)
#
# The mpiexec form is the pattern ALCF documents for MPI workloads in containers
# (https://docs.alcf.anl.gov/aurora/containers/containers/). It is required for
# multi-node because mpiexec inside a container cannot read PBS's hostfile, and
# it also inherits the ALCF-canonical --cpu-bind that the torchrun path lacks.
#
# The command runs ONE rank per process, so pass the entry point directly
# (biom3_train_stage3 ...), NOT scripts/stage3_train_*node.sh — those dispatch
# to a launcher that would spawn ranks a second time.
#
# USAGE:
#   scripts/aurora/apptainer_mpi_run.sh <command...>
#
# EXAMPLES:
#   # single node, 12 tiles (use this first to validate the path)
#   NGPU_PER_NODE=12 NGPU_TOTAL=12 \
#   BIOM3_SIF=/flare/.../biom3_xpu-<sha>.sif \
#   scripts/aurora/apptainer_mpi_run.sh \
#       biom3_train_stage3 --config_path configs/stage3_training/pretrain_scratch_v1.json \
#       --device xpu --devices_per_node 12 --num_nodes 1 --run_id mpi001
#
#   # two nodes, 24 tiles
#   NGPU_PER_NODE=12 NGPU_TOTAL=24 ... (same, --num_nodes 2 --run_id mpi002)
#
# ENV:
#   NGPU_PER_NODE      ranks per node (required; 12 for full Aurora nodes)
#   NGPU_TOTAL         total ranks across all nodes (required)
#   PBS_NODEFILE       set by PBS; required for >1 node
#   BIOM3_SIF          path to the .sif (default: ./biom3_xpu.sif)
#   BIOM3_WEIGHTS_DIR  host weights dir bound to /app/weights (ro)
#   BIOM3_DATA_DIR     host data dir    bound to /app/data    (ro)
#   BIOM3_OUTPUTS_DIR  host outputs dir bound to /app/outputs (rw; default ./outputs)
#   BIOM3_CONFIGS_DIR  host configs dir bound to /app/configs (ro)
#   BIOM3_BIND_EXTRA   extra colon/comma paths to --bind
#   BIOM3_FI_PROVIDER  libfabric provider (default tcp; see the CXI note below)
#   BIOM3_FABRIC_DIR   host libfabric to bind over the container's (CXI; below)
#   BIOM3_PMIX         host PMIx library (default /usr/lib64/libpmix.so.2)
#   WANDB_API_KEY      forwarded into the container if set
#
# CXI: the default provider is tcp, which works across nodes but does not use
# Aurora's Slingshot fabric — two nodes end up slower in aggregate than one.
# Driving CXI needs HPE's Cray libfabric bound in. Intel MPI's own bundled
# libfabric has no cxi provider, in the image or under /opt/aurora:
#   BIOM3_FABRIC_DIR=/opt/cray/libfabric/1.22.0/lib64 \
#   BIOM3_FI_PROVIDER=cxi \
#   scripts/aurora/apptainer_mpi_run.sh ...
# See setup_aurora_container.md.
#
#=============================================================================
set -euo pipefail

[[ $# -ge 1 ]] || { echo "USAGE: $0 <command...>   (see --help header)" >&2; exit 1; }
[[ "$1" == "-h" || "$1" == "--help" ]] && { sed -n '3,57p' "$0"; exit 0; }

SIF="${BIOM3_SIF:-./biom3_xpu.sif}"
[[ -f "${SIF}" ]] || { echo "ERROR: sif '${SIF}' not found; set BIOM3_SIF." >&2; exit 1; }

command -v mpiexec >/dev/null 2>&1 || { echo "ERROR: mpiexec not found (host MPI)." >&2; exit 1; }
command -v apptainer >/dev/null 2>&1 || { echo "ERROR: apptainer not found; module load apptainer." >&2; exit 1; }

NGPU_PER_NODE="${NGPU_PER_NODE:?NGPU_PER_NODE env var required}"
NGPU_TOTAL="${NGPU_TOTAL:?NGPU_TOTAL env var required}"

PMIX="${BIOM3_PMIX:-/usr/lib64/libpmix.so.2}"
[[ -f "${PMIX}" ]] || { echo "ERROR: PMIx library '${PMIX}' not found; set BIOM3_PMIX." >&2; exit 1; }

O="${BIOM3_OUTPUTS_DIR:-$PWD/outputs}"
mkdir -p "${O}"

# --- Binds ---------------------------------------------------------------
# Do NOT bind /dev/dri: apptainer mounts /dev by default, and adding it as a
# user bind remounts it nodev, which hides the GPUs (see apptainer_run.sh).
#
# /hostlib  : host PMIx, so the container's MPI can bootstrap against PALS.
# /hostevent: host /usr/lib64, prepended to LD_LIBRARY_PATH for libevent, which
#             PMIx links against. Prepending the whole directory is why this is
#             scoped to a subdirectory rather than binding over /usr/lib64.
BINDS=("/flare" "${O}:/app/outputs" "${PMIX}:/hostlib/libpmix.so.2" "/usr/lib64:/hostevent")

# BIOM3_FABRIC_DIR binds a host libfabric over the container's, so cross-node
# collectives can use Aurora's CXI provider instead of tcp, which otherwise caps
# multi-node throughput below a single node's.
#
# It must be the directory *containing* libfabric.so.1, and it must be HPE's
# Cray build -- Intel MPI's own bundled libfabric ships efa/psm3/rxm/tcp/verbs
# and no cxi, on Aurora as elsewhere:
#   BIOM3_FABRIC_DIR=/opt/cray/libfabric/1.22.0/lib64 BIOM3_FI_PROVIDER=cxi
# In that build cxi is compiled in rather than a loadable plugin, which is why
# FI_PROVIDER_PATH is empty on bare metal and only set below when a prov/
# directory actually exists.
if [[ -n "${BIOM3_FABRIC_DIR:-}" ]]; then
    [[ -e "${BIOM3_FABRIC_DIR}/libfabric.so.1" ]] || {
        echo "ERROR: no libfabric.so.1 in BIOM3_FABRIC_DIR='${BIOM3_FABRIC_DIR}'." >&2
        echo "       Point it at the directory containing it, e.g." >&2
        echo "       /opt/cray/libfabric/<ver>/lib64 (the build with the cxi provider)." >&2
        exit 1; }
    BINDS+=("${BIOM3_FABRIC_DIR}:/hostfabric:ro")
fi
[[ -n "${BIOM3_WEIGHTS_DIR:-}" ]] && BINDS+=("${BIOM3_WEIGHTS_DIR}:/app/weights:ro")
[[ -n "${BIOM3_DATA_DIR:-}"    ]] && BINDS+=("${BIOM3_DATA_DIR}:/app/data:ro")
[[ -n "${BIOM3_CONFIGS_DIR:-}" ]] && BINDS+=("${BIOM3_CONFIGS_DIR}:/app/configs:ro")
[[ -n "${BIOM3_BIND_EXTRA:-}"  ]] && BINDS+=("${BIOM3_BIND_EXTRA}")
BIND_ARG="$(IFS=,; echo "${BINDS[*]}")"

# --- Env into each rank's container ---------------------------------------
# CCL_PROCESS_LAUNCHER=torchrun, not pmix. mpiexec spawns the ranks, but rank
# identity reaches the workload through the env vars the prelude derives from
# PALS, not through MPI -- so oneCCL should read those same variables rather
# than try to join a PMIx namespace. With pmix it blocks during init instead of
# failing. CCL_ROOT is overridden because the host path does not exist here.
ENVS=(--env "ZE_FLAT_DEVICE_HIERARCHY=FLAT"
      --env "CCL_ROOT=/opt/venv"
      --env "CCL_PROCESS_LAUNCHER=${BIOM3_CCL_LAUNCHER:-torchrun}"
      --env "FI_PROVIDER=${BIOM3_FI_PROVIDER:-tcp}"
      --env "I_MPI_PMI_LIBRARY=/hostlib/libpmix.so.2"
      --env "CCL_ZE_IPC_EXCHANGE=${BIOM3_CCL_ZE_IPC_EXCHANGE:-sockets}"
      --env "CCL_ATL_TRANSPORT=${BIOM3_CCL_ATL_TRANSPORT:-mpi}")

# CCL_ATL_TRANSPORT=mpi (oneCCL's own default) rather than the ofi that ALCF's
# recipe sets. Their recipe has no usable MPI inside the container, so oneCCL
# must open libfabric providers itself; this image does have one, working over
# cxi. Left on ofi, oneCCL opens its own providers and fails on cxi with
#   fi_getinfo error: ret -61, providers 0 / can't create providers for name cxi
# even with Cray's libfabric preloaded and fi_info -p cxi listing every domain --
# it asks for capabilities the provider will not grant it. Riding the MPI that
# already works sidesteps that entirely.

# CCL_ZE_IPC_EXCHANGE=sockets is required on this path, not merely advisable.
# oneCCL's default (pidfd) exchanges Level-Zero IPC handles with pidfd_getfd,
# which needs ptrace-level access to the peer process. Under apptainer_run.sh
# every rank is a torchrun child inside ONE container, so that succeeds. Here
# each rank is its own container with its own PID namespace and it is denied:
#   pidfd_getfd failed: ... errno: Operation not permitted
#   ze_fd_manager.cpp:390 convert_fd_pidfd
# Sockets are the exchange ALCF's container recipe uses for the same reason.

# LD_LIBRARY_PATH is deliberately NOT set with --env: that replaces the image's
# own value rather than extending it, and on an oneAPI base that value carries
# every /opt/intel/oneapi/*/lib directory. Dropping it makes anything built with
# the Intel compiler unloadable --
#   ImportError: libsvml.so: cannot open shared object file
# and hides Intel MPI's bundled libfabric providers, which surfaces later as
#   MPI_Init_thread ... MPIDI_OFI_mpi_init_hook: Other MPI error
# The prelude appends /hostevent instead, inside the container, where the image's
# value is already in place.
# Intel MPI prefers its own bundled libfabric unless told otherwise, which would
# defeat the bind above.
[[ -n "${BIOM3_FABRIC_DIR:-}" ]] && ENVS+=(--env "I_MPI_OFI_LIBRARY_INTERNAL=0")

[[ -n "${WANDB_API_KEY:-}" ]] && ENVS+=(--env "WANDB_API_KEY=${WANDB_API_KEY}")

# ALCF-canonical 12-tile binding, matching launchers/aurora_multinode.sh.
CPU_BIND_SCHEME="--cpu-bind=list:1-8:9-16:17-24:25-32:33-40:41-48:53-60:61-68:69-76:77-84:85-92:93-100"
if [[ "${NGPU_PER_NODE}" != "12" ]]; then
    echo "WARNING (apptainer_mpi_run.sh): NGPU_PER_NODE=${NGPU_PER_NODE} but the" \
         "CPU_BIND list assumes 12 tiles per node. Adjust the list if needed." >&2
fi

# -genv values configure the HOST launcher (Intel MPI), per the ALCF container
# recipe in _misc/sample_script.sh. Distinct from the --env values below, which
# configure oneCCL/libfabric *inside* each rank's container.
#
# NOTE: the recipe also sets ZE_AFFINITY_MASK=0..11. Do NOT copy that here:
# backend/xpu.py resolves to xpu:0 whenever ZE_AFFINITY_MASK is set, so every
# rank would land on tile 0.
MPI_ARGS=(--envall -n "${NGPU_TOTAL}" --ppn "${NGPU_PER_NODE}" "${CPU_BIND_SCHEME}"
          -genv I_MPI_HYDRA_BOOTSTRAP pmi
          -genv I_MPI_FABRICS ofi
          -genv I_MPI_OFI_PROVIDER "${BIOM3_FI_PROVIDER:-tcp}"
          -genv FI_PROVIDER "${BIOM3_FI_PROVIDER:-tcp}")
if [[ "${NGPU_TOTAL}" -gt "${NGPU_PER_NODE}" ]]; then
    : "${PBS_NODEFILE:?PBS_NODEFILE required for multi-node (set by PBS)}"
    MPI_ARGS+=(--hostfile "${PBS_NODEFILE}")
fi

# Rendezvous over the high-speed network interface, per the ALCF container recipe.
export MASTER_ADDR="${MASTER_ADDR:-$(hostname -s).hsn.cm.aurora.alcf.anl.gov}"
export MASTER_PORT="${MASTER_PORT:-29500}"
ENVS+=(--env "MASTER_ADDR=${MASTER_ADDR}" --env "MASTER_PORT=${MASTER_PORT}")

ENVS+=(--env "BIOM3_WORLD_SIZE=${NGPU_TOTAL}")

# Rank translation, done INSIDE the container because PALS_RANKID differs per
# rank and --env would apply one value to all of them.
#
# Lightning auto-detects its ClusterEnvironment. On bare metal MPIEnvironment
# wins, because it asks mpi4py, which is built against Aurora's Intel MPI. The
# container's mpi4py is OpenMPI and cannot bootstrap under PALS, so detection
# falls through to LightningEnvironment, whose global_rank() is 0 for every
# process -- each rank then tries to spawn its own children and the run dies.
#
# PALS exports per-rank ids into the container regardless, so translate them
# into the standard torch variables. TorchElasticEnvironment reads exactly
# these and reports creates_processes_externally=True, which is accurate here:
# mpiexec already created the processes.
#
# BIOM3_RANK_SOURCE=mpi skips the translation entirely and lets MPIEnvironment
# detect via mpi4py, which is the native path and only works in an image whose
# MPI matches the launcher's (Dockerfile.xpu-oneapi). Note TorchElasticEnvironment
# is checked BEFORE MPIEnvironment, so the variables below suppress it.
# BIOM3_SETVARS=1 puts the base oneAPI's MPI libraries ahead of pip's. Required
# for Dockerfile.xpu-oneapi, which has two Intel MPIs: the base's (what mpi4py
# was compiled against) and pip's impi-rt, pulled in as a dependency. Left to
# itself the loader mixes them and mpi4py fails with
#   libmpifort.so.12: undefined symbol: MPIR_F_MPI_BUFFER_AUTOMATIC
#
# Only the MPI directories, NOT setvars.sh. Sourcing setvars also puts the base's
# oneAPI 2025.3 compiler runtime first, which is not the one the pip torch was
# built against, and torch then fails to import with
#   libur_loader.so.0: version `LIBUR_LOADER_0.11' not found (by libsycl.so.8)
# Leave unset for Dockerfile.xpu, which has no oneAPI installation.
SETVARS=""
[[ "${BIOM3_SETVARS:-0}" == "1" ]] && \
    SETVARS='export LD_LIBRARY_PATH="/opt/intel/oneapi/mpi/latest/lib:/opt/intel/oneapi/mpi/latest/lib/release:${LD_LIBRARY_PATH}"
'

# Appended, never prepended, and inside the container so the image's own
# LD_LIBRARY_PATH survives. /hostevent exists only so PMIx can find the host's
# libevent; nothing else should resolve there in preference to the image.
HOSTEVENT='export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/hostevent"
'

# Prepended, unlike /hostevent: the whole point is for the host libfabric to win
# over the container's, which has no cxi provider. FI_PROVIDER_PATH points
# libfabric at the host's provider plugins rather than the image's.
HOSTFABRIC=""
if [[ -n "${BIOM3_FABRIC_DIR:-}" ]]; then
    # LD_PRELOAD, not just LD_LIBRARY_PATH. pip's oneccl ships its own
    # libfabric at /opt/venv/lib/libfabric.so.1 and finds it through RPATH,
    # which LD_LIBRARY_PATH does not override -- so oneCCL kept loading a
    # libfabric with no cxi provider and failed with
    #   fi_getinfo error: ret -61, providers 0 / can't create providers for name cxi
    # while Intel MPI, which honours I_MPI_OFI_LIBRARY_INTERNAL=0, was already
    # using Cray's. Cray's libfabric needs libcxi.so.1, which lives in
    # /usr/lib64 and so resolves through the /hostevent bind.
    HOSTFABRIC='export LD_PRELOAD="/hostfabric/libfabric.so.1${LD_PRELOAD:+:${LD_PRELOAD}}"
export LD_LIBRARY_PATH="/hostfabric:${LD_LIBRARY_PATH}"
'
    # Only when the build uses loadable providers. The Cray build compiles cxi
    # in, and setting this to a directory without plugins hides the built-ins.
    [[ -d "${BIOM3_FABRIC_DIR}/prov" ]] && \
        HOSTFABRIC+='export FI_PROVIDER_PATH="/hostfabric/prov"
'
fi

if [[ "${BIOM3_RANK_SOURCE:-pals}" == "mpi" ]]; then
PRELUDE="cd /app
${HOSTEVENT}${HOSTFABRIC}${SETVARS}"'source environment.sh >&2
exec "$@"'
else
PRELUDE="cd /app
${HOSTEVENT}${HOSTFABRIC}${SETVARS}"'export RANK="${PALS_RANKID:?PALS_RANKID not set; was this launched by mpiexec?}"
export LOCAL_RANK="${PALS_LOCAL_RANKID:?PALS_LOCAL_RANKID not set}"
export LOCAL_WORLD_SIZE="${PALS_LOCAL_SIZE:?PALS_LOCAL_SIZE not set}"
export WORLD_SIZE="${BIOM3_WORLD_SIZE:?BIOM3_WORLD_SIZE not set}"
export GROUP_RANK=$(( RANK / LOCAL_WORLD_SIZE ))
export NODE_RANK="${GROUP_RANK}"
export TORCHELASTIC_RUN_ID="${TORCHELASTIC_RUN_ID:-biom3-mpi}"
source environment.sh >&2
exec "$@"'
fi

# environment.sh runs inside each rank so BIOM3_MACHINE and the Aurora oneCCL
# settings apply; the --env values above win over anything it sets.
set -- bash -lc "${PRELUDE}" _ "$@"

echo "+ mpiexec ${MPI_ARGS[*]} apptainer exec --writable-tmpfs --bind ${BIND_ARG} ${SIF} <cmd>" >&2
exec mpiexec "${MPI_ARGS[@]}" \
    apptainer exec --writable-tmpfs --bind "${BIND_ARG}" "${ENVS[@]}" "${SIF}" "$@"

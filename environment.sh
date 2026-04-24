# BioM3 environment variables
# Source this file before running tests or scripts: source environment.sh
#
# Common variables are set unconditionally. Machine-specific variables are
# added based on hostname detection (Polaris, Aurora, DGX Spark).

# --- Common (all machines) ---
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

# --- Machine detection ---
_hostname="$(hostname)"

if [[ "$_hostname" == x3* ]] || [[ "$_hostname" == polaris-login* ]]; then
    # Polaris (ALCF) — NVIDIA GPUs
    echo "[environment.sh] Detected Polaris"

elif [[ "$_hostname" == x4* ]] || [[ "$_hostname" == aurora-uan* ]]; then
    # Aurora (ALCF) — Intel GPUs
    export NUMEXPR_MAX_THREADS=64
    # Override the frameworks/2025.3.1 default `opencl:gpu;level_zero:gpu`,
    # which ALCF explicitly flags as potentially problematic for distributed jobs.
    export ONEAPI_DEVICE_SELECTOR="level_zero:gpu"

    # ALCF-recommended oneCCL environment (see user-guides/aurora/data-science/
    # frameworks/oneCCL.md). The MPI transport variables below are honored only
    # under an MPI launcher — launch via the stage*_train_{singlenode,multinode}.sh
    # wrappers, which use mpiexec. Without mpiexec, oneCCL silently falls back
    # to OFI/ATL and these are ignored.
    export CCL_PROCESS_LAUNCHER=pmix
    export CCL_ATL_TRANSPORT=mpi
    export CCL_KVS_MODE=mpi
    export FI_MR_CACHE_MONITOR=userfaultfd
    # Hang avoidance — pairs with CCL_OP_SYNC (already set by frameworks module).
    export CCL_ATL_SYNC_COLL=1
    # Pin CCL progress threads to last 6 cores of each socket, disjoint from
    # framework cores (see --cpu-bind list in the train scripts).
    export CCL_WORKER_AFFINITY=42,43,44,45,46,47,94,95,96,97,98,99
    # Avoid `AF_UNIX path too long` on Lightning DataLoader workers
    # (known-issues.md #7).
    export TMPDIR=/tmp

    echo "[environment.sh] Detected Aurora"

elif [[ "$_hostname" == spark* ]]; then
    # DGX Spark — single NVIDIA GPU
    echo "[environment.sh] Detected DGX Spark"

else
    echo "[environment.sh] Unknown machine: $_hostname (using common settings only)"
fi

unset _hostname

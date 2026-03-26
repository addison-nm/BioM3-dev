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
    export ONEAPI_DEVICE_SELECTOR="level_zero:gpu"
    echo "[environment.sh] Detected Aurora"

elif [[ "$_hostname" == spark* ]]; then
    # DGX Spark — single NVIDIA GPU
    echo "[environment.sh] Detected DGX Spark"

else
    echo "[environment.sh] Unknown machine: $_hostname (using common settings only)"
fi

unset _hostname

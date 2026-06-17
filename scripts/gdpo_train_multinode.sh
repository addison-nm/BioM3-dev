#!/usr/bin/env bash
#=============================================================================
#
# FILE: gdpo_train_multinode.sh
#
# USAGE: gdpo_train_multinode.sh CONFIG_PATH NUM_NODES DEVICE RUN_ID \
#        [additional --key value overrides]
#
# DESCRIPTION: Multi-node wrapper for biom3_gdpo_train. One rank per node by
#   design (NGPU_PER_NODE=1) — within each rank, the existing RolloutPool
#   fans diffusion rollouts across all local tiles via threads. World_size
#   equals num_nodes; cross-node comm is only the per-step state_dict
#   broadcast + the (small) ids/seqs/rewards gather. ESMFold is loaded once
#   per node, not per tile, which keeps HBM pressure reasonable.
#
#   --rollout_devices auto is injected unless the caller overrides it,
#   so each rank's RolloutPool spans all visible local tiles.
#
#   Dispatches to scripts/launchers/${BIOM3_MACHINE}_multinode_rl.sh.
#
#   Requires: source environment.sh first so BIOM3_MACHINE is set.
#   PBS_NODEFILE must be set by PBS at submission time.
#
#=============================================================================
set -euo pipefail

if [ "$#" -lt 4 ]; then
    echo "Usage: $0 CONFIG_PATH NUM_NODES DEVICE RUN_ID [--key value ...]"
    echo "  CONFIG_PATH  e.g. configs/grpo/production_gdpo.json"
    echo "  NUM_NODES    number of allocated nodes (= world_size, one rank per node)"
    echo "  DEVICE       cuda | xpu | cpu (forwarded as --device)"
    echo "  RUN_ID       unique identifier; outputs land at OUTPUT_ROOT/RUN_ID"
    exit 1
fi

config_path=$1
NUM_NODES=$2
device=$3
run_id=$4
shift 4

# One rank per node. RolloutPool inside each rank spans local tiles.
NGPU_PER_NODE=1
NGPU_TOTAL="${NUM_NODES}"

echo "NUM_NODES: ${NUM_NODES}  NGPU_PER_NODE: ${NGPU_PER_NODE}  NGPU_TOTAL: ${NGPU_TOTAL} (${device})"

export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=true

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

MACHINE="${BIOM3_MACHINE:?BIOM3_MACHINE not set; source environment.sh first}"
LAUNCHER="${SCRIPT_DIR}/launchers/${MACHINE}_multinode_rl.sh"

if [ ! -x "${LAUNCHER}" ]; then
    echo "ERROR: no RL multinode launcher for ${MACHINE} at ${LAUNCHER}"
    exit 1
fi

export NGPU_PER_NODE NGPU_TOTAL

# Inject --rollout_devices auto unless the caller already passed one.
inject_rollout_devices=1
for arg in "$@"; do
    if [ "${arg}" = "--rollout_devices" ]; then
        inject_rollout_devices=0
        break
    fi
done
rollout_devices_args=()
if [ "${inject_rollout_devices}" = "1" ]; then
    rollout_devices_args=(--rollout_devices auto)
fi

exec "${LAUNCHER}" \
    biom3_gdpo_train \
        --config_path "${config_path}" \
        --run_id "${run_id}" \
        --device "${device}" \
        "${rollout_devices_args[@]}" \
        "$@"

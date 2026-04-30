#!/usr/bin/env bash
#=============================================================================
#
# FILE: test_stage3_multinode.sh
#
# USAGE: test_stage3_multinode.sh NUM_NODES NGPU_PER_NODE [extra pytest args...]
#
# DESCRIPTION: Multi-node multi-GPU launcher for the Stage 3 sampling smoke
#   test suite (tests/stage3_tests/test_multinode_sample.py). Dispatches to
#   the machine-specific launcher under
#   scripts/launchers/${BIOM3_MACHINE}_multinode.sh, which wraps the call
#   in mpiexec with the appropriate per-tile CPU binding.
#
#   The smoke suite verifies that, under the requested topology:
#     * every rank loads the model with identical weights,
#     * every rank can run sampling end-to-end,
#     * gathered outputs match a single-rank reference (world-size invariance).
#
#   Requires: source environment.sh first (sets BIOM3_MACHINE).
#   PBS_NODEFILE must be set by PBS at submission time.
#
#=============================================================================
set -euo pipefail

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 NUM_NODES NGPU_PER_NODE [extra pytest args...]"
    exit 1
fi

NUM_NODES=$1
NGPU_PER_NODE=$2
shift 2

NGPU_TOTAL="$((NUM_NODES * NGPU_PER_NODE))"

echo "NUM_NODES: ${NUM_NODES}, NGPU_PER_NODE: ${NGPU_PER_NODE}, NGPU_TOTAL: ${NGPU_TOTAL}"

export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=true
export NGPU_PER_NODE NGPU_TOTAL

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MACHINE="${BIOM3_MACHINE:?BIOM3_MACHINE not set; source environment.sh first}"
LAUNCHER="${SCRIPT_DIR}/launchers/${MACHINE}_multinode.sh"

if [ ! -x "${LAUNCHER}" ]; then
    echo "ERROR: no launcher for ${MACHINE} multinode at ${LAUNCHER}"
    exit 1
fi

exec "${LAUNCHER}" \
    python -m pytest \
        tests/stage3_tests/test_multinode_sample.py \
        --multinode "${NUM_NODES}" \
        --multidevice "${NGPU_PER_NODE}" \
        -m multinode \
        -p no:randomly \
        -v \
        "$@"

#!/usr/bin/env bash
#=============================================================================
#
# FILE: aurora_multinode.sh
#
# USAGE: aurora_multinode.sh ENTRYPOINT [args...]
#
# DESCRIPTION: Aurora multi-node launcher. Wraps the entry point in
#   `mpiexec --envall -n NGPU_TOTAL --ppn NGPU_PER_NODE --hostfile=$PBS_NODEFILE`
#   with the same 12-tile CPU binding as aurora_singlenode.sh applied per node.
#   Required env: NGPU_PER_NODE (12 for full Aurora node), NGPU_TOTAL
#   (NGPU_PER_NODE * num_nodes), and PBS_NODEFILE (set by PBS).
#
#=============================================================================
set -euo pipefail

NGPU_PER_NODE="${NGPU_PER_NODE:?NGPU_PER_NODE env var required}"
NGPU_TOTAL="${NGPU_TOTAL:?NGPU_TOTAL env var required}"
PBS_NODEFILE="${PBS_NODEFILE:?PBS_NODEFILE env var required (set by PBS)}"

CPU_BIND_SCHEME="--cpu-bind=list:1-8:9-16:17-24:25-32:33-40:41-48:53-60:61-68:69-76:77-84:85-92:93-100"
GPU_BIND_SCHEME="--gpu-bind=list:0:1:2:3:4:5:6:7:8:9:10:11"

if [ "${NGPU_PER_NODE}" != "12" ]; then
    echo "WARNING (aurora_multinode.sh): NGPU_PER_NODE=${NGPU_PER_NODE} but" \
         "the CPU_BIND list assumes 12 tiles per node. Adjust if needed."
fi

exec mpiexec \
    --verbose \
    --envall \
    -n "${NGPU_TOTAL}" \
    --ppn "${NGPU_PER_NODE}" \
    --hostfile="${PBS_NODEFILE}" \
    ${CPU_BIND_SCHEME} \
    ${GPU_BIND_SCHEME} \
    "$@"

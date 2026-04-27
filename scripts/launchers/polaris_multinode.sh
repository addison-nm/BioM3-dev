#!/usr/bin/env bash
#=============================================================================
#
# FILE: polaris_multinode.sh
#
# USAGE: polaris_multinode.sh ENTRYPOINT [args...]
#
# DESCRIPTION: Polaris (ALCF) multi-node launcher. 4 NVIDIA A100 GPUs per
#   node. Required env: NGPU_PER_NODE (typically 4), NGPU_TOTAL (4 *
#   num_nodes), PBS_NODEFILE (set by PBS).
#
# STATUS: Initial template — verify on first multi-node run. Polaris uses
#   PALS so this mirrors aurora_multinode.sh structurally; CPU binding is
#   left out pending the right AMD EPYC core layout. NCCL is the default
#   backend on Polaris (handled by Lightning automatically when XPU isn't
#   detected).
#
#=============================================================================
set -euo pipefail

NGPU_PER_NODE="${NGPU_PER_NODE:?NGPU_PER_NODE env var required}"
NGPU_TOTAL="${NGPU_TOTAL:?NGPU_TOTAL env var required}"
PBS_NODEFILE="${PBS_NODEFILE:?PBS_NODEFILE env var required (set by PBS)}"

exec mpiexec \
    --verbose \
    --envall \
    -n "${NGPU_TOTAL}" \
    --ppn "${NGPU_PER_NODE}" \
    --hostfile="${PBS_NODEFILE}" \
    "$@"

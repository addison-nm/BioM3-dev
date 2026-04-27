#!/usr/bin/env bash
#=============================================================================
#
# FILE: polaris_singlenode.sh
#
# USAGE: polaris_singlenode.sh ENTRYPOINT [args...]
#
# DESCRIPTION: Polaris (ALCF) single-node launcher. Each Polaris node has
#   4 NVIDIA A100 GPUs and a 32-core AMD EPYC CPU. Required env: NGPU
#   (number of GPUs to use, typically 4).
#
# STATUS: Initial template — the mpiexec invocation below mirrors Aurora's
#   PALS-style launch since Polaris also uses Cray PALS. CPU binding is
#   commented out pending the right Polaris core layout (AMD EPYC, 32
#   physical cores, NUMA per chiplet). When you run this for the first
#   time, verify with `taskset -cp` and add a --cpu-bind list once the
#   binding is settled.
#
#=============================================================================
set -euo pipefail

NGPU="${NGPU:?NGPU env var required}"

exec mpiexec \
    --verbose \
    --envall \
    -n "${NGPU}" \
    --ppn "${NGPU}" \
    "$@"

#!/usr/bin/env bash
#=============================================================================
#
# FILE: aurora_singlenode.sh
#
# USAGE: aurora_singlenode.sh ENTRYPOINT [args...]
#
# DESCRIPTION: Aurora single-node launcher. Wraps the entry point in
#   `mpiexec --envall -n NGPU --ppn NGPU` with the ALCF-canonical 12-tile
#   CPU binding. Required env: NGPU (number of tiles, must be 12 for full
#   node use). Pairs ranks with adjacent GPUs:
#     ranks  0,1 -> GPU 0  (cores  1-8 / 9-16)   socket 0
#     ranks  2,3 -> GPU 1  (cores 17-24 / 25-32) socket 0
#     ranks  4,5 -> GPU 2  (cores 33-40 / 41-48) socket 0
#     ranks  6,7 -> GPU 3  (cores 53-60 / 61-68) socket 1
#     ranks  8,9 -> GPU 4  (cores 69-76 / 77-84) socket 1
#     ranks 10,11-> GPU 5  (cores 85-92 / 93-100) socket 1
#   See docs/debug_aurora.md for topology rationale.
#
#=============================================================================
set -euo pipefail

NGPU="${NGPU:?NGPU env var required}"

CPU_BIND_SCHEME="--cpu-bind=list:1-8:9-16:17-24:25-32:33-40:41-48:53-60:61-68:69-76:77-84:85-92:93-100"
GPU_BIND_SCHEME="--gpu-bind=list:0.0:0.1:1.0:1.1:2.0:2.1:3.0:3.1:4.0:4.1:5.0:5.1"

if [ "${NGPU}" != "12" ]; then
    echo "WARNING (aurora_singlenode.sh): NGPU=${NGPU} but the CPU_BIND list" \
         "assumes NGPU=12. Adjust the list if running with a different rank count."
fi

exec mpiexec \
    --verbose \
    --envall \
    -n "${NGPU}" \
    --ppn "${NGPU}" \
    ${CPU_BIND_SCHEME} \
    ${GPU_BIND_SCHEME} \
    "$@"

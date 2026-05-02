#!/usr/bin/env bash
# Run the 4 smoke probes under direct mpiexec at one or more NGPU values
# (default: 2 12). Edit the two BIND vars below to test different bind
# configurations without touching the production launcher scripts.
# Run from the project root.

CPU_BIND="--cpu-bind=list:1-8:9-16:17-24:25-32:33-40:41-48:53-60:61-68:69-76:77-84:85-92:93-100"
GPU_BIND="--gpu-bind=list:0:1:2:3:4:5:6:7:8:9:10:11"

[ $# -eq 0 ] && set -- 2 12

for n in "$@"; do
    MPIEXEC="mpiexec --envall -n $n --ppn $n $CPU_BIND $GPU_BIND"
    $MPIEXEC python tests/_smoke/env_dump.py -l
    $MPIEXEC python tests/_smoke/device_probe.py
    $MPIEXEC python tests/_smoke/rank_audit.py
    timeout 120 $MPIEXEC python tests/_smoke/pg_probe.py
done

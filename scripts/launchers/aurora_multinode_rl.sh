#!/usr/bin/env bash
#=============================================================================
#
# FILE: aurora_multinode_rl.sh
#
# USAGE: aurora_multinode_rl.sh ENTRYPOINT [args...]
#
# DESCRIPTION: Aurora multi-node launcher for RL trainers (GDPO/GRPO).
#   One rank per node (`--ppn 1`), no tile-level CPU/GPU binding — the
#   single rank's process must see all 12 local Aurora tiles so the
#   in-process RolloutPool can fan rollouts across them via
#   `torch.xpu.device_count()`.
#
#   Differs from aurora_multinode.sh (used by Stage 3 distributed training):
#     - `--ppn 1` instead of `--ppn ${NGPU_PER_NODE}` (=12).
#     - No `--cpu-bind` list (that scheme assumes 12 ranks per node).
#     - `ZE_AFFINITY_MASK` is unset before exec so each rank sees all tiles.
#
#   Required env: NGPU_TOTAL (= NUM_NODES); PBS_NODEFILE (set by PBS).
#
#=============================================================================
set -euo pipefail

NGPU_TOTAL="${NGPU_TOTAL:?NGPU_TOTAL env var required (= num_nodes)}"
PBS_NODEFILE="${PBS_NODEFILE:?PBS_NODEFILE env var required (set by PBS)}"

# Optional ``HOSTFILE_OVERRIDE``: a path to a custom hostfile that is a
# subset of the PBS allocation. Used by the combined scaling-study job
# which requests 512 nodes (the prod-queue minimum) and runs sub-jobs
# at N ∈ {8, 16, ..., 512} sequentially against subset hostfiles
# carved from ``$PBS_NODEFILE``. When unset, falls back to the full
# PBS_NODEFILE (the standard per-N submit path).
HOSTFILE="${HOSTFILE_OVERRIDE:-${PBS_NODEFILE}}"

# Each rank's process must see all local tiles. Strip any tile pin the
# parent shell may have inherited from PBS/PALS defaults.
unset ZE_AFFINITY_MASK || true

# CCL_WORKER_AFFINITY is set in environment.sh to a 12-slot mask
# (8,16,…,100) that's designed for the Stage 3 training launcher's
# 12-ranks-per-node + 8-cores-per-rank layout. The RL launcher uses
# ``--ppn 1`` (one rank per node, RolloutPool fans across local tiles
# in-process), so a single rank inherits all 12 affinity slots and
# oneCCL spawns 12 worker threads contending in one process. Unset so
# oneCCL falls back to its auto-affinity policy.
unset CCL_WORKER_AFFINITY || true

# Export WORLD_SIZE explicitly so each rank's torch.distributed init
# resolves the right size without depending on PBS_NODEFILE surviving
# the env chain (the PBS path counts PBS_NODEFILE lines; interactive ssh
# sessions can lose that env var). Cheap and defensive.
export WORLD_SIZE="${NGPU_TOTAL}"

# Per-rank identity echo. Each spawned rank prints its PMI/PALS env
# BEFORE the trainer starts, so the log captures whether the sub-run's
# ranks all see the same MASTER_ADDR/PORT/WORLD_SIZE (i.e. the env is
# correctly isolated from any concurrent sub-run in the same PBS
# allocation). $0 is set to ``rank-diag-wrapper`` so the bash -c
# positional args expand cleanly.
exec mpiexec \
    --verbose \
    --envall \
    -n "${NGPU_TOTAL}" \
    --ppn 1 \
    --hostfile="${HOSTFILE}" \
    bash -c '
        echo "[rank-diag] $(date -Iseconds) host=$(hostname -s) PMI_RANK=${PMI_RANK:-?} PMI_SIZE=${PMI_SIZE:-?} PALS_RANKID=${PALS_RANKID:-?} PALS_NODEID=${PALS_NODEID:-?} PALS_LOCAL_RANKID=${PALS_LOCAL_RANKID:-?} WORLD_SIZE=${WORLD_SIZE:-?} RANK=${RANK:-?} MASTER_ADDR=${MASTER_ADDR:-?} MASTER_PORT=${MASTER_PORT:-?} HOSTFILE_OVERRIDE=${HOSTFILE_OVERRIDE:-?} CCL_LOG_FILE=${CCL_LOG_FILE:-?}"
        exec "$@"
    ' rank-diag-wrapper "$@"

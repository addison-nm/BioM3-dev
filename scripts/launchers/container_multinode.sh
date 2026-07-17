#!/usr/bin/env bash
#=============================================================================
#
# FILE: container_multinode.sh
#
# USAGE: container_multinode.sh ENTRYPOINT [args...]
#
# DESCRIPTION: Multi-node launcher for containerized commercial-cloud
#   environments (BIOM3_MACHINE=container) under SkyPilot. Whereas the ALCF
#   launchers use mpiexec/PALS, here we use a torchrun static rendezvous:
#   torchrun spawns one process per GPU and sets RANK / LOCAL_RANK /
#   WORLD_SIZE / GROUP_RANK / MASTER_ADDR / MASTER_PORT, which
#   biom3.core._dist_env and PyTorch Lightning already read. No Python change.
#
#   SkyPilot runs the task once per NODE (not per GPU) and sets
#   SKYPILOT_NODE_RANK / SKYPILOT_NUM_NODES / SKYPILOT_NODE_IPS. This launcher
#   fills NUM_NODES / NODE_RANK / MASTER_ADDR from those (head = first IP)
#   unless they are preset.
#
#   Required env: NGPU_PER_NODE (GPUs per node). NCCL runs over the cloud
#   private network: the private-net interface is auto-detected for
#   NCCL_SOCKET_IFNAME and InfiniBand is disabled (spot instances have none).
#
#=============================================================================
set -euo pipefail

NGPU_PER_NODE="${NGPU_PER_NODE:?NGPU_PER_NODE env var required}"
NUM_NODES="${NUM_NODES:-${SKYPILOT_NUM_NODES:?NUM_NODES or SKYPILOT_NUM_NODES required}}"
NODE_RANK="${NODE_RANK:-${SKYPILOT_NODE_RANK:-0}}"

# Head node = first line of SKYPILOT_NODE_IPS (SkyPilot lists the head first).
if [ -z "${MASTER_ADDR:-}" ]; then
    MASTER_ADDR="$(printf '%s\n' "${SKYPILOT_NODE_IPS:?MASTER_ADDR or SKYPILOT_NODE_IPS required}" | awk 'NF{print; exit}')"
fi
MASTER_PORT="${MASTER_PORT:-29500}"

# NCCL over the cloud private network. Auto-detect the private-net (10.x) iface
# unless NCCL_SOCKET_IFNAME is preset; assume no InfiniBand on spot instances.
if [ -z "${NCCL_SOCKET_IFNAME:-}" ]; then
    iface="$(ip -o -4 addr show 2>/dev/null | awk '/ 10\./{print $2; exit}')"
    [ -n "${iface}" ] && export NCCL_SOCKET_IFNAME="${iface}"
fi
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"

# torchrun launches a Python *script file*, not a console-script name on PATH.
# Resolve the first token to its path (as container_singlenode.sh does).
entry="$1"; shift
entry_path="$(command -v "${entry}" || true)"
[ -n "${entry_path}" ] || entry_path="${entry}"

echo "[container_multinode] node_rank=${NODE_RANK}/${NUM_NODES} nproc=${NGPU_PER_NODE}" \
     "master=${MASTER_ADDR}:${MASTER_PORT} iface=${NCCL_SOCKET_IFNAME:-auto}" >&2

exec torchrun \
    --nnodes="${NUM_NODES}" \
    --node-rank="${NODE_RANK}" \
    --master-addr="${MASTER_ADDR}" \
    --master-port="${MASTER_PORT}" \
    --nproc-per-node="${NGPU_PER_NODE}" \
    "${entry_path}" \
    "$@"

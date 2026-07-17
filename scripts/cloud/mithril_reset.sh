#!/usr/bin/env bash
# Reset local SkyPilot/Mithril state after a failed or interrupted launch.
#
# A failed/Ctrl-C'd `mithril launch` can leave a zombie `SkyPilot:executor:long`
# process behind; the next launch then fails with a *misleading*
# `ResourcesUnavailableError` (it is NOT actual GPU scarcity). Stopping the API
# server reaps the zombie; the next `mithril launch` starts a fresh server.
#
# Usage: scripts/cloud/mithril_reset.sh   (then re-run your mithril launch)
set -uo pipefail

echo "[reset] stopping SkyPilot API server (reaps zombie executors)..."
mithril sky api stop || true

echo "[reset] reaping any orphaned executors the stop missed..."
pkill -f 'SkyPilot:executor' 2>/dev/null || true

echo "[reset] clearing stale per-cluster lock files..."
rm -f ~/.sky/locks/.*_status.lock ~/.sky/locks/.ssh_config_*.lock 2>/dev/null || true

remaining=$(pgrep -fc 'SkyPilot:executor|sky\.server' 2>/dev/null || echo 0)
echo "[reset] done. Residual sky processes: ${remaining}. Re-run your mithril launch."

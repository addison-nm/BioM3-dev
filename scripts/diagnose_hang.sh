#!/usr/bin/env bash
#=============================================================================
#
# FILE: scripts/diagnose_hang.sh
#
# USAGE: scripts/diagnose_hang.sh [OUTPUT_DIR]
#
# DESCRIPTION: Sudo-less, gdb-less diagnostic script to run on an Aurora
#   compute node while a biom3 distributed training job is hung. Collects
#   per-process /proc state, kernel wait channels, and per-thread snapshots
#   for every rank process of the hung job. Takes two snapshots 30 seconds
#   apart so context-switch counter deltas can distinguish "blocked in
#   kernel" from "busy spinning in userspace".
#
#   OUTPUT_DIR defaults to ./hang_diag_<timestamp>/. Inside you'll get:
#     summary.txt          — top-of-stack + proc state per rank, both snapshots
#     procstate_T0.txt     — /proc/<pid>/{status,wchan,syscall} at T=0
#     procstate_T30.txt    — same at T+30s
#     pyspy_threads_<pid>.txt — per-thread py-spy dump for each rank
#     threads_<pid>.txt    — /proc/<pid>/task/<tid>/{status,wchan} per thread
#     stack_<pid>.txt      — /proc/<pid>/stack if readable
#
# PREREQUISITES:
#   - py-spy installed in the active venv (tried via absolute path from
#     `which python`). Missing py-spy is non-fatal; native /proc info is
#     captured either way.
#   - /proc/<pid>/wchan and /proc/<pid>/syscall readable for user-owned
#     processes (always true on Aurora compute nodes).
#
#=============================================================================
set -uo pipefail

OUTPUT_DIR="${1:-./hang_diag_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "${OUTPUT_DIR}"

# Try to locate py-spy via the active Python's bin directory so a PATH
# hiccup doesn't silently lose per-thread dumps.
PY_SPY=""
if command -v py-spy >/dev/null 2>&1; then
    PY_SPY="$(command -v py-spy)"
elif command -v python >/dev/null 2>&1; then
    PYBIN="$(dirname "$(command -v python)")"
    if [ -x "${PYBIN}/py-spy" ]; then
        PY_SPY="${PYBIN}/py-spy"
    fi
fi

if [ -n "${PY_SPY}" ]; then
    echo "[diagnose_hang.sh] using py-spy: ${PY_SPY}"
else
    echo "[diagnose_hang.sh] WARNING: py-spy not found; skipping Python-level dumps."
    echo "    install with: \"\$(which python)\" -m pip install py-spy"
fi

# Identify all rank PIDs of the hung job. Matches biom3 stage 1/2/3 entry
# points, plus a generic biom3-pretrain fallback.
RANK_PIDS=$(pgrep -u "$USER" -f 'biom3_pretrain_stage[123]' 2>/dev/null | sort -n)
if [ -z "${RANK_PIDS}" ]; then
    echo "[diagnose_hang.sh] ERROR: no biom3_pretrain_stage{1,2,3} rank processes found."
    echo "    hint: pgrep -u $USER -af python  to see what's running."
    exit 1
fi

# Rough heuristic: the main rank PIDs are the ones whose parent PID is
# the mpiexec launcher, not other ranks (which would indicate dataloader
# worker children). We emit everything into the diagnostic dir but print
# rank-vs-child hints in the summary.
echo "[diagnose_hang.sh] identified $(echo "${RANK_PIDS}" | wc -l) candidate PIDs"
echo "${RANK_PIDS}" > "${OUTPUT_DIR}/pids_all.txt"

# Snapshot /proc state at T=0 and again at T+30s. ctxt-switch counters
# that don't move across 30s => process isn't being scheduled => kernel
# wait (driver / ioctl / collective), not userspace spin.
snapshot_procstate() {
    local label="$1"
    local outfile="${OUTPUT_DIR}/procstate_${label}.txt"
    for pid in ${RANK_PIDS}; do
        if [ ! -d "/proc/${pid}" ]; then
            echo "=== pid ${pid} :: GONE ===" >> "${outfile}"
            continue
        fi
        {
            echo "=== pid ${pid} @ ${label} ==="
            # ppid helps separate launcher vs ranks vs dataloader workers
            echo "ppid:       $(cat /proc/${pid}/status | awk '/^PPid:/{print $2}')"
            echo "state:      $(awk '/^State:/{for(i=2;i<=NF;++i) printf "%s ",$i; print ""}' /proc/${pid}/status)"
            echo "wchan:      $(cat /proc/${pid}/wchan 2>/dev/null || echo '(unreadable)')"
            echo "syscall:    $(cat /proc/${pid}/syscall 2>/dev/null | awk '{print $1}' || echo '(unreadable)')"
            echo "vctxt:      $(awk '/^voluntary_ctxt_switches:/{print $2}' /proc/${pid}/status)"
            echo "nvctxt:     $(awk '/^nonvoluntary_ctxt_switches:/{print $2}' /proc/${pid}/status)"
            echo "threads:    $(awk '/^Threads:/{print $2}' /proc/${pid}/status)"
            echo "cmd:        $(tr '\0' ' ' < /proc/${pid}/cmdline | cut -c1-160)"
            echo ""
        } >> "${outfile}"
    done
}

echo "[diagnose_hang.sh] snapshot 1 of 2 (T=0)"
snapshot_procstate T0

# Per-thread snapshots of every rank. Fast.
echo "[diagnose_hang.sh] per-thread /proc dumps"
for pid in ${RANK_PIDS}; do
    outfile="${OUTPUT_DIR}/threads_${pid}.txt"
    if [ ! -d "/proc/${pid}/task" ]; then
        continue
    fi
    {
        echo "=== pid ${pid} threads ==="
        for tdir in /proc/${pid}/task/*; do
            tid=$(basename "${tdir}")
            state=$(awk '/^State:/{print $2,$3}' "${tdir}/status" 2>/dev/null)
            wchan=$(cat "${tdir}/wchan" 2>/dev/null || echo '(unreadable)')
            echo "  tid=${tid} state=${state} wchan=${wchan}"
        done
    } > "${outfile}"
done

# Try to read /proc/<pid>/stack — often readable for user-owned procs.
echo "[diagnose_hang.sh] /proc/<pid>/stack attempts"
for pid in ${RANK_PIDS}; do
    outfile="${OUTPUT_DIR}/stack_${pid}.txt"
    if [ -r "/proc/${pid}/stack" ]; then
        {
            echo "=== pid ${pid} kernel stack ==="
            cat "/proc/${pid}/stack" 2>/dev/null
        } > "${outfile}"
    else
        echo "(not readable: /proc/${pid}/stack)" > "${outfile}"
    fi
done

# Full py-spy dumps (if py-spy is available). `py-spy dump` already prints
# every thread in the process by default, so no `--threads` flag is needed
# (and it's rejected by the py-spy on Aurora). The resulting files capture
# CCL worker threads and DDP hook threads alongside MainThread.
if [ -n "${PY_SPY}" ]; then
    echo "[diagnose_hang.sh] py-spy full dumps (all threads per rank)"
    for pid in ${RANK_PIDS}; do
        outfile="${OUTPUT_DIR}/pyspy_threads_${pid}.txt"
        "${PY_SPY}" dump --pid "${pid}" > "${outfile}" 2>&1
    done
fi

# Wait 30s for the second snapshot (ctxt-switch delta).
echo "[diagnose_hang.sh] sleeping 30s for second snapshot..."
sleep 30
echo "[diagnose_hang.sh] snapshot 2 of 2 (T=30)"
snapshot_procstate T30

# Build a small summary diffing the two snapshots — which PIDs moved,
# which are frozen.
SUMMARY="${OUTPUT_DIR}/summary.txt"
{
    echo "=========================================================="
    echo "  biom3 hang diagnostic summary"
    echo "  node: $(hostname)"
    echo "  time: $(date -Iseconds)"
    echo "  ${PY_SPY:-no py-spy}"
    echo "=========================================================="
    echo ""
    echo "## PIDs captured"
    echo "$(echo "${RANK_PIDS}" | wc -l) PIDs"
    echo ""
    echo "## State + wchan + ctxt-switch delta (T30 - T0)"
    echo ""
    printf "%-10s %-6s %-6s %-32s %-12s %-12s %s\n" \
        PID PPID STATE WCHAN VCTXT_DELTA NVCTXT_DELTA SYSCALL
    for pid in ${RANK_PIDS}; do
        if [ ! -d "/proc/${pid}" ]; then
            printf "%-10s (gone)\n" "${pid}"
            continue
        fi
        ppid=$(awk '/^PPid:/{print $2}' /proc/${pid}/status 2>/dev/null)
        state=$(awk '/^State:/{print $2}' /proc/${pid}/status 2>/dev/null)
        wchan=$(cat /proc/${pid}/wchan 2>/dev/null | head -c 32)
        syscall=$(cat /proc/${pid}/syscall 2>/dev/null | awk '{print $1}')
        vc_t30=$(awk '/^voluntary_ctxt_switches:/{print $2}' /proc/${pid}/status 2>/dev/null)
        nvc_t30=$(awk '/^nonvoluntary_ctxt_switches:/{print $2}' /proc/${pid}/status 2>/dev/null)
        vc_t0=$(grep -A 20 "^=== pid ${pid} @ T0 ===" "${OUTPUT_DIR}/procstate_T0.txt" | awk '/^vctxt:/{print $2; exit}')
        nvc_t0=$(grep -A 20 "^=== pid ${pid} @ T0 ===" "${OUTPUT_DIR}/procstate_T0.txt" | awk '/^nvctxt:/{print $2; exit}')
        vc_delta=$(( ${vc_t30:-0} - ${vc_t0:-0} ))
        nvc_delta=$(( ${nvc_t30:-0} - ${nvc_t0:-0} ))
        printf "%-10s %-6s %-6s %-32s %-12s %-12s %s\n" \
            "${pid}" "${ppid}" "${state}" "${wchan:-?}" "${vc_delta}" "${nvc_delta}" "${syscall:-?}"
    done
    echo ""
    echo "How to read the columns:"
    echo "  STATE: R=running, S=interruptible sleep, D=kernel wait (driver/ioctl)"
    echo "  WCHAN: kernel function the process is parked in"
    echo "         futex_*      -> userspace lock"
    echo "         ze_*, i915_* -> blocked in Intel GPU driver"
    echo "         do_sys_poll  -> epoll/select wait"
    echo "  VCTXT_DELTA, NVCTXT_DELTA: context-switch counters over the 30s window"
    echo "         both ~0 -> process is not being scheduled (true kernel wait)"
    echo "         >0      -> process is making some progress"
    echo ""
    if [ -n "${PY_SPY}" ]; then
        echo "## py-spy MainThread top-of-stack (T30, refreshed)"
        for pid in ${RANK_PIDS}; do
            echo "=== pid ${pid} ==="
            "${PY_SPY}" dump --pid "${pid}" 2>/dev/null | grep -E "^    " | head -3 || \
                echo "  (py-spy dump failed)"
        done
    fi
} > "${SUMMARY}"

echo ""
echo "[diagnose_hang.sh] done. output in: ${OUTPUT_DIR}/"
echo "[diagnose_hang.sh] summary: ${SUMMARY}"
echo ""
cat "${SUMMARY}"

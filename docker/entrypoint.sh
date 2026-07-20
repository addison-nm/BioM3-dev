#!/usr/bin/env bash
#=============================================================================
#
# FILE: entrypoint.sh  (BioM3 GPU image — AWS / Mithril)
#
# Sets up the BioM3 environment, optionally syncs weights/data from an object
# store, then exec's the requested command (an interactive shell, a biom3_*
# CLI, or a scripts/*_train_*.sh wrapper).
#
# DEFAULT (no sync env vars): a no-op besides sourcing environment.sh. Supply
# weights/data/outputs via bind-mounts (see docker/run.sh, docker/README.md).
#
# OPTIONAL object-store sync (for fresh spot instances with no pre-staged
# volume). Tool-agnostic — set BIOM3_SYNC_CMD to use any tool (rclone, gsutil,
# ...), otherwise s3:// URIs use awscli IFF it is installed (build with
# --build-arg INSTALL_AWSCLI=true). Env vars:
#   BIOM3_WEIGHTS_URI       e.g. s3://bucket/biom3/weights  -> /app/weights
#   BIOM3_DATA_URI          e.g. s3://bucket/biom3/data     -> /app/data
#   BIOM3_WEIGHTS_INCLUDES  required with BIOM3_WEIGHTS_URI; space-separated
#                           --include globs (weights only), prefixed by
#                           --exclude "*". Use "*" to sync the whole tree.
#   BIOM3_SYNC_MODE         auto (default) | always | never
#                             auto   = skip if the dest dir already has files
#                             always = sync (aws/rclone only re-transfer deltas)
#                             never  = skip entirely
#   BIOM3_SYNC_CMD          custom pull command; runs via `bash -c` with
#                           BIOM3_SYNC_URI and BIOM3_SYNC_DEST exported. When
#                           set, it is used for ALL pulls instead of awscli.
#   BIOM3_OUTPUTS_PUSH_URI  optional; push /app/outputs here on exit
#                           (best-effort; the mounted volume is the primary
#                           persistence mechanism). Uses BIOM3_SYNC_CMD_OUT if
#                           set, else awscli for s3:// targets.
#
#=============================================================================
set -euo pipefail

# BIOM3_MACHINE and TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD come from the image's ENV.
# A device-specific env step (as environment.sh does for XPU) will be needed here
# for a non-CUDA variant.

SYNC_MODE="${BIOM3_SYNC_MODE:-auto}"

_have_aws() { command -v aws >/dev/null 2>&1; }

# _sync_in URI DEST LABEL [extra aws filter args...]
_sync_in() {
    local uri="$1" dest="$2" label="$3"; shift 3
    local extra=("$@")
    [[ -z "${uri}" ]] && return 0
    if [[ "${SYNC_MODE}" == "never" ]]; then
        echo "[entrypoint] BIOM3_SYNC_MODE=never; skipping ${label} sync." >&2
        return 0
    fi
    if [[ "${SYNC_MODE}" == "auto" \
          && -n "$(find "${dest}" -mindepth 1 -type f -print -quit 2>/dev/null)" ]]; then
        echo "[entrypoint] ${dest} already populated; skipping ${label} sync (auto)." >&2
        return 0
    fi
    mkdir -p "${dest}"
    if [[ -n "${BIOM3_SYNC_CMD:-}" ]]; then
        echo "[entrypoint] ${label}: BIOM3_SYNC_CMD '${uri}' -> '${dest}'" >&2
        BIOM3_SYNC_URI="${uri}" BIOM3_SYNC_DEST="${dest}" bash -c "${BIOM3_SYNC_CMD}"
    elif [[ "${uri}" == s3://* ]] && _have_aws; then
        echo "[entrypoint] ${label}: aws s3 sync '${uri}' -> '${dest}'" >&2
        aws s3 sync "${uri}" "${dest}" --no-progress "${extra[@]}"
    else
        echo "[entrypoint] ERROR: cannot sync ${label} from '${uri}'." >&2
        echo "[entrypoint]   Set BIOM3_SYNC_CMD, or use an s3:// URI with awscli" >&2
        echo "[entrypoint]   (build --build-arg INSTALL_AWSCLI=true), or bind-mount ${dest}." >&2
        return 1
    fi
}

# _can_push DEST_URI — is a push mechanism available for this destination?
_can_push() {
    [[ -n "${BIOM3_SYNC_CMD_OUT:-}" ]] && return 0
    [[ "$1" == s3://* ]] && _have_aws && return 0
    return 1
}

# _sync_out DEST_URI SRC LABEL — returns non-zero if the push failed.
_sync_out() {
    local dest_uri="$1" src="$2" label="$3"
    [[ -z "${dest_uri}" ]] && return 0
    if [[ -n "${BIOM3_SYNC_CMD_OUT:-}" ]]; then
        echo "[entrypoint] Pushing ${label}: BIOM3_SYNC_CMD_OUT '${src}' -> '${dest_uri}'" >&2
        BIOM3_SYNC_SRC="${src}" BIOM3_SYNC_URI="${dest_uri}" bash -c "${BIOM3_SYNC_CMD_OUT}"
    else
        echo "[entrypoint] Pushing ${label}: aws s3 sync '${src}' -> '${dest_uri}'" >&2
        aws s3 sync "${src}" "${dest_uri}" --no-progress
    fi
}

# Build the weights --include filter list. A URI without includes is refused rather
# than pulling the whole tree; "*" opts in explicitly.
WEIGHTS_FILTER=()
if [[ -n "${BIOM3_WEIGHTS_URI:-}" ]]; then
    if [[ -z "${BIOM3_WEIGHTS_INCLUDES:-}" ]]; then
        echo "[entrypoint] ERROR: BIOM3_WEIGHTS_URI set without BIOM3_WEIGHTS_INCLUDES." >&2
        echo "[entrypoint]   Refusing a full-tree sync. Set globs, or '*' for everything." >&2
        exit 1
    fi
    WEIGHTS_FILTER+=(--exclude "*")
    set -f      # split on spaces without expanding globs against the local fs
    for pat in ${BIOM3_WEIGHTS_INCLUDES}; do
        WEIGHTS_FILTER+=(--include "${pat}")
    done
    set +f
fi

# Pulls (each a no-op when its URI is unset).
_sync_in "${BIOM3_WEIGHTS_URI:-}" /app/weights weights "${WEIGHTS_FILTER[@]}"
_sync_in "${BIOM3_DATA_URI:-}"    /app/data    data

# Run the command. With no outputs-push configured, exec for clean signal
# semantics. Otherwise run as a child so we can push /app/outputs on exit
# (forwards SIGTERM/SIGINT for graceful spot preemption).
if [[ -z "${BIOM3_OUTPUTS_PUSH_URI:-}" ]]; then
    exec "$@"
fi

# Check before running rather than after: on an ephemeral instance an unpushable
# destination means the outputs are lost when the node goes away.
if ! _can_push "${BIOM3_OUTPUTS_PUSH_URI}"; then
    echo "[entrypoint] ERROR: cannot push outputs to '${BIOM3_OUTPUTS_PUSH_URI}'" \
         "(need BIOM3_SYNC_CMD_OUT, or an s3:// target with awscli installed)." >&2
    exit 1
fi

"$@" &
child=$!
trap 'kill -TERM "${child}" 2>/dev/null || true' TERM INT
wait "${child}"; rc=$?

if ! _sync_out "${BIOM3_OUTPUTS_PUSH_URI}" /app/outputs outputs; then
    echo "[entrypoint] ERROR: outputs push to '${BIOM3_OUTPUTS_PUSH_URI}' failed;" \
         "outputs remain only on this node." >&2
    # A job failure's own code is more informative, so only override a success.
    if [[ "${rc}" -eq 0 ]]; then rc=1; fi
fi
exit "${rc}"

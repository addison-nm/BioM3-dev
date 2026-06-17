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
#   BIOM3_WEIGHTS_INCLUDES  optional space-separated --include globs (weights
#                           only); when set, prefixes --exclude "*".
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

# environment.sh honors the image's BIOM3_MACHINE=container and sets common
# vars (TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD, etc.). Required before any training
# wrapper, which reads BIOM3_MACHINE. Send its banner to stderr so the exec'd
# command's stdout stays clean (the redirect applies to `source` only).
# shellcheck disable=SC1091
source /app/environment.sh 1>&2

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

# _sync_out DEST_URI SRC LABEL  (best-effort; never fails the run)
_sync_out() {
    local dest_uri="$1" src="$2" label="$3"
    [[ -z "${dest_uri}" ]] && return 0
    if [[ -n "${BIOM3_SYNC_CMD_OUT:-}" ]]; then
        echo "[entrypoint] Pushing ${label}: BIOM3_SYNC_CMD_OUT '${src}' -> '${dest_uri}'" >&2
        BIOM3_SYNC_SRC="${src}" BIOM3_SYNC_URI="${dest_uri}" bash -c "${BIOM3_SYNC_CMD_OUT}" \
            || echo "[entrypoint] WARN: ${label} push failed." >&2
    elif [[ "${dest_uri}" == s3://* ]] && _have_aws; then
        echo "[entrypoint] Pushing ${label}: aws s3 sync '${src}' -> '${dest_uri}'" >&2
        aws s3 sync "${src}" "${dest_uri}" --no-progress \
            || echo "[entrypoint] WARN: ${label} push failed." >&2
    else
        echo "[entrypoint] WARN: cannot push ${label} to '${dest_uri}'" \
             "(need BIOM3_SYNC_CMD_OUT, or s3:// + awscli)." >&2
    fi
}

# Build the optional weights --include filter list.
WEIGHTS_FILTER=()
if [[ -n "${BIOM3_WEIGHTS_INCLUDES:-}" ]]; then
    WEIGHTS_FILTER+=(--exclude "*")
    for pat in ${BIOM3_WEIGHTS_INCLUDES}; do
        WEIGHTS_FILTER+=(--include "${pat}")
    done
fi

# Pulls (each a no-op when its URI is unset).
_sync_in "${BIOM3_WEIGHTS_URI:-}" /app/weights weights "${WEIGHTS_FILTER[@]}"
_sync_in "${BIOM3_DATA_URI:-}"    /app/data    data

# Run the command. With no outputs-push configured, exec for clean signal
# semantics. Otherwise run as a child so we can push /app/outputs on exit
# (best-effort; forwards SIGTERM/SIGINT for graceful spot preemption).
if [[ -z "${BIOM3_OUTPUTS_PUSH_URI:-}" ]]; then
    exec "$@"
fi

"$@" &
child=$!
trap 'kill -TERM "${child}" 2>/dev/null || true' TERM INT
wait "${child}"; rc=$?
_sync_out "${BIOM3_OUTPUTS_PUSH_URI}" /app/outputs outputs
exit "${rc}"

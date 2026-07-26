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
# OPTIONAL GHCR weights bundle (no object store, no credentials):
#   BIOM3_WEIGHTS_BUNDLE    e.g. run1_base -> `oras pull` the published bundle
#                           into /app/weights. The bundle's weights/ tree uses
#                           the same layout and filenames as configs/weights/,
#                           so `--weight_set configs/weights/<name>.json` then
#                           resolves. Honours BIOM3_SYNC_MODE. Public bundles
#                           need no login; a private one needs an oras login
#                           (or GHCR_TOKEN, used here to authenticate).
#   BIOM3_WEIGHTS_BUNDLE_REPO  override the default bundle repo.
#
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

# GHCR weights bundle. `oras pull` lays the artifact out as <dir>/weights/... and
# <dir>/configs/..., so pulling to a staging dir and moving the weights subtree up
# gives /app/weights the layout configs/weights/*.json expects. The bundle's own
# config fragments are provenance, not inputs — the image's configs/inference/
# already carries the matching architecture.
_pull_bundle() {
    local tag="$1" dest=/app/weights
    [[ -z "${tag}" ]] && return 0
    if [[ "${SYNC_MODE}" == "never" ]]; then
        echo "[entrypoint] BIOM3_SYNC_MODE=never; skipping weights bundle." >&2
        return 0
    fi
    if [[ "${SYNC_MODE}" == "auto" \
          && -n "$(find "${dest}" -mindepth 1 -type f -print -quit 2>/dev/null)" ]]; then
        echo "[entrypoint] ${dest} already populated; skipping bundle pull (auto)." >&2
        return 0
    fi
    command -v oras >/dev/null 2>&1 || {
        echo "[entrypoint] ERROR: BIOM3_WEIGHTS_BUNDLE set but oras is not installed." >&2
        return 1
    }
    local repo="${BIOM3_WEIGHTS_BUNDLE_REPO:-ghcr.io/natural-machine/biom3-weights}"
    local stage=/tmp/biom3-bundle
    if [[ -n "${GHCR_TOKEN:-}" ]]; then
        echo "${GHCR_TOKEN}" | oras login ghcr.io -u "${GHCR_USER:-x}" --password-stdin
    fi
    echo "[entrypoint] weights: oras pull '${repo}:${tag}' -> '${dest}'" >&2
    rm -rf "${stage}"; mkdir -p "${stage}"
    oras pull "${repo}:${tag}" -o "${stage}"
    [[ -d "${stage}/weights" ]] || {
        echo "[entrypoint] ERROR: ${repo}:${tag} has no weights/ tree." >&2
        return 1
    }
    mkdir -p "${dest}"
    cp -a "${stage}/weights/." "${dest}/"
    rm -rf "${stage}"
}

# Pulls (each a no-op when its variable is unset).
_pull_bundle "${BIOM3_WEIGHTS_BUNDLE:-}"
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

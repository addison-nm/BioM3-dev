#!/usr/bin/env bash
#=============================================================================
#
# FILE: docker/build.sh
#
# Build a BioM3 device image with the repo root as the build context. Works
# from any directory.
#
# USAGE:
#   docker/build.sh [--variant V] [--platform P] [--tag T] [--awscli] [--push] [-- <buildx args>]
#
#   --variant V    cuda | xpu (default: cuda). Selects docker/Dockerfile.<V>
#                  and defaults the tag to biom3:<V>. xpu is amd64-only.
#   --platform P   linux/amd64 | linux/arm64 | linux/amd64,linux/arm64
#                  (default: builder's native platform). Cross-arch builds
#                  need QEMU/binfmt registered in the buildx builder.
#   --tag T        image tag (default: biom3:<variant>)
#   --awscli       bake awscli in (for the entrypoint's S3 sync hook)
#   --push         push to the registry instead of loading locally
#                  (required for multi-platform builds — buildx --load is
#                  single-platform only)
#
# NETWORK NOTE: the ~3 GB torch wheel comes from download.pytorch.org (cuda) or
# Intel's index (xpu). If you're on a VPN with upstream DNS filtering (e.g.
# Tailscale MagicDNS) and the wheel download stalls on "Failed to resolve",
# disconnect the VPN for the build.
#
#=============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

VARIANT="cuda"
PLATFORM=""
TAG=""
AWSCLI="false"
PUSH=0
EXTRA=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --variant)  VARIANT="$2"; shift 2 ;;
        --platform) PLATFORM="$2"; shift 2 ;;
        --tag)      TAG="$2"; shift 2 ;;
        --awscli)   AWSCLI="true"; shift ;;
        --push)     PUSH=1; shift ;;
        --)         shift; EXTRA=("$@"); break ;;
        -h|--help)  sed -n '3,27p' "$0"; exit 0 ;;
        *)          echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

case "${VARIANT}" in
    cuda|xpu) ;;
    *) echo "ERROR: --variant must be cuda or xpu (got '${VARIANT}')." >&2; exit 1 ;;
esac

DOCKERFILE="${SCRIPT_DIR}/Dockerfile.${VARIANT}"
[[ -f "${DOCKERFILE}" ]] || { echo "ERROR: ${DOCKERFILE} not found." >&2; exit 1; }
TAG="${TAG:-biom3:${VARIANT}}"

# Intel GPU wheels are amd64-only; reject arm64 requests for xpu.
if [[ "${VARIANT}" == "xpu" && "${PLATFORM}" == *arm64* ]]; then
    echo "ERROR: the xpu variant is amd64-only (no arm64 Intel GPU wheels)." >&2
    exit 1
fi

ARGS=(buildx build -f "${DOCKERFILE}" -t "${TAG}"
      --build-arg "INSTALL_AWSCLI=${AWSCLI}")

[[ -n "${PLATFORM}" ]] && ARGS+=(--platform "${PLATFORM}")

if [[ "${PUSH}" -eq 1 ]]; then
    ARGS+=(--push)
elif [[ "${PLATFORM}" == *,* ]]; then
    echo "ERROR: multi-platform build requires --push (buildx --load is single-platform)." >&2
    exit 1
else
    ARGS+=(--load)
fi

# Append passthrough args only if any (empty-array expansion under `set -u`
# errors on macOS's bash 3.2).
if [[ ${#EXTRA[@]} -gt 0 ]]; then
    ARGS+=("${EXTRA[@]}")
fi
ARGS+=("${REPO_ROOT}")

echo "+ docker ${ARGS[*]}" >&2
exec docker "${ARGS[@]}"

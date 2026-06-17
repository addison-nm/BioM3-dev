#!/usr/bin/env bash
#=============================================================================
#
# FILE: docker/build.sh
#
# Build the BioM3 GPU image (docker/Dockerfile) with the repo root as the
# build context. Works from any directory.
#
# USAGE:
#   docker/build.sh [--platform P] [--tag T] [--awscli] [--push] [-- <buildx args>]
#
#   --platform P   linux/amd64 | linux/arm64 | linux/amd64,linux/arm64
#                  (default: builder's native platform). Cross-arch builds
#                  need QEMU/binfmt registered in the buildx builder.
#   --tag T        image tag (default: biom3:gpu)
#   --awscli       bake awscli in (for the entrypoint's S3 sync hook)
#   --push         push to the registry instead of loading locally
#                  (required for multi-platform builds — buildx --load is
#                  single-platform only)
#
# NETWORK NOTE: the ~3 GB torch cu129 wheel comes from download.pytorch.org.
# If you're on a VPN with upstream DNS filtering (e.g. Tailscale MagicDNS) and
# the wheel download stalls on "Failed to resolve download.pytorch.org",
# disconnect the VPN for the build.
#
#=============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PLATFORM=""
TAG="biom3:gpu"
AWSCLI="false"
PUSH=0
EXTRA=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --platform) PLATFORM="$2"; shift 2 ;;
        --tag)      TAG="$2"; shift 2 ;;
        --awscli)   AWSCLI="true"; shift ;;
        --push)     PUSH=1; shift ;;
        --)         shift; EXTRA=("$@"); break ;;
        -h|--help)  sed -n '3,30p' "$0"; exit 0 ;;
        *)          echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

ARGS=(buildx build -f "${SCRIPT_DIR}/Dockerfile" -t "${TAG}"
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

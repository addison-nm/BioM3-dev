#!/usr/bin/env bash
#=============================================================================
#
# FILE: docker/push.sh
#
# Tag + push a locally-built BioM3 image to GHCR with git-derived version tags, so
# cloud instances pull it instead of rebuilding (~10-40 min build -> seconds). The
# image is published PUBLIC, so cloud/*.yaml pull it anonymously (no registry token).
#
# Pushes TWO tags per call:
#   <variant>-<shortsha>   immutable, tied to the exact commit (reproducible)
#   <variant>-dev          moving pointer for the dev line (what cloud/*.yaml track)
#
# MULTI-ARCH (amd64 for cloud instances + arm64 for DGX Spark) is built natively on
# each architecture and merged, because emulating the other arch costs hours on an
# image this size. Run --arch on a host of each architecture, then --join once:
#   host A:  docker/build.sh --variant cuda --awscli && docker/push.sh --arch
#   host B:  docker/build.sh --variant cuda --awscli && docker/push.sh --arch
#   either:  docker/push.sh --join
# --join needs both <variant>-<sha>-amd64 and <variant>-<sha>-arm64 to exist.
#
# Prereqs: docker; a local biom3:<variant> image (docker/build.sh --variant ...); and a
# GHCR login on the PUSH side (pulling a public image needs no login):
#   echo "$GHCR_TOKEN" | docker login ghcr.io -u <github-user> --password-stdin
# GHCR_TOKEN = a classic PAT with write:packages (+ read:packages), SSO-authorized for
# the org. See cloud/README.md for the full one-time runbook.
#
# USAGE:
#   docker/push.sh [--variant cuda|xpu] [--repo R] [--local-tag T] [--allow-dirty]
#                  [--arch | --join]
#
#   --variant V    cuda | xpu (default cuda) -> pushes biom3:<V> as <V>-<sha> + <V>-dev
#   --repo R       registry repo WITHOUT the tag
#                  (default: ghcr.io/natural-machine/biom3)
#   --local-tag T  local image to push (default: biom3:<variant>)
#   --allow-dirty  push from a dirty tree; the sha tag gets a -dirty suffix
#   --arch         push only <variant>-<sha>-<arch>, taking <arch> from the image
#   --join         merge the per-arch tags into <variant>-<sha> + <variant>-dev
#
#=============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

VARIANT="cuda"
REPO="ghcr.io/natural-machine/biom3"
LOCAL_TAG=""
ALLOW_DIRTY=0
MODE="single"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --variant)     VARIANT="$2"; shift 2 ;;
        --repo)        REPO="$2"; shift 2 ;;
        --local-tag)   LOCAL_TAG="$2"; shift 2 ;;
        --allow-dirty) ALLOW_DIRTY=1; shift ;;
        --arch)        MODE="arch"; shift ;;
        --join)        MODE="join"; shift ;;
        -h|--help)     sed -n '3,38p' "$0"; exit 0 ;;
        *)             echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

LOCAL_TAG="${LOCAL_TAG:-biom3:${VARIANT}}"

command -v docker >/dev/null 2>&1 || { echo "ERROR: docker not found." >&2; exit 1; }

# Version from git: <shortsha>, guarded so the tag is truthful (the image is built
# from the working tree, so a dirty tree means the tag would not match the commit).
SHA="$(git -C "${REPO_ROOT}" rev-parse --short HEAD 2>/dev/null || echo nogit)"
if [[ -n "$(git -C "${REPO_ROOT}" status --porcelain 2>/dev/null)" ]]; then
    if [[ "${ALLOW_DIRTY}" -eq 1 ]]; then
        SHA="${SHA}-dirty"
        echo "WARNING: working tree is dirty; tagging ${VARIANT}-${SHA} (NOT reproducible)." >&2
    else
        echo "ERROR: working tree is dirty. Commit first so ${VARIANT}-<sha> is truthful," >&2
        echo "       or pass --allow-dirty to tag ${VARIANT}-${SHA}." >&2
        exit 1
    fi
fi

VERSION_TAG="${REPO}:${VARIANT}-${SHA}"
MOVING_TAG="${REPO}:${VARIANT}-dev"

case "${MODE}" in

single)
    echo "+ docker tag ${LOCAL_TAG} -> ${VERSION_TAG} , ${MOVING_TAG}" >&2
    docker tag "${LOCAL_TAG}" "${VERSION_TAG}"
    docker tag "${LOCAL_TAG}" "${MOVING_TAG}"

    echo "+ docker push ${VERSION_TAG}" >&2
    docker push "${VERSION_TAG}"
    echo "+ docker push ${MOVING_TAG}" >&2
    docker push "${MOVING_TAG}"

    echo "Pushed:" >&2
    echo "  ${VERSION_TAG}   (immutable, single-arch)" >&2
    echo "  ${MOVING_TAG}    (moving dev pointer — what cloud/*.yaml reference)" >&2
    ;;

arch)
    # Read the arch off the image rather than the host, so a cross-built image is
    # still tagged truthfully.
    ARCH="$(docker image inspect --format '{{.Architecture}}' "${LOCAL_TAG}")"
    [[ -n "${ARCH}" ]] || { echo "ERROR: could not read arch of ${LOCAL_TAG}." >&2; exit 1; }
    ARCH_TAG="${VERSION_TAG}-${ARCH}"

    echo "+ docker tag ${LOCAL_TAG} -> ${ARCH_TAG}" >&2
    docker tag "${LOCAL_TAG}" "${ARCH_TAG}"
    echo "+ docker push ${ARCH_TAG}" >&2
    docker push "${ARCH_TAG}"

    echo "Pushed ${ARCH_TAG}. Run --join once every arch is pushed." >&2
    ;;

join)
    MISSING=()
    for a in amd64 arm64; do
        docker buildx imagetools inspect "${VERSION_TAG}-${a}" >/dev/null 2>&1 \
            || MISSING+=("${VERSION_TAG}-${a}")
    done
    if [[ ${#MISSING[@]} -gt 0 ]]; then
        echo "ERROR: missing per-arch tag(s): ${MISSING[*]}" >&2
        echo "       Build on that architecture and push with --arch first." >&2
        exit 1
    fi

    echo "+ docker buildx imagetools create -t ${VERSION_TAG} -t ${MOVING_TAG}" >&2
    docker buildx imagetools create \
        -t "${VERSION_TAG}" -t "${MOVING_TAG}" \
        "${VERSION_TAG}-amd64" "${VERSION_TAG}-arm64"

    echo "Published multi-arch:" >&2
    echo "  ${VERSION_TAG}   (immutable, amd64+arm64)" >&2
    echo "  ${MOVING_TAG}    (moving dev pointer — what cloud/*.yaml reference)" >&2
    echo "Verify: docker manifest inspect ${MOVING_TAG}" >&2
    ;;

esac

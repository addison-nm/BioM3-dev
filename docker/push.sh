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
# Prereqs: docker; a local biom3:<variant> image (docker/build.sh --variant ...); and a
# GHCR login on the PUSH side (pulling a public image needs no login):
#   echo "$GHCR_TOKEN" | docker login ghcr.io -u <github-user> --password-stdin
# GHCR_TOKEN = a classic PAT with write:packages (+ read:packages), SSO-authorized for
# the org. See cloud/README.md for the full one-time runbook.
#
# USAGE:
#   docker/push.sh [--variant cuda|xpu] [--repo R] [--local-tag T] [--allow-dirty]
#
#   --variant V    cuda | xpu (default cuda) -> pushes biom3:<V> as <V>-<sha> + <V>-dev
#   --repo R       registry repo WITHOUT the tag
#                  (default: ghcr.io/natural-machine/biom3)
#   --local-tag T  local image to push (default: biom3:<variant>)
#   --allow-dirty  push from a dirty tree; the sha tag gets a -dirty suffix
#
#=============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

VARIANT="cuda"
REPO="ghcr.io/natural-machine/biom3"
LOCAL_TAG=""
ALLOW_DIRTY=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --variant)     VARIANT="$2"; shift 2 ;;
        --repo)        REPO="$2"; shift 2 ;;
        --local-tag)   LOCAL_TAG="$2"; shift 2 ;;
        --allow-dirty) ALLOW_DIRTY=1; shift ;;
        -h|--help)     sed -n '3,27p' "$0"; exit 0 ;;
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

echo "+ docker tag ${LOCAL_TAG} -> ${VERSION_TAG} , ${MOVING_TAG}" >&2
docker tag "${LOCAL_TAG}" "${VERSION_TAG}"
docker tag "${LOCAL_TAG}" "${MOVING_TAG}"

echo "+ docker push ${VERSION_TAG}" >&2
docker push "${VERSION_TAG}"
echo "+ docker push ${MOVING_TAG}" >&2
docker push "${MOVING_TAG}"

echo "Pushed:" >&2
echo "  ${VERSION_TAG}   (immutable)" >&2
echo "  ${MOVING_TAG}    (moving dev pointer — what cloud/*.yaml reference)" >&2

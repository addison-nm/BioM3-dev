#!/usr/bin/env bash
#=============================================================================
#
# FILE: scripts/weights_bundle/push_bundle.sh
#
# Push a built weights bundle to GHCR as an OCI artifact, so consumers pull a
# consistent weights + config set instead of hunting for matched files. Mirrors
# docker/push.sh's git-sha derivation and dirty-tree guard.
#
# Pushes ONE immutable, sha-pinned tag:
#   <tag>-<shortsha>   tied to the exact commit that built the bundle
#
# There is no moving pointer: weights are a versioned artifact, so consumers pin
# an exact sha rather than tracking a tag that shifts under them.
#
# Each file becomes its own content-addressed blob, so republishing a bundle
# that only changes one stage re-uploads only that stage. Weights are
# architecture-independent — there is no multi-arch step.
#
# Prereqs: oras (https://oras.land, single static binary); a bundle built by
# scripts/weights_bundle/build_bundle.py; and a GHCR login on the PUSH side
# (pulling a public artifact needs no login):
#   echo "$GHCR_TOKEN" | oras login ghcr.io -u <github-user> --password-stdin
# GHCR_TOKEN = a classic PAT with write:packages (+ read:packages), SSO-authorized
# for the org. See docs/setup/weights_bundle.md for the full runbook.
#
# NOTE: GHCR enforces a 10-minute upload timeout per blob and oras has no
# resumable upload. The largest blob here is ~3 GB, which needs a sustained
# ~41 Mbps upstream. Rehearse with --smoke on a slow link.
#
# USAGE:
#   scripts/weights_bundle/push_bundle.sh <bundle_dir> [--repo R] [--tag T]
#                                         [--allow-dirty] [--smoke] [--dry-run]
#
#   <bundle_dir>   directory produced by build_bundle.py
#   --repo R       registry repo WITHOUT the tag
#                  (default: ghcr.io/natural-machine/biom3-weights)
#   --tag T        bundle tag stem (default: run1_base)
#   --allow-dirty  push from a dirty tree; the sha tag gets a -dirty suffix
#   --smoke        push only the small files (configs, manifest, Stage 2 + 3)
#                  under <tag>-smoke, to validate auth and pull behavior first
#   --dry-run      print the oras command without running it
#
#=============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

REPO="ghcr.io/natural-machine/biom3-weights"
TAG="run1_base"
ARTIFACT_TYPE="application/vnd.biom3.weights.bundle.v1"
ALLOW_DIRTY=0
SMOKE=0
DRY_RUN=0
BUNDLE_DIR=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --repo)        REPO="$2"; shift 2 ;;
        --tag)         TAG="$2"; shift 2 ;;
        --allow-dirty) ALLOW_DIRTY=1; shift ;;
        --smoke)       SMOKE=1; shift ;;
        --dry-run)     DRY_RUN=1; shift ;;
        -h|--help)     sed -n '3,43p' "$0"; exit 0 ;;
        -*)            echo "Unknown arg: $1" >&2; exit 1 ;;
        *)             BUNDLE_DIR="$1"; shift ;;
    esac
done

[[ -n "${BUNDLE_DIR}" ]] || { echo "ERROR: bundle directory required." >&2; exit 1; }
BUNDLE_DIR="$(cd "${BUNDLE_DIR}" && pwd)"
[[ -f "${BUNDLE_DIR}/MANIFEST.json" ]] || {
    echo "ERROR: ${BUNDLE_DIR} has no MANIFEST.json — not a built bundle." >&2
    exit 1
}

command -v oras >/dev/null 2>&1 || {
    echo "ERROR: oras not found. Install from https://oras.land/docs/installation" >&2
    exit 1
}

# The bundle records the commit it was built from; refuse to publish under a tag
# that would misattribute it.
BUILT_SHA="$(python3 -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["source_repo_git_sha"])' \
    "${BUNDLE_DIR}/MANIFEST.json")"
SHA="$(git -C "${REPO_ROOT}" rev-parse --short HEAD 2>/dev/null || echo nogit)"

if [[ "${BUILT_SHA}" != "${SHA}" ]]; then
    echo "ERROR: bundle was built at ${BUILT_SHA} but HEAD is ${SHA}." >&2
    echo "       Rebuild the bundle so the published tag is truthful." >&2
    exit 1
fi

if [[ -n "$(git -C "${REPO_ROOT}" status --porcelain 2>/dev/null)" ]]; then
    if [[ "${ALLOW_DIRTY}" -eq 1 ]]; then
        SHA="${SHA}-dirty"
        echo "WARNING: working tree is dirty; tagging ${TAG}-${SHA} (NOT reproducible)." >&2
    else
        echo "ERROR: working tree is dirty. Commit first so ${TAG}-<sha> is truthful," >&2
        echo "       or pass --allow-dirty." >&2
        exit 1
    fi
fi

if [[ "$(python3 -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["incomplete"])' \
    "${BUNDLE_DIR}/MANIFEST.json")" == "True" && "${SMOKE}" -eq 0 ]]; then
    echo "ERROR: bundle is marked incomplete (built with --skip_llms)." >&2
    echo "       Rebuild without --skip_llms, or push it as --smoke." >&2
    exit 1
fi

# Media type per file: JSON for the manifest and configs, opaque bytes for
# weights. oras records each path as the blob's title annotation, so the
# directory layout survives the round trip.
media_type_for() {
    case "$1" in
        *.json) echo "application/json" ;;
        *.md)   echo "text/markdown" ;;
        *)      echo "application/octet-stream" ;;
    esac
}

cd "${BUNDLE_DIR}"

FILES=()
while IFS= read -r rel; do
    if [[ "${SMOKE}" -eq 1 && "${rel}" == weights/LLMs/* ]]; then continue; fi
    if [[ "${SMOKE}" -eq 1 && "${rel}" == weights/PenCL/* ]]; then continue; fi
    FILES+=("${rel}:$(media_type_for "${rel}")")
done < <(find . -type f -printf '%P\n' | sort)

if [[ "${SMOKE}" -eq 1 ]]; then
    VERSION_TAG="${REPO}:${TAG}-smoke-${SHA}"
    echo "SMOKE: omitting LLMs/ and PenCL/ — this artifact is NOT usable for Stage 1." >&2
else
    VERSION_TAG="${REPO}:${TAG}-${SHA}"
fi

TOTAL_BYTES="$(du -sb "${BUNDLE_DIR}" | cut -f1)"
echo "Bundle:  ${BUNDLE_DIR}" >&2
echo "Files:   ${#FILES[@]} ($(numfmt --to=iec "${TOTAL_BYTES}"))" >&2
echo "Tag:     ${VERSION_TAG}" >&2

if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "+ oras push --artifact-type ${ARTIFACT_TYPE} ${VERSION_TAG} <${#FILES[@]} files>" >&2
    printf '    %s\n' "${FILES[@]}" >&2
    exit 0
fi

echo "+ oras push (this uploads ${TOTAL_BYTES} bytes; GHCR times out at 10 min per blob)" >&2
oras push \
    --artifact-type "${ARTIFACT_TYPE}" \
    --annotation "org.opencontainers.image.source=https://github.com/natural-machine/BioM3-dev" \
    --annotation "org.opencontainers.image.revision=${SHA}" \
    "${VERSION_TAG}" \
    "${FILES[@]}"

echo "Pushed: ${VERSION_TAG}   (immutable)" >&2
echo "Verify: oras manifest fetch --pretty ${VERSION_TAG}" >&2
echo "Fetch:  scripts/weights_bundle/fetch_bundle.sh <dest> --tag ${VERSION_TAG##*:}" >&2

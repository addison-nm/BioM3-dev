#!/usr/bin/env bash
# Pull a published weights bundle from GHCR and verify it against its manifest.
#
#   fetch_bundle.sh <dest> --tag <tag> [--link] [--repo R] [--quick-verify]
#
# By default this only pulls + verifies and does NOT touch your checkout; it prints
# the commands to wire the bundle in. Pass --link to also do that wiring: symlink
# the weights into ./weights (via link_weights.sh, which never overwrites) and the
# bundle's configs into ./configs/bundles/<name>/.
#
# Pulling a public artifact needs no login; for a private one:
#   echo "$GHCR_TOKEN" | oras login ghcr.io -u <github-user> --password-stdin
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

REPO="ghcr.io/natural-machine/biom3-weights"
TAG=""
DEST=""
LINK=0
VERIFY_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --repo)         REPO="$2"; shift 2 ;;
        --tag)          TAG="$2"; shift 2 ;;
        --link)         LINK=1; shift ;;
        --quick-verify) VERIFY_ARGS+=("--quick"); shift ;;
        -h|--help)      echo "usage: fetch_bundle.sh <dest> --tag <tag> [--link] [--repo R] [--quick-verify]"; exit 0 ;;
        -*)             echo "Unknown arg: $1" >&2; exit 1 ;;
        *)              DEST="$1"; shift ;;
    esac
done

[[ -n "${DEST}" ]] || { echo "ERROR: destination directory required." >&2; exit 1; }
[[ -n "${TAG}" ]]  || {
    echo "ERROR: --tag is required (e.g. --tag run1_base). List published tags with:" >&2
    echo "       oras repo tags ${REPO}" >&2
    exit 1
}
command -v oras >/dev/null 2>&1 || {
    echo "ERROR: oras not found. Install from https://oras.land/docs/installation" >&2
    exit 1
}

mkdir -p "${DEST}"
DEST="$(cd "${DEST}" && pwd)"
BUNDLE_DIR="${DEST}/${TAG}"

echo "+ oras pull ${REPO}:${TAG} -> ${BUNDLE_DIR}" >&2
mkdir -p "${BUNDLE_DIR}"
oras pull "${REPO}:${TAG}" -o "${BUNDLE_DIR}"

echo "+ verifying against MANIFEST.json" >&2
python3 "${SCRIPT_DIR}/verify_bundle.py" "${BUNDLE_DIR}" "${VERIFY_ARGS[@]}"

# Link name comes from the bundle's own manifest, so any bundle works.
NAME="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["name"])' "${BUNDLE_DIR}/MANIFEST.json")"
LINK_NAME="${NAME#biom3-weights-}"

if [[ "${LINK}" -eq 0 ]]; then
    cat >&2 <<EOF

Pulled and verified: ${BUNDLE_DIR}
(nothing in your checkout was touched.)

To wire it into a BioM3-dev clone, re-run with --link, or from the repo root:

  ./scripts/link_weights.sh ${BUNDLE_DIR}/weights weights
  ln -s ${BUNDLE_DIR}/configs configs/bundles/${LINK_NAME}
EOF
    exit 0
fi

# --link: wire the bundle into this checkout. link_weights.sh never overwrites —
# it links only absent files and reports MATCH/MISMATCH for the rest.
[[ -d "${BUNDLE_DIR}/weights" ]] || {
    echo "ERROR: ${BUNDLE_DIR}/weights missing — is this a weights bundle?" >&2
    exit 1
}
echo "+ scripts/link_weights.sh ${BUNDLE_DIR}/weights weights" >&2
"${REPO_ROOT}/scripts/link_weights.sh" "${BUNDLE_DIR}/weights" "${REPO_ROOT}/weights"

BUNDLE_CONFIG_LINK="${REPO_ROOT}/configs/bundles/${LINK_NAME}"
mkdir -p "$(dirname "${BUNDLE_CONFIG_LINK}")"
if [[ -L "${BUNDLE_CONFIG_LINK}" ]]; then
    rm "${BUNDLE_CONFIG_LINK}"
elif [[ -e "${BUNDLE_CONFIG_LINK}" ]]; then
    echo "ERROR: ${BUNDLE_CONFIG_LINK} exists and is not a symlink. Move it aside." >&2
    exit 1
fi
ln -s "${BUNDLE_DIR}/configs" "${BUNDLE_CONFIG_LINK}"
echo "+ configs/bundles/${LINK_NAME} -> ${BUNDLE_DIR}/configs" >&2
echo "" >&2
echo "Linked. Review the link_weights.sh summary above for any MISMATCH (nothing was" >&2
echo "overwritten). The symlinks point into ${BUNDLE_DIR} — keep it around." >&2

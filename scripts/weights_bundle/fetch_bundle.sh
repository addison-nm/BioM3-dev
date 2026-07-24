#!/usr/bin/env bash
# Pull a published weights bundle from GHCR and verify it against its manifest.
#
#   fetch_bundle.sh <dest> --tag <tag> [--repo R] [--quick-verify]
#
# This does NOT touch your checkout. Wiring the bundle into ./weights and
# ./configs is a separate, explicit step (the commands are printed on success),
# so a fetch never overwrites or shadows anything in your working tree.
#
# Pulling a public artifact needs no login; for a private one:
#   echo "$GHCR_TOKEN" | oras login ghcr.io -u <github-user> --password-stdin
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

REPO="ghcr.io/natural-machine/biom3-weights"
TAG=""
DEST=""
VERIFY_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --repo)         REPO="$2"; shift 2 ;;
        --tag)          TAG="$2"; shift 2 ;;
        --quick-verify) VERIFY_ARGS+=("--quick"); shift ;;
        -h|--help)      echo "usage: fetch_bundle.sh <dest> --tag <tag> [--repo R] [--quick-verify]"; exit 0 ;;
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

# Link name (for the hint below) comes from the bundle's own manifest.
NAME="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["name"])' "${BUNDLE_DIR}/MANIFEST.json")"
LINK_NAME="${NAME#biom3-weights-}"

cat >&2 <<EOF

Pulled and verified: ${BUNDLE_DIR}
(nothing in your checkout was touched.)

To wire it into a BioM3-dev clone, from the repo root — link_weights.sh never
overwrites: it symlinks only files that are absent and reports MATCH/MISMATCH for
the rest, so review its output before relying on the result:

  ./scripts/link_weights.sh ${BUNDLE_DIR}/weights weights
  ln -s ${BUNDLE_DIR}/configs configs/bundles/${LINK_NAME}
EOF

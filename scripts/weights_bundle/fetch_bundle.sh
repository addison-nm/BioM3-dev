#!/usr/bin/env bash
# Pull a published weights bundle from GHCR and wire it into this checkout.
#
#   fetch_bundle.sh <dest> --tag <tag> [--repo R] [--no-link] [--quick-verify]
#
# Steps (no BioM3 install needed):
#   1. oras pull  -> <dest>/<tag>/{weights,configs,MANIFEST.json}
#   2. verify     -> every file's sha256 against MANIFEST.json
#   3. link       -> weights into ./weights/ (via link_weights.sh) and the
#                    bundle's configs into ./configs/bundles/<name>/. <name> is
#                    read from the manifest, so any bundle works (run1_base,
#                    run2_base, ...).
#
# Pulling a public artifact needs no login; for a private one:
#   echo "$GHCR_TOKEN" | oras login ghcr.io -u <github-user> --password-stdin
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

REPO="ghcr.io/natural-machine/biom3-weights"
TAG=""
DEST=""
NO_LINK=0
VERIFY_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --repo)         REPO="$2"; shift 2 ;;
        --tag)          TAG="$2"; shift 2 ;;
        --no-link)      NO_LINK=1; shift ;;
        --quick-verify) VERIFY_ARGS+=("--quick"); shift ;;
        -h|--help)      echo "usage: fetch_bundle.sh <dest> --tag <tag> [--repo R] [--no-link] [--quick-verify]"; exit 0 ;;
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

if [[ "${NO_LINK}" -eq 1 ]]; then
    echo "Pulled and verified: ${BUNDLE_DIR} (not linked)" >&2
    exit 0
fi

# Local link name comes from the bundle itself, so this works for any bundle.
NAME="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["name"])' "${BUNDLE_DIR}/MANIFEST.json")"
LINK_NAME="${NAME#biom3-weights-}"

BUNDLE_WEIGHTS="${BUNDLE_DIR}/weights"
[[ -d "${BUNDLE_WEIGHTS}" ]] || {
    echo "ERROR: ${BUNDLE_WEIGHTS} missing — is this a weights bundle?" >&2
    exit 1
}

echo "+ scripts/link_weights.sh ${BUNDLE_WEIGHTS} weights" >&2
"${REPO_ROOT}/scripts/link_weights.sh" "${BUNDLE_WEIGHTS}" "${REPO_ROOT}/weights"

# Expose the bundle's configs at a stable path so consumer configs can reference
# them relatively (../bundles/<name>/...).
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
echo "Done. Weights linked into ./weights/, configs at configs/bundles/${LINK_NAME}/." >&2

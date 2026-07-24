#!/usr/bin/env bash
#=============================================================================
#
# FILE: scripts/weights_bundle/fetch_bundle.sh
#
# Pull a published BioM3 weights bundle from GHCR and wire it into this
# checkout. Three steps, none of which need a BioM3 install:
#
#   1. oras pull      -> <dest>/biom3-weights-run1_base/{weights,configs}
#   2. verify         -> every file's sha256 against the bundle MANIFEST.json
#   3. link           -> symlinks into ./weights/ and ./configs/bundles/
#
# Step 3 reuses scripts/link_weights.sh, so the bundle behaves exactly like the
# shared BioM3-data-share weights directory the repo already consumes. Existing
# configs continue to reference ./weights/LLMs/... and keep working — nothing in
# the config loader changes.
#
# Pulling a public artifact needs no login. For a private package:
#   echo "$GHCR_TOKEN" | oras login ghcr.io -u <github-user> --password-stdin
#
# USAGE:
#   scripts/weights_bundle/fetch_bundle.sh <dest> [--repo R] [--tag T]
#                                          [--no-link] [--quick-verify]
#
#   <dest>          directory to pull into (created if absent)
#   --repo R        registry repo WITHOUT the tag
#                   (default: ghcr.io/natural-machine/biom3-weights)
#   --tag T         tag to pull (default: run1_base)
#   --no-link       pull and verify only; skip the symlink step
#   --quick-verify  check size only, skipping sha256 (fast, less thorough)
#
#=============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

REPO="ghcr.io/natural-machine/biom3-weights"
TAG="run1_base"
BUNDLE_NAME="biom3-weights-run1_base"
DEST=""
NO_LINK=0
VERIFY_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --repo)         REPO="$2"; shift 2 ;;
        --tag)          TAG="$2"; shift 2 ;;
        --no-link)      NO_LINK=1; shift ;;
        --quick-verify) VERIFY_ARGS+=("--quick"); shift ;;
        -h|--help)      sed -n '3,32p' "$0"; exit 0 ;;
        -*)             echo "Unknown arg: $1" >&2; exit 1 ;;
        *)              DEST="$1"; shift ;;
    esac
done

[[ -n "${DEST}" ]] || { echo "ERROR: destination directory required." >&2; exit 1; }

command -v oras >/dev/null 2>&1 || {
    echo "ERROR: oras not found. Install from https://oras.land/docs/installation" >&2
    exit 1
}

mkdir -p "${DEST}"
DEST="$(cd "${DEST}" && pwd)"
BUNDLE_DIR="${DEST}/${BUNDLE_NAME}"

echo "+ oras pull ${REPO}:${TAG} -> ${BUNDLE_DIR}" >&2
mkdir -p "${BUNDLE_DIR}"
oras pull "${REPO}:${TAG}" -o "${BUNDLE_DIR}"

echo "+ verifying against MANIFEST.json" >&2
python3 "${SCRIPT_DIR}/verify_bundle.py" "${BUNDLE_DIR}" "${VERIFY_ARGS[@]}"

if [[ "${NO_LINK}" -eq 1 ]]; then
    echo "Pulled and verified: ${BUNDLE_DIR} (not linked)" >&2
    exit 0
fi

# link_weights.sh expects the directory that CONTAINS LLMs/, PenCL/, ... — that
# is <bundle>/weights, not the bundle root. Pointing it at the root would create
# weights/configs/ and weights/weights/.
BUNDLE_WEIGHTS="${BUNDLE_DIR}/weights"
[[ -d "${BUNDLE_WEIGHTS}/LLMs" ]] || {
    echo "ERROR: ${BUNDLE_WEIGHTS} does not look like a weights root (no LLMs/)." >&2
    exit 1
}

echo "+ scripts/link_weights.sh ${BUNDLE_WEIGHTS} weights" >&2
"${REPO_ROOT}/scripts/link_weights.sh" "${BUNDLE_WEIGHTS}" "${REPO_ROOT}/weights"

# Expose the bundle's partial configs at a stable path so configs can reference
# them relatively (../bundles/run1_base/...) instead of by absolute path.
BUNDLE_CONFIG_LINK="${REPO_ROOT}/configs/bundles/${TAG}"
mkdir -p "$(dirname "${BUNDLE_CONFIG_LINK}")"
if [[ -L "${BUNDLE_CONFIG_LINK}" ]]; then
    rm "${BUNDLE_CONFIG_LINK}"
elif [[ -e "${BUNDLE_CONFIG_LINK}" ]]; then
    echo "ERROR: ${BUNDLE_CONFIG_LINK} exists and is not a symlink. Move it aside." >&2
    exit 1
fi
ln -s "${BUNDLE_DIR}/configs" "${BUNDLE_CONFIG_LINK}"
echo "+ configs/bundles/${TAG} -> ${BUNDLE_DIR}/configs" >&2

cat >&2 <<EOF

Done. Run a stage with the bundle's pinned architecture:

  biom3_PenCL_inference \\
      --input_data_path None \\
      --config_path configs/inference/stage1_PenCL_${TAG}.json \\
      --model_path weights/PenCL/BioM3_PenCL_${TAG}.bin \\
      --output_path outputs/pencl_embeddings.pt
EOF

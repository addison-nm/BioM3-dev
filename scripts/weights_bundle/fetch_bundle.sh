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
#   scripts/weights_bundle/fetch_bundle.sh <dest> --tag run1_base-<sha>
#                                          [--repo R] [--no-link] [--quick-verify]
#
#   <dest>          directory to pull into (created if absent)
#   --tag T         immutable tag to pull, e.g. run1_base-<sha>  (REQUIRED)
#   --repo R        registry repo WITHOUT the tag
#                   (default: ghcr.io/natural-machine/biom3-weights)
#   --no-link       pull and verify only; skip the symlink step
#   --quick-verify  check size only, skipping sha256 (fast, less thorough)
#
#=============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

REPO="ghcr.io/natural-machine/biom3-weights"
TAG=""                        # required: the immutable run1_base-<sha> tag
LINK_NAME="run1_base"         # stable local name; configs reference ../bundles/run1_base/
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
[[ -n "${TAG}" ]] || {
    echo "ERROR: --tag is required (e.g. --tag run1_base-<sha>). Weights are pinned by" >&2
    echo "       sha; there is no moving tag. Find the published tag in cloud/README.md," >&2
    echo "       or list them: oras repo tags ${REPO}" >&2
    exit 1
}

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

# Expose the bundle's partial configs at a stable path so the committed consumer
# configs can reference them relatively (../bundles/run1_base/...). The link name
# is fixed regardless of which sha tag was pulled.
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

cat >&2 <<EOF

Done. Run a stage with the bundle's pinned architecture:

  biom3_PenCL_inference \\
      --input_data_path None \\
      --config_path configs/inference/stage1_PenCL_${LINK_NAME}.json \\
      --model_path weights/PenCL/BioM3_PenCL_${LINK_NAME}.bin \\
      --output_path outputs/pencl_embeddings.pt
EOF

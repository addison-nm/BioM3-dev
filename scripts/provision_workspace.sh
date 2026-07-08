#!/bin/bash
#=============================================================================
#
# FILE: provision_workspace.sh
#
# USAGE: provision_workspace.sh new-workspace-path
#
# DESCRIPTION: Provision a new directory with the minimal infrastructure to
#   quickly use BioM3. Includes data, configs, outputs, and weights directories.
#
# EXAMPLE: sh provision_workspace.sh /path/to/new/workspace
#=============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 new-workspace-path"
    exit 1
fi

target="$1"

if [ -e "$target" ]; then
    echo "ERROR: Path already exists: $target"
    echo "Refusing to provision over an existing path. Choose a new location."
    exit 1
fi

mkdir "$target"
echo "Provisioning a new BioM3 workspace at: $target"

dirs=(data weights configs outputs scripts)
for d in "${dirs[@]}"; do
    mkdir -p "$target/$d"
done

# Pre-create the canonical per-component weights subdirs (see weights/README.md).
weights_subdirs=(LLMs PenCL Facilitator ProteoScribe)
for d in "${weights_subdirs[@]}"; do
    mkdir -p "$target/weights/$d"
done

# Copy the environment setup script into the workspace.
if [ -f "$REPO_ROOT/environment.sh" ]; then
    cp "$REPO_ROOT/environment.sh" "$target/environment.sh"
    echo "Copied environment.sh into workspace."
else
    echo "WARNING: environment.sh not found at $REPO_ROOT; skipping."
fi

# Optionally link a local weights tree into $target/weights via link_weights.sh.
read -r -p "Local path to weights to link (press Enter or 'No' to skip): " weights_path

if [ -z "$weights_path" ] || [[ "$weights_path" =~ ^[Nn][Oo]?$ ]]; then
    echo "Skipping weights linking."
elif [ ! -d "$weights_path" ]; then
    echo "ERROR: Weights path does not exist or is not a directory: $weights_path"
    echo "Skipping weights linking."
else
    "$SCRIPT_DIR/link_weights.sh" "$weights_path" "$target/weights"
fi

echo "Done. Workspace ready at: $target"

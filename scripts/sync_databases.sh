#!/usr/bin/env bash
# sync_databases.sh - Sync reference databases from a shared directory into a
# local directory using symlinks.
#
# For each top-level entry (file or directory) inside each subdirectory of
# SRC_DIR, this script:
#   1. Skips .git* and README* files
#   2. If the entry does not exist in TGT_DIR, creates a symlink
#   3. If the entry already exists, compares via md5sum (recursively for dirs)
#      and reports matches/mismatches
#
# Usage:
#   ./sync_databases.sh <source_dir> <target_dir> [--dry-run]
#
# Arguments:
#   source_dir   Directory containing shared databases (e.g.
#                /data/data-share/BioM3-data-share/databases)
#   target_dir   Local databases directory to populate with symlinks (e.g.
#                ./data/databases)
#   --dry-run    Show what would be done without making changes
#
# Examples:
#   # Preview changes
#   ./sync_databases.sh /data/data-share/BioM3-data-share/databases data/databases --dry-run
#
#   # Apply symlinks
#   ./sync_databases.sh /data/data-share/BioM3-data-share/databases data/databases

set -euo pipefail

usage() {
    echo "Usage: $0 <source_dir> <target_dir> [--dry-run]"
    exit 1
}

if [ $# -lt 2 ] || [ $# -gt 3 ]; then
    usage
fi

SRC_DIR="$(realpath "$1")"
TGT_DIR="$(realpath "$2")"
DRY_RUN=false

if [ $# -eq 3 ]; then
    if [ "$3" = "--dry-run" ]; then
        DRY_RUN=true
    else
        usage
    fi
fi

if [ ! -d "$SRC_DIR" ]; then
    echo "ERROR: Source directory does not exist: $SRC_DIR"
    exit 1
fi

if [ ! -d "$TGT_DIR" ]; then
    echo "ERROR: Target directory does not exist: $TGT_DIR"
    exit 1
fi

echo "Source: $SRC_DIR"
echo "Target: $TGT_DIR"
$DRY_RUN && echo "Mode:   DRY RUN"
echo ""

# Compute a comparable hash for a path (file or directory).
# For directories, hashes all regular files recursively and produces a combined hash.
compute_hash() {
    local path="$1"
    if [ -f "$path" ]; then
        md5sum "$path" | awk '{print $1}'
    elif [ -d "$path" ]; then
        (cd "$path" && find . -type f ! -path '*/.git/*' | sort | while read -r f; do
            md5sum "$f"
        done | md5sum | awk '{print $1}')
    fi
}

linked=0
skipped=0
matched=0
mismatched=0

# Iterate over subdirectories in the source
for subdir in "$SRC_DIR"/*/; do
    subdir_name="$(basename "$subdir")"
    echo "--- $subdir_name ---"

    # Ensure the subdirectory exists in the target
    if [ ! -d "$TGT_DIR/$subdir_name" ]; then
        echo "  MKDIR: $subdir_name/"
        if ! $DRY_RUN; then
            mkdir -p "$TGT_DIR/$subdir_name"
        fi
    fi

    for entry in "$subdir"*; do
        # Skip if glob didn't match anything (empty directory)
        [ -e "$entry" ] || continue

        name="$(basename "$entry")"

        # Skip .git and README files
        [[ "$name" == .git* ]] && continue
        [[ "$name" == README* ]] && continue

        target="$TGT_DIR/$subdir_name/$name"

        if [ -e "$target" ] || [ -L "$target" ]; then
            # Entry exists - compare hashes
            src_hash=$(compute_hash "$entry")
            tgt_hash=$(compute_hash "$target")

            if [ "$src_hash" = "$tgt_hash" ]; then
                echo "  MATCH:    $name"
                matched=$((matched + 1))
            else
                echo "  MISMATCH: $name"
                echo "    src md5: $src_hash"
                echo "    tgt md5: $tgt_hash"
                mismatched=$((mismatched + 1))
            fi
        else
            echo "  LINK:     $name -> $entry"
            if ! $DRY_RUN; then
                ln -s "$entry" "$target"
            fi
            linked=$((linked + 1))
        fi
    done
    echo ""
done

# Also link top-level files (e.g. provenance.tsv)
for entry in "$SRC_DIR"/*; do
    [ -d "$entry" ] && continue
    name="$(basename "$entry")"
    [[ "$name" == .git* ]] && continue
    [[ "$name" == README* ]] && continue

    target="$TGT_DIR/$name"
    if [ -e "$target" ] || [ -L "$target" ]; then
        src_hash=$(compute_hash "$entry")
        tgt_hash=$(compute_hash "$target")
        if [ "$src_hash" = "$tgt_hash" ]; then
            echo "MATCH:    $name"
            matched=$((matched + 1))
        else
            echo "MISMATCH: $name"
            mismatched=$((mismatched + 1))
        fi
    else
        echo "LINK:     $name -> $entry"
        if ! $DRY_RUN; then
            ln -s "$entry" "$target"
        fi
        linked=$((linked + 1))
    fi
done

echo ""
echo "=== Summary ==="
echo "  Linked:     $linked"
echo "  Matched:    $matched"
echo "  Mismatched: $mismatched"

if [ "$mismatched" -gt 0 ]; then
    echo ""
    echo "WARNING: $mismatched file(s) differ between source and target."
fi

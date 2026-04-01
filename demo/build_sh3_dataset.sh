#!/usr/bin/env bash
# build_sh3_dataset.sh — Build an SH3 domain (PF00018) fine-tuning dataset
#
# Demonstrates the biom3_build_dataset entrypoint with different enrichment
# options. SH3 domains are small protein interaction modules (~60 residues)
# that bind proline-rich peptide sequences.
#
# Prerequisites:
#   pip install -e .
#   source environment.sh
#   ./scripts/sync_databases.sh <shared_path> data/databases
#
# Usage:
#   bash demo/build_sh3_dataset.sh

set -euo pipefail

OUTDIR="demo/outputs/sh3_dataset"

# The taxonomy index is a derived file (~25 GB) that must be writable.
# Store it locally alongside the symlinked database files.
TAXID_INDEX="data/databases/ncbi_taxonomy/accession2taxid.sqlite"

# Source file for building the index (read-only, may be a symlink).
TAXID_SOURCE="data/databases/ncbi_taxonomy/prot.accession2taxid.gz"

echo "============================================================"
echo "  Demo: Build SH3 Domain Dataset (PF00018)"
echo "============================================================"
echo ""

# --- Step 0: Build taxonomy SQLite index if needed ---
# This is a one-time operation (~10-15 min) that makes all subsequent
# taxonomy lookups instant (seconds instead of streaming 1.55B rows).
if [ ! -f "$TAXID_INDEX" ]; then
    echo "[0] Building SQLite taxonomy index (one-time)..."
    echo "    This takes ~10-15 minutes but only needs to be done once."
    echo ""
    biom3_build_taxid_index "$TAXID_SOURCE" -o "$TAXID_INDEX"
    echo ""
fi

# --- Option 1: Basic extraction (no enrichment) ---
# Subsets SwissProt and Pfam CSVs by Pfam ID.
# Paths resolved from configs/dbio_config.json.
echo "[1] Basic extraction..."
biom3_build_dataset \
    -p PF00018 \
    -o "${OUTDIR}/basic"

echo ""

# --- Option 2: With UniProt enrichment (via REST API) ---
# Enriches Pfam captions with protein name, function, GO terms, etc.
# Results are cached in .uniprot_cache/ for subsequent runs.
echo "[2] With UniProt API enrichment..."
biom3_build_dataset \
    -p PF00018 \
    --enrich-pfam \
    -o "${OUTDIR}/enriched_api"

echo ""

# --- Option 3: With local .dat file enrichment (offline) ---
# Uses uniprot_sprot.dat.gz instead of API calls.
# Note: only covers reviewed Swiss-Prot entries (~568K).
# For full Pfam coverage, also include uniprot_trembl.dat.gz.
echo "[3] With local .dat enrichment..."
biom3_build_dataset \
    -p PF00018 \
    --enrich-pfam \
    --uniprot-dat data/databases/swissprot/uniprot_sprot.dat.gz \
    -o "${OUTDIR}/enriched_local"

echo ""

# --- Option 4: With enrichment + taxonomy filtering ---
# Adds NCBI taxonomy lineage and filters to bacteria only.
# Uses the pre-built SQLite index for fast accession-to-taxid lookups.
echo "[4] With enrichment + taxonomy (bacteria only)..."
biom3_build_dataset \
    -p PF00018 \
    --enrich-pfam \
    --add-taxonomy \
    --taxid-index "$TAXID_INDEX" \
    --taxonomy-filter "superkingdom=Bacteria" \
    -o "${OUTDIR}/enriched_bacteria"

echo ""
echo "============================================================"
echo "  Done. Outputs in ${OUTDIR}/"
echo ""
echo "  Each subdirectory contains:"
echo "    dataset.csv              — final dataset (4-column format)"
echo "    dataset_annotations.csv  — intermediate with annot_* columns"
echo "    build.log                — full log output"
echo "    build_manifest.json      — reproducibility manifest"
echo "    pfam_ids.csv             — Pfam IDs used"
echo "============================================================"

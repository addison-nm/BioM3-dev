#!/usr/bin/env bash
# build_source_datasets.sh — Build training CSVs from raw databases
#
# Demonstrates the biom3_build_source_swissprot and biom3_build_source_pfam
# entrypoints, which produce the two intermediate training CSVs
# (fully_annotated_swiss_prot.csv and Pfam_protein_text_dataset.csv) directly
# from the raw database files.
#
# This closes the reproducibility gap: instead of depending on pre-curated CSVs
# from an unknown pipeline, the entire dataset lineage is now:
#
#   Raw databases (UniProt .dat, Pfam FASTA, NCBI taxonomy)
#     → biom3_build_source_swissprot / biom3_build_source_pfam
#       → training CSVs
#         → biom3_build_dataset (subset by Pfam ID, enrich, filter)
#           → fine-tuning dataset
#
# For the Pfam step, this demo extracts only SH3 domain entries (PF00018) from
# the full FASTA to keep runtime short (~3 min total). For a full build, use:
#   scripts/dataset_building/build_pretraining_pfam.py
#
# Prerequisites:
#   pip install -e .
#   source environment.sh
#   ./scripts/sync_databases.sh <shared_path> data/databases
#
# Usage:
#   bash demos/build_source_datasets.sh

set -euo pipefail

OUTDIR="demos/outputs/source_datasets"
mkdir -p "$OUTDIR"

# --- Database paths ---
DAT_FILE="data/databases/swissprot/uniprot_sprot.dat.gz"
PFAM_FASTA="data/databases/pfam/Pfam-A.fasta.gz"
PFAM_STOCKHOLM="data/databases/pfam/Pfam-A.full.gz"

echo "============================================================"
echo "  Demo: Build Source Training CSVs from Raw Databases"
echo "============================================================"
echo ""

# ---------------------------------------------------------------
# Step 1: Build fully_annotated_swiss_prot.csv
# ---------------------------------------------------------------
# Parses uniprot_sprot.dat.gz to extract:
#   - protein sequences (SQ lines)
#   - annotations (CC blocks: function, catalytic activity, etc.)
#   - Pfam cross-references (DR Pfam lines)
#   - organism lineage (OC lines)
#   - Pfam family long names (from Pfam-A.full.gz metadata)
#
# Only entries with at least one Pfam annotation are included.
# Expected output: ~547K rows, ~780 MB. Takes ~1 min.

SWISSPROT_OUT="${OUTDIR}/fully_annotated_swiss_prot.csv"

echo "[1/3] Building SwissProt training CSV from ${DAT_FILE}..."
echo "      (Pfam metadata from ${PFAM_STOCKHOLM})"
echo ""

biom3_build_source_swissprot \
    --dat "$DAT_FILE" \
    --pfam-metadata "$PFAM_STOCKHOLM" \
    -o "$SWISSPROT_OUT"

echo ""
echo "      → $(wc -l < "$SWISSPROT_OUT") lines (incl. header)"
echo ""

# ---------------------------------------------------------------
# Step 2: Build Pfam CSV (SH3 subset for demo)
# ---------------------------------------------------------------
# For the demo, we extract only PF00018 (SH3 domain) entries from the
# full Pfam-A.fasta.gz to keep runtime short. This produces ~25K rows
# in ~2.5 min instead of ~63M rows / 52 GB for the full database.
#
# For the full build, use:
#   python scripts/dataset_building/build_pretraining_pfam.py

PFAM_SUBSET="${OUTDIR}/pfam_sh3_subset.fasta"
PFAM_OUT="${OUTDIR}/pfam_sh3_subset.csv"

echo "[2/3] Extracting SH3 (PF00018) entries from Pfam FASTA..."

# Extract SH3 entries from the full FASTA
zcat "$PFAM_FASTA" | awk '
    /^>/ { keep = /PF00018\./ }
    keep { print }
' > "$PFAM_SUBSET"

echo "      $(grep -c '^>' "$PFAM_SUBSET") FASTA entries extracted"
echo ""
echo "      Building Pfam CSV from subset..."

biom3_build_source_pfam \
    --fasta "$PFAM_SUBSET" \
    --pfam-metadata "$PFAM_STOCKHOLM" \
    -o "$PFAM_OUT"

echo ""
echo "      → $(wc -l < "$PFAM_OUT") lines (incl. header)"
echo ""

# ---------------------------------------------------------------
# Step 3: Use the newly built CSVs with biom3_build_dataset
# ---------------------------------------------------------------
# Feed the freshly-built source CSVs into the existing subsetting
# pipeline, demonstrating the full raw → finetuning path.

DATASET_OUT="${OUTDIR}/sh3_from_source"

echo "[3/3] Building SH3 fine-tuning dataset from the new source CSVs..."
echo ""

biom3_build_dataset \
    -p PF00018 \
    --swissprot "$SWISSPROT_OUT" \
    --pfam "$PFAM_OUT" \
    -o "$DATASET_OUT"

echo ""
echo "============================================================"
echo "  Done. Outputs in ${OUTDIR}/"
echo ""
echo "  Source CSVs (reproducible from raw databases):"
echo "    $(basename "$SWISSPROT_OUT")         — SwissProt proteins with captions"
echo "    $(basename "$PFAM_OUT")              — SH3 domain hits with captions"
echo ""
echo "  Fine-tuning dataset (from source CSVs):"
echo "    sh3_from_source/dataset.csv          — SH3 domain dataset"
echo ""
echo "  To compare with legacy-sourced dataset:"
echo "    bash demos/build_sh3_dataset.sh"
echo "============================================================"

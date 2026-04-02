#!/usr/bin/env python3
"""Build a finetuning dataset for a specific Pfam family.

Subsets the pretraining CSVs by Pfam ID using biom3_build_dataset.
This is a thin wrapper showing how the CLI entry point can be called
programmatically for scripted workflows.

Edit the configuration below, then run:
    python scripts/dataset_building/build_finetuning_dataset.py
"""

import subprocess
import sys

# --- Configuration (edit these) ---
PFAM_IDS = ["PF00018"]  # SH3 domain
OUTPUT_DIR = "outputs/finetuning/SH3"

SWISSPROT_CSV = "data/datasets/fully_annotated_swiss_prot.csv"
PFAM_CSV = "data/datasets/Pfam_protein_text_dataset.csv"

# Optional enrichment (uncomment to enable)
ENRICH = False
UNIPROT_DAT = "data/databases/swissprot/uniprot_sprot.dat.gz"
ADD_TAXONOMY = False
TAXID_INDEX = "data/databases/ncbi_taxonomy/accession2taxid.sqlite"
TAXONOMY_FILTER = None  # e.g., "superkingdom=Bacteria"


def main():
    cmd = [
        sys.executable, "-m", "biom3.dbio",
    ]
    # biom3_build_dataset args
    cmd = [
        "biom3_build_dataset",
        "-p", *PFAM_IDS,
        "--swissprot", SWISSPROT_CSV,
        "--pfam", PFAM_CSV,
        "-o", OUTPUT_DIR,
    ]

    if ENRICH:
        cmd.append("--enrich-pfam")
        cmd.extend(["--uniprot-dat", UNIPROT_DAT])

    if ADD_TAXONOMY:
        cmd.append("--add-taxonomy")
        cmd.extend(["--taxid-index", TAXID_INDEX])

    if TAXONOMY_FILTER:
        cmd.extend(["--taxonomy-filter", TAXONOMY_FILTER])

    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()

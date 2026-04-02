#!/usr/bin/env python3
"""Demo: Extract Pfam domain sequences for a single family.

Shows how to use iter_pfam_fasta() and _parse_fasta_header() directly
for custom processing without going through the full CSV builder.

Run:
    python demos/minimal_pfam_extract.py
"""

import csv
import os

from biom3.dbio.build_source_pfam import iter_pfam_fasta, _parse_fasta_header

FASTA_PATH = "data/databases/pfam/Pfam-A.fasta.gz"
TARGET_FAMILY = "PF00018"  # SH3 domain
OUTDIR = "demos/outputs/pfam_extract"
OUTPUT_PATH = os.path.join(OUTDIR, "sh3_domains.csv")


if __name__ == "__main__":
    os.makedirs(OUTDIR, exist_ok=True)
    count = 0
    with open(OUTPUT_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["accession", "range", "sequence"])

        for header, sequence in iter_pfam_fasta(FASTA_PATH):
            parsed = _parse_fasta_header(header)
            if parsed is None:
                continue
            if parsed["pfam_label"] != TARGET_FAMILY:
                continue

            writer.writerow([parsed["id"], parsed["range"], sequence])
            count += 1

    print(f"Extracted {count} {TARGET_FAMILY} domain sequences to {OUTPUT_PATH}")

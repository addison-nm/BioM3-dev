#!/usr/bin/env python3
"""Demo: Build SwissProt CSVs with custom caption formats.

Shows how CaptionSpec controls which annotation fields appear in captions,
their labels, ordering, and formatting — without modifying library code.

Run:
    python demo/custom_caption_format.py
"""

import os

from biom3.dbio.caption import CaptionSpec
from biom3.dbio.pfam_metadata import PfamMetadataParser
from biom3.dbio.build_source_swissprot import build_swissprot_csv

OUTDIR = "demo/outputs/custom_captions"

# Only include protein name and function — no lineage, no family names
minimal_spec = CaptionSpec(
    fields=[
        ("PROTEIN NAME", "annot_protein_name"),
        ("FUNCTION", "annot_function"),
    ],
    strip_pubmed=True,
    family_names_label=None,
)

# Include everything but with lowercase labels and semicolon separators
lowercase_spec = CaptionSpec(
    fields=[
        ("Protein name", "annot_protein_name"),
        ("Function", "annot_function"),
        ("Catalytic activity", "annot_catalytic_activity"),
        ("Lineage", "annot_lineage"),
    ],
    separator="; ",
    strip_pubmed=True,
    family_names_label="Family",
    family_names_template="{names}",
)

if __name__ == "__main__":
    os.makedirs(OUTDIR, exist_ok=True)
    metadata = PfamMetadataParser("data/databases/pfam/Pfam-A.full.gz").parse()

    print("Building with minimal spec (protein name + function only)...")
    build_swissprot_csv(
        dat_path="data/databases/swissprot/uniprot_sprot.dat.gz",
        pfam_metadata=metadata,
        output_path=os.path.join(OUTDIR, "swissprot_minimal.csv"),
        caption_spec=minimal_spec,
    )

    print("Building with lowercase spec...")
    build_swissprot_csv(
        dat_path="data/databases/swissprot/uniprot_sprot.dat.gz",
        pfam_metadata=metadata,
        output_path=os.path.join(OUTDIR, "swissprot_lowercase.csv"),
        caption_spec=lowercase_spec,
    )

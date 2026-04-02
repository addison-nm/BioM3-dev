"""Build fully_annotated_swiss_prot.csv from raw uniprot_sprot.dat.gz.

This produces a CSV that is a drop-in replacement for the legacy
fully_annotated_swiss_prot.csv used by build_dataset.py and SwissProtReader.

Usage:
    biom3_build_source_swissprot \\
        --dat data/databases/swissprot/uniprot_sprot.dat.gz \\
        --pfam-metadata data/databases/pfam/Pfam-A.full.gz \\
        -o data/datasets/fully_annotated_swiss_prot.csv
"""

import argparse
import csv
import sys

from biom3.backend.device import setup_logger
from biom3.dbio.caption import CaptionSpec, build_lineage_string, compose_row_caption
from biom3.dbio.pfam_metadata import PfamMetadataParser
from biom3.dbio.swissprot_dat import SwissProtDatParser

logger = setup_logger(__name__)

# Default caption spec matching the original fully_annotated_swiss_prot.csv.
SWISSPROT_SPEC = CaptionSpec(
    fields=[
        ("PROTEIN NAME",                  "annot_protein_name"),
        ("FUNCTION",                      "annot_function"),
        ("CATALYTIC ACTIVITY",            "annot_catalytic_activity"),
        ("COFACTOR",                      "annot_cofactor"),
        ("ACTIVITY REGULATION",           "annot_activity_regulation"),
        ("BIOPHYSICOCHEMICAL PROPERTIES", "annot_biophysicochemical_properties"),
        ("PATHWAY",                       "annot_pathway"),
        ("SUBUNIT",                       "annot_subunit"),
        ("SUBCELLULAR LOCATION",          "annot_subcellular_location"),
        ("TISSUE SPECIFICITY",            "annot_tissue_specificity"),
        ("DOMAIN",                        "annot_domain"),
        ("PTM",                           "annot_ptm"),
        ("SIMILARITY",                    "annot_similarity"),
        ("MISCELLANEOUS",                 "annot_miscellaneous"),
        ("INDUCTION",                     "annot_induction"),
        ("DEVELOPMENTAL STAGE",           "annot_developmental_stage"),
        ("BIOTECHNOLOGY",                 "annot_biotechnology"),
        ("GENE ONTOLOGY",                 "annot_gene_ontology"),
        ("LINEAGE",                       "annot_lineage"),
    ],
    strip_pubmed=True,
    strip_evidence=True,
    family_names_label="FAMILY NAMES",
    family_names_template="Family names are {names}",
)

OUTPUT_COLUMNS = [
    "primary_Accession",
    "protein_sequence",
    "[final]text_caption",
    "pfam_label",
]


def _format_pfam_label(pfam_ids):
    """Format Pfam IDs as a Python list string like the original CSV."""
    return repr(pfam_ids)


def build_swissprot_csv(dat_path, pfam_metadata, output_path,
                        caption_spec=None, taxonomy_tree=None,
                        taxonomy_ranks=None, chunk_size=10_000):
    """Build fully_annotated_swiss_prot.csv from raw Swiss-Prot .dat file.

    Args:
        dat_path: path to uniprot_sprot.dat.gz
        pfam_metadata: dict from PfamMetadataParser.parse()
        output_path: output CSV path
        caption_spec: CaptionSpec controlling caption format. Defaults to
            SWISSPROT_SPEC (original BioM3 format).
        taxonomy_tree: optional TaxonomyTree instance. When provided, uses
            NCBI ranked lineage (via OX tax_id) instead of flat OC lineage.
            This enables partial lineage via taxonomy_ranks.
        taxonomy_ranks: optional list of rank names to include in lineage
            (e.g. ["superkingdom", "phylum"]). Only used when taxonomy_tree
            is provided. If None, all non-empty ranks are included.
        chunk_size: rows to buffer before writing
    """
    if caption_spec is None:
        caption_spec = SWISSPROT_SPEC

    parser = SwissProtDatParser(dat_path)
    buffer = []
    total = 0

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(OUTPUT_COLUMNS)

        for accession, entry in parser.parse_all(require_pfam=True):
            pfam_ids = entry["pfam_ids"]
            annotations = entry["annotations"]

            # Override lineage with rank-structured NCBI lineage if tree provided
            if taxonomy_tree is not None:
                tax_id = entry.get("tax_id")
                if tax_id:
                    lineage_dict = taxonomy_tree.get_lineage(tax_id)
                    if lineage_dict:
                        lineage_str = build_lineage_string(
                            lineage_dict, ranks=taxonomy_ranks,
                        )
                        if lineage_str:
                            annotations["annot_lineage"] = lineage_str
                        else:
                            annotations.pop("annot_lineage", None)
                    else:
                        annotations.pop("annot_lineage", None)

            family_names = []
            for pid in pfam_ids:
                meta = pfam_metadata.get(pid)
                if meta:
                    family_names.append(meta["family_name"])
                else:
                    family_names.append(pid)

            caption = compose_row_caption(
                annotations, caption_spec, pfam_family_names=family_names,
            )

            buffer.append([
                accession,
                entry["sequence"],
                caption,
                _format_pfam_label(pfam_ids),
            ])

            if len(buffer) >= chunk_size:
                writer.writerows(buffer)
                total += len(buffer)
                buffer = []
                logger.info("Written %s rows", f"{total:,}")

        if buffer:
            writer.writerows(buffer)
            total += len(buffer)

    logger.info("Build complete: %s rows written to %s", f"{total:,}", output_path)


def parse_arguments(args):
    parser = argparse.ArgumentParser(
        description="Build fully_annotated_swiss_prot.csv from raw Swiss-Prot .dat file."
    )
    parser.add_argument(
        "--dat", type=str, required=True,
        help="Path to uniprot_sprot.dat.gz",
    )
    parser.add_argument(
        "--pfam-metadata", type=str, required=True,
        help="Path to Pfam-A.full.gz or Pfam-A.hmm.gz for family name lookup",
    )
    parser.add_argument(
        "-o", "--output", type=str, required=True,
        help="Output CSV path",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=10_000,
        help="Rows to buffer before writing (default: 10000)",
    )
    return parser.parse_args(args)


def main(args):
    logger.info("Loading Pfam family metadata from %s", args.pfam_metadata)
    pfam_metadata = PfamMetadataParser(args.pfam_metadata).parse()

    build_swissprot_csv(
        dat_path=args.dat,
        pfam_metadata=pfam_metadata,
        output_path=args.output,
        chunk_size=args.chunk_size,
    )


if __name__ == "__main__":
    main(parse_arguments(sys.argv[1:]))

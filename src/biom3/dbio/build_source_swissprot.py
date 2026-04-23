"""Build fully_annotated_swiss_prot.csv from raw uniprot_sprot.dat.gz.

This produces a CSV that is a drop-in replacement for the legacy
fully_annotated_swiss_prot.csv used by build_dataset.py and SwissProtReader.

Usage:
    biom3_build_source_swissprot \\
        --dat data/databases/swissprot/uniprot_sprot.dat.gz \\
        --pfam_metadata data/databases/pfam/Pfam-A.full.gz \\
        -o data/datasets/fully_annotated_swiss_prot.csv
"""

import argparse
import csv
import os
import re
import sys
from dataclasses import replace
from datetime import datetime

from biom3.backend.device import setup_logger
from biom3.core.run_utils import (
    get_file_metadata,
    write_manifest,
)
from biom3.dbio.caption import CaptionSpec, build_lineage_string, compose_row_caption
from biom3.dbio.pfam_metadata import PfamMetadataParser
from biom3.dbio.stats import IncrementalStatsBuilder, write_stats_markdown
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
    "annot_ec_numbers",
]

OUTPUT_COLUMNS_WITH_INTERMEDIATES = [
    "primary_Accession",
    "protein_sequence",
    "text_caption",
    "[clean]text_caption",
    "[final]text_caption",
    "pfam_label",
    "annot_ec_numbers",
]


def _format_pfam_label(pfam_ids, require_pfam=False):
    """Format Pfam IDs as a Python list string like the original CSV.

    When ``pfam_ids`` is empty, emit ``['nan']`` to match the legacy
    SwissProt CSV sentinel for Pfam-less entries. If ``require_pfam`` is
    True the caller filters those entries upstream and this branch is
    never hit.
    """
    if not pfam_ids and not require_pfam:
        return repr(["nan"])
    return repr(pfam_ids)


def _read_release_version(filepath, pattern):
    """Read a release file and extract a version string matching *pattern*."""
    try:
        with open(filepath) as f:
            text = f.read(2048)
        match = re.search(pattern, text)
        return match.group(1).strip() if match else None
    except Exception:
        return None


def build_swissprot_csv(dat_path, pfam_metadata, output_path,
                        caption_spec=None, taxonomy_tree=None,
                        taxonomy_ranks=None, chunk_size=10_000,
                        require_pfam=False, keep_intermediate_captions=False):
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
        require_pfam: if True, skip entries without Pfam DR cross-refs.
            Default False, which keeps them with ``pfam_label=['nan']`` to
            match the legacy CSV format.
        keep_intermediate_captions: if True, also emit ``text_caption``
            (raw, with PubMed refs and ECO tags intact) and
            ``[clean]text_caption`` (ECO tags stripped, PubMed refs kept)
            columns alongside ``[final]text_caption``. Default: False.

    Returns:
        int: number of rows written.
    """
    if caption_spec is None:
        caption_spec = SWISSPROT_SPEC

    if keep_intermediate_captions:
        raw_spec = replace(caption_spec, strip_pubmed=False, strip_evidence=False)
        clean_spec = replace(caption_spec, strip_pubmed=False, strip_evidence=True)
        output_columns = OUTPUT_COLUMNS_WITH_INTERMEDIATES
    else:
        raw_spec = None
        clean_spec = None
        output_columns = OUTPUT_COLUMNS

    annotation_fields = [field for (_, field) in caption_spec.fields]
    stats_builder = IncrementalStatsBuilder(
        annotation_fields=annotation_fields,
        seq_field="sequence",
        pfam_field="pfam_ids",
        caption_field="caption",
    )

    parser = SwissProtDatParser(dat_path)
    buffer = []
    total = 0

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(output_columns)

        for accession, entry in parser.parse_all(require_pfam=require_pfam):
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

            final_caption = compose_row_caption(
                annotations, caption_spec, pfam_family_names=family_names,
            )

            ec_numbers = annotations.get("annot_ec_numbers", "")

            if keep_intermediate_captions:
                raw_caption = compose_row_caption(
                    annotations, raw_spec, pfam_family_names=family_names,
                )
                clean_caption = compose_row_caption(
                    annotations, clean_spec, pfam_family_names=family_names,
                )
                row = [
                    accession,
                    entry["sequence"],
                    raw_caption,
                    clean_caption,
                    final_caption,
                    _format_pfam_label(pfam_ids, require_pfam=require_pfam),
                    ec_numbers,
                ]
            else:
                row = [
                    accession,
                    entry["sequence"],
                    final_caption,
                    _format_pfam_label(pfam_ids, require_pfam=require_pfam),
                    ec_numbers,
                ]

            stats_builder.update({
                "sequence": entry["sequence"],
                "pfam_ids": pfam_ids,
                "caption": final_caption,
                **annotations,
            })
            buffer.append(row)

            if len(buffer) >= chunk_size:
                writer.writerows(buffer)
                total += len(buffer)
                buffer = []
                logger.info("Written %s rows", f"{total:,}")

        if buffer:
            writer.writerows(buffer)
            total += len(buffer)

    logger.info("Build complete: %s rows written to %s", f"{total:,}", output_path)
    return total, stats_builder


def _collect_database_versions(dat_path, pfam_metadata_path):
    """Collect provenance metadata for the UniProt and Pfam source files."""
    versions = {"uniprot_dat": get_file_metadata(dat_path),
                "pfam_metadata": get_file_metadata(pfam_metadata_path)}

    dat_dir = os.path.dirname(os.path.abspath(dat_path))
    reldate_path = os.path.join(dat_dir, "reldate.txt")
    if os.path.exists(reldate_path):
        versions["uniprot_release"] = _read_release_version(
            reldate_path, r"Release\s+(\S+)",
        )

    pfam_dir = os.path.dirname(os.path.abspath(pfam_metadata_path))
    relnotes_path = os.path.join(pfam_dir, "relnotes.txt")
    if os.path.exists(relnotes_path):
        versions["pfam_release"] = _read_release_version(
            relnotes_path, r"RELEASE\s+(\S+)",
        )

    return versions


def parse_arguments(args):
    parser = argparse.ArgumentParser(
        description="Build fully_annotated_swiss_prot.csv from raw Swiss-Prot .dat file."
    )
    parser.add_argument(
        "--dat", type=str, required=True,
        help="Path to uniprot_sprot.dat.gz",
    )
    parser.add_argument(
        "--pfam_metadata", type=str, required=True,
        help="Path to Pfam-A.full.gz or Pfam-A.hmm.gz for family name lookup",
    )
    parser.add_argument(
        "-o", "--output", type=str, required=True,
        help="Output CSV path",
    )
    parser.add_argument(
        "--chunk_size", type=int, default=10_000,
        help="Rows to buffer before writing (default: 10000)",
    )

    require_group = parser.add_mutually_exclusive_group()
    require_group.add_argument(
        "--require_pfam", dest="require_pfam", action="store_true",
        help="Skip entries without Pfam DR cross-refs.",
    )
    require_group.add_argument(
        "--no_require_pfam", dest="require_pfam", action="store_false",
        help="Keep entries without Pfam DR cross-refs; their pfam_label "
             "is emitted as ['nan'] for legacy parity (default).",
    )
    parser.set_defaults(require_pfam=False)

    parser.add_argument(
        "--keep_intermediate_captions", action="store_true", default=False,
        help="Emit text_caption (raw) and [clean]text_caption "
             "(evidence-stripped only) columns alongside [final]text_caption "
             "for auditing the PubMed/ECO stripping passes.",
    )
    return parser.parse_args(args)


def main(args):
    start_time = datetime.now()

    logger.info("Loading Pfam family metadata from %s", args.pfam_metadata)
    pfam_metadata = PfamMetadataParser(args.pfam_metadata).parse()

    row_count, stats_builder = build_swissprot_csv(
        dat_path=args.dat,
        pfam_metadata=pfam_metadata,
        output_path=args.output,
        chunk_size=args.chunk_size,
        require_pfam=args.require_pfam,
        keep_intermediate_captions=args.keep_intermediate_captions,
    )

    elapsed = datetime.now() - start_time
    stats = stats_builder.finalize()

    outdir = os.path.dirname(os.path.abspath(args.output))
    stem = os.path.splitext(os.path.basename(args.output))[0]

    stats_path = os.path.join(outdir, f"{stem}.stats.md")
    write_stats_markdown(stats, stats_path, title=f"{stem} — coverage stats")
    logger.info("Saved stats report to %s", stats_path)

    database_versions = _collect_database_versions(args.dat, args.pfam_metadata)
    resolved_paths = {
        "dat": os.path.abspath(args.dat),
        "pfam_metadata": os.path.abspath(args.pfam_metadata),
        "output": os.path.abspath(args.output),
        "stats_markdown": stats_path,
    }
    manifest_path = write_manifest(
        args, outdir, start_time, elapsed,
        outputs={"row_counts": {"swissprot": row_count}},
        resolved_paths=resolved_paths,
        database_versions=database_versions,
        stats=stats,
        manifest_filename=f"{stem}.build_manifest.json",
    )
    logger.info("Saved build manifest to %s", manifest_path)


if __name__ == "__main__":
    main(parse_arguments(sys.argv[1:]))

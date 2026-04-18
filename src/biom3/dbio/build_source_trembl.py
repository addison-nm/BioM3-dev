"""Build fully_annotated_trembl.csv from raw uniprot_trembl.dat.gz.

TrEMBL shares the UniProt .dat format with SwissProt, so this builder reuses
SwissProtDatParser and the SWISSPROT_SPEC CaptionSpec. The key difference is
that TrEMBL annotations are largely auto-propagated (ECO:0000256, ECO:0000313),
so this builder adds an evidence filter that defaults to skipping purely
automatic entries.

Usage:
    biom3_build_source_trembl \\
        --dat data/databases/trembl/uniprot_trembl.dat.gz \\
        --pfam_metadata data/databases/pfam/Pfam-A.full.gz \\
        -o data/datasets/fully_annotated_trembl.csv \\
        --evidence_filter lenient
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
from biom3.dbio.build_source_swissprot import (
    OUTPUT_COLUMNS,
    OUTPUT_COLUMNS_WITH_INTERMEDIATES,
    SWISSPROT_SPEC,
    _format_pfam_label,
    _read_release_version,
)
from biom3.dbio.caption import build_lineage_string, compose_row_caption
from biom3.dbio.pfam_metadata import PfamMetadataParser
from biom3.dbio.stats import IncrementalStatsBuilder, write_stats_markdown
from biom3.dbio.swissprot_dat import SwissProtDatParser

logger = setup_logger(__name__)

# ECO codes that UniProt marks as pure-automatic. Entries whose annotations
# cite only these are dropped in the default 'lenient' evidence mode.
AUTOMATIC_ECO_CODES = {"ECO:0000256", "ECO:0000313"}
EXPERIMENTAL_ECO_CODE = "ECO:0000269"
ECO_PATTERN = re.compile(r"ECO:\d{7}")


def _iter_eco_codes(annotations):
    """Yield every ECO code that appears in any annotation value."""
    for value in annotations.values():
        if not isinstance(value, str):
            continue
        for match in ECO_PATTERN.findall(value):
            yield match


def _passes_evidence_filter(annotations, mode):
    """Return True if *annotations* satisfy the evidence-filter *mode*.

    Modes:
      - 'any': no filter, always True.
      - 'lenient': at least one ECO code not in AUTOMATIC_ECO_CODES.
      - 'strict': at least one ECO:0000269 (experimental) code.
    """
    if mode == "any":
        return True
    if mode == "strict":
        return any(eco == EXPERIMENTAL_ECO_CODE for eco in _iter_eco_codes(annotations))
    if mode == "lenient":
        return any(eco not in AUTOMATIC_ECO_CODES for eco in _iter_eco_codes(annotations))
    raise ValueError(f"Unknown evidence filter mode: {mode}")


def build_trembl_csv(dat_path, pfam_metadata, output_path,
                     caption_spec=None, taxonomy_tree=None,
                     taxonomy_ranks=None, chunk_size=10_000,
                     require_pfam=False, keep_intermediate_captions=False,
                     evidence_filter="lenient", limit=None):
    """Build fully_annotated_trembl.csv from raw TrEMBL .dat file.

    Args match `build_swissprot_csv` plus:
        evidence_filter: 'strict' | 'lenient' | 'any'. See _passes_evidence_filter.
        limit: if given, stop after writing this many rows (for testing).

    Returns:
        (int, int, int): (rows_written, skipped_no_pfam, skipped_by_evidence).
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
    skipped_eco = 0
    skipped_pfam = 0

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(output_columns)

        for accession, entry in parser.parse_all(require_pfam=False):
            pfam_ids = entry["pfam_ids"]
            annotations = entry["annotations"]

            if require_pfam and not pfam_ids:
                skipped_pfam += 1
                continue

            if not _passes_evidence_filter(annotations, evidence_filter):
                skipped_eco += 1
                continue

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
                ]
            else:
                row = [
                    accession,
                    entry["sequence"],
                    final_caption,
                    _format_pfam_label(pfam_ids, require_pfam=require_pfam),
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
                logger.info("Written %s rows (skipped %s by evidence, %s by pfam)",
                            f"{total:,}", f"{skipped_eco:,}", f"{skipped_pfam:,}")

            if limit is not None and total + len(buffer) >= limit:
                break

        if buffer:
            writer.writerows(buffer)
            total += len(buffer)

    logger.info(
        "Build complete: %s rows written to %s (skipped %s by evidence, %s by pfam)",
        f"{total:,}", output_path, f"{skipped_eco:,}", f"{skipped_pfam:,}",
    )
    return total, skipped_pfam, skipped_eco, stats_builder


def _collect_database_versions(dat_path, pfam_metadata_path):
    versions = {"trembl_dat": get_file_metadata(dat_path),
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
        description="Build fully_annotated_trembl.csv from raw TrEMBL .dat file."
    )
    parser.add_argument(
        "--dat", type=str, required=True,
        help="Path to uniprot_trembl.dat.gz",
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
        help="Keep entries without Pfam DR cross-refs with pfam_label=['nan'] (default).",
    )
    parser.set_defaults(require_pfam=False)

    parser.add_argument(
        "--keep_intermediate_captions", action="store_true", default=False,
        help="Emit text_caption (raw) and [clean]text_caption columns "
             "alongside [final]text_caption for auditing.",
    )
    parser.add_argument(
        "--evidence_filter", choices=["strict", "lenient", "any"],
        default="lenient",
        help="Filter entries by UniProt evidence code. "
             "'strict': require ECO:0000269 (experimental evidence). "
             "'lenient' (default): require at least one ECO code other than "
             "ECO:0000256/ECO:0000313 (pure-automatic). "
             "'any': no filter — includes fully automatic entries.",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Stop after writing this many rows (for testing).",
    )
    return parser.parse_args(args)


def main(args):
    start_time = datetime.now()

    logger.info("Loading Pfam family metadata from %s", args.pfam_metadata)
    pfam_metadata = PfamMetadataParser(args.pfam_metadata).parse()

    row_count, skipped_pfam, skipped_eco, stats_builder = build_trembl_csv(
        dat_path=args.dat,
        pfam_metadata=pfam_metadata,
        output_path=args.output,
        chunk_size=args.chunk_size,
        require_pfam=args.require_pfam,
        keep_intermediate_captions=args.keep_intermediate_captions,
        evidence_filter=args.evidence_filter,
        limit=args.limit,
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
        outputs={
            "row_counts": {"trembl": row_count},
            "skipped": {"by_pfam": skipped_pfam, "by_evidence": skipped_eco},
        },
        resolved_paths=resolved_paths,
        database_versions=database_versions,
        stats=stats,
        manifest_filename=f"{stem}.build_manifest.json",
    )
    logger.info("Saved build manifest to %s", manifest_path)


if __name__ == "__main__":
    main(parse_arguments(sys.argv[1:]))

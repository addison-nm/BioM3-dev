"""Build expasy_enzyme.csv from ExPASy `enzyme.dat`.

The resulting CSV is the canonical EC-number -> UniProt/description/cofactor
bridge used by `enrich.py` for EC-based joins (and downstream by BRENDA).

Usage:
    biom3_build_source_expasy \\
        --dat data/databases/expasy/enzyme.dat \\
        -o data/datasets/expasy_enzyme.csv
"""

import argparse
import csv
import os
import re
import sys
from datetime import datetime

import pandas as pd

from biom3.backend.device import setup_logger
from biom3.core.run_utils import get_file_metadata, write_manifest
from biom3.dbio.caption import CaptionSpec, compose_row_caption
from biom3.dbio.expasy import ExPASyEnzymeParser
from biom3.dbio.stats import compute_coverage_stats, write_stats_markdown

logger = setup_logger(__name__)

EXPASY_SPEC = CaptionSpec(
    fields=[
        ("ENZYME NAME",          "annot_name"),
        ("ALTERNATIVE NAMES",    "annot_alternative_names"),
        ("CATALYTIC ACTIVITY",   "annot_catalytic_activity"),
        ("COFACTOR",             "annot_cofactor"),
        ("COMMENTS",             "annot_comments"),
    ],
    strip_pubmed=False,
    strip_evidence=False,
    family_names_label=None,
)

OUTPUT_COLUMNS = [
    "ec",
    "annot_name",
    "annot_alternative_names",
    "annot_catalytic_activity",
    "annot_cofactor",
    "annot_comments",
    "annot_uniprot_accessions",
    "uniprot_count",
    "transferred_to",
    "deleted",
    "[final]text_caption",
]


def _join(values, sep="; "):
    return sep.join(v for v in values if v)


def build_expasy_csv(dat_path, output_path, caption_spec=None,
                     include_obsolete=True, chunk_size=1_000):
    """Build expasy_enzyme.csv.

    Args:
        dat_path: path to enzyme.dat
        output_path: output CSV
        caption_spec: CaptionSpec for [final]text_caption. Defaults to EXPASY_SPEC.
        include_obsolete: if False, skip Transferred and Deleted entries.
            Default True — obsolete entries are preserved so EC-number
            lookups for legacy data still resolve to a forwarding record.
        chunk_size: rows to buffer before writing

    Returns:
        (int, pd.DataFrame): row count and the full DataFrame (safe to keep
        in memory — ExPASy is ~8k rows).
    """
    if caption_spec is None:
        caption_spec = EXPASY_SPEC

    parser = ExPASyEnzymeParser(dat_path)
    rows = []

    for entry in parser.iter_entries():
        is_obsolete = bool(entry.deleted or entry.transferred_to)
        if is_obsolete and not include_obsolete:
            continue

        annotations = {
            "annot_name": entry.name,
            "annot_alternative_names": _join(entry.alternative_names),
            "annot_catalytic_activity": _join(entry.catalytic_activities, sep=" | "),
            "annot_cofactor": _join(entry.cofactors),
            "annot_comments": _join(entry.comments, sep=" "),
        }
        caption = compose_row_caption(annotations, caption_spec)

        rows.append({
            "ec": entry.ec,
            **annotations,
            "annot_uniprot_accessions": ",".join(entry.uniprot_accessions),
            "uniprot_count": len(entry.uniprot_accessions),
            "transferred_to": ",".join(entry.transferred_to),
            "deleted": entry.deleted,
            "[final]text_caption": caption,
        })

    df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    df.to_csv(output_path, index=False, quoting=csv.QUOTE_MINIMAL)

    logger.info("Build complete: %s rows written to %s", f"{len(df):,}", output_path)
    return len(df), df


def _read_release_version(dat_path):
    """Extract ExPASy release date from the header block of enzyme.dat."""
    with open(dat_path) as fh:
        head = fh.read(4096)
    match = re.search(r"Release of (\S+?)$", head, re.MULTILINE)
    return match.group(1).strip() if match else None


def _collect_database_versions(dat_path):
    return {
        "expasy_dat": get_file_metadata(dat_path),
        "expasy_release": _read_release_version(dat_path),
    }


def parse_arguments(args):
    parser = argparse.ArgumentParser(
        description="Build expasy_enzyme.csv from ExPASy enzyme.dat."
    )
    parser.add_argument(
        "--dat", type=str, required=True,
        help="Path to data/databases/expasy/enzyme.dat",
    )
    parser.add_argument(
        "-o", "--output", type=str, required=True,
        help="Output CSV path",
    )

    obsolete_group = parser.add_mutually_exclusive_group()
    obsolete_group.add_argument(
        "--include_obsolete", dest="include_obsolete", action="store_true",
        help="Include Transferred and Deleted EC entries (default).",
    )
    obsolete_group.add_argument(
        "--exclude_obsolete", dest="include_obsolete", action="store_false",
        help="Skip Transferred and Deleted EC entries.",
    )
    parser.set_defaults(include_obsolete=True)

    return parser.parse_args(args)


def main(args):
    start_time = datetime.now()

    row_count, df = build_expasy_csv(
        dat_path=args.dat,
        output_path=args.output,
        include_obsolete=args.include_obsolete,
    )

    elapsed = datetime.now() - start_time

    stats = compute_coverage_stats(df)
    stats["uniprot_crosslinks"] = {
        "total": int(df["uniprot_count"].sum()),
        "mean_per_ec": round(float(df["uniprot_count"].mean()), 2) if len(df) else 0.0,
        "max": int(df["uniprot_count"].max()) if len(df) else 0,
    }
    stats["obsolete"] = {
        "transferred": int((df["transferred_to"] != "").sum()),
        "deleted": int(df["deleted"].sum()),
    }

    outdir = os.path.dirname(os.path.abspath(args.output))
    stem = os.path.splitext(os.path.basename(args.output))[0]

    stats_path = os.path.join(outdir, f"{stem}.stats.md")
    write_stats_markdown(stats, stats_path, title=f"{stem} — coverage stats")
    logger.info("Saved stats report to %s", stats_path)

    resolved_paths = {
        "dat": os.path.abspath(args.dat),
        "output": os.path.abspath(args.output),
        "stats_markdown": stats_path,
    }
    manifest_path = write_manifest(
        args, outdir, start_time, elapsed,
        outputs={"row_counts": {"expasy": row_count}},
        resolved_paths=resolved_paths,
        database_versions=_collect_database_versions(args.dat),
        stats=stats,
        manifest_filename=f"{stem}.build_manifest.json",
    )
    logger.info("Saved build manifest to %s", manifest_path)


if __name__ == "__main__":
    main(parse_arguments(sys.argv[1:]))

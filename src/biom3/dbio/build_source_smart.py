"""Build smart_domains.csv from SMART `SMART_domains.txt`.

SMART provides a domain-name + description lookup table keyed by SMART
accessions (SM\\d{5}). The resulting CSV is joined against UniProt's
`DR SMART;` cross-refs during dataset enrichment.

Usage:
    biom3_build_source_smart \\
        --input data/databases/smart/SMART_domains.txt \\
        -o data/datasets/smart_domains.csv
"""

import argparse
import csv
import os
import sys
from datetime import datetime

import pandas as pd

from biom3.backend.device import setup_logger
from biom3.core.run_utils import get_file_metadata, write_manifest
from biom3.dbio.caption import CaptionSpec, compose_row_caption
from biom3.dbio.smart import SmartReader
from biom3.dbio.stats import compute_coverage_stats, write_stats_markdown

logger = setup_logger(__name__)

SMART_SPEC = CaptionSpec(
    fields=[
        ("SMART DOMAIN",      "annot_domain_name"),
        ("DEFINITION",        "annot_definition"),
        ("DESCRIPTION",       "annot_description"),
    ],
    strip_pubmed=False,
    strip_evidence=False,
    family_names_label=None,
)

OUTPUT_COLUMNS = [
    "domain_id",
    "annot_domain_name",
    "annot_definition",
    "annot_description",
    "[final]text_caption",
]


def build_smart_csv(input_path, output_path, caption_spec=None):
    """Parse SMART_domains.txt and write smart_domains.csv.

    Returns:
        (int, pd.DataFrame): row count and the DataFrame (small, ~1400 rows).
    """
    if caption_spec is None:
        caption_spec = SMART_SPEC

    reader = SmartReader(input_path)
    rows = []
    for dom in reader.iter_domains():
        annotations = {
            "annot_domain_name": dom["domain_name"],
            "annot_definition": dom["definition"],
            "annot_description": dom["description"],
        }
        caption = compose_row_caption(annotations, caption_spec)
        rows.append({
            "domain_id": dom["accession"],
            **annotations,
            "[final]text_caption": caption,
        })

    df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    df.to_csv(output_path, index=False, quoting=csv.QUOTE_MINIMAL)
    logger.info("Build complete: %s rows written to %s", f"{len(df):,}", output_path)
    return len(df), df


def parse_arguments(args):
    parser = argparse.ArgumentParser(
        description="Build smart_domains.csv from SMART_domains.txt."
    )
    parser.add_argument(
        "--input", type=str, required=True,
        help="Path to data/databases/smart/SMART_domains.txt",
    )
    parser.add_argument(
        "-o", "--output", type=str, required=True,
        help="Output CSV path",
    )
    return parser.parse_args(args)


def main(args):
    start_time = datetime.now()

    row_count, df = build_smart_csv(input_path=args.input, output_path=args.output)

    elapsed = datetime.now() - start_time
    stats = compute_coverage_stats(df)

    outdir = os.path.dirname(os.path.abspath(args.output))
    stem = os.path.splitext(os.path.basename(args.output))[0]

    stats_path = os.path.join(outdir, f"{stem}.stats.md")
    write_stats_markdown(stats, stats_path, title=f"{stem} — coverage stats")
    logger.info("Saved stats report to %s", stats_path)

    resolved_paths = {
        "input": os.path.abspath(args.input),
        "output": os.path.abspath(args.output),
        "stats_markdown": stats_path,
    }
    manifest_path = write_manifest(
        args, outdir, start_time, elapsed,
        outputs={"row_counts": {"smart": row_count}},
        resolved_paths=resolved_paths,
        database_versions={"smart_domains": get_file_metadata(args.input)},
        stats=stats,
        manifest_filename=f"{stem}.build_manifest.json",
    )
    logger.info("Saved build manifest to %s", manifest_path)


if __name__ == "__main__":
    main(parse_arguments(sys.argv[1:]))

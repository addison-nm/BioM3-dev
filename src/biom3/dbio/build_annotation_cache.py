"""Build and load annotation Parquet caches from UniProt .dat files.

One-time build:
    biom3_build_annotation_cache --dat uniprot_trembl.dat.gz -o trembl_annotations.parquet

Per-build lookup:
    biom3_build_dataset -p PF00018 --enrich_pfam \
        --annotation_cache trembl_annotations.parquet -o outputs/SH3
"""

import argparse
import os
import sys
from datetime import datetime

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from biom3.backend.device import setup_logger
from biom3.dbio.enrich import ANNOTATION_COLUMNS

logger = setup_logger(__name__)

# Annotation columns produced by SwissProtDatParser._parse_entry_full().
# This is a superset of enrich.ANNOTATION_COLUMNS: it includes the 3 extra
# columns (tissue_specificity, developmental_stage, biotechnology) that
# _parse_entry_full() extracts but that are not yet in the canonical
# ANNOTATION_FIELDS list.  Including them future-proofs the cache.
_EXTRA_COLUMNS = [
    "annot_tissue_specificity",
    "annot_developmental_stage",
    "annot_biotechnology",
]

# Columns stored in the cache.  Excludes annot_family_name and
# annot_family_description, which come from Pfam metadata, not .dat files.
CACHE_ANNOTATION_COLUMNS = [
    col for col in ANNOTATION_COLUMNS
    if col not in ("annot_family_name", "annot_family_description")
] + [
    col for col in _EXTRA_COLUMNS
    if col not in ANNOTATION_COLUMNS
]

CACHE_SCHEMA = pa.schema(
    [pa.field("primary_Accession", pa.string(), nullable=False)]
    + [pa.field(col, pa.string(), nullable=True) for col in CACHE_ANNOTATION_COLUMNS]
)


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def build_annotation_cache(
    dat_path,
    output_path,
    chunk_size=100_000,
    require_annotation=True,
):
    """Parse a UniProt .dat(.gz) file and write annotations to Parquet.

    Args:
        dat_path: path to .dat or .dat.gz file.
        output_path: destination .parquet file.
        chunk_size: rows buffered before writing a row group.
        require_annotation: if True, skip entries with zero annotations.
    """
    from biom3.dbio.swissprot_dat import SwissProtDatParser

    logger.info("Building annotation cache: %s -> %s", dat_path, output_path)

    parser = SwissProtDatParser(dat_path)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    writer = pq.ParquetWriter(output_path, CACHE_SCHEMA)
    buffer = []
    total_written = 0
    skipped = 0

    for accession, entry in parser.parse_all(require_pfam=False):
        annots = entry["annotations"]
        if require_annotation and not annots:
            skipped += 1
            continue

        row = {"primary_Accession": accession}
        for col in CACHE_ANNOTATION_COLUMNS:
            row[col] = annots.get(col)
        buffer.append(row)

        if len(buffer) >= chunk_size:
            _flush(writer, buffer)
            total_written += len(buffer)
            logger.info("Written %s rows...", f"{total_written:,}")
            buffer = []

    if buffer:
        _flush(writer, buffer)
        total_written += len(buffer)

    writer.close()

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    logger.info(
        "Done: %s rows written (%s skipped), %.1f MB",
        f"{total_written:,}", f"{skipped:,}", size_mb,
    )


def _flush(writer, buffer):
    """Sort buffer by accession and write as a Parquet row group."""
    table = pa.Table.from_pylist(buffer, schema=CACHE_SCHEMA)
    table = table.sort_by("primary_Accession")
    writer.write_table(table)


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------

def load_annotation_cache(cache_paths, accessions):
    """Load annotations for a set of accessions from Parquet cache(s).

    Args:
        cache_paths: list of paths to annotation .parquet files.
        accessions: iterable of accession strings to look up.

    Returns:
        dict mapping accession -> annotation dict (annot_* keys only,
        sparse — missing annotations are absent, not None).  Same format
        as SwissProtDatParser.parse().
    """
    accession_set = set(accessions)
    results = {}

    for path in cache_paths:
        remaining = accession_set - set(results.keys())
        if not remaining:
            break

        logger.info("Loading annotation cache: %s (%s accessions to find)",
                     path, f"{len(remaining):,}")

        table = pq.read_table(
            path,
            filters=[("primary_Accession", "in", list(remaining))],
        )
        df = table.to_pandas()

        for _, row in df.iterrows():
            acc = row["primary_Accession"]
            annots = {}
            for col in CACHE_ANNOTATION_COLUMNS:
                val = row.get(col)
                if pd.notna(val) and str(val).strip():
                    annots[col] = str(val)
            results[acc] = annots

        logger.info("Cache %s: %s/%s accessions found",
                     os.path.basename(path), f"{len(results):,}",
                     f"{len(accession_set):,}")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_arguments(args):
    parser = argparse.ArgumentParser(
        description="Build an annotation Parquet cache from a UniProt .dat file.",
    )
    parser.add_argument(
        "--dat", type=str, required=True,
        help="Path to UniProt .dat or .dat.gz file",
    )
    parser.add_argument(
        "-o", "--output", type=str, required=True,
        help="Output .parquet file path",
    )
    parser.add_argument(
        "--chunk_size", type=int, default=100_000,
        help="Rows to buffer before writing a row group (default: 100000)",
    )
    parser.add_argument(
        "--no_require_annotation", action="store_true",
        help="Include entries with zero annotation fields",
    )
    return parser.parse_args(args)


def main(args):
    start = datetime.now()
    build_annotation_cache(
        dat_path=args.dat,
        output_path=args.output,
        chunk_size=args.chunk_size,
        require_annotation=not args.no_require_annotation,
    )
    elapsed = datetime.now() - start
    logger.info("Elapsed: %s", elapsed)

"""Convert large CSV files to Parquet for faster repeated queries.

Parquet is columnar and compressed, making reads ~5-10x faster than CSV.
The Pfam CSV (35 GB, 44.8M rows) shrinks to ~5-8 GB in Parquet with
predicate pushdown support for instant Pfam ID filtering.
"""

import argparse
import os
import sys

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from biom3.backend.device import setup_logger

logger = setup_logger(__name__)


def csv_to_parquet(csv_path, parquet_path=None, chunk_size=500_000,
                   row_group_size=500_000, dtype_overrides=None):
    """Convert a CSV file to Parquet format.

    Reads the CSV in chunks and writes to a single Parquet file with
    row groups for efficient partial reads.

    Args:
        csv_path: path to input CSV file.
        parquet_path: path to output Parquet file. Defaults to replacing
            the .csv extension with .parquet.
        chunk_size: rows per chunk when reading the CSV.
        row_group_size: rows per row group in the Parquet file.
        dtype_overrides: optional dict of column name -> dtype string
            to force specific dtypes (e.g., {"family_description": "str"}).
    """
    if parquet_path is None:
        base, _ = os.path.splitext(csv_path)
        parquet_path = base + ".parquet"

    logger.info("Converting %s → %s", csv_path, parquet_path)
    logger.info("Chunk size: %s, row group size: %s",
                f"{chunk_size:,}", f"{row_group_size:,}")

    writer = None
    rows_written = 0

    dtype = dtype_overrides or {}
    reader = pd.read_csv(csv_path, chunksize=chunk_size, dtype=dtype)

    for chunk in tqdm(reader, desc="Converting to Parquet", unit="chunk"):
        table = pa.Table.from_pandas(chunk, preserve_index=False)

        if writer is None:
            writer = pq.ParquetWriter(parquet_path, table.schema)

        writer.write_table(table, row_group_size=row_group_size)
        rows_written += len(chunk)

    if writer is not None:
        writer.close()

    file_size_mb = os.path.getsize(parquet_path) / (1024 * 1024)
    logger.info("Done: %s rows written, %.1f MB", f"{rows_written:,}", file_size_mb)
    return parquet_path


def parse_arguments(args):
    parser = argparse.ArgumentParser(
        description="Convert a CSV file to Parquet format for faster queries."
    )
    parser.add_argument(
        "csv_path", type=str,
        help="Path to input CSV file",
    )
    parser.add_argument(
        "-o", "--output", type=str, default=None,
        help="Output Parquet path (default: replaces .csv with .parquet)",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=500_000,
        help="Rows per chunk when reading CSV (default: 500000)",
    )
    parser.add_argument(
        "--row-group-size", type=int, default=500_000,
        help="Rows per row group in Parquet file (default: 500000)",
    )
    return parser.parse_args(args)


def main(args):
    dtype = {}
    # Auto-detect Pfam CSV by column presence and apply known dtype fix
    try:
        cols = pd.read_csv(args.csv_path, nrows=0).columns.tolist()
        if "family_description" in cols:
            dtype["family_description"] = str
    except Exception:
        pass

    csv_to_parquet(
        args.csv_path,
        parquet_path=args.output,
        chunk_size=args.chunk_size,
        row_group_size=args.row_group_size,
        dtype_overrides=dtype or None,
    )

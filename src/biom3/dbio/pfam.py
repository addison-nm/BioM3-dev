"""Pfam CSV/Parquet reader with chunked filtering."""

import os

import pandas as pd
from tqdm import tqdm

from biom3.backend.device import setup_logger
from biom3.dbio.base import DatabaseReader

logger = setup_logger(__name__)

OUTPUT_COLS = [
    "primary_Accession",
    "protein_sequence",
    "[final]text_caption",
    "pfam_label",
]

COLUMN_MAP = {"id": "primary_Accession", "sequence": "protein_sequence"}


def _parquet_path_for(csv_path):
    """Return the corresponding .parquet path for a .csv path."""
    base, ext = os.path.splitext(csv_path)
    if ext.lower() == ".csv":
        return base + ".parquet"
    return None


class PfamReader(DatabaseReader):
    """Reads Pfam protein-text dataset (~44.8M rows).

    Supports two file formats:
    - **Parquet** (preferred): uses pyarrow predicate pushdown for instant
      filtering by pfam_label. Convert with ``biom3_convert_to_parquet``.
    - **CSV** (fallback): reads in chunks with tqdm progress bar.

    If the data_path points to a CSV and a corresponding .parquet file exists
    alongside it, the Parquet file is used automatically.
    """

    name = "pfam"
    DEFAULT_CHUNK_SIZE = 500_000

    def __init__(self, data_path, chunk_size=DEFAULT_CHUNK_SIZE):
        super().__init__(data_path)
        self.chunk_size = chunk_size

    def _resolve_path(self):
        """Return (path, is_parquet). Auto-detects Parquet if available."""
        if self.data_path.endswith(".parquet"):
            return self.data_path, True
        parquet = _parquet_path_for(self.data_path)
        if parquet and os.path.exists(parquet):
            logger.info("Parquet file found, using fast path: %s", parquet)
            return parquet, True
        return self.data_path, False

    def query_by_pfam(self, pfam_ids, keep_family_cols=False, **kwargs):
        """Filter rows by exact match on pfam_label.

        Args:
            pfam_ids: list of Pfam ID strings.
            keep_family_cols: if True, preserve family_name and family_description
                columns (needed for enrichment).
        """
        path, is_parquet = self._resolve_path()
        if is_parquet:
            return self._query_parquet(path, pfam_ids, keep_family_cols)
        return self._query_csv(path, pfam_ids, keep_family_cols)

    def _query_parquet(self, path, pfam_ids, keep_family_cols):
        """Fast Parquet query with predicate pushdown."""
        import pyarrow.parquet as pq

        logger.info("Reading Pfam Parquet: %s", path)
        pfam_id_set = set(pfam_ids)

        # Read with row-group-level filtering
        table = pq.read_table(
            path,
            filters=[("pfam_label", "in", pfam_id_set)],
        )
        df = table.to_pandas()
        df = df.rename(columns=COLUMN_MAP)

        cols = OUTPUT_COLS + (["family_name", "family_description"]
                              if keep_family_cols else [])
        result = df[[c for c in cols if c in df.columns]].copy()
        logger.info("Pfam (Parquet): %s rows matched for %s",
                     f"{len(result):,}", pfam_ids)
        return result

    def _query_csv(self, path, pfam_ids, keep_family_cols):
        """Chunked CSV reading with tqdm progress bar."""
        pfam_id_set = set(pfam_ids)
        chunks = []
        logger.info("Reading Pfam CSV in chunks of %s: %s",
                     f"{self.chunk_size:,}", path)

        reader = pd.read_csv(path, chunksize=self.chunk_size,
                              dtype={"family_description": str})
        rows_matched = 0
        for chunk in tqdm(reader, desc="Scanning Pfam", unit="chunk"):
            match = chunk[chunk["pfam_label"].isin(pfam_id_set)]
            if len(match) > 0:
                chunks.append(match)
                rows_matched += len(match)

        if not chunks:
            cols = OUTPUT_COLS + (["family_name", "family_description"]
                                  if keep_family_cols else [])
            logger.info("Pfam: 0 rows matched for %s", pfam_ids)
            return pd.DataFrame(columns=cols)

        df = pd.concat(chunks, ignore_index=True)
        df = df.rename(columns=COLUMN_MAP)

        cols = OUTPUT_COLS + (["family_name", "family_description"]
                              if keep_family_cols else [])
        result = df[cols].copy()
        logger.info("Pfam: %s rows matched for %s", f"{len(result):,}", pfam_ids)
        return result

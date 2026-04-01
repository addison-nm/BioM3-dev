"""Pfam CSV reader with chunked filtering."""

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


class PfamReader(DatabaseReader):
    """Reads Pfam_protein_text_dataset.csv (~44.8M rows, ~35 GB).

    Uses chunked reading to avoid loading the full file into memory.
    The pfam_label column contains a single Pfam ID per row, so filtering
    uses exact isin() matching (unlike SwissProt's regex approach).
    """

    name = "pfam"
    DEFAULT_CHUNK_SIZE = 500_000

    def __init__(self, data_path, chunk_size=DEFAULT_CHUNK_SIZE):
        super().__init__(data_path)
        self.chunk_size = chunk_size

    def query_by_pfam(self, pfam_ids, keep_family_cols=False, **kwargs):
        """Filter rows by exact match on pfam_label.

        Args:
            pfam_ids: list of Pfam ID strings.
            keep_family_cols: if True, preserve family_name and family_description
                columns (needed for enrichment).
        """
        pfam_id_set = set(pfam_ids)
        chunks = []
        logger.info("Reading Pfam CSV in chunks of %s: %s",
                     f"{self.chunk_size:,}", self.data_path)

        reader = pd.read_csv(self.data_path, chunksize=self.chunk_size,
                              dtype={"family_description": str})
        rows_scanned = 0
        rows_matched = 0
        for chunk in tqdm(reader, desc="Scanning Pfam", unit="chunk"):
            rows_scanned += len(chunk)
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

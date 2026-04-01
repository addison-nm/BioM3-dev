"""SwissProt CSV/Parquet reader with Pfam filtering."""

import os

import pandas as pd

from biom3.backend.device import setup_logger
from biom3.dbio.base import DatabaseReader

logger = setup_logger(__name__)

OUTPUT_COLS = [
    "primary_Accession",
    "protein_sequence",
    "[final]text_caption",
    "pfam_label",
]


def _parquet_path_for(csv_path):
    """Return the corresponding .parquet path for a .csv path."""
    base, ext = os.path.splitext(csv_path)
    if ext.lower() == ".csv":
        return base + ".parquet"
    return None


class SwissProtReader(DatabaseReader):
    """Reads fully_annotated_swiss_prot dataset (~570K rows).

    Supports Parquet (preferred) and CSV formats. If the data_path points
    to a CSV and a corresponding .parquet file exists alongside it, the
    Parquet file is used automatically.
    """

    name = "swissprot"

    def __init__(self, data_path):
        super().__init__(data_path)
        self._df = None

    def _resolve_path(self):
        """Return (path, is_parquet). Auto-detects Parquet if available."""
        if self.data_path.endswith(".parquet"):
            return self.data_path, True
        parquet = _parquet_path_for(self.data_path)
        if parquet and os.path.exists(parquet):
            logger.info("Parquet file found, using fast path: %s", parquet)
            return parquet, True
        return self.data_path, False

    def _load(self):
        if self._df is None:
            path, is_parquet = self._resolve_path()
            if is_parquet:
                logger.info("Loading SwissProt Parquet: %s", path)
                self._df = pd.read_parquet(path)
            else:
                logger.info("Loading SwissProt CSV: %s", path)
                self._df = pd.read_csv(path)
        return self._df

    def query_by_pfam(self, pfam_ids, **kwargs):
        """Filter rows where pfam_label contains any of the given Pfam IDs.

        Uses regex matching because pfam_label stores stringified Python lists.
        """
        df = self._load()
        pattern = "|".join(pfam_ids)
        mask = df["pfam_label"].str.contains(pattern, na=False)
        result = df.loc[mask, OUTPUT_COLS].copy()
        logger.info("SwissProt: %s rows matched for %s", f"{len(result):,}", pfam_ids)
        return result

"""SwissProt CSV reader with Pfam filtering."""

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


class SwissProtReader(DatabaseReader):
    """Reads fully_annotated_swiss_prot.csv (~570K rows).

    Loads the entire file into memory (acceptable at ~1.5 GB).
    The pfam_label column stores stringified Python lists like
    "['PF00018', 'PF07714']", so filtering uses regex substring matching.
    """

    name = "swissprot"

    def __init__(self, data_path):
        super().__init__(data_path)
        self._df = None

    def _load(self):
        if self._df is None:
            logger.info("Loading SwissProt CSV: %s", self.data_path)
            self._df = pd.read_csv(self.data_path)
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

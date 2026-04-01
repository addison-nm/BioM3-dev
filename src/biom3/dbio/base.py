"""Abstract base class for database readers."""

from abc import ABC, abstractmethod


class DatabaseReader(ABC):
    """Base class for local protein database readers."""

    def __init__(self, data_path):
        self.data_path = data_path

    @property
    @abstractmethod
    def name(self):
        """Short identifier for this database (e.g., 'swissprot', 'pfam')."""

    @abstractmethod
    def query_by_pfam(self, pfam_ids, **kwargs):
        """Return a DataFrame of rows matching the given Pfam IDs.

        Args:
            pfam_ids: list of Pfam ID strings (e.g., ["PF00018"])

        Returns:
            pandas DataFrame with standardized output columns.
        """

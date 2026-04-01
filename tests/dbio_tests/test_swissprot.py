"""Tests for SwissProtReader."""

import os
import pytest

from biom3.dbio.swissprot import SwissProtReader, OUTPUT_COLS

DATDIR = os.path.join("tests", "_data", "dbio")
SWISSPROT_PATH = os.path.join(DATDIR, "mini_swissprot.csv")


class TestSwissProtReader:

    def test_query_single_pfam_id(self):
        reader = SwissProtReader(SWISSPROT_PATH)
        result = reader.query_by_pfam(["PF00018"])
        # PF00018 appears in: P12345, P12347, P12351, P12352
        assert len(result) == 4
        assert list(result.columns) == OUTPUT_COLS

    def test_query_multiple_pfam_ids(self):
        reader = SwissProtReader(SWISSPROT_PATH)
        result = reader.query_by_pfam(["PF00018", "PF00042"])
        # PF00018: P12345, P12347, P12351, P12352 (4 rows)
        # PF00042: P12346, P12364 (2 rows)
        assert len(result) == 6

    def test_query_no_match(self):
        reader = SwissProtReader(SWISSPROT_PATH)
        result = reader.query_by_pfam(["PF99999"])
        assert len(result) == 0
        assert list(result.columns) == OUTPUT_COLS

    def test_regex_matches_within_list(self):
        """PF07714 appears as second element in P12345's list and alone in P12359."""
        reader = SwissProtReader(SWISSPROT_PATH)
        result = reader.query_by_pfam(["PF07714"])
        accessions = set(result["primary_Accession"])
        assert "P12345" in accessions
        assert "P12359" in accessions

    def test_lazy_loading(self):
        reader = SwissProtReader(SWISSPROT_PATH)
        assert reader._df is None
        reader.query_by_pfam(["PF00018"])
        assert reader._df is not None
        # Second call reuses cached data
        reader.query_by_pfam(["PF00042"])
        assert reader._df is not None

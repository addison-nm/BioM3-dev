"""Tests for PfamReader."""

import os
import pytest

from biom3.dbio.pfam import PfamReader, OUTPUT_COLS

DATDIR = os.path.join("tests", "_data", "dbio")
PFAM_PATH = os.path.join(DATDIR, "mini_pfam.csv")


class TestPfamReader:

    def test_query_single_pfam_id(self):
        reader = PfamReader(PFAM_PATH)
        result = reader.query_by_pfam(["PF00018"])
        assert len(result) == 5
        assert list(result.columns) == OUTPUT_COLS

    def test_query_multiple_pfam_ids(self):
        reader = PfamReader(PFAM_PATH)
        result = reader.query_by_pfam(["PF00018", "PF07714"])
        assert len(result) == 8  # 5 SH3 + 3 Pkinase_Tyr

    def test_query_no_match(self):
        reader = PfamReader(PFAM_PATH)
        result = reader.query_by_pfam(["PF99999"])
        assert len(result) == 0
        assert list(result.columns) == OUTPUT_COLS

    def test_column_rename(self):
        """Verify id->primary_Accession and sequence->protein_sequence."""
        reader = PfamReader(PFAM_PATH)
        result = reader.query_by_pfam(["PF00042"])
        assert "primary_Accession" in result.columns
        assert "protein_sequence" in result.columns
        assert "id" not in result.columns
        assert "sequence" not in result.columns

    def test_keep_family_cols(self):
        reader = PfamReader(PFAM_PATH)
        result = reader.query_by_pfam(["PF00018"], keep_family_cols=True)
        assert "family_name" in result.columns
        assert "family_description" in result.columns

    def test_exact_match_not_substring(self):
        """PF0007 should not match PF00071."""
        reader = PfamReader(PFAM_PATH)
        result = reader.query_by_pfam(["PF0007"])
        assert len(result) == 0

    def test_chunked_consistency(self):
        """Results should be identical regardless of chunk size."""
        reader_small = PfamReader(PFAM_PATH, chunk_size=3)
        reader_large = PfamReader(PFAM_PATH, chunk_size=100)
        result_small = reader_small.query_by_pfam(["PF00018"])
        result_large = reader_large.query_by_pfam(["PF00018"])
        assert len(result_small) == len(result_large)
        assert set(result_small["primary_Accession"]) == set(result_large["primary_Accession"])

    def test_parquet_roundtrip(self, tmp_path):
        """Parquet file should produce identical results to CSV."""
        from biom3.dbio.convert import csv_to_parquet

        parquet_path = str(tmp_path / "mini_pfam.parquet")
        csv_to_parquet(PFAM_PATH, parquet_path, chunk_size=10)

        csv_reader = PfamReader(PFAM_PATH)
        parquet_reader = PfamReader(parquet_path)

        csv_result = csv_reader.query_by_pfam(["PF00018"])
        parquet_result = parquet_reader.query_by_pfam(["PF00018"])

        assert len(csv_result) == len(parquet_result)
        assert set(csv_result["primary_Accession"]) == set(parquet_result["primary_Accession"])

    def test_parquet_auto_detect(self, tmp_path):
        """PfamReader should auto-detect .parquet alongside .csv."""
        import shutil
        from biom3.dbio.convert import csv_to_parquet

        # Copy CSV to tmp and create Parquet alongside it
        csv_copy = str(tmp_path / "test.csv")
        shutil.copy(PFAM_PATH, csv_copy)
        csv_to_parquet(csv_copy, chunk_size=10)

        reader = PfamReader(csv_copy)
        path, is_parquet = reader._resolve_path()
        assert is_parquet

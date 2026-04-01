"""Tests for TaxonomyTree and AccessionTaxidMapper."""

import os
import pytest

from biom3.dbio.taxonomy import TaxonomyTree, AccessionTaxidMapper

DATDIR = os.path.join("tests", "_data", "dbio")
TAXONOMY_DIR = DATDIR  # mini .dmp files live here
ACCESSION2TAXID_PATH = os.path.join(DATDIR, "mini_accession2taxid.tsv")


class TestTaxonomyTree:

    @pytest.fixture
    def tree(self):
        t = TaxonomyTree(TAXONOMY_DIR)
        t.load()
        return t

    def test_load_node_count(self, tree):
        assert len(tree._lineages) == 10

    def test_get_lineage_human(self, tree):
        lineage = tree.get_lineage(9606)
        assert lineage["superkingdom"] == "Eukaryota"
        assert lineage["phylum"] == "Chordata"
        assert lineage["species"] == "Homo sapiens"
        assert lineage["genus"] == "Homo"

    def test_get_lineage_bacteria(self, tree):
        lineage = tree.get_lineage(562)
        assert lineage["superkingdom"] == "Bacteria"
        assert lineage["genus"] == "Escherichia"

    def test_get_lineage_missing(self, tree):
        lineage = tree.get_lineage(99999999)
        assert lineage == {}

    def test_get_lineage_string(self, tree):
        s = tree.get_lineage_string(9606)
        assert s.startswith("The organism lineage is")
        assert "Eukaryota" in s
        assert "Homo sapiens" in s

    def test_get_lineage_string_missing(self, tree):
        assert tree.get_lineage_string(99999999) is None

    def test_filter_by_rank_include(self, tree):
        all_ids = {9606, 10090, 562, 287, 4932, 2697049, 3702}
        bacteria = tree.filter_by_rank(all_ids, "superkingdom", include={"Bacteria"})
        assert bacteria == {562, 287}

    def test_filter_by_rank_exclude(self, tree):
        all_ids = {9606, 10090, 562, 287, 4932, 2697049, 3702}
        no_viruses = tree.filter_by_rank(all_ids, "superkingdom", exclude={"Viruses"})
        assert 2697049 not in no_viruses
        assert 9606 in no_viruses

    def test_idempotent_load(self, tree):
        count_before = len(tree._lineages)
        tree.load()  # should be no-op
        assert len(tree._lineages) == count_before


class TestAccessionTaxidMapper:

    def test_streaming_lookup(self):
        mapper = AccessionTaxidMapper(ACCESSION2TAXID_PATH)
        results = mapper._lookup_streaming(["A0A001", "A0A004", "P12345"], chunk_size=5)
        assert results["A0A001"] == 9606
        assert results["A0A004"] == 562
        assert results["P12345"] == 9606

    def test_streaming_missing(self):
        mapper = AccessionTaxidMapper(ACCESSION2TAXID_PATH)
        results = mapper._lookup_streaming(["NONEXISTENT"], chunk_size=5)
        assert len(results) == 0

    def test_sqlite_roundtrip(self, tmp_path):
        db_path = str(tmp_path / "test.sqlite")
        mapper = AccessionTaxidMapper(ACCESSION2TAXID_PATH)

        # Build index
        mapper.build_sqlite_index(db_path)
        assert os.path.exists(db_path)

        # Lookup via SQLite
        results = mapper.lookup_sqlite(["A0A001", "A0A008", "P12350"], db_path)
        assert results["A0A001"] == 9606
        assert results["A0A008"] == 562
        assert results["P12350"] == 9606

    def test_sqlite_missing(self, tmp_path):
        db_path = str(tmp_path / "test.sqlite")
        mapper = AccessionTaxidMapper(ACCESSION2TAXID_PATH)
        mapper.build_sqlite_index(db_path)

        results = mapper.lookup_sqlite(["NONEXISTENT"], db_path)
        assert len(results) == 0

    def test_auto_detect_fallback(self):
        """Without SQLite index, lookup() falls back to streaming."""
        mapper = AccessionTaxidMapper(ACCESSION2TAXID_PATH)
        results = mapper.lookup(["A0A001"])
        assert results["A0A001"] == 9606

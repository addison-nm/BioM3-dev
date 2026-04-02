import os

import pyarrow.parquet as pq
import pytest

from biom3.dbio.build_annotation_cache import (
    CACHE_ANNOTATION_COLUMNS,
    CACHE_SCHEMA,
    build_annotation_cache,
    load_annotation_cache,
)
from biom3.dbio.swissprot_dat import SwissProtDatParser

DATDIR = os.path.join(os.path.dirname(__file__), "..", "_data", "dbio")
TMPDIR = os.path.join(os.path.dirname(__file__), "..", "_tmp")


@pytest.fixture(autouse=True)
def ensure_tmpdir():
    os.makedirs(TMPDIR, exist_ok=True)


@pytest.fixture
def dat_path():
    return os.path.join(DATDIR, "mini_swissprot.dat")


@pytest.fixture
def cache_path():
    return os.path.join(TMPDIR, "test_annotation_cache.parquet")


class TestBuildAnnotationCache:

    def test_basic_build(self, dat_path, cache_path):
        build_annotation_cache(dat_path, cache_path)
        assert os.path.exists(cache_path)
        table = pq.read_table(cache_path)
        assert len(table) > 0

    def test_schema(self, dat_path, cache_path):
        build_annotation_cache(dat_path, cache_path)
        table = pq.read_table(cache_path)
        col_names = table.column_names
        assert "primary_Accession" in col_names
        for col in CACHE_ANNOTATION_COLUMNS:
            assert col in col_names

    def test_accessions_present(self, dat_path, cache_path):
        build_annotation_cache(dat_path, cache_path)
        table = pq.read_table(cache_path)
        accessions = set(table.column("primary_Accession").to_pylist())
        # Q6GZX4 has FUNCTION annotation, should be included
        assert "Q6GZX4" in accessions

    def test_require_annotation_skips_empty(self, dat_path, cache_path):
        build_annotation_cache(dat_path, cache_path, require_annotation=True)
        table_strict = pq.read_table(cache_path)

        cache_path_all = cache_path.replace(".parquet", "_all.parquet")
        build_annotation_cache(dat_path, cache_path_all, require_annotation=False)
        table_all = pq.read_table(cache_path_all)

        assert len(table_all) >= len(table_strict)

    def test_small_chunk_size(self, dat_path, cache_path):
        build_annotation_cache(dat_path, cache_path, chunk_size=1)
        table = pq.read_table(cache_path)
        assert len(table) > 0


class TestLoadAnnotationCache:

    def test_load_specific_accessions(self, dat_path, cache_path):
        build_annotation_cache(dat_path, cache_path)
        result = load_annotation_cache([cache_path], {"Q6GZX4"})
        assert "Q6GZX4" in result
        assert isinstance(result["Q6GZX4"], dict)

    def test_load_missing_accessions(self, dat_path, cache_path):
        build_annotation_cache(dat_path, cache_path)
        result = load_annotation_cache([cache_path], {"NONEXISTENT"})
        assert "NONEXISTENT" not in result

    def test_load_partial_accessions(self, dat_path, cache_path):
        build_annotation_cache(dat_path, cache_path)
        result = load_annotation_cache(
            [cache_path], {"Q6GZX4", "NONEXISTENT"},
        )
        assert "Q6GZX4" in result
        assert "NONEXISTENT" not in result

    def test_sparse_dict_format(self, dat_path, cache_path):
        build_annotation_cache(dat_path, cache_path)
        result = load_annotation_cache([cache_path], {"Q6GZX4"})
        annots = result["Q6GZX4"]
        # All values should be non-empty strings (sparse format)
        for val in annots.values():
            assert isinstance(val, str)
            assert val.strip()
        # No None or empty values
        assert None not in annots.values()

    def test_roundtrip_matches_parser(self, dat_path, cache_path):
        """Cache output should match SwissProtDatParser.parse() output."""
        build_annotation_cache(dat_path, cache_path)
        parser = SwissProtDatParser(dat_path)

        # Get all accessions from the cache
        table = pq.read_table(cache_path, columns=["primary_Accession"])
        all_accs = set(table.column("primary_Accession").to_pylist())

        # Load from cache
        cached = load_annotation_cache([cache_path], all_accs)

        # Load from parser
        parsed = parser.parse(all_accs)

        # Compare: every accession in parsed should be in cached
        for acc in parsed:
            assert acc in cached, f"{acc} in parser output but not in cache"
            # The annotation keys should match
            for key, val in parsed[acc].items():
                if key in CACHE_ANNOTATION_COLUMNS:
                    assert key in cached[acc], (
                        f"{acc}: key {key} in parser but not in cache"
                    )
                    assert cached[acc][key] == val, (
                        f"{acc}: {key} differs: parser={val!r}, cache={cached[acc][key]!r}"
                    )

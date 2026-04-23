"""Tests for biom3.dbio.stats — coverage computation + markdown formatter."""

import pandas as pd
import pytest

from biom3.dbio.stats import (
    IncrementalStatsBuilder,
    _is_populated,
    _parse_pfam_label,
    compute_coverage_stats,
    format_stats_markdown,
)


class TestIsPopulated:

    @pytest.mark.parametrize("value", [
        None, "", "nan", "NaN", "None", "null", "[]", "['nan']", "<NA>", "<na>",
        pd.NA, float("nan"),
    ])
    def test_empty_sentinels(self, value):
        assert _is_populated(value) is False

    @pytest.mark.parametrize("value", [
        "real value", "x", "['PF00018']", 0, 42, "0",
    ])
    def test_populated_values(self, value):
        assert _is_populated(value) is True


class TestParsePfamLabel:

    def test_single_id_list_string(self):
        assert _parse_pfam_label("['PF00018']") == ["PF00018"]

    def test_multi_id_list_string(self):
        assert _parse_pfam_label("['PF00018', 'PF00001']") == ["PF00018", "PF00001"]

    def test_nan_sentinel(self):
        assert _parse_pfam_label("['nan']") == []

    def test_bare_id(self):
        assert _parse_pfam_label("PF00018") == ["PF00018"]

    def test_list_input(self):
        assert _parse_pfam_label(["PF00018", "PF00001"]) == ["PF00018", "PF00001"]

    def test_na_and_none(self):
        assert _parse_pfam_label(None) == []
        assert _parse_pfam_label("") == []
        assert _parse_pfam_label("nan") == []


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "primary_Accession": ["A", "B", "C", "D"],
        "protein_sequence": ["MK", "MPRQ", "M", "MAAA"],
        "pfam_label": [
            "['PF00018']",
            "['PF00018', 'PF00001']",
            "['nan']",
            "PF00001",
        ],
        "annot_function": ["real", pd.NA, "", "another"],
        "annot_cofactor": ["", "", "", "Zn2+"],
    })


class TestComputeCoverageStats:

    def test_row_count(self, sample_df):
        stats = compute_coverage_stats(sample_df)
        assert stats["row_count"] == 4

    def test_sequence_length_stats(self, sample_df):
        seq = compute_coverage_stats(sample_df)["sequence_length"]
        assert seq["min"] == 1
        assert seq["max"] == 4
        assert seq["median"] == 3

    def test_annotation_coverage_ignores_empty(self, sample_df):
        cov = compute_coverage_stats(sample_df)["annotation_coverage"]
        assert cov["annot_function"]["populated"] == 2
        assert cov["annot_function"]["percentage"] == 50.0
        assert cov["annot_cofactor"]["populated"] == 1
        assert cov["annot_cofactor"]["percentage"] == 25.0

    def test_pfam_stats_drops_nan_sentinel(self, sample_df):
        pfam = compute_coverage_stats(sample_df)["pfam_label"]
        assert pfam["rows_with_pfam"] == 3  # A, B, D
        assert pfam["distinct_families"] == 2
        top_ids = {entry["pfam_id"] for entry in pfam["top"]}
        assert top_ids == {"PF00018", "PF00001"}

    def test_source_breakdown(self, sample_df):
        df = sample_df.assign(_source=["sp", "sp", "pfam", "pfam"])
        stats = compute_coverage_stats(df, source_col="_source")
        assert stats["sources"] == {"sp": 2, "pfam": 2}

    def test_join_metadata_passthrough(self, sample_df):
        stats = compute_coverage_stats(
            sample_df, join_metadata={"expasy_hit_rate": 0.25},
        )
        assert stats["joins"] == {"expasy_hit_rate": 0.25}

    def test_missing_seq_column_skipped(self):
        df = pd.DataFrame({
            "primary_Accession": ["A"],
            "annot_function": ["x"],
        })
        stats = compute_coverage_stats(df)
        assert stats["sequence_length"] is None


class TestFormatStatsMarkdown:

    def test_renders_all_sections(self, sample_df):
        stats = compute_coverage_stats(
            sample_df.assign(_source=["sp", "sp", "pfam", "pfam"]),
            source_col="_source",
            join_metadata={"expasy_hit_rate": 0.25},
        )
        md = format_stats_markdown(stats, "test")
        assert "# test" in md
        assert "**Rows:** 4" in md
        assert "## Sequence length" in md
        assert "## Rows by source" in md
        assert "## Annotation coverage" in md
        assert "## Pfam family distribution" in md
        assert "## Enrichment join hit rates" in md
        assert "| `annot_function` |" in md

    def test_renders_generic_extras(self):
        stats = compute_coverage_stats(pd.DataFrame({
            "primary_Accession": ["A"],
            "protein_sequence": ["M"],
        }))
        stats["my_custom_section"] = {"foo": 42, "bar": 3.14}
        md = format_stats_markdown(stats, "custom")
        assert "## My Custom Section" in md
        assert "| foo | 42 |" in md
        assert "| bar | 3.14 |" in md


class TestIncrementalStatsBuilder:

    def test_matches_compute_coverage_for_same_rows(self):
        annotation_fields = ["annot_function", "annot_cofactor"]
        builder = IncrementalStatsBuilder(
            annotation_fields=annotation_fields,
            seq_field="sequence",
            pfam_field="pfam_ids",
            caption_field="caption",
        )
        rows = [
            {"sequence": "MK", "pfam_ids": ["PF00018"], "caption": "c1",
             "annot_function": "real", "annot_cofactor": ""},
            {"sequence": "MPRQ", "pfam_ids": ["PF00018", "PF00001"], "caption": "c2",
             "annot_function": "x", "annot_cofactor": "Zn2+"},
            {"sequence": "M", "pfam_ids": [], "caption": "",
             "annot_function": "", "annot_cofactor": ""},
            {"sequence": "MAAA", "pfam_ids": ["PF00001"], "caption": "c4",
             "annot_function": "", "annot_cofactor": "Mg2+"},
        ]
        for row in rows:
            builder.update(row)
        stats = builder.finalize()

        assert stats["row_count"] == 4
        assert stats["annotation_coverage"]["annot_function"]["populated"] == 2
        assert stats["annotation_coverage"]["annot_cofactor"]["populated"] == 2
        assert stats["caption_coverage"]["populated"] == 3
        assert stats["pfam_label"]["distinct_families"] == 2
        assert stats["pfam_label"]["rows_with_pfam"] == 3

    def test_empty_builder_finalizes_cleanly(self):
        builder = IncrementalStatsBuilder(annotation_fields=["annot_x"])
        stats = builder.finalize()
        assert stats["row_count"] == 0
        assert stats["sequence_length"] is None
        assert stats["annotation_coverage"]["annot_x"]["populated"] == 0
        assert stats["pfam_label"]["distinct_families"] == 0

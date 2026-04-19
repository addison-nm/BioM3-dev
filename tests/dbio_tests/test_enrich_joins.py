"""Tests for the source-CSV join layer in enrich.py.

Covers extract_ec_numbers + _join_expasy / _join_brenda / _join_smart
through enrich_dataframe's public API (with in-memory lookups built by
hand so the tests don't depend on a real ExPASy/BRENDA/SMART CSV).
"""

import pandas as pd
import pytest

from biom3.dbio.enrich import (
    enrich_dataframe,
    extract_ec_numbers,
)


# ---------------------------------------------------------------------------
# extract_ec_numbers
# ---------------------------------------------------------------------------

class TestExtractECNumbers:

    def test_extracts_ec_equals(self):
        assert extract_ec_numbers("Reaction=ATP... EC=3.6.1.3") == ["3.6.1.3"]

    def test_extracts_ec_space(self):
        assert extract_ec_numbers("citing EC 3.6.1.3 in prose") == ["3.6.1.3"]

    def test_extracts_bare_number_in_reaction(self):
        # Bare EC numbers embedded in reaction strings still match.
        assert extract_ec_numbers("1.14.11.1") == ["1.14.11.1"]

    def test_handles_partial_ec(self):
        assert "1.2.-.-" in extract_ec_numbers("partial 1.2.-.-")

    def test_deduplicates_repeated_ecs(self):
        text = "EC=3.6.1.3 ... also EC 3.6.1.3 elsewhere"
        assert extract_ec_numbers(text) == ["3.6.1.3"]

    def test_preserves_insertion_order(self):
        text = "first EC=3.6.1.3 then EC=1.1.1.1"
        assert extract_ec_numbers(text) == ["3.6.1.3", "1.1.1.1"]

    def test_none_and_na_return_empty(self):
        assert extract_ec_numbers(None) == []
        assert extract_ec_numbers(pd.NA) == []
        assert extract_ec_numbers(float("nan")) == []

    def test_empty_string_returns_empty(self):
        assert extract_ec_numbers("") == []

    def test_no_ec_returns_empty(self):
        assert extract_ec_numbers("no enzyme here, just a plain sentence") == []


# ---------------------------------------------------------------------------
# _join_expasy (exercised via enrich_dataframe)
# ---------------------------------------------------------------------------

@pytest.fixture
def expasy_lookup():
    return {
        "3.6.1.3": {
            "name": "adenosinetriphosphatase",
            "alternative_names": "ATPase",
            "catalytic_activity": "ATP + H2O = ADP + phosphate",
            "comments": "Broadly distributed hydrolase.",
            "uniprot_accessions": "P12345,P67890",
            "transferred_to": "",
            "deleted": False,
        },
        "1.1.1.1": {
            "name": "alcohol dehydrogenase",
            "alternative_names": "aldehyde reductase",
            "catalytic_activity": "alcohol + NAD+ = aldehyde + NADH",
            "comments": "Acts on primary alcohols.",
            "uniprot_accessions": "P07327",
            "transferred_to": "",
            "deleted": False,
        },
    }


def test_expasy_join_populates_ec_columns(expasy_lookup):
    df = pd.DataFrame({
        "primary_Accession": ["A", "B", "C"],
        "protein_sequence": ["MK", "MPR", "MG"],
        "pfam_label": ["['PF1']", "['PF2']", "['PF3']"],
        "annot_catalytic_activity": [
            "Reaction=ATP + H2O = ADP. EC=3.6.1.3",
            "Reaction=alcohol oxidation. EC=1.1.1.1",
            "no EC here",
        ],
    })
    result, stats = enrich_dataframe(df, expasy_lookup=expasy_lookup)
    assert result.loc[0, "annot_ec_numbers"] == "3.6.1.3"
    assert "adenosinetriphosphatase" in result.loc[0, "annot_ec_names"]
    assert "Broadly distributed" in result.loc[0, "annot_ec_description"]
    assert result.loc[1, "annot_ec_numbers"] == "1.1.1.1"
    assert pd.isna(result.loc[2, "annot_ec_numbers"])


def test_expasy_join_hit_rate_stats(expasy_lookup):
    df = pd.DataFrame({
        "primary_Accession": ["A", "B", "C", "D"],
        "protein_sequence": ["M", "M", "M", "M"],
        "pfam_label": ["['PF1']"] * 4,
        "annot_catalytic_activity": [
            "EC=3.6.1.3",          # hits
            "EC=1.1.1.1",          # hits
            "EC=9.9.9.9",          # EC extracted, no ExPASy match
            "no enzyme",           # no EC at all
        ],
    })
    _, stats = enrich_dataframe(df, expasy_lookup=expasy_lookup)
    assert stats["ec_extraction_rate"] == 0.75  # 3/4 rows had an EC
    assert stats["expasy_hit_rate"] == 0.5      # 2/4 rows had an ExPASy match


# ---------------------------------------------------------------------------
# _join_smart (exercised via enrich_dataframe)
# ---------------------------------------------------------------------------

@pytest.fixture
def smart_lookup():
    return {
        "SM00101": {
            "name": "14_3_3",
            "definition": "14-3-3 homologues",
            "description": "Signal transduction domain.",
        },
        "SM00474": {
            "name": "35EXOc",
            "definition": "3'-5' exonuclease",
            "description": "Proofreading domain.",
        },
    }


def test_smart_join_populates_domain_column(smart_lookup):
    df = pd.DataFrame({
        "primary_Accession": ["A", "B", "C"],
        "protein_sequence": ["M", "M", "M"],
        "pfam_label": ["['PF1']"] * 3,
        "xref_smart_ids": [["SM00101"], ["SM00101", "SM00474"], []],
    })
    result, stats = enrich_dataframe(df, smart_lookup=smart_lookup)
    assert "SM00101: 14-3-3 homologues" in result.loc[0, "annot_smart_domains"]
    # Multi-domain row combines both.
    assert "SM00101" in result.loc[1, "annot_smart_domains"]
    assert "SM00474" in result.loc[1, "annot_smart_domains"]
    # No xrefs -> NA.
    assert pd.isna(result.loc[2, "annot_smart_domains"])
    assert stats["smart_hit_rate"] == pytest.approx(2 / 3)


def test_smart_join_no_xref_column_gives_zero_hit(smart_lookup):
    df = pd.DataFrame({
        "primary_Accession": ["A"],
        "protein_sequence": ["M"],
        "pfam_label": ["['PF1']"],
    })
    result, stats = enrich_dataframe(df, smart_lookup=smart_lookup)
    assert pd.isna(result.loc[0, "annot_smart_domains"])
    assert stats["smart_hit_rate"] == 0.0


def test_smart_join_missing_id_skipped(smart_lookup):
    df = pd.DataFrame({
        "primary_Accession": ["A"],
        "protein_sequence": ["M"],
        "pfam_label": ["['PF1']"],
        "xref_smart_ids": [["SM99999"]],  # not in lookup
    })
    result, stats = enrich_dataframe(df, smart_lookup=smart_lookup)
    assert pd.isna(result.loc[0, "annot_smart_domains"])
    assert stats["smart_hit_rate"] == 0.0


# ---------------------------------------------------------------------------
# _join_brenda (exercised via enrich_dataframe)
# ---------------------------------------------------------------------------

@pytest.fixture
def brenda_lookup():
    return {
        "by_ec_org": {
            ("1.1.1.1", "homo sapiens"): {
                "substrates_products": "ethanol + NAD+ = acetaldehyde",
                "km_values": "0.05 {ethanol}",
                "ph_optimum": "7.4",
                "temperature_optimum": "25",
            },
            ("1.1.1.1", "saccharomyces cerevisiae"): {
                "substrates_products": "butanol + NAD+ = butyraldehyde",
                "km_values": "0.3 {ethanol}",
                "ph_optimum": "6.5",
                "temperature_optimum": "",
            },
        },
        "shared_by_ec": {
            "1.1.1.1": {
                "recommended_name": "alcohol dehydrogenase",
                "systematic_name": "alcohol:NAD+ oxidoreductase",
                "synonyms": "aldehyde reductase",
                "reactions": "primary alcohol + NAD+ = aldehyde + NADH",
            },
        },
    }


def test_brenda_join_strict_species_match(brenda_lookup):
    df = pd.DataFrame({
        "primary_Accession": ["A"],
        "protein_sequence": ["M"],
        "pfam_label": ["['PF1']"],
        "annot_catalytic_activity": ["EC=1.1.1.1"],
        "annot_lineage": [
            "The organism lineage is Eukaryota, Metazoa, Chordata, Homo sapiens"
        ],
    })
    result, stats = enrich_dataframe(df, brenda_lookup=brenda_lookup)
    assert "ethanol" in result.loc[0, "annot_brenda_substrates"]
    assert "0.05" in result.loc[0, "annot_brenda_km_values"]
    assert result.loc[0, "annot_brenda_ph_optimum"] == "7.4"
    assert stats["brenda_strict_hits"] == 1
    assert stats["brenda_hit_rate"] == 1.0


def test_brenda_join_strict_miss_on_wrong_organism(brenda_lookup):
    df = pd.DataFrame({
        "primary_Accession": ["A"],
        "protein_sequence": ["M"],
        "pfam_label": ["['PF1']"],
        "annot_catalytic_activity": ["EC=1.1.1.1"],
        "annot_lineage": [
            "The organism lineage is Bacteria, Pseudomonadota, Escherichia coli"
        ],
    })
    result, stats = enrich_dataframe(df, brenda_lookup=brenda_lookup)
    assert pd.isna(result.loc[0, "annot_brenda_substrates"])
    assert stats["brenda_strict_hits"] == 0
    assert stats["brenda_hit_rate"] == 0.0


def test_brenda_join_handles_no_ec(brenda_lookup):
    df = pd.DataFrame({
        "primary_Accession": ["A"],
        "protein_sequence": ["M"],
        "pfam_label": ["['PF1']"],
        "annot_catalytic_activity": ["no EC here"],
        "annot_lineage": ["The organism lineage is Eukaryota, Homo sapiens"],
    })
    result, stats = enrich_dataframe(df, brenda_lookup=brenda_lookup)
    assert pd.isna(result.loc[0, "annot_brenda_substrates"])
    assert stats["brenda_hit_rate"] == 0.0


# ---------------------------------------------------------------------------
# enrich_dataframe: no lookups -> empty join_stats
# ---------------------------------------------------------------------------

def test_no_lookups_returns_empty_join_stats():
    df = pd.DataFrame({
        "primary_Accession": ["A"],
        "protein_sequence": ["M"],
        "pfam_label": ["['PF1']"],
    })
    result, stats = enrich_dataframe(df)
    assert stats == {}
    # Canonical annotation schema is initialized regardless of joins so
    # downstream code can trust the column set; all values are NA when
    # no enrichment data was supplied.
    assert "annot_ec_numbers" in result.columns
    assert "annot_ec_names" in result.columns
    assert pd.isna(result.loc[0, "annot_ec_numbers"])
    assert pd.isna(result.loc[0, "annot_ec_names"])

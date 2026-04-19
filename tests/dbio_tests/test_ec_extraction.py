"""Tests for EC-number extraction from UniProt .dat files.

Covers two sources of truth that both feed into annot_ec_numbers:
  - CC CATALYTIC ACTIVITY blocks (via _map_cc_catalytic_activity)
  - DE lines (standalone `EC=N.N.N.N` lines)

and the precedence order the enrichment layer uses when reading them back.
"""

import textwrap

import pandas as pd
import pytest

from biom3.dbio.enrich import (
    _extract_row_ec_numbers,
    enrich_dataframe,
    load_expasy_lookup,
)
from biom3.dbio.swissprot_dat import (
    _extract_ec_numbers_from_lines,
    _parse_entry,
)


# ---------------------------------------------------------------------------
# Low-level: _extract_ec_numbers_from_lines
# ---------------------------------------------------------------------------

class TestExtractECFromLines:

    def test_single_ec_in_de(self):
        lines = [
            "RecName: Full=Alcohol dehydrogenase 1;",
            "         EC=1.1.1.1;",
        ]
        assert _extract_ec_numbers_from_lines(lines) == ["1.1.1.1"]

    def test_multiple_ecs_deduplicated(self):
        lines = [
            "RecName: Full=Bifunctional enzyme;",
            "         EC=1.1.1.1;",
            "         EC=2.7.11.1;",
            "AltName: Short=BF {EC=1.1.1.1};",  # duplicate
        ]
        assert _extract_ec_numbers_from_lines(lines) == ["1.1.1.1", "2.7.11.1"]

    def test_no_ec_returns_empty(self):
        lines = [
            "RecName: Full=Non-enzyme protein;",
            "         Short=NE;",
        ]
        assert _extract_ec_numbers_from_lines(lines) == []


# ---------------------------------------------------------------------------
# End-to-end: .dat entry -> annotations with annot_ec_numbers
# ---------------------------------------------------------------------------

def _entry_lines(text):
    return [line + "\n" for line in text.splitlines()]


class TestDatEntryECExtraction:

    def test_de_line_standalone_ec(self):
        # Entry with only a standalone DE EC line (no CATALYTIC ACTIVITY).
        entry = textwrap.dedent("""\
            ID   TEST_ENTRY              Reviewed;  10 AA.
            AC   P12345;
            DE   RecName: Full=Some enzyme;
            DE            EC=1.2.3.4;
            OS   Escherichia coli.
            OC   Bacteria; Pseudomonadota.
            //
        """)
        ann = _parse_entry(_entry_lines(entry))
        assert ann["annot_ec_numbers"] == "1.2.3.4"

    def test_cc_catalytic_activity_extracts_ec(self):
        entry = textwrap.dedent("""\
            ID   TEST_ENTRY              Reviewed;  10 AA.
            AC   P12345;
            DE   RecName: Full=Some hydrolase;
            OS   Escherichia coli.
            OC   Bacteria.
            CC   -!- CATALYTIC ACTIVITY:
            CC       Reaction=A + H2O = B + C; Xref=Rhea:RHEA:00001; EC=3.1.1.1;
            CC         Evidence={ECO:0000269|PubMed:12345};
            //
        """)
        ann = _parse_entry(_entry_lines(entry))
        assert ann["annot_ec_numbers"] == "3.1.1.1"
        # Prose reaction stays in annot_catalytic_activity, EC xref stripped out.
        assert "EC=" not in ann.get("annot_catalytic_activity", "")
        assert "A + H2O = B + C" in ann["annot_catalytic_activity"]

    def test_merges_de_and_cc_ecs(self):
        # Entry with EC in both DE and CC — merged + deduplicated.
        entry = textwrap.dedent("""\
            ID   TEST_ENTRY              Reviewed;  10 AA.
            AC   P12345;
            DE   RecName: Full=Bifunctional;
            DE            EC=2.7.11.1;
            OS   Homo sapiens.
            OC   Eukaryota.
            CC   -!- CATALYTIC ACTIVITY:
            CC       Reaction=ATP + X = ADP + X-phosphate; EC=2.7.11.1;
            CC   -!- CATALYTIC ACTIVITY:
            CC       Reaction=X-phosphate + H2O = X + phosphate; EC=3.1.3.16;
            //
        """)
        ann = _parse_entry(_entry_lines(entry))
        ecs = ann["annot_ec_numbers"].split(", ")
        assert "2.7.11.1" in ecs
        assert "3.1.3.16" in ecs
        assert len(ecs) == len(set(ecs))  # no duplicates

    def test_no_ec_for_non_enzymes(self):
        entry = textwrap.dedent("""\
            ID   TEST_ENTRY              Reviewed;  10 AA.
            AC   P12345;
            DE   RecName: Full=Structural protein;
            OS   Homo sapiens.
            OC   Eukaryota.
            CC   -!- FUNCTION: Binds things.
            //
        """)
        ann = _parse_entry(_entry_lines(entry))
        assert "annot_ec_numbers" not in ann


# ---------------------------------------------------------------------------
# Enrichment: _extract_row_ec_numbers precedence
# ---------------------------------------------------------------------------

class TestExtractRowECPrecedence:

    def test_uses_annot_ec_numbers_column_first(self):
        row = pd.Series({
            "annot_ec_numbers": "3.1.1.74",
            "annot_catalytic_activity": "cutin + H2O = cutin monomers",  # no EC
            "[final]text_caption": "PROTEIN NAME: Cutinase",
        })
        assert _extract_row_ec_numbers(row) == ["3.1.1.74"]

    def test_falls_back_to_catalytic_activity_when_column_empty(self):
        row = pd.Series({
            "annot_ec_numbers": pd.NA,
            "annot_catalytic_activity": "Reaction=ATP + H2O = ADP. EC=3.6.1.3",
            "[final]text_caption": "something",
        })
        assert _extract_row_ec_numbers(row) == ["3.6.1.3"]

    def test_falls_back_to_caption_when_both_empty(self):
        row = pd.Series({
            "annot_ec_numbers": pd.NA,
            "annot_catalytic_activity": pd.NA,
            "[final]text_caption": "Something with EC=1.1.1.1 in it",
        })
        assert _extract_row_ec_numbers(row) == ["1.1.1.1"]

    def test_multiple_ec_numbers_from_column(self):
        row = pd.Series({
            "annot_ec_numbers": "2.7.11.1, 3.1.3.16",
            "annot_catalytic_activity": pd.NA,
            "[final]text_caption": pd.NA,
        })
        assert _extract_row_ec_numbers(row) == ["2.7.11.1", "3.1.3.16"]


# ---------------------------------------------------------------------------
# _join_expasy honors pre-populated annot_ec_numbers
# ---------------------------------------------------------------------------

def test_join_expasy_hits_with_source_populated_ec():
    # Row has annot_ec_numbers from the source CSV but no EC in catalytic
    # text or caption — join should still hit.
    df = pd.DataFrame({
        "primary_Accession": ["A"],
        "protein_sequence": ["M"],
        "pfam_label": ["['PF1']"],
        "annot_catalytic_activity": ["cutin + H2O = cutin monomers"],  # no EC
        "[final]text_caption": ["PROTEIN NAME: Cutinase"],
        "annot_ec_numbers": ["3.1.1.74"],
    })
    expasy_lookup = {
        "3.1.1.74": {
            "name": "cutinase",
            "alternative_names": "",
            "catalytic_activity": "cutin + H2O = cutin monomers",
            "comments": "Degrades plant cutin.",
            "uniprot_accessions": "A0A024SC78",
            "transferred_to": "",
            "deleted": False,
        },
    }
    result, stats = enrich_dataframe(df, expasy_lookup=expasy_lookup)
    assert stats["expasy_hit_rate"] == 1.0
    assert result.loc[0, "annot_ec_numbers"] == "3.1.1.74"  # not clobbered
    assert "EC 3.1.1.74: cutinase" in result.loc[0, "annot_ec_names"]
    assert "Degrades plant cutin" in result.loc[0, "annot_ec_description"]

"""Tests for the ExPASy enzyme.dat parser."""

import textwrap

import pytest

from biom3.dbio.expasy import ExPASyEnzymeParser


HEADER = textwrap.dedent("""\
    CC   -----------------------------------------------------------------------
    CC   ENZYME nomenclature database
    CC   -----------------------------------------------------------------------
    CC   Release of 28-Jan-2026
    CC   -----------------------------------------------------------------------
    //
""")

FIXTURE = HEADER + textwrap.dedent("""\
    ID   1.1.1.1
    DE   alcohol dehydrogenase.
    AN   aldehyde reductase.
    AN   ethanol dehydrogenase.
    CA   (1) a primary alcohol + NAD(+) = an aldehyde + NADH + H(+).
    CA   (2) a secondary alcohol + NAD(+) = a ketone + NADH + H(+).
    CC   -!- Acts on primary or secondary alcohols with very broad specificity.
    CC   -!- Formerly EC 1.1.1.32.
    DR   P07327, ADH1A_HUMAN;  P28469, ADH1A_MACMU;  Q5RBP7, ADH1A_PONAB;
    DR   P25405, ADH1A_SAAHA;
    //
    ID   1.1.1.5
    DE   Transferred entry: 1.1.1.303 and 1.1.1.304.
    //
    ID   1.1.1.74
    DE   Deleted entry.
    //
    ID   1.14.11.1
    DE   gamma-butyrobetaine dioxygenase.
    CA   gamma-butyrobetaine + O2 = (R)-carnitine.
    DR   Q9UH18, BBOX1_HUMAN;
    //
""")


@pytest.fixture
def enzyme_dat(tmp_path):
    path = tmp_path / "enzyme.dat"
    path.write_text(FIXTURE)
    return path


def test_parses_header_and_skips_it(enzyme_dat):
    entries = list(ExPASyEnzymeParser(str(enzyme_dat)).iter_entries())
    assert len(entries) == 4
    assert [e.ec for e in entries] == ["1.1.1.1", "1.1.1.5", "1.1.1.74", "1.14.11.1"]


def test_extracts_name_and_alternative_names(enzyme_dat):
    entries = list(ExPASyEnzymeParser(str(enzyme_dat)).iter_entries())
    entry = entries[0]
    assert entry.name == "alcohol dehydrogenase"
    assert entry.alternative_names == [
        "aldehyde reductase",
        "ethanol dehydrogenase",
    ]


def test_splits_numbered_catalytic_activities(enzyme_dat):
    entries = list(ExPASyEnzymeParser(str(enzyme_dat)).iter_entries())
    cas = entries[0].catalytic_activities
    assert len(cas) == 2
    assert "primary alcohol" in cas[0]
    assert "secondary alcohol" in cas[1]


def test_bullet_comments_accumulated(enzyme_dat):
    entries = list(ExPASyEnzymeParser(str(enzyme_dat)).iter_entries())
    comments = entries[0].comments
    assert len(comments) == 2
    assert comments[0].startswith("Acts on primary or secondary")
    assert comments[1] == "Formerly EC 1.1.1.32."


def test_collects_dr_uniprot_accessions(enzyme_dat):
    entries = list(ExPASyEnzymeParser(str(enzyme_dat)).iter_entries())
    # EC 1.1.1.1 has four pairs split across two DR lines
    assert entries[0].uniprot_accessions == [
        "P07327", "P28469", "Q5RBP7", "P25405",
    ]
    assert entries[3].uniprot_accessions == ["Q9UH18"]


def test_marks_transferred_entry(enzyme_dat):
    entries = list(ExPASyEnzymeParser(str(enzyme_dat)).iter_entries())
    transferred = [e for e in entries if e.transferred_to]
    assert len(transferred) == 1
    assert transferred[0].ec == "1.1.1.5"
    assert transferred[0].transferred_to == ["1.1.1.303", "1.1.1.304"]


def test_marks_deleted_entry(enzyme_dat):
    entries = list(ExPASyEnzymeParser(str(enzyme_dat)).iter_entries())
    deleted = [e for e in entries if e.deleted]
    assert len(deleted) == 1
    assert deleted[0].ec == "1.1.1.74"


def test_empty_file_yields_nothing(tmp_path):
    path = tmp_path / "empty.dat"
    path.write_text("")
    assert list(ExPASyEnzymeParser(str(path)).iter_entries()) == []

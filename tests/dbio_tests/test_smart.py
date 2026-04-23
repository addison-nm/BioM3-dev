"""Tests for the SMART domains TSV reader."""

import textwrap

import pytest

from biom3.dbio.smart import SmartReader


FIXTURE = textwrap.dedent("""\
    DOMAIN\tACC\tDEFINITION\tDESCRIPTION
    --------------------------------------------------------------------------------
    14_3_3\tSM00101\t14-3-3 homologues\tMediates signal transduction.
    35EXOc\tSM00474\t3'-5' exonuclease\t3'-5' exonuclease proofreading domain.
    4.1m\tSM00294\tputative band 4.1 homologues' binding motif\t
    53EXOc\tSM00475\t5'-3' exonuclease\t

    AAA\tSM00382\tATPases associated with a variety of cellular activities\tAAA ATPase family.
""")


@pytest.fixture
def smart_txt(tmp_path):
    path = tmp_path / "SMART_domains.txt"
    path.write_text(FIXTURE)
    return path


def test_skips_header_and_separator(smart_txt):
    domains = list(SmartReader(str(smart_txt)).iter_domains())
    assert len(domains) == 5
    assert [d["accession"] for d in domains] == [
        "SM00101", "SM00474", "SM00294", "SM00475", "SM00382",
    ]


def test_parses_four_column_row(smart_txt):
    domains = list(SmartReader(str(smart_txt)).iter_domains())
    first = domains[0]
    assert first == {
        "domain_name": "14_3_3",
        "accession": "SM00101",
        "definition": "14-3-3 homologues",
        "description": "Mediates signal transduction.",
    }


def test_handles_empty_description(smart_txt):
    domains = list(SmartReader(str(smart_txt)).iter_domains())
    empty_desc = next(d for d in domains if d["accession"] == "SM00294")
    assert empty_desc["definition"].startswith("putative band 4.1")
    assert empty_desc["description"] == ""


def test_skips_blank_lines(smart_txt):
    domains = list(SmartReader(str(smart_txt)).iter_domains())
    # Blank line between SM00475 and SM00382 in the fixture is skipped silently.
    accessions = [d["accession"] for d in domains]
    assert "" not in accessions
    assert "SM00475" in accessions
    assert "SM00382" in accessions


def test_skips_row_missing_accession(tmp_path):
    # A row with a name but no accession is silently dropped.
    path = tmp_path / "SMART_domains.txt"
    path.write_text(
        "DOMAIN\tACC\tDEFINITION\tDESCRIPTION\n"
        "----\n"
        "OK\tSM00001\tOkay domain\tA real domain.\n"
        "BAD\t\tNo accession\tDefective row.\n"
    )
    domains = list(SmartReader(str(path)).iter_domains())
    assert [d["accession"] for d in domains] == ["SM00001"]


def test_empty_file_yields_nothing(tmp_path):
    path = tmp_path / "empty.txt"
    path.write_text("")
    assert list(SmartReader(str(path)).iter_domains()) == []

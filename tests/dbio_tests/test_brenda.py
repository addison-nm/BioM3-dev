"""Tests for the BRENDA flatfile parser."""

import textwrap

import pytest

from biom3.dbio.brenda import BrendaParser


FIXTURE = textwrap.dedent("""\
    BR\t2026.1

    ID\t1.1.1.1
    ****************************************

    PROTEIN
    PR\t#1# Homo sapiens <1,2,3>
    PR\t#2# Saccharomyces cerevisiae (yeast) <4,5>
    PR\t#3# Escherichia coli <6>

    RECOMMENDED_NAME
    RN\talcohol dehydrogenase

    SYSTEMATIC_NAME
    SN\talcohol:NAD+ oxidoreductase

    SYNONYMS
    SY\taldehyde reductase
    SY\tethanol dehydrogenase

    REACTION
    RE\ta primary alcohol + NAD+ = an aldehyde + NADH + H+

    SUBSTRATE_PRODUCT
    SP\t#1# ethanol + NAD+ = acetaldehyde + NADH + H+ <1>
    SP\t#1,2# butanol + NAD+ = butyraldehyde + NADH + H+ <1,4>
    SP\t#3# methanol + NAD+ = formaldehyde + NADH + H+ <6>

    KM_VALUE
    KM\t#1# 0.05 {ethanol} (#1# pH 7.4 <1>) <1>
    KM\t#2# 0.3 {ethanol} (#2# at 25 C <4>) <4>

    PH_OPTIMUM
    PHO\t#1# 7.4 <1>
    PHO\t#2# 6.5 <4>

    TEMPERATURE_OPTIMUM
    TO\t#1# 25 <1>

    ///
    ID\t1.1.1.2
    ****************************************

    PROTEIN
    PR\t#1# Mus musculus <1>

    RECOMMENDED_NAME
    RN\talcohol dehydrogenase (NADP+)

    ///
""")


@pytest.fixture
def brenda_txt(tmp_path):
    path = tmp_path / "brenda.txt"
    path.write_text(FIXTURE)
    return path


def test_parses_two_entries(brenda_txt):
    entries = list(BrendaParser(str(brenda_txt)).iter_entries())
    assert len(entries) == 2
    assert [e.ec for e in entries] == ["1.1.1.1", "1.1.1.2"]


def test_protein_block_maps_organism_numbers(brenda_txt):
    entry = list(BrendaParser(str(brenda_txt)).iter_entries())[0]
    assert set(entry.organisms) == {1, 2, 3}
    assert entry.organisms[1].name == "Homo sapiens"
    assert entry.organisms[2].name == "Saccharomyces cerevisiae"
    assert entry.organisms[2].details == "yeast"
    assert entry.organisms[3].name == "Escherichia coli"


def test_ec_level_fields(brenda_txt):
    entry = list(BrendaParser(str(brenda_txt)).iter_entries())[0]
    assert entry.recommended_name == "alcohol dehydrogenase"
    assert entry.systematic_name == "alcohol:NAD+ oxidoreductase"
    assert entry.synonyms == ["aldehyde reductase", "ethanol dehydrogenase"]
    assert entry.reactions == [
        "a primary alcohol + NAD+ = an aldehyde + NADH + H+",
    ]


def test_substrate_product_bucketed_by_organism(brenda_txt):
    entry = list(BrendaParser(str(brenda_txt)).iter_entries())[0]
    # Organism 1 gets its own SP record plus the shared #1,2# record.
    sp_1 = entry.substrates_products.get(1, [])
    sp_2 = entry.substrates_products.get(2, [])
    sp_3 = entry.substrates_products.get(3, [])
    assert any("ethanol" in s for s in sp_1)
    assert any("butanol" in s for s in sp_1)
    assert any("butanol" in s for s in sp_2)
    assert any("methanol" in s for s in sp_3)
    assert 4 not in entry.substrates_products


def test_kinetics_stored_per_organism(brenda_txt):
    entry = list(BrendaParser(str(brenda_txt)).iter_entries())[0]
    assert len(entry.km_values[1]) == 1
    assert "0.05" in entry.km_values[1][0]
    assert "{ethanol}" in entry.km_values[1][0]
    assert len(entry.km_values[2]) == 1
    assert "0.3" in entry.km_values[2][0]


def test_ph_and_temperature_parsed(brenda_txt):
    entry = list(BrendaParser(str(brenda_txt)).iter_entries())[0]
    assert entry.ph_optimum[1] == ["7.4"]
    assert entry.ph_optimum[2] == ["6.5"]
    assert entry.temperature_optimum[1] == ["25"]
    assert 2 not in entry.temperature_optimum


def test_missing_sections_left_empty(brenda_txt):
    # EC 1.1.1.2 has only PROTEIN + RECOMMENDED_NAME.
    entry = list(BrendaParser(str(brenda_txt)).iter_entries())[1]
    assert entry.ec == "1.1.1.2"
    assert entry.recommended_name == "alcohol dehydrogenase (NADP+)"
    assert entry.synonyms == []
    assert entry.reactions == []
    assert entry.substrates_products == {}
    assert entry.km_values == {}


def test_skips_pre_entry_header(tmp_path):
    # BRENDA files start with a "spontaneous" ID that is not a numeric EC.
    path = tmp_path / "brenda.txt"
    path.write_text(textwrap.dedent("""\
        BR\t2026.1

        ID\tspontaneous

        RECOMMENDED_NAME
        RN\tspontaneous reaction

        ///
        ID\t1.1.1.1

        RECOMMENDED_NAME
        RN\talcohol dehydrogenase

        ///
    """))
    entries = list(BrendaParser(str(path)).iter_entries())
    # Non-numeric 'spontaneous' ID is rejected, only 1.1.1.1 yielded.
    assert [e.ec for e in entries] == ["1.1.1.1"]


def test_multi_line_continuation(tmp_path):
    # Records that wrap to a second line (leading TAB) should be joined.
    path = tmp_path / "brenda.txt"
    path.write_text(textwrap.dedent("""\
        BR\t2026.1

        ID\t1.1.1.1

        RECOMMENDED_NAME
        RN\talcohol dehydrogenase

        SUBSTRATE_PRODUCT
        SP\t#1# a very long substrate name that wraps
        \tacross lines + NAD+ = product + NADH <1>

        ///
    """))
    entry = list(BrendaParser(str(path)).iter_entries())[0]
    assert len(entry.substrates_products[1]) == 1
    record = entry.substrates_products[1][0]
    assert "a very long substrate name that wraps" in record
    assert "across lines + NAD+" in record

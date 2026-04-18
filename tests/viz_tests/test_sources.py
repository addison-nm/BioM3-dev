from biom3.viz.sources import parse_pdb_id, parse_uniprot_id


class TestParsePdbId:
    def test_bar_format(self):
        assert parse_pdb_id("pdb|4HHB|A") == "4HHB"

    def test_bar_format_lowercase(self):
        assert parse_pdb_id("pdb|4hhb|A") == "4HHB"

    def test_underscore_chain_format(self):
        assert parse_pdb_id("4HHB_A") == "4HHB"

    def test_gi_wrapped(self):
        assert parse_pdb_id("gi|12345|pdb|1ABC|A") == "1ABC"

    def test_from_def_when_id_misses(self):
        assert parse_pdb_id("gi|99|ref|NP_1|", "pdb|2XYZ|B") == "2XYZ"

    def test_none_for_refseq(self):
        assert parse_pdb_id("ref|XP_12345.1|") is None

    def test_none_for_swissprot(self):
        assert parse_pdb_id("sp|P12345|NAME_ORG") is None

    def test_none_for_empty(self):
        assert parse_pdb_id("") is None


class TestParseUniprotId:
    def test_swissprot(self):
        assert parse_uniprot_id("sp|P12345|NAME_ORG") == "P12345"

    def test_trembl(self):
        assert parse_uniprot_id("tr|A1B2C3|UNK") == "A1B2C3"

    def test_falls_through_to_def(self):
        assert parse_uniprot_id("gi|12345", "sp|Q8N9H8|foo") == "Q8N9H8"

    def test_none_for_pdb(self):
        assert parse_uniprot_id("pdb|4HHB|A") is None

    def test_none_for_refseq(self):
        assert parse_uniprot_id("ref|XP_12345.1|") is None

    def test_longer_accession(self):
        assert parse_uniprot_id("sp|A0A024R1R8|HUMAN") == "A0A024R1R8"

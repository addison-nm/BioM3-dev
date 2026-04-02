import os

import pytest

from biom3.dbio.pfam_metadata import PfamMetadataParser

DATDIR = os.path.join(os.path.dirname(__file__), "..", "_data", "dbio")


@pytest.fixture
def sto_path():
    return os.path.join(DATDIR, "mini_pfam_metadata.sto")


class TestPfamMetadataParser:

    def test_parse_stockholm_family_count(self, sto_path):
        result = PfamMetadataParser(sto_path).parse()
        assert len(result) == 3

    def test_parse_stockholm_short_id(self, sto_path):
        result = PfamMetadataParser(sto_path).parse()
        assert result["PF04947"]["short_id"] == "Pox_VLTF3"
        assert result["PF01083"]["short_id"] == "Cutinase"
        assert result["PF00018"]["short_id"] == "SH3_1"

    def test_parse_stockholm_family_name(self, sto_path):
        result = PfamMetadataParser(sto_path).parse()
        assert result["PF04947"]["family_name"] == "Viral late transcription factor 3"
        assert result["PF01083"]["family_name"] == "Cutinase"
        assert result["PF00018"]["family_name"] == "SH3 domain"

    def test_parse_stockholm_family_description(self, sto_path):
        result = PfamMetadataParser(sto_path).parse()
        desc = result["PF04947"]["family_description"]
        assert "viral late transcription factor 3" in desc.lower()
        assert "poxviruses" in desc.lower()

    def test_parse_stockholm_empty_description(self, sto_path):
        result = PfamMetadataParser(sto_path).parse()
        # SH3_1 has no CC lines
        assert result["PF00018"]["family_description"] == ""

    def test_accession_version_stripped(self, sto_path):
        result = PfamMetadataParser(sto_path).parse()
        assert "PF04947" in result
        assert "PF04947.18" not in result

    def test_cc_lines_concatenated(self, sto_path):
        result = PfamMetadataParser(sto_path).parse()
        desc = result["PF01083"]["family_description"]
        # CC lines should be joined with spaces
        assert "serine esterases" in desc
        assert "cuticles" in desc

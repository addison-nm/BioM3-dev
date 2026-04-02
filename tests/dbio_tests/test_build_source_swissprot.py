import csv
import os

import pytest

from biom3.dbio.build_source_swissprot import build_swissprot_csv, SWISSPROT_SPEC
from biom3.dbio.caption import compose_row_caption
from biom3.dbio.pfam_metadata import PfamMetadataParser

DATDIR = os.path.join(os.path.dirname(__file__), "..", "_data", "dbio")
TMPDIR = os.path.join(os.path.dirname(__file__), "..", "_tmp")


@pytest.fixture(autouse=True)
def ensure_tmpdir():
    os.makedirs(TMPDIR, exist_ok=True)


@pytest.fixture
def pfam_metadata():
    sto_path = os.path.join(DATDIR, "mini_pfam_metadata.sto")
    return PfamMetadataParser(sto_path).parse()


@pytest.fixture
def dat_path():
    return os.path.join(DATDIR, "mini_swissprot.dat")


@pytest.fixture
def output_path():
    return os.path.join(TMPDIR, "test_swissprot_source.csv")


def _read_csv(path):
    with open(path) as f:
        reader = csv.DictReader(f)
        return list(reader)


class TestBuildSwissprotCsv:

    def test_basic_build(self, dat_path, pfam_metadata, output_path):
        build_swissprot_csv(dat_path, pfam_metadata, output_path)
        rows = _read_csv(output_path)
        # 3 entries have Pfam (Q6GZX4, A0A024SC78, P99999); Q197F8 does not
        assert len(rows) == 3

    def test_output_columns(self, dat_path, pfam_metadata, output_path):
        build_swissprot_csv(dat_path, pfam_metadata, output_path)
        with open(output_path) as f:
            reader = csv.reader(f)
            header = next(reader)
        assert header == [
            "primary_Accession",
            "protein_sequence",
            "[final]text_caption",
            "pfam_label",
        ]

    def test_only_pfam_entries(self, dat_path, pfam_metadata, output_path):
        build_swissprot_csv(dat_path, pfam_metadata, output_path)
        rows = _read_csv(output_path)
        accessions = {r["primary_Accession"] for r in rows}
        assert "Q6GZX4" in accessions
        assert "A0A024SC78" in accessions
        assert "P99999" in accessions
        # Q197F8 has no Pfam DR lines
        assert "Q197F8" not in accessions

    def test_sequence_extraction(self, dat_path, pfam_metadata, output_path):
        build_swissprot_csv(dat_path, pfam_metadata, output_path)
        rows = _read_csv(output_path)
        q6 = next(r for r in rows if r["primary_Accession"] == "Q6GZX4")
        seq = q6["protein_sequence"]
        assert seq.startswith("MAFSAEDVLK")
        assert seq.endswith("WKFTPL")
        assert len(seq) == 256

    def test_pfam_label_format(self, dat_path, pfam_metadata, output_path):
        build_swissprot_csv(dat_path, pfam_metadata, output_path)
        rows = _read_csv(output_path)
        q6 = next(r for r in rows if r["primary_Accession"] == "Q6GZX4")
        assert q6["pfam_label"] == "['PF04947']"

    def test_multi_pfam_label(self, dat_path, pfam_metadata, output_path):
        build_swissprot_csv(dat_path, pfam_metadata, output_path)
        rows = _read_csv(output_path)
        multi = next(r for r in rows if r["primary_Accession"] == "P99999")
        label = multi["pfam_label"]
        assert "PF01083" in label
        assert "PF04947" in label

    def test_caption_has_protein_name(self, dat_path, pfam_metadata, output_path):
        build_swissprot_csv(dat_path, pfam_metadata, output_path)
        rows = _read_csv(output_path)
        cut = next(r for r in rows if r["primary_Accession"] == "A0A024SC78")
        assert "PROTEIN NAME: Cutinase." in cut["[final]text_caption"]

    def test_caption_family_names_at_end(self, dat_path, pfam_metadata, output_path):
        build_swissprot_csv(dat_path, pfam_metadata, output_path)
        rows = _read_csv(output_path)
        cut = next(r for r in rows if r["primary_Accession"] == "A0A024SC78")
        caption = cut["[final]text_caption"]
        assert "FAMILY NAMES: Family names are Cutinase." in caption
        # FAMILY NAMES should be at the end
        family_idx = caption.index("FAMILY NAMES:")
        assert family_idx > caption.index("PROTEIN NAME:")

    def test_pubmed_refs_stripped(self, dat_path, pfam_metadata, output_path):
        build_swissprot_csv(dat_path, pfam_metadata, output_path)
        rows = _read_csv(output_path)
        cut = next(r for r in rows if r["primary_Accession"] == "A0A024SC78")
        assert "PubMed:" not in cut["[final]text_caption"]

    def test_evidence_tags_stripped(self, dat_path, pfam_metadata, output_path):
        build_swissprot_csv(dat_path, pfam_metadata, output_path)
        rows = _read_csv(output_path)
        q6 = next(r for r in rows if r["primary_Accession"] == "Q6GZX4")
        assert "{ECO:" not in q6["[final]text_caption"]

    def test_additional_cc_fields(self, dat_path, pfam_metadata, output_path):
        build_swissprot_csv(dat_path, pfam_metadata, output_path)
        rows = _read_csv(output_path)
        multi = next(r for r in rows if r["primary_Accession"] == "P99999")
        caption = multi["[final]text_caption"]
        assert "TISSUE SPECIFICITY:" in caption
        assert "DEVELOPMENTAL STAGE:" in caption


class TestComposeCaption:

    def test_empty_annotations(self):
        assert compose_row_caption({}, SWISSPROT_SPEC) == ""

    def test_single_field(self):
        result = compose_row_caption(
            {"annot_protein_name": "Kinase"}, SWISSPROT_SPEC,
        )
        assert result == "PROTEIN NAME: Kinase."

    def test_with_family_names(self):
        result = compose_row_caption(
            {"annot_protein_name": "Kinase"},
            SWISSPROT_SPEC,
            pfam_family_names=["SH3 domain", "Cutinase"],
        )
        assert "FAMILY NAMES: Family names are SH3 domain, Cutinase." in result

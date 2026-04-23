import csv
import json
import os

import pytest

from biom3.dbio.build_source_pfam import (
    _parse_fasta_header,
    build_pfam_csv,
    main,
    parse_arguments,
)
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
def fasta_path():
    return os.path.join(DATDIR, "mini_pfam.fasta")


@pytest.fixture
def output_path():
    return os.path.join(TMPDIR, "test_pfam_source.csv")


def _read_csv(path):
    with open(path) as f:
        reader = csv.DictReader(f)
        return list(reader)


class TestParseFastaHeader:

    def test_basic_header(self):
        header = "Q6GZX4_FRG3G/1-256 Q6GZX4.1 PF04947.18;Pox_VLTF3;"
        result = _parse_fasta_header(header)
        assert result["id"] == "Q6GZX4"
        assert result["range"] == "1-256"
        assert result["pfam_label"] == "PF04947"
        assert result["description"] == "Q6GZX4.1 PF04947.18;Pox_VLTF3;"

    def test_accession_version_stripped(self):
        header = "A0A024SC78_TRIHA/10-120 A0A024SC78.1 PF01083.27;Cutinase;"
        result = _parse_fasta_header(header)
        assert result["id"] == "A0A024SC78"
        assert result["pfam_label"] == "PF01083"

    def test_short_header_returns_none(self):
        assert _parse_fasta_header("too short") is None


class TestBuildPfamCsv:

    def test_basic_build(self, fasta_path, pfam_metadata, output_path):
        build_pfam_csv(fasta_path, pfam_metadata, output_path)
        rows = _read_csv(output_path)
        assert len(rows) == 5

    def test_output_columns(self, fasta_path, pfam_metadata, output_path):
        build_pfam_csv(fasta_path, pfam_metadata, output_path)
        with open(output_path) as f:
            reader = csv.reader(f)
            header = next(reader)
        assert header == [
            "id", "range", "description", "pfam_label",
            "sequence", "family_name", "family_description",
            "[final]text_caption",
        ]

    def test_domain_sequence(self, fasta_path, pfam_metadata, output_path):
        build_pfam_csv(fasta_path, pfam_metadata, output_path)
        rows = _read_csv(output_path)
        first = rows[0]
        assert first["sequence"].startswith("MAFSAEDVLK")

    def test_family_name_populated(self, fasta_path, pfam_metadata, output_path):
        build_pfam_csv(fasta_path, pfam_metadata, output_path)
        rows = _read_csv(output_path)
        pox = next(r for r in rows if r["pfam_label"] == "PF04947")
        assert pox["family_name"] == "Viral late transcription factor 3"

    def test_family_description_populated(self, fasta_path, pfam_metadata, output_path):
        build_pfam_csv(fasta_path, pfam_metadata, output_path)
        rows = _read_csv(output_path)
        cut = next(r for r in rows if r["pfam_label"] == "PF01083")
        assert "serine esterases" in cut["family_description"]

    def test_caption_lowercase_format(self, fasta_path, pfam_metadata, output_path):
        build_pfam_csv(fasta_path, pfam_metadata, output_path)
        rows = _read_csv(output_path)
        first = rows[0]
        caption = first["[final]text_caption"]
        assert caption.startswith("Protein name:")

    def test_caption_has_family_description(self, fasta_path, pfam_metadata, output_path):
        build_pfam_csv(fasta_path, pfam_metadata, output_path)
        rows = _read_csv(output_path)
        cut = next(r for r in rows if r["pfam_label"] == "PF01083")
        assert "Family description:" in cut["[final]text_caption"]

    def test_missing_metadata_graceful(self, fasta_path, output_path):
        # Empty metadata dict — should still produce rows with empty family info
        build_pfam_csv(fasta_path, {}, output_path)
        rows = _read_csv(output_path)
        assert len(rows) == 5
        for row in rows:
            assert row["family_name"] == ""

    def test_pfam_label_no_version(self, fasta_path, pfam_metadata, output_path):
        build_pfam_csv(fasta_path, pfam_metadata, output_path)
        rows = _read_csv(output_path)
        for row in rows:
            assert "." not in row["pfam_label"]


class TestMainWritesManifest:

    def test_main_writes_build_manifest(self, fasta_path, tmp_path):
        pfam_sto = os.path.join(DATDIR, "mini_pfam_metadata.sto")
        output_csv = tmp_path / "pfam_source.csv"
        args = parse_arguments([
            "--fasta", fasta_path,
            "--pfam_metadata", pfam_sto,
            "-o", str(output_csv),
        ])
        main(args)

        manifest_path = tmp_path / "pfam_source.build_manifest.json"
        assert manifest_path.exists()

        with open(manifest_path) as f:
            manifest = json.load(f)
        for key in ("biom3_version", "git_hash", "timestamp",
                    "command", "args", "database_versions", "outputs"):
            assert key in manifest, f"missing top-level key: {key}"
        assert manifest["outputs"]["row_counts"]["pfam"] == 5
        assert manifest["database_versions"]["pfam_fasta"] is not None
        assert manifest["database_versions"]["pfam_metadata"] is not None

"""Tests for the build_dataset pipeline."""

import os
import pytest

import pandas as pd

from biom3.dbio.build_dataset import parse_arguments, main
from biom3.dbio.swissprot import OUTPUT_COLS

DATDIR = os.path.join("tests", "_data", "dbio")
SWISSPROT_PATH = os.path.join(DATDIR, "mini_swissprot.csv")
PFAM_PATH = os.path.join(DATDIR, "mini_pfam.csv")


class TestBuildDataset:

    @pytest.fixture
    def outdir(self, tmp_path):
        return str(tmp_path / "test_output")

    def test_basic_build(self, outdir):
        args = parse_arguments([
            "-p", "PF00018",
            "--swissprot", SWISSPROT_PATH,
            "--pfam", PFAM_PATH,
            "-o", outdir,
        ])
        main(args)

        dataset_path = os.path.join(outdir, "dataset.csv")
        assert os.path.exists(dataset_path)

        df = pd.read_csv(dataset_path)
        # SwissProt: 4 rows (P12345, P12347, P12351, P12352)
        # Pfam: 5 rows (A0A001-A0A005)
        assert len(df) == 9
        assert list(df.columns) == OUTPUT_COLS

    def test_annotations_csv_saved(self, outdir):
        args = parse_arguments([
            "-p", "PF00018",
            "--swissprot", SWISSPROT_PATH,
            "--pfam", PFAM_PATH,
            "-o", outdir,
        ])
        main(args)

        annotations_path = os.path.join(outdir, "dataset_annotations.csv")
        assert os.path.exists(annotations_path)

        df_annot = pd.read_csv(annotations_path)
        assert len(df_annot) == 9
        # Should have at least the output columns
        for col in OUTPUT_COLS:
            assert col in df_annot.columns

    def test_multiple_pfam_ids(self, outdir):
        args = parse_arguments([
            "-p", "PF00018", "PF00042",
            "--swissprot", SWISSPROT_PATH,
            "--pfam", PFAM_PATH,
            "-o", outdir,
        ])
        main(args)

        df = pd.read_csv(os.path.join(outdir, "dataset.csv"))
        # SwissProt: 4 (PF00018) + 2 (PF00042) = 6
        # Pfam: 5 (PF00018) + 3 (PF00042) = 8
        assert len(df) == 14

    def test_pfam_ids_metadata(self, outdir):
        args = parse_arguments([
            "-p", "PF00018", "PF00042",
            "--swissprot", SWISSPROT_PATH,
            "--pfam", PFAM_PATH,
            "-o", outdir,
        ])
        main(args)

        pfam_ids = pd.read_csv(os.path.join(outdir, "pfam_ids.csv"))
        assert set(pfam_ids["pfam_id"]) == {"PF00018", "PF00042"}

    def test_no_match(self, outdir):
        args = parse_arguments([
            "-p", "PF99999",
            "--swissprot", SWISSPROT_PATH,
            "--pfam", PFAM_PATH,
            "-o", outdir,
        ])
        main(args)

        df = pd.read_csv(os.path.join(outdir, "dataset.csv"))
        assert len(df) == 0

    def test_output_columns(self, outdir):
        args = parse_arguments([
            "-p", "PF00071",
            "--swissprot", SWISSPROT_PATH,
            "--pfam", PFAM_PATH,
            "-o", outdir,
        ])
        main(args)

        df = pd.read_csv(os.path.join(outdir, "dataset.csv"))
        assert list(df.columns) == OUTPUT_COLS

    def test_pfam_caption_from_family_columns(self, outdir):
        """Pfam rows should get captions composed from family_name/description."""
        args = parse_arguments([
            "-p", "PF00018",
            "--swissprot", SWISSPROT_PATH,
            "--pfam", PFAM_PATH,
            "-o", outdir,
        ])
        main(args)

        df = pd.read_csv(os.path.join(outdir, "dataset.csv"))
        # Find a Pfam row (accession starts with A0A)
        pfam_rows = df[df["primary_Accession"].str.startswith("A0A")]
        assert len(pfam_rows) > 0
        caption = pfam_rows.iloc[0]["[final]text_caption"]
        assert "FAMILY NAME:" in caption
        assert "FAMILY DESCRIPTION:" in caption

    def test_build_log_and_manifest(self, outdir):
        """Build should produce a log file and a JSON manifest."""
        args = parse_arguments([
            "-p", "PF00018",
            "--swissprot", SWISSPROT_PATH,
            "--pfam", PFAM_PATH,
            "-o", outdir,
        ])
        main(args)

        # Log file
        log_path = os.path.join(outdir, "build.log")
        assert os.path.exists(log_path)
        log_content = open(log_path).read()
        assert "Build fine-tuning dataset" in log_content
        assert "PF00018" in log_content

        # Manifest
        import json
        manifest_path = os.path.join(outdir, "build_manifest.json")
        assert os.path.exists(manifest_path)
        manifest = json.load(open(manifest_path))
        assert "biom3_version" in manifest
        assert "git_hash" in manifest
        assert "command" in manifest
        assert "args" in manifest
        assert manifest["args"]["pfam_ids"] == ["PF00018"]
        assert manifest["outputs"]["row_counts"]["swissprot"] == 4
        assert manifest["outputs"]["row_counts"]["pfam"] == 5
        assert manifest["outputs"]["row_counts"]["combined"] == 9

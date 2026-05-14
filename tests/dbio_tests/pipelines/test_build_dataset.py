"""Tests for the build_dataset pipeline."""

import os
import pytest

import pandas as pd

from biom3.dbio.pipelines.build_dataset import parse_arguments, main
from biom3.dbio.readers.swissprot_csv import OUTPUT_COLS

DATDIR = os.path.join("tests", "_data", "dbio")
SWISSPROT_PATH = os.path.join(DATDIR, "mini_swissprot.csv")
PFAM_PATH = os.path.join(DATDIR, "mini_pfam.csv")
PFAM_ALT_PATH = os.path.join(DATDIR, "mini_pfam_alt.csv")


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


class TestTaxonomyFilterNormalization:
    """parse_arguments should treat literal 'None' values as not-set."""

    def test_filter_none_string_normalized_to_none(self):
        args = parse_arguments([
            "-p", "PF00018",
            "--swissprot", SWISSPROT_PATH,
            "--pfam", PFAM_PATH,
            "-o", "/tmp/x",
            "--taxonomy_filter", "None",
        ])
        assert args.taxonomy_filter is None

    def test_filter_mixed_drops_only_none(self):
        args = parse_arguments([
            "-p", "PF00018",
            "--swissprot", SWISSPROT_PATH,
            "--pfam", PFAM_PATH,
            "-o", "/tmp/x",
            "--taxonomy_filter", "superkingdom=Bacteria", "None",
        ])
        assert args.taxonomy_filter == ["superkingdom=Bacteria"]

    def test_filter_real_value_unchanged(self):
        args = parse_arguments([
            "-p", "PF00018",
            "--swissprot", SWISSPROT_PATH,
            "--pfam", PFAM_PATH,
            "-o", "/tmp/x",
            "--taxonomy_filter", "phylum=Pseudomonadota",
        ])
        assert args.taxonomy_filter == ["phylum=Pseudomonadota"]

    def test_filter_unset_remains_none(self):
        args = parse_arguments([
            "-p", "PF00018",
            "--swissprot", SWISSPROT_PATH,
            "--pfam", PFAM_PATH,
            "-o", "/tmp/x",
        ])
        assert args.taxonomy_filter is None


class TestTaxonomyDirOverride:
    """--taxonomy_dir should bypass get_database_path('ncbi_taxonomy', ...)."""

    def test_flag_parsed(self):
        args = parse_arguments([
            "-p", "PF00018",
            "--swissprot", SWISSPROT_PATH,
            "--pfam", PFAM_PATH,
            "-o", "/tmp/x",
            "--taxonomy_dir", "/some/explicit/path",
        ])
        assert args.taxonomy_dir == "/some/explicit/path"

    def test_resolve_uses_explicit_path_when_set(self):
        from biom3.dbio.pipelines.build_dataset import _resolve_taxonomy_dir
        args = parse_arguments([
            "-p", "PF00018",
            "--swissprot", SWISSPROT_PATH,
            "--pfam", PFAM_PATH,
            "-o", "/tmp/x",
            "--taxonomy_dir", "/explicit/ncbi_taxonomy",
        ])
        assert _resolve_taxonomy_dir(args) == "/explicit/ncbi_taxonomy"

    def test_resolve_falls_through_to_config_when_unset(self, monkeypatch, tmp_path):
        """Without --taxonomy_dir, _resolve_taxonomy_dir should reach for config."""
        from biom3.dbio.pipelines.build_dataset import _resolve_taxonomy_dir

        # Point BIOM3_DATABASES_ROOT at a tmp dir so the config path
        # resolves cleanly without depending on the host env.
        fake_root = tmp_path / "databases"
        (fake_root / "ncbi_taxonomy").mkdir(parents=True)
        monkeypatch.setenv("BIOM3_DATABASES_ROOT", str(fake_root))

        args = parse_arguments([
            "-p", "PF00018",
            "--swissprot", SWISSPROT_PATH,
            "--pfam", PFAM_PATH,
            "-o", "/tmp/x",
        ])
        assert args.taxonomy_dir is None
        resolved = _resolve_taxonomy_dir(args)
        assert resolved.endswith("ncbi_taxonomy")


class TestPfamMultiInput:
    """--pfam accepts multiple paths; concat + dedupe semantics."""

    @pytest.fixture
    def outdir(self, tmp_path):
        return str(tmp_path / "test_output")

    def test_single_pfam_path_still_works(self, outdir):
        """Regression: one --pfam path produces the same output as before."""
        args = parse_arguments([
            "-p", "PF00018",
            "--swissprot", SWISSPROT_PATH,
            "--pfam", PFAM_PATH,
            "-o", outdir,
        ])
        # nargs="+" wraps a single path in a 1-element list
        assert args.pfam == [PFAM_PATH]
        main(args)
        df = pd.read_csv(os.path.join(outdir, "dataset.csv"))
        assert len(df) == 9  # same as the basic single-path test

    def test_multi_pfam_paths_concatenate_with_dedupe_default(self, outdir):
        """Two --pfam paths with one overlapping row → deduped on (acc, pfam_label)."""
        args = parse_arguments([
            "-p", "PF00018", "PF07714",
            "--swissprot", SWISSPROT_PATH,
            "--pfam", PFAM_PATH, PFAM_ALT_PATH,
            "-o", outdir,
        ])
        assert args.pfam == [PFAM_PATH, PFAM_ALT_PATH]
        assert args.no_dedupe_pfam is False
        main(args)
        df = pd.read_csv(os.path.join(outdir, "dataset.csv"))
        # The Pfam side must have unique (primary_Accession, pfam_label).
        # mini_pfam.csv contributes PF00018 rows for A0A001..A0A005 and
        # mini_pfam_alt.csv contributes A0A001 (overlap), A0A100, A0A101.
        # After dedupe the Pfam side has the 5 originals + 2 new (A0A100,
        # A0A101) = 7 unique Pfam rows. SwissProt contributes 4. Total: 11.
        pfam_side = df[df["primary_Accession"].str.startswith("A0A")]
        accs = list(pfam_side["primary_Accession"])
        assert "A0A001" in accs
        assert "A0A100" in accs
        assert "A0A101" in accs
        # Only one A0A001 row despite appearing in both inputs:
        assert accs.count("A0A001") == 1

    def test_no_dedupe_pfam_preserves_duplicates(self, outdir):
        """--no_dedupe_pfam: A0A001 appears twice (once per input)."""
        args = parse_arguments([
            "-p", "PF00018", "PF07714",
            "--swissprot", SWISSPROT_PATH,
            "--pfam", PFAM_PATH, PFAM_ALT_PATH,
            "--no_dedupe_pfam",
            "-o", outdir,
        ])
        assert args.no_dedupe_pfam is True
        main(args)
        df = pd.read_csv(os.path.join(outdir, "dataset.csv"))
        accs = list(df[df["primary_Accession"].str.startswith("A0A")]["primary_Accession"])
        # A0A001 appears in both fixtures, multiset preserves both
        assert accs.count("A0A001") == 2

    def test_manifest_records_list_of_pfam_paths(self, outdir):
        """Manifest's pfam_csv resolved_paths and database_versions are lists."""
        import json
        args = parse_arguments([
            "-p", "PF00018", "PF07714",
            "--swissprot", SWISSPROT_PATH,
            "--pfam", PFAM_PATH, PFAM_ALT_PATH,
            "-o", outdir,
        ])
        main(args)
        manifest = json.load(open(os.path.join(outdir, "build_manifest.json")))
        assert isinstance(manifest["resolved_paths"]["pfam_csv"], list)
        assert len(manifest["resolved_paths"]["pfam_csv"]) == 2
        assert isinstance(manifest["database_versions"]["pfam_csv"], list)
        assert len(manifest["database_versions"]["pfam_csv"]) == 2

    def test_resolve_pfam_paths_returns_list_in_config_fallback(
        self, monkeypatch, tmp_path,
    ):
        """Without --pfam, the resolver returns a 1-element list."""
        from biom3.dbio.pipelines.build_dataset import _resolve_pfam_paths

        # Build a minimal fake databases_root with a writable pfam_csv
        fake_root = tmp_path / "databases"
        (fake_root / "datasets").mkdir(parents=True)
        fake_pfam = fake_root / "datasets" / "Pfam_protein_text_dataset.csv"
        fake_pfam.write_text("id,range,description,pfam_label,sequence\n")
        monkeypatch.setenv("BIOM3_DATABASES_ROOT", str(fake_root))

        # Trick config: training_data_root defaults to databases_root if no
        # config; the resolver calls get_training_data_path which under the
        # hood uses get_training_data_root → falls back to databases_root.
        args = parse_arguments([
            "-p", "PF00018",
            "--swissprot", SWISSPROT_PATH,
            "-o", str(tmp_path / "out"),
        ])
        assert args.pfam is None
        resolved = _resolve_pfam_paths(args)
        assert isinstance(resolved, list)
        assert len(resolved) == 1

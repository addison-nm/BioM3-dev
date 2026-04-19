import csv
import json
import os

import pytest

from biom3.dbio.build_source_swissprot import (
    SWISSPROT_SPEC,
    build_swissprot_csv,
    main,
    parse_arguments,
)
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
        # Default keeps all 4 entries; Q197F8 has no Pfam so its pfam_label
        # is ['nan'] for legacy parity.
        assert len(rows) == 4

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
            "annot_ec_numbers",
        ]

    def test_only_pfam_entries_when_require_pfam(self, dat_path, pfam_metadata, output_path):
        build_swissprot_csv(
            dat_path, pfam_metadata, output_path, require_pfam=True,
        )
        rows = _read_csv(output_path)
        accessions = {r["primary_Accession"] for r in rows}
        assert "Q6GZX4" in accessions
        assert "A0A024SC78" in accessions
        assert "P99999" in accessions
        # Q197F8 has no Pfam DR lines — filtered out by require_pfam=True
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

    def test_ec_numbers_extracted_from_catalytic_activity(
        self, dat_path, pfam_metadata, output_path,
    ):
        # The Cutinase fixture entry (A0A024SC78) has
        #   CC   -!- CATALYTIC ACTIVITY:
        #   CC       Reaction=cutin + H2O = cutin monomers; EC=3.1.1.74;
        # so annot_ec_numbers should carry 3.1.1.74.
        build_swissprot_csv(dat_path, pfam_metadata, output_path)
        rows = _read_csv(output_path)
        cutinase = next(r for r in rows if r["primary_Accession"] == "A0A024SC78")
        assert cutinase["annot_ec_numbers"] == "3.1.1.74"

    def test_ec_numbers_absent_for_non_enzymes(
        self, dat_path, pfam_metadata, output_path,
    ):
        build_swissprot_csv(dat_path, pfam_metadata, output_path)
        rows = _read_csv(output_path)
        non_enzyme = next(r for r in rows if r["primary_Accession"] == "Q6GZX4")
        assert non_enzyme["annot_ec_numbers"] == ""

    def test_catalytic_activity_prose_excludes_ec_xref(
        self, dat_path, pfam_metadata, output_path,
    ):
        # annot_catalytic_activity stays prose-only; the EC xref lives in
        # annot_ec_numbers instead of being embedded in the caption.
        build_swissprot_csv(dat_path, pfam_metadata, output_path)
        rows = _read_csv(output_path)
        cutinase = next(r for r in rows if r["primary_Accession"] == "A0A024SC78")
        caption = cutinase["[final]text_caption"]
        assert "EC=" not in caption
        assert "CATALYTIC ACTIVITY: cutin + H2O = cutin monomers" in caption

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


class TestRequirePfamFlag:

    def test_no_require_pfam_keeps_pfamless_entry(self, dat_path, pfam_metadata, output_path):
        build_swissprot_csv(
            dat_path, pfam_metadata, output_path, require_pfam=False,
        )
        rows = _read_csv(output_path)
        # All 4 entries should now appear (Q6GZX4, A0A024SC78, P99999, Q197F8)
        assert len(rows) == 4
        accessions = {r["primary_Accession"] for r in rows}
        assert "Q197F8" in accessions

    def test_no_require_pfam_uses_nan_label(self, dat_path, pfam_metadata, output_path):
        build_swissprot_csv(
            dat_path, pfam_metadata, output_path, require_pfam=False,
        )
        rows = _read_csv(output_path)
        q197 = next(r for r in rows if r["primary_Accession"] == "Q197F8")
        assert q197["pfam_label"] == "['nan']"

    def test_no_require_pfam_preserves_pfam_entries(self, dat_path, pfam_metadata, output_path):
        build_swissprot_csv(
            dat_path, pfam_metadata, output_path, require_pfam=False,
        )
        rows = _read_csv(output_path)
        q6 = next(r for r in rows if r["primary_Accession"] == "Q6GZX4")
        # Pfam-having entries still get their real IDs, not ['nan']
        assert q6["pfam_label"] == "['PF04947']"

    def test_default_matches_legacy_behavior(self, dat_path, pfam_metadata, output_path):
        # Default is require_pfam=False — matches the legacy CSV, which kept
        # Pfam-less entries with pfam_label=['nan'].
        build_swissprot_csv(dat_path, pfam_metadata, output_path)
        rows = _read_csv(output_path)
        assert len(rows) == 4
        q197 = next(r for r in rows if r["primary_Accession"] == "Q197F8")
        assert q197["pfam_label"] == "['nan']"

    def test_require_pfam_filters_pfamless(self, dat_path, pfam_metadata, output_path):
        build_swissprot_csv(
            dat_path, pfam_metadata, output_path, require_pfam=True,
        )
        rows = _read_csv(output_path)
        assert len(rows) == 3
        accessions = {r["primary_Accession"] for r in rows}
        assert "Q197F8" not in accessions


class TestKeepIntermediateCaptions:

    def test_default_emits_five_columns(self, dat_path, pfam_metadata, output_path):
        build_swissprot_csv(dat_path, pfam_metadata, output_path)
        with open(output_path) as f:
            header = next(csv.reader(f))
        assert header == [
            "primary_Accession",
            "protein_sequence",
            "[final]text_caption",
            "pfam_label",
            "annot_ec_numbers",
        ]

    def test_keep_intermediate_emits_seven_columns(self, dat_path, pfam_metadata, output_path):
        build_swissprot_csv(
            dat_path, pfam_metadata, output_path,
            keep_intermediate_captions=True,
        )
        with open(output_path) as f:
            header = next(csv.reader(f))
        assert header == [
            "primary_Accession",
            "protein_sequence",
            "text_caption",
            "[clean]text_caption",
            "[final]text_caption",
            "pfam_label",
            "annot_ec_numbers",
        ]

    def test_raw_text_caption_retains_pubmed_and_eco(self, dat_path, pfam_metadata, output_path):
        build_swissprot_csv(
            dat_path, pfam_metadata, output_path,
            keep_intermediate_captions=True,
        )
        rows = _read_csv(output_path)
        cut = next(r for r in rows if r["primary_Accession"] == "A0A024SC78")
        raw = cut["text_caption"]
        # The mini fixture for A0A024SC78 has PubMed refs — they should be
        # retained in the raw column and stripped in the final column.
        assert "(PubMed:" in raw
        assert "(PubMed:" not in cut["[final]text_caption"]

    def test_clean_text_caption_strips_eco_only(self, dat_path, pfam_metadata, output_path):
        build_swissprot_csv(
            dat_path, pfam_metadata, output_path,
            keep_intermediate_captions=True,
        )
        rows = _read_csv(output_path)
        cut = next(r for r in rows if r["primary_Accession"] == "A0A024SC78")
        clean = cut["[clean]text_caption"]
        # ECO evidence tags are stripped but PubMed refs are kept
        assert "{ECO:" not in clean
        # The mini_swissprot fixture carries PubMed refs on this entry
        assert "PubMed:" in clean


class TestMainWritesManifest:

    def test_main_writes_build_manifest(self, dat_path, tmp_path):
        pfam_sto = os.path.join(DATDIR, "mini_pfam_metadata.sto")
        output_csv = tmp_path / "swissprot_source.csv"
        args = parse_arguments([
            "--dat", dat_path,
            "--pfam_metadata", pfam_sto,
            "-o", str(output_csv),
        ])
        main(args)

        manifest_path = tmp_path / "swissprot_source.build_manifest.json"
        assert manifest_path.exists()

        with open(manifest_path) as f:
            manifest = json.load(f)
        for key in ("biom3_version", "git_hash", "timestamp",
                    "command", "args", "database_versions", "outputs"):
            assert key in manifest, f"missing top-level key: {key}"
        # Default is require_pfam=False, so all 4 fixture entries are kept.
        assert manifest["args"]["require_pfam"] is False
        assert manifest["outputs"]["row_counts"]["swissprot"] == 4
        assert manifest["database_versions"]["uniprot_dat"] is not None
        assert manifest["database_versions"]["pfam_metadata"] is not None

    def test_main_manifest_records_require_pfam_arg(self, dat_path, tmp_path):
        pfam_sto = os.path.join(DATDIR, "mini_pfam_metadata.sto")
        output_csv = tmp_path / "swissprot_source.csv"
        args = parse_arguments([
            "--dat", dat_path,
            "--pfam_metadata", pfam_sto,
            "-o", str(output_csv),
            "--require_pfam",
        ])
        main(args)

        with open(tmp_path / "swissprot_source.build_manifest.json") as f:
            manifest = json.load(f)
        assert manifest["args"]["require_pfam"] is True
        assert manifest["outputs"]["row_counts"]["swissprot"] == 3

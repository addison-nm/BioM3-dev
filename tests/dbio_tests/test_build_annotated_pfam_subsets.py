import argparse
import csv
import json
import os

import pytest

from biom3.dbio.build_annotated_pfam_subsets import (
    OUTPUT_COLUMNS,
    _clean_sequence,
    build_annotated_pfam_subsets_csv,
    iter_annotated_pfam_rows,
    main,
)

DATDIR = os.path.join(os.path.dirname(__file__), "..", "_data", "dbio")
STO_PATH = os.path.join(DATDIR, "mini_pfam_full.sto")


def _make_args(pfam_ids, pfam_full, output, chunk_size=100):
    return argparse.Namespace(
        pfam_ids=pfam_ids,
        pfam_full=pfam_full,
        output=output,
        chunk_size=chunk_size,
    )


def _read_csv(path):
    with open(path) as f:
        reader = csv.DictReader(f)
        return list(reader)


class TestCleanSequence:

    def test_strip_dot_and_dash(self):
        assert _clean_sequence("MK.LV-DE") == "MKLVDE"

    def test_uppercase_insertion_residues(self):
        assert _clean_sequence("YVrc.DE-F") == "YVRCDEF"

    def test_leading_and_trailing_gaps(self):
        assert _clean_sequence("...MKLV---") == "MKLV"


class TestIterAnnotatedPfamRows:

    def test_single_family_row_count(self):
        rows = list(iter_annotated_pfam_rows(STO_PATH, ["PF00018"]))
        assert len(rows) == 3

    def test_multi_family_row_count(self):
        rows = list(iter_annotated_pfam_rows(STO_PATH, ["PF00018", "PF01083"]))
        assert len(rows) == 6

    def test_sequence_is_uppercased_and_ungapped(self):
        rows = list(iter_annotated_pfam_rows(STO_PATH, ["PF00018"]))
        # H6SH3A3_TEST3 has mixed-case residues and `.` gap chars
        h6 = next(r for r in rows if r["id"] == "H6SH3A3")
        assert h6["sequence"] == "YVRALYDYTAKEDDCLSFKEGDIIINLK"

    def test_accession_resolved_from_gs_lines(self):
        rows = list(iter_annotated_pfam_rows(STO_PATH, ["PF01083"]))
        assert {r["id"] for r in rows} == {"C1CUTI1", "D2CUTI2", "E3CUTI3"}

    def test_range_parsed(self):
        rows = list(iter_annotated_pfam_rows(STO_PATH, ["PF01083"]))
        c1 = next(r for r in rows if r["id"] == "C1CUTI1")
        assert c1["range"] == "30-44"

    def test_pfam_label_unversioned(self):
        rows = list(iter_annotated_pfam_rows(STO_PATH, ["PF00018"]))
        for r in rows:
            assert r["pfam_label"] == "PF00018"

    def test_family_type_populated(self):
        rows = list(iter_annotated_pfam_rows(STO_PATH, ["PF00018"]))
        assert all(r["family_type"] == "Domain" for r in rows)

    def test_family_clan_populated(self):
        rows = list(iter_annotated_pfam_rows(STO_PATH, ["PF00018"]))
        assert all(r["family_clan"] == "CL0010" for r in rows)

    def test_family_wikipedia_populated(self):
        rows = list(iter_annotated_pfam_rows(STO_PATH, ["PF00018"]))
        assert all(r["family_wikipedia"] == "SH3_domain" for r in rows)

    def test_family_references_joined(self):
        rows = list(iter_annotated_pfam_rows(STO_PATH, ["PF00018"]))
        refs = rows[0]["family_references"]
        assert "Diverse recognition of non-PxxP peptide ligands by the SH3" in refs
        assert "Grb2 and Pex13p" in refs

    def test_clanless_and_no_wikipedia(self):
        # PF04947 (Pox_VLTF3) is deliberately clanless and has no #=GF WK.
        rows = list(iter_annotated_pfam_rows(STO_PATH, ["PF04947"]))
        assert len(rows) == 2
        assert rows[0]["family_clan"] == ""
        assert rows[0]["family_wikipedia"] == ""
        assert rows[0]["family_type"] == "Family"

    def test_missing_family_warns_and_emits_zero_rows(self, caplog, monkeypatch):
        # biom3's logger has propagate=False (rank-aware StreamHandler).
        # Temporarily flip propagate so caplog can intercept the warning.
        import logging
        mod_logger = logging.getLogger("biom3.dbio.build_annotated_pfam_subsets")
        monkeypatch.setattr(mod_logger, "propagate", True)
        caplog.set_level(logging.WARNING, logger=mod_logger.name)

        rows = list(iter_annotated_pfam_rows(STO_PATH, ["PF99999"]))
        assert rows == []
        assert any("PF99999" in rec.message for rec in caplog.records)

    def test_early_exit_after_all_targets_found(self):
        # The fixture has PF04947 (first) → PF01083 (middle) → PF00018 (last).
        # Requesting only PF04947 should stop after the first block without
        # scanning the rest. We verify by confirming it's correct; a direct
        # timing test would be flaky.
        rows = list(iter_annotated_pfam_rows(STO_PATH, ["PF04947"]))
        assert len(rows) == 2


class TestBuildAnnotatedPfamSubsetsCsv:

    def test_header_and_row_count(self, tmp_path):
        out = tmp_path / "subsets.csv"
        row_count, _ = build_annotated_pfam_subsets_csv(
            STO_PATH, ["PF00018", "PF01083"], str(out),
        )
        assert row_count == 6
        with open(out) as f:
            reader = csv.reader(f)
            header = next(reader)
            data_rows = list(reader)
        assert header == OUTPUT_COLUMNS
        assert len(data_rows) == 6

    def test_final_caption_composed(self, tmp_path):
        out = tmp_path / "subsets.csv"
        build_annotated_pfam_subsets_csv(STO_PATH, ["PF00018"], str(out))
        rows = _read_csv(str(out))
        caption = rows[0]["[final]text_caption"]
        assert "SH3 domain" in caption

    def test_family_columns_in_output(self, tmp_path):
        out = tmp_path / "subsets.csv"
        build_annotated_pfam_subsets_csv(STO_PATH, ["PF00018"], str(out))
        rows = _read_csv(str(out))
        assert rows[0]["family_type"] == "Domain"
        assert rows[0]["family_clan"] == "CL0010"
        assert rows[0]["family_wikipedia"] == "SH3_domain"
        assert "Diverse recognition" in rows[0]["family_references"]

    def test_stats_builder_row_count(self, tmp_path):
        out = tmp_path / "subsets.csv"
        _, stats_builder = build_annotated_pfam_subsets_csv(
            STO_PATH, ["PF00018", "PF01083"], str(out),
        )
        stats = stats_builder.finalize()
        assert stats["row_count"] == 6


class TestMain:

    def test_writes_csv_stats_and_manifest(self, tmp_path):
        out = tmp_path / "subsets.csv"
        args = _make_args(["PF00018"], STO_PATH, str(out))
        main(args)

        assert out.exists()
        stats_path = tmp_path / "subsets.stats.md"
        manifest_path = tmp_path / "subsets.build_manifest.json"
        assert stats_path.exists()
        assert manifest_path.exists()

        with open(manifest_path) as f:
            manifest = json.load(f)
        assert manifest["outputs"]["row_counts"]["pfam_annotated_subsets"] == 3
        assert manifest["outputs"]["pfam_ids"] == ["PF00018"]
        assert "stats" in manifest and manifest["stats"]["row_count"] == 3

    def test_multi_family_manifest_row_counts(self, tmp_path):
        out = tmp_path / "combined.csv"
        args = _make_args(["PF00018", "PF01083", "PF04947"], STO_PATH, str(out))
        main(args)
        with open(tmp_path / "combined.build_manifest.json") as f:
            manifest = json.load(f)
        assert manifest["outputs"]["row_counts"]["pfam_annotated_subsets"] == 8

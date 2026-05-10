import argparse
import csv
import json
import os

import pytest

from biom3.dbio.builders.pfam_subsets import (
    OUTPUT_COLUMNS,
    _clean_sequence,
    build_annotated_pfam_subsets_csv,
    iter_annotated_pfam_rows,
    main,
)

DATDIR = os.path.join(os.path.dirname(__file__), "..", "..", "_data", "dbio")
STO_PATH = os.path.join(DATDIR, "mini_pfam_full.sto")


def _make_args(pfam_ids, pfam_full, output, chunk_size=100,
               per_pfam_output=False, outdir=None):
    return argparse.Namespace(
        pfam_ids=pfam_ids,
        pfam_full=pfam_full,
        output=output,
        chunk_size=chunk_size,
        per_pfam_output=per_pfam_output,
        outdir=outdir,
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
        mod_logger = logging.getLogger("biom3.dbio.builders.pfam_subsets")
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


class TestPerPfamOutput:
    """--per_pfam_output writes one CSV per Pfam ID in a single pass."""

    def test_writes_one_csv_per_id(self, tmp_path):
        outdir = tmp_path / "per_pfam"
        args = _make_args(
            ["PF00018", "PF01083"], STO_PATH, output=None,
            per_pfam_output=True, outdir=str(outdir),
        )
        main(args)
        assert (outdir / "PF00018.csv").exists()
        assert (outdir / "PF01083.csv").exists()
        # Both CSVs should be non-empty (header + at least one data row)
        for pid in ("PF00018", "PF01083"):
            with open(outdir / f"{pid}.csv") as f:
                lines = f.read().splitlines()
            assert len(lines) >= 2  # header + ≥1 row

    def test_row_pfam_label_matches_filename(self, tmp_path):
        """Per-family routing: every row in PF00018.csv has pfam_label=PF00018."""
        outdir = tmp_path / "per_pfam"
        args = _make_args(
            ["PF00018", "PF01083"], STO_PATH, output=None,
            per_pfam_output=True, outdir=str(outdir),
        )
        main(args)
        for pid in ("PF00018", "PF01083"):
            rows = _read_csv(outdir / f"{pid}.csv")
            assert all(r["pfam_label"] == pid for r in rows), (
                f"Cross-contamination in {pid}.csv: {[r['pfam_label'] for r in rows]}"
            )

    def test_emits_manifest_per_file(self, tmp_path):
        outdir = tmp_path / "per_pfam"
        args = _make_args(
            ["PF00018", "PF01083"], STO_PATH, output=None,
            per_pfam_output=True, outdir=str(outdir),
        )
        main(args)
        for pid in ("PF00018", "PF01083"):
            manifest_path = outdir / f"{pid}.build_manifest.json"
            stats_path = outdir / f"{pid}.stats.md"
            assert manifest_path.exists(), f"{manifest_path} not found"
            assert stats_path.exists(), f"{stats_path} not found"

            manifest = json.load(open(manifest_path))
            assert manifest["outputs"]["pfam_ids"] == [pid]
            assert manifest["outputs"]["row_counts"]["pfam_annotated_subsets"] >= 1

    def test_per_pfam_row_counts_match_single_output_total(self, tmp_path):
        """Per-Pfam mode should produce the same row population as -o, just split."""
        single_out = tmp_path / "single.csv"
        single_args = _make_args(["PF00018", "PF01083"], STO_PATH, str(single_out))
        main(single_args)
        single_rows = _read_csv(single_out)
        single_per_pid = {}
        for r in single_rows:
            single_per_pid.setdefault(r["pfam_label"], 0)
            single_per_pid[r["pfam_label"]] += 1

        per_pfam_dir = tmp_path / "per_pfam"
        per_args = _make_args(
            ["PF00018", "PF01083"], STO_PATH, output=None,
            per_pfam_output=True, outdir=str(per_pfam_dir),
        )
        main(per_args)
        for pid, expected in single_per_pid.items():
            split_rows = _read_csv(per_pfam_dir / f"{pid}.csv")
            assert len(split_rows) == expected, (
                f"{pid}: single-output had {expected} rows, "
                f"per-pfam had {len(split_rows)}"
            )


class TestArgparseOutputGroup:
    """CLI mutex + outdir validation."""

    def test_per_pfam_output_requires_outdir(self):
        from biom3.dbio.builders.pfam_subsets import parse_arguments
        with pytest.raises(SystemExit):
            parse_arguments([
                "-p", "PF00018",
                "--pfam_full", STO_PATH,
                "--per_pfam_output",
                # no --outdir
            ])

    def test_output_and_per_pfam_mutually_exclusive(self):
        from biom3.dbio.builders.pfam_subsets import parse_arguments
        with pytest.raises(SystemExit):
            parse_arguments([
                "-p", "PF00018",
                "--pfam_full", STO_PATH,
                "-o", "/tmp/x.csv",
                "--per_pfam_output",
                "--outdir", "/tmp/per_pfam",
            ])

    def test_neither_output_nor_per_pfam_errors(self):
        """Mutex group is required=True, so omitting both raises."""
        from biom3.dbio.builders.pfam_subsets import parse_arguments
        with pytest.raises(SystemExit):
            parse_arguments([
                "-p", "PF00018",
                "--pfam_full", STO_PATH,
            ])

    def test_single_output_path_parses_cleanly(self):
        from biom3.dbio.builders.pfam_subsets import parse_arguments
        args = parse_arguments([
            "-p", "PF00018",
            "--pfam_full", STO_PATH,
            "-o", "/tmp/x.csv",
        ])
        assert args.output == "/tmp/x.csv"
        assert args.per_pfam_output is False
        assert args.outdir is None

    def test_per_pfam_output_with_outdir_parses_cleanly(self):
        from biom3.dbio.builders.pfam_subsets import parse_arguments
        args = parse_arguments([
            "-p", "PF00018",
            "--pfam_full", STO_PATH,
            "--per_pfam_output",
            "--outdir", "/tmp/per_pfam",
        ])
        assert args.output is None
        assert args.per_pfam_output is True
        assert args.outdir == "/tmp/per_pfam"

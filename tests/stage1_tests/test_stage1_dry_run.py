"""Integration tests for ``--dry_run`` on biom3_train_stage1."""

import json
import os

import pytest

from tests.conftest import DATDIR, TMPDIR, get_args

from biom3.Stage1.run_PL_training import parse_arguments, main


pytestmark = [pytest.mark.slow]

ARGS_DIR = os.path.join(DATDIR, "entrypoint_args", "training")


def _stage1_argv():
    return get_args(f"{ARGS_DIR}/stage1_training_args_scratch_v1.txt")


def test_dry_run_stage1_stdout_only(capsys, tmp_path):
    argv = _stage1_argv() + [
        "--dry_run", "True",
        "--output_root", str(tmp_path / "outputs"),
        "--run_id", "stage1_dryrun_stdout",
    ]
    args = parse_arguments(argv)
    rc = main(args)
    out = capsys.readouterr().out
    assert rc == 0
    assert "DRY RUN" in out
    assert "[Effective configuration]" in out
    assert "[Output paths the run would create]" in out
    assert "[Memory estimate]" in out
    assert not (tmp_path / "outputs" / "runs").exists()


def test_dry_run_stage1_writes_to_artifacts(tmp_path):
    argv = _stage1_argv() + [
        "--dry_run", "True",
        "--dry_run_output", "True",
        "--output_root", str(tmp_path / "outputs"),
        "--run_id", "stage1_dryrun_artifacts",
    ]
    args = parse_arguments(argv)
    rc = main(args)
    assert rc == 0
    report = tmp_path / "outputs" / "runs" / "stage1_dryrun_artifacts" \
        / "artifacts" / "dry_run_report.json"
    assert report.exists()
    data = json.loads(report.read_text())
    assert data["stage"] == "stage1"
    # Stage 1 single-device is the expected strategy at devices_per_node=1
    assert data["memory_estimate"]["strategy"] in ("ddp", "single_device")

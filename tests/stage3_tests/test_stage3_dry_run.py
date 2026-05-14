"""Integration tests for ``--dry_run`` on biom3_train_stage3."""

import json
import os

import pytest

from tests.conftest import DATDIR, TMPDIR, get_args

from biom3.Stage3.run_PL_training import parse_arguments, main


pytestmark = [pytest.mark.slow]

ARGS_DIR = os.path.join(DATDIR, "entrypoint_args", "training")


def _stage3_argv():
    """Return CLI args for a minimal Stage 3 dry-run.

    Resolves data_root path relative to DATDIR.
    """
    argv = get_args(f"{ARGS_DIR}/training_args_scratch_v1.txt")
    # Resolve --swissprot_data_root path against DATDIR (the entrypoint args
    # use a relative path).
    new = []
    skip = False
    for i, tok in enumerate(argv):
        if skip:
            skip = False
            continue
        if tok == "--swissprot_data_root":
            new.append(tok)
            new.append(os.path.join(DATDIR, argv[i + 1]))
            skip = True
        else:
            new.append(tok)
    return new


def test_dry_run_stage3_stdout_only(capsys, tmp_path):
    argv = _stage3_argv() + [
        "--dry_run", "True",
        "--output_root", str(tmp_path / "outputs"),
        "--run_id", "stage3_dryrun_stdout",
    ]
    args = parse_arguments(argv)
    rc = main(args)
    out = capsys.readouterr().out
    assert rc == 0
    # Stdout sections are present
    assert "DRY RUN" in out
    assert "[Effective configuration]" in out
    assert "[Output paths the run would create]" in out
    assert "[Distributed / batch math]" in out
    assert "[Memory estimate]" in out
    # No JSON written
    assert not (tmp_path / "outputs" / "runs").exists()
    assert not (tmp_path / "outputs" / "history").exists()


def test_dry_run_stage3_writes_to_artifacts(tmp_path):
    argv = _stage3_argv() + [
        "--dry_run", "True",
        "--dry_run_output", "True",
        "--output_root", str(tmp_path / "outputs"),
        "--run_id", "stage3_dryrun_artifacts",
    ]
    args = parse_arguments(argv)
    rc = main(args)
    assert rc == 0
    report = tmp_path / "outputs" / "runs" / "stage3_dryrun_artifacts" \
        / "artifacts" / "dry_run_report.json"
    assert report.exists()
    data = json.loads(report.read_text())
    assert data["stage"] == "stage3"
    assert "effective_config" in data
    assert "output_paths" in data
    assert "memory_estimate" in data
    assert data["memory_estimate"]["strategy"] == "deepspeed_zero2"


def test_dry_run_stage3_distributed_strategy_ddp(tmp_path):
    """Selecting --distributed_strategy ddp flips the report's strategy
    field and the per-rank memory math to full replication (no
    world_size sharding for optimizer states)."""
    argv = _stage3_argv() + [
        "--dry_run", "True",
        "--dry_run_output", "True",
        "--distributed_strategy", "ddp",
        "--devices_per_node", "4",
        "--num_nodes", "2",
        "--output_root", str(tmp_path / "outputs"),
        "--run_id", "stage3_dryrun_ddp",
    ]
    args = parse_arguments(argv)
    rc = main(args)
    assert rc == 0
    report = tmp_path / "outputs" / "runs" / "stage3_dryrun_ddp" \
        / "artifacts" / "dry_run_report.json"
    data = json.loads(report.read_text())
    mem = data["memory_estimate"]
    assert mem["strategy"] == "ddp"
    # DDP: optimizer state is replicated, not sharded across world_size=8.
    assert mem["per_rank_optimizer_gb"] == pytest.approx(
        (mem["total_params"] * 8) / (1024 ** 3)
    )


def test_dry_run_stage3_writes_to_custom_path(tmp_path):
    custom = tmp_path / "preflight.json"
    argv = _stage3_argv() + [
        "--dry_run", "True",
        "--dry_run_output", str(custom),
        "--output_root", str(tmp_path / "outputs"),
        "--run_id", "stage3_dryrun_custom",
    ]
    args = parse_arguments(argv)
    rc = main(args)
    assert rc == 0
    assert custom.exists()
    artifacts_report = tmp_path / "outputs" / "runs" / "stage3_dryrun_custom" \
        / "artifacts" / "dry_run_report.json"
    assert not artifacts_report.exists()

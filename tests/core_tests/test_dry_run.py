"""Tests for biom3.core.dry_run."""

import argparse
import io
import json
import os
from contextlib import redirect_stdout
from types import SimpleNamespace

import pytest
import torch

from biom3.core.dry_run import (
    batch_footprint_bytes,
    build_args_table,
    coerce_dry_run_output,
    detect_cli_keys,
    infer_strategy,
    render_distributed_summary,
    render_memory_estimate,
    render_paths,
    resolve_dry_run_output,
    run_dry_run,
    write_report,
)


# ---------- detect_cli_keys / build_args_table ----------

def _build_test_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--name", type=str, default="default_name")
    return parser


def test_detect_cli_keys_picks_only_explicit():
    cli_keys = detect_cli_keys(["lr", "batch_size", "name"], ["--lr", "0.5"])
    assert cli_keys == {"lr"}


def test_detect_cli_keys_handles_equals_form():
    cli_keys = detect_cli_keys(["lr", "name"], ["--lr=0.5", "--name", "x"])
    assert cli_keys == {"lr", "name"}


def test_detect_cli_keys_empty_argv():
    assert detect_cli_keys(["lr"], []) == set()


def test_build_args_table_cli_overrides_json(tmp_path):
    config_path = tmp_path / "c.json"
    config_path.write_text(json.dumps({"lr": 0.01, "batch_size": 64}))
    parser = _build_test_parser()
    parser.set_defaults(lr=0.01, batch_size=64)  # mimic the JSON-as-defaults pattern
    argv = ["--name", "alpha"]
    args = parser.parse_args(argv)
    table = build_args_table(args, argv, str(config_path))
    sources = {k: s for (k, _, s) in table}
    assert sources["name"] == "CLI"
    assert sources["lr"].startswith("JSON: ") and sources["lr"].endswith("c.json")
    assert sources["batch_size"].startswith("JSON: ")


def test_build_args_table_no_config(tmp_path):
    parser = _build_test_parser()
    argv = ["--lr", "0.5"]
    args = parser.parse_args(argv)
    table = build_args_table(args, argv, None)
    sources = {k: s for (k, _, s) in table}
    assert sources["lr"] == "CLI"
    assert sources["batch_size"] == "default"
    assert sources["name"] == "default"


# ---------- render_paths ----------

def test_render_paths_creates_no_directories(tmp_path):
    args = SimpleNamespace(
        output_root=str(tmp_path / "out"),
        runs_folder="runs",
        checkpoints_folder="checkpoints",
        run_id="run_xyz",
    )
    paths = render_paths(args, stage="stage3", dry_run_report_path=None)
    assert paths["run_dir"].endswith(os.path.join("out", "runs", "run_xyz"))
    assert paths["checkpoint_dir"].endswith(os.path.join("out", "checkpoints", "run_xyz"))
    assert paths["args_json"].endswith("args.json")
    assert paths["build_manifest"].endswith("build_manifest.json")
    # nothing should have been created on disk
    for p in paths.values():
        assert not os.path.exists(p), f"{p} should not exist"


def test_render_paths_includes_dry_run_report_when_given(tmp_path):
    args = SimpleNamespace(
        output_root=str(tmp_path / "out"),
        runs_folder="runs",
        checkpoints_folder="checkpoints",
        run_id="r",
    )
    paths = render_paths(args, stage="stage3",
                         dry_run_report_path="/tmp/foo.json")
    assert paths["dry_run_report"] == "/tmp/foo.json"


# ---------- render_distributed_summary ----------

def test_render_distributed_summary_math():
    args = SimpleNamespace(
        devices_per_node=4, num_nodes=2, batch_size=8, acc_grad_batches=2,
        training_strategy="primary_only", epochs=10,
    )
    s = render_distributed_summary(args, train_dataset_len=10_000,
                                   val_dataset_len=1_000)
    assert s["world_size"] == 8
    assert s["effective_batch_size"] == 8 * 8 * 2
    assert s["batches_per_epoch_per_rank"] == (10_000 // 8) // 8
    assert s["steps_per_epoch"] == s["batches_per_epoch_per_rank"] // 2
    assert s["epochs"] == 10
    assert s["training_strategy"] == "primary_only"


def test_render_distributed_summary_handles_none_dataset():
    args = SimpleNamespace(devices_per_node=1, num_nodes=1, batch_size=1, acc_grad_batches=1)
    s = render_distributed_summary(args, None, None)
    assert s["train_dataset_len"] is None
    assert "steps_per_epoch" not in s


# ---------- batch_footprint_bytes ----------

def test_batch_footprint_bytes_recursive():
    batch = {
        "x": torch.zeros(4, 16, dtype=torch.float32),  # 4*16*4 = 256
        "y": [torch.zeros(2, dtype=torch.int64), torch.zeros(3, dtype=torch.int8)],
        # 2*8 + 3*1 = 19
        "meta": "ignored",
    }
    assert batch_footprint_bytes(batch) == 256 + 16 + 3


def test_batch_footprint_bytes_handles_none():
    assert batch_footprint_bytes(None) == 0


# ---------- infer_strategy ----------

def test_infer_strategy_stage3_is_zero2():
    args = SimpleNamespace(device="cuda", devices_per_node=1, num_nodes=1)
    assert infer_strategy(args, "stage3") == "deepspeed_zero2"


def test_infer_strategy_stage1_multi_gpu_is_ddp():
    args = SimpleNamespace(device="cuda", devices_per_node=4, num_nodes=1)
    assert infer_strategy(args, "stage1") == "ddp"


def test_infer_strategy_stage1_single_xpu_is_single_device():
    args = SimpleNamespace(device="xpu", devices_per_node=1, num_nodes=1)
    assert infer_strategy(args, "stage1") == "single_device"


# ---------- render_memory_estimate ----------

class _Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(10, 10, bias=False)  # 100 params


def test_render_memory_estimate_ddp_full_replication():
    out = render_memory_estimate(
        _Tiny(), sample_batch=None, strategy="ddp",
        devices_per_node=4, num_nodes=2,
    )
    assert out["total_params"] == 100
    # full replication: 4N + 4N + 8N bytes (fp32 AdamW)
    expected = (100 * 4 + 100 * 4 + 100 * 8) / (1024 ** 3)
    assert abs(out["per_rank_static_total_gb"] - expected) < 1e-12


def test_render_memory_estimate_zero2_shards_optimizer():
    out = render_memory_estimate(
        _Tiny(), sample_batch=None, strategy="deepspeed_zero2",
        devices_per_node=4, num_nodes=2,
    )
    # ZeRO-2: optimizer/grad sharded across world_size=8, params replicated
    assert out["per_rank_optimizer_gb"] == pytest.approx(
        (100 * 8) / 8 / (1024 ** 3)
    )
    assert out["per_rank_grad_gb"] == pytest.approx(
        (100 * 4) / 8 / (1024 ** 3)
    )
    assert out["per_rank_param_gb"] == pytest.approx(
        (100 * 4) / (1024 ** 3)
    )
    assert "deepspeed_breakdown" in out


def test_render_memory_estimate_no_model():
    out = render_memory_estimate(
        None, sample_batch=None, strategy="ddp", devices_per_node=1, num_nodes=1,
    )
    assert "error" in out


def test_render_memory_estimate_with_sample_batch():
    sample = torch.zeros(2, 16, dtype=torch.float32)  # 128 bytes
    out = render_memory_estimate(
        _Tiny(), sample_batch=sample, strategy="single_device",
        devices_per_node=1, num_nodes=1,
    )
    assert out["per_minibatch_input_bytes"] == 128


# ---------- coerce_dry_run_output / resolve_dry_run_output ----------

@pytest.mark.parametrize("value,expected", [
    (False, False),
    (True, True),
    ("True", True),
    ("true", True),
    ("False", False),
    ("false", False),
    ("", False),
    ("None", False),
    ("./report.json", "./report.json"),
    ("/tmp/r.json", "/tmp/r.json"),
])
def test_coerce_dry_run_output(value, expected):
    assert coerce_dry_run_output(value) == expected


def test_resolve_dry_run_output_false_returns_none():
    args = SimpleNamespace(dry_run_output=False)
    assert resolve_dry_run_output(args, {"artifacts_dir": "/tmp/a"}) is None


def test_resolve_dry_run_output_true_uses_artifacts_dir():
    args = SimpleNamespace(dry_run_output=True)
    out = resolve_dry_run_output(args, {"artifacts_dir": "/tmp/a"})
    assert out == "/tmp/a/dry_run_report.json"


def test_resolve_dry_run_output_string_is_path(tmp_path):
    args = SimpleNamespace(dry_run_output=str(tmp_path / "foo.json"))
    out = resolve_dry_run_output(args, {})
    assert out == os.path.abspath(str(tmp_path / "foo.json"))


# ---------- write_report ----------

def test_write_report_skipped_when_path_none(tmp_path):
    write_report({"a": 1}, None)
    assert list(tmp_path.iterdir()) == []


def test_write_report_creates_file(tmp_path):
    target = tmp_path / "x" / "r.json"
    write_report({"a": 1}, str(target))
    assert target.exists()
    assert json.loads(target.read_text()) == {"a": 1}


def test_write_report_backs_up_existing(tmp_path):
    target = tmp_path / "r.json"
    write_report({"v": 1}, str(target))
    write_report({"v": 2}, str(target))
    assert json.loads(target.read_text()) == {"v": 2}
    backups = [p for p in tmp_path.iterdir() if ".bak." in p.name]
    assert len(backups) == 1


# ---------- run_dry_run end-to-end (no model/data) ----------

def test_run_dry_run_stdout_only(tmp_path, capsys):
    parser = _build_test_parser()
    parser.add_argument("--output_root", default=str(tmp_path / "out"))
    parser.add_argument("--runs_folder", default="runs")
    parser.add_argument("--checkpoints_folder", default="checkpoints")
    parser.add_argument("--run_id", default="r")
    parser.add_argument("--devices_per_node", default=1, type=int)
    parser.add_argument("--num_nodes", default=1, type=int)
    parser.add_argument("--acc_grad_batches", default=1, type=int)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dry_run_output", default=False)
    args = parser.parse_args([])

    rc = run_dry_run(args, stage="stage3", argv=[],
                     dataset_probe=None, model_probe=None)
    captured = capsys.readouterr()
    assert rc == 0
    assert "DRY RUN" in captured.out
    assert "[Effective configuration]" in captured.out
    # No JSON should have been written
    assert not (tmp_path / "out").exists()


def test_run_dry_run_writes_json_when_requested(tmp_path, capsys):
    parser = _build_test_parser()
    parser.add_argument("--output_root", default=str(tmp_path / "out"))
    parser.add_argument("--runs_folder", default="runs")
    parser.add_argument("--checkpoints_folder", default="checkpoints")
    parser.add_argument("--run_id", default="r")
    parser.add_argument("--devices_per_node", default=1, type=int)
    parser.add_argument("--num_nodes", default=1, type=int)
    parser.add_argument("--acc_grad_batches", default=1, type=int)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dry_run_output", default=True)
    args = parser.parse_args([])

    rc = run_dry_run(args, stage="stage3", argv=[],
                     dataset_probe=None, model_probe=None)
    assert rc == 0
    report = tmp_path / "out" / "runs" / "r" / "artifacts" / "dry_run_report.json"
    assert report.exists()
    data = json.loads(report.read_text())
    assert data["stage"] == "stage3"
    assert "effective_config" in data
    assert "output_paths" in data
    assert "memory_estimate" in data

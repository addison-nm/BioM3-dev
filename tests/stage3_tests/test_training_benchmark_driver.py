"""Tests for the Stage 3 training benchmark sweep driver (pure helpers).

The driver itself invokes ``biom3_pretrain_stage3`` in a subprocess per
sweep cell, which needs weights; those are out of scope here. We cover the
pure-Python helpers: config validation, cell enumeration, cell_id
formatting, and summary aggregation from synthetic benchmark outputs.
"""

import json
import os

import pytest

from biom3.benchmarks.Stage3 import training as driver


def test_validate_config_requires_sweep():
    with pytest.raises(ValueError, match="sweep"):
        driver._validate_config({
            "base_training_config_path": "x.json",
            "output_root": "out/",
        })


def test_validate_config_rejects_unknown_axis():
    with pytest.raises(ValueError, match="unknown sweep axis"):
        driver._validate_config({
            "base_training_config_path": "x.json",
            "output_root": "out/",
            "sweep": {"nonsense": [1]},
        })


def test_validate_config_requires_nonempty_axis():
    with pytest.raises(ValueError, match="non-empty"):
        driver._validate_config({
            "base_training_config_path": "x.json",
            "output_root": "out/",
            "sweep": {"batch_size": []},
        })


def test_enumerate_cells_cartesian_product():
    cells = list(driver._enumerate_cells({
        "batch_size": [8, 16],
        "precision": ["32", "bf16"],
    }))
    assert len(cells) == 4
    assert {
        (c["batch_size"], c["precision"]) for c in cells
    } == {(8, "32"), (8, "bf16"), (16, "32"), (16, "bf16")}


def test_cell_id_stable_and_sortable():
    a = driver._cell_id({"batch_size": 16, "precision": "bf16"})
    b = driver._cell_id({"batch_size": 16, "precision": "bf16"})
    assert a == b
    assert "16" in a and "bf16" in a


def test_summarize_from_history_and_steps():
    history = {
        "epochs": [
            {"epoch_wall_time_sec": 5.0, "samples_per_sec": 10.0,
             "steps_per_sec": 1.0, "peak_memory_allocated_gb": 3.0,
             "peak_memory_reserved_gb": 5.0},
            {"epoch_wall_time_sec": 4.0, "samples_per_sec": 12.0,
             "steps_per_sec": 1.5, "peak_memory_allocated_gb": 3.5,
             "peak_memory_reserved_gb": 5.5},
            {"epoch_wall_time_sec": 4.2, "samples_per_sec": 11.0,
             "steps_per_sec": 1.4, "peak_memory_allocated_gb": 3.2,
             "peak_memory_reserved_gb": 5.2},
        ],
    }
    step_times = [0.5, 0.2, 0.21, 0.22, 0.23, 0.20]

    summary = driver._summarize(history, step_times, skip_first_epoch=True)

    # First epoch dropped
    assert summary["epochs_observed"] == 2
    assert summary["mean_epoch_wall_time_sec"] == pytest.approx((4.0 + 4.2) / 2)
    assert summary["peak_memory_allocated_gb"] == pytest.approx(3.5)
    assert summary["peak_memory_reserved_gb"] == pytest.approx(5.5)

    # First step is dropped as warmup when there's more than one sample
    assert summary["steps_observed"] == len(step_times)
    assert summary["mean_step_wall_time_sec"] == pytest.approx(
        sum(step_times[1:]) / len(step_times[1:])
    )


def test_summarize_handles_empty_history():
    summary = driver._summarize(history=None, step_times=[],
                                skip_first_epoch=True)
    assert summary["epochs_observed"] == 0
    assert summary["mean_epoch_wall_time_sec"] is None
    assert summary["steps_observed"] == 0
    assert summary["mean_step_wall_time_sec"] is None


def test_load_cell_metrics_from_disk(tmp_path):
    artifacts = tmp_path / "runs" / "myrun" / "artifacts"
    artifacts.mkdir(parents=True)
    history = {"epochs": [{"epoch_wall_time_sec": 1.0, "samples_per_sec": 50.0,
                           "steps_per_sec": 2.0,
                           "peak_memory_allocated_gb": 1.0,
                           "peak_memory_reserved_gb": 2.0}]}
    (artifacts / "benchmark_history.json").write_text(json.dumps(history))
    with open(artifacts / "benchmark_steps.jsonl", "w") as fh:
        fh.write(json.dumps({"global_step": 1, "step_wall_time_sec": 0.1}) + "\n")
        fh.write(json.dumps({"global_step": 2, "step_wall_time_sec": 0.12}) + "\n")

    loaded_history, step_times, artifacts_dir = driver._load_cell_metrics(
        str(tmp_path), "myrun", cfg={},
    )
    assert loaded_history == history
    assert step_times == [0.1, 0.12]
    assert os.path.samefile(artifacts_dir, str(artifacts))


def test_build_training_cmd_includes_overrides():
    cfg = {
        "base_training_config_path": "configs/stage3_training/x.json",
        "benchmark_per_step": False,
        "training_extra_args": {"epochs": 2, "wandb": "False"},
    }
    cmd = driver._build_training_cmd(
        cfg, cell={"batch_size": 16, "precision": "bf16"},
        cell_output_root="/tmp/cell", run_id="r1",
    )
    assert cmd[0] == "biom3_pretrain_stage3"
    # Cell sweep values are passed as CLI overrides
    assert "--batch_size" in cmd and "16" in cmd
    assert "--precision" in cmd and "bf16" in cmd
    # Extra args are passed through
    assert "--epochs" in cmd and "2" in cmd
    assert "--wandb" in cmd and "False" in cmd
    # Benchmark flags come from cfg
    assert "--save_benchmark" in cmd
    assert "--benchmark_per_step" in cmd

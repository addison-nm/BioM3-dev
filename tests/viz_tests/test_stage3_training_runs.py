import json
from pathlib import Path

import numpy as np
import pytest
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.figure
import matplotlib.pyplot as plt

from biom3.viz.stage3_training_runs import (
    load_run_artifacts,
    discover_metric_bases,
    plot_metric,
    plot_benchmark_history,
)


def _make_train_metrics(n_steps=20, n_epochs=4):
    rng = np.random.default_rng(0)
    global_step = np.arange(1, n_steps + 1, dtype=np.int64)
    epoch = np.repeat(np.arange(n_epochs), n_steps // n_epochs).astype(np.int64)
    if len(epoch) < n_steps:
        epoch = np.concatenate([epoch, np.full(n_steps - len(epoch), n_epochs - 1)])

    # Step-level loss (dense)
    train_loss_step = rng.uniform(0.1, 0.3, n_steps)
    # Bare form equals step form (as produced by MetricsHistoryCallback)
    train_loss = train_loss_step.copy()

    # Epoch-level loss (one value per unique epoch, but the array length in the
    # callback matches the record count — it's simplest to reuse step-length
    # arrays for test synthesis since that's how records accumulate)
    train_loss_epoch = rng.uniform(0.1, 0.3, n_steps)
    train_hard_acc_step = rng.uniform(0.5, 0.9, n_steps)
    train_hard_acc_epoch = rng.uniform(0.5, 0.9, n_steps)

    return {
        "global_step": global_step,
        "epoch": epoch,
        "train_loss": train_loss,
        "train_loss_step": train_loss_step,
        "train_loss_epoch": train_loss_epoch,
        "train_prev_hard_acc_step": train_hard_acc_step,
        "train_prev_hard_acc_epoch": train_hard_acc_epoch,
    }


def _make_val_metrics(n_epochs=4, include_nan=False):
    rng = np.random.default_rng(1)
    # Val is recorded once per validation epoch — arrays are length n_epochs.
    global_step = np.array([13, 26, 39, 52][:n_epochs], dtype=np.int64)
    epoch = np.arange(n_epochs, dtype=np.int64)
    val_loss = rng.uniform(0.1, 0.3, n_epochs)
    val_loss_epoch = val_loss.copy()
    val_acc_epoch = rng.uniform(0.5, 0.9, n_epochs)
    loss_gap = rng.uniform(-0.05, 0.05, n_epochs)
    if include_nan and n_epochs > 1:
        val_loss[0] = np.nan
        val_loss_epoch[0] = np.nan

    return {
        "global_step": global_step,
        "epoch": epoch,
        "val_loss": val_loss,
        "val_loss_epoch": val_loss_epoch,
        "val_prev_hard_acc_epoch": val_acc_epoch,
        "loss_gap": loss_gap,
    }


def _write_metrics_history(artifacts_dir: Path, train=None, val=None):
    payload = {"train": train or {}, "val": val or {}}
    torch.save(payload, artifacts_dir / "metrics_history.pt")


def _write_rank_file(artifacts_dir: Path, rank: int, val_dict: dict):
    torch.save({"val": val_dict}, artifacts_dir / f"metrics_history.rank{rank:03d}.pt")


@pytest.fixture
def artifacts_dir(tmp_path):
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()
    _write_metrics_history(
        artifacts_dir,
        train=_make_train_metrics(),
        val=_make_val_metrics(),
    )
    # Sibling metadata
    (artifacts_dir / "args.json").write_text(json.dumps({
        "lr": 1e-4, "batch_size": 16, "epochs": 4,
    }))
    (artifacts_dir / "checkpoint_summary.json").write_text(json.dumps({
        "primary": {"metric": "val_loss", "best_score": 0.12, "best_path": "/tmp/best.pth"},
        "additional": [],
    }))
    (artifacts_dir / "build_manifest.json").write_text(json.dumps({
        "git_hash": "abc123", "git_branch": "addison-dev", "outputs": {"seed": 42}
    }))
    (artifacts_dir / "benchmark_history.json").write_text(json.dumps({
        "config": {"batch_size": 16, "gpu_devices": 12, "num_nodes": 8, "backend": "xpu"},
        "epochs": [
            {"epoch": 0, "samples_per_sec": 1200.0, "steps_per_sec": 18.0,
             "peak_memory_allocated_gb": 40.1, "peak_memory_reserved_gb": 45.3,
             "peak_memory_allocated_gb_per_rank": [40.1, 40.0, 39.9, 40.2]},
            {"epoch": 1, "samples_per_sec": 1500.0, "steps_per_sec": 22.0,
             "peak_memory_allocated_gb": 41.2, "peak_memory_reserved_gb": 46.1,
             "peak_memory_allocated_gb_per_rank": [41.1, 41.0, 41.3, 41.2]},
        ],
    }))
    return artifacts_dir


class TestLoadRunArtifacts:
    def test_basic_structure(self, artifacts_dir):
        run = load_run_artifacts(artifacts_dir)
        assert run["available_ranks"] == []
        assert run["per_rank_val_loss"] == {}
        assert "loss" in run["metrics"]["train"]
        assert "loss" in run["metrics"]["val"]
        assert "prev_hard_acc" in run["metrics"]["train"]
        assert "prev_hard_acc" in run["metrics"]["val"]
        assert "loss_gap" in run["metrics"]["val"]
        # sibling jsons parsed
        assert run["args"]["lr"] == 1e-4
        assert run["checkpoint_summary"]["primary"]["best_score"] == 0.12
        assert run["build_manifest"]["git_hash"] == "abc123"
        assert run["benchmark_history"]["config"]["gpu_devices"] == 12

    def test_missing_metrics_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_run_artifacts(tmp_path)

    def test_with_ranks(self, artifacts_dir):
        _write_rank_file(artifacts_dir, 0, {
            "global_step": np.array([13, 26, 39, 52], dtype=np.int64),
            "epoch": np.arange(4, dtype=np.int64),
            "val_loss": np.array([0.25, 0.2, 0.18, 0.17]),
            "val_loss_epoch": np.array([0.25, 0.2, 0.18, 0.17]),
            "rank": np.zeros(4, dtype=np.int64),
        })
        _write_rank_file(artifacts_dir, 1, {
            "global_step": np.array([13, 26, 39, 52], dtype=np.int64),
            "epoch": np.arange(4, dtype=np.int64),
            "val_loss": np.array([0.24, 0.19, 0.17, 0.16]),
            "val_loss_epoch": np.array([0.24, 0.19, 0.17, 0.16]),
            "rank": np.ones(4, dtype=np.int64),
        })
        run = load_run_artifacts(artifacts_dir)
        assert run["available_ranks"] == [0, 1]
        assert "by_step" in run["per_rank_val_loss"][0]
        assert "by_epoch" in run["per_rank_val_loss"][0]

    def test_val_has_step_axis(self, artifacts_dir):
        """Val has no _step metrics, but global_step is recorded so by_step populates."""
        run = load_run_artifacts(artifacts_dir)
        val_loss_series = run["metrics"]["val"]["loss"]
        assert "by_step" in val_loss_series
        x, y = val_loss_series["by_step"]
        assert len(x) == len(y)
        assert len(x) > 0


class TestDiscoverMetricBases:
    def test_returns_sorted_lists(self, artifacts_dir):
        run = load_run_artifacts(artifacts_dir)
        bases = discover_metric_bases(run)
        assert bases["train"] == sorted(bases["train"])
        assert bases["val"] == sorted(bases["val"])
        assert "loss" in bases["train"]
        assert "loss" in bases["val"]
        assert "loss_gap" in bases["val"]


class TestPlotMetric:
    def test_returns_figure(self, artifacts_dir):
        run = load_run_artifacts(artifacts_dir)
        fig = plot_metric(run, "loss")
        assert isinstance(fig, matplotlib.figure.Figure)
        assert len(fig.axes[0].get_lines()) >= 2  # train + val
        plt.close(fig)

    def test_val_sparse_on_step_axis(self, artifacts_dir):
        run = load_run_artifacts(artifacts_dir)
        fig = plot_metric(run, "loss", x_axis="step", splits=("val",))
        assert isinstance(fig, matplotlib.figure.Figure)
        lines = fig.axes[0].get_lines()
        assert len(lines) == 1
        x_data = lines[0].get_xdata()
        # Val uses global_step markers, not dense per-step
        assert len(x_data) <= 10
        plt.close(fig)

    def test_epoch_axis(self, artifacts_dir):
        run = load_run_artifacts(artifacts_dir)
        fig = plot_metric(run, "loss", x_axis="epoch")
        assert fig.axes[0].get_xlabel() == "Epoch"
        plt.close(fig)

    def test_log_y(self, artifacts_dir):
        run = load_run_artifacts(artifacts_dir)
        fig = plot_metric(run, "loss", log_y=True)
        assert fig.axes[0].get_yscale() == "log"
        plt.close(fig)

    def test_non_finite_masked(self, tmp_path):
        artifacts_dir = tmp_path / "artifacts"
        artifacts_dir.mkdir()
        _write_metrics_history(
            artifacts_dir,
            train=_make_train_metrics(),
            val=_make_val_metrics(include_nan=True),
        )
        run = load_run_artifacts(artifacts_dir)
        fig = plot_metric(run, "loss")
        val_line = [l for l in fig.axes[0].get_lines() if "val" in l.get_label()][0]
        # NaN value at epoch 0 should have been masked before plotting
        assert not np.any(np.isnan(val_line.get_ydata()))
        plt.close(fig)

    def test_missing_metric_no_error(self, artifacts_dir):
        run = load_run_artifacts(artifacts_dir)
        fig = plot_metric(run, "nonexistent_metric")
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)

    def test_rank_overlay_for_loss(self, artifacts_dir):
        _write_rank_file(artifacts_dir, 0, {
            "global_step": np.array([13, 26, 39, 52], dtype=np.int64),
            "epoch": np.arange(4, dtype=np.int64),
            "val_loss": np.array([0.25, 0.2, 0.18, 0.17]),
            "val_loss_epoch": np.array([0.25, 0.2, 0.18, 0.17]),
            "rank": np.zeros(4, dtype=np.int64),
        })
        _write_rank_file(artifacts_dir, 1, {
            "global_step": np.array([13, 26, 39, 52], dtype=np.int64),
            "epoch": np.arange(4, dtype=np.int64),
            "val_loss": np.array([0.24, 0.19, 0.17, 0.16]),
            "val_loss_epoch": np.array([0.24, 0.19, 0.17, 0.16]),
            "rank": np.ones(4, dtype=np.int64),
        })
        run = load_run_artifacts(artifacts_dir)
        fig = plot_metric(run, "loss", ranks=[0, 1])
        labels = [l.get_label() for l in fig.axes[0].get_lines()]
        assert any("rank 0" in l for l in labels)
        assert any("rank 1" in l for l in labels)
        plt.close(fig)

    def test_rank_overlay_ignored_for_non_loss(self, artifacts_dir):
        _write_rank_file(artifacts_dir, 0, {
            "global_step": np.array([13, 26, 39, 52], dtype=np.int64),
            "epoch": np.arange(4, dtype=np.int64),
            "val_loss": np.array([0.25, 0.2, 0.18, 0.17]),
            "val_loss_epoch": np.array([0.25, 0.2, 0.18, 0.17]),
            "rank": np.zeros(4, dtype=np.int64),
        })
        run = load_run_artifacts(artifacts_dir)
        fig = plot_metric(run, "prev_hard_acc", ranks=[0])
        labels = [l.get_label() for l in fig.axes[0].get_lines()]
        assert not any("rank" in l for l in labels)
        plt.close(fig)


class TestPlotBenchmark:
    def test_throughput(self, artifacts_dir):
        run = load_run_artifacts(artifacts_dir)
        fig = plot_benchmark_history(run, kind="throughput")
        assert isinstance(fig, matplotlib.figure.Figure)
        # twin y-axis → 2 axes
        assert len(fig.axes) == 2
        plt.close(fig)

    def test_memory(self, artifacts_dir):
        run = load_run_artifacts(artifacts_dir)
        fig = plot_benchmark_history(run, kind="memory")
        labels = [l.get_label() for l in fig.axes[0].get_lines()]
        assert any("allocated" in l for l in labels)
        assert any("reserved" in l for l in labels)
        plt.close(fig)

    def test_memory_per_rank(self, artifacts_dir):
        run = load_run_artifacts(artifacts_dir)
        fig = plot_benchmark_history(run, kind="memory", per_rank=True)
        labels = [l.get_label() for l in fig.axes[0].get_lines()]
        assert any("rank 0" in l for l in labels)
        assert any("rank 3" in l for l in labels)
        plt.close(fig)

    def test_missing_benchmark_graceful(self, tmp_path):
        artifacts_dir = tmp_path / "artifacts"
        artifacts_dir.mkdir()
        _write_metrics_history(artifacts_dir, train=_make_train_metrics(), val=_make_val_metrics())
        run = load_run_artifacts(artifacts_dir)
        fig = plot_benchmark_history(run, kind="throughput")
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)

    def test_invalid_kind_raises(self, artifacts_dir):
        run = load_run_artifacts(artifacts_dir)
        with pytest.raises(ValueError):
            plot_benchmark_history(run, kind="bogus")

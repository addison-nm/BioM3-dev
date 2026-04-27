"""Tests for Stage 3 custom callbacks."""

import json
import os
import time
from unittest.mock import MagicMock, patch

import pytest
import torch

from biom3.Stage3.callbacks import (
    BestArtifactSyncCallback,
    LoggingModelCheckpoint,
    MetricsHistoryCallback,
    SyncSafeModelCheckpoint,
    TrainingBenchmarkCallback,
    build_checkpoint_callbacks,
    rebuild_metrics_history_pt,
)

TMPDIR = "tests/_tmp"


@pytest.fixture
def tmp_output_dir(tmp_path):
    return str(tmp_path / "artifacts")


def _make_callback(tmp_output_dir, **overrides):
    defaults = dict(
        output_dir=tmp_output_dir,
        batch_size=16,
        acc_grad_batches=4,
        gpu_devices=2,
        num_nodes=1,
        precision="bf16",
        training_strategy="primary_only",
        num_workers=0,
        skip_first_epoch=True,
        all_ranks_memory=False,
    )
    defaults.update(overrides)
    return TrainingBenchmarkCallback(**defaults)


def _mock_trainer(global_rank=0, global_step=0, current_epoch=0):
    trainer = MagicMock()
    trainer.global_rank = global_rank
    trainer.global_step = global_step
    trainer.current_epoch = current_epoch
    return trainer


def _mock_pl_module(all_gather_return=None):
    pl_module = MagicMock()
    pl_module.device = torch.device("cpu")
    if all_gather_return is not None:
        pl_module.all_gather = MagicMock(return_value=all_gather_return)
    return pl_module


class TestEffectiveBatchSize:

    def test_basic(self, tmp_output_dir):
        cb = _make_callback(tmp_output_dir, batch_size=16, acc_grad_batches=4,
                            gpu_devices=2, num_nodes=1)
        assert cb.effective_batch_size == 16 * 4 * 2 * 1

    def test_multinode(self, tmp_output_dir):
        cb = _make_callback(tmp_output_dir, batch_size=32, acc_grad_batches=1,
                            gpu_devices=4, num_nodes=2)
        assert cb.effective_batch_size == 32 * 1 * 4 * 2

    def test_single_device(self, tmp_output_dir):
        cb = _make_callback(tmp_output_dir, batch_size=8, acc_grad_batches=1,
                            gpu_devices=1, num_nodes=1)
        assert cb.effective_batch_size == 8


class TestBuildRecord:

    def test_record_fields(self, tmp_output_dir):
        cb = _make_callback(tmp_output_dir, batch_size=16, acc_grad_batches=4,
                            gpu_devices=2, num_nodes=1)
        record = cb._build_record(epoch=0, global_step=100, steps=100,
                                  elapsed=10.0)
        assert record["epoch"] == 0
        assert record["global_step"] == 100
        assert record["steps_in_epoch"] == 100
        assert record["samples_in_epoch"] == 100 * 128  # 16*4*2*1
        assert record["epoch_wall_time_sec"] == 10.0
        assert record["samples_per_sec"] == round(100 * 128 / 10.0, 1)
        assert record["steps_per_sec"] == round(100 / 10.0, 3)
        # Memory fields default to None when not provided
        assert record["peak_memory_allocated_gb"] is None
        assert record["peak_memory_reserved_gb"] is None

    def test_zero_elapsed(self, tmp_output_dir):
        cb = _make_callback(tmp_output_dir)
        record = cb._build_record(epoch=0, global_step=50, steps=50,
                                  elapsed=0.0)
        assert record["samples_per_sec"] == 0.0
        assert record["steps_per_sec"] == 0.0

    def test_interval_step_range(self, tmp_output_dir):
        cb = _make_callback(tmp_output_dir)
        record = cb._build_record(epoch=0, global_step=200, steps=100,
                                  elapsed=5.0,
                                  interval_step_range=[100, 200])
        assert record["interval_step_range"] == [100, 200]

    def test_no_interval_by_default(self, tmp_output_dir):
        cb = _make_callback(tmp_output_dir)
        record = cb._build_record(epoch=0, global_step=50, steps=50,
                                  elapsed=5.0)
        assert "interval_step_range" not in record

    def test_memory_fields_converted_to_gb(self, tmp_output_dir):
        cb = _make_callback(tmp_output_dir)
        one_gb = 1024 ** 3
        record = cb._build_record(
            epoch=0, global_step=100, steps=100, elapsed=10.0,
            peak_alloc=2 * one_gb, peak_reserved=4 * one_gb,
        )
        assert record["peak_memory_allocated_gb"] == 2.0
        assert record["peak_memory_reserved_gb"] == 4.0


class TestEpochHookSequence:

    def test_epoch_based_produces_records(self, tmp_output_dir):
        cb = _make_callback(tmp_output_dir, training_strategy="primary_only")
        trainer = _mock_trainer(global_step=0, current_epoch=0)
        pl_module = _mock_pl_module()

        cb.on_train_start(trainer, pl_module)
        for epoch in range(3):
            trainer.current_epoch = epoch
            cb.on_train_epoch_start(trainer, pl_module)
            trainer.global_step += 100
            cb.on_train_epoch_end(trainer, pl_module)

        assert len(cb._epoch_records) == 3
        assert cb._epoch_records[0]["epoch"] == 0
        assert cb._epoch_records[1]["epoch"] == 1
        assert cb._epoch_records[2]["epoch"] == 2
        for r in cb._epoch_records:
            assert r["steps_in_epoch"] == 100
            assert r["epoch_wall_time_sec"] >= 0

    def test_combine_strategy_uses_validation_intervals(self, tmp_output_dir):
        cb = _make_callback(tmp_output_dir, training_strategy="combine")
        trainer = _mock_trainer(global_step=0, current_epoch=0)
        pl_module = _mock_pl_module()

        cb.on_train_start(trainer, pl_module)
        cb.on_train_epoch_start(trainer, pl_module)

        # Simulate two validation intervals
        trainer.global_step = 500
        cb.on_validation_epoch_end(trainer, pl_module)

        trainer.global_step = 1000
        cb.on_validation_epoch_end(trainer, pl_module)

        assert len(cb._epoch_records) == 2
        assert cb._epoch_records[0]["interval_step_range"] == [0, 500]
        assert cb._epoch_records[1]["interval_step_range"] == [500, 1000]

    def test_combine_skips_zero_step_intervals(self, tmp_output_dir):
        cb = _make_callback(tmp_output_dir, training_strategy="combine")
        trainer = _mock_trainer(global_step=0, current_epoch=0)
        pl_module = _mock_pl_module()

        cb.on_train_start(trainer, pl_module)
        # Validation at step 0 (sanity check) should be skipped
        cb.on_validation_epoch_end(trainer, pl_module)
        assert len(cb._epoch_records) == 0

    def test_epoch_based_ignores_validation_hook(self, tmp_output_dir):
        cb = _make_callback(tmp_output_dir, training_strategy="primary_only")
        trainer = _mock_trainer(global_step=100, current_epoch=0)
        pl_module = _mock_pl_module()

        cb.on_train_start(trainer, pl_module)
        cb.on_validation_epoch_end(trainer, pl_module)
        assert len(cb._epoch_records) == 0


class TestMemoryTracking:

    def test_memory_captured_in_epoch_end(self, tmp_output_dir):
        cb = _make_callback(tmp_output_dir, training_strategy="primary_only")
        trainer = _mock_trainer(global_step=0, current_epoch=0)
        pl_module = _mock_pl_module()

        one_gb = 1024 ** 3
        with patch(
            "biom3.Stage3.callbacks._get_peak_memory_stats",
            return_value=(3 * one_gb, 5 * one_gb),
        ), patch(
            "biom3.Stage3.callbacks._reset_peak_memory_stats",
        ) as reset_mock:
            cb.on_train_start(trainer, pl_module)
            cb.on_train_epoch_start(trainer, pl_module)
            trainer.global_step = 100
            cb.on_train_epoch_end(trainer, pl_module)

        assert reset_mock.call_count >= 2  # on_train_start + on_train_epoch_start
        assert len(cb._epoch_records) == 1
        r = cb._epoch_records[0]
        assert r["peak_memory_allocated_gb"] == 3.0
        assert r["peak_memory_reserved_gb"] == 5.0

    def test_memory_none_on_cpu(self, tmp_output_dir):
        """When backend returns (None, None), memory fields should be None."""
        cb = _make_callback(tmp_output_dir, training_strategy="primary_only")
        trainer = _mock_trainer(global_step=0, current_epoch=0)
        pl_module = _mock_pl_module()

        with patch(
            "biom3.Stage3.callbacks._get_peak_memory_stats",
            return_value=(None, None),
        ), patch(
            "biom3.Stage3.callbacks._reset_peak_memory_stats",
        ):
            cb.on_train_start(trainer, pl_module)
            cb.on_train_epoch_start(trainer, pl_module)
            trainer.global_step = 100
            cb.on_train_epoch_end(trainer, pl_module)

        r = cb._epoch_records[0]
        assert r["peak_memory_allocated_gb"] is None
        assert r["peak_memory_reserved_gb"] is None

    def test_memory_in_saved_json(self, tmp_output_dir):
        cb = _make_callback(tmp_output_dir, training_strategy="primary_only")
        trainer = _mock_trainer(global_step=0, current_epoch=0)
        pl_module = _mock_pl_module()

        one_gb = 1024 ** 3
        with patch(
            "biom3.Stage3.callbacks._get_peak_memory_stats",
            return_value=(2 * one_gb, 4 * one_gb),
        ), patch(
            "biom3.Stage3.callbacks._reset_peak_memory_stats",
        ):
            cb.on_train_start(trainer, pl_module)
            cb.on_train_epoch_start(trainer, pl_module)
            trainer.global_step = 100
            cb.on_train_epoch_end(trainer, pl_module)
            cb.on_train_end(trainer, pl_module)

        out_path = os.path.join(tmp_output_dir, "benchmark_history.json")
        with open(out_path) as f:
            data = json.load(f)
        assert data["epochs"][0]["peak_memory_allocated_gb"] == 2.0
        assert data["epochs"][0]["peak_memory_reserved_gb"] == 4.0
        assert "backend" in data["config"]


class TestPerRankMemoryGather:

    def test_gather_disabled_by_default(self, tmp_output_dir):
        cb = _make_callback(tmp_output_dir)  # all_ranks_memory=False
        pl_module = _mock_pl_module()
        per_alloc, per_reserved = cb._gather_across_ranks(
            pl_module, peak_alloc=1024 ** 3, peak_reserved=2 * 1024 ** 3,
        )
        assert per_alloc is None
        assert per_reserved is None
        # all_gather should not have been called
        pl_module.all_gather.assert_not_called() if hasattr(
            pl_module.all_gather, "assert_not_called"
        ) else None

    def test_gather_none_when_peak_is_none(self, tmp_output_dir):
        cb = _make_callback(tmp_output_dir, all_ranks_memory=True)
        pl_module = _mock_pl_module()
        per_alloc, per_reserved = cb._gather_across_ranks(
            pl_module, peak_alloc=None, peak_reserved=None,
        )
        assert per_alloc is None
        assert per_reserved is None

    def test_gather_single_process(self, tmp_output_dir):
        """all_gather returns 1D tensor (2,) in single-process mode."""
        cb = _make_callback(tmp_output_dir, all_ranks_memory=True)
        one_gb = 1024 ** 3
        # Single-process all_gather returns the local tensor unchanged
        pl_module = _mock_pl_module(
            all_gather_return=torch.tensor([2.0 * one_gb, 4.0 * one_gb],
                                           dtype=torch.float64),
        )
        per_alloc, per_reserved = cb._gather_across_ranks(
            pl_module, peak_alloc=2 * one_gb, peak_reserved=4 * one_gb,
        )
        assert per_alloc == [2.0]
        assert per_reserved == [4.0]

    def test_gather_distributed(self, tmp_output_dir):
        """all_gather returns 2D tensor (world_size, 2) in distributed mode."""
        cb = _make_callback(tmp_output_dir, all_ranks_memory=True)
        one_gb = 1024 ** 3
        # Simulate 4-rank all_gather result
        gathered = torch.tensor([
            [1.0 * one_gb, 2.0 * one_gb],
            [1.5 * one_gb, 2.5 * one_gb],
            [0.8 * one_gb, 1.8 * one_gb],
            [1.2 * one_gb, 2.2 * one_gb],
        ], dtype=torch.float64)
        pl_module = _mock_pl_module(all_gather_return=gathered)
        per_alloc, per_reserved = cb._gather_across_ranks(
            pl_module, peak_alloc=1.0 * one_gb, peak_reserved=2.0 * one_gb,
        )
        assert per_alloc == [1.0, 1.5, 0.8, 1.2]
        assert per_reserved == [2.0, 2.5, 1.8, 2.2]

    def test_gather_failure_returns_none(self, tmp_output_dir):
        cb = _make_callback(tmp_output_dir, all_ranks_memory=True)
        pl_module = _mock_pl_module()
        pl_module.all_gather = MagicMock(side_effect=RuntimeError("boom"))
        per_alloc, per_reserved = cb._gather_across_ranks(
            pl_module, peak_alloc=1024 ** 3, peak_reserved=2 * 1024 ** 3,
        )
        assert per_alloc is None
        assert per_reserved is None

    def test_per_rank_fields_in_record(self, tmp_output_dir):
        cb = _make_callback(tmp_output_dir, all_ranks_memory=True,
                            training_strategy="primary_only")
        trainer = _mock_trainer(global_step=0, current_epoch=0)
        one_gb = 1024 ** 3
        pl_module = _mock_pl_module(
            all_gather_return=torch.tensor(
                [[1.0 * one_gb, 2.0 * one_gb],
                 [1.5 * one_gb, 2.5 * one_gb]],
                dtype=torch.float64,
            ),
        )

        with patch(
            "biom3.Stage3.callbacks._get_peak_memory_stats",
            return_value=(1 * one_gb, 2 * one_gb),
        ), patch(
            "biom3.Stage3.callbacks._reset_peak_memory_stats",
        ):
            cb.on_train_start(trainer, pl_module)
            cb.on_train_epoch_start(trainer, pl_module)
            trainer.global_step = 100
            cb.on_train_epoch_end(trainer, pl_module)

        r = cb._epoch_records[0]
        assert r["peak_memory_allocated_gb_per_rank"] == [1.0, 1.5]
        assert r["peak_memory_reserved_gb_per_rank"] == [2.0, 2.5]
        # Rank-0 scalar fields still populated
        assert r["peak_memory_allocated_gb"] == 1.0
        assert r["peak_memory_reserved_gb"] == 2.0

    def test_per_rank_fields_in_saved_json(self, tmp_output_dir):
        cb = _make_callback(tmp_output_dir, all_ranks_memory=True,
                            training_strategy="primary_only")
        trainer = _mock_trainer(global_step=0, current_epoch=0)
        one_gb = 1024 ** 3
        pl_module = _mock_pl_module(
            all_gather_return=torch.tensor(
                [[1.0 * one_gb, 2.0 * one_gb],
                 [1.5 * one_gb, 2.5 * one_gb]],
                dtype=torch.float64,
            ),
        )

        with patch(
            "biom3.Stage3.callbacks._get_peak_memory_stats",
            return_value=(1 * one_gb, 2 * one_gb),
        ), patch(
            "biom3.Stage3.callbacks._reset_peak_memory_stats",
        ):
            cb.on_train_start(trainer, pl_module)
            cb.on_train_epoch_start(trainer, pl_module)
            trainer.global_step = 100
            cb.on_train_epoch_end(trainer, pl_module)
            cb.on_train_end(trainer, pl_module)

        out_path = os.path.join(tmp_output_dir, "benchmark_history.json")
        with open(out_path) as f:
            data = json.load(f)
        ep = data["epochs"][0]
        assert ep["peak_memory_allocated_gb_per_rank"] == [1.0, 1.5]
        assert ep["peak_memory_reserved_gb_per_rank"] == [2.0, 2.5]


class TestSaveOutput:

    def test_saves_json(self, tmp_output_dir):
        cb = _make_callback(tmp_output_dir, batch_size=32, acc_grad_batches=2,
                            gpu_devices=4, num_nodes=1, precision="bf16")
        trainer = _mock_trainer(global_step=0, current_epoch=0)
        pl_module = _mock_pl_module()

        cb.on_train_start(trainer, pl_module)
        cb.on_train_epoch_start(trainer, pl_module)
        trainer.global_step = 50
        cb.on_train_epoch_end(trainer, pl_module)
        cb.on_train_end(trainer, pl_module)

        out_path = os.path.join(tmp_output_dir, "benchmark_history.json")
        assert os.path.exists(out_path)

        with open(out_path) as f:
            data = json.load(f)

        assert data["config"]["batch_size"] == 32
        assert data["config"]["acc_grad_batches"] == 2
        assert data["config"]["gpu_devices"] == 4
        assert data["config"]["num_nodes"] == 1
        assert data["config"]["effective_batch_size"] == 32 * 2 * 4 * 1
        assert data["config"]["precision"] == "bf16"
        assert len(data["epochs"]) == 1
        assert data["epochs"][0]["steps_in_epoch"] == 50

    def test_no_save_on_non_zero_rank(self, tmp_output_dir):
        cb = _make_callback(tmp_output_dir)
        trainer = _mock_trainer(global_rank=1, global_step=0, current_epoch=0)
        pl_module = _mock_pl_module()

        cb.on_train_start(trainer, pl_module)
        cb.on_train_epoch_start(trainer, pl_module)
        trainer.global_step = 50
        cb.on_train_epoch_end(trainer, pl_module)
        cb.on_train_end(trainer, pl_module)

        out_path = os.path.join(tmp_output_dir, "benchmark_history.json")
        assert not os.path.exists(out_path)


class TestSkipFirstEpoch:

    def test_summary_skips_first_epoch(self, tmp_output_dir):
        cb = _make_callback(tmp_output_dir, skip_first_epoch=True)
        trainer = _mock_trainer(global_step=0, current_epoch=0)
        pl_module = _mock_pl_module()

        cb.on_train_start(trainer, pl_module)
        for epoch in range(3):
            trainer.current_epoch = epoch
            cb.on_train_epoch_start(trainer, pl_module)
            trainer.global_step += 100
            cb.on_train_epoch_end(trainer, pl_module)
        cb.on_train_end(trainer, pl_module)

        # All 3 epochs should be in the raw data
        out_path = os.path.join(tmp_output_dir, "benchmark_history.json")
        with open(out_path) as f:
            data = json.load(f)
        assert len(data["epochs"]) == 3

    def test_no_skip_when_disabled(self, tmp_output_dir):
        cb = _make_callback(tmp_output_dir, skip_first_epoch=False)
        trainer = _mock_trainer(global_step=0, current_epoch=0)
        pl_module = _mock_pl_module()

        cb.on_train_start(trainer, pl_module)
        cb.on_train_epoch_start(trainer, pl_module)
        trainer.global_step = 100
        cb.on_train_epoch_end(trainer, pl_module)
        cb.on_train_end(trainer, pl_module)

        # Record should still exist
        assert len(cb._epoch_records) == 1


class TestPerStepJsonl:

    def test_disabled_by_default(self, tmp_output_dir):
        cb = _make_callback(tmp_output_dir)
        trainer = _mock_trainer(global_step=0, current_epoch=0)
        pl_module = _mock_pl_module()
        cb.on_train_start(trainer, pl_module)
        cb.on_train_batch_start(trainer, pl_module, batch=None, batch_idx=0)
        trainer.global_step = 1
        cb.on_train_batch_end(trainer, pl_module, outputs=None, batch=None,
                              batch_idx=0)
        cb.on_train_end(trainer, pl_module)
        assert not os.path.exists(
            os.path.join(tmp_output_dir, "benchmark_steps.jsonl")
        )

    def test_writes_step_records(self, tmp_output_dir):
        cb = _make_callback(tmp_output_dir, per_step=True)
        trainer = _mock_trainer(global_step=0, current_epoch=0)
        pl_module = _mock_pl_module()
        cb.on_train_start(trainer, pl_module)
        for i in range(3):
            cb.on_train_batch_start(trainer, pl_module, batch=None, batch_idx=i)
            trainer.global_step = i + 1
            cb.on_train_batch_end(trainer, pl_module, outputs=None, batch=None,
                                  batch_idx=i)
        cb.on_train_end(trainer, pl_module)

        jsonl = os.path.join(tmp_output_dir, "benchmark_steps.jsonl")
        assert os.path.exists(jsonl)
        with open(jsonl) as fh:
            lines = [json.loads(l) for l in fh if l.strip()]
        assert len(lines) == 3
        assert lines[0]["batch_idx"] == 0
        assert lines[2]["global_step"] == 3
        assert all("step_wall_time_sec" in r for r in lines)

    def test_non_zero_rank_skipped(self, tmp_output_dir):
        cb = _make_callback(tmp_output_dir, per_step=True)
        trainer = _mock_trainer(global_rank=1, global_step=0, current_epoch=0)
        pl_module = _mock_pl_module()
        cb.on_train_start(trainer, pl_module)
        cb.on_train_batch_start(trainer, pl_module, batch=None, batch_idx=0)
        cb.on_train_batch_end(trainer, pl_module, outputs=None, batch=None,
                              batch_idx=0)
        cb.on_train_end(trainer, pl_module)
        assert not os.path.exists(
            os.path.join(tmp_output_dir, "benchmark_steps.jsonl")
        )

    def test_on_exception_flushes(self, tmp_output_dir):
        cb = _make_callback(tmp_output_dir, per_step=True)
        trainer = _mock_trainer(global_step=0, current_epoch=0)
        pl_module = _mock_pl_module()
        cb.on_train_start(trainer, pl_module)
        cb.on_train_batch_start(trainer, pl_module, batch=None, batch_idx=0)
        trainer.global_step = 1
        cb.on_train_batch_end(trainer, pl_module, outputs=None, batch=None,
                              batch_idx=0)
        cb.on_exception(trainer, pl_module, RuntimeError("boom"))
        jsonl = os.path.join(tmp_output_dir, "benchmark_steps.jsonl")
        with open(jsonl) as fh:
            lines = [l for l in fh if l.strip()]
        assert len(lines) == 1


def _make_metrics_trainer(global_rank=0, global_step=0, current_epoch=0,
                          callback_metrics=None):
    trainer = MagicMock()
    trainer.global_rank = global_rank
    trainer.global_step = global_step
    trainer.current_epoch = current_epoch
    trainer.callback_metrics = callback_metrics or {}
    return trainer


class TestMetricsHistoryJsonlStreaming:

    def test_jsonl_written_on_each_step(self, tmp_output_dir):
        cb = MetricsHistoryCallback(output_dir=tmp_output_dir, every_n_steps=1)
        pl_module = _mock_pl_module()
        metrics = {"train_loss": torch.tensor(0.5)}
        trainer = _make_metrics_trainer(callback_metrics=metrics)
        cb.on_train_start(trainer, pl_module)
        for step in range(1, 4):
            trainer.global_step = step
            metrics["train_loss"] = torch.tensor(0.5 / step)
            cb.on_train_batch_end(trainer, pl_module, outputs=None,
                                  batch=None, batch_idx=step - 1)
        cb.on_train_end(trainer, pl_module)

        jsonl = os.path.join(tmp_output_dir,
                             MetricsHistoryCallback.TRAIN_JSONL)
        with open(jsonl) as fh:
            records = [json.loads(l) for l in fh if l.strip()]
        assert len(records) == 3
        assert records[0]["global_step"] == 1
        assert records[0]["source"] == "step"
        assert "train_loss" in records[0]

    def test_jsonl_respects_every_n_steps(self, tmp_output_dir):
        cb = MetricsHistoryCallback(output_dir=tmp_output_dir, every_n_steps=2)
        pl_module = _mock_pl_module()
        metrics = {"train_loss": torch.tensor(0.5)}
        trainer = _make_metrics_trainer(callback_metrics=metrics)
        cb.on_train_start(trainer, pl_module)
        for step in range(1, 6):
            trainer.global_step = step
            cb.on_train_batch_end(trainer, pl_module, outputs=None,
                                  batch=None, batch_idx=step - 1)
        cb.on_train_end(trainer, pl_module)

        jsonl = os.path.join(tmp_output_dir,
                             MetricsHistoryCallback.TRAIN_JSONL)
        with open(jsonl) as fh:
            records = [json.loads(l) for l in fh if l.strip()]
        # Only steps 2 and 4 should be recorded (step % 2 == 0, step > 0)
        assert [r["global_step"] for r in records] == [2, 4]

    def test_every_n_epochs_writes_source_epoch_records(self, tmp_output_dir):
        cb = MetricsHistoryCallback(
            output_dir=tmp_output_dir, every_n_steps=10_000,
            every_n_epochs=1,
        )
        pl_module = _mock_pl_module()
        metrics = {"train_loss_epoch": torch.tensor(0.25)}
        trainer = _make_metrics_trainer(callback_metrics=metrics)
        cb.on_train_start(trainer, pl_module)
        for epoch in range(3):
            trainer.current_epoch = epoch
            trainer.global_step = (epoch + 1) * 100
            cb.on_train_epoch_end(trainer, pl_module)
        cb.on_train_end(trainer, pl_module)

        jsonl = os.path.join(tmp_output_dir,
                             MetricsHistoryCallback.TRAIN_JSONL)
        with open(jsonl) as fh:
            records = [json.loads(l) for l in fh if l.strip()]
        assert len(records) == 3
        assert all(r["source"] == "epoch" for r in records)
        assert [r["epoch"] for r in records] == [0, 1, 2]

    def test_every_n_epochs_cadence(self, tmp_output_dir):
        cb = MetricsHistoryCallback(
            output_dir=tmp_output_dir, every_n_steps=10_000,
            every_n_epochs=2,
        )
        pl_module = _mock_pl_module()
        trainer = _make_metrics_trainer(
            callback_metrics={"train_loss": torch.tensor(0.1)}
        )
        cb.on_train_start(trainer, pl_module)
        for epoch in range(5):
            trainer.current_epoch = epoch
            cb.on_train_epoch_end(trainer, pl_module)
        cb.on_train_end(trainer, pl_module)

        jsonl = os.path.join(tmp_output_dir,
                             MetricsHistoryCallback.TRAIN_JSONL)
        with open(jsonl) as fh:
            records = [json.loads(l) for l in fh if l.strip()]
        # Epoch 1 (0+1=1%2!=0), 3 (3+1=4%2==0) — (epoch+1)%N==0
        assert [r["epoch"] for r in records] == [1, 3]

    def test_val_records_streamed(self, tmp_output_dir):
        cb = MetricsHistoryCallback(output_dir=tmp_output_dir)
        pl_module = _mock_pl_module()
        metrics = {
            "val_loss": torch.tensor(1.2),
            "val_loss_epoch": torch.tensor(1.2),
            "train_loss_epoch": torch.tensor(1.0),
        }
        trainer = _make_metrics_trainer(
            global_step=500, current_epoch=3, callback_metrics=metrics,
        )
        cb.on_train_start(trainer, pl_module)
        cb.on_validation_epoch_end(trainer, pl_module)
        cb.on_train_end(trainer, pl_module)

        jsonl = os.path.join(tmp_output_dir,
                             MetricsHistoryCallback.VAL_JSONL)
        with open(jsonl) as fh:
            records = [json.loads(l) for l in fh if l.strip()]
        assert len(records) == 1
        assert records[0]["val_loss"] == pytest.approx(1.2)
        assert records[0]["loss_gap"] == pytest.approx(0.2)


class TestMetricsHistoryFlush:

    def test_flush_every_n_steps_fsyncs(self, tmp_output_dir):
        cb = MetricsHistoryCallback(
            output_dir=tmp_output_dir, flush_every_n_steps=2,
        )
        pl_module = _mock_pl_module()
        metrics = {"train_loss": torch.tensor(0.5)}
        trainer = _make_metrics_trainer(callback_metrics=metrics)

        with patch("biom3.Stage3.callbacks.os.fsync") as fsync_mock:
            cb.on_train_start(trainer, pl_module)
            for step in range(1, 5):
                trainer.global_step = step
                cb.on_train_batch_end(trainer, pl_module, outputs=None,
                                      batch=None, batch_idx=step - 1)
            # fsync fires on steps 2 and 4 (global_step % 2 == 0)
            # Each fsync call covers both train and val file handles.
            assert fsync_mock.call_count >= 4  # 2 events * 2 file handles

    def test_flush_disabled_by_default(self, tmp_output_dir):
        cb = MetricsHistoryCallback(output_dir=tmp_output_dir)
        pl_module = _mock_pl_module()
        metrics = {"train_loss": torch.tensor(0.5)}
        trainer = _make_metrics_trainer(callback_metrics=metrics)

        with patch("biom3.Stage3.callbacks.os.fsync") as fsync_mock:
            cb.on_train_start(trainer, pl_module)
            for step in range(1, 5):
                trainer.global_step = step
                cb.on_train_batch_end(trainer, pl_module, outputs=None,
                                      batch=None, batch_idx=step - 1)
            # No periodic flushes; only the end-of-train close fsyncs
            assert fsync_mock.call_count == 0


class TestRebuildMetricsHistoryPt:

    def test_rebuild_from_jsonl(self, tmp_output_dir):
        os.makedirs(tmp_output_dir, exist_ok=True)
        train_path = os.path.join(
            tmp_output_dir, MetricsHistoryCallback.TRAIN_JSONL,
        )
        val_path = os.path.join(
            tmp_output_dir, MetricsHistoryCallback.VAL_JSONL,
        )
        with open(train_path, "w") as fh:
            for step in (1, 2, 3):
                fh.write(json.dumps({
                    "global_step": step, "epoch": 0, "source": "step",
                    "train_loss": 1.0 / step,
                }) + "\n")
        with open(val_path, "w") as fh:
            fh.write(json.dumps({
                "global_step": 3, "epoch": 0,
                "val_loss": 0.4, "loss_gap": 0.1,
            }) + "\n")

        out_path, n_train, n_val = rebuild_metrics_history_pt(tmp_output_dir)
        assert os.path.exists(out_path)
        assert n_train == 3
        assert n_val == 1
        payload = torch.load(out_path, weights_only=False)
        assert list(payload["train"]["train_loss"]) == [1.0, 0.5, 1.0 / 3]
        assert payload["val"]["val_loss"][0] == 0.4

    def test_rebuild_tolerates_truncated_last_line(self, tmp_output_dir):
        os.makedirs(tmp_output_dir, exist_ok=True)
        train_path = os.path.join(
            tmp_output_dir, MetricsHistoryCallback.TRAIN_JSONL,
        )
        with open(train_path, "w") as fh:
            fh.write(json.dumps({
                "global_step": 1, "epoch": 0, "source": "step",
                "train_loss": 0.9,
            }) + "\n")
            # Truncated final line (simulating a process killed mid-write)
            fh.write('{"global_step": 2, "epoch": 0, "source": "st')

        out_path, n_train, n_val = rebuild_metrics_history_pt(tmp_output_dir)
        assert n_train == 1
        assert n_val == 0
        payload = torch.load(out_path, weights_only=False)
        assert list(payload["train"]["global_step"]) == [1]


class TestBuildCheckpointCallbacks:
    """Verifies the orthogonal monitored / periodic checkpoint helper.

    The pre-refactor bug was that periodic ``every_n_*`` knobs were attached
    to the same monitored ``ModelCheckpoint`` that had ``save_top_k=2``, so
    periodic-trigger saves were silently pruned by the metric-based retention.
    These tests pin the orthogonal split and the parameter wiring.
    """

    def test_default_no_periodic(self, tmp_path):
        ckpt_dir = str(tmp_path / "ckpt")
        monitored, periodic = build_checkpoint_callbacks(checkpoint_dir=ckpt_dir)
        assert periodic is None
        assert len(monitored) == 1
        cb = monitored[0]
        assert cb.monitor == "val_loss"
        assert cb.mode == "min"
        assert cb.save_top_k == 2
        assert cb.save_last == "link"
        assert cb.dirpath == ckpt_dir
        # The bug fixed by the refactor: monitored callbacks must NOT carry
        # periodic cadences (which would compete with save_top_k pruning).
        assert cb._every_n_train_steps == 0
        assert cb._every_n_epochs == 1  # PL default; no explicit override

    def test_periodic_every_n_steps(self, tmp_path):
        ckpt_dir = str(tmp_path / "ckpt")
        monitored, periodic = build_checkpoint_callbacks(
            checkpoint_dir=ckpt_dir, periodic_every_n_steps=50,
        )
        assert periodic is not None
        # Monitored callback stays clean.
        assert monitored[0]._every_n_train_steps == 0
        # Periodic callback: monitor=None, keep-all, separate subdir, step cadence wired.
        assert periodic.monitor is None
        assert periodic.save_top_k == -1
        assert periodic.save_last is False
        assert periodic.dirpath == os.path.join(ckpt_dir, "periodic")
        assert periodic._every_n_train_steps == 50

    def test_periodic_every_n_epochs(self, tmp_path):
        ckpt_dir = str(tmp_path / "ckpt")
        _, periodic = build_checkpoint_callbacks(
            checkpoint_dir=ckpt_dir, periodic_every_n_epochs=5,
        )
        assert periodic is not None
        assert periodic._every_n_epochs == 5
        assert periodic._every_n_train_steps == 0

    def test_periodic_both_cadences_rejected(self, tmp_path):
        # Lightning's ModelCheckpoint enforces step-vs-epoch exclusivity;
        # we surface that as a clear error at helper-call time.
        ckpt_dir = str(tmp_path / "ckpt")
        with pytest.raises(ValueError, match="mutually exclusive"):
            build_checkpoint_callbacks(
                checkpoint_dir=ckpt_dir,
                periodic_every_n_steps=100,
                periodic_every_n_epochs=2,
            )

    def test_periodic_max_keep_keep_last_one(self, tmp_path):
        # save_top_k=1 with monitor=None means "keep most recent only" — the
        # smallest legal positive bound under Lightning's constraint.
        ckpt_dir = str(tmp_path / "ckpt")
        _, periodic = build_checkpoint_callbacks(
            checkpoint_dir=ckpt_dir,
            periodic_every_n_epochs=1,
            periodic_max_keep=1,
        )
        assert periodic.save_top_k == 1

    def test_periodic_max_keep_invalid_raises(self, tmp_path):
        # Anything outside {-1, 0, 1} is rejected; Lightning's monitor=None
        # constraint makes other values unsupportable without custom pruning.
        ckpt_dir = str(tmp_path / "ckpt")
        with pytest.raises(ValueError, match="periodic_max_keep"):
            build_checkpoint_callbacks(
                checkpoint_dir=ckpt_dir,
                periodic_every_n_epochs=1,
                periodic_max_keep=3,
            )

    def test_multiple_monitors(self, tmp_path):
        ckpt_dir = str(tmp_path / "ckpt")
        monitored, _ = build_checkpoint_callbacks(
            checkpoint_dir=ckpt_dir,
            checkpoint_monitors=[
                {"metric": "val_loss", "mode": "min"},
                {"metric": "val_acc", "mode": "max"},
            ],
        )
        assert len(monitored) == 2
        # Primary keeps top-2 + last symlink.
        assert monitored[0].monitor == "val_loss"
        assert monitored[0].save_top_k == 2
        assert monitored[0].save_last == "link"
        # Secondary monitors keep best-1 with metric-slug filename, no last.
        assert monitored[1].monitor == "val_acc"
        assert monitored[1].mode == "max"
        assert monitored[1].save_top_k == 1
        assert monitored[1].save_last is False
        assert "val_acc" in monitored[1].filename

    def test_checkpoint_monitors_accepts_json_string(self, tmp_path):
        ckpt_dir = str(tmp_path / "ckpt")
        monitors_json = json.dumps([{"metric": "loss_gap", "mode": "min"}])
        monitored, _ = build_checkpoint_callbacks(
            checkpoint_dir=ckpt_dir, checkpoint_monitors=monitors_json,
        )
        assert len(monitored) == 1
        assert monitored[0].monitor == "loss_gap"

    def test_use_sync_safe_swaps_class(self, tmp_path):
        ckpt_dir = str(tmp_path / "ckpt")
        # Default: monitored callbacks are LoggingModelCheckpoint.
        mon_default, _ = build_checkpoint_callbacks(checkpoint_dir=ckpt_dir)
        assert isinstance(mon_default[0], LoggingModelCheckpoint)
        assert not isinstance(mon_default[0], SyncSafeModelCheckpoint)
        # use_sync_safe=True: monitored callbacks become SyncSafeModelCheckpoint.
        mon_sync, periodic = build_checkpoint_callbacks(
            checkpoint_dir=ckpt_dir,
            use_sync_safe=True,
            periodic_every_n_steps=10,
        )
        assert isinstance(mon_sync[0], SyncSafeModelCheckpoint)
        # Periodic always uses LoggingModelCheckpoint regardless — SyncSafe's
        # override only matters when comparing against a monitored metric.
        assert isinstance(periodic, LoggingModelCheckpoint)
        assert not isinstance(periodic, SyncSafeModelCheckpoint)


class TestProgressiveMetricsHistorySnapshot:
    """metrics_history.pt must exist on disk after each validation epoch end,
    not only after on_train_end. Pins the timeout-resilience behavior."""

    def test_pt_snapshot_on_validation_epoch_end(self, tmp_output_dir):
        cb = MetricsHistoryCallback(output_dir=tmp_output_dir)
        pl_module = _mock_pl_module()
        metrics = {
            "val_loss": torch.tensor(1.5),
            "val_loss_epoch": torch.tensor(1.5),
            "train_loss_epoch": torch.tensor(1.0),
        }
        trainer = _make_metrics_trainer(
            global_step=100, current_epoch=1, callback_metrics=metrics,
        )
        cb.on_train_start(trainer, pl_module)
        cb.on_validation_epoch_end(trainer, pl_module)

        pt_path = os.path.join(tmp_output_dir, "metrics_history.pt")
        assert os.path.exists(pt_path), (
            "metrics_history.pt must exist after a validation epoch — not "
            "only after on_train_end. Otherwise a SIGTERM mid-run loses "
            "the consolidated metrics file."
        )
        payload = torch.load(pt_path, weights_only=False)
        assert payload["val"]["val_loss"][0] == pytest.approx(1.5)

    def test_jsonl_fsynced_at_validation_end(self, tmp_output_dir):
        # Even with flush_every_n_steps=None (default), val-end must fsync.
        cb = MetricsHistoryCallback(output_dir=tmp_output_dir)
        pl_module = _mock_pl_module()
        trainer = _make_metrics_trainer(
            global_step=10, current_epoch=0,
            callback_metrics={"val_loss": torch.tensor(1.0)},
        )
        cb.on_train_start(trainer, pl_module)
        with patch("biom3.Stage3.callbacks.os.fsync") as fsync_mock:
            cb.on_validation_epoch_end(trainer, pl_module)
            # Both train and val JSONL handles get fsynced.
            assert fsync_mock.call_count >= 2


class TestProgressiveBenchmarkHistorySnapshot:
    """benchmark_history.json must be re-written on each train epoch end."""

    def test_json_snapshot_on_train_epoch_end(self, tmp_output_dir):
        cb = _make_callback(tmp_output_dir, batch_size=4, gpu_devices=1,
                            num_nodes=1, acc_grad_batches=1)
        pl_module = _mock_pl_module()
        trainer = _make_metrics_trainer(global_step=100, current_epoch=0)
        cb.on_train_start(trainer, pl_module)
        cb.on_train_epoch_start(trainer, pl_module)
        # Move time forward so wall-time math doesn't divide by zero.
        time.sleep(0.01)
        trainer.global_step = 110
        cb.on_train_epoch_end(trainer, pl_module)

        json_path = os.path.join(tmp_output_dir, "benchmark_history.json")
        assert os.path.exists(json_path), (
            "benchmark_history.json must be written after each training "
            "epoch — otherwise a timeout loses the per-epoch summary."
        )
        with open(json_path) as fh:
            payload = json.load(fh)
        assert len(payload["epochs"]) == 1


class TestBestArtifactSyncCallback:
    """The mid-training best-artifact sync must:
       - Skip until the monitored callback has produced a best.
       - Run sync_fn on each new best (best_model_path changed).
       - NOT re-run sync_fn when the best path is unchanged (idempotent).
       - Throttle by every_n_val.
       - Tolerate sync_fn exceptions without aborting training.
    """

    def _make(self, every_n_val=1):
        primary = MagicMock()
        primary.best_model_path = ""
        sync_fn = MagicMock()
        cb = BestArtifactSyncCallback(
            sync_fn=sync_fn, primary_callback=primary,
            extra_callbacks=[], every_n_val=every_n_val,
        )
        trainer = _mock_trainer(global_rank=0)
        pl_module = _mock_pl_module()
        return cb, primary, sync_fn, trainer, pl_module

    def test_skips_when_no_best_yet(self):
        cb, primary, sync_fn, trainer, pl_module = self._make()
        primary.best_model_path = ""  # no checkpoint yet
        cb.on_validation_epoch_end(trainer, pl_module)
        sync_fn.assert_not_called()

    def test_runs_on_first_best(self, tmp_path):
        cb, primary, sync_fn, trainer, pl_module = self._make()
        # Simulate ModelCheckpoint having just promoted a best path.
        primary.best_model_path = str(tmp_path / "best.ckpt")
        cb.on_validation_epoch_end(trainer, pl_module)
        sync_fn.assert_called_once()

    def test_idempotent_when_best_unchanged(self, tmp_path):
        cb, primary, sync_fn, trainer, pl_module = self._make()
        primary.best_model_path = str(tmp_path / "best.ckpt")
        cb.on_validation_epoch_end(trainer, pl_module)
        cb.on_validation_epoch_end(trainer, pl_module)
        cb.on_validation_epoch_end(trainer, pl_module)
        # Three val cycles, one new best — only one sync call.
        assert sync_fn.call_count == 1

    def test_re_runs_when_best_changes(self, tmp_path):
        cb, primary, sync_fn, trainer, pl_module = self._make()
        primary.best_model_path = str(tmp_path / "best_v1.ckpt")
        cb.on_validation_epoch_end(trainer, pl_module)
        primary.best_model_path = str(tmp_path / "best_v2.ckpt")
        cb.on_validation_epoch_end(trainer, pl_module)
        assert sync_fn.call_count == 2

    def test_throttle_every_n_val(self, tmp_path):
        cb, primary, sync_fn, trainer, pl_module = self._make(every_n_val=3)
        # Even with a new best every cycle, sync only fires every 3rd cycle.
        for i in range(6):
            primary.best_model_path = str(tmp_path / f"best_v{i}.ckpt")
            cb.on_validation_epoch_end(trainer, pl_module)
        # Cycles 3 and 6 trigger.
        assert sync_fn.call_count == 2

    def test_sync_exception_does_not_propagate(self, tmp_path):
        cb, primary, sync_fn, trainer, pl_module = self._make()
        sync_fn.side_effect = RuntimeError("DeepSpeed conversion failed")
        primary.best_model_path = str(tmp_path / "best.ckpt")
        # Must not raise — losing an artifact-sync should not abort training.
        cb.on_validation_epoch_end(trainer, pl_module)
        assert sync_fn.call_count == 1
        # Failed sync must NOT mark the path as synced; the next val cycle
        # will retry against the same best path.
        cb.on_validation_epoch_end(trainer, pl_module)
        assert sync_fn.call_count == 2

    def test_non_zero_rank_skips(self, tmp_path):
        cb, primary, sync_fn, _, pl_module = self._make()
        trainer = _mock_trainer(global_rank=1)
        primary.best_model_path = str(tmp_path / "best.ckpt")
        cb.on_validation_epoch_end(trainer, pl_module)
        sync_fn.assert_not_called()

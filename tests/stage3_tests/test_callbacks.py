"""Tests for Stage 3 custom callbacks."""

import json
import os
import time
from unittest.mock import MagicMock, patch

import pytest
import torch

from biom3.Stage3.callbacks import TrainingBenchmarkCallback

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

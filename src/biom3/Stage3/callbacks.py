"""Custom PyTorch Lightning callbacks for Stage 3 training."""

import json
import os
import socket
import time
from datetime import datetime

import numpy as np
import torch

from biom3.backend.device import BACKEND_NAME, _CUDA, _XPU, setup_logger

if BACKEND_NAME == _XPU:
    import lightning as pl
    from lightning.pytorch.callbacks import ModelCheckpoint as _ModelCheckpoint
else:
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint as _ModelCheckpoint

logger = setup_logger(__name__)


class MetricsHistoryCallback(pl.Callback):
    """Accumulates training and validation metrics and saves to a .pt file.

    The saved file (``metrics_history.pt``) is a dict with two keys:

    - ``"train"``: dict mapping metric names to 1-D numpy arrays, plus
      ``"global_step"`` and ``"epoch"`` index arrays.
    - ``"val"``: same structure for validation metrics (including ``"loss_gap"``).

    Parameters
    ----------
    output_dir : str
        Directory where ``metrics_history.pt`` is written.
    save_ranks : list[int]
        Rank indices on which to record and save train/val metrics
        (default: ``[0]``).
    every_n_steps : int
        Record training metrics every *n* global steps (default: ``1``).
    all_ranks_val_loss : bool
        When True, record ``val_loss`` and ``val_loss_epoch`` on **every** rank
        at validation epoch end and dump one ``metrics_history.rank{N}.pt`` per
        rank under ``output_dir``. Used to diagnose cross-rank sync_dist issues
        in ``ModelCheckpoint`` (the consensus ``reduce_boolean_decision`` call
        requires all ranks to agree that val_loss improved). Default: False.
    """

    def __init__(self, output_dir, save_ranks=None, every_n_steps=1,
                 all_ranks_val_loss=False):
        super().__init__()
        self.output_dir = output_dir
        self.save_ranks = save_ranks or [0]
        self.every_n_steps = max(1, every_n_steps)
        self.all_ranks_val_loss = all_ranks_val_loss
        self.train_step_metrics: list[dict] = []
        self.val_epoch_metrics: list[dict] = []
        self.per_rank_val_loss: list[dict] = []

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_rank not in self.save_ranks:
            return
        if trainer.global_step % self.every_n_steps != 0:
            return
        logged = trainer.callback_metrics
        record = {"global_step": trainer.global_step, "epoch": trainer.current_epoch}
        for k, v in logged.items():
            if k.startswith("train_"):
                record[k] = v.item() if hasattr(v, "item") else v
        self.train_step_metrics.append(record)

    def on_validation_epoch_end(self, trainer, pl_module):
        if self.all_ranks_val_loss:
            logged = trainer.callback_metrics
            rec = {
                "global_step": trainer.global_step,
                "epoch": trainer.current_epoch,
                "rank": trainer.global_rank,
            }
            for key in ("val_loss", "val_loss_epoch", "val_loss_step"):
                v = logged.get(key)
                if v is not None:
                    rec[key] = v.item() if hasattr(v, "item") else v
            self.per_rank_val_loss.append(rec)

        if trainer.global_rank not in self.save_ranks:
            return
        logged = trainer.callback_metrics
        record = {"global_step": trainer.global_step, "epoch": trainer.current_epoch}
        for k, v in logged.items():
            if k.startswith("val_"):
                record[k] = v.item() if hasattr(v, "item") else v
        # Compute loss gap (val_loss - train_loss at epoch level)
        val_loss = record.get("val_loss_epoch") or record.get("val_loss")
        train_loss = logged.get("train_loss_epoch")
        if train_loss is not None:
            train_loss = train_loss.item() if hasattr(train_loss, "item") else train_loss
        if val_loss is not None and train_loss is not None:
            record["loss_gap"] = val_loss - train_loss
        self.val_epoch_metrics.append(record)

        # Log epoch summary to file logger (rank 0 only)
        if trainer.global_rank == 0:
            epoch = trainer.current_epoch
            step = trainer.global_step
            parts = [f"Epoch {epoch} (step {step})"]
            if train_loss is not None:
                parts.append(f"train_loss={train_loss:.4f}")
            if val_loss is not None:
                parts.append(f"val_loss={val_loss:.4f}")
            if val_loss is not None and train_loss is not None:
                parts.append(f"gap={val_loss - train_loss:+.4f}")
            for key in ("val_prev_hard_acc_epoch", "val_fut_hard_acc_epoch",
                        "val_current_hard_acc_epoch"):
                v = record.get(key)
                if v is not None:
                    short = key.replace("val_", "").replace("_epoch", "")
                    parts.append(f"{short}={v:.3f}")
            lr = logged.get("lr-AdamW")
            if lr is not None:
                lr_val = lr.item() if hasattr(lr, "item") else lr
                parts.append(f"lr={lr_val:.2e}")
            logger.info("  ".join(parts))

    def on_train_end(self, trainer, pl_module):
        if self.all_ranks_val_loss:
            self._save_per_rank_val_loss(trainer.global_rank)
        if trainer.global_rank not in self.save_ranks:
            return
        self._save()

    @staticmethod
    def _records_to_arrays(records: list[dict]) -> dict[str, np.ndarray]:
        """Convert a list of dicts into a dict of numpy arrays (one per key)."""
        if not records:
            return {}
        all_keys: list[str] = []
        for r in records:
            for k in r:
                if k not in all_keys:
                    all_keys.append(k)
        arrays = {}
        for k in all_keys:
            arrays[k] = np.array([r.get(k, np.nan) for r in records])
        return arrays

    def _save(self):
        os.makedirs(self.output_dir, exist_ok=True)
        payload = {
            "train": self._records_to_arrays(self.train_step_metrics),
            "val": self._records_to_arrays(self.val_epoch_metrics),
        }
        out_path = os.path.join(self.output_dir, "metrics_history.pt")
        torch.save(payload, out_path)
        logger.info(
            "Saved metrics history (%d train, %d val records) to %s",
            len(self.train_step_metrics),
            len(self.val_epoch_metrics),
            out_path,
        )

    def _save_per_rank_val_loss(self, rank):
        os.makedirs(self.output_dir, exist_ok=True)
        payload = {"val": self._records_to_arrays(self.per_rank_val_loss)}
        out_path = os.path.join(
            self.output_dir, f"metrics_history.rank{rank:03d}.pt"
        )
        torch.save(payload, out_path)
        logger.info(
            "Saved per-rank val_loss history (%d records) to %s",
            len(self.per_rank_val_loss),
            out_path,
        )


def _reset_peak_memory_stats():
    """Reset peak memory counters for the active device backend.

    No-op on CPU.  CUDA and XPU expose the same-named function.
    """
    if BACKEND_NAME == _CUDA:
        torch.cuda.reset_peak_memory_stats()
    elif BACKEND_NAME == _XPU:
        torch.xpu.reset_peak_memory_stats()


def _get_peak_memory_stats():
    """Return ``(peak_allocated_bytes, peak_reserved_bytes)`` for the local device.

    Returns ``(None, None)`` on CPU or if the backend does not expose the
    relevant APIs.  CUDA and XPU expose identical function names.
    """
    if BACKEND_NAME == _CUDA:
        return (torch.cuda.max_memory_allocated(),
                torch.cuda.max_memory_reserved())
    if BACKEND_NAME == _XPU:
        try:
            return (torch.xpu.max_memory_allocated(),
                    torch.xpu.max_memory_reserved())
        except AttributeError:
            return (None, None)
    return (None, None)


class TrainingBenchmarkCallback(pl.Callback):
    """Records per-epoch training wall-clock time, throughput, and peak memory.

    Saves ``benchmark_history.json`` in the artifacts directory.  Opt-in via
    ``--save_benchmark True``.

    Per-epoch peak memory is captured by resetting the device peak memory
    counters at epoch start and reading ``max_memory_allocated`` /
    ``max_memory_reserved`` at epoch end.  By default only rank 0's local
    device is sampled.  Set ``all_ranks_memory=True`` to all-gather peak
    memory from every rank and save ``peak_memory_{allocated,reserved}_gb_per_rank``
    lists in each record.

    Parameters
    ----------
    output_dir : str
        Directory where ``benchmark_history.json`` is written.
    batch_size : int
        Per-device micro-batch size.
    acc_grad_batches : int
        Gradient accumulation steps.
    gpu_devices : int
        Number of GPU devices per node.
    num_nodes : int
        Number of nodes.
    precision : str
        Training precision string (e.g. ``'bf16'``).
    training_strategy : str
        ``'primary_only'`` or ``'combine'``.
    num_workers : int
        DataLoader workers.
    skip_first_epoch : bool
        Exclude the first epoch from summary statistics (warmup effects).
    all_ranks_memory : bool
        All-gather peak memory from every rank and report per-rank lists.
        Default: False (only rank 0's local device is reported).
    """

    def __init__(self, output_dir, *, batch_size, acc_grad_batches,
                 gpu_devices, num_nodes, precision="32",
                 training_strategy="primary_only", num_workers=0,
                 skip_first_epoch=True, all_ranks_memory=False):
        super().__init__()
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.acc_grad_batches = acc_grad_batches
        self.gpu_devices = gpu_devices
        self.num_nodes = num_nodes
        self.effective_batch_size = (
            batch_size * acc_grad_batches * gpu_devices * num_nodes
        )
        self.precision = precision
        self.training_strategy = training_strategy
        self.num_workers = num_workers
        self.skip_first_epoch = skip_first_epoch
        self.all_ranks_memory = all_ranks_memory
        self.backend = BACKEND_NAME

        self._epoch_start_time = None
        self._epoch_start_step = None
        self._interval_start_time = None
        self._interval_start_step = None
        self._epoch_records: list[dict] = []
        self._train_start_timestamp = None

    def on_train_start(self, trainer, pl_module):
        self._train_start_timestamp = datetime.now().isoformat()
        self._interval_start_time = time.perf_counter()
        self._interval_start_step = trainer.global_step
        _reset_peak_memory_stats()

    def on_train_epoch_start(self, trainer, pl_module):
        self._epoch_start_time = time.perf_counter()
        self._epoch_start_step = trainer.global_step
        _reset_peak_memory_stats()

    def on_train_epoch_end(self, trainer, pl_module):
        if self._epoch_start_time is None:
            return
        elapsed = time.perf_counter() - self._epoch_start_time
        steps = trainer.global_step - self._epoch_start_step
        peak_alloc, peak_reserved = _get_peak_memory_stats()
        per_rank_alloc, per_rank_reserved = self._gather_across_ranks(
            pl_module, peak_alloc, peak_reserved,
        )
        record = self._build_record(
            trainer.current_epoch, trainer.global_step, steps, elapsed,
            peak_alloc=peak_alloc, peak_reserved=peak_reserved,
            per_rank_alloc=per_rank_alloc, per_rank_reserved=per_rank_reserved,
        )
        self._epoch_records.append(record)
        self._log_record(trainer, pl_module, record)

    def on_validation_epoch_end(self, trainer, pl_module):
        if self.training_strategy != "combine":
            return
        if self._interval_start_time is None:
            return
        elapsed = time.perf_counter() - self._interval_start_time
        steps = trainer.global_step - self._interval_start_step
        if steps <= 0:
            return
        peak_alloc, peak_reserved = _get_peak_memory_stats()
        per_rank_alloc, per_rank_reserved = self._gather_across_ranks(
            pl_module, peak_alloc, peak_reserved,
        )
        record = self._build_record(
            trainer.current_epoch, trainer.global_step, steps, elapsed,
            peak_alloc=peak_alloc, peak_reserved=peak_reserved,
            per_rank_alloc=per_rank_alloc, per_rank_reserved=per_rank_reserved,
            interval_step_range=[self._interval_start_step,
                                 trainer.global_step],
        )
        self._epoch_records.append(record)
        self._log_record(trainer, pl_module, record)
        self._interval_start_time = time.perf_counter()
        self._interval_start_step = trainer.global_step
        _reset_peak_memory_stats()

    def _gather_across_ranks(self, pl_module, peak_alloc, peak_reserved):
        """All-gather peak memory from every rank.

        Runs on ALL ranks (collective op) but the result is only used on rank 0.
        Returns ``(alloc_list_gb, reserved_list_gb)`` or ``(None, None)`` when
        the flag is off, peak stats are unavailable, or the gather fails.
        """
        if not self.all_ranks_memory:
            return None, None
        if peak_alloc is None:
            return None, None
        local = torch.tensor(
            [float(peak_alloc), float(peak_reserved)],
            device=pl_module.device, dtype=torch.float64,
        )
        try:
            gathered = pl_module.all_gather(local)
        except Exception as e:
            logger.warning("Memory all_gather failed: %s", e)
            return None, None
        if gathered.ndim == 1:
            # Single process: shape (2,)
            alloc_bytes = [float(gathered[0].item())]
            reserved_bytes = [float(gathered[1].item())]
        else:
            # Distributed: shape (world_size, 2)
            alloc_bytes = [float(x.item()) for x in gathered[:, 0]]
            reserved_bytes = [float(x.item()) for x in gathered[:, 1]]
        one_gb = 1024 ** 3
        return (
            [round(a / one_gb, 3) for a in alloc_bytes],
            [round(r / one_gb, 3) for r in reserved_bytes],
        )

    def on_train_end(self, trainer, pl_module):
        if trainer.global_rank != 0:
            return
        self._save()
        records = self._epoch_records
        if self.skip_first_epoch and len(records) > 1:
            records = records[1:]
        if records:
            avg_sps = sum(r["samples_per_sec"] for r in records) / len(records)
            avg_time = (
                sum(r["epoch_wall_time_sec"] for r in records) / len(records)
            )
            mem_parts = []
            peak_allocs = [r["peak_memory_allocated_gb"] for r in records
                           if r.get("peak_memory_allocated_gb") is not None]
            peak_reserveds = [r["peak_memory_reserved_gb"] for r in records
                              if r.get("peak_memory_reserved_gb") is not None]
            if peak_allocs:
                mem_parts.append(
                    f"max {max(peak_allocs):.2f} GB allocated"
                )
            if peak_reserveds:
                mem_parts.append(
                    f"max {max(peak_reserveds):.2f} GB reserved"
                )
            mem_summary = f", {', '.join(mem_parts)}" if mem_parts else ""
            logger.info(
                "Benchmark summary (%d epochs%s): "
                "avg %.1f samples/sec, avg %.1f sec/epoch%s",
                len(records),
                ", first skipped" if self.skip_first_epoch else "",
                avg_sps, avg_time, mem_summary,
            )

    def _build_record(self, epoch, global_step, steps, elapsed,
                      peak_alloc=None, peak_reserved=None,
                      per_rank_alloc=None, per_rank_reserved=None,
                      interval_step_range=None):
        samples = steps * self.effective_batch_size
        record = {
            "epoch": epoch,
            "global_step": global_step,
            "steps_in_epoch": steps,
            "samples_in_epoch": samples,
            "epoch_wall_time_sec": round(elapsed, 3),
            "samples_per_sec": round(samples / elapsed, 1) if elapsed > 0 else 0.0,
            "steps_per_sec": round(steps / elapsed, 3) if elapsed > 0 else 0.0,
            "peak_memory_allocated_gb": (
                round(peak_alloc / (1024 ** 3), 3)
                if peak_alloc is not None else None
            ),
            "peak_memory_reserved_gb": (
                round(peak_reserved / (1024 ** 3), 3)
                if peak_reserved is not None else None
            ),
        }
        if per_rank_alloc is not None:
            record["peak_memory_allocated_gb_per_rank"] = per_rank_alloc
        if per_rank_reserved is not None:
            record["peak_memory_reserved_gb_per_rank"] = per_rank_reserved
        if interval_step_range is not None:
            record["interval_step_range"] = interval_step_range
        return record

    def _log_record(self, trainer, pl_module, record):
        if trainer.global_rank != 0:
            return
        pl_module.log("benchmark/epoch_wall_time_sec",
                      float(record["epoch_wall_time_sec"]),
                      on_step=False, on_epoch=True, rank_zero_only=True)
        pl_module.log("benchmark/samples_per_sec",
                      float(record["samples_per_sec"]),
                      on_step=False, on_epoch=True, rank_zero_only=True)
        pl_module.log("benchmark/steps_per_sec",
                      float(record["steps_per_sec"]),
                      on_step=False, on_epoch=True, rank_zero_only=True)
        if record.get("peak_memory_allocated_gb") is not None:
            pl_module.log("benchmark/peak_memory_allocated_gb",
                          float(record["peak_memory_allocated_gb"]),
                          on_step=False, on_epoch=True, rank_zero_only=True)
        if record.get("peak_memory_reserved_gb") is not None:
            pl_module.log("benchmark/peak_memory_reserved_gb",
                          float(record["peak_memory_reserved_gb"]),
                          on_step=False, on_epoch=True, rank_zero_only=True)
        mem_str = ""
        per_rank_alloc = record.get("peak_memory_allocated_gb_per_rank")
        per_rank_reserved = record.get("peak_memory_reserved_gb_per_rank")
        if per_rank_alloc is not None:
            alloc_fmt = ",".join(f"{v:.2f}" for v in per_rank_alloc)
            mem_str += f"  peak_alloc_per_rank=[{alloc_fmt}]GB"
        elif record.get("peak_memory_allocated_gb") is not None:
            mem_str += f"  peak_alloc={record['peak_memory_allocated_gb']:.2f}GB"
        if per_rank_reserved is not None:
            reserved_fmt = ",".join(f"{v:.2f}" for v in per_rank_reserved)
            mem_str += f"  peak_reserved_per_rank=[{reserved_fmt}]GB"
        elif record.get("peak_memory_reserved_gb") is not None:
            mem_str += f"  peak_reserved={record['peak_memory_reserved_gb']:.2f}GB"
        logger.info(
            "Benchmark  epoch=%d  step=%d  time=%.1fs  "
            "samples/sec=%.1f  steps/sec=%.3f  "
            "effective_batch=%d%s",
            record["epoch"], record["global_step"],
            record["epoch_wall_time_sec"],
            record["samples_per_sec"], record["steps_per_sec"],
            self.effective_batch_size, mem_str,
        )

    def _save(self):
        os.makedirs(self.output_dir, exist_ok=True)
        payload = {
            "config": {
                "batch_size": self.batch_size,
                "acc_grad_batches": self.acc_grad_batches,
                "gpu_devices": self.gpu_devices,
                "num_nodes": self.num_nodes,
                "effective_batch_size": self.effective_batch_size,
                "num_workers": self.num_workers,
                "precision": self.precision,
                "training_strategy": self.training_strategy,
                "backend": self.backend,
                "hostname": socket.gethostname(),
                "train_start_timestamp": self._train_start_timestamp,
            },
            "epochs": self._epoch_records,
        }
        out_path = os.path.join(self.output_dir, "benchmark_history.json")
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
        logger.info("Saved benchmark history (%d records) to %s",
                     len(self._epoch_records), out_path)


class _CheckpointLogMixin:
    """Mixin that logs checkpoint saves and best-score updates to the file logger."""

    def _save_checkpoint(self, trainer, filepath):
        super()._save_checkpoint(trainer, filepath)
        if trainer.global_rank != 0:
            return
        score = self.current_score
        score_str = f"{float(score):.5f}" if score is not None else "n/a"
        best = self.best_model_score
        best_str = f"{float(best):.5f}" if best is not None else "n/a"
        logger.info(
            "Checkpoint saved  epoch=%d  step=%d  %s=%s  best=%s  path=%s",
            trainer.current_epoch, trainer.global_step,
            self.monitor or "none", score_str, best_str,
            os.path.basename(filepath),
        )


class LoggingModelCheckpoint(_CheckpointLogMixin, _ModelCheckpoint):
    """Standard ModelCheckpoint with file-logger output on each save."""
    pass


class SyncSafeModelCheckpoint(_CheckpointLogMixin, _ModelCheckpoint):
    """ModelCheckpoint that bypasses ``reduce_boolean_decision``.

    Lightning's ``check_monitor_top_k`` calls
    ``strategy.reduce_boolean_decision(decision, all=True)`` which does a
    ``ReduceOp.SUM`` all-reduce on an **integer** tensor and checks
    ``sum == world_size``.  On Intel XPU with the CCL backend this
    all-reduce silently returns an incorrect value, causing the
    checkpoint callback to reject every improvement after the first
    ``save_top_k`` saves (those bypass the consensus entirely).

    Since ``sync_dist=True`` on the logged metric already ensures all
    ranks see an identical ``val_loss`` value (verified empirically), the
    per-rank comparison is deterministic and consensus is redundant.
    This subclass simply removes the ``reduce_boolean_decision`` call.
    """

    def check_monitor_top_k(self, trainer, current=None):
        if current is None:
            return False
        if self.save_top_k == -1:
            return True

        less_than_k_models = len(self.best_k_models) < self.save_top_k
        if less_than_k_models:
            return True

        monitor_op = {"min": torch.lt, "max": torch.gt}[self.mode]
        return bool(monitor_op(
            current, self.best_k_models[self.kth_best_model_path]
        ))

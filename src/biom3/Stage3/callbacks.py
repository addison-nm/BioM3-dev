"""Custom PyTorch Lightning callbacks for Stage 3 training."""

import json
import os
import socket
import time
from datetime import datetime

import numpy as np
import torch

from biom3.backend.device import (
    BACKEND_NAME,
    _CUDA,
    _XPU,
    setup_logger,
    reset_peak_memory_stats as _reset_peak_memory_stats,
    get_peak_memory_stats as _get_peak_memory_stats,
)

if BACKEND_NAME == _XPU:
    import lightning as pl
    from lightning.pytorch.callbacks import ModelCheckpoint as _ModelCheckpoint
else:
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint as _ModelCheckpoint

logger = setup_logger(__name__)


def _json_default(value):
    """Coerce common torch/numpy scalar types to JSON-safe primitives."""
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    raise TypeError(f"Not JSON serializable: {type(value).__name__}")


def _read_jsonl(path):
    if not os.path.exists(path):
        return []
    records = []
    with open(path, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                # Last line may be truncated after a crash; skip it.
                continue
    return records


def rebuild_metrics_history_pt(output_dir):
    """Reconstruct ``metrics_history.pt`` from the streaming JSONL files.

    Call this after a crashed/killed training run to recover whatever
    metrics were flushed to disk before the process died.
    """
    train = _read_jsonl(
        os.path.join(output_dir, MetricsHistoryCallback.TRAIN_JSONL)
    )
    val = _read_jsonl(
        os.path.join(output_dir, MetricsHistoryCallback.VAL_JSONL)
    )
    payload = {
        "train": MetricsHistoryCallback._records_to_arrays(train),
        "val": MetricsHistoryCallback._records_to_arrays(val),
    }
    out_path = os.path.join(output_dir, "metrics_history.pt")
    torch.save(payload, out_path)
    return out_path, len(train), len(val)


class MetricsHistoryCallback(pl.Callback):
    """Accumulates training and validation metrics and saves to a .pt file.

    The saved file (``metrics_history.pt``) is a dict with two keys:

    - ``"train"``: dict mapping metric names to 1-D numpy arrays, plus
      ``"global_step"`` and ``"epoch"`` index arrays.
    - ``"val"``: same structure for validation metrics (including ``"loss_gap"``).

    Streaming ``metrics_history.{train,val}.jsonl`` files are also written
    incrementally so metrics survive crashes and timeouts even if
    ``on_train_end`` never fires.

    Parameters
    ----------
    output_dir : str
        Directory where ``metrics_history.pt`` is written.
    save_ranks : list[int]
        Rank indices on which to record and save train/val metrics
        (default: ``[0]``).
    every_n_steps : int
        Record training metrics every *n* global steps (default: ``1``).
    every_n_epochs : int | None
        Additionally record training metrics at the end of every *n* training
        epochs. Captures the epoch-averaged ``train_*_epoch`` values rather
        than noisy step-level values. ``None`` disables epoch-level records
        (default: ``None``).
    flush_every_n_steps : int | None
        Flush pending JSONL records to disk (fsync) every *n* global steps.
        ``None`` disables periodic flushing — records are only guaranteed on
        disk once ``on_train_end`` runs (default: ``None``).
    all_ranks_val_loss : bool
        When True, record ``val_loss`` and ``val_loss_epoch`` on **every** rank
        at validation epoch end and dump one ``metrics_history.rank{N}.pt`` per
        rank under ``output_dir``. Used to diagnose cross-rank sync_dist issues
        in ``ModelCheckpoint`` (the consensus ``reduce_boolean_decision`` call
        requires all ranks to agree that val_loss improved). Default: False.
    """

    TRAIN_JSONL = "metrics_history.train.jsonl"
    VAL_JSONL = "metrics_history.val.jsonl"

    def __init__(self, output_dir, save_ranks=None, every_n_steps=1,
                 every_n_epochs=None, flush_every_n_steps=None,
                 all_ranks_val_loss=False):
        super().__init__()
        self.output_dir = output_dir
        self.save_ranks = save_ranks or [0]
        self.every_n_steps = max(1, every_n_steps)
        self.every_n_epochs = (
            max(1, int(every_n_epochs)) if every_n_epochs else None
        )
        self.flush_every_n_steps = (
            max(1, int(flush_every_n_steps)) if flush_every_n_steps else None
        )
        self.all_ranks_val_loss = all_ranks_val_loss
        self.train_step_metrics: list[dict] = []
        self.val_epoch_metrics: list[dict] = []
        self.per_rank_val_loss: list[dict] = []
        self._train_jsonl_fh = None
        self._val_jsonl_fh = None
        self._unflushed_since_fsync = 0

    def _open_jsonl_streams(self):
        if self._train_jsonl_fh is not None:
            return
        os.makedirs(self.output_dir, exist_ok=True)
        self._train_jsonl_fh = open(
            os.path.join(self.output_dir, self.TRAIN_JSONL), "a"
        )
        self._val_jsonl_fh = open(
            os.path.join(self.output_dir, self.VAL_JSONL), "a"
        )

    def _close_jsonl_streams(self):
        for attr in ("_train_jsonl_fh", "_val_jsonl_fh"):
            fh = getattr(self, attr, None)
            if fh is not None:
                try:
                    fh.flush()
                    os.fsync(fh.fileno())
                except OSError:
                    pass
                fh.close()
                setattr(self, attr, None)

    def _append_jsonl(self, fh, record):
        if fh is None:
            return
        fh.write(json.dumps(record, default=_json_default) + "\n")

    def _fsync_streams(self):
        for fh in (self._train_jsonl_fh, self._val_jsonl_fh):
            if fh is None:
                continue
            try:
                fh.flush()
                os.fsync(fh.fileno())
            except OSError:
                pass
        self._unflushed_since_fsync = 0

    def _maybe_fsync(self, trainer):
        if self.flush_every_n_steps is None:
            return
        self._unflushed_since_fsync += 1
        if trainer.global_step % self.flush_every_n_steps != 0:
            return
        self._fsync_streams()

    def on_train_start(self, trainer, pl_module):
        if trainer.global_rank in self.save_ranks:
            self._open_jsonl_streams()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_rank not in self.save_ranks:
            return
        if trainer.global_step % self.every_n_steps != 0:
            return
        logged = trainer.callback_metrics
        record = {
            "global_step": trainer.global_step,
            "epoch": trainer.current_epoch,
            "source": "step",
        }
        for k, v in logged.items():
            if k.startswith("train_"):
                record[k] = v.item() if hasattr(v, "item") else v
        self.train_step_metrics.append(record)
        self._append_jsonl(self._train_jsonl_fh, record)
        self._maybe_fsync(trainer)

    def on_train_epoch_end(self, trainer, pl_module):
        if self.every_n_epochs is None:
            return
        if trainer.global_rank not in self.save_ranks:
            return
        epoch = trainer.current_epoch
        if (epoch + 1) % self.every_n_epochs != 0:
            return
        logged = trainer.callback_metrics
        record = {
            "global_step": trainer.global_step,
            "epoch": epoch,
            "source": "epoch",
        }
        for k, v in logged.items():
            if k.startswith("train_"):
                record[k] = v.item() if hasattr(v, "item") else v
        self.train_step_metrics.append(record)
        self._append_jsonl(self._train_jsonl_fh, record)
        self._maybe_fsync(trainer)

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
        self._append_jsonl(self._val_jsonl_fh, record)
        # Validation epochs are coarse — always fsync the JSONL streams here so
        # at most one val cycle's worth of metrics is at risk on a SIGKILL,
        # regardless of flush_every_n_steps. Then snapshot the consolidated
        # metrics_history.pt so it's recoverable without rebuild_metrics_history_pt.
        self._fsync_streams()
        self._save()

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
            self._close_jsonl_streams()
            return
        self._save()
        out_path = os.path.join(self.output_dir, "metrics_history.pt")
        logger.info(
            "Final metrics history (%d train, %d val records) at %s",
            len(self.train_step_metrics),
            len(self.val_epoch_metrics),
            out_path,
        )
        self._close_jsonl_streams()

    def on_exception(self, trainer, pl_module, exception):
        """Best-effort flush if training crashes before ``on_train_end``."""
        self._close_jsonl_streams()

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
        # _save runs at every validation-epoch end (timeout-resilient
        # snapshotting) and once at train end; demote the per-epoch lines to
        # debug to keep run.log clean. The end-of-train summary is logged by
        # on_train_end's downstream callers.
        logger.debug(
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
    per_step : bool
        Record per-step wall-clock timing to ``benchmark_steps.jsonl``
        (rank 0 only). Each record contains ``epoch``, ``global_step``,
        ``batch_idx``, and ``step_wall_time_sec``. Memory is not sampled
        per-step to avoid the device sync required by a peak-memory reset
        (peak memory stays at the epoch granularity). Default: False.
    """

    STEP_JSONL = "benchmark_steps.jsonl"

    def __init__(self, output_dir, *, batch_size, acc_grad_batches,
                 gpu_devices, num_nodes, precision="32",
                 training_strategy="primary_only", num_workers=0,
                 skip_first_epoch=True, all_ranks_memory=False,
                 per_step=False):
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
        self.per_step = per_step
        self.backend = BACKEND_NAME

        self._epoch_start_time = None
        self._epoch_start_step = None
        self._interval_start_time = None
        self._interval_start_step = None
        self._epoch_records: list[dict] = []
        self._train_start_timestamp = None
        self._step_start_time = None
        self._step_jsonl_fh = None
        self._step_records_count = 0

    def on_train_start(self, trainer, pl_module):
        self._train_start_timestamp = datetime.now().isoformat()
        self._interval_start_time = time.perf_counter()
        self._interval_start_step = trainer.global_step
        _reset_peak_memory_stats()
        if self.per_step and trainer.global_rank == 0:
            os.makedirs(self.output_dir, exist_ok=True)
            self._step_jsonl_fh = open(
                os.path.join(self.output_dir, self.STEP_JSONL), "a"
            )

    def on_train_epoch_start(self, trainer, pl_module):
        self._epoch_start_time = time.perf_counter()
        self._epoch_start_step = trainer.global_step
        _reset_peak_memory_stats()

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if not self.per_step or self._step_jsonl_fh is None:
            return
        self._step_start_time = time.perf_counter()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not self.per_step or self._step_jsonl_fh is None:
            return
        if self._step_start_time is None:
            return
        elapsed = time.perf_counter() - self._step_start_time
        record = {
            "epoch": trainer.current_epoch,
            "global_step": trainer.global_step,
            "batch_idx": batch_idx,
            "step_wall_time_sec": round(elapsed, 6),
        }
        self._step_jsonl_fh.write(json.dumps(record) + "\n")
        self._step_records_count += 1
        self._step_start_time = None

    def _close_step_jsonl(self):
        fh = self._step_jsonl_fh
        if fh is None:
            return
        try:
            fh.flush()
            os.fsync(fh.fileno())
        except OSError:
            pass
        fh.close()
        self._step_jsonl_fh = None

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
        # Snapshot the rolled-up benchmark_history.json each epoch so a
        # timeout-killed run still has the per-epoch summary on disk.
        if trainer.global_rank == 0:
            self._save()

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
        # Step-based ('combine') strategy: snapshot benchmark_history.json on
        # each val cycle so it's not lost on timeout.
        if trainer.global_rank == 0:
            self._save()
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
        self._close_step_jsonl()
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

    def on_exception(self, trainer, pl_module, exception):
        """Best-effort flush of per-step JSONL on crash."""
        self._close_step_jsonl()

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
                "per_step": self.per_step,
            },
            "epochs": self._epoch_records,
        }
        if self.per_step:
            payload["config"]["per_step_records"] = self._step_records_count
        out_path = os.path.join(self.output_dir, "benchmark_history.json")
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
        # _save runs per-epoch (timeout-resilient snapshotting) plus once at
        # train end. The per-epoch invocations would spam the file log; emit
        # only at debug level. on_train_end already logs its own summary.
        logger.debug("Saved benchmark history (%d records) to %s",
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


def build_checkpoint_callbacks(
    *,
    checkpoint_dir,
    checkpoint_monitors=None,
    periodic_every_n_steps=None,
    periodic_every_n_epochs=None,
    periodic_max_keep=-1,
    periodic_subdir="periodic",
    use_sync_safe=False,
    primary_save_top_k=2,
    primary_save_last="link",
):
    """Build orthogonal monitored and periodic checkpoint callbacks.

    Two independent families of callbacks are returned:

    * **Monitored** — one ModelCheckpoint per entry in
      ``checkpoint_monitors``. Index 0 is the primary monitor
      (``save_top_k=primary_save_top_k`` + ``save_last=primary_save_last``);
      the rest keep best-1 each, with metric-slug filenames. These never
      use the ``every_n_*`` knobs — keeping them off the monitored callbacks
      prevents periodic-trigger saves from being silently pruned by the
      ``save_top_k`` retention cap (the bug Sophie's prototype fixed).
    * **Periodic** — a separate, monitor-free ModelCheckpoint that snapshots
      every N training steps and/or every M epochs into a ``periodic_subdir``
      subdirectory. ``save_top_k=periodic_max_keep`` (default ``-1`` keeps
      everything). Returned as ``None`` when both periodic cadences are unset.

    Parameters
    ----------
    checkpoint_dir : str
        Root checkpoint directory. Monitored checkpoints land here directly;
        periodic snapshots go into ``checkpoint_dir/periodic_subdir``.
    checkpoint_monitors : list[dict] | str | None
        List of ``{"metric": ..., "mode": ...}`` dicts, or a JSON string of
        the same. ``None`` defaults to ``[{"metric": "val_loss", "mode": "min"}]``.
    periodic_every_n_steps, periodic_every_n_epochs : int | None
        Cadence(s) for the periodic snapshot callback. At least one must be
        set for the periodic callback to be created.
    periodic_max_keep : int
        ``save_top_k`` for the periodic callback. Must be one of:
        ``-1`` (keep all, default), ``0`` (disable saving), or ``1`` (keep
        only the most recent). Lightning rejects other values when
        ``monitor`` is None — implementing arbitrary "keep last N" retention
        would require a custom pruning layer.
    periodic_subdir : str
        Subdirectory name under ``checkpoint_dir`` for periodic snapshots.
    use_sync_safe : bool
        Use ``SyncSafeModelCheckpoint`` instead of ``LoggingModelCheckpoint``
        for monitored callbacks. The periodic callback always uses
        ``LoggingModelCheckpoint`` (the SyncSafe override only matters when
        comparing against a monitored metric).
    primary_save_top_k, primary_save_last : int, "link"|bool|str
        Forwarded to the primary monitored callback.

    Returns
    -------
    monitored : list[ModelCheckpoint]
    periodic : ModelCheckpoint | None
    """
    if checkpoint_monitors is None:
        checkpoint_monitors = [{"metric": "val_loss", "mode": "min"}]
    elif isinstance(checkpoint_monitors, str):
        checkpoint_monitors = json.loads(checkpoint_monitors)

    cls = SyncSafeModelCheckpoint if use_sync_safe else LoggingModelCheckpoint

    monitored = []
    for i, mon in enumerate(checkpoint_monitors):
        kwargs = dict(
            dirpath=checkpoint_dir,
            verbose=True,
            monitor=mon["metric"],
            mode=mon["mode"],
            enable_version_counter=False,
        )
        if i == 0:
            kwargs["save_top_k"] = primary_save_top_k
            kwargs["save_last"] = primary_save_last
        else:
            metric_slug = mon["metric"].replace("/", "_")
            kwargs["save_top_k"] = 1
            kwargs["save_last"] = False
            kwargs["filename"] = f"best-{metric_slug}-{{epoch}}"
        monitored.append(cls(**kwargs))

    periodic = None
    if periodic_every_n_steps is not None and periodic_every_n_epochs is not None:
        # Lightning's ModelCheckpoint requires these to be mutually exclusive
        # (see ModelCheckpoint.__validate_init_configuration). Surface a clear
        # error here rather than at Trainer construction time.
        raise ValueError(
            "periodic_every_n_steps and periodic_every_n_epochs are mutually "
            "exclusive — pass one or the other, not both."
        )
    if int(periodic_max_keep) not in (-1, 0, 1):
        # Lightning rejects monitor=None with save_top_k outside {-1, 0, 1}
        # ("No quantity for top_k to track"). Implementing custom "keep last
        # N" retention would mean writing our own pruning layer; out of scope.
        raise ValueError(
            f"periodic_max_keep={periodic_max_keep} is not supported. "
            "Allowed values: -1 (keep all, default), 0 (disable), 1 (keep "
            "only the most recent). Lightning's ModelCheckpoint requires "
            "save_top_k in {-1, 0, 1} when monitor is None."
        )
    if periodic_every_n_steps is not None or periodic_every_n_epochs is not None:
        kwargs = dict(
            dirpath=os.path.join(checkpoint_dir, periodic_subdir),
            filename="periodic-{epoch:04d}-{step:08d}",
            monitor=None,
            save_top_k=int(periodic_max_keep),
            save_last=False,
            enable_version_counter=False,
            verbose=True,
        )
        if periodic_every_n_steps is not None:
            kwargs["every_n_train_steps"] = int(periodic_every_n_steps)
        if periodic_every_n_epochs is not None:
            kwargs["every_n_epochs"] = int(periodic_every_n_epochs)
        periodic = LoggingModelCheckpoint(**kwargs)

    return monitored, periodic


class BestArtifactSyncCallback(pl.Callback):
    """Materialize the ``state_dict.best.pth`` artifacts mid-training.

    Lightning's ``ModelCheckpoint`` writes its best ``.ckpt`` progressively,
    but the *derived* artifacts (``state_dict.best.pth``, the per-monitor
    ``state_dict.best_<metric>.pth`` copies, and ``checkpoint_summary.json``)
    are only produced by ``save_model`` after ``trainer.fit()`` returns. A
    SIGTERM/timeout mid-run leaves those artifacts missing and forces a
    manual DeepSpeed-conversion to recover the weights.

    This callback runs the same conversion pipeline at validation-epoch end,
    throttled by ``every_n_val`` and skipped when no new "best" has been
    written since the last sync. Saves are rank-0 only.

    Parameters
    ----------
    sync_fn : callable
        Function with signature ``(primary_callback, extra_callbacks) -> None``.
        Provided by ``run_PL_training`` so this callback stays model-agnostic
        (avoids a circular import between callbacks.py and run_PL_training.py).
    primary_callback : ModelCheckpoint
        The monitored callback whose ``best_model_path`` drives the sync.
    extra_callbacks : list[ModelCheckpoint] | None
        Additional monitored callbacks whose best checkpoints should also be
        synced.
    every_n_val : int
        Sync every Nth validation epoch (default ``1`` = every val cycle).
    """

    def __init__(self, *, sync_fn, primary_callback, extra_callbacks=None,
                 every_n_val=1):
        super().__init__()
        self.sync_fn = sync_fn
        self.primary_callback = primary_callback
        self.extra_callbacks = list(extra_callbacks or [])
        self.every_n_val = max(1, int(every_n_val))
        self._val_cycle_counter = 0
        self._last_synced_best_path = None

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.global_rank != 0:
            return
        self._val_cycle_counter += 1
        if self._val_cycle_counter % self.every_n_val != 0:
            return
        current_best = self.primary_callback.best_model_path
        if not current_best:
            return
        if current_best == self._last_synced_best_path:
            # ModelCheckpoint hasn't promoted a new best since our last sync;
            # the on-disk artifact already reflects this checkpoint.
            return
        try:
            self.sync_fn(
                primary_callback=self.primary_callback,
                extra_callbacks=self.extra_callbacks,
            )
            self._last_synced_best_path = current_best
        except Exception as e:
            # Don't let an artifact-sync failure abort training. The .ckpt is
            # still preserved by ModelCheckpoint and recoverable post-hoc.
            logger.warning(
                "BestArtifactSyncCallback: sync failed (%s). Training "
                "continues; checkpoint .ckpt is still on disk.", e,
            )


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

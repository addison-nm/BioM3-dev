"""Custom PyTorch Lightning callbacks for Stage 3 training."""

import os

import numpy as np
import torch

from biom3.backend.device import BACKEND_NAME, _XPU, setup_logger

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


class SyncSafeModelCheckpoint(_ModelCheckpoint):
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
        should_update = bool(monitor_op(
            current, self.best_k_models[self.kth_best_model_path]
        ))

        if trainer.global_rank == 0:
            logger.info(
                "[SyncSafeCkpt] epoch=%d step=%d: current=%.6f  "
                "kth_best=%.6f  improved=%s",
                trainer.current_epoch, trainer.global_step,
                float(current),
                float(self.best_k_models[self.kth_best_model_path]),
                should_update,
            )

        return should_update

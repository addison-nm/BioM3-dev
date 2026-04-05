"""Custom PyTorch Lightning callbacks for Stage 3 training."""

import os

import numpy as np
import torch

from biom3.backend.device import BACKEND_NAME, _XPU, setup_logger

if BACKEND_NAME == _XPU:
    import lightning as pl
else:
    import pytorch_lightning as pl

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
        Rank indices on which to record and save metrics (default: ``[0]``).
    every_n_steps : int
        Record training metrics every *n* global steps (default: ``1``).
    """

    def __init__(self, output_dir, save_ranks=None, every_n_steps=1):
        super().__init__()
        self.output_dir = output_dir
        self.save_ranks = save_ranks or [0]
        self.every_n_steps = max(1, every_n_steps)
        self.train_step_metrics: list[dict] = []
        self.val_epoch_metrics: list[dict] = []

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

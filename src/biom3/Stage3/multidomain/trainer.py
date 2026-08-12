"""Trainer and distributed-strategy construction for multidomain runs.

Sized to what multidomain needs, rather than reusing the Stage 3 HDF5 trainer:
that one carries a two-corpus phase-transfer mode and Pfam step-based switching
that have no meaning here, and reads them off ``args`` unconditionally.

The strategy kwargs, however, are *not* free choices. Each encodes a distinct
failure observed on Aurora, so they are copied deliberately and pinned by
``test_strategy_flags`` rather than re-derived:

* ``overlap_comm=False`` on XPU — overlapping reduction produces nondeterministic
  bucket ordering across ranks on oneCCL, which deadlocks on a mismatched
  collective.
* ``process_group_backend='xccl'`` on XPU — frameworks/2025.3.1 removed the
  ``oneccl-bindings-for-pytorch`` module and replaced the ``ccl`` backend with
  torch's native ``xccl``.
* ``static_graph=True`` for DDP — precomputes the gradient-bucket ready order at
  iteration 0, removing the dynamic-hook race behind the same class of deadlock.
"""

import os

from biom3.backend.device import BACKEND_NAME, _XPU, setup_logger

if BACKEND_NAME == _XPU:
    import lightning as pl
    from lightning import Trainer
    from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
    from lightning.pytorch.loggers import CSVLogger
    from lightning.pytorch.strategies import DDPStrategy, DeepSpeedStrategy
else:
    import pytorch_lightning as pl
    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
    from pytorch_lightning.loggers import CSVLogger
    from pytorch_lightning.strategies import DDPStrategy, DeepSpeedStrategy

logger = setup_logger(__name__)

DEEPSPEED = "deepspeed"
DDP = "ddp"


def build_deepspeed_strategy():
    """ZeRO-2, with the XPU-conditional flags described in the module docstring."""
    return DeepSpeedStrategy(
        stage=2,
        allgather_bucket_size=int(5e8),
        reduce_bucket_size=int(5e8),
        contiguous_gradients=True,
        overlap_comm=(BACKEND_NAME != _XPU),
        process_group_backend="xccl" if BACKEND_NAME == _XPU else None,
    )


def build_ddp_strategy():
    return DDPStrategy(
        process_group_backend="xccl" if BACKEND_NAME == _XPU else None,
        static_graph=True,
        gradient_as_bucket_view=True,
    )


def resolve_accelerator():
    if BACKEND_NAME == _XPU:
        return "xpu"
    return "gpu" if BACKEND_NAME == "cuda" else "cpu"


def build_strategy(name, *, devices, num_nodes):
    """Distributed strategy, or ``"auto"`` for a single-process run."""
    if devices <= 1 and num_nodes <= 1:
        return "auto"
    if name == DEEPSPEED:
        return build_deepspeed_strategy()
    if name == DDP:
        return build_ddp_strategy()
    raise ValueError(
        f"unknown distributed_strategy {name!r}; expected {DEEPSPEED!r} or {DDP!r}")


def build_callbacks(args, checkpoint_dir):
    """Checkpointing plus an LR monitor.

    ``save_last`` is what makes a preempted run resumable, and the periodic
    checkpoint is kept separate from the monitored one so a diverging val loss
    cannot leave a long run with nothing on disk.
    """
    monitored = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="best-{epoch:03d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=getattr(args, "save_top_k", 3),
        save_last=True,
        auto_insert_metric_name=False,
    )
    callbacks = [monitored, LearningRateMonitor(logging_interval="step")]

    every_n_epochs = getattr(args, "checkpoint_every_n_epochs", None)
    if every_n_epochs:
        callbacks.append(ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="epoch-{epoch:03d}",
            every_n_epochs=every_n_epochs,
            save_top_k=-1,
            auto_insert_metric_name=False,
        ))
    return callbacks


def find_resume_checkpoint(checkpoint_dir):
    """``last.ckpt`` when it exists, so chained or preempted jobs continue."""
    last = os.path.join(checkpoint_dir, "last.ckpt")
    if os.path.exists(last):
        return last
    return None


def build_trainer(args, *, checkpoint_dir, logs_dir):
    devices = int(getattr(args, "devices_per_node", 1))
    num_nodes = int(getattr(args, "num_nodes", 1))
    strategy = build_strategy(
        getattr(args, "distributed_strategy", DEEPSPEED),
        devices=devices, num_nodes=num_nodes,
    )
    accelerator = resolve_accelerator()
    logger.info(
        "Trainer: accelerator=%s devices=%s num_nodes=%s strategy=%s",
        accelerator, devices, num_nodes,
        strategy if isinstance(strategy, str) else type(strategy).__name__,
    )
    return Trainer(
        max_epochs=args.epochs,
        max_steps=getattr(args, "max_steps", -1),
        accelerator=accelerator,
        devices=devices,
        num_nodes=num_nodes,
        strategy=strategy,
        precision=getattr(args, "precision", "bf16"),
        accumulate_grad_batches=getattr(args, "acc_grad_batches", 1),
        log_every_n_steps=getattr(args, "log_every_n_steps", 50),
        limit_val_batches=getattr(args, "limit_val_batches", 1.0),
        callbacks=build_callbacks(args, checkpoint_dir),
        logger=CSVLogger(logs_dir, name="multidomain"),
        # The data module attaches its own DistributedSampler with drop_last=True.
        use_distributed_sampler=False,
    )

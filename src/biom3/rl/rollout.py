"""Multi-device rollout pool for GDPO/GRPO.

Replicates the trainable Stage 3 policy onto each device in a list,
dispatches rollout chunks in parallel via threads, and aggregates
results back to the master device. Gradient updates remain
single-device on the master — this pool is *only* for the no-grad
diffusion rollout, which dominates per-step wallclock at L=1024 / G=8.

Threading model
---------------
PyTorch's per-device kernel queues run independently; the GIL only
serializes Python-side dispatch (small relative to a 128-step
diffusion rollout). ``torch.autocast`` is thread-local, so each
worker's autocast context (set inside
``Stage3.sampling_analysis._inference_autocast``) doesn't leak across
threads.

Two non-obvious correctness details:

1. ``cfg3.device`` is mutated in-place by the trainer
   (``biom3.rl.gdpo.gdpo_train``). To avoid races, the pool keeps a
   ``copy.deepcopy(cfg3)`` per tile with its ``.device`` patched once
   at construction; the master's cfg3 is never touched here.
2. Replicas are kept in ``.eval()`` with ``requires_grad=False``. They
   exist only to forward-rollout, never to backward through. The
   ``sync_from`` call broadcasts a fresh ``state_dict`` from the
   master each outer iteration.

Usage::

    pool = RolloutPool(
        s3_master=s3,
        cfg3=cfg3,
        rollout_fn=_gdpo_rollout,                     # the per-tile worker
        devices=[torch.device("xpu:0"), ..., torch.device("xpu:5")],
    )
    pool.sync_from(old_s3)                            # once per outer iter
    ids = pool.rollout(z_c, G=24)                     # (G, L_total) on devices[0]
"""

import copy
import threading
from argparse import Namespace
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, List, Optional, Sequence

import torch

from biom3.backend.device import setup_logger

logger = setup_logger(__name__)


def _split_evenly(total: int, n_buckets: int) -> List[int]:
    """Round-robin split: ``sum(out) == total``, ``len(out) == n_buckets``,
    bucket sizes differ by at most 1. Earlier buckets get the extras."""
    if n_buckets <= 0:
        raise ValueError(f"n_buckets must be positive, got {n_buckets}")
    base = total // n_buckets
    extras = total % n_buckets
    return [base + (1 if i < extras else 0) for i in range(n_buckets)]


def _device_synchronize(device: torch.device) -> None:
    """Best-effort device sync, mirroring Stage3.sampling_analysis._device_synchronize."""
    if device.type == "xpu" and hasattr(torch, "xpu"):
        try:
            torch.xpu.synchronize(device)
        except Exception:
            pass
    elif device.type == "cuda":
        try:
            torch.cuda.synchronize(device)
        except Exception:
            pass


class RolloutPool:
    """Holds replica policies on N devices and dispatches rollouts in parallel.

    Args:
        s3_master: The trainable policy on the master device. Used as the
            seed for replica weights. Never mutated by the pool itself.
        cfg3: Stage 3 namespace. Deep-copied per tile; the originals are
            untouched.
        rollout_fn: Callable with the signature
            ``rollout_fn(model, cfg3, z_c, K, device) -> Tensor``.
            For GDPO this is ``biom3.rl.gdpo._gdpo_rollout``. Pulled in
            as an arg to keep this module decoupled from gdpo's internals
            (and to ease testing with a stub).
        devices: List of devices. ``devices[0]`` is the master; replicas
            are created on ``devices[1:]``.
    """

    def __init__(
        self,
        s3_master: torch.nn.Module,
        cfg3: Namespace,
        rollout_fn: Callable[..., torch.Tensor],
        devices: Sequence[torch.device],
    ):
        if not devices:
            raise ValueError("RolloutPool needs at least one device")
        self.devices: List[torch.device] = [torch.device(d) for d in devices]
        self.master_device = self.devices[0]
        self.rollout_fn = rollout_fn
        self._lock = threading.Lock()  # serializes sync_from to be safe

        # The master tile reuses s3_master directly (no extra copy → no
        # extra HBM). Replicas live on devices[1:].
        self._models: List[torch.nn.Module] = [s3_master]
        for d in self.devices[1:]:
            replica = copy.deepcopy(s3_master).to(d).eval()
            for p in replica.parameters():
                p.requires_grad_(False)
            self._models.append(replica)

        # Per-tile deep-copied cfg3 with .device patched. We never touch
        # the master cfg3 (the trainer relies on it being correct).
        self._cfgs: List[Namespace] = []
        for d in self.devices:
            cfg_d = copy.deepcopy(cfg3)
            cfg_d.device = str(d)
            self._cfgs.append(cfg_d)

        # ThreadPool reused across rollouts — startup cost is small but
        # not free; reusing avoids per-step thread creation churn.
        self._pool = ThreadPoolExecutor(max_workers=len(self.devices))

        logger.info(
            "RolloutPool initialized: master=%s replicas=%s",
            self.master_device, [str(d) for d in self.devices[1:]],
        )

    def sync_from(self, source_model: torch.nn.Module) -> None:
        """Broadcast ``source_model``'s weights to all replicas.

        Called once per outer iteration to make the replicas match
        ``π_old`` (the deepcopy of the trainable policy taken at the
        start of the iteration). Replicas on ``devices[1:]`` get the
        state_dict; the master tile shares state with the original.
        """
        if len(self.devices) == 1:
            return  # nothing to sync
        with self._lock:
            sd = source_model.state_dict()
            for d, replica in zip(self.devices[1:], self._models[1:]):
                # ``map_location``-style: load the state_dict onto the
                # replica's device. PyTorch's load_state_dict handles
                # device placement when the destination tensors are
                # already on the target device (which they are — replica
                # was .to(d) at construction).
                replica.load_state_dict(sd)

    def rollout(self, z_c: torch.Tensor, G: int) -> torch.Tensor:
        """Run ``G`` rollouts under the replicated policy.

        ``z_c`` is the conditioning vector for a single prompt
        (shape ``(1, emb_dim)``), produced on the master device. We
        ``.to(d)`` it per worker. Returns ``(G, L_total)`` int64 on
        the master device.
        """
        n_dev = len(self.devices)
        chunks = _split_evenly(G, n_dev)

        if n_dev == 1:
            # Fast path — same as the single-device call site.
            cfg_d = self._cfgs[0]
            return self.rollout_fn(self._models[0], cfg_d, z_c, G, self.master_device)

        def _work(idx: int, k: int) -> torch.Tensor:
            if k <= 0:
                # Sentinel: empty contribution. Returning an empty tensor
                # of the right shape keeps the cat path simple.
                L_total = getattr(self._cfgs[idx], "sequence_length",
                                  self._cfgs[idx].diffusion_steps)
                return torch.empty(0, L_total, dtype=torch.int64, device=self.master_device)
            d = self.devices[idx]
            cfg_d = self._cfgs[idx]
            model_d = self._models[idx]
            z_d = z_c.to(d, non_blocking=True)
            ids = self.rollout_fn(model_d, cfg_d, z_d, k, d)
            _device_synchronize(d)
            return ids.to(self.master_device, non_blocking=True)

        futures = [self._pool.submit(_work, i, chunks[i]) for i in range(n_dev)]
        results = [f.result() for f in futures]
        results = [r for r in results if r.shape[0] > 0]
        return torch.cat(results, dim=0)

    def shutdown(self) -> None:
        """Release the thread pool. Idempotent."""
        try:
            self._pool.shutdown(wait=True)
        except Exception:
            pass

    def __del__(self):
        self.shutdown()

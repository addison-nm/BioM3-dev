"""ESMFold pLDDT reward.

Lazy-loads ``facebook/esmfold_v1`` on first call; requires the
optional ``[grpo]`` extras (``fair-esm`` + a transformers version that
ships ``EsmForProteinFolding``).

Supports two dispatch modes:

  - **Single-device** (``device='xpu:0'``): loads one ESMFold replica
    on the given device; scores sequences sequentially. Original
    behavior — used by single-tile / single-process flows.
  - **Multi-device** (``devices=['xpu:0', 'xpu:1', ...]``): loads a
    replica on every device, scores sequences in parallel via a
    ``ThreadPoolExecutor``. Each sequence dispatches to one tile
    (round-robin); the threads run concurrently so K sequences take
    ``ceil(K / n_tiles)`` × per-sequence-time instead of K ×
    per-sequence-time. At K=12 on 12 tiles this is a ~12× speedup on
    the ESMFold portion of the per-step wallclock.

    HBM cost: ~14 GB × n_tiles per node. At 12 tiles × 14 GB = 168 GB
    out of ~768 GB total node HBM. Comfortable.
"""

import copy
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Sequence, Union

import torch

from biom3.backend.device import setup_logger
from biom3.rl.rewards.base import _clean_aa

logger = setup_logger(__name__)


class ESMFoldReward:
    """Mean pLDDT in [0, 100] from ESMFold structure prediction.

    Pass either ``device`` (single-device, legacy) or ``devices``
    (multi-device, parallel scoring). Both arguments accept anything
    convertible to a string device spec.
    """

    def __init__(
        self,
        device: Optional[Union[str, torch.device]] = None,
        devices: Optional[Sequence[Union[str, torch.device]]] = None,
        max_length: int = 500,
        min_length: int = 10,
        model_name: str = "facebook/esmfold_v1",
    ):
        if devices is not None and device is not None:
            raise ValueError("Pass either `device` (single) or `devices` (list), not both")
        if devices is not None:
            self._devices = [str(d) for d in devices]
        elif device is not None:
            self._devices = [str(device)]
        else:
            raise ValueError("Must pass either `device` or `devices`")

        # Backwards compatibility: external code may inspect ``.device``.
        # We point it at the master (first) device.
        self.device = self._devices[0]
        self.max_length = max_length
        self.min_length = min_length
        self.model_name = model_name
        self._models: Optional[List] = None
        self._tok = None
        self._pool: Optional[ThreadPoolExecutor] = None

    @property
    def devices(self) -> List[str]:
        return list(self._devices)

    def _load(self):
        if self._models is not None:
            return
        try:
            from transformers import AutoTokenizer, EsmForProteinFolding
        except ImportError as e:
            raise ImportError(
                "ESMFoldReward requires the [grpo] extras. "
                "Install with: pip install -e '.[grpo]'"
            ) from e
        n = len(self._devices)
        logger.info(
            "Loading ESMFold (%s) on %d device%s...",
            self.model_name, n, "s" if n != 1 else "",
        )
        t0 = time.time()
        self._tok = AutoTokenizer.from_pretrained(self.model_name)

        # Load once on CPU via `from_pretrained` (HF Transformers uses
        # Accelerate's meta-device init machinery internally; calling it
        # from multiple threads simultaneously corrupts that shared
        # state and yields `NotImplementedError: Cannot copy out of meta
        # tensor`). Then deepcopy per target device on CPU, then `.to(d)`.
        # The deepcopy phase is fast (~1s per copy at ESMFold's size)
        # and `.to(d)` is bus-limited, so sequential per-device staging
        # is plenty fast — ~3s/tile vs the corrupt 12-way parallel load.
        base = EsmForProteinFolding.from_pretrained(self.model_name).eval()
        if n == 1:
            self._models = [base.to(self._devices[0])]
        else:
            self._models = []
            for d in self._devices:
                m = copy.deepcopy(base).to(d)
                self._models.append(m)
            del base
            # Long-lived pool for scoring (this is the parallelism that
            # actually matters — load is one-time, scoring is per-step).
            self._pool = ThreadPoolExecutor(max_workers=n)
        logger.info("ESMFold loaded in %.1fs", time.time() - t0)

    def _score_one(self, tile_idx: int, seq: str) -> float:
        """Score one sequence on the tile at ``self._devices[tile_idx]``.

        Called from worker threads under multi-device mode, and directly
        in the sequential loop under single-device mode. ``torch.no_grad``
        context is thread-local so we apply it here explicitly.
        """
        d = self._devices[tile_idx]
        m = self._models[tile_idx]
        clean = _clean_aa(seq, self.max_length)
        if len(clean) < self.min_length:
            return 0.0
        try:
            inp = self._tok(
                clean, return_tensors="pt", add_special_tokens=False
            )
            inp = {k: v.to(d) for k, v in inp.items()}
            with torch.no_grad():
                out = m(**inp)
            plddt = out.plddt[0, : len(clean)].mean().item()
            if plddt <= 1.0:
                plddt *= 100.0
            return float(round(plddt, 2))
        except Exception as e:
            logger.warning(
                "ESMFold failure on length=%d (tile=%s): %s",
                len(clean), d, e,
            )
            return 0.0

    @torch.no_grad()
    def __call__(self, completions: List[str], **kwargs) -> List[float]:
        self._load()
        n = len(self._devices)
        if n == 1 or len(completions) <= 1:
            return [self._score_one(0, seq) for seq in completions]
        # Multi-tile: round-robin sequences across tiles. ThreadPoolExecutor
        # with max_workers=n means n threads run concurrently; additional
        # work queues per tile. With K=12 and n=12, each tile handles 1.
        # With K=24, each tile handles 2 sequentially. Always order-preserving
        # because we collect futures in submission order.
        futures = [
            self._pool.submit(self._score_one, i % n, completions[i])
            for i in range(len(completions))
        ]
        return [f.result() for f in futures]

    def shutdown(self) -> None:
        """Release the multi-device thread pool. Idempotent."""
        try:
            if self._pool is not None:
                self._pool.shutdown(wait=True)
                self._pool = None
        except Exception:
            pass

    def __del__(self):
        try:
            self.shutdown()
        except Exception:
            pass

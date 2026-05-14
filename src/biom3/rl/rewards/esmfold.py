"""ESMFold pLDDT reward.

Lazy-loads ``facebook/esmfold_v1`` on first call; requires the
optional ``[grpo]`` extras (``fair-esm`` + a transformers version that
ships ``EsmForProteinFolding``).
"""

import time
from typing import List, Optional

import torch

from biom3.backend.device import setup_logger
from biom3.rl.rewards.base import _clean_aa

logger = setup_logger(__name__)


class ESMFoldReward:
    """Mean pLDDT in [0, 100] from ESMFold structure prediction."""

    def __init__(
        self,
        device: torch.device,
        max_length: int = 500,
        min_length: int = 10,
        model_name: str = "facebook/esmfold_v1",
    ):
        self.device = device
        self.max_length = max_length
        self.min_length = min_length
        self.model_name = model_name
        self._model = None
        self._tok: Optional[object] = None

    def _load(self):
        if self._model is not None:
            return
        try:
            from transformers import AutoTokenizer, EsmForProteinFolding
        except ImportError as e:
            raise ImportError(
                "ESMFoldReward requires the [grpo] extras. "
                "Install with: pip install -e '.[grpo]'"
            ) from e
        logger.info("Loading ESMFold (%s)...", self.model_name)
        t0 = time.time()
        self._tok = AutoTokenizer.from_pretrained(self.model_name)
        self._model = (
            EsmForProteinFolding.from_pretrained(self.model_name)
            .eval()
            .to(self.device)
        )
        logger.info("ESMFold loaded in %.1fs", time.time() - t0)

    @torch.no_grad()
    def __call__(self, completions: List[str], **kwargs) -> List[float]:
        self._load()
        rewards: List[float] = []
        for seq in completions:
            clean = _clean_aa(seq, self.max_length)
            if len(clean) < self.min_length:
                rewards.append(0.0)
                continue
            try:
                inp = self._tok(
                    clean, return_tensors="pt", add_special_tokens=False
                )
                inp = {k: v.to(self.device) for k, v in inp.items()}
                out = self._model(**inp)
                plddt = out.plddt[0, : len(clean)].mean().item()
                if plddt <= 1.0:
                    plddt *= 100.0
                rewards.append(float(round(plddt, 2)))
            except Exception as e:
                logger.warning("ESMFold failure on length=%d: %s", len(clean), e)
                rewards.append(0.0)
        return rewards

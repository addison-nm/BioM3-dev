"""Reward functions for GRPO fine-tuning.

Each reward is a callable taking ``List[str]`` of decoded amino-acid
sequences and returning ``List[float]`` scalar rewards (one per sequence).
"""

import time
from typing import List, Optional

import torch

from biom3.backend.device import setup_logger

logger = setup_logger(__name__)

VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")


class StubReward:
    """Deterministic reward keyed off sequence length and composition.

    Useful for CPU smoke tests and unit tests where ESMFold is unavailable.
    Returns scores in [0, 100] so it shares the pLDDT-style scale.
    """

    def __init__(self, target_length: int = 100):
        self.target_length = target_length

    def __call__(self, completions: List[str], **kwargs) -> List[float]:
        scores = []
        for seq in completions:
            clean = "".join(c for c in seq if c in VALID_AA)
            if not clean:
                scores.append(0.0)
                continue
            length_score = max(0.0, 100.0 - abs(len(clean) - self.target_length))
            diversity = len(set(clean)) / 20.0 * 100.0
            scores.append(0.5 * length_score + 0.5 * diversity)
        return scores


class ESMFoldReward:
    """Mean pLDDT in [0, 100] from ESMFold structure prediction.

    Loads ``facebook/esmfold_v1`` lazily on first call. Requires the
    optional ``[grpo]`` extras (``fair-esm`` + a transformers version that
    ships ``EsmForProteinFolding``).
    """

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
            clean = "".join(c for c in seq if c in VALID_AA)[: self.max_length]
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


def build_reward(name: str, device: torch.device, **kwargs):
    if name == "esmfold_plddt":
        return ESMFoldReward(device=device, **kwargs)
    if name == "stub":
        return StubReward(**kwargs)
    raise ValueError(f"Unknown reward '{name}'. Known: esmfold_plddt, stub")

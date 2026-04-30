"""Deterministic stub reward for CPU smoke tests."""

from typing import List

from biom3.rl.rewards.base import _clean_aa


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
            clean = _clean_aa(seq)
            if not clean:
                scores.append(0.0)
                continue
            length_score = max(0.0, 100.0 - abs(len(clean) - self.target_length))
            diversity = len(set(clean)) / 20.0 * 100.0
            scores.append(0.5 * length_score + 0.5 * diversity)
        return scores

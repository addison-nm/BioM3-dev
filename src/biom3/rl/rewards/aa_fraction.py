"""Closed-form amino-acid fraction reward (synthetic ground truth)."""

from typing import List

from biom3.rl.rewards.base import VALID_AA, _clean_aa


class AAFractionReward:
    """Score a sequence by how close its target-AA fraction is to a goal.

    .. math::
        r(s) = \\text{scale} \\cdot \\max\\!\\Big(0,\\; 1 - 2 \\,\\big|\\, f_a(s) - f^* \\,\\big|\\Big)

    where :math:`f_a(s)` is the fraction of canonical AAs equal to
    ``target_aa`` in ``s``, and :math:`f^*` is ``target_fraction``.
    Peaks at ``scale`` when :math:`f_a = f^*`, falls linearly to 0 at
    :math:`f_a = f^* \\pm 0.5`, and clamps to 0 outside that band. Empty
    cleaned sequences score 0.

    Used as the closed-form ground truth for the synthetic Phase-3.5
    surrogate-in-the-loop demo (see docs/grpo_finetuning.md). The band
    avoids the trivial collapse mode where the policy would otherwise
    learn to emit a constant string of ``target_aa``.
    """

    def __init__(
        self,
        target_aa: str = "A",
        target_fraction: float = 0.4,
        scale: float = 100.0,
    ):
        if len(target_aa) != 1 or target_aa not in VALID_AA:
            raise ValueError(f"target_aa must be a single canonical AA, got {target_aa!r}")
        if not 0.0 <= target_fraction <= 1.0:
            raise ValueError(f"target_fraction must be in [0, 1], got {target_fraction}")
        self.target_aa = target_aa
        self.target_fraction = float(target_fraction)
        self.scale = float(scale)

    def __call__(self, completions: List[str], **kwargs) -> List[float]:
        out: List[float] = []
        for seq in completions:
            clean = _clean_aa(seq)
            if not clean:
                out.append(0.0)
                continue
            frac = clean.count(self.target_aa) / len(clean)
            r = max(0.0, 1.0 - 2.0 * abs(frac - self.target_fraction))
            out.append(self.scale * r)
        return out

"""Within-group diversity reward.

Penalizes mode collapse by giving each replica a reward proportional to
its average divergence (1 − identity) from peers in its own group. The
metric helper lives in ``biom3.rl.diversity`` so the trainer can log
diversity even when no diversity reward is in use.
"""

from typing import Dict, List

from biom3.rl.diversity import diversity_stats


class DiversityReward:
    """Per-replica within-group diversity reward.

    ``group_size`` is fixed at construction (the trainer's
    ``num_generations``). On each call, the flat ``completions`` list is
    split into ``len(completions) / group_size`` groups, per-replica
    diversity is computed via ``diversity_stats``, and rewards are
    derived from that:

    - ``mode="monotone"``: ``r_i = scale * per_replica_diversity[i]``,
      range ``[0, scale]``. Encourages divergence directly.
    - ``mode="targeted"``: ``r_i = scale * max(0, 1 - 2*|d_i - target|)``,
      mirrors ``AAFractionReward``'s band — useful when full divergence
      is undesirable.

    ``last_components()`` returns ``{"diversity": [...]}`` of length
    ``len(completions)`` so the trainer's existing per-component logger
    surfaces it in ``train_log.json`` and the components figure.
    """

    def __init__(
        self,
        group_size: int,
        scale: float = 100.0,
        mode: str = "monotone",
        target: float = 0.5,
    ):
        if group_size <= 0:
            raise ValueError(f"group_size must be positive, got {group_size}")
        if mode not in ("monotone", "targeted"):
            raise ValueError(
                f"mode must be 'monotone' or 'targeted', got {mode!r}"
            )
        if not 0.0 <= target <= 1.0:
            raise ValueError(f"target must be in [0, 1], got {target}")
        self.group_size = int(group_size)
        self.scale = float(scale)
        self.mode = mode
        self.target = float(target)
        self._last: Dict[str, List[float]] = {}

    def __call__(self, completions: List[str], **kwargs) -> List[float]:
        stats = diversity_stats(completions, group_size=self.group_size)
        per_replica = stats["per_replica_diversity"]
        if self.mode == "monotone":
            rewards = [self.scale * float(d) for d in per_replica]
        else:
            rewards = [
                self.scale * max(0.0, 1.0 - 2.0 * abs(float(d) - self.target))
                for d in per_replica
            ]
        self._last = {"diversity": list(rewards)}
        return rewards

    def last_components(self) -> Dict[str, List[float]]:
        return dict(self._last)

"""Multi-objective reward composition with weighted reductions."""

from typing import Callable, Dict, List, Mapping, Tuple

from biom3.rl.rewards.base import Reward


_REDUCTIONS: Dict[str, Callable[[List[List[float]], List[float]], List[float]]] = {}


def _register_reduction(name: str):
    def deco(fn):
        _REDUCTIONS[name] = fn
        return fn
    return deco


@_register_reduction("weighted_sum")
def _weighted_sum(component_rewards: List[List[float]], weights: List[float]) -> List[float]:
    n = len(component_rewards[0])
    return [
        sum(weights[i] * component_rewards[i][j] for i in range(len(weights)))
        for j in range(n)
    ]


@_register_reduction("product")
def _product(component_rewards: List[List[float]], weights: List[float]) -> List[float]:
    n = len(component_rewards[0])
    out: List[float] = []
    for j in range(n):
        x = 1.0
        for i in range(len(weights)):
            x *= component_rewards[i][j] ** weights[i]
        out.append(x)
    return out


class CompositeReward:
    """Combine multiple named ``Reward`` components with explicit weights.

    ``components`` maps a name (used in logs) to ``(reward, weight)``.
    ``reduction`` controls how the per-component scores are combined:

    - ``"weighted_sum"`` (default): ``sum(w_i * r_i)``. Prefer when
      components are on similar scales (or you've normalized them).
    - ``"product"``: ``prod(r_i ** w_i)``. Useful for "all of these
      must be high" — a near-zero on any component nukes the score.

    Per-component values from the most recent call are available via
    ``last_components()`` for diagnostic logging. The trainer picks
    that up automatically if present.
    """

    def __init__(
        self,
        components: Mapping[str, Tuple[Reward, float]],
        reduction: str = "weighted_sum",
    ):
        if not components:
            raise ValueError("CompositeReward needs at least one component")
        if reduction not in _REDUCTIONS:
            raise ValueError(
                f"reduction must be one of {sorted(_REDUCTIONS)}, got {reduction!r}"
            )
        self.names = list(components.keys())
        self.rewards = [components[n][0] for n in self.names]
        self.weights = [float(components[n][1]) for n in self.names]
        self.reduction = reduction
        self._last: Dict[str, List[float]] = {}

    def __call__(self, completions: List[str], **kwargs) -> List[float]:
        per_component = [r(completions, **kwargs) for r in self.rewards]
        for name, vals in zip(self.names, per_component):
            self._last[name] = list(vals)
        return _REDUCTIONS[self.reduction](per_component, self.weights)

    def last_components(self) -> Dict[str, List[float]]:
        return dict(self._last)

"""Shared building blocks for the rl rewards subpackage.

A reward conforms to the ``Reward`` protocol: a callable that takes a
``List[str]`` of decoded amino-acid sequences and returns a ``List[float]``
of scalar rewards (one per sequence).

A reward MAY also expose ``last_components() -> Dict[str, List[float]]``
returning the most recent call's per-component breakdown. The trainer
uses this for diagnostic logging when present (CompositeReward does;
single-objective rewards generally don't).
"""

from typing import List, Optional, Protocol


VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")


class Reward(Protocol):
    def __call__(self, completions: List[str], **kwargs) -> List[float]: ...


def _clean_aa(seq: str, max_length: Optional[int] = None) -> str:
    out = "".join(c for c in seq if c in VALID_AA)
    if max_length is not None:
        out = out[:max_length]
    return out

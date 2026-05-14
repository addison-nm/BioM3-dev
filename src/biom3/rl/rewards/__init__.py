"""Reward functions for GRPO/GDPO fine-tuning.

A reward conforms to the ``Reward`` protocol: a callable that takes a
``List[str]`` of decoded amino-acid sequences and returns a ``List[float]``
of scalar rewards (one per sequence). A reward MAY also expose
``last_components()`` for per-component diagnostic logging
(``CompositeReward`` and ``DiversityReward`` do).

The package is split by reward family — folding model, closed-form,
lookup, surrogate, composite, diversity — so per-family dependencies
(e.g. ``transformers`` for ESMFold, ``biopython`` for diversity) can stay
behind their own lazy imports. All public symbols re-export here for
``from biom3.rl.rewards import X`` back-compat.
"""

from biom3.rl.rewards.aa_fraction import AAFractionReward
from biom3.rl.rewards.base import Reward
from biom3.rl.rewards.composite import CompositeReward
from biom3.rl.rewards.diversity import DiversityReward
from biom3.rl.rewards.esmfold import ESMFoldReward
from biom3.rl.rewards.registry import build_reward
from biom3.rl.rewards.stub import StubReward
from biom3.rl.rewards.surrogate import SurrogateReward
from biom3.rl.rewards.tsv_lookup import TsvLookupReward

__all__ = [
    "Reward",
    "StubReward",
    "ESMFoldReward",
    "AAFractionReward",
    "SurrogateReward",
    "TsvLookupReward",
    "CompositeReward",
    "DiversityReward",
    "build_reward",
]

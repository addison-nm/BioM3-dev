"""Name-based reward dispatcher."""

import torch

from biom3.rl.rewards.aa_fraction import AAFractionReward
from biom3.rl.rewards.diversity import DiversityReward
from biom3.rl.rewards.esmfold import ESMFoldReward
from biom3.rl.rewards.stub import StubReward
from biom3.rl.rewards.tsv_lookup import TsvLookupReward


def build_reward(name: str, device: torch.device, **kwargs):
    """Build a single named reward.

    For ``CompositeReward`` you typically construct it directly in code
    rather than going through this dispatcher (the components are
    themselves rewards and need their own kwargs). ``DiversityReward``
    requires an explicit ``group_size`` kwarg.
    """
    if name == "esmfold_plddt":
        return ESMFoldReward(device=device, **kwargs)
    if name == "stub":
        return StubReward(**kwargs)
    if name == "tsv_lookup":
        return TsvLookupReward(**kwargs)
    if name == "aa_fraction":
        return AAFractionReward(**kwargs)
    if name == "diversity":
        return DiversityReward(**kwargs)
    raise ValueError(
        f"Unknown reward '{name}'. Known: esmfold_plddt, stub, tsv_lookup, "
        "aa_fraction, diversity. For multi-objective use CompositeReward "
        "directly; for SurrogateReward construct it in code (predictor + featurizer)."
    )

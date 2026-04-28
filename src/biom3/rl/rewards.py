"""Reward functions for GRPO fine-tuning.

A reward conforms to the ``Reward`` protocol: a callable that takes a
``List[str]`` of decoded amino-acid sequences and returns a ``List[float]``
of scalar rewards (one per sequence).

A reward MAY also expose ``last_components() -> Dict[str, List[float]]``
returning the most recent call's per-component breakdown. The trainer
uses this for diagnostic logging when present (CompositeReward does;
single-objective rewards generally don't).
"""

import csv
import time
from typing import Callable, Dict, List, Mapping, Optional, Protocol, Sequence, Tuple

import torch

from biom3.backend.device import setup_logger

logger = setup_logger(__name__)

VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")


class Reward(Protocol):
    def __call__(self, completions: List[str], **kwargs) -> List[float]: ...


def _clean_aa(seq: str, max_length: Optional[int] = None) -> str:
    out = "".join(c for c in seq if c in VALID_AA)
    if max_length is not None:
        out = out[:max_length]
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Computational rewards
# ─────────────────────────────────────────────────────────────────────────────


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


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic / closed-form rewards
# ─────────────────────────────────────────────────────────────────────────────


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


# ─────────────────────────────────────────────────────────────────────────────
# Experimental-data rewards
# ─────────────────────────────────────────────────────────────────────────────


class SurrogateReward:
    """Wrap a fitted regressor + featurizer as a per-sequence reward.

    ``predictor`` is anything with a ``.predict(features) -> np.ndarray``
    method (sklearn estimators conform; a small wrapper around a torch
    MLP would too). ``featurizer`` is a ``Featurizer`` (see
    ``biom3.rl.featurizers``) that turns sequences into a ``(N, D)``
    array.

    On call, returns a Python ``List[float]`` of length ``len(completions)``.

    Use this for the surrogate-in-the-loop workflow (Phase 3.5):

    1. Train a regressor offline on a TSV of (sequence, scalar) lab
       measurements via ``scripts/train_grpo_surrogate.py``.
    2. Reload the joblib + the featurizer config in your GRPO config
       and pass the assembled ``SurrogateReward`` to ``grpo_train``.
    """

    def __init__(self, predictor, featurizer):
        if not hasattr(predictor, "predict"):
            raise TypeError(
                f"predictor must expose .predict(features); got {type(predictor).__name__}"
            )
        self.predictor = predictor
        self.featurizer = featurizer

    def __call__(self, completions: List[str], **kwargs) -> List[float]:
        feats = self.featurizer(list(completions))
        preds = self.predictor.predict(feats)
        return [float(x) for x in preds]


class TsvLookupReward:
    """Look up a per-sequence scalar reward from a TSV/CSV file.

    The file must have a header. ``key_column`` names the column holding
    sequences (cleaned to canonical AA before lookup); ``value_column``
    names the column holding the scalar.

    On a miss (sequence not in the table), behavior is controlled by
    ``miss_strategy``:

    - ``"penalty"`` — return ``miss_value`` (default 0.0). Simple, but
      zero advantage when *every* group member misses.
    - ``"skip"`` — return ``float('nan')``. The trainer treats NaN
      rewards as missing; if the entire group is NaN, the step is
      skipped. (Not yet wired through grpo_train; for now it behaves
      like ``penalty`` with miss_value=NaN.)

    Closed-world tasks (ranking known variants) are the natural fit for
    pure lookup. For novel-sequence generation where most rollouts will
    miss, train a regressor on the TSV and use ``SurrogateReward``
    instead (not in this revision — see docs/grpo_finetuning.md).
    """

    def __init__(
        self,
        path: str,
        key_column: str = "sequence",
        value_column: str = "value",
        delimiter: str = "\t",
        miss_value: float = 0.0,
        miss_strategy: str = "penalty",
        clean: bool = True,
    ):
        if miss_strategy not in ("penalty", "skip"):
            raise ValueError(f"miss_strategy must be 'penalty' or 'skip', got {miss_strategy!r}")
        self.path = path
        self.key_column = key_column
        self.value_column = value_column
        self.delimiter = delimiter
        self.miss_value = float("nan") if miss_strategy == "skip" else float(miss_value)
        self.miss_strategy = miss_strategy
        self.clean = clean
        self._table: Dict[str, float] = {}
        self._loaded = False

    def _load(self):
        if self._loaded:
            return
        with open(self.path, newline="") as f:
            reader = csv.DictReader(f, delimiter=self.delimiter)
            if self.key_column not in (reader.fieldnames or []):
                raise KeyError(f"key_column {self.key_column!r} not in {reader.fieldnames}")
            if self.value_column not in (reader.fieldnames or []):
                raise KeyError(f"value_column {self.value_column!r} not in {reader.fieldnames}")
            for row in reader:
                seq = row[self.key_column].strip()
                if self.clean:
                    seq = _clean_aa(seq)
                if not seq:
                    continue
                try:
                    self._table[seq] = float(row[self.value_column])
                except ValueError:
                    continue
        logger.info("TsvLookupReward: loaded %d entries from %s", len(self._table), self.path)
        self._loaded = True

    def __call__(self, completions: List[str], **kwargs) -> List[float]:
        self._load()
        out: List[float] = []
        for seq in completions:
            key = _clean_aa(seq) if self.clean else seq
            out.append(self._table.get(key, self.miss_value))
        return out


# ─────────────────────────────────────────────────────────────────────────────
# Composition
# ─────────────────────────────────────────────────────────────────────────────


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


# ─────────────────────────────────────────────────────────────────────────────
# Dispatcher
# ─────────────────────────────────────────────────────────────────────────────


def build_reward(name: str, device: torch.device, **kwargs):
    """Build a single named reward.

    For ``CompositeReward`` you typically construct it directly in code
    rather than going through this dispatcher (the components are
    themselves rewards and need their own kwargs).
    """
    if name == "esmfold_plddt":
        return ESMFoldReward(device=device, **kwargs)
    if name == "stub":
        return StubReward(**kwargs)
    if name == "tsv_lookup":
        return TsvLookupReward(**kwargs)
    if name == "aa_fraction":
        return AAFractionReward(**kwargs)
    raise ValueError(
        f"Unknown reward '{name}'. Known: esmfold_plddt, stub, tsv_lookup, "
        "aa_fraction. For multi-objective use CompositeReward directly; "
        "for SurrogateReward construct it in code (predictor + featurizer)."
    )

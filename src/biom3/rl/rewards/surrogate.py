"""Wrap a fitted regressor + featurizer as a per-sequence reward."""

from typing import List


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

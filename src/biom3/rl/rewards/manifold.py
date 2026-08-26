"""Distance-from-a-latent-manifold as a per-sequence reward.

Scores generated sequences by their Mahalanobis distance ``d_E`` from a
reference cloud fitted with :mod:`biom3.geometry` — lower means "more like
the reference family". Structurally a sibling of ``SurrogateReward``:
featurize, then score with a fitted offline object.

The reward interface is higher-is-better and ``d_E`` is lower-is-better and
unbounded above, so a transform is mandatory. See ``TRANSFORMS``.

Correctness note, because ``d_E`` fails quietly rather than loudly: the fit
vectors and the scored queries must come from the same embedding *and* the
same preprocessing. A mismatch returns plausible numbers. Scoring the 911
reference vectors of ``manifold_sho1_911_run1.npz`` with the manifold they
were fitted on reproduces its band to 1e-4; L2-normalising those same
vectors first moves the median from 8.79 to 422.68. Hence:

- ``check_norm`` stays on by default, so ``NormMismatchWarning`` fires on
  the normalise/don't-normalise mismatch it was written for.
- ``validate_against_band()`` is the acceptance test worth running once per
  (manifold, embedder) pair before any training: push the reference
  sequences through the *reward's own* featurizer and confirm the stored
  band comes back.
"""

from typing import Callable, Dict, List, Optional

import numpy as np

from biom3.backend.device import setup_logger
from biom3.geometry import load_manifold, manifold_score
from biom3.geometry.gaussian import band_position

logger = setup_logger(__name__)


# Each transform maps d_E (lower better) to a reward (higher better),
# given the manifold's own reference band.
#
# A caveat specific to GDPO/GRPO: the trainer's advantage is
# ``A_g = R_g - mean(R)`` within a group, so any *affine* transform of d_E
# collapses to the same advantage up to a constant factor — "band" and
# "neg" differ only by 1/p95, which is an effective learning-rate rescale,
# not a different objective. Only the nonlinear transforms ("clipped",
# "exp") genuinely reshape within-group spread. Worth knowing before
# sweeping all four as if they were four ideas.
TRANSFORMS: Dict[str, Callable[[np.ndarray, dict, float], np.ndarray]] = {
    # Simplest. Unbounded negative tail: one catastrophic sample can
    # dominate its group's advantage.
    "neg": lambda d, band, scale: -d,
    # Clipped at `scale` multiples of the band p95 — bounds that tail.
    "clipped": lambda d, band, scale: -np.minimum(d, scale * band["p95"]),
    # Bounded in (0, 1], smooth. `scale` multiplies p95 to set the decay.
    "exp": lambda d, band, scale: np.exp(-d / (scale * band["p95"])),
    # Band-relative: zero exactly at the reference p95, so "inside the
    # natural band" is the natural origin. Travels with the manifold, so
    # it is comparable across reference sets.
    "band": lambda d, band, scale: -(d - band["p95"]) / band["p95"],
}

# Sequences that fail to embed (empty after cleaning) get this d_E, so a
# degenerate rollout cannot score well by accident. Multiplied by p95.
FAILED_EMBED_P95_MULTIPLE = 10.0


class ManifoldReward:
    """Reward a sequence by its distance from a fitted latent manifold.

    Args:
        manifold: a fitted ``ManifoldModel``, or a path to one saved by
            ``biom3.geometry.io.save_manifold``.
        featurizer: sequences -> ``(N, D)`` embedding matrix in the same
            space and preprocessing the manifold was fitted in. For
            BioM3 z_p manifolds this is ``PenCLZpFeaturizer``. May be
            ``None`` at construction: the trainers build the reward via
            ``build_reward`` before Stage 1 exists, then call
            :func:`bind_pencl_rewards` once it is loaded, which fills
            this in with a featurizer over the *already-resident* PenCL
            rather than loading a second copy per rank.
        transform: key into ``TRANSFORMS``.
        scale: transform-specific multiplier on the band p95 (unused by
            ``"neg"`` and ``"band"``).
        check_norm: pass through to the manifold's norm check. Leave on.
        strict_norm: promote the norm mismatch from a warning to an
            error on the first call. Use for unattended runs.
    """

    def __init__(
        self,
        manifold,
        featurizer=None,
        transform: str = "band",
        scale: float = 1.0,
        check_norm: bool = True,
        strict_norm: bool = False,
        featurizer_kwargs: Optional[dict] = None,
    ):
        if transform not in TRANSFORMS:
            raise ValueError(
                f"transform must be one of {sorted(TRANSFORMS)}, got {transform!r}"
            )
        if isinstance(manifold, (str, bytes)):
            manifold = load_manifold(manifold)
        if featurizer is not None and not callable(featurizer):
            raise TypeError(
                f"featurizer must be callable; got {type(featurizer).__name__}"
            )
        self.manifold = manifold
        self.featurizer = None
        self.transform = transform
        self.scale = float(scale)
        self.check_norm = check_norm
        self.strict_norm = strict_norm
        self.featurizer_kwargs = dict(featurizer_kwargs or {})
        self._last: Dict[str, List[float]] = {}
        self._checked_norm = False
        if featurizer is not None:
            self._set_featurizer(featurizer)
        logger.info(
            "ManifoldReward: %r | transform=%s scale=%.3g%s",
            manifold, transform, scale,
            "" if featurizer is not None else " | featurizer: awaiting bind_pencl",
        )

    def _set_featurizer(self, featurizer) -> None:
        dim = getattr(featurizer, "dim", None)
        if dim is not None and int(dim) != int(self.manifold.n_dim):
            raise ValueError(
                f"featurizer emits dim {dim} but the manifold was fitted at "
                f"{self.manifold.n_dim}; these are not the same space"
            )
        self.featurizer = featurizer

    def bind_pencl(self, pencl) -> None:
        """Attach a ``PenCLZpFeaturizer`` over an already-loaded PenCL.

        No-op if a featurizer was supplied explicitly, so an experiment
        that wants a different embedding is not silently overridden.
        """
        if self.featurizer is not None:
            return
        from biom3.rl.featurizers import PenCLZpFeaturizer
        self._set_featurizer(PenCLZpFeaturizer(pencl, **self.featurizer_kwargs))
        logger.info(
            "ManifoldReward bound to PenCL z_p featurizer (dim=%d, device=%s)",
            self.featurizer.dim, self.featurizer.device,
        )

    def scores(self, completions: List[str]) -> np.ndarray:
        """Raw ``d_E`` per sequence — no transform, no polarity flip."""
        if self.featurizer is None:
            raise RuntimeError(
                "ManifoldReward has no featurizer. Either pass one at "
                "construction, or call bind_pencl_rewards(reward, s1) once "
                "Stage 1 is loaded (the trainers do this automatically)."
            )
        z = np.asarray(self.featurizer(list(completions)), dtype=np.float64)
        embedded = np.linalg.norm(z, axis=1) > 0
        d = np.full(len(completions), np.nan, dtype=np.float64)
        if embedded.any():
            d[embedded] = manifold_score(
                z[embedded], self.manifold, check_norm=self.check_norm
            )
            self._verify_norm(z[embedded])
        n_failed = int((~embedded).sum())
        if n_failed:
            logger.warning(
                "%d/%d sequence(s) did not embed (empty after cleaning); "
                "assigning d_E = %.1f x band p95",
                n_failed, len(completions), FAILED_EMBED_P95_MULTIPLE,
            )
            d[~embedded] = FAILED_EMBED_P95_MULTIPLE * self.manifold.band["p95"]
        return d

    def _verify_norm(self, z: np.ndarray) -> None:
        """Fail loudly on the first batch when strict_norm is set.

        The manifold's own check already warns; this turns that into a
        hard stop for unattended runs, where a warning scrolls past and
        the mismatch is only noticed as a strange loss curve.
        """
        if self._checked_norm or not self.strict_norm:
            return
        self._checked_norm = True
        ref = self.manifold.ref_mean_norm
        got = float(np.linalg.norm(z, axis=1).mean())
        if ref > 0 and not (1 / 1.5 <= got / ref <= 1.5):
            raise ValueError(
                f"query mean L2 norm {got:.4g} is {got / ref:.3g}x the manifold's "
                f"reference mean norm {ref:.4g}. The embedder's preprocessing "
                f"almost certainly does not match the fit's (strict_norm=True)"
            )

    def __call__(self, completions: List[str], **kwargs) -> List[float]:
        d = self.scores(completions)
        r = TRANSFORMS[self.transform](d, self.manifold.band, self.scale)
        pos = band_position(d, self.manifold)
        self._last = {
            "d_E": [float(x) for x in d],
            # -1 below / 0 within / +1 above the reference band. Mean this
            # over a step to get the "fraction inside the natural band"
            # diagnostic that mode collapse shows up in.
            "band_position": [
                float({"below": -1.0, "within": 0.0, "above": 1.0}[p]) for p in pos
            ],
        }
        return [float(x) for x in r]

    def last_components(self) -> Dict[str, List[float]]:
        return dict(self._last)

    def validate_against_band(
        self,
        reference_sequences: List[str],
        tol: float = 0.05,
        stored_vectors: Optional[np.ndarray] = None,
    ) -> dict:
        """Acceptance test: do the reference sequences reproduce the band?

        Embeds ``reference_sequences`` through this reward's own
        featurizer and compares the resulting p5/median/p95 against the
        manifold's stored band. If the naturals do not reproduce their
        own band, the embedder does not match the fit and nothing
        downstream means anything.

        ``tol`` is a relative tolerance on each band statistic. Pass
        ``stored_vectors`` (the matrix the manifold was fitted on) to
        additionally report per-vector agreement, which localises a
        failure to the embedder rather than to the scoring.

        Returns a dict; raises nothing, so a caller can log and decide.
        """
        # Embed once and reuse: on the full 911-member reference set a second
        # pass is another full ESM-2 forward for no new information.
        z = np.asarray(self.featurizer(list(reference_sequences)), dtype=np.float64)
        d = manifold_score(z, self.manifold, check_norm=self.check_norm)
        got = {
            k: float(v) for k, v in zip(
                ("p5", "median", "p95"), np.percentile(d, [5, 50, 95])
            )
        }
        rel = {
            k: abs(got[k] - self.manifold.band[k]) / max(abs(self.manifold.band[k]), 1e-12)
            for k in got
        }
        # A band is a population statistic. Handing this method a subset of the
        # reference set produces a real deviation that means nothing about the
        # embedder — so say so rather than reporting a bare FAIL. (24 of the 911
        # SHO1 naturals shifts the median 8.79 -> 10.00 with a per-vector error
        # of 2e-06, i.e. a perfect embedder and a "failing" band.)
        n_given, n_fit = len(reference_sequences), int(self.manifold.n_reference)
        is_subset = n_given < n_fit
        result = {
            "stored_band": dict(self.manifold.band),
            "recomputed_band": got,
            "relative_deviation": rel,
            "n_given": n_given,
            "n_reference": n_fit,
            "band_comparable": not is_subset,
            "passed": all(v <= tol for v in rel.values()),
        }
        if is_subset:
            logger.warning(
                "validate_against_band got %d of the manifold's %d reference "
                "vectors; band statistics are not comparable on a subset. Trust "
                "max_relative_vector_error instead, or pass the full set.",
                n_given, n_fit,
            )
        if stored_vectors is not None:
            sv = np.asarray(stored_vectors, dtype=np.float64)
            if z.shape == sv.shape:
                num = np.linalg.norm(z - sv, axis=1)
                den = np.linalg.norm(sv, axis=1)
                result["max_relative_vector_error"] = float((num / den).max())
            else:
                result["max_relative_vector_error"] = None
                result["vector_shape_mismatch"] = (z.shape, sv.shape)
        logger.info(
            "validate_against_band: %s (stored median=%.4f p95=%.4f | "
            "recomputed median=%.4f p95=%.4f)",
            "PASS" if result["passed"] else "FAIL",
            self.manifold.band["median"], self.manifold.band["p95"],
            got["median"], got["p95"],
        )
        return result


def bind_pencl_rewards(reward_fn, pencl) -> int:
    """Bind ``pencl`` into every ``ManifoldReward`` reachable from ``reward_fn``.

    Recurses into ``CompositeReward`` components, so a manifold term
    combined with a diversity term is bound too. Returns how many rewards
    were bound; anything without a ``bind_pencl`` method is skipped, so
    this is safe to call unconditionally on any reward.
    """
    bound = 0
    seen = set()

    def walk(r):
        nonlocal bound
        if id(r) in seen:
            return
        seen.add(id(r))
        if hasattr(r, "bind_pencl"):
            r.bind_pencl(pencl)
            bound += 1
        for child in getattr(r, "rewards", []) or []:
            walk(child)

    walk(reward_fn)
    return bound

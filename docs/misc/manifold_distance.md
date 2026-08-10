# Manifold distance (`biom3.geometry`)

A score for "does this embedding look like it came from the reference family?"

Fit a description of a reference cloud of embeddings, then score how far any query vector sits from
it. The default method — and the only one implemented today — is a flat Gaussian: the cloud's
centroid and its Ledoit-Wolf shrinkage precision matrix, with the query's Mahalanobis distance as
the score:

```
d(x) = sqrt((x - mu) @ P @ (x - mu))
```

That distance is in units of reference-cloud standard deviations. Low means the query sits inside
the reference distribution. Applied to PenCL `z_p` with a family's natural homologs as the
reference, it is the metric used to rank designed sequences by predicted function.

This is the flat metric — a single global covariance stand-in, not a position-dependent
(curved/geodesic) one.

## API

```python
import numpy as np
from biom3.geometry import fit_manifold, manifold_score, save_manifold, load_manifold

reference = np.load("zp_naturals.npy")        # (M, D), M >= 8
manifold = fit_manifold(reference, label="PenCL z_p run1_trackC_step187000 / Sho1 naturals")
save_manifold("manifold_sho1.npz", manifold)

queries = np.load("zp_designs.npy")           # (N, D), same D
scores = manifold_score(queries, load_manifold("manifold_sho1.npz"))
```

Everything is arrays in, arrays out: no sequences, no model, no GPU, no network. Producing the
embeddings is somebody else's job.

The split between `fit_manifold` and `manifold_score` is the point of the design. Under the default
method, fitting is a `D × D` shrinkage inverse and scoring is a cheap quadratic form — so fit once,
persist, and score many query sets later without re-touching the reference. Other methods invert
that cost balance; the interface does not change.

## Structure

| Module | Holds |
|---|---|
| [base.py](../../src/biom3/geometry/base.py) | `ManifoldModel` — the fit/score contract and the method registry |
| [gaussian.py](../../src/biom3/geometry/gaussian.py) | `GaussianManifold` — the Ledoit-Wolf method and everything specific to it |
| [manifold.py](../../src/biom3/geometry/manifold.py) | `fit_manifold` / `manifold_score` — the functional facade |
| [io.py](../../src/biom3/geometry/io.py) | `.npz` persistence and embedding-matrix loading |

The base promises exactly two operations:

```
fit(R: (M, D))          -> model
model.score(X: (N, D))  -> (N,) float, lower = more on-manifold
```

Note that this is a *score*, not a metric. For `GaussianManifold` the score happens to be a true
Mahalanobis distance — a norm, in shared units, poolable across query sets — but a neighbour
distance or a reconstruction error is none of those, and a log-density is not even
sign-constrained. Scores are comparable across query sets against **the same** model; they are not
comparable across methods.

Everything beyond fit and score belongs to the method that needs it, because the right answer
differs per method. `GaussianManifold` carries three such things:

- **`band`** — the p5 / median / p95 of the reference set's own distances, the yardstick a query
  distance is read against. `band_position(scores, manifold)` labels each query `below` / `within` /
  `above`. A global fit can measure this in-sample; a memorizing method cannot (under kNN every
  reference point is its own nearest neighbour, so self-scores collapse to zero without
  leave-one-out) — which is exactly why the base does not impose the recipe.
- **`ref_mean_norm`** — the reference vectors' mean L2 norm, checked against queries at score time.
- **`MIN_REFERENCE_SIZE = 8`** — below which a covariance is not worth fitting.

### Adding a method

Subclass `ManifoldModel`, declare `METHOD`, add the fitted state as dataclass fields, and implement
four hooks:

| Hook | Does |
|---|---|
| `_fit_state(reference, **params)` | fit; return the field values as a dict |
| `_score(queries)` | score a validated `(N, D)` float64 matrix |
| `_arrays()` | the state to persist, as arrays/scalars |
| `_from_arrays(arrays)` | rebuild the field values on load |

Declaring `METHOD` registers the class, which is what lets `load_manifold` rebuild a persisted model
as the right type — the method name is stored in the file, so scoring never has to be told which
method produced it. Add validation (minimum sizes, provenance guards) and extra state in the
subclass, not the base. `tests/geometry_tests/test_base.py` shows a minimal method that carries no
band and no norm guard at all.

## CLI

See [CLI_reference.md](../CLI_reference.md#analysis-entrypoints) for `biom3_fit_manifold` and
`biom3_score_manifold`.

## Things that will bite you

- **`M < D` is normal, not a bug.** A 177-homolog reference in 512 dimensions has a singular
  covariance. Ledoit-Wolf shrinkage is exactly what makes the inverse well-defined. Reducing the
  dimension first to "fix" the singularity changes the metric.
- **Scores shift with the scale of the queries.** Scoring L2-normalized vectors (norm ≈ 1) against a
  manifold fit on full-head PenCL `z_p` (norm ≈ 4) produces distances that reflect the scale gap
  rather than the content, and nothing raises. `GaussianManifold.score` compares the queries' mean
  L2 norm against the stored reference mean norm and emits a `NormMismatchWarning` when the ratio
  leaves `[1/norm_ratio_tol, norm_ratio_tol]` (default `1.5`). The warning reports both norms, the
  ratio, and the threshold that tripped it — it says the norms differ, not why, so read it as a
  prompt to check how each set was produced. Raise `norm_ratio_tol` if a genuine scale difference is
  expected.
- **Do not z-score per design group.** The distance is already normalized — it is measured in
  standard deviations of the reference cloud, a yardstick shared across every set scored against the
  same manifold. Re-standardizing per group divides out the variance that carries the signal.
- **`M >= 8` is enforced** with a `ValueError` rather than a `LinAlgError` from inside numpy.
- **Nothing hard-codes `D`.** The dimension comes from the input.

## Provenance and validation

Ported from two implementations of the same metric: the production `mdist`/`build_manifold` in
`biom3-mcp` (`daemon/src/decoder_epoch_eval.py`, branch `fix/mcp-phase-a-endpoints`) and the
library version in `PenCL_geometry` (`pencl_geometry/score.py`). Numerically checked against both
on synthetic 512-d data across three regimes (`M = 177 < D`, the `M = 8` floor, and `M > D`):
bit-identical to the production route, and agreeing with the `score.py` route to ~1e-13 absolute
(~1e-15 relative), which differs only in taking `pinv` of the shrunk covariance instead of
sklearn's `precision_`. `tests/geometry_tests/test_gaussian.py` pins both routes.

Two deliberate departures from production:

- The quadratic form is clamped at zero before the square root (`sqrt(max(q, 0))`), so round-off
  cannot produce a NaN for a query sitting on the centroid.
- There is exactly one code path per method. `score.py` prefers `zp_dms.metric.family_metric` when
  that package is installed and silently falls back to Ledoit-Wolf otherwise; we never substitute.

Out of scope here, by design: embedding, the tangent/normal (`tanperp`) split, and the curved
geodesic successor.

# The d_E manifold reward

Rewards a generated sequence by its **Mahalanobis distance from a reference
cloud of natural sequences**, in PenCL's joint latent space:

```
generated s -> z_p(s) -> d_E(s, M) -> reward -> GDPO update
```

Lower `d_E` means "more like the reference family". A design that drifts off
the family manifold is penalised; one that lands inside the natural cloud is
not. It costs one ESM-2 forward plus a quadratic form — no folding model, no
assay.

## Why it is worth rewarding

Fitting on 177 Sho1 orthologs (`NOG09120`) and scoring the 317-design assayed
panel, `d_E` from `z_p` alone ranks assayed function at **AUC 0.8966**
(functional n=22, mean 21.44 ± 3.97; non-functional n=295, mean 38.93 ± 25.34).
Reproduced on Aurora 2026-08-25 from
`Sho1_manifold_reproduction/biom3_geometry/`.

## Usage

```bash
biom3_gdpo_train \
    --config_path configs/grpo/gdpo_edist_sh3.json \
    --run_id edist_001
```

or directly:

```bash
--reward manifold \
--manifold_path /path/to/manifold_sho1_911_run1.npz \
--manifold_transform band
```

The embedder is **not** configurable from the CLI, deliberately. It is always
PenCL `z_p` over the Stage 1 weights the run already loads, attached after
Stage 1 comes up via `bind_pencl_rewards`. That reuses the resident PenCL
instead of loading a second copy per rank, and removes the most likely way to
get silently wrong numbers — see below.

## The failure mode to respect

`d_E` is a Mahalanobis distance, so it amplifies small differences in
low-variance directions. **A mismatch between the fit's embedding and the
query's does not announce itself — it returns plausible-looking numbers.**

Measured on the 911 reference vectors of `manifold_sho1_911_run1.npz`, changing
only the preprocessing:

| what changed | d_E median | d_E p95 |
|---|---|---|
| correct (run1 `step_187000`, raw, fp32) | 8.7943 | 17.4581 |
| L2-normalise the queries first | 422.68 | 423.92 |
| bf16 autocast instead of fp32 | 12.1863 | 19.5852 |

Two guardrails, and one gap between them:

- **Normalisation mismatches** trip `NormMismatchWarning`. Do not suppress it.
  `--manifold_strict_norm` promotes it to a hard error for unattended runs.
- **Precision mismatches do not trip anything.** Mean ‖z_p‖ is 4.5055 under
  both fp32 and bf16, so the norm check is blind to it, and the cosine to the
  fp32 vector is still 0.99984 — which reads as fine and is not. Per-row `d_E`
  moves by up to **8.68**, against a band spanning 4.14–17.46.

`PenCLZpFeaturizer` therefore sets `torch.autocast(..., enabled=False)`
explicitly rather than simply not enabling autocast, so an ambient autocast in
a caller cannot change the numbers. Do not remove that.

Padding mode, by contrast, is free: dynamic-to-batch-max and fixed-1024 agree
to cosine 1.000000, because ESM-2 masks pads internally. The featurizer uses
dynamic padding, which for SH3-length inputs is a ~70-token forward rather than
a 1024-token one.

### Acceptance test

Before trusting a new (manifold, Stage 1) pair, push the reference sequences
through the reward's own featurizer and confirm the stored band comes back:

```python
result = reward.validate_against_band(reference_seqs, stored_vectors=ref_matrix)
assert result["passed"], result
```

If the naturals do not reproduce their own band, the embedder does not match
the fit and nothing downstream means anything.
`_misc/edist_scripts/check_zp_embedder.py` does this end-to-end for the
911/run1 pair.

## Choosing a manifold

The reference set is a scientific choice, not a config detail. A whole-family
manifold (n=59,893) asks "is this an SH3 domain at all"; the Sho1-ortholog
manifold (n≈177–911) asks "is this a Sho1 ortholog". Same design, very
different scores.

`manifold_sho1_911_run1.npz` is the default recommendation: specific enough to
steer, large enough to estimate. At n=177 in 512 dimensions the empirical
covariance has rank 176 and ~1.4% of a design's variance is silently discarded
by the pseudo-inverse; that worsens as the ensemble shrinks.

Whichever you pick, the manifold must be fitted in the **same Stage 1's** z_p as
the run uses. `manifold_sho1_911_run1.npz` requires
`Run1_TrackC_step_187000_3b7e39f9.ckpt`; `manifold_sho1_177_run0.npz` requires
Run0 and L2-normalised rows.

## Choosing a transform

`d_E` is lower-better and unbounded above; the reward interface is
higher-better. Four transforms, set with `--manifold_transform`:

| name | reward | notes |
|---|---|---|
| `neg` | `-d` | simplest; unbounded negative tail lets one catastrophic sample dominate its group |
| `clipped` | `-min(d, scale·p95)` | bounds that tail |
| `exp` | `exp(-d / (scale·p95))` | bounded in (0, 1], smooth |
| `band` | `-(d - p95) / p95` | zero at the reference p95, so "inside the band" is the origin; travels with the manifold |

**`band` and `neg` are not really different choices.** GDPO's advantage is
`A_g = R_g - mean(R)` within a group, so any affine transform of `d_E` gives the
same advantage up to a constant factor — here `1/p95`, i.e. an effective
learning-rate rescale. Only `clipped` and `exp` genuinely reshape within-group
spread. Sweep those two against a linear baseline rather than all four.

`band` is the default because it is self-calibrating across reference sets and
its sign is interpretable: positive reward means inside the natural band.

## Guarding against collapse

`d_E` is minimised by generating something maximally typical — plausibly a
near-copy of a reference sequence. This is not hypothetical: the April 2026
pLDDT production run had all 48 replicas converged to length 62 by step 98.

`--diversity_weight` wraps the manifold reward in a `CompositeReward` with
`DiversityReward`; `bind_pencl_rewards` recurses into composite components, so
that combination works without extra wiring. Budget for it from the start.

`last_components()` exposes `d_E` and `band_position` per sequence (-1 below /
0 within / +1 above the reference band); the trainer logs those automatically.
Mean `band_position` over a step is the "fraction inside the natural band"
diagnostic to watch alongside the diversity stats the trainer already prints.

# Weights Bundle (GHCR)

The BioM3 Run 1 base weights are published to GHCR as an OCI artifact, alongside the
software images. One `oras pull` gets a consistent, checksummed set of weights for all
three stages along with the architecture configs those weights require in order to load
successfully.

| | |
|---|---|
| Package | `ghcr.io/natural-machine/biom3-weights` |
| Tag | `run1_base` (whatever the publisher pushed it as) |
| Size | ~6.44 GB across 13 files |
| Prereq | [`oras`](https://oras.land/docs/installation) — a single static binary |

## Contents

The published artifact unpacks to a self-contained directory:

```txt
biom3-weights-run1_base/
├── MANIFEST.json        sha256 and size for every file
├── configs/             one architecture config fragment per stage
│   ├── _base_PenCL.json
│   ├── _base_Facilitator.json
│   └── _base_ProteoScribe.json
└── weights/             weight files, one subdirectory per component
    ├── LLMs/
    ├── PenCL/
    ├── Facilitator/
    └── ProteoScribe/
```

- **`configs/`** holds a small JSON fragment per stage carrying only the settings needed
  to construct and load that stage. Shipping them with the weights keeps configuration and
  weights from drifting apart — a fragment and its weights are always a matched set.
- **`weights/`** holds the weight files, grouped into one subdirectory per pipeline
  component. The layout mirrors the repo's own `weights/`, so it links straight into a
  checkout (see [Pulling](#pulling)).
- **`MANIFEST.json`** records a sha256 and byte size for every file, and `oras pull`
  verifies each blob's digest against the manifest as it downloads.

## Pulling

### Standalone — nothing but `oras`

The weights need neither the `BioM3-dev` repo, the software image, nor a login (once the
package is public). Pull them, with their configs, straight to a directory:

```bash
oras pull ghcr.io/natural-machine/biom3-weights:run1_base -o biom3-weights-run1_base
```

`oras pull` checks each blob's digest against the manifest as it downloads, so a clean
pull is already integrity-verified. You get a self-contained
`biom3-weights-run1_base/{configs,weights,MANIFEST.json}` to point any `biom3` install at.

The software image pulls the same standalone way (see
[../../cloud/README.md](../../cloud/README.md) for running it):

```bash
docker pull ghcr.io/natural-machine/biom3:cuda-dev
```

Each artifact stands on its own — pull either without the other, and neither needs a
checkout.

### Into a checkout

By default `fetch_bundle.sh` pulls and verifies — it does **not** touch your checkout:

```bash
cd /path/to/BioM3-dev
./scripts/weights_bundle/fetch_bundle.sh ~/biom3-bundles --tag run1_base
```

1. `oras pull` into `~/biom3-bundles/run1_base/`
2. verifies every file's sha256 against the bundle's `MANIFEST.json`

Wiring it into the checkout is opt-in. Either pass `--link` to have fetch do it after
verifying, or run the two commands yourself from the repo root (fetch prints them on
success):

```bash
./scripts/weights_bundle/fetch_bundle.sh ~/biom3-bundles --tag run1_base --link
# equivalently, by hand:
./scripts/link_weights.sh ~/biom3-bundles/run1_base/weights weights
ln -s ~/biom3-bundles/run1_base/configs configs/bundles/run1_base
```

[`link_weights.sh`](../../scripts/link_weights.sh) never overwrites: it symlinks only the
files that are absent and reports `MATCH`/`MISMATCH` for anything already present, so review
its output before relying on the result. The bundle's `weights/` subtree mirrors the repo's
own layout, so once linked, a path like `weights/PenCL/BioM3_PenCL_run1_base.bin` resolves
through the symlink. Keep the pulled bundle dir around — the symlinks point into it.

`--tag` is required — it names the bundle to pull; list published tags with `oras repo tags
ghcr.io/natural-machine/biom3-weights`. `--link` wires the bundle into the checkout;
`--quick-verify` checks sizes without hashing.

### Into an image

The published image does not bake in the weights. Pull the bundle on the host and bind its
`weights/` into the container at run time:

```bash
oras pull ghcr.io/natural-machine/biom3-weights:run1_base -o ~/biom3-bundles/run1_base
# then at container run time, bind:
#   ~/biom3-bundles/run1_base/weights -> /app/weights
```

You bind only the weights: run1_base's architecture is already the default baked into the
image's `configs/inference/` configs, so you point `--config_path` at those and don't need
to bind the bundle's configs at run time. See [APPTAINER.md](../APPTAINER.md) for a worked
container invocation.

### Verifying an existing bundle

`verify_bundle.py` is stdlib-only and needs neither torch nor a `biom3` install, so it
works on a bare machine:

```bash
python scripts/weights_bundle/verify_bundle.py ~/biom3-bundles/run1_base
```

## Running against the bundle's weights

Each bundle ships a per-stage architecture fragment (`configs/_base_*.json`) alongside its
weights, so the settings the weights were built with are published *with* them — you can
always see exactly what a bundle expects.

For **run1_base**, that architecture is already the default the repo's `configs/inference/`
configs encode, so there is nothing to compose: point `--config_path` straight at the stock
inference config and pass the bundle's weight file. (Weight paths are relative to the run
dir, so run from the repo root — or `/app` in the container — with the bundle's `weights/`
linked or bound there.)

```bash
biom3_PenCL_inference \
    --config_path configs/inference/stage1_PenCL.json \
    --model_path weights/PenCL/BioM3_PenCL_run1_base.bin \
    --input_data_path None --output_path outputs/pencl_embeddings.pt

biom3_Facilitator_sample \
    --config_path configs/inference/stage2_Facilitator.json \
    --model_path weights/Facilitator/BioM3_Facilitator_run1_base.bin \
    --input_data_path outputs/pencl_embeddings.pt \
    --output_data_path outputs/facilitator_embeddings.pt

biom3_ProteoScribe_sample \
    --config_path configs/inference/stage3_ProteoScribe_sample.json \
    --model_path weights/ProteoScribe/BioM3_ProteoScribe_run1_base.bin \
    --input_path outputs/facilitator_embeddings.pt \
    --output_path outputs/generated_sequences.csv
```

A future weight set whose architecture differs from these defaults would ship a fragment that
no longer matches `configs/inference/`; that's when you compose the shipped fragment into a
config via `_overwrite_configs` (or the planned bundle-config integration tool). run1_base
does not need it.

Image: bind the pulled bundle's `configs/` to `/app/configs/bundles/run1_base` and your own
`configs/` to `/app/configs` (see [Into an image](#into-an-image)); the same
`../bundles/run1_base/...` reference then resolves inside the container.

Until the bundle is linked or bound, these configs raise
`FileNotFoundError: .../configs/bundles/run1_base/_base_PenCL.json` — that means fetch the
bundle first, not that the config is broken. `--input_data_path None` uses Stage 1's
built-in 5-protein test set, so the whole chain runs with no external data.

## Publishing (maintainers)

The publish runbook — `oras` login, build from a spec, push, and the visibility flip —
lives alongside the image's runbook in
[../../cloud/README.md](../../cloud/README.md) under **Publishing the weights bundle
(GHCR)**, so both GHCR publish flows sit in one place. A bundle's contents are declared by a
spec under `scripts/weights_bundle/bundle_specs/`.

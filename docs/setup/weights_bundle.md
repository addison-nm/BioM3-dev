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
through the symlink, and `configs/bundles/run1_base/` is what the consumer configs reference.
Keep the pulled bundle dir around — the symlinks point into it.

`--tag` is required — it names the bundle to pull; list published tags with `oras repo tags
ghcr.io/natural-machine/biom3-weights`. `--link` wires the bundle into the checkout;
`--quick-verify` checks sizes without hashing.

### Verifying an existing bundle

`verify_bundle.py` is stdlib-only and needs neither torch nor a `biom3` install, so it
works on a bare machine:

```bash
python scripts/weights_bundle/verify_bundle.py ~/biom3-bundles/run1_base
```

## Running with it

Three configs compose the bundle's pinned architecture over the repo defaults:

```bash
biom3_PenCL_inference \
    --input_data_path None \
    --config_path configs/inference/stage1_PenCL_run1_base.json \
    --model_path weights/PenCL/BioM3_PenCL_run1_base.bin \
    --output_path outputs/pencl_embeddings.pt

biom3_Facilitator_sample \
    --input_data_path outputs/pencl_embeddings.pt \
    --config_path configs/inference/stage2_Facilitator_run1_base.json \
    --model_path weights/Facilitator/BioM3_Facilitator_run1_base.bin \
    --output_data_path outputs/facilitator_embeddings.pt

biom3_ProteoScribe_sample \
    --input_path outputs/facilitator_embeddings.pt \
    --config_path configs/inference/stage3_ProteoScribe_sample_run1_base.json \
    --model_path weights/ProteoScribe/BioM3_ProteoScribe_run1_base.bin \
    --output_path outputs/generated_sequences.csv
```

Each config lists the bundle partial under `_overwrite_configs`, so bundle values win over
the repo's defaults but CLI flags still win over everything.

These three configs only resolve once the bundle is linked, because
`configs/bundles/run1_base/` is the symlink that `fetch_bundle.sh --link` (or the manual
`ln -s`) creates. Running them on a checkout that has not linked the bundle raises:

```
FileNotFoundError: .../configs/bundles/run1_base/_base_PenCL.json
```

That indicates one should fetch the bundle first, not that the config is itsel broken.

`--input_data_path None` uses Stage 1's built-in 5-protein test set, which makes the whole
chain runnable with no external data.

## Publishing (maintainers)

The publish runbook — `oras` login, build from a spec, push, and the visibility flip —
lives alongside the image's runbook in
[../../cloud/README.md](../../cloud/README.md) under **Publishing the weights bundle
(GHCR)**, so both GHCR publish flows sit in one place. A bundle's contents are declared by a
spec under `scripts/weights_bundle/bundle_specs/`.

# Weights Bundle (GHCR)

The BioM3 Run 1 base weights are published to GHCR as an OCI artifact, alongside the
software images. One `oras pull` gets a consistent, checksummed set of weights for all
three stages **plus** the architecture configs those weights belong to.

This is the alternative to [setup_shared_weights.md](setup_shared_weights.md) for anyone
without access to a cluster's shared BioM3-data-share directory. On Spark, Polaris and
Aurora, keep using the shared directory — it is already local.

| | |
|---|---|
| Package | `ghcr.io/natural-machine/biom3-weights` |
| Tags | `run1_base-<sha>` (immutable) · `run1_base` (moving pointer) |
| Size | ~6.44 GB across 14 files |
| Prereq | [`oras`](https://oras.land/docs/installation) — a single static binary |

## Why the configs ship with the weights

Nothing inside a weight file records which model graph it belongs to.
`configs/stage3_training/models/` has two ProteoScribe base configs differing in exactly
one key — `transformer_blocks`, 1 vs 16. The Run 1 Stage 3 weights are a **1-block**
model; loading them under the 16-block config fails. The bundle ships
`_base_ProteoScribe.json` with that value pinned, so the pairing travels with the weights
instead of living in someone's memory.

## Pulling

```bash
cd /path/to/BioM3-dev
./scripts/weights_bundle/fetch_bundle.sh ~/biom3-bundles
```

That does three things:

1. `oras pull` into `~/biom3-bundles/biom3-weights-run1_base/`
2. verifies every file's sha256 against the bundle's `MANIFEST.json`
3. symlinks the bundle into this checkout via the existing
   [`scripts/link_weights.sh`](../../scripts/link_weights.sh), and links the bundle's
   configs to `configs/bundles/run1_base/`

Step 3 is why nothing in the codebase needed to change. The bundle's `weights/` subtree
mirrors the repo's own layout, so after linking, the config path
`./weights/LLMs/esm2_t33_650M_UR50D.pt` resolves through a symlink into the bundle. Every
existing config keeps working.

Useful flags: `--no-link` to pull and verify only, `--quick-verify` to check sizes without
hashing, `--tag` to pull a specific immutable tag.

### Verifying an existing bundle

`verify_bundle.py` is stdlib-only and needs neither torch nor a `biom3` install, so it
works on a bare machine:

```bash
python3 scripts/weights_bundle/verify_bundle.py ~/biom3-bundles/biom3-weights-run1_base
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

These three configs only resolve **after** a fetch, because `configs/bundles/run1_base/` is
the symlink that `fetch_bundle.sh` creates. Running them on a checkout that has not pulled
the bundle raises:

```
FileNotFoundError: .../configs/bundles/run1_base/_base_PenCL.json
```

That means "fetch the bundle first", not "the config is broken".

`--input_data_path None` uses Stage 1's built-in 5-protein test set, which makes the whole
chain runnable with no external data.

## Contents

```
biom3-weights-run1_base/
  MANIFEST.json                        sha256 + size + provenance for every file
  README.md
  configs/
    _base_PenCL.json                   10 architecture keys
    _base_Facilitator.json              3
    _base_ProteoScribe.json            20, incl. transformer_blocks: 1
  weights/
    LLMs/
      esm2_t33_650M_UR50D.pt                          2.605 GB
      esm2_t33_650M_UR50D-contact-regression.pt       3.7 KB
      BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext/
        config.json  pytorch_model.bin
        tokenizer_config.json  vocab.txt  LICENSE.md  440.7 MB
    PenCL/BioM3_PenCL_run1_base.bin                   3.049 GB
    Facilitator/BioM3_Facilitator_run1_base.bin         4.2 MB
    ProteoScribe/BioM3_ProteoScribe_run1_base.bin     344.8 MB
```

Two things that look like they could be trimmed but cannot:

- **`esm2_..._-contact-regression.pt`.** `esm.pretrained.load_model_and_alphabet_local`
  loads this sibling file unconditionally for any model that is not esm1v/esm_if.
  Omitting it is a crash, not a degraded mode.
- **The ESM-2 and BiomedBERT backbones**, even though PenCL's own state dict overwrites
  every one of their tensors. They are needed to *construct* the graph before the PenCL
  weights load onto it. Removing that scaffolding is a planned follow-up that would cut
  the bundle to ~3.4 GB, but it requires code changes in `Stage1/model.py`.

Checkpoints are flattened to core weights at build time, which drops ~880 MB of optimizer
state relative to shipping the frozen `.ckpt` files whole.

## Publishing (maintainers)

### One-time setup

`oras` and a GHCR login with a classic PAT carrying `write:packages` (+ `read:packages`),
SSO-authorized for the org:

```bash
echo "$GHCR_TOKEN" | oras login ghcr.io -u <github-user> --password-stdin
```

### Build, rehearse, push

```bash
# 1. Build from the frozen checkpoints. Aborts if any source sha256 has drifted
#    from weights/Run1_frozen_ckpts/FROZEN_MANIFEST.md.
python scripts/weights_bundle/build_bundle.py -o ~/biom3-bundles

# 2. Rehearse on a ~350 MB subset — validates auth, artifact type, and the pull
#    side without committing to a 6.4 GB upload.
python scripts/weights_bundle/build_bundle.py -o ~/smoke --skip_llms
./scripts/weights_bundle/push_bundle.sh ~/smoke/biom3-weights-run1_base --smoke

# 3. Real push: two tags, run1_base-<sha> and run1_base.
./scripts/weights_bundle/push_bundle.sh ~/biom3-bundles/biom3-weights-run1_base
```

`push_bundle.sh` refuses to publish if the bundle's recorded build commit does not match
`HEAD`, or if the tree is dirty (`--allow-dirty` overrides, and suffixes the tag).

### Upload constraints

GHCR enforces a **10 GB limit per layer** and a **10-minute upload timeout**. Layer size
is not a concern — the largest blob is 3.05 GB. The timeout is: that blob needs a sustained
**~41 Mbps** upstream, and `oras` has no resumable upload, so a slow link fails at the end
of the largest file. Rehearse with `--smoke` first.

Because blobs are content-addressed, a later bundle that reuses the same ESM-2 file
re-uploads only what actually changed.

## Licensing

The package is intended to be public, which makes redistribution terms load-bearing:

- **BiomedBERT** — MIT (Microsoft), per the `LICENSE.md` shipped inside the bundle.
- **ESM-2** — terms must be confirmed before publishing. Not yet verified.
- **BioM3 weights** — BioM3-dev currently has no `LICENSE` file. This needs an explicit
  decision before the artifact is made public.

Keep the package private until all three are settled.

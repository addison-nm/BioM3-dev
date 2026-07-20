# BioM3 cloud jobs (`cloud/`)

`run.mithril.yaml` provisions a GPU on Mithril, pulls the public GHCR image
`ghcr.io/natural-machine/biom3:cuda-dev`, and runs whatever `CMD` you give it. The job
lives in `CMD`; the yaml only describes the machine. Nothing about where data lives is
baked into the image.

The image itself: [`../docker/README.md`](../docker/README.md).

## Launch

```bash
scripts/cloud/mithril_launch.sh cloud/run.mithril.yaml <name-prefix> \
  --config mithril.limit_price=<max $/hr> \
  --env CMD="<command>" \
  2>&1 | tee run.log
```

`mithril_launch.sh` harvests AWS credentials, picks a unique cluster name (Mithril
retains bid names, so reuse fails with a misleading `ResourcesUnavailableError`), and
auto-loads `configs/jobs/local.env`. The launch streams the job output — the trailing
`tee` is what keeps a local copy.

Add `--num-nodes N` for multi-node, `--gpus A100:4` for a different GPU count.

## Data staging

The container entrypoint stages inputs before `CMD` and pushes outputs after, all from
runtime env:

| Var | Direction |
| --- | --- |
| `BIOM3_WEIGHTS_URI` + `BIOM3_WEIGHTS_INCLUDES` | → `/app/weights` |
| `BIOM3_DATA_URI` | → `/app/data` |
| `BIOM3_OUTPUTS_PUSH_URI` | `/app/outputs` → (rank-0 node only) |
| `BIOM3_SYNC_CMD` / `BIOM3_SYNC_CMD_OUT` | custom pull/push command; `s3://` uses awscli |
| `BIOM3_SYNC_MODE` | `auto` (skip if dest populated) / `always` / `never` |

`BIOM3_WEIGHTS_URI` comes from the gitignored `configs/jobs/local.env`, so the examples
below only set `INCLUDES`. A URI with no `INCLUDES` is refused rather than pulling the
whole tree — pass `INCLUDES="*"` to opt into everything, or `BIOM3_WEIGHTS_URI=""` to
skip the weight sync.

Not on S3: set `BIOM3_SYNC_CMD` to any pull command. It runs with `BIOM3_SYNC_URI` and
`BIOM3_SYNC_DEST` exported, so rclone/gsutil/curl work without changing the image.

## Examples

### test

```bash
scripts/cloud/mithril_launch.sh cloud/run.mithril.yaml biom3-test \
  --config mithril.limit_price=6.00 \
  --env CMD="pytest tests/ --include_requires_gpu" \
  --env BIOM3_WEIGHTS_INCLUDES="LLMs/esm2_t33_650M_UR50D* LLMs/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext/* PenCL/BioM3_PenCL_epoch20.bin PenCL/PenCL_V09152023_last.ckpt* Facilitator/BioM3_Facilitator_epoch20.bin ProteoScribe/BioM3_ProteoScribe_pfam_epoch20_v1*.bin ProteoScribe/epoch200_full.ckpt/single_model.pth" \
  2>&1 | tee test.log
```

No weights (weight-gated tests skip):

```bash
scripts/cloud/mithril_launch.sh cloud/run.mithril.yaml biom3-test \
  --config mithril.limit_price=6.00 \
  --env CMD="pytest tests/ --quick" --env BIOM3_WEIGHTS_URI="" \
  2>&1 | tee test.log
```

### pretrain — Stage 3 from scratch

From-scratch needs no pretrained weights. This uses the test HDF5 baked into the image:

```bash
scripts/cloud/mithril_launch.sh cloud/run.mithril.yaml biom3-pt \
  --config mithril.limit_price=6.00 \
  --env BIOM3_WEIGHTS_URI="" \
  --env CMD="bash scripts/stage3_train_singlenode.sh \
      configs/stage3_training/pretrain_scratch_v1.json 1 cuda pt1 \
      --primary_data_path tests/_data/data/Stage2_MMD_swissprot_embedding_subset_1000.hdf5 \
      --epochs 1 --limit_val_batches 1.0" \
  2>&1 | tee pretrain.log
```

Positional args are `CONFIG_PATH NGPU DEVICE RUN_ID`; everything after is forwarded to
`biom3_train_stage3`. For a real dataset, sync it with `--env BIOM3_DATA_URI=s3://…` and
point `--primary_data_path` at `/app/data/…`.

### finetune — Stage 3

```bash
scripts/cloud/mithril_launch.sh cloud/run.mithril.yaml biom3-ft \
  --config mithril.limit_price=6.00 \
  --env BIOM3_WEIGHTS_INCLUDES="ProteoScribe/run1_base_proteoscribe.ckpt*" \
  --env CMD="bash scripts/stage3_train_singlenode.sh \
      configs/stage3_training/finetune_v1.json 1 cuda ft1 \
      --finetune True \
      --pretrained_weights weights/ProteoScribe/run1_base_proteoscribe.ckpt \
      --primary_data_path tests/_data/data/Stage2_MMD_swissprot_embedding_subset_1000.hdf5 \
      --epochs 1 --limit_val_batches 1.0" \
  2>&1 | tee finetune.log
```

Starting from a CSV instead of a precompiled HDF5, chain the embedding pipeline first
(this also needs the PenCL + Facilitator weights in `INCLUDES`):

```bash
  --env CMD="biom3_embedding_pipeline -i /app/data/my.csv -o /app/outputs/emb --prefix ds1 \
        --weight_set configs/weights/run1_base.json \
        --pencl_config configs/inference/stage1_PenCL.json \
        --facilitator_config configs/inference/stage2_Facilitator.json \
      && bash scripts/stage3_train_singlenode.sh \
        configs/stage3_training/finetune_v1.json 1 cuda ft1 \
        --finetune True --pretrained_weights weights/ProteoScribe/run1_base_proteoscribe.ckpt \
        --primary_data_path /app/outputs/emb/ds1.compiled_emb.hdf5 --epochs 1"
```

### generate — Stage 1 → 2 → 3

```bash
scripts/cloud/mithril_launch.sh cloud/run.mithril.yaml biom3-gen \
  --config mithril.limit_price=6.00 \
  --env BIOM3_WEIGHTS_INCLUDES="LLMs/esm2_t33_650M_UR50D* LLMs/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext/* PenCL/run1_base_pencl.ckpt* Facilitator/run1_base_facilitator.ckpt* ProteoScribe/run1_base_proteoscribe.ckpt*" \
  --env CMD="biom3_embedding_pipeline --generate \
      -i tests/_data/stage1_inputs/sample_text_seqs1.csv \
      -o /app/outputs/gen1 --prefix gen1 \
      --weight_set configs/weights/run1_base.json \
      --pencl_config configs/inference/stage1_PenCL.json \
      --facilitator_config configs/inference/stage2_Facilitator.json \
      --proteoscribe_config configs/inference/stage3_ProteoScribe_sample.json" \
  2>&1 | tee generate.log
```

The prompts CSV needs the columns the PenCL config expects (`protein_sequence` +
`primary_Accession`). FASTA is written by default; `--no_fasta` disables it.

Multi-GPU or multi-node sampling: run the same command under a launcher, e.g.
`NGPU=4 bash scripts/launchers/container_singlenode.sh biom3_embedding_pipeline --generate …`.
Stages 1-2 are deterministic, so every rank computes identical embeddings; the Stage 3
sampler shards by rank and only rank 0 writes.

### keeping outputs

Add `--env BIOM3_OUTPUTS_PUSH_URI=s3://<bucket>/<prefix>/<run>` to any of the above.
`/app/outputs` is pushed when `CMD` exits, from the rank-0 node only.

---

## Publishing the image (GHCR) — runbook

The image is published **public** at **`ghcr.io/natural-machine/biom3`**, tagged
`cuda-<sha>` (immutable, per commit) and `cuda-dev` (moving; what `run.mithril.yaml`
tracks). Both are **multi-arch manifest lists** covering `linux/amd64` (cloud
instances) and `linux/arm64` (DGX Spark), so the same tag runs on either.

Why GHCR + public:
- **Cost**: Mithril instances are ephemeral, so *every launch pulls the whole ~11.6 GB
  image*. From ECR that is internet egress (~$0.09/GB ≈ **$1/launch**, billed under the
  AWS EC2/"EC2-Other" line). GitHub Packages is **free for public packages**.
- **Simplicity**: a public image needs **no pull authentication**, so the launch path
  carries no registry token at all.

> **Before you publish**: the image bakes `src/`, `scripts/`, `tests/` (incl. the test
> HDF5s) and `configs/`. Publishing it **makes all of that world-readable** — confirm
> that is intended.

### One-time: create a token and publish

```bash
# 1. Create a classic PAT (push side only).
#    GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)
#    → Generate new token (classic) → scopes: write:packages, read:packages
#    → If the org enforces SAML SSO: click "Configure SSO" → Authorize for natural-machine
export GHCR_TOKEN=ghp_xxxxxxxxxxxxxxxxxxxx

# 2. Log in to GHCR (needed to PUSH; pulling a public image needs no login).
echo "$GHCR_TOKEN" | docker login ghcr.io -u addison-nm --password-stdin

# 3. Build and publish (see the multi-arch section below).

# 4. Make the package PUBLIC — the first push creates it PRIVATE by default.
#    Web: https://github.com/orgs/natural-machine/packages → biom3
#         → Package settings → Danger Zone → Change visibility → Public

# 5. Verify an ANONYMOUS pull — this is exactly what a Mithril instance does.
docker logout ghcr.io
docker pull ghcr.io/natural-machine/biom3:cuda-dev
```

### On every image change — build each arch natively, then merge

The image bakes `src/ scripts/ tests/ configs/`, so any change to them — or to
`docker/entrypoint.sh` — needs a rebuild and repush before it reaches a cloud run.

Each architecture is built on matching hardware and the results merged. Emulating the
other architecture costs hours on this image (~1 GB torch wheel, DeepSpeed compiles
from source), so cross-building is a fallback, not the default.

On an **arm64** host (DGX Spark) *and* an **amd64** host (x86 box, cloud instance, or
CI runner), from the same clean commit:

```bash
echo "$GHCR_TOKEN" | docker login ghcr.io -u addison-nm --password-stdin
docker/build.sh --variant cuda --awscli          # builder's native platform
docker/push.sh  --variant cuda --arch            # -> cuda-<sha>-amd64 / -arm64
```

Then once, from either host:

```bash
docker/push.sh --variant cuda --join             # -> cuda-<sha> + cuda-dev
docker manifest inspect ghcr.io/natural-machine/biom3:cuda-dev
```

The `inspect` should list both `linux/amd64` and `linux/arm64`. `--arch` reads the
architecture from the image itself, and `--join` refuses unless both per-arch tags
exist. The dirty-tree guard applies to both, so `cuda-<sha>` still matches the commit.

#### Adopting an existing single-arch image

`imagetools` composes manifest lists from images already in the registry — no rebuild,
no pull. To reuse an image pushed before this flow existed:

```bash
REPO=ghcr.io/natural-machine/biom3
docker buildx imagetools create -t ${REPO}:cuda-<sha>-amd64 ${REPO}:cuda-<sha>
```

Brace the variable — zsh reads an unbraced `$REPO:c…` as the `:c` history modifier
and silently drops the `:c`, producing a garbage image reference.

#### Cross-building on one host (slow fallback)

Emulates the non-native architecture. `buildx` cannot `--load` a multi-platform build,
so it must push directly and `push.sh` is not involved:

```bash
docker/build.sh --variant cuda --awscli --platform linux/amd64,linux/arm64 \
    --tag ghcr.io/natural-machine/biom3:cuda-<sha> --push
```

### What this does NOT remove

**AWS credentials are still required** when your data is on S3 — GHCR only replaces the
*image registry*. The entrypoint still syncs from S3, so launches pass the `AWS_*` trio
and that egress still bills to AWS. Narrow it with `BIOM3_WEIGHTS_INCLUDES`.

---

## Launching from a separate machine

You can drive Mithril launches from any host — e.g. an EC2 instance hosting a web app.
The launching machine is a **thin client**: no GPU, no Docker, no repo, no weights.

### It needs

1. **`uv`** (or pip) — to install the CLI.
2. **`mithril-client`** — `uv tool install -U mithril-client` (provides `mithril` + the
   bundled `sky`).
3. **Mithril auth** *(separate from AWS)*: `~/.config/mithril/config.yaml` with
   `api_key` + `project_id`, or the `MITHRIL_API_KEY` / `MITHRIL_PROJECT` env vars.
4. **AWS CLI + credentials** — only if your data is on S3. The identity needs
   `s3:GetObject`/`ListBucket` on the bucket. On EC2 an **IAM instance role** is cleanest.
5. **`cloud/run.mithril.yaml`** and `scripts/cloud/mithril_launch.sh` — **not** the whole
   repo; the code is baked into the image.
6. **Outbound HTTPS** to `api.mithril.ai`, `ghcr.io`, and your object store.
7. A little local disk — `mithril sky launch` runs a local SkyPilot API server and keeps
   state in `~/.sky/`.

### It does NOT need

- ❌ **Docker** — the pull/run happen on the remote GPU instance.
- ❌ a **registry token** — the GHCR image is public.
- ❌ a **GPU**, the **weights**, or the **BioM3-dev repo**.

### Operational notes

- **Use a unique cluster name every launch.** Mithril retains bid names indefinitely, so
  reuse fails with a *misleading* `ResourcesUnavailableError`. The helper appends a
  timestamp; a web app must do the same.
- **`--down` only fires after a job finishes.** A provision-stage failure leaves the
  instance billing — reconcile with `mithril sky status` / `mithril sky down <c>`.
- **Instances that never get an SSH IP** are a Mithril-side failure: the bid clears, an
  instance is allocated, then SkyPilot waits out a hardcoded 3600 s timeout and cancels
  the bid. Ctrl-C rather than waiting the hour.
- **The local API server accumulates state.** Run
  [`scripts/cloud/mithril_reset.sh`](../scripts/cloud/mithril_reset.sh) when a launch wedges.
- **Temporary creds expire** (SSO/role, ~1–12 h). The helper re-harvests every launch.

# BioM3 cloud jobs — test / finetune / generate on Mithril & AWS

A provider-agnostic way to run BioM3 jobs on commercial GPU cloud. The same job
runs on **Mithril** or **AWS**, with inputs/outputs on **S3** or a **local
filesystem**, by setting a handful of variables and launching.

For the underlying image and the original bring-up, see
[`docker/README.md`](../../docker/README.md) and
[`mithril/INSTRUCTIONS.md`](../../mithril/INSTRUCTIONS.md).

---

## Mental model — three orthogonal axes

| Axis | Choices | Where it's set |
| --- | --- | --- |
| **Job** (what runs) | `test` / `finetune` / `generate` | `scripts/cloud/<job>.sh` (baked into the image) |
| **Compute** (where) | Mithril / AWS | `cloud/<job>.<provider>.yaml` (`resources.infra`) |
| **Data** (in/out) | S3 / local FS | env vars (`BIOM3_*_URI`) or SkyPilot `file_mounts` |

- **Job scripts** (`scripts/cloud/{test,finetune,generate}.sh`) are env-var driven
  and provider/data agnostic. They only ever read `/app/...` paths and reuse the
  existing `biom3_*` entry points.
- **Task files** (`cloud/*.yaml`) are thin SkyPilot specs that run *inside* the
  prebuilt ECR image (`resources.image_id`). The `envs:` block IS the job spec —
  the "couple of variables" you edit per run.
- **Data movement** is handled by the container entrypoint
  ([`docker/entrypoint.sh`](../../docker/entrypoint.sh)): set `BIOM3_WEIGHTS_URI`
  / `BIOM3_DATA_URI` to sync S3 → container, and `BIOM3_OUTPUTS_PUSH_URI` to push
  `/app/outputs` → S3 on exit.

The same job scripts also run under a bare `docker/run.sh` (local Docker, bind
mounts) — see [§ Local Docker](#local-docker-no-skypilot).

---

## One-time setup

### 1. Register the image in ECR (so instances pull, not build)

Build on a `linux/amd64` host (or the cloud instance), then push:

```bash
eval "$(aws configure export-credentials --format env)"      # AWS creds in shell

# Build + push in one step:
docker/build.sh --awscli --platform linux/amd64 \
    --tag 955510722784.dkr.ecr.us-east-2.amazonaws.com/biom3:gpu --push

# …or push an image you already built locally:
docker/push.sh --create-repo        # tag + ECR login + push biom3:gpu
```

The `cloud/*.yaml` files already point at
`955510722784.dkr.ecr.us-east-2.amazonaws.com/biom3:gpu`. Rebuild + repush
whenever you change `scripts/cloud/*.sh` or `src/` (image_id mode runs the baked
code, not your working tree).

### 2. Install the launcher CLI

`ml` (Mithril client) or `sky` (SkyPilot) — see
[`mithril/INSTRUCTIONS.md` §0.b](../../mithril/INSTRUCTIONS.md).

### 3. Launch-shell credentials (every session)

```bash
eval "$(aws configure export-credentials --format env)"      # AWS_* trio (temp creds)
export ECR_PASSWORD="$(aws ecr get-login-password --region us-east-2)"  # ECR pull
```

SkyPilot substitutes these into the task's `secrets:` block. On AWS you may
instead leave `SKYPILOT_DOCKER_USERNAME`/`PASSWORD` empty and let the instance
IAM role authenticate to ECR.

---

## Launch a job

Edit the `envs:` block of the relevant `cloud/<job>.<provider>.yaml`, then:

```bash
# Test suite
ml launch cloud/test.mithril.yaml     -c biom3-test
ml launch cloud/test.aws.yaml         -c biom3-test

# Finetuning (Stage 3)
ml launch cloud/finetune.mithril.yaml -c biom3-ft
ml launch cloud/finetune.aws.yaml     -c biom3-ft

# Generation (Stage 3 inference)
ml launch cloud/generate.mithril.yaml -c biom3-gen
ml launch cloud/generate.aws.yaml     -c biom3-gen

ml logs <cluster>      # stream output
ml down <cluster>      # tear down — the cluster bills until you do
```

Override any `envs:` value at launch instead of editing the file:

```bash
ml launch cloud/finetune.mithril.yaml -c biom3-ft \
    --env RUN_ID=my_run --env EPOCHS=5 --env PRIMARY_HDF5=data/my.hdf5
```

---

## Job env-var reference

### `finetune` — `scripts/cloud/finetune.sh`

Embed (optional) → split (optional) → finetune ProteoScribe. Outputs land under
the trainer's `--output_root` (default `outputs/Stage3/finetuning/{checkpoints,runs}/<RUN_ID>/`).

| Var | Default | Meaning |
| --- | --- | --- |
| `DATASET_CSV` **or** `PRIMARY_HDF5` | — (one required) | input CSV (→ embed) or precompiled Stage-2 HDF5 |
| `WEIGHT_SET` | `configs/weights/run1_base.json` | bundle: pencl+facilitator (embed) + proteoscribe (init) |
| `PROTEOSCRIBE_INIT` | proteoscribe_weights from `WEIGHT_SET` | finetune init weights |
| `FINETUNE_CONFIG` | `configs/stage3_training/finetune_v1.json` | Stage-3 training config |
| `RUN_ID` | auto (`<config>_ft_n1_d<NGPU>_e<EPOCHS>_V<ts>`) | run identifier |
| `DEVICE` / `NGPU` / `EPOCHS` | `cuda` / `1` / — | device, GPUs (>1 → torchrun), epochs |
| `SPLIT` | `none` | `none` (trainer's `valid_size`) or `cluster` (needs `mmseqs`†) |
| `SPLIT_MANIFEST` | — | use a prebuilt split manifest directly |

† `mmseqs` is **not** in the base image. `SPLIT=cluster` requires adding it to the
image or providing `CLUSTERS_TSV`; otherwise the split step fails with a clear
message. Extra args are forwarded to the trainer (e.g. `--batch_size 8`).

### `generate` — `scripts/cloud/generate.sh`

PenCL → Facilitator → ProteoScribe sampling. Outputs `.pt` (+ `.fasta`) under
`outputs/<OUTPUT_PREFIX>/`.

| Var | Default | Meaning |
| --- | --- | --- |
| `PROMPTS_CSV` | — (required) | input CSV of prompts |
| `WEIGHT_SET` | `configs/weights/run1_base.json` | all three stage weights |
| `PENCL_WEIGHTS` / `FACILITATOR_WEIGHTS` / `PROTEOSCRIBE_WEIGHTS` | from `WEIGHT_SET` | per-stage overrides |
| `OUTPUT_PREFIX` | `gen_<ts>` | filename prefix / output subdir |
| `DEVICE` | `cuda` | cpu/cuda/xpu |
| `FASTA` | `true` | also write FASTA |
| `TOKEN_STRATEGY` / `UNMASKING_ORDER` / `SEED` | — / — / `0` | sampling controls |

### `test` — `scripts/cloud/test.sh`

| Var | Default | Meaning |
| --- | --- | --- |
| `PYTEST_ARGS` | `tests/ --use_gpu` | passed verbatim to `pytest` |

Copy-pasteable specs: [`configs/jobs/finetune.example.env`](../../configs/jobs/finetune.example.env),
[`configs/jobs/generate.example.env`](../../configs/jobs/generate.example.env).

---

## Data: S3 vs local

**S3 (recommended for cloud):** set in `envs:` —
- `BIOM3_WEIGHTS_URI` (+ `BIOM3_WEIGHTS_INCLUDES` to narrow the pull) → `/app/weights`
- `BIOM3_DATA_URI` → `/app/data`
- `BIOM3_OUTPUTS_PUSH_URI` → pushes `/app/outputs` on exit

Then reference in-container paths (e.g. `PRIMARY_HDF5=data/foo.hdf5`).

**Local filesystem:** upload local dirs to the instance with SkyPilot `file_mounts`,
mapping them into the container's `/app/...`:

```yaml
file_mounts:
  /app/data: ./data/datasets/SH3        # local dir -> container path
```

(Leave the matching `BIOM3_*_URI` unset so the entrypoint doesn't try to sync.)

---

## Local Docker (no SkyPilot)

The job scripts are the same unit of work locally. `docker/run.sh` bind-mounts
`./weights`, `./data`, `./outputs` and forwards env vars:

```bash
set -a; source configs/jobs/finetune.example.env; set +a
docker/run.sh scripts/cloud/finetune.sh --epochs 1

# CPU smoke, no weights/GPU, using the baked test subset:
PRIMARY_HDF5=tests/_data/data/Stage2_MMD_swissprot_embedding_subset_1000.hdf5 \
FINETUNE_CONFIG=configs/stage3_training/pretrain_scratch_v1.json \
RUN_ID=smoke01 DEVICE=cpu NGPU=1 PROTEOSCRIBE_INIT=none \
BIOM3_DATA_DIR=$PWD/tests/_data/data \
docker/run.sh scripts/cloud/finetune.sh --epochs 1
```

---

## Gotchas

- **image_id bypasses the image ENTRYPOINT.** The `cloud/*.yaml` `run:` blocks
  invoke `/usr/local/bin/entrypoint.sh <job>.sh` explicitly so the S3 sync and
  `environment.sh` sourcing still happen. Keep that wrapper if you write new tasks.
- **Baked code, not your working tree.** image_id runs `/app` from the image —
  rebuild + repush after editing `scripts/cloud/*.sh`.
- **Temp AWS creds expire** (1–12 h); refresh the launch-shell `eval` + `ECR_PASSWORD`
  for long runs.
- **`mmseqs` absent** → `SPLIT=cluster` fails; use `SPLIT=none` or add the binary.
- See also the Mithril-specific gotchas in [`HANDOFF.md`](../../HANDOFF.md).

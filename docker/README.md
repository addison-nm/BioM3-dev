# BioM3 Docker image (AWS / Mithril GPU cloud)

Recipes for running BioM3 — **training (all stages), finetuning, and generation** — in a
container on commercial GPU cloud (AWS GPU instances, Mithril), e.g. spot instances.

**One image, all uses.** A single CUDA image holds the full BioM3 install; you pick what
to run at `docker run` time. Built per architecture:

- **x86_64 NVIDIA** — AWS `p3/p4/p5`, `g4/g5/g6`; Mithril H100/H200.
- **ARM64 NVIDIA** — Grace / GH200.

> **Not for ALCF HPC.** Polaris and Aurora use apptainer and a different software stack —
> see [`docs/setup/setup_polaris.md`](../docs/setup/setup_polaris.md) and
> [`setup_aurora.md`](../docs/setup/setup_aurora.md).

| File | Purpose |
| ---- | ------- |
| `Dockerfile` | The image: `nvidia/cuda:12.9` → py3.12 → torch 2.8 (cu129) → BioM3 (`pip install -e .[app]`). |
| `Dockerfile.dockerignore` | Trims the build context (BuildKit picks it up for `-f docker/Dockerfile`). |
| `entrypoint.sh` | Sources `environment.sh`, runs the optional object-store sync, then exec's your command. |
| `build.sh` | `docker buildx` wrapper (platform, tag, awscli, push). |
| `run.sh` | `docker run` wrapper for GPU hosts: standard mounts + env passthrough. |
| `docker-compose.yml` | Optional services for the streamlit `app` (port 8501) and a `shell`. |

---

## Build

Requires Docker with BuildKit/buildx. The image is ~12–15 GB; the first build downloads
the ~3 GB torch cu129 wheel (10–40 min). Subsequent builds reuse the buildx layer cache.

```bash
# Native architecture (Apple Silicon → arm64; x86 Linux/Mac → amd64):
docker/build.sh                      # tags biom3:cuda

# Explicit platform:
docker/build.sh --platform linux/amd64
docker/build.sh --platform linux/arm64

# Bake in awscli for the S3 sync hook (off by default):
docker/build.sh --awscli

# Multi-arch (requires --push; buildx --load is single-platform only):
docker/build.sh --platform linux/amd64,linux/arm64 --tag <registry>/biom3:cuda --push
```

Cross-arch builds (building arm64 on an x86 host or vice-versa) need QEMU/binfmt
registered in the buildx builder. Building natively on each arch is simplest.

**Network note:** the torch wheel comes from `download.pytorch.org`. On a VPN with
upstream DNS filtering (e.g. Tailscale MagicDNS) the download can stall on `Failed to
resolve download.pytorch.org` — disconnect the VPN for the build.

---

## Getting weights and data into the container

BioM3 needs pretrained **weights** (ESM-2 + BioBERT + per-stage checkpoints) and, for
training, **datasets**. Neither is baked into the image. Two ways to supply them:

### 1. Bind-mount (default, cloud-agnostic)

Stage the files on the host (an attached EBS / Mithril volume) and mount them.
`docker/run.sh` wires up the conventional layout automatically:

| Host (default) | Container | Mode | Contents |
| --- | --- | --- | --- |
| `./weights` | `/app/weights` | ro | `LLMs/`, `PenCL/`, `Facilitator/`, `ProteoScribe/` |
| `./data` | `/app/data` | ro | training datasets (CSV / HDF5) |
| `./outputs` | `/app/outputs` | **rw** | checkpoints, logs, generated sequences |
| `./configs` | `/app/configs` | ro | *(optional)* overrides the configs baked into the image |

Override the host dirs with `BIOM3_WEIGHTS_DIR`, `BIOM3_DATA_DIR`, `BIOM3_OUTPUTS_DIR`,
`BIOM3_CONFIGS_DIR`. The weights layout mirrors
[`docs/setup/setup_shared_weights.md`](../docs/setup/setup_shared_weights.md).

### 2. Object-store sync (optional, for fresh spot instances)

The entrypoint can pull weights/data from an object store on container start. Default
path uses `aws s3 sync` (build with `--awscli`); any other tool works via
`BIOM3_SYNC_CMD`. Set on `docker run` (or export before `docker/run.sh`):

| Env var | Effect |
| --- | --- |
| `BIOM3_WEIGHTS_URI` | e.g. `s3://bucket/biom3/weights` → synced to `/app/weights`. |
| `BIOM3_DATA_URI` | e.g. `s3://bucket/biom3/data` → synced to `/app/data`. |
| `BIOM3_WEIGHTS_INCLUDES` | space-separated `--include` globs (weights only); narrows the sync. |
| `BIOM3_SYNC_MODE` | `auto` (default; skip if dir already populated) \| `always` \| `never`. |
| `BIOM3_SYNC_CMD` | custom pull command (`rclone`, `gsutil`, …); receives `$BIOM3_SYNC_URI`, `$BIOM3_SYNC_DEST`. |
| `BIOM3_OUTPUTS_PUSH_URI` | push `/app/outputs` here on exit (best-effort; a mounted volume is the primary persistence). |
| `AWS_ENDPOINT_URL` | point `aws` at an S3-compatible store (Mithril, MinIO, …). |

> **Recommended for spot:** the simplest robust pattern is host-side staging — your
> instance bootstrap pulls weights/datasets to a local dir, then `docker/run.sh`
> bind-mounts them. The container then needs no cloud tools or credentials.

---

## Run

`docker/run.sh <command...>` assembles `docker run --gpus all` with the mounts above and
forwards `WANDB_API_KEY`, `NGPU`, and `AWS_*`/`BIOM3_*` env vars. Examples below use it;
the raw `docker run` equivalents are in the per-section notes.

Requires the **NVIDIA Container Toolkit** on the host (`--gpus all`).

### Generation (Stage 1 → 2 → 3)

Single-process; no launcher needed. Run the three stages in sequence:

```bash
docker/run.sh biom3_PenCL_inference \
    --input_data_path data/my_proteins.csv \
    --config_path configs/inference/stage1_PenCL.json \
    --model_path weights/PenCL/BioM3_PenCL_epoch20.bin \
    --output_path outputs/pencl_embeddings.pt --device cuda

docker/run.sh biom3_Facilitator_sample \
    --input_data_path outputs/pencl_embeddings.pt \
    --config_path configs/inference/stage2_Facilitator.json \
    --model_path weights/Facilitator/BioM3_Facilitator_epoch20.bin \
    --output_data_path outputs/facilitator_embeddings.pt --device cuda

docker/run.sh biom3_ProteoScribe_sample \
    --input_path outputs/facilitator_embeddings.pt \
    --config_path configs/inference/stage3_ProteoScribe_sample.json \
    --model_path weights/ProteoScribe/BioM3_ProteoScribe_pfam_epoch20_v1.bin \
    --output_path outputs/generated_sequences.pt --device cuda --fasta
```

(Use `--device cpu` for a cheap smoke test on tiny inputs without a GPU.)

### Training (Stages 1, 2, 3)

The existing wrapper scripts work unchanged in the container. Inside, `BIOM3_MACHINE` is
`container`, so they dispatch to `scripts/launchers/container_singlenode.sh`, which uses
**`torchrun`** for multi-GPU (see [How it works](#how-it-works-inside-the-container)).

Wrapper signature: `scripts/stageN_train_singlenode.sh CONFIG_PATH NGPU DEVICE RUN_ID [--overrides…]`

```bash
# Stage 3 pretrain from scratch, single GPU:
docker/run.sh scripts/stage3_train_singlenode.sh \
    configs/stage3_training/pretrain_scratch_v1.json 1 cuda run001 --epochs 1

# Stage 3, 4 GPUs (NGPU must match the wrapper's NGPU arg; torchrun spawns 4 ranks):
BIOM3_GPUS=all NGPU=4 docker/run.sh scripts/stage3_train_singlenode.sh \
    configs/stage3_training/pretrain_scratch_v1.json 4 cuda run001 --epochs 5

# Stage 1 (PenCL) and Stage 2 (Facilitator):
docker/run.sh scripts/stage1_train_singlenode.sh \
    configs/stage1_training/pretrain_scratch_v1.json 1 cuda s1run001
docker/run.sh scripts/stage2_train_singlenode.sh \
    configs/stage2_training/pretrain_scratch_v1.json 1 cuda s2run001
```

`WANDB_API_KEY` (forwarded by `run.sh` when set) enables Weights & Biases logging
automatically; otherwise it defaults off. Pass `--wandb True|False` to force it.

### Finetuning (Stage 3)

Finetuning is the Stage 3 trainer with `--finetune` flags + base weights:

```bash
docker/run.sh scripts/stage3_train_singlenode.sh \
    configs/stage3_training/finetune_v1.json 1 cuda ft001 \
    --finetune True \
    --pretrained_weights weights/ProteoScribe/BioM3_ProteoScribe_pfam_epoch20_v1.bin \
    --finetune_last_n_blocks 1 --finetune_last_n_layers 1 \
    --primary_data_path data/my_finetune_set.hdf5 --epochs 10
```

### Resume (spot preemption)

Checkpoints land in the mounted `outputs/` and survive container exit. After a
preemption, resume from `last.ckpt`:

```bash
docker/run.sh scripts/stage3_train_singlenode.sh \
    configs/stage3_training/pretrain_scratch_v1.json 1 cuda run001 \
    --epochs 5 --resume_from_checkpoint outputs/<...>/checkpoints/run001/last.ckpt
```

### Web app

```bash
docker compose -f docker/docker-compose.yml up app    # http://localhost:8501
```

### Interactive shell

```bash
docker/run.sh bash
# or: docker compose -f docker/docker-compose.yml run --rm shell
```

### Spot instance with S3, no persistent disk (e.g. Mithril)

When there's no persistent volume, pull inputs from S3 on start and push results
back on exit. Mithril gives you an SSH GPU VM (Ubuntu, x86) — ensure Docker + the
NVIDIA Container Toolkit, then:

```bash
# Build with awscli so the entrypoint can sync (off by default):
docker/build.sh --awscli --platform linux/amd64

# Credentials reach the container via env vars (forwarded) or ~/.aws (bind-mounted):
export AWS_ACCESS_KEY_ID=...  AWS_SECRET_ACCESS_KEY=...  AWS_DEFAULT_REGION=us-east-2

# S3 trees mirror the local weights/ and data/ layout:
export BIOM3_WEIGHTS_URI=s3://your-bucket/biom3/weights
export BIOM3_DATA_URI=s3://your-bucket/biom3/data
export BIOM3_OUTPUTS_PUSH_URI=s3://your-bucket/biom3/outputs/run001
export BIOM3_WEIGHTS_INCLUDES="LLMs/* ProteoScribe/*"   # optional: narrow the pull

docker/run.sh scripts/stage3_train_singlenode.sh \
    configs/stage3_training/pretrain_scratch_v1.json 1 cuda run001 --epochs 1 --max_steps 5
```

`run.sh` skips the `weights/`/`data/` bind-mounts when their `*_URI` is set, so the
entrypoint syncs into the container's own dirs. Two caveats for the disk-less model:

- `docker run --rm` re-pulls weights each run. While iterating, use `docker/run.sh bash`
  (sync once, run many commands, push on exit) or a Docker named volume for
  `/app/weights` (`BIOM3_SYNC_MODE=auto` then skips the re-pull).
- The outputs push is **best-effort on exit** — fine for a short test, but a spot
  preemption can kill a long run before the final push. Periodic checkpoint upload is
  the real fix (a follow-up).

---

## How it works inside the container

- The image sets `ENV BIOM3_MACHINE=container`. The entrypoint sources `environment.sh`,
  which honors that value and applies the (minimal) container settings.
- **Single-node multi-GPU uses `torchrun`, not `mpiexec`/PBS.**
  `scripts/launchers/container_singlenode.sh`: 1 GPU → `exec` directly; N GPUs →
  `torchrun --standalone --nproc-per-node=N`, which sets `RANK`/`LOCAL_RANK`/`WORLD_SIZE`
  /`MASTER_ADDR`/`MASTER_PORT`. BioM3 already reads these
  ([`core/_dist_env.py`](../src/biom3/core/_dist_env.py)) and PyTorch Lightning
  auto-detects the torchelastic environment (it does not re-spawn).
- OpenMPI is in the image only so `mpi4py` (a pinned dependency) builds; it is not used
  to launch.

## Quick local sanity check (no GPU required)

```bash
docker/build.sh
docker run --rm biom3:cuda python -c "import biom3, torch; print(torch.__version__)"
docker run --rm biom3:cuda pytest tests/ --quick
```

## Multi-node training (SkyPilot)

Finetuning, pretraining, and generation run across multiple instances via
[`scripts/launchers/container_multinode.sh`](../scripts/launchers/container_multinode.sh),
the multi-node analog of `container_singlenode.sh`. SkyPilot launches the task once per
**node** and sets `SKYPILOT_NODE_RANK` / `SKYPILOT_NUM_NODES` / `SKYPILOT_NODE_IPS`; the
launcher translates those into a **`torchrun` static rendezvous**
(`--nnodes --node-rank --master-addr --nproc-per-node`), which sets
`RANK`/`LOCAL_RANK`/`WORLD_SIZE`/`GROUP_RANK` — the same env BioM3
([`core/_dist_env.py`](../src/biom3/core/_dist_env.py)) and Lightning already read. No
Python changes.

- **Enable it:** in `cloud/{finetune,generate,pretrain}.mithril.yaml`, set
  `resources.num_nodes > 1` and `accelerators` to the **per-node** GPU count (`NGPU` must
  match). The job scripts dispatch on `NNODES` (defaulting to `$SKYPILOT_NUM_NODES`) to
  `stage3_train_multinode.sh`.
- **Checkpointing = DDP, not DeepSpeed.** Mithril/AWS spot clusters have **no shared
  filesystem**, so DeepSpeed ZeRO's per-rank optimizer shards would scatter across nodes'
  local disks and can't be consolidated. The scripts default `--distributed_strategy ddp`
  when `NNODES>1`: a single `.ckpt` is written entirely by **global rank 0** (on the
  `SKYPILOT_NODE_RANK==0` node), so the task yaml gates `BIOM3_OUTPUTS_PUSH_URI` to that
  node. Revisit DeepSpeed multi-node only with a shared FS.
- **NCCL:** the launcher auto-detects the private-net (`10.x`) interface for
  `NCCL_SOCKET_IFNAME` and disables InfiniBand (`NCCL_IB_DISABLE=1`); the task uses
  `--net=host --ipc=host`. Debug a first run with `NCCL_DEBUG=INFO`.
- **Data:** each node's container S3-syncs its own copy — prefer `PRIMARY_HDF5`/
  `BIOM3_DATA_URI` (identical bytes per node) over on-the-fly CSV embedding so every rank
  builds identical `DistributedSampler` shards.
- **Generation** parallelizes only Stage-3 sampling (the sampler is rank-aware; only rank
  0 writes). Stages 1–2 run per node and Facilitator sampling is stochastic — verify
  determinism with a fixed `SEED` before trusting multi-node output.

The ALCF path (`scripts/launchers/{aurora,polaris}_multinode.sh`, mpiexec/PBS) remains the
reference for the HPC clusters.

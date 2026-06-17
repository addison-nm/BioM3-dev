# Setup: Docker (AWS / Mithril GPU cloud)

For running BioM3 in a container on commercial GPU cloud (AWS GPU instances, Mithril) —
training (all stages), finetuning, and generation. Unlike the other targets there is no
conda/venv to create: you build one CUDA image and run it.

> ALCF HPC (Polaris/Aurora) is a separate case — those use apptainer; see their setup
> docs. This guide is for Docker on commercial cloud only.

The full reference (build options, all run recipes, the object-store sync, the streamlit
app, multi-arch) lives in [`docker/README.md`](../../docker/README.md). Quick start:

## Prerequisites

- Docker with BuildKit/buildx.
- For GPU runs: an NVIDIA GPU host with the **NVIDIA Container Toolkit** (`--gpus all`).

## Build

One image, all uses (train/finetune/generate + the app). Built per architecture:

```bash
cd /path/to/BioM3-dev
docker/build.sh                          # native arch → tags biom3:gpu
# or pin: docker/build.sh --platform linux/amd64   (AWS/Mithril x86)
#         docker/build.sh --platform linux/arm64   (Grace / GH200)
```

Verify the build (no GPU needed):

```bash
docker run --rm biom3:gpu python -c "import biom3, torch; print(torch.__version__)"
docker run --rm biom3:gpu pytest tests/ --quick
```

## Supplying weights and data

Not baked into the image. Either bind-mount host dirs (default) or sync from an object
store. `docker/run.sh` mounts the conventional layout (`weights/` ro, `data/` ro,
`outputs/` rw). Populate `weights/` per
[`setup_shared_weights.md`](./setup_shared_weights.md). See
[`docker/README.md`](../../docker/README.md) for the sync-from-S3 option and the
recommended host-side staging pattern for spot instances.

## Usage

`docker/run.sh <command…>` runs on a GPU host with the standard mounts and forwards
`WANDB_API_KEY` / `NGPU` / `AWS_*`. `environment.sh` is sourced inside the container
automatically (it auto-detects `BIOM3_MACHINE=container`).

```bash
# Train Stage 3 (single GPU)
docker/run.sh scripts/stage3_train_singlenode.sh \
    configs/stage3_training/pretrain_scratch_v1.json 1 cuda run001 --epochs 1

# Train on 4 GPUs (torchrun spawns one rank per GPU inside the container)
NGPU=4 docker/run.sh scripts/stage3_train_singlenode.sh \
    configs/stage3_training/pretrain_scratch_v1.json 4 cuda run001 --epochs 5

# Generate (Stage 1 → 2 → 3): see docker/README.md for the full three-command pipeline
docker/run.sh biom3_PenCL_inference --help
```

Single-node multi-GPU is supported via `torchrun`; multi-node (across instances) is a
documented follow-up — see [`docker/README.md`](../../docker/README.md).

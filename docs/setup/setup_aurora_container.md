# Running BioM3 on Aurora via Apptainer (Intel XPU container)

This is the containerized path for Aurora, as an alternative to the bare-metal
`module load frameworks` install in [setup_aurora.md](./setup_aurora.md). It uses
the Intel-XPU image ([docker/Dockerfile.xpu](../../docker/Dockerfile.xpu)),
converted to an Apptainer `.sif`.

**Scope: single node.** Multi-node oneCCL over Aurora's Slingshot/CXI fabric is
not wired up here yet — see [Multi-node](#multi-node-not-yet-supported).

## Why a separate image from the CUDA one

Aurora's GPUs are Intel Data Center GPU Max (Ponte Vecchio), driven by oneAPI /
Level-Zero — there is no CUDA. The `biom3:cuda` image would run on Aurora only on
CPU. The XPU image instead installs `torch==2.8.0+xpu` with the matching
`intel-extension-for-pytorch==2.8.10+xpu` — the newest public XPU pair, and one
the `addison-nm/lightning` fork requires, since its `XPUAccelerator` raises
without IPEX. Aurora's `module load frameworks` runs torch `2.10.0a0` with a
non-public IPEX `2.10.10`, which the container cannot reproduce (native
`torch.xpu`;
Intel's IPEX is upstreamed into mainline torch, and `torch.distributed` uses the
`xccl` backend). This mirrors how the CUDA image swaps in the cu129 wheel — see
[PyTorch on Aurora](https://docs.alcf.anl.gov/aurora/data-science/frameworks/pytorch/).

## Prerequisites

- An x86_64 host with Docker to build + push the image (any dev box; Aurora nodes
  have no Docker). Intel GPU torch wheels are x86_64-only.
- GHCR push access for the one-time publish (see [cloud/README.md](../../cloud/README.md)
  and [docker/push.sh](../../docker/push.sh)). The published image is public, so
  the Aurora-side pull needs no login.

## Workflow

### 1. Build + push the XPU image (off Aurora)

```bash
docker/build.sh --variant xpu --release       # -> ghcr.io/natural-machine/biom3:xpu-dev (+ :xpu-<sha>)
```

`--release` builds and pushes in one pass. Unlike the cuda variant it stays
**amd64-only**, so there is no manifest list to assemble. To publish an image you have
already built locally, `docker/push.sh --variant xpu` pushes the same two tags without
rebuilding.

### 2. Convert to a .sif (Aurora login node)

Apptainer ([Containers on Aurora](https://docs.alcf.anl.gov/aurora/containers/containers/))
converts the Docker image to a `.sif`. Point the cache/tmp at a roomy filesystem
if `$HOME` is tight — the unpack is several GB.

```bash
export APPTAINER_CACHEDIR=/flare/NLDesignProtein/$USER/.apptainer/cache
export APPTAINER_TMPDIR=/flare/NLDesignProtein/$USER/.apptainer/tmp
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"

apptainer build biom3_xpu.sif docker://ghcr.io/natural-machine/biom3:xpu-dev
```

Offline alternative (no registry): `docker save biom3:xpu -o biom3_xpu.tar` on the
build host, `scp` it over, then
`apptainer build biom3_xpu.sif docker-archive://biom3_xpu.tar`.

### 3. Smoke-test the GPUs (interactive, on a compute node)

Grab an interactive node, then check that torch sees the 12 tiles:

```bash
scripts/aurora/apptainer_run.sh python -c \
  "import torch; print('xpu', torch.xpu.is_available(), torch.xpu.device_count())"
# expect: xpu True 12
```

`apptainer_run.sh` binds
`/flare`, sets `ZE_FLAT_DEVICE_HIERARCHY=FLAT` (so each tile is its own device,
matching `num_devices=12` in the PBS templates), and sources `environment.sh`
inside the container — which fingerprints `/flare` to select the `aurora` profile
and apply the oneCCL/`xccl`/NUMEXPR settings.

### 4. Run a stage (single node)

```bash
BIOM3_WEIGHTS_DIR=/flare/NLDesignProtein/sharepoint/BioM3-data-share/weights \
BIOM3_DATA_DIR=/flare/NLDesignProtein/sharepoint/BioM3-data-share/data \
scripts/aurora/apptainer_run.sh scripts/stage3_train_singlenode.sh \
    configs/stage3_training/pretrain_scratch_v1.json 12 xpu run001 --epochs 1
```

Host dirs bind onto `/app/{weights,data,outputs}`; `outputs/` is writable, weights
and data are read-only. See the script header for all `BIOM3_*` knobs.

## Multi-node (not yet supported)

A self-contained image's bundled oneCCL cannot drive Aurora's CXI fabric on its
own. Enabling multi-node means binding the host libfabric/CXI + MPICH into the
container and launching `apptainer exec` under the host `mpiexec` with an
ABI-compatible MPICH — see [oneCCL on Aurora](https://docs.alcf.anl.gov/aurora/data-science/frameworks/oneCCL/).
Until that is added, use the bare-metal path ([setup_aurora.md](./setup_aurora.md))
for multi-node jobs.

## Troubleshooting

- **`torch.xpu.device_count()` is 0.** The container's Level-Zero GPU driver
  (`libze-intel-gpu1`, baked into the image) may not match Aurora's kernel driver.
  Fallback: bind the host runtime instead, e.g. add the host Level-Zero libs via
  `BIOM3_BIND_EXTRA=/usr/lib/x86_64-linux-gnu` (adjust to the actual host path) so
  the container uses Aurora's driver. Confirm the device nodes are visible with
  `clinfo` inside the container.
- **Dependency-conflict warning during build.** Expected — the `addison-nm/lightning`
  fork vs pyproject's pin — and harmless, same as the bare-metal Aurora install.
- **`OSError: [Errno 30] Read-only file system` (e.g. running the test suite).**
  The `.sif` is read-only, and some code writes into the image tree
  (`/app/tests/_tmp`, `.pytest_cache`). `apptainer_run.sh` passes
  `--writable-tmpfs` (an ephemeral RAM-backed overlay) to absorb these; if you
  invoke `apptainer exec` by hand, add `--writable-tmpfs` yourself.

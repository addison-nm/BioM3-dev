# Running BioM3 on Aurora via Apptainer (Intel XPU container)

This is the containerized path for Aurora, as an alternative to the bare-metal
`module load frameworks` install in [setup_aurora.md](./setup_aurora.md).

There are **two** Aurora images, and which one you want depends on node count:

- **Single node** — [docker/Dockerfile.xpu](../../docker/Dockerfile.xpu), run
  with [apptainer_run.sh](../../scripts/aurora/apptainer_run.sh). Validated: the
  full test suite passes and 12-tile training runs within ~8% of bare metal.
  Most of this document describes this path.
- **Multi-node** — [docker/Dockerfile.xpu-oneapi](../../docker/Dockerfile.xpu-oneapi),
  run with [apptainer_mpi_run.sh](../../scripts/aurora/apptainer_mpi_run.sh).
  Runs across nodes over Slingshot/CXI and scales — two nodes at roughly twice a
  single node's throughput — provided the host's Cray libfabric is bound in.
  See [Multi-node](#multi-node).

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

## Collectives: what works and what doesn't

oneCCL inside the container needs three things that bare metal gets for free.
All three are handled by [apptainer_run.sh](../../scripts/aurora/apptainer_run.sh);
they are recorded here because the failure modes are opaque.

| Need | Why | Failure if missing |
| --- | --- | --- |
| `libze_loader.so` symlink | oneCCL `dlopen`s the unversioned name, which only the `-dev` package ships. torch is unaffected — it links `.so.1` directly. | `could not open the library: libze_loader.so`, then `ze_data was not initialized` on every collective |
| `FI_PROVIDER=tcp` | A shell with `module load frameworks` exports `cxi,tcp;ofi_rxm`; apptainer forwards it, and the container's libfabric has no cxi provider. | `fi_getinfo error: ret -61, providers 0` → `failed to initialize ATL` |
| `CCL_PROCESS_LAUNCHER=torchrun` | The host sets `pmix`, but no PMIx server is reachable in the container. `torchrun` reads `LOCAL_RANK`/`LOCAL_WORLD_SIZE`. | `PMIx_Init failed: PMIX_ERR_UNREACH` → `local_idx >= 0 && local_idx < local_count failed` |

**Single node** works with the above. GPU-to-GPU transfers use Level-Zero IPC
rather than the fabric, so the tcp provider carries only out-of-band traffic.

## Multi-node

Multi-node works, over Slingshot/CXI, and scales: two nodes run roughly twice a
single node's throughput. It needs the host's Cray libfabric bound in — see
[CXI](#cxi) — without which it falls back to `tcp` and ends up *slower* than one
node.

It needs a different image and a different launcher from the single-node path:

| | Single node | Multi-node |
| --- | --- | --- |
| Image | [Dockerfile.xpu](../../docker/Dockerfile.xpu) | [Dockerfile.xpu-oneapi](../../docker/Dockerfile.xpu-oneapi) |
| Launcher | [apptainer_run.sh](../../scripts/aurora/apptainer_run.sh) — one container, torchrun spawns ranks | [apptainer_mpi_run.sh](../../scripts/aurora/apptainer_mpi_run.sh) — host mpiexec spawns one container per rank |
| Rank source | PALS env vars translated to `RANK`/`LOCAL_RANK` | `MPIEnvironment` via mpi4py (`BIOM3_RANK_SOURCE=mpi`) |

The oneapi image exists because the Ubuntu-based one cannot do this: its mpi4py
is built against OpenMPI, so under Aurora's Intel MPI it never bootstraps,
Lightning falls back to a local environment, and every rank reports global rank
0. `intel/oneapi-hpckit` supplies an Intel MPI that matches the host launcher.

```bash
# 2 nodes, 24 tiles. Do not `module load frameworks` — the container carries its
# own stack, and the module only exports host values the wrapper must override.
module load apptainer

NGPU_PER_NODE=12 NGPU_TOTAL=24 BIOM3_RANK_SOURCE=mpi \
BIOM3_FABRIC_DIR=/opt/cray/libfabric/1.22.0/lib64 BIOM3_FI_PROVIDER=cxi \
BIOM3_SIF=/flare/.../biom3_xpu-oneapi-<sha>.sif \
BIOM3_WEIGHTS_DIR=./weights BIOM3_DATA_DIR=./data \
scripts/aurora/apptainer_mpi_run.sh \
    biom3_train_stage3 --config_path configs/stage3_training/pretrain_scratch_v1.json \
    --device xpu --devices_per_node 12 --num_nodes 2 --run_id run001 --epochs 2
```

One additional setting this path needs, handled by the wrapper:
`CCL_ZE_IPC_EXCHANGE=sockets`. Each rank is its own container with its own PID
namespace, so oneCCL's default `pidfd` handle exchange is denied
(`pidfd_getfd failed: ... Operation not permitted`). Under `apptainer_run.sh`
every rank is a torchrun child of one container, so it never arises.

### Measured throughput

Stage 3 pretraining, `pretrain_scratch_v1.json`, 86.2M params, batch 32/rank:

| Config | it/s | samples/s | read at step |
| --- | --- | --- | --- |
| bare metal, 12 ranks, 1 node | 0.62 | ~238 | 16 |
| container, 12 ranks, 1 node | 0.57 | ~219 | 7 |
| container, 24 ranks, 2 nodes, tcp | 0.17 | ~131 | 334 |
| container, 24 ranks, 2 nodes, **cxi** | **0.64** | **~492** | 499 |

Single-node containers cost roughly 8% against bare metal. Two nodes over CXI is
3.8x the tcp figure and about twice a single node; the single-node container
number was read very early, so treat the scaling ratio as approximate until both
are measured at the same step.

### CXI

Aurora's CXI provider is in **HPE's Cray libfabric**, not the image and not
Intel MPI's bundled libfabric — that one ships efa/psm3/rxm/shm/tcp/verbs and no
cxi, under `/opt/aurora` as well as in the image. Point `BIOM3_FABRIC_DIR` at
the directory holding Cray's `libfabric.so.1`:

```
BIOM3_FABRIC_DIR=/opt/cray/libfabric/1.22.0/lib64 BIOM3_FI_PROVIDER=cxi
```

The wrapper then binds it at `/hostfabric`, prepends it to `LD_LIBRARY_PATH`,
`LD_PRELOAD`s it, and sets `I_MPI_OFI_LIBRARY_INTERNAL=0`. All four are needed:
`LD_PRELOAD` because pip's oneCCL ships its own libfabric at
`/opt/venv/lib/libfabric.so.1` and finds it through RPATH, which
`LD_LIBRARY_PATH` cannot override; `I_MPI_OFI_LIBRARY_INTERNAL=0` because Intel
MPI otherwise prefers its own. Cray's libfabric needs `libcxi.so.1`, which is in
`/usr/lib64` and so arrives via the existing `/hostevent` bind.

`CCL_ATL_TRANSPORT` must be `mpi` here, which is the wrapper's default. On `ofi`
oneCCL opens libfabric providers itself and fails on cxi with
`fi_getinfo error: ret -61, providers 0`, even with Cray's library loaded and
`fi_info -p cxi` listing every domain from inside the container — it requests
capabilities the provider will not grant. Riding the Intel MPI that already works
over cxi avoids the question. (ALCF's recipe sets `ofi` because its container has
no usable MPI; that constraint does not apply to this image.)

See [oneCCL on Aurora](https://docs.alcf.anl.gov/aurora/data-science/frameworks/oneCCL/)
and the container recipe in `_misc/sample_script.sh`.
Until then, use the bare-metal path ([setup_aurora.md](./setup_aurora.md)) for
multi-node jobs.

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

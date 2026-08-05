# Session: Aurora Containers — First Build, Single-Node Validation, Multi-Node over CXI

**Date:** 2026-08-04 → 2026-08-05
**Branch:** addison-dev
**Commit:** `331f8dc` — `docs: multi-node containers work over CXI and scale`
**Pre-session state:** `git checkout 070e35e` (`feat: multidomain data path over bioparsers JSONL + split manifest`)

## Goal

Establish whether we had ever published a container capable of running on Aurora (Intel XPU),
and if not, get one working. Requirement stated mid-session: **portability and
reproducibility** — one artifact per target, rebuilt on a BioM3 version bump — which ruled
out the alternative of a thin overlay on Aurora's `module load frameworks` stack.

## Summary

We had a `Dockerfile.xpu` that had **never been built**. By the end there are two working
Aurora images and two launchers:

| | Single node | Multi-node |
| --- | --- | --- |
| Image | `docker/Dockerfile.xpu` (Ubuntu + pip wheels, torch 2.8.0+xpu) | `docker/Dockerfile.xpu-oneapi` (`intel/oneapi-hpckit`, torch 2.10.0+xpu) |
| Launcher | `scripts/aurora/apptainer_run.sh` — one container, torchrun spawns ranks | `scripts/aurora/apptainer_mpi_run.sh` — host `mpiexec`, one container per rank |
| Rank source | PALS env vars translated to `RANK`/`LOCAL_RANK` | `MPIEnvironment` via mpi4py |
| Status | 1134 tests pass; 12 tiles ~8% off bare metal | 24 ranks / 2 nodes over CXI, ~2x a single node |

## Measured throughput

Stage 3 pretraining, `pretrain_scratch_v1.json`, 86.2M params, batch 32/rank:

| Config | it/s | samples/s | read at step |
| --- | --- | --- | --- |
| bare metal, 12 ranks, 1 node | 0.62 | ~238 | 16 |
| container, 12 ranks, 1 node | 0.57 | ~219 | 7 |
| container, 24 ranks, 2 nodes, tcp | 0.17 | ~131 | 334 |
| container, 24 ranks, 2 nodes, **cxi** | **0.64** | **~492** | 499 |

The single-node container figure was read very early, so the 2x scaling claim is
approximate. Re-measure both at ~step 500 before quoting it.

## What actually broke, in order

Every failure was one of three kinds. Recording them because the symptoms were consistently
opaque and pointed away from the cause.

### A. Image build — dependency version chains

1. **`pkg_resources` gone.** setuptools removed it in 82.0.0; the lightning fork's `setup.py`
   imports it. Pinned `setuptools<82`.
2. **`dpctl` and `tensorboardX` missing.** Supplied by `module load frameworks` on bare metal,
   so `requirements/aurora-container.txt` never listed them. `backend/xpu.py` imports `dpctl`
   at module scope, so *any* `import biom3` would have died on Aurora — and could not fail on
   Spark, where the CPU backend is selected and `xpu.py` never loads.
3. **pip silently swapping torch.** A transitive `torch` requirement resolved from PyPI, which
   serves CPU-only wheels, replacing `torch 2.8.0+xpu` and stranding `torchvision`.
   `requirements/constraints-xpu.txt` makes that fail loudly instead.
4. **The constraint then exposed the real conflict:** `dpctl 0.21.1` needs
   `intel-cmplr-lib-rt>=2025.3` while torch 2.8.0+xpu pins `==2025.1.1`. pip had been
   "resolving" this by upgrading torch. Pinned `dpctl==0.19.0` — 0.20+ also links
   `sycl::khr_get_default_context`, absent from the 2025.1.1 runtime.
5. **IPEX.** The fork's `XPUAccelerator` gates on `intel_extension_for_pytorch>=1.13`. Public
   IPEX stops at 2.8.10+xpu, which caps torch at 2.8 — the constraint that eventually forced
   the second image.

### B. Host state leaking into the container

The dominant failure class. Apptainer forwards the host environment, and on Aurora those
values are all wrong inside a container:

| Leak | Symptom |
| --- | --- |
| `--bind /dev/dri` (ours, not a leak) | remounts `/dev` `nodev`; `clinfo -l` empty, `torch.xpu.device_count()` 0 while the host saw 12 |
| `CCL_ROOT=/opt/aurora/...` | `failed to load file containing oneCCL SPIR-V kernels` |
| `FI_PROVIDER=cxi,tcp;ofi_rxm` | `fi_getinfo error: ret -61, providers 0` |
| `CCL_PROCESS_LAUNCHER=pmix` | `PMIx_Init failed: PMIX_ERR_UNREACH` |
| `LD_LIBRARY_PATH` set with `--env` | *replaces* the image's value; `ImportError: libsvml.so`, then `MPI_Init_thread` failing in `MPIDI_OFI_mpi_init_hook` |

`apptainer_run.sh` was itself the bug for the first of these: it bound `/dev/dri` explicitly
to "enable" the GPUs and thereby hid them. A plain `apptainer exec` with no `/dev` bind saw
all 12 tiles.

### C. Multi-node — one container per rank changes the rules

1. **mpi4py couldn't bootstrap.** The Ubuntu image ships OpenMPI; Aurora's launcher is Intel
   MPI. Lightning's `MPIEnvironment` asks mpi4py for the rank, detection failed, it fell back
   to `LightningEnvironment`, and all 24 ranks reported `global_rank 0` and tried to spawn
   their own children. This is why `Dockerfile.xpu-oneapi` exists.
2. **Two oneAPI versions.** Keeping torch 2.8 (deps: oneAPI 2025.1.1) on an oneAPI 2025.3 base
   put both in the image; the loader resolved per-library and one side always broke —
   `LIBUR_LOADER_0.11 not found` for torch, or `MPIR_F_MPI_BUFFER_AUTOMATIC` for mpi4py,
   depending on `LD_LIBRARY_PATH` order. No ordering satisfies both. Aligning everything on
   2025.3 (torch 2.10.0+xpu, dpctl 0.21.1, no IPEX) resolved it structurally.
3. **Lightning has two XPU accelerator copies.** `pytorch/accelerators/xpu.py` gates the
   accelerator; `fabric/accelerators/xpu.py` holds the `num_xpu_devices()` that
   `device_parser._sanitize_gpu_ids` consults. Patching only the first gave
   `You requested gpu: [0..11] But your machine only has: []` while
   `torch.xpu.device_count()` reported 12 in the same ranks.
4. **`pidfd_getfd` denied.** Each rank is its own container with its own PID namespace, so
   oneCCL's default IPC handle exchange has no ptrace access to peers.
   `CCL_ZE_IPC_EXCHANGE=sockets`.
5. **CXI.** The provider is in **HPE's Cray libfabric** (`/opt/cray/libfabric/1.22.0/lib64`),
   not Intel MPI's bundled one, which ships efa/psm3/rxm/shm/tcp/verbs and no cxi anywhere
   under `/opt/aurora`. Binding it needs `LD_PRELOAD` — pip's oneCCL ships its own libfabric
   and finds it through RPATH, which `LD_LIBRARY_PATH` cannot override — plus
   `I_MPI_OFI_LIBRARY_INTERNAL=0`.
6. **`CCL_ATL_TRANSPORT` must be `mpi`.** On `ofi`, oneCCL opens providers itself and fails on
   cxi with `providers 0` even with Cray's library confirmed loaded and `fi_info -p cxi`
   listing all eight domains in-container. ALCF's recipe sets `ofi` because its container has
   no usable MPI; ours does, so oneCCL should ride it.

## Decisions worth remembering

- **Two images, not one.** `Dockerfile.xpu` stays as the validated single-node path.
  `Dockerfile.xpu-oneapi` was added rather than mutating it, at the user's instruction, so a
  partially-working artifact was never put at risk.
- **The overlay option was rejected.** Binding Aurora's frameworks stack and shipping only
  `biom3` would have avoided most of section A and C, but it is Aurora-only, and the stated
  requirement was portability plus reproducibility.
- **Do not `module load frameworks` for container runs.** It only exports the host values the
  wrapper then has to override.
- **The Aurora-specific parts of frameworks are unobtainable** — torch `2.10.0a0+git449b176`,
  IPEX `2.10.10+gitd0f992f`, triton, torchvision and torchao are all ALCF source builds. What
  *is* reproducible is the oneAPI version underneath (2025.3), which is what the second image
  aligns to.

## Open items

- ~~`xpu-oneapi` has never been through the test suite~~ — both images pass
  (1134 passed / 162 skipped), `xpu-oneapi` confirmed at `6d1497e`. Notably it passes with
  torch 2.10, dpctl 0.21.1 and no IPEX, which is evidence that removing the fork's IPEX gate
  has no observable effect beyond letting the accelerator construct.
- Re-measure single-node container throughput at ~step 500 for an exact scaling ratio.
- `xpu-oneapi` image not rebuilt since `a8fe97f`; published `xpu-oneapi-6ed0e87` is
  functionally current (that commit removed a no-op layer and edited comments).
- `CCL_WORKER_AFFINITY` in `environment.sh` assumes the bare-metal 12-rank layout and warns at
  other rank counts.
- `ProcessGroupXCCL` warns per rank that the device mapping is unknown; passing `device_id` to
  `init_process_group` would silence it and remove a real hang risk.
- Commits from `db00a7f` onward were unpushed at session end.

## Process note

Most of the session was single-symptom-at-a-time debugging, and several hypotheses were
asserted without evidence and turned out wrong (the `pmix` hang attribution; "host env leakage
is ruled out"; the `LD_LIBRARY_PATH` override, dismissed on a test that could not have
detected it). What consistently worked was making the system explain itself —
`CCL_LOG_LEVEL=info` named the missing `libze_loader.so` symlink immediately, `/proc` state
showed 12 ranks spinning with idle GPUs, and `fi_info -p cxi` settled whether the provider was
usable at all. The reusable heuristic: when a container misbehaves on Aurora, suspect a host
value that is wrong inside it before suspecting the image.

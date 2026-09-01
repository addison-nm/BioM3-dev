# Session: Slimming the Docker images

**Date:** 2026-09-01
**Branch:** `dev`

The published `biom3:cuda` image was 17.9 GB on disk (~8 GB compressed pull).
It is now 8.63 GB, and a new CPU-only inference variant ships at 1.88 GB.

## What was wrong

Not duplicate PyTorch installs — **two complete CUDA stacks**.

`site-packages/torch/lib` is 4.9 GB and fully self-contained. The cu129 wheel
bundles its own `libcudart`, `libcublas`/`libcublasLt`, all of cuDNN (976 MB),
`libnccl` (410 MB), cuFFT, cuSPARSE, cuSPARSELt, cuSOLVER, cuRAND, nvrtc,
nvJitLink and cuPTI, and resolves them through its own `RUNPATH`
(`$ORIGIN:/usr/local/cuda/lib64`). `triton` likewise ships its own `ptxas`,
`cuobjdump` and `nvdisasm`.

The `nvidia/cuda:12.9.0-cudnn-devel-ubuntu24.04` base supplied a second copy of
every one of those, and almost none of it was opened at run time:

| Base layer | Size |
| ---------- | ---- |
| `cuda-libraries-12-9` (runtime `.so`) | 3.26 GB |
| dev packages (nvcc, nsight-compute, `*-dev`, 3.5 GB of static `.a`) | 5.12 GB |
| cuDNN + cuDNN-dev | 1.13 GB |
| cudart + cuda-compat | 238 MB |
| ubuntu rootfs + keyring | 112 MB |

Confirmed empirically: with `LD_LIBRARY_PATH` cleared in the old image, CUDA
matmul and cuDNN `conv2d` both still work, resolving entirely from `torch/lib`.
(`LD_LIBRARY_PATH=/usr/local/cuda/lib64` is set by the base image and is searched
*before* `RUNPATH`, which is why the old image was actually loading the base
image's cuBLAS/cuSPARSE and leaving torch's bundled copies as dead weight.)

Secondary waste: no multi-stage build, so `build-essential`, `python3.12-dev` and
`libopenmpi-dev` (429 MB) shipped in the final image even though `mpi4py` is the
only source build.

## What changed

**`docker/Dockerfile.cuda` — two stages.** Builder is plain `ubuntu:24.04`;
nothing in the build needs CUDA, since every CUDA library arrives inside a pip
wheel. Runtime is `nvidia/cuda:12.9.0-base-ubuntu24.04` plus `python3.12`,
`cuda-nvcc`, `openmpi-bin`, `libgomp1`, `git`/`curl`/`ca-certificates`, then
`COPY --from=builder` of `oras`, `/opt/venv` and `/app`.

**`cuda-nvcc` is required, not optional.** This was the one real trap. `import
deepspeed` shells out to `$CUDA_HOME/bin/nvcc -V` from `op_builder`'s
`is_compatible()` scan and raises `FileNotFoundError` without it. Since
`biom3.Stage3.PL_wrapper` imports deepspeed at module scope, the first slim build
broke *every* Stage 3 entry point — training and generation. `cuda-nvcc-12-9`
costs 553 MB against the 5.1 GB the full devel layer costs.

`INSTALL_BUILD_TOOLS=false` remains as an escape hatch for the genuinely optional
part: DeepSpeed JIT-compiles `cpu_adam` with `c++ -isystem /usr/local/cuda/include
-lcudart -lcublas -lcurand` (no nvcc) on first use of `DeepSpeedCPUAdam` or ZeRO
offload. No BioM3 config takes that path — every `configs/stage3_training/*.json`
uses `choose_optim: AdamW`, and `DeepSpeedStrategy` is constructed without offload.

**`docker/Dockerfile.cpu` + `requirements/container-cpu.txt` — new.** CPU torch
wheel on `ubuntu:24.04`, same two-stage shape. Scope is Stage 1/2 embedding, the
geometry manifold tools and Stage 3 sampling. Drops `mpi4py`, `nvidia-ml-py`,
`numba`/`llvmlite`, `optuna`, `diffusers`, `wordcloud`, `seaborn`, `tensorboard`,
`linformer`, `torchvision` (none of which `src/` imports) and the `app` extra.
Keeps `deepspeed` — pure Python here, under 20 MB, and required by
`Stage3.PL_wrapper`.

`wandb` and `py3Dmol` had to stay: `run_PL_training.py` and `biom3.viz.viewer`
import them at module scope, so without them 13 test modules fail to collect.
Only `streamlit` is genuinely absent, which costs exactly one test module.

**Tooling.** `build.sh`/`push.sh` accept `--variant cpu`. `run.sh` gained a `cpu`
device kind so it stops passing `--gpus` to a CPU image, and now forwards the
`BIOM3_WEIGHTS_BUNDLE` / `BIOM3_WEIGHTS_BUNDLE_REPO` / `GHCR_TOKEN` / `GHCR_USER`
vars that `entrypoint.sh` documents but `run.sh` was silently dropping.

## What is verified

| Image | Size |
| ----- | ---- |
| `ghcr.io/natural-machine/biom3:cuda-dev` (previous) | 17.90 GB |
| `biom3:cuda` (this work) | 8.63 GB |
| `biom3:cpu` (new) | 1.88 GB |

- **cuda, on the GB10:** matmul, cuDNN `conv2d`, `is_nccl_available()`, imports of
  `deepspeed` / `mpi4py` / `streamlit` / `pytorch_lightning`, all Stage 3 and
  pipeline modules, and `oras` + `aws` + `mpiexec` + `nvcc` on PATH.
  `pytest tests/ --quick`: **1267 passed, 212 skipped**.
- **cpu:** `torch 2.8.0+cpu`, all six target entry points respond to `--help`.
  `pytest tests/ --quick --ignore=tests/viz_tests/test_data_browser.py`:
  **1236 passed, 238 skipped**.
- `pip list` diff between the old and new cuda images shows only unpinned-
  dependency version drift (streamlit and awscli transitives). The restructure
  dropped no package.

## Follow-ups (not done)

- **`requirements/cpu.txt` is byte-identical to `requirements/spark.txt`**, so the
  CPU install path documented in `docs/setup/setup_cpu.md` pulls `mpi4py` and
  `nvidia-ml-py`. Worth collapsing.
- **`requirements/spark.txt` carries ~300 MB nothing in `src/` imports**
  (`numba`+`llvmlite`, `optuna`, `diffusers`, `wordcloud`, `seaborn`,
  `tensorboard`, `linformer`, `torchvision`). Pruning changes what the cuda image
  provides, so it belongs in its own pass.
- **The XPU images have the same duplication.** `biom3:xpu-oneapi` is 22.3 GB and
  `biom3:xpu` is 10.6 GB; neither is multi-stage. Deferred deliberately.

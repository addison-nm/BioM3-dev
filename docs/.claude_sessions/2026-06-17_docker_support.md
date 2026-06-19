# 2026-06-17 — Docker support for AWS / Mithril GPU cloud

## Context

Added first-class Docker support so BioM3 — training (all stages), finetuning, and
generation — can run in containers on commercial GPU cloud (AWS GPU instances, Mithril),
e.g. spot instances. ALCF HPC (Polaris/Aurora) is explicitly out of scope: those use
apptainer and a different software stack.

Prior art existed in the sibling repo `nm-portal/docker/biom3/` (`spark.Dockerfile`,
`entrypoint.sh`, `job_run.py`) — a solid CUDA image recipe, but inference-only and tightly
coupled to nm-portal's AWS infra (S3 weight/config sync, SQS-dispatched `BIOM3_JOB_SPEC`,
Roles Anywhere creds), targeting the ARM64 DGX Spark. We wanted a general, repo-owned
image that supports training.

Work done on branch `feat-docker` (off `addison-dev`).

## What was built

A self-contained `docker/` plus two small source-tree edits that let the existing
training scripts run unchanged in-container.

**New `docker/`:**
- `Dockerfile` — one arch-portable CUDA image (`nvidia/cuda:12.9.0-cudnn-devel-ubuntu24.04`
  → py3.12 venv → `torch==2.8.0 torchvision` cu129 → `requirements/spark.txt` →
  `pip install -e .[app]`). Build context = repo root; explicit COPYs (never `COPY . .`)
  so weights/data/outputs are never baked in; `configs/` IS baked in. `ENV
  BIOM3_MACHINE=container`, `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`. `ARG INSTALL_AWSCLI=false`.
- `Dockerfile.dockerignore`, `entrypoint.sh`, `build.sh`, `run.sh`, `docker-compose.yml`,
  `README.md` (full reference: build, run recipes, sync, app, multi-node-as-followup).

**Source edits (minimal):**
- `scripts/launchers/container_singlenode.sh` (new) — `NGPU=1` → `exec`; `NGPU>1` →
  `torchrun --standalone --nnodes=1 --nproc-per-node=N`. Resolves the console-script entry
  (`biom3_train_stageN`) to its file path via `command -v` because torchrun launches a
  *script file*, not a PATH name.
- `environment.sh` — honor a preset `BIOM3_MACHINE` (auto-detect only when unset, so the
  image's `ENV` wins); add `/.dockerenv` → `container` fallback (after the `/flare`,`/grand`
  HPC checks); add a minimal `container` settings branch.
- `README.md` — one-line pointer; `docs/setup/setup_docker.md` — concise setup guide.

## Key decisions (and why)

- **One image, all uses** (not per-entrypoint). Training, finetuning, and generation share
  the same install (finetuning is literally `biom3_train_stage3 --finetune True`); the only
  dep delta is the streamlit `app` extra, which we fold in. Per-entrypoint images would be
  ~4 near-identical multi-GB artifacts. User-confirmed.
- **One Dockerfile for x86_64 + ARM64**, no arch conditionals. The cuda base is a multi-arch
  manifest and the cu129 torch index serves both x86_64 and aarch64 wheels (confirmed:
  `torch-2.8.0+cu129-...-aarch64.whl` downloaded during the arm64 build). Arch is chosen
  solely by `buildx --platform`. User-confirmed.
- **`torchrun`, not `mpiexec`, inside the container.** Torch-native, no PBS/MPI/CPU-binding
  machinery; the codebase already reads `RANK`/`LOCAL_RANK`/`WORLD_SIZE`
  (`core/_dist_env.py`) and PL auto-detects the torchelastic env (no re-spawn). The training
  path does not call `init_distributed_if_launched` — PL's strategy initializes the process
  group from torchrun's env vars. OpenMPI is kept in the image only so `mpi4py` (a pinned
  dep) builds; it is not used to launch.
- **Bind-mounts by default + optional object-store sync.** Base image is cloud-agnostic
  (no awscli unless `--build-arg INSTALL_AWSCLI=true`); the entrypoint sync hook is a no-op
  unless `BIOM3_*_URI` vars are set, and is tool-agnostic via `BIOM3_SYNC_CMD`. Host-side
  staging documented as the recommended spot pattern. User-confirmed.
- **Single-node multi-GPU only** for v1; multi-node documented as a follow-up. User-confirmed.

## Verification (local, macOS arm64 — no NVIDIA GPU)

- Image builds natively for arm64 (`docker/build.sh`), 28.1 GB. `BIOM3_MACHINE=container`
  honored; `import biom3` + `torch 2.8.0+cu129` OK; `cuda? False` (expected).
- `pytest tests/ --quick` in-container: **680 passed, 228 skipped, 0 failed**.
- Launcher dry-run: `torchrun` on PATH, all 7 entrypoints resolve, `NGPU=1` dispatch execs
  directly, `container_multinode.sh` correctly absent.
- Entrypoint sends the `environment.sh` banner to stderr → command stdout stays clean.

## Deferred to a real GPU instance (documented in docker/README.md)

- Single- and multi-GPU training via `torchrun` (the one fact needing real hardware:
  N ranks form the PL world correctly).
- End-to-end GPU generation (Stage 1→2→3) and finetune/resume.

## Follow-ups / notes

- Image is 28.1 GB (the `-devel` cuda base is heavy). Could slim with a `-runtime` base +
  multi-stage build later if pull time on spot becomes a concern; `-devel` is currently
  needed so `mpi4py`/`deepspeed` build.
- Multi-node across instances (cross-host rendezvous) is the natural next step.
- Changes are uncommitted on `feat-docker` pending review.

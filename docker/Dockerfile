# BioM3 — NVIDIA GPU image for commercial cloud (AWS / Mithril).
#
# One image, all uses: training (all stages), finetuning, generation, and
# the streamlit app. The command is chosen at `docker run` time.
#
# Arch-portable: build for x86_64 (AWS p3/p4/p5, g4/g5/g6; Mithril H100/H200)
# or ARM64 (Grace / GH200) from THIS SAME Dockerfile — there are no arch
# conditionals. The nvidia/cuda base is a multi-arch manifest and the cu129
# torch index serves both x86_64 and aarch64 wheels; pip selects by platform.
# Choose the arch with `docker buildx build --platform linux/{amd64,arm64}`
# (see docker/build.sh).
#
# Build context is the BioM3-dev repo ROOT (so source + requirements are in
# scope). Build with docker/build.sh, or:
#   docker buildx build -f docker/Dockerfile -t biom3:gpu --load .
#
# ALCF HPC (Polaris/Aurora) is NOT a target here — those use apptainer and a
# different software stack (see docs/setup/setup_{polaris,aurora}.md).

FROM nvidia/cuda:12.9.0-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
    BIOM3_MACHINE=container

# System deps:
#  - python3.12: matches BioM3-dev's required Python (Ubuntu 24.04 ships 3.12).
#  - build-essential, python3.12-dev: source builds for deps without wheels.
#  - libopenmpi-dev, openmpi-bin: kept ONLY so mpi4py (in requirements) builds
#    and imports. Multi-GPU launching uses torchrun, not mpiexec.
#  - git, ca-certificates: pip may resolve git+https URLs from dependencies.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-dev python3.12-venv \
        build-essential git ca-certificates \
        libopenmpi-dev openmpi-bin \
    && rm -rf /var/lib/apt/lists/*

ENV VIRTUAL_ENV=/opt/venv
RUN python3.12 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN pip install --upgrade pip setuptools wheel

# Torch with the cu129 wheel — must come before requirements so the pinned
# torch isn't replaced by a CPU wheel during dependency resolution. pip picks
# the x86_64 or aarch64 wheel automatically based on --platform.
RUN pip install torch==2.8.0 torchvision \
        --index-url https://download.pytorch.org/whl/cu129

# Requirements layer is cached independently of the source. requirements/spark.txt
# is the general post-torch pinned set (numpy/transformers/deepspeed/etc.); the
# "spark" name is historical — it is not DGX-Spark-specific.
WORKDIR /app
COPY requirements/ /app/requirements/
RUN pip install -r requirements/spark.txt

# Optional: bake awscli in for the entrypoint's S3 sync hook. Off by default to
# keep the image cloud-agnostic — the default data path is bind-mounts (no cloud
# tools needed). Enable with `--build-arg INSTALL_AWSCLI=true`.
ARG INSTALL_AWSCLI=false
RUN if [ "$INSTALL_AWSCLI" = "true" ]; then pip install awscli; fi

# BioM3 source. Explicit paths (no `COPY . .`) so we never bake weights/, data/,
# outputs/, logs/, or jobs/ into the image. configs/ IS baked in: it is tiny,
# repo-versioned, and matches the code revision (unlike weights/data, which are
# large and mounted/synced at runtime). Add new BioM3 top-level dirs here as
# they are introduced.
COPY pyproject.toml environment.sh README.md /app/
COPY src/      /app/src/
COPY scripts/  /app/scripts/
COPY tests/    /app/tests/
COPY configs/  /app/configs/

# Editable install with the `app` extra so the same image can also serve
# `biom3_app` (streamlit). Create the runtime mount points (empty in the image;
# bind-mounted or synced at run time).
RUN pip install -e ".[app]" \
    && mkdir -p /app/weights /app/data /app/outputs /app/logs

# Entrypoint from the repo's docker/ directory.
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["bash"]

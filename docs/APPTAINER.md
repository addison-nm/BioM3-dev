# Notes for using apptainer

## Polaris instructions

Refer to [ALCF's Polaris docs](https://docs.alcf.anl.gov/polaris/containers/containers/).

Start by requesting an interactive node on Polaris. Once allocated, `cd` to a project directory and execute the following commands to load apptainer and run `biom3` commands from within the container.

Load the apptainer module for use on Polaris:

```bash
module use /soft/modulefiles
module load spack-pe-base
module load apptainer
```

Set up some basic folders and internet access tooling.

```bash
export BASE_SCRATCH_DIR=/local/scratch/ # For Polaris
export APPTAINER_TMPDIR=$BASE_SCRATCH_DIR/apptainer-tmpdir
mkdir -p $APPTAINER_TMPDIR

export APPTAINER_CACHEDIR=$BASE_SCRATCH_DIR/apptainer-cachedir
mkdir -p $APPTAINER_CACHEDIR

# For internet access
export HTTP_PROXY=http://proxy.alcf.anl.gov:3128
export HTTPS_PROXY=http://proxy.alcf.anl.gov:3128
export http_proxy=http://proxy.alcf.anl.gov:3128
export https_proxy=http://proxy.alcf.anl.gov:3128

apptainer version # should return 1.4.1
```

Build the container. The `--fakeroot` option is needed to emulate sudo permissions. This creates a `biom3.sif` object in your current directory. It builds it from the image that is hosted at the subsequent URL. Below, the version tag is `cuda-dev`. We will plan on creating tagged images as we release new versions. The `.sif` object is the container that allows us to run biom3 commands without cloning the repo. The container should be built from a directory in your project space within the `/grand` filesystem, not under your home directory.

```bash
# Only needed once, to build the container
biom3_image=/path/to/biom3.sif  # <-- Specify this
apptainer build --fakeroot ${biom3_image} docker://ghcr.io/natural-machine/biom3:cuda-dev
```

Run the BioM3 test suite from within the container. The test suite needs no
repo checkout, no bind mounts, and no folders created. It reads its data from
the baked-in `/app/tests/_data` and writes only to `tests/_tmp` and
`.pytest_cache`. Two flags are necessary:

- `--writable-tmpfs` gives the read-only image an ephemeral, RAM-backed overlay
  so those test writes succeed (nothing persists after the run).
- `--pwd /app` runs from the image's baked-in source tree, so `tests/` resolves.

```bash
# Point to the built container
biom3_image=/path/to/biom3.sif  # <-- Make sure this matches as above
biom3_image=/grand/NLDesignProtein/ahowe/biom3.sif  # TODO: remove this line

# Quick test suite (CPU only, ~3 min)
apptainer exec --writable-tmpfs --pwd /app ${biom3_image} python -m pytest tests/ --quick

# Full test suite — add --nv so the GPU-marked tests actually run (they skip without it)
apptainer exec --nv --writable-tmpfs --pwd /app ${biom3_image} python -m pytest tests/
```

### Running real biom3 commands

Real runs need model weights. On Polaris, **do not point at the shared
`BioM3-data-share` weights** — use the versioned **weights bundle** published to
GHCR (`ghcr.io/natural-machine/biom3-weights`). Like the container, it pulls
standalone with no checkout, and every tag is an immutable sha, so a run is
reproducible and self-contained. Full detail:
[setup/weights_bundle.md](setup/weights_bundle.md).

Provision the weights and an outputs dir. `oras` is a single static binary
([install](https://oras.land/docs/installation)); the proxy exports set up above
give it internet on a compute node.

```bash
# Pull the Run 1 base weights + their configs (list tags: oras repo tags ghcr.io/natural-machine/biom3-weights)
oras pull ghcr.io/natural-machine/biom3-weights:run1_base -o $PWD/biom3-weights-run1_base
bundle=$PWD/biom3-weights-run1_base

# The one folder you must create — real outputs persist here, outside the overlay
mkdir -p $PWD/outputs
```

Bind the bundle's `weights/` over the image's empty `/app/weights` — that's the
only bundle bind you need. run1_base's architecture is already the default baked
into the image's `configs/inference/` configs, so you point `--config_path` at
those and never bind the bundle's configs. A small helper keeps the three stages
tidy — with `--pwd /app`, no shell wrapper is needed:

```bash
run() {
    apptainer exec --nv --writable-tmpfs --pwd /app \
        --bind "$PWD/outputs:/app/outputs" \
        --bind "$bundle/weights:/app/weights:ro" \
        ${biom3_image} "$@"
}

# Stage 1 -> 2 -> 3. `--input_data_path None` uses Stage 1's built-in 5-protein
# test set, so the whole chain runs with no external data.
run biom3_PenCL_inference --input_data_path None \
    --config_path configs/inference/stage1_PenCL.json \
    --model_path weights/PenCL/BioM3_PenCL_run1_base.bin \
    --output_path outputs/pencl_embeddings.pt

run biom3_Facilitator_sample --input_data_path outputs/pencl_embeddings.pt \
    --config_path configs/inference/stage2_Facilitator.json \
    --model_path weights/Facilitator/BioM3_Facilitator_run1_base.bin \
    --output_data_path outputs/facilitator_embeddings.pt

run biom3_ProteoScribe_sample --input_path outputs/facilitator_embeddings.pt \
    --config_path configs/inference/stage3_ProteoScribe_sample.json \
    --model_path weights/ProteoScribe/BioM3_ProteoScribe_run1_base.bin \
    --output_path outputs/generated_sequences.csv
```

Paths like `configs/...`, `weights/...`, `outputs/...` are relative to `/app`
(`--pwd /app`): the config comes from the baked-in image, the weights from the
bundle you bound. Generated sequences land in `$PWD/outputs/generated_sequences.csv`.

> **Multi-GPU training is different.** The single-node training launcher spawns
> ranks with Cray PALS `mpiexec`, which must run *outside* the container (the
> image's bundled OpenMPI can't drive Polaris's launcher). The single-process
> commands above cover inference/generation; multi-rank training needs the
> `mpiexec ... apptainer exec ...` form and is left as a follow-up.


## Aurora instructions

Refer to [ALCF's Aurora docs](https://docs.alcf.anl.gov/aurora/containers/containers/).

An analogous set of instructions should work on Aurora. The differences include 1) that we need to specify an xpu build instead of cuda, since Aurora has Intel machines, and 2) we load apptainer and execute the build in a slightly different manner.

Load the apptainer module.

```bash
module load apptainer
```

Build the image (one time).

```bash
apptainer build --fakeroot biom3.sif docker://ghcr.io/natural-machine/biom3:cuda-dev
```


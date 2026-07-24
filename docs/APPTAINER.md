# Notes for using apptainer

## Polaris instructions

Start by requesting an interactive node on Polaris. Once allocated, `cd` to a project directory and execute the following commands to load apptainer and run `biom3` commands from within the container.

Load the apptainer module for use on Polaris:

```bash
module use /soft/modulefiles
module load spack-pe-base
module load apptainer
```

Build the container. The `--fakeroot` option is needed to emulate sudo permissions. This creates a `biom3.sif` object in your current directory. It builds it from the image that is hosted at the subsequent URL. Below, the version tag is `cuda-dev`. We will plan on creating tagged images as we release new versions. The `.sif` object is the container that allows us to run biom3 commands without cloning the repo.

```bash
# Only needed once, to build the container
apptainer build --fakeroot biom3.sif docker://ghcr.io/natural-machine/biom3:cuda-dev
```

Run the BioM3 test suite from within the container.
The --writable-tmpfs option is necessary to permit writing to the tests/_tmp directory, an operation that is carried
out in a number of tests. The app/tests/ directory is read only, but this option enables some sort of ephemeral write
permissions.

```bash
# Quick test suite
apptainer exec --nv --writable-tmpfs \
    --bind /grand \
    --bind "$PWD/outputs:/app/outputs" \
    /grand/NLDesignProtein/ahowe/BioM3-dev/biom3.sif \
    bash -lc 'cd /app && unset BIOM3_MACHINE && source environment.sh && exec "$@"' _ \
    python -m pytest tests/ --quick

# Full test suite
apptainer exec --nv --writable-tmpfs \
    --bind /grand \
    --bind "$PWD/outputs:/app/outputs" \
    /grand/NLDesignProtein/ahowe/BioM3-dev/biom3.sif \
    bash -lc 'cd /app && unset BIOM3_MACHINE && source environment.sh && exec "$@"' _ \
    python -m pytest tests/
```

What about something other than a test? Can we run arbitrary `biom3` commands? Specifically, can we perform finetuning and generate sequences?

```bash
# TODO: Figure out what command does this...
```


## Aurora instructions

An analogous set of instructions should work on Aurora. The differences include 1) that we need to specify an xpu build instead of cuda, since Aurora has Intel machines, and 2) we load apptainer and execute the build in a slightly different manner.

```bash
module load /soft/modulefiles
module spack-pe-base
module apptainer
```
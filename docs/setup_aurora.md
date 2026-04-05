# Installation and setup instructions for Aurora

The Aurora cluster provides access to Intel GPUs, so a different installation is required. Like Polaris, there is a prebuilt environment that we can extend using a virtual environment.

In order to run on Intel GPUs, we have modified a custom `lightning` package, which must be included in the installation. This [lightning source code](https://github.com/addison-nm/lightning) is available on GitHub, if one needs to edit this directly. The commands below install this version of lightning, originally forked from the ALCF lightning repo.

Create the environment:

```bash
env_name="biom3-env"
module load frameworks
cd /path/to/BioM3-dev
# Create environment, using packages from prebuilt one
python -m venv venvs/${env_name} --system-site-packages
source "venvs/${env_name}/bin/activate"
# Install custom lightning package
python -m pip install git+https://github.com/addison-nm/lightning.git --no-build-isolation
# Install BioM3
python -m pip install -e .
# Install additional dependencies
python -m pip install -r requirements/aurora.txt
```

Note that presently an error message may be raised due to package conflicts, but the installation should still work.

## Usage

Load the Aurora frameworks module, activate the environment, and source `environment.sh` at the start of each session. The script auto-detects Aurora and sets additional variables required for Intel GPUs (`NUMEXPR_MAX_THREADS`, `ONEAPI_DEVICE_SELECTOR`).

```bash
cd /path/to/BioM3-dev
module load frameworks
source venvs/biom3-env/bin/activate
source environment.sh
```

### Running tests

```bash
python -m pytest tests/test_imports.py           # Quick import check
python -m pytest tests                           # Full suite (may take some time)
python -m pytest tests -rs                       # Full suite, report skipped tests
```

Some tests will be skipped if pretrained weights have not been synced. See [setup_shared_weights.md](./setup_shared_weights.md) for the list of required files.

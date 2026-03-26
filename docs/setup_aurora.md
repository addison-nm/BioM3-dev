# Installation and setup instructions for Aurora

The Aurora cluster provides access to Intel GPUs, so a different installation is required. Like Polaris, there is a prebuilt conda environment that we can extend using a virtual environment.
In order to run on Intel GPUs, we have modified a custom `lightning` package, which must be included in the installation. Download the zip file and unzip it in the `BioM3-dev` directory.

```bash
cd /path/to/BioM3-dev
# Copy the custom lightning source code into your BioM3-dev directory
cp /flare/NLDesignProtein/lightning.tar
# Extract the tar file
tar -xf lightning.tar
```

Now, create the conda environment.

```bash
ENV_NAME="biom3-env"
module load frameworks
cd /path/to/BioM3-dev
# Create environment, using packages from prebuilt one
python -m venv venvs/${ENV_NAME} --system-site-packages
source "venvs/${ENV_NAME}/bin/activate"
# Install custom lightning package
python -m pip install -e ./lightning --no-build-isolation
# Install BioM3
python -m pip install -e .
# Install additional dependencies
python -m pip install -r requirements_aurora.txt
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

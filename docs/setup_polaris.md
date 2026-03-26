# Installation and setup instructions for Polaris

Polaris provides access to NVIDIA GPUs. The prebuilt conda environment on Polaris is theoretically equipped with up to date versions of ML packages optimized to run on the cluster. In order to take advantage of this optimization, while also installing additional requirements, we will create a virtual environment extending the existing environment. Note the use of pip instead of conda.

```bash
env_name="biom3-env"
module use /soft/modulefiles; module load conda; conda activate base
cd /path/to/BioM3-dev
mkdir -p venvs
# Create environment, using packages from prebuilt one
python -m venv venvs/${env_name} --system-site-packages
source "venvs/${env_name}/bin/activate"
python -m pip install -r requirements_polaris.txt --ignore-installed
python -m pip install -e .
```

Note that presently an error message may be raised due to package conflicts, but the installation should still work.

## Usage

Load the Polaris modules, activate the environment, and source `environment.sh` at the start of each session. The script auto-detects the machine and applies the appropriate settings.

```bash
cd /path/to/BioM3-dev
module use /soft/modulefiles
module load conda
conda activate base
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

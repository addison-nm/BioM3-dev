# Installation and setup instructions for MacOS (CPU only, no GPU support)

Note that for MPI imports to succeed, mpi4py must be installed locally via brew in addition to the installation included in the pip install command (via requirements) below.

```bash
# If mpi4py is not available already, run the following:
# brew install mpi4py

env_name="biom3-env-py312"
cd /path/to/BioM3-dev
mkdir -p venvs
conda create -p venvs/${env_name} python=3.12
conda activate venvs/${env_name}
python -m pip install torch==2.8 torchvision
python -m pip install -r requirements/cpu.txt
python -m pip install -e .
```

Verify the installation by running a small set of import tests.

```bash
python -m pytest tests/test_imports.py
```

## Usage

Source `environment.sh` at the start of each session to set required environment variables. The script auto-detects the machine and applies the appropriate settings.

```bash
cd /path/to/BioM3-dev
conda activate venvs/biom3-env-py312
source environment.sh
```

### Running tests

```bash
python -m pytest tests/test_imports.py           # Quick import check
python -m pytest tests                           # Full suite (may take some time)
python -m pytest tests -rs                       # Full suite, report skipped tests
```

Some tests will be skipped if pretrained weights have not been synced. See [setup_shared_weights.md](./setup_shared_weights.md) for the list of required files.

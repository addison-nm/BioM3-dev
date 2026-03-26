# Installation and setup instructions for DGX Spark

The DGX Spark has a single NVIDIA GPU. The following commands should allow one to setup a working conda environment.

```bash
ENV_NAME="biom3-env-py312"
cd /path/to/BioM3-dev
mkdir -p venvs
conda create -p venvs/${ENV_NAME} python=3.12
conda activate venvs/${ENV_NAME}
python -m pip install torch==2.8 torchvision --index-url https://download.pytorch.org/whl/cu129
python -m pip install -r requirements_spark_py312.txt
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

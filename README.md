# BioM3 development

## About

## Setup

### Polaris

Clone the repository:

```bash 
git clone https://github.com/addison-nm/BioM3-dev.git
cd BioM3-dev
```

Create the environment, copying packages from custom base environment on Polaris.

```bash
# Create environment, copying packages from custom base
cd /path/to/BioM3-dev
module use /soft/modulefiles; module load conda; conda activate base
CONDA_NAME="biom3-env"
VENV_DIR="$(pwd)/venvs/${CONDA_NAME}"
mkdir -p "${VENV_DIR}"
python -m venv "${VENV_DIR}" --system-site-packages
source "${VENV_DIR}/bin/activate"
```

Install dependencies specified in `requirements_polaris.txt`, and project source code. Note that presently an error message may be raised due to package conflicts, but the installation should still work. 

```bash
# Install dependencies
python -m pip install -r requirements_polaris.txt --ignore-installed
```

Install source code:

```bash
python -m pip install -e .
```

To activate (on compute node, with access to GPU):

```bash
cd /path/to/BioM3-dev
module use /soft/modulefiles
module load conda
source venvs/biom3-env/bin/activate
```

Basic verification. **Note: might only succeed on compute node.**

```bash
python -c 'import torch; import pytorch_lightning as pl; import deepspeed; import Stage3_source.preprocess as prep'
```

Run tests with

```bash
python -m pytest tests
```

## Usage

## References


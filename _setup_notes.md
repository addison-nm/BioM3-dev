# Setup Notes

## Polaris 

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

Install source code with dependencies (specified in pyproject.toml).

```bash
# Install source code (with dependencies)
python -m pip install -e .

# The following are included in the pyproject.toml file
#python -m pip install pytorch-lightning==2.5.1.post0
#python -m pip install axial-positional-embedding==0.3.12
#python -m pip install linear-attention-transformer==0.19.1
#python -m pip install biopython==1.85
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

## Aurora

TODO...

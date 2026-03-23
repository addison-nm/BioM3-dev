# Installation and setup instructions for Polaris

Polaris provides access to NVIDIA GPUs. The prebuilt conda environment on Polaris is theoretically equipped with up to date versions of ML packages optimized to run on the cluster. In order to take advantage of this optimization, while also installing additional requirements, we will create a virtual environment extending the existing environment. Note the use of pip instead of conda.

```bash
ENV_NAME="biom3-env"
module use /soft/modulefiles; module load conda; conda activate base
cd /path/to/BioM3-dev
mkdir -p venvs
# Create environment, using packages from prebuilt one
python -m venv venvs/${ENV_NAME} --system-site-packages
source "venvs/${ENV_NAME}/bin/activate"
python -m pip install -r requirements_polaris.txt --ignore-installed
python -m pip install -e .
```

Note that presently an error message may be raised due to package conflicts, but the installation should still work. 

Note that in order to load pretrained weights, it may be necessary to set the environment variable `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` . For convenience, include the following line in a file `environment.sh` and source it before running tests or scripts.

```bash
echo "export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1" >> environment.sh
source environment.sh
python -m pytest tests  # Runs all tests. Warning: may take some time
```

Test the setup with a small set of tests:

```bash
cd /path/to/BioM3-dev
module use /soft/modulefiles
module load conda
source venvs/biom3-env/bin/activate

source environment.sh
python -m pytest tests/test_imports.py
# python -m pytest tests  # Runs all tests. Warning: may take some time
```

Note that some tests will be skipped if the necessary weights have not been downloaded. Running with flags `-rs` should report these issues.

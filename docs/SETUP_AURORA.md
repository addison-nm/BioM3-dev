# Installation and setup instructions for Aurora

The Aurora cluster provides access to Intel GPUs, so a different installation is required. Like Polaris, there is a prebuilt conda environment that we can extend using a virtual environment.
In order to run on Intel GPUs, we have modified a custom `lightning` package, which must be included in the installation. Download the zip file and unzip it in the `BioM3-dev` directory.

```bash
cd /path/to/BioM3-dev
cp /flare/NLDesignProtein/lightning.zip .
unzip lightning.zip
```

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

Note that in order to load pretrained weights, it may be necessary to set the environment variable `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` . For convenience, include the following line in a file `environment.sh` and source it before running tests or scripts.

```bash
echo "export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1" >> environment.sh
source environment.sh
python -m pytest tests  # Runs all tests. Warning: may take some time
```

Test the setup with a small set of tests:

```bash
cd /path/to/BioM3-dev
module load frameworks
source venvs/biom3-env/bin/activate

source environment.sh
python -m pytest tests/test_imports.py
# python -m pytest tests  # Runs all tests. Warning: may take some time
```

Note that some tests will be skipped if the necessary weights have not been downloaded. Running with flags `-rs` should report these issues.

# Installation and setup instructions for Aurora

The Aurora cluster provides access to Intel GPUs, so a different installation is required. Like Polaris, there is a prebuilt conda environment that we can extend using a virtual environment.
In order to run on Intel GPUs, we have modified a custom `lightning` package, which must be included in the installation. Download the zip file and unzip it in the `BioM3-dev` directory.

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

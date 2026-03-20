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

The full suite of tests may take some time to run. Note that in order to load pretrained weights, it may be necessary to set the environment variable `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` . For convenience, include the following line in a file `environment.sh` and source it before running tests or scripts.

```bash
echo "export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1" >> environment.sh
source environment.sh
python -m pytest tests  # Runs all tests. Warning: may take some time
```

Note that some tests will be skipped if the necessary weights have not been downloaded. Running with flags `-rs` should report these issues.

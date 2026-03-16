# BioM3 development

## About

## Installation and setup

The `BioM3-dev` repo is available to clone from GitHub.

```bash
git clone https://github.com/addison-nm/BioM3-dev.git && cd BioM3-dev
```

**Note:** *In order to run the complete suite of tests, one needs to download pretrained weights.
Instructions to do so are included in the README located in the `weights` directory.*

The following sections detail installation procedures on different machines.

### DGX Spark

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

The full suite of tests may take some time to run. Note that in order to load pretrained weights, it may be necessary to set the environment variable `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` . For convenience, include the following line in a file environment.sh and source it before running tests or scripts.

```bash
echo "export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1" >> environment.sh
source environment.sh
python -m pytest tests  # Runs all tests. Warning: may take some time
```

### Polaris

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
Again, test the setup with a small set of tests:

```bash
cd /path/to/BioM3-dev
module use /soft/modulefiles
module load conda
source venvs/biom3-env/bin/activate

python -m pytest tests/test_imports.py
# python -m pytest tests  # Runs all tests. Warning: may take some time
```

Note that some tests will be skipped if the necessary weights have not been downloaded. Running with flags `-rs` should report these issues.

<!-- ```
conda create -p venvs/env-orig python=3.10
conda activate venvs/env-orig
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements_orig.txt
python -m pip install -e .
``` -->

### Aurora

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

## Usage

After the pip installation, a number of entrypoints should be available from the command line. These include scripts to run Stages 1, 2, and 3 in inference mode, as well as a general Stage 3 training script for both pretraining and finetuning ProteoScribe.

### Stage 1 (inference)

Run PenCL inference from the entrypoint `biom3_PenCL_inference`, which accesses the script `src/biom3/Stage1/run_PenCL_inference.py`.

```bash
biom3_PenCL_inference \
    -i None \
    -c configs/stage1_config_PenCL_inference.json \
    -m ./weights/PenCL/BioM3_PenCL_epoch20.bin \
    -o outputs/test_PenCL_embeddings.pt
```

### Stage 2 (inference)

Run Facilitator sampling from the entrypoint `biom3_Facilitator_sample`, which accesses the script `src/biom3/Stage2/run_Facilitator_sample.py`.

```bash
biom3_Facilitator_sample \
    -i None \
    -c configs/stage2_config_Facilitator_sample.json \
    -m ./weights/Facilitator/BioM3_Facilitator_epoch20.bin \
    -o outputs/test_Facilitator_embeddings.pt
```

### Stage 3

#### Pretraining

The script `src/biom3/Stage3/run_PL_training.py` (also available as an entrypoint `biom3_pretrain_stage3`) contains the code necessary to pretrain and finetune the BioM3 Stage 3 component ProteoScribe.
This script takes a number of command line arguments specifying the transformer architecture, data sources, and computing environment (number of nodes, GPUs, etc.).
It also allows one to continue training from a specified checkpoint.

In order to organize and document different training runs, we use a config file and wrapper shell script to specify the many command line arguments, and pass these to the training script. 
Example config files are stored in the `arglists` directory.
The wrapper script `scripts/stage3_pretraining.sh` takes as arguments the config directory and config file name (without extension) and uses this file to source the command line arguments contained within. 
It also takes additional arguments, including a Weights&Biases API key for logging with W&B; a version name to identify the particular training run; the number of nodes and GPUs per node available; the number of training epochs; and a string specifying a checkpoint from which to resume training (or None if training from scratch).

Running this script will perform model training using the arguments specified in the config file, as well as those specified from the command line.

A final wrapper script can be found at `scripts/pretraining/pretrain_multinode.sh`. 
This script wraps the `stage3_pretraining.sh` script described above, and executes it using an `mpiexec` call.
This allows us to easily run model training from a job submission script, as demonstrated in `jobs/pretraining/job_pretrain_scratch_v1_n2.pbs`.
In this file, we request and specify 2 nodes on Polaris. We also specify the desired configuration file and number of training epochs. The W&B API key should be available as an environment variable. A version name is automatically produced from the given configurations.

#### Finetuning

The finetuning pipeline is similar to the pretraining one. 
The key difference is that we must specify a pretrained model and the number of transformer blocks or layers that we wish to freeze/finetune. 
In addition, we specify a finetuning dataset. The pretraining of ProteoScribe as described in the BioM3 paper uses a two-phase approach, in which the model is first trained for a specified number of epochs on a SwissProt dataset of around 500,000 sequence-text pairs (Phase 1), and then further trained for a given number of steps on a union of SwissProt and Pfam data (Phase 2).

For finetuning, we will want to train typically on a single dataset, and thus this logic should be expected to change.
Currently, we can achieve finetuning by passing the dataset of interest in as the `swiss_prot_data_root`, and leaving the Pfam dataset unspecified (None). 

#### Inference (Generation)

```bash
biom3_ProteoScribe_sample \
    -i outputs/test_Facilitator_embeddings.pt \
    -c configs/stage3_config_ProteoScribe_sample.json \
    -m ./weights/ProteoScribe/BioM3_ProteoScribe_pfam_epoch20_v1.bin \
    -o outputs/test_ProteoScribe_samples.pt
```

## References

[1] Natural Language Prompts Guide the Design of Novel Functional Protein Sequences. Nikša Praljak, Hugh Yeh, Miranda Moore, Michael Socolich, Rama Ranganathan, Andrew L. Ferguson. bioRxiv 2024.11.11.622734; doi: https://doi.org/10.1101/2024.11.11.622734

# BioM3 development

## About

## Setup

### Installation

#### Polaris

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

To activate:

```bash
cd /path/to/BioM3-dev
module use /soft/modulefiles
module load conda
source venvs/biom3-env/bin/activate
```

Basic verification.

```bash
python -c 'import torch; import pytorch_lightning as pl; import deepspeed; import Stage3_source.preprocess'
```

Run tests with

```bash
python -m pytest tests
```

#### Aurora

TODO...

### Data downloads

Check the `weights` directory, and follow instructions in the READMEs there to download pretrained weights for the various model components.

## Usage

### Stage 1

TODO...

### Stage 2

TODO... 

### Stage 3

#### Pretraining

The python script `scripts/PL_train_stage3.py` contains the code necessary to pretrain and finetune the BioM3 Stage 3 component ProteoScribe.
This script takes a number of command line arguments specifying the transformer architecture, data sources, and computing environment (number of nodes, GPUs, etc.).
It also allows one to continue training from a specified checkpoint.

In order to organize and document different training runs, we use a config file and wrapper shell script to specify the many command line arguments, and pass these to the training script. 
Example config files are stored in the `arglists` directory.
The wrapper script `scripts/stage3_pretraining.sh` takes as arguments the config directory and config file name (without extension) and uses this file to source the command line arguments contained within. 
It also takes additional arguments, including a Weights&Biases API key for logging with W&B; a version name to identify the particular training run; the number of nodes and GPUs per node available; the number of training epochs; and a string specifying a checkpoint from which to resume training (or None if training from scratch).

Running this script will essentially perform model training using the arguments specified in the config file, as well as those specified from the command line.

A final wrapper script can be found at `scripts/pretraining/pretrain_multinode.sh`. 
This script wraps the `stage3_pretraining.sh` script described above, and executes it using an `mpiexec` call.
This allows us to easily run model training from a job submission script, as demonstrated in `jobs/pretraining/job_pretrain_scratch_v1_n2.pbs`.
In this file, we request and specify 2 nodes on Polaris. We also specify the desired configuration file and number of training epochs. The W&B API key should be available as an environment variable. A version name is automatically produced from the given configurations.

#### Finetuning

The finetuning pipeline is similar to the pretraining one. 
The key difference is that we must specify a pretrained model and the number of transformer blocks that we wish to finetune. 
In addition, we specify a finetuning dataset. The pretraining of ProteoScribe as described in the BioM3 paper uses a two-phase approach, in which the model is first trained for a specified number of epochs on a SwissProt dataset of around 500,000 sequence-text pairs (Phase 1), and then further trained for a given number of steps on a union of SwissProt and Pfam data (Phase 2).

For finetuning, we will want to train typically on a single dataset, and thus this logic should be expected to change.
Currently, we can achieve finetuning by passing the dataset of interest in as the `swiss_prot_data_root`, and leaving the Pfam dataset unspecified (None). 

#### Generation

TODO...

## References

[1] Natural Language Prompts Guide the Design of Novel Functional Protein Sequences. Nikša Praljak, Hugh Yeh, Miranda Moore, Michael Socolich, Rama Ranganathan, Andrew L. Ferguson. bioRxiv 2024.11.11.622734; doi: https://doi.org/10.1101/2024.11.11.622734
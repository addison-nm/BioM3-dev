# BioM3 development

A working project developing and investigating the [BioM3 framework](https://openreview.net/forum?id=L1MyyRCAjX) (NeurIPS 2024).

## About

## Installation and setup

The `BioM3-dev` repo is available to clone from GitHub.

```bash
git clone https://github.com/addison-nm/BioM3-dev.git && cd BioM3-dev
```

**Note:** *In order to run the complete suite of tests, one needs to download pretrained weights.
Instructions to do so are included in the README located in the `weights` directory.*

For installation and setup instructions on the following machines, refer to the setup instructions located in the `docs/` folder.

| Machine | Instructions |
| ------- | ------------ |
| Polaris (ALCF) | [setup_polaris.md](./docs/setup_polaris.md) |
| Aurora (ALCF) | [setup_aurora.md](./docs/setup_aurora.md) |
| DGX Spark | [setup_spark.md](./docs/setup_spark.md) |


## Usage

After the pip installation, a number of entrypoints should be available from the command line. These include scripts to run Stages 1, 2, and 3 in inference mode, as well as a general Stage 3 training script for both pretraining and finetuning ProteoScribe.

### End-to-end inference pipeline

The three inference stages form a sequential pipeline. The output of each stage feeds into the next:

```txt
Input text prompts / protein sequences
        │
        ▼
biom3_PenCL_inference         → outputs/pencl_embeddings.pt       (z_t, z_p)
        │
        ▼
biom3_Facilitator_sample      → outputs/facilitator_embeddings.pt  (z_t, z_p, z_c)
        │
        ▼
biom3_ProteoScribe_sample     → outputs/generated_sequences.pt
```

### Configuration files

Each inference entrypoint takes a JSON config file from `configs/` (e.g. `configs/stage1_config_PenCL_inference.json`) that controls model hyperparameters and paths to backbone LLM weights.
Training uses a separate set of shell-based config files in `arglists/` (e.g. `arglists/config_pretrain_scratch_v1.sh`), which are sourced by the wrapper scripts in `scripts/`.

### Stage 1 (inference)

Run PenCL inference from the entrypoint `biom3_PenCL_inference`, which accesses the script `src/biom3/Stage1/run_PenCL_inference.py`.
This stage encodes protein sequences and text descriptions into a shared latent space using ESM2 (protein) and BiomedBERT (text) encoders.

**Preparation:** Edit `configs/stage1_config_PenCL_inference.json` and set `seq_model_path` and `text_model_path` to the paths of the downloaded LLM weights.

```json
"seq_model_path": "/path/to/weights/LLMs/esm2_t33_650M_UR50D.pt",
"text_model_path": "/path/to/weights/LLMs/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
```

**Arguments:**

| Flag | Long form | Required | Description |
| ---- | --------- | -------- | ----------- |
| `-i` | `--input_data_path` | Yes | Path to input CSV file, or `None` to use the built-in test dataset |
| `-c` | `--json_path` | Yes | Path to JSON configuration file |
| `-m` | `--model_path` | Yes | Path to model weights (`.bin`/`.pt`) or Lightning checkpoint (`.ckpt`) |
| `-o` | `--output_path` | Yes | Path to save output embeddings (`.pt`) |
| | `--device` | No | Device to use: `cuda` (default), `cpu`, or `xpu` |
| | `--batch_size` | No | Batch size for inference (default: 32) |
| | `--num_workers` | No | Number of DataLoader workers (default: 0) |
| | `--load_from_checkpoint` | No | Force loading `--model_path` as a Lightning checkpoint (auto-detected from `.ckpt` extension) |

#### Example: using the built-in test dataset with raw weights

```bash
biom3_PenCL_inference \
    --input_data_path None \
    --json_path configs/stage1_config_PenCL_inference.json \
    --model_path ./weights/PenCL/BioM3_PenCL_epoch20.bin \
    --output_path outputs/pencl_embeddings.pt
```

#### Example: using a custom CSV input, larger batch size, and CPU

```bash
biom3_PenCL_inference \
    --input_data_path data/my_proteins.csv \
    --json_path configs/stage1_config_PenCL_inference.json \
    --model_path ./weights/PenCL/BioM3_PenCL_epoch20.bin \
    --output_path outputs/pencl_embeddings.pt \
    --device cpu \
    --batch_size 16 \
    --num_workers 4
```

#### Example: loading from a Lightning checkpoint

```bash
biom3_PenCL_inference \
    --input_data_path None \
    --json_path configs/stage1_config_PenCL_inference.json \
    --model_path ./weights/PenCL/BioM3_PenCL_epoch20.ckpt \
    --output_path outputs/pencl_embeddings.pt
```

The `.ckpt` extension is detected automatically; alternatively, use `--load_from_checkpoint` to force checkpoint loading regardless of extension.

### Stage 2 (inference)

Run Facilitator sampling from the entrypoint `biom3_Facilitator_sample`, which accesses the script `src/biom3/Stage2/run_Facilitator_sample.py`.
This stage maps text embeddings (`z_t`) into the protein embedding distribution (`z_c`) using a learned MMD-based alignment model.

**Arguments:**

| Flag | Long form | Required | Description |
| ---- | --------- | -------- | ----------- |
| `-i` | `--input_data_path` | Yes | Path to Stage 1 output embeddings (`.pt` file containing `z_t` and `z_p`) |
| `-c` | `--json_path` | Yes | Path to JSON configuration file |
| `-m` | `--model_path` | Yes | Path to Facilitator model weights (`.bin`/`.pt`) |
| `-o` | `--output_data_path` | Yes | Path to save output embeddings (`.pt`) |
| | `--device` | No | Device to use: `cuda` (default), `cpu`, or `xpu` |
| | `--mmd_sample_limit` | No | Max number of samples used to compute MMD diagnostics; `-1` uses all (default: -1) |

#### Example: standard usage following Stage 1

```bash
biom3_Facilitator_sample \
    --input_data_path outputs/pencl_embeddings.pt \
    --json_path configs/stage2_config_Facilitator_sample.json \
    --model_path ./weights/Facilitator/BioM3_Facilitator_epoch20.bin \
    --output_data_path outputs/facilitator_embeddings.pt
```

#### Example: limiting MMD computation to 256 samples (useful for large datasets)

```bash
biom3_Facilitator_sample \
    --input_data_path outputs/pencl_embeddings.pt \
    --json_path configs/stage2_config_Facilitator_sample.json \
    --model_path ./weights/Facilitator/BioM3_Facilitator_epoch20.bin \
    --output_data_path outputs/facilitator_embeddings.pt \
    --device cpu \
    --mmd_sample_limit 256
```

### Stage 3

#### Pretraining

The script `src/biom3/Stage3/run_PL_training.py` (also available as an entrypoint `biom3_pretrain_stage3`) contains the code necessary to pretrain and finetune the BioM3 Stage 3 component ProteoScribe.
This script takes a number of command line arguments specifying the transformer architecture, data sources, and computing environment (device, number of nodes, GPUs, etc.).
It also allows one to continue training from a specified checkpoint or model weights.

In order to organize and document different training runs, we use a config file and wrapper shell script to specify the many command line arguments, and pass these to the training script.
Example config files are stored in the `arglists` directory.
The wrapper script `scripts/stage3_pretraining.sh` takes as arguments the config directory and config file name (without extension) and uses this file to source the command line arguments contained within.
It also takes additional arguments, including a Weights&Biases API key for logging with W&B; a version name to identify the particular training run; the device, number of nodes, and GPUs per node available; the number of training epochs; and a string specifying a checkpoint or model weights from which to resume training (or None if training from scratch).

Running this script will perform model training using the arguments specified in the config file, as well as those specified from the command line, allowing one to override particular configurations (e.g. the particular device available).

A final wrapper script can be found at `scripts/pretraining/pretrain_multinode.sh`.
This script wraps the `stage3_pretraining.sh` script described above, and executes it using an `mpiexec` call.
This allows us to easily run model training from a job submission script, as demonstrated in the Polaris and Aurora demo job files: `jobs/polaris/_template_pretrain_scratch.pbs` and `jobs/aurora/_template_pretrain_scratch.pbs`.
In these files, we request and specify 2 nodes. We also specify the desired configuration file and number of training epochs. The W&B API key should be available as an environment variable. A version name is automatically produced from the given configurations.

#### Finetuning

The finetuning pipeline is similar to the pretraining one.
The key difference is that we must specify a pretrained model and the number of transformer blocks or layers that we wish to freeze/finetune.
In addition, we specify a finetuning dataset. The pretraining of ProteoScribe as described in the BioM3 paper uses a two-phase approach, in which the model is first trained for a specified number of epochs on a SwissProt dataset of around 500,000 sequence-text pairs (Phase 1), and then further trained for a given number of steps on a union of SwissProt and Pfam data (Phase 2).

For finetuning, we will want to train typically on a single dataset, and thus this logic should be expected to change.
Currently, we achieve finetuning by passing the dataset of interest in as the `swiss_prot_data_root`, and leaving the Pfam dataset unspecified (None).

#### Inference (Generation)

Run ProteoScribe sampling from the entrypoint `biom3_ProteoScribe_sample`, which accesses the script `src/biom3/Stage3/run_ProteoScribe_sample.py`.
This stage generates protein sequences from the facilitated text embeddings (`z_c`) produced by Stage 2, using the conditional diffusion transformer.

**Arguments:**

| Flag | Long form | Required | Description |
| ---- | --------- | -------- | ----------- |
| `-i` | `--input_path` | Yes | Path to Stage 2 output embeddings (`.pt` file containing `z_c`) |
| `-c` | `--json_path` | Yes | Path to JSON configuration file |
| `-m` | `--model_path` | Yes | Path to ProteoScribe model weights (`.bin`/`.pt`) or Lightning checkpoint (`.ckpt`) |
| `-o` | `--output_path` | Yes | Path to save generated sequences (`.pt`) |
| | `--seed` | No | Random seed for reproducibility. Pass `0` or omit for a random seed (default: 0) |
| | `--device` | No | Device to use: `cuda` (default), `cpu`, or `xpu` |

> **Note:** To control sampling behavior (number of sequences per prompt, batch size, diffusion steps), edit `num_replicas`, `batch_size_sample`, and `diffusion_steps` in the JSON config.

#### Example: standard usage following Stage 2

```bash
biom3_ProteoScribe_sample \
    --input_path outputs/facilitator_embeddings.pt \
    --json_path configs/stage3_config_ProteoScribe_sample.json \
    --model_path ./weights/ProteoScribe/BioM3_ProteoScribe_pfam_epoch20_v1.bin \
    --output_path outputs/generated_sequences.pt
```

#### Example: reproducible run with a fixed seed

```bash
biom3_ProteoScribe_sample \
    --input_path outputs/facilitator_embeddings.pt \
    --json_path configs/stage3_config_ProteoScribe_sample.json \
    --model_path ./weights/ProteoScribe/BioM3_ProteoScribe_pfam_epoch20_v1.bin \
    --output_path outputs/generated_sequences.pt \
    --seed 42
```

## References

[1] Natural Language Prompts Guide the Design of Novel Functional Protein Sequences. Nikša Praljak, Hugh Yeh, Miranda Moore, Michael Socolich, Rama Ranganathan, Andrew L. Ferguson. bioRxiv 2024.11.11.622734; doi: [10.1101/2024.11.11.622734](https://doi.org/10.1101/2024.11.11.622734)

# BioM3 development

A working project developing and investigating the [BioM3 framework](https://openreview.net/forum?id=L1MyyRCAjX) (NeurIPS 2024).

## About

BioM3 development is organized across several repositories:

| Repository | Role | Description |
|------------|------|-------------|
| **BioM3-dev** (this repo) | Core library | Python package: 3-stage pipeline, dataset construction, training |
| [BioM3-data-share](https://github.com/natural-machine/BioM3-data-share) | Shared data | Model weights, datasets, and reference databases synced across clusters |
| [BioM3-workflow-demo](https://github.com/natural-machine/BioM3-workflow-demo) | Demo workflows | End-to-end finetuning and generation demonstration pipeline |
| BioM3-workspace-template | Workspace setup | *(Planned)* Standardized workspace template for new research projects |

See [docs/biom3_ecosystem.md](./docs/biom3_ecosystem.md) for cross-repo workflows, version compatibility, and shared data architecture.

## Installation and setup

The `BioM3-dev` repo is available to clone from GitHub.

```bash
git clone https://github.com/addison-nm/BioM3-dev.git && cd BioM3-dev
```

### pip install

Install the core package:

```bash
# Editable (development) install
pip install -e .

# From GitHub (specific branch or tag)
pip install 'biom3 @ git+https://github.com/addison-nm/BioM3-dev.git'
pip install 'biom3 @ git+https://github.com/addison-nm/BioM3-dev.git@v0.1.0a1'
```

To include the Streamlit web app and visualization tools, install with the `app` extra:

```bash
# Editable install with app dependencies
pip install -e '.[app]'

# From GitHub with app dependencies
pip install 'biom3[app] @ git+https://github.com/addison-nm/BioM3-dev.git'
```

> **Note:** The quotes around `'.[app]'` and `'biom3[app] @ ...'` are required.
> Without them, shells like `zsh` interpret the square brackets as glob patterns
> and the command will fail.

### Environment setup

**Important:** Before running tests or scripts, source the `environment.sh` file to set required
environment variables. The environment variables needed may differ across machines — see the
Usage section in each machine's setup doc for details.

```bash
source environment.sh
```

**Note:** *Some tests require pretrained weights that are too large to commit to git. These tests
are skipped automatically when the weights are absent. To run the full test suite, populate the
`weights/` directory using the shared weights sync script — see
[docs/setup_shared_weights.md](./docs/setup_shared_weights.md) for machine-specific paths, the
list of required files, and setup instructions.*

For installation and setup instructions on the following machines, refer to the setup instructions located in the `docs/` folder.

| Machine | Instructions |
| ------- | ------------ |
| Polaris (ALCF) | [setup_polaris.md](./docs/setup_polaris.md) |
| Aurora (ALCF) | [setup_aurora.md](./docs/setup_aurora.md) |
| DGX Spark | [setup_spark.md](./docs/setup_spark.md) |


## Reference databases

Building fine-tuning datasets with `biom3.dbio` requires access to protein reference databases (NCBI Taxonomy, Pfam, Swiss-Prot, etc.) and pre-processed training CSVs. These files are too large to commit to git and are stored in a shared directory on each machine.

The local `data/databases/` directory is populated with symlinks to the shared databases using a sync script — the same pattern used for model weights. To set up:

```bash
# Preview what will be linked
./scripts/sync_databases.sh <shared_databases_path> data/databases --dry-run

# Create symlinks
./scripts/sync_databases.sh <shared_databases_path> data/databases
```

For example, on DGX Spark:

```bash
./scripts/sync_databases.sh /data/data-share/BioM3-data-share/databases data/databases
```

Once synced, you can use `biom3_build_dataset` to construct fine-tuning datasets by subsetting Swiss-Prot and Pfam by Pfam ID, optionally enriched with UniProt annotations and taxonomy data:

```bash
# Basic extraction
biom3_build_dataset -p PF00018 -o outputs/SH3_dataset

# With UniProt-enriched captions (PROTEIN NAME, FUNCTION, GENE ONTOLOGY, etc.)
biom3_build_dataset -p PF00018 --enrich_pfam -o outputs/SH3_dataset

# With enrichment + NCBI taxonomy lineage and filtering
biom3_build_dataset -p PF00018 --enrich_pfam --add_taxonomy --taxonomy_filter "superkingdom=Bacteria" -o outputs/SH3_bacteria
```

See [docs/setup_databases.md](./docs/setup_databases.md) for machine-specific shared paths, the full list of databases, and configuration details.

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

Each inference entrypoint takes a JSON config file from `configs/` (e.g. `configs/inference/stage1_PenCL.json`) that controls model hyperparameters and paths to backbone LLM weights.
Training uses a separate set of shell-based config files in `arglists/` (e.g. `arglists/config_pretrain_scratch_v1.sh`), which are sourced by the wrapper scripts in `scripts/`.

### Stage 1 (inference)

Run PenCL inference from the entrypoint `biom3_PenCL_inference`, which accesses the script `src/biom3/Stage1/run_PenCL_inference.py`.
This stage encodes protein sequences and text descriptions into a shared latent space using ESM2 (protein) and BiomedBERT (text) encoders.

**Preparation:** Edit `configs/inference/stage1_PenCL.json` and set `seq_model_path` and `text_model_path` to the paths of the downloaded LLM weights.

```json
"seq_model_path": "/path/to/weights/LLMs/esm2_t33_650M_UR50D.pt",
"text_model_path": "/path/to/weights/LLMs/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
```

**Arguments:**

| Flag | Long form | Required | Description |
| ---- | --------- | -------- | ----------- |
| `-i` | `--input_data_path` | Yes | Path to input CSV file, or `None` to use the built-in test dataset |
| `-c` | `--config_path` | Yes | Path to JSON configuration file |
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
    --config_path configs/inference/stage1_PenCL.json \
    --model_path ./weights/PenCL/BioM3_PenCL_epoch20.bin \
    --output_path outputs/pencl_embeddings.pt
```

#### Example: using a custom CSV input, larger batch size, and CPU

```bash
biom3_PenCL_inference \
    --input_data_path data/my_proteins.csv \
    --config_path configs/inference/stage1_PenCL.json \
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
    --config_path configs/inference/stage1_PenCL.json \
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
| `-c` | `--config_path` | Yes | Path to JSON configuration file |
| `-m` | `--model_path` | Yes | Path to Facilitator model weights (`.bin`/`.pt`) |
| `-o` | `--output_data_path` | Yes | Path to save output embeddings (`.pt`) |
| | `--device` | No | Device to use: `cuda` (default), `cpu`, or `xpu` |
| | `--mmd_sample_limit` | No | Max number of samples used to compute MMD diagnostics; `-1` uses all (default: -1) |

#### Example: standard usage following Stage 1

```bash
biom3_Facilitator_sample \
    --input_data_path outputs/pencl_embeddings.pt \
    --config_path configs/inference/stage2_Facilitator.json \
    --model_path ./weights/Facilitator/BioM3_Facilitator_epoch20.bin \
    --output_data_path outputs/facilitator_embeddings.pt
```

#### Example: limiting MMD computation to 256 samples (useful for large datasets)

```bash
biom3_Facilitator_sample \
    --input_data_path outputs/pencl_embeddings.pt \
    --config_path configs/inference/stage2_Facilitator.json \
    --model_path ./weights/Facilitator/BioM3_Facilitator_epoch20.bin \
    --output_data_path outputs/facilitator_embeddings.pt \
    --device cpu \
    --mmd_sample_limit 256
```

### Stage 3

#### Pretraining

The entrypoint `biom3_pretrain_stage3` (script `src/biom3/Stage3/run_PL_training.py`) handles both pretraining and finetuning of ProteoScribe.
Training configuration is specified via a JSON config file passed with `--config_path`, with per-job overrides (device, number of nodes, run ID, etc.) passed as CLI arguments.
CLI arguments override JSON values, which override argparse defaults.

Example JSON configs are in `configs/stage3_training/`.
Configs support layered composition: `_base_configs` for shared model architectures (`configs/stage3_training/models/`) and `_overwrite_configs` for per-machine settings (`configs/stage3_training/machines/`). See [docs/stage3_training.md](./docs/stage3_training.md) for details.
Wrapper scripts `scripts/stage3_train_multinode.sh` (multi-node via `mpiexec`) and `scripts/stage3_train_singlenode.sh` (single-node) handle environment setup and launch the entrypoint.
HPC job templates in `jobs/{polaris,aurora,spark}/` demonstrate how to use these wrappers.

```bash
# Direct usage
biom3_pretrain_stage3 \
    --config_path configs/stage3_training/pretrain_scratch_v2.json \
    --run_id my_run_v1 \
    --device cuda \
    --epochs 10

# Via multinode wrapper (from an HPC job template)
./scripts/stage3_train_multinode.sh \
    configs/stage3_training/pretrain_scratch_v2.json \
    2 4 cuda $WANDB_API_KEY my_run_v1 \
    --epochs 10
```

#### Finetuning

Finetuning uses the same entrypoint with `--finetune True` and additional flags for specifying pretrained weights and which transformer blocks/layers to unfreeze.
The key difference is that we must specify a pretrained model and the number of transformer blocks or layers that we wish to freeze/finetune.
Example finetuning configs are in `configs/stage3_training/finetune_*.json`.

```bash
biom3_pretrain_stage3 \
    --config_path configs/stage3_training/finetune_v1.json \
    --run_id finetune_v1 \
    --device cuda \
    --pretrained_weights ./weights/ProteoScribe/BioM3_ProteoScribe_pfam_epoch20_v1.bin \
    --finetune_last_n_blocks 1 \
    --finetune_last_n_layers -1
```

#### Inference (Generation)

Run ProteoScribe sampling from the entrypoint `biom3_ProteoScribe_sample`, which accesses the script `src/biom3/Stage3/run_ProteoScribe_sample.py`.
This stage generates protein sequences from the facilitated text embeddings (`z_c`) produced by Stage 2, using the conditional diffusion transformer.

**Arguments:**

| Flag | Long form | Required | Description |
| ---- | --------- | -------- | ----------- |
| `-i` | `--input_path` | Yes | Path to Stage 2 output embeddings (`.pt` file containing `z_c`) |
| `-c` | `--config_path` | Yes | Path to JSON configuration file |
| `-m` | `--model_path` | Yes | Path to ProteoScribe model weights (`.bin`/`.pt`) or Lightning checkpoint (`.ckpt`) |
| `-o` | `--output_path` | Yes | Path to save generated sequences (`.pt`) |
| | `--seed` | No | Random seed for reproducibility. Pass `0` or omit for a random seed (default: 0) |
| | `--device` | No | Device to use: `cuda` (default), `cpu`, or `xpu` |
| | `--animate_prompts` | No | Prompt indices to animate (e.g. `0 1 2`), `all`, or `none`. Omit to disable animation. |
| | `--animate_replicas` | No | Replicas to animate: integer `i` selects `range(0, i)`, `all`, or `none` (default: `1`) |
| | `--animation_dir` | No | Directory for GIF output. Default: `<output_dir>/animations/` |
| | `--fasta` | No | Write one FASTA file per prompt to `<output_dir>/fasta/` |
| | `--fasta_merge` | No | Also write a single merged FASTA with all sequences (requires `--fasta`) |
| | `--fasta_dir` | No | Custom output directory for FASTA files |

> **Note:** To control sampling behavior (number of sequences per prompt, batch size, diffusion steps), edit `num_replicas`, `batch_size_sample`, and `diffusion_steps` in the JSON config.

#### Example: standard usage following Stage 2

```bash
biom3_ProteoScribe_sample \
    --input_path outputs/facilitator_embeddings.pt \
    --config_path configs/inference/stage3_ProteoScribe_sample.json \
    --model_path ./weights/ProteoScribe/BioM3_ProteoScribe_pfam_epoch20_v1.bin \
    --output_path outputs/generated_sequences.pt
```

#### Example: reproducible run with a fixed seed

```bash
biom3_ProteoScribe_sample \
    --input_path outputs/facilitator_embeddings.pt \
    --config_path configs/inference/stage3_ProteoScribe_sample.json \
    --model_path ./weights/ProteoScribe/BioM3_ProteoScribe_pfam_epoch20_v1.bin \
    --output_path outputs/generated_sequences.pt \
    --seed 42
```

#### Example: animate the denoising process for selected prompts

```bash
biom3_ProteoScribe_sample \
    --input_path outputs/facilitator_embeddings.pt \
    --config_path configs/inference/stage3_ProteoScribe_sample.json \
    --model_path ./weights/ProteoScribe/BioM3_ProteoScribe_pfam_epoch20_v1.bin \
    --output_path outputs/generated_sequences.pt \
    --animate_prompts 0 1 2 \
    --animate_replicas 2
```

GIFs are written to `outputs/animations/prompt_<P>_replica_<R>.gif`. See [docs/sequence_generation_animation.md](./docs/sequence_generation_animation.md) for details.

### Dataset construction

The `biom3.dbio` subpackage provides tools for constructing fine-tuning datasets from local protein databases. The main entrypoint is `biom3_build_dataset`, which subsets the Swiss-Prot and Pfam training CSVs by one or more Pfam IDs.

**Arguments:**

| Flag | Long form | Required | Description |
| ---- | --------- | -------- | ----------- |
| `-p` | `--pfam_ids` | Yes | One or more Pfam IDs to extract (e.g. `PF00018 PF00169`) |
| `-o` | `--outdir` | Yes | Output directory (created if it doesn't exist) |
| | `--swissprot` | No | Path to Swiss-Prot CSV (default: from `configs/dbio_config.json`) |
| | `--pfam` | No | Path to Pfam CSV (default: from config) |
| | `--enrich_pfam` | No | Enrich Pfam captions with UniProt annotations (API by default) |
| | `--uniprot_dat` | No | Use local `.dat.gz` file(s) instead of API. Accepts multiple paths for full coverage (e.g. Swiss-Prot + TrEMBL) |
| | `--add_taxonomy` | No | Add NCBI taxonomy lineage to Pfam captions (local, no API) |
| | `--taxonomy_filter` | No | Filter by taxonomy rank (e.g. `"superkingdom=Bacteria"`) |
| | `--chunk_size` | No | Pfam CSV chunk size (default: 500000) |

#### Example: build an SH3 domain dataset

```bash
biom3_build_dataset \
    -p PF00018 \
    -o outputs/SH3_dataset
```

#### Example: with taxonomy lineage and filtering

```bash
biom3_build_dataset \
    -p PF00018 \
    --add_taxonomy \
    --taxonomy_filter "superkingdom=Bacteria" \
    -o outputs/SH3_bacteria
```

#### Example: multiple Pfam IDs with UniProt enrichment

```bash
biom3_build_dataset \
    -p PF00018 PF07714 \
    --enrich_pfam \
    --add_taxonomy \
    -o outputs/SH3_Pkinase
```

#### Example: offline enrichment with local `.dat` files

```bash
biom3_build_dataset \
    -p PF00018 \
    --enrich_pfam \
    --uniprot_dat data/databases/swissprot/uniprot_sprot.dat.gz \
                  data/databases/swissprot/uniprot_trembl.dat.gz \
    -o outputs/SH3_dataset
```

A separate utility builds a SQLite index for fast taxonomy lookups:

```bash
biom3_build_taxid_index data/databases/ncbi_taxonomy/prot.accession2taxid.gz
```

### Web app and visualization

BioM3 includes a Streamlit web app for browsing data and visualizing protein structures, and a Python visualization library for 3D structure rendering and sequence analysis.

Install with the `app` extra (see [Installation](#pip-install)):

```bash
pip install -e '.[app]'
```

Launch the web app:

```bash
biom3_app
```

See [docs/web_app.md](./docs/web_app.md) for app pages, configuration, and architecture. See [docs/structure_visualization.md](./docs/structure_visualization.md) for the `biom3.viz` Python API (3D rendering, structural alignment, BLAST, unmasking-order visualization).

## References

[1] Natural Language Prompts Guide the Design of Novel Functional Protein Sequences. Nikša Praljak, Hugh Yeh, Miranda Moore, Michael Socolich, Rama Ranganathan, Andrew L. Ferguson. bioRxiv 2024.11.11.622734; doi: [10.1101/2024.11.11.622734](https://doi.org/10.1101/2024.11.11.622734)

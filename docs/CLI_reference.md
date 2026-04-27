# BioM3 CLI Reference

Reference for the seven user-facing entrypoints declared in [pyproject.toml](../pyproject.toml). Each section gives the synopsis, required and optional arguments, the canonical config file (where applicable), and a representative example. For per-stage prose context (output layouts, metrics, per-machine job submission), see the deeper docs linked from each section.

dbio entrypoints (`biom3_build_dataset`, `biom3_build_taxid_index`, `biom3_convert_to_parquet`, `biom3_build_source_*`, `biom3_build_annotation_cache`, `biom3_build_annotated_pfam_subsets`) are documented separately in [dbio_examples.md](dbio_examples.md). Benchmark and Streamlit-app entrypoints (`biom3_benchmark_*`, `biom3_app`, `biom3_plot_benchmark`, `biom3_compile_hdf5`) are not covered here — invoke them with `--help` for current options.

> All entrypoints accept `--help`. The tables below mirror the source-of-truth `add_argument` declarations; if behavior diverges, the source wins.

> **Config composition**: every entrypoint that accepts `--config_path` loads JSON via `core.helpers.load_json_config`, which honors two special keys: `_base_configs` (loaded before the current file — current file overrides) and `_overwrite_configs` (loaded after — they override). CLI args override everything. See [stage3_training.md#config-composition](stage3_training.md#config-composition).

---

## Inference Entrypoints

### `biom3_PenCL_inference` — Stage 1 PenCL inference

Produces joint protein/text embeddings (`z_p`, `z_t`) from a CSV of (sequence, prompt) pairs.

**Source:** [src/biom3/Stage1/run_PenCL_inference.py](../src/biom3/Stage1/run_PenCL_inference.py)
**Config:** `configs/inference/stage1_PenCL.json`
**Deeper doc:** none (this reference + module docstring are authoritative).

#### Required arguments

| Arg | Type | Description |
|---|---|---|
| `-i`, `--input_data_path` | str | Path to input CSV (sequences + prompts). Pass `None` to use the built-in test dataset. |
| `-c`, `--config_path` | str | Path to JSON config (e.g. `configs/inference/stage1_PenCL.json`). |
| `-m`, `--model_path` | str | Path to pretrained PenCL weights (`.bin`) or Lightning checkpoint (`.ckpt`). |
| `-o`, `--output_path` | str | Path to write output embeddings (`.pt`). |

#### Optional arguments

| Arg | Type | Default | Description |
|---|---|---|---|
| `--device` | str | `cuda` | One of `cpu`, `cuda`, `xpu`. |
| `--batch_size` | int | 32 | Inference batch size. |
| `--num_workers` | int | 0 | DataLoader worker count. |
| `--load_from_checkpoint` | flag | False | Force loading `model_path` as a Lightning `.ckpt` (otherwise inferred from extension). |
| `--cross_comparison_sample_limit` | int | -1 | Cap samples used for O(n²) cross-comparison metrics (dot-product probabilities, homology matrix). `-1` = all. **Print-only** — saved embeddings are unaffected. Use to avoid OOM on large datasets. |

#### Example

```bash
biom3_PenCL_inference \
    --input_data_path data/my_proteins.csv \
    --config_path configs/inference/stage1_PenCL.json \
    --model_path ./weights/PenCL/BioM3_PenCL_epoch20.bin \
    --output_path outputs/pencl_embeddings.pt \
    --batch_size 64 \
    --cross_comparison_sample_limit 1000
```

---

### `biom3_Facilitator_sample` — Stage 2 Facilitator sampling

Maps Stage 1 text embeddings (`z_t`) into the protein-embedding space (`z_c`). Consumes the `.pt` output of `biom3_PenCL_inference`.

**Source:** [src/biom3/Stage2/run_Facilitator_sample.py](../src/biom3/Stage2/run_Facilitator_sample.py)
**Config:** `configs/inference/stage2_Facilitator.json`

#### Required arguments

| Arg | Type | Description |
|---|---|---|
| `-i`, `--input_data_path` | str | Path to Stage 1 output embeddings (`.pt`). |
| `-c`, `--config_path` | str | Path to JSON config. |
| `-m`, `--model_path` | str | Path to Facilitator weights (`.bin`). |
| `-o`, `--output_data_path` | str | Path to write Stage 2 output embeddings (`.pt`). |

#### Optional arguments

| Arg | Type | Default | Description |
|---|---|---|---|
| `--device` | str | `cuda` | One of `cpu`, `cuda`, `xpu`. |
| `--mmd_sample_limit` | int | -1 | Cap samples for MMD computation. `-1` = all. **Print-only** — saved `z_c` embeddings are unaffected. |

#### Example

```bash
biom3_Facilitator_sample \
    --input_data_path outputs/pencl_embeddings.pt \
    --config_path configs/inference/stage2_Facilitator.json \
    --model_path ./weights/Facilitator/BioM3_Facilitator_epoch20.bin \
    --output_data_path outputs/facilitator_embeddings.pt \
    --mmd_sample_limit 256
```

---

### `biom3_ProteoScribe_sample` — Stage 3 sequence generation

Generates protein sequences from facilitated embeddings via diffusion sampling. Consumes the `.pt` output of `biom3_Facilitator_sample`.

**Source:** [src/biom3/Stage3/run_ProteoScribe_sample.py](../src/biom3/Stage3/run_ProteoScribe_sample.py)
**Config:** `configs/inference/stage3_ProteoScribe_sample.json`
**Deeper docs:** [sequence_generation_animation.md](sequence_generation_animation.md) for `--animate_*` and `--animation_*` flags.

#### Required arguments

| Arg | Type | Description |
|---|---|---|
| `-i`, `--input_path` | str | Path to Stage 2 output embeddings (`.pt`). |
| `-c`, `--config_path` | str | Path to JSON config. |
| `-m`, `--model_path` | str | Path to pretrained ProteoScribe weights (`pytorch_model.bin`). |
| `-o`, `--output_path` | str | Path to write generated sequences (`.pt`). |

#### Optional arguments — sampling

| Arg | Type | Default | Description |
|---|---|---|---|
| `--seed` | int | 0 | RNG seed. |
| `--device` | str | `cuda` | One of `cpu`, `cuda`, `xpu`. |
| `--unmasking_order` | str | None | One of `random`, `confidence`, `confidence_no_pad`. Defaults to `random`. |
| `--token_strategy` | str | None | One of `sample` (Gumbel-max, default) or `argmax` (deterministic). |
| `--pre_unmask` | flag | False | Start diffusion from a partially-unmasked state. Requires `--pre_unmask_config`. |
| `--pre_unmask_config` | str | None | Path to JSON describing the pre-unmask strategy. |

#### Optional arguments — output

| Arg | Type | Default | Description |
|---|---|---|---|
| `--fasta` | flag | False | Write one FASTA file per prompt to `<output_dir>/fasta/`. |
| `--fasta_merge` | flag | False | Also write a single merged FASTA (requires `--fasta`). |
| `--fasta_dir` | str | None | Output directory for FASTA files. |
| `--store_probabilities` | flag | False | Save per-step probabilities as `.npz`. Memory-intensive. |

#### Optional arguments — animation

See [sequence_generation_animation.md](sequence_generation_animation.md) for full details.

| Arg | Type | Default | Description |
|---|---|---|---|
| `--animate_prompts` | str (n+) | None | Prompt indices to animate (e.g. `0 1 2`), `all`, or `none`. |
| `--animate_replicas` | str | `1` | Replicas to animate: integer `i` = `range(0, i)`, `all`, or `none`. |
| `--animation_dir` | str | None | Output directory for GIFs. Default: `<output_dir>/animations/`. |
| `--animation_style` | str | `brightness` | One of `brightness`, `colorbar`, `logo`, `gauge`. |
| `--animation_metrics` | str (n*) | None | Per-position metric overlays (e.g. `confidence`). |

#### Example

```bash
biom3_ProteoScribe_sample \
    --input_path outputs/facilitator_embeddings.pt \
    --config_path configs/inference/stage3_ProteoScribe_sample.json \
    --model_path ./weights/ProteoScribe/BioM3_ProteoScribe_pfam_epoch20_v1.bin \
    --output_path outputs/generated_sequences.pt \
    --seed 42 \
    --fasta
```

---

### `biom3_embedding_pipeline` — End-to-end embedding pipeline

Runs `biom3_PenCL_inference` → `biom3_Facilitator_sample` → HDF5 compilation in sequence. Intermediate paths are constructed automatically from `--output_dir` and `--prefix`.

**Source:** [src/biom3/pipeline/embedding_pipeline.py](../src/biom3/pipeline/embedding_pipeline.py)
**Config:** delegates to two configs (one per stage).
**Deeper doc:** [embedding_pipeline.md](embedding_pipeline.md).

#### Required arguments

| Arg | Type | Description |
|---|---|---|
| `-i`, `--input_data_path` | str | Path to input CSV (sequences + prompts). |
| `-o`, `--output_dir` | str | Directory for all output files. |
| `--pencl_weights` | str | Path to PenCL weights or checkpoint. |
| `--facilitator_weights` | str | Path to Facilitator weights or checkpoint. |
| `--pencl_config` | str | Path to Stage 1 JSON config. |
| `--facilitator_config` | str | Path to Stage 2 JSON config. |
| `--prefix` | str | Filename prefix for intermediate and final output files. |

#### Optional arguments

| Arg | Type | Default | Description |
|---|---|---|---|
| `--device` | str | `cuda` | One of `cpu`, `cuda`, `xpu`. |
| `--batch_size` | int | 256 | Stage 1 batch size. |
| `--num_workers` | int | 0 | Stage 1 DataLoader worker count. |
| `--mmd_sample_limit` | int | 1000 | Stage 2 MMD sample cap. |
| `--dataset_key` | str | `MMD_data` | HDF5 group name for the compiled output. |

---

## Training Entrypoints

All three training entrypoints share a similar argparser shape (assembled from `get_args`, `get_model_args`, `get_path_args`, `get_wrapper_args`) and read CLI < JSON < argparse defaults. Stringified booleans (e.g. `--wandb True`, `--scale_learning_rate False`) are converted via `str_to_bool` inside the script.

### `biom3_pretrain_stage1` — Stage 1 PenCL training

Trains the joint protein/text encoder (PenCL) from a CSV dataset.

**Source:** [src/biom3/Stage1/run_PL_training.py](../src/biom3/Stage1/run_PL_training.py)
**Config dir:** `configs/stage1_training/`
**Deeper doc:** none yet — see [stage3_training.md](stage3_training.md) for the analogous training conventions (output layout, checkpoint formats, metric history).

#### Required arguments

None positional. `--config_path` is technically optional but expected in practice.

#### Key arguments

The argparser declares ~50 flags. Highlights below; run `biom3_pretrain_stage1 --help` for the complete list.

| Arg | Type | Default | Description |
|---|---|---|---|
| `--config_path`, `-c` | str | None | Path to JSON config. CLI overrides JSON. |
| `--run_id` | str | None | Unique identifier for this run; drives output directory naming. |
| `--data_path` | str | None | Path to Swiss-Prot CSV. |
| `--pfam_data_path` | str | `'None'` | Path to Pfam CSV (required when `dataset_type=pfam`). |
| `--dataset_type` | str | `default` | One of `default`, `masked`, `pfam`, `pfam_ablated`. |
| `--device` | str | `cuda` | One of `cuda`, `xpu`, `cpu`. |
| `--gpu_devices` | int | 1 | GPUs (CUDA) or tiles (XPU) per node. |
| `--num_nodes` | int | 1 | Nodes participating in training. |
| `--batch_size` | int | 8 | Per-device mini-batch size. |
| `--epochs` | int | 20 | Training epochs. |
| `--valid_size` | float | 0.1 | Train/val split fraction (0.1 = 90/10). |
| `--limit_val_batches` | float | 1.0 | See [Choosing limit_val_batches](stage3_training.md#choosing-limit_val_batches-and-limit_train_batches). |
| `--limit_train_batches` | float | None | Same convention as above. |
| `--head_lr` / `--protein_encoder_lr` / `--text_encoder_lr` | float | varies | Per-component learning rates. |
| `--scale_learning_rate` | str | `'False'` | `'True'`/`'False'`. Scale LR by world size. |
| `--precision` | str | `'32'` | One of `'32'`, `'16'`, `'bf16'`, `'bf16-mixed'`. |
| `--resume_from_checkpoint` | str | `'None'` | Path to a Lightning `.ckpt` to resume from. |
| `--pretrained_weights` | str | `'None'` | Path to a raw weights file (no optimizer state). |
| `--wandb` | str | `'False'` | `'True'`/`'False'`. Enable wandb logging (requires `WANDB_API_KEY`). |
| `--output_root` | str | `./outputs/Stage1/pretraining` | Base output directory. |

#### Example (Aurora job submission via wrapper)

```bash
# In a PBS template
./scripts/stage1_train_singlenode.sh \
    configs/stage1_training/pretrain_pfam_v1.json \
    12 xpu my_run_001 \
    --epochs 20 --resume_from_checkpoint None --wandb ${use_wandb}
```

---

### `biom3_pretrain_stage2` — Stage 2 Facilitator training

Trains the Facilitator that aligns text embeddings (`z_t`) to protein embeddings (`z_p`). Consumes Stage 1 output `.pt` dicts.

**Source:** [src/biom3/Stage2/run_PL_training.py](../src/biom3/Stage2/run_PL_training.py)
**Config dir:** `configs/stage2_training/`

#### Required arguments

None positional.

#### Key arguments

| Arg | Type | Default | Description |
|---|---|---|---|
| `--config_path`, `-c` | str | None | Path to JSON config. CLI overrides JSON. |
| `--run_id` | str | None | Unique identifier for this run. |
| `--swissprot_data_path` | str | `'None'` | Path to Stage 1 SwissProt embeddings `.pt` dict. |
| `--pfam_data_path` | str | `'None'` | Path to Stage 1 Pfam embeddings `.pt` dict. |
| `--output_swissprot_dict_path` | str | None | Where to save Stage 2 SwissProt embeddings dict. |
| `--output_pfam_dict_path` | str | None | Where to save Stage 2 Pfam embeddings dict. |
| `--device` | str | `cuda` | One of `cuda`, `xpu`, `cpu`. |
| `--gpu_devices` | int | 1 | GPUs/tiles per node. |
| `--num_nodes` | int | 1 | Nodes. |
| `--batch_size` | int | 32 | Per-device batch size. |
| `--epochs` | int | 20 | Training epochs. |
| `--valid_size` | float | 0.2 | Train/val split (0.2 = 80/20). |
| `--limit_val_batches` | float | 1.0 | See [Choosing limit_val_batches](stage3_training.md#choosing-limit_val_batches-and-limit_train_batches). |
| `--lr` | float | 1e-3 | Base learning rate. |
| `--loss_type` | str | `MSE` | One of `MSE` (point-wise) or `MMD` (distribution-matching). |
| `--emb_dim` / `--hid_dim` | int | 512 / 1024 | Facilitator I/O and hidden dims. |
| `--wandb` | str | `'False'` | Enable wandb (requires `WANDB_API_KEY`). |

Run `biom3_pretrain_stage2 --help` for the complete list.

---

### `biom3_pretrain_stage3` — Stage 3 ProteoScribe training and finetuning

Trains the conditional diffusion transformer that generates protein sequences. Supports pretraining from scratch, secondary-data continuation, and selective-layer finetuning.

**Source:** [src/biom3/Stage3/run_PL_training.py](../src/biom3/Stage3/run_PL_training.py)
**Config dir:** `configs/stage3_training/`
**Deeper doc:** [stage3_training.md](stage3_training.md) — output layout, metric definitions, checkpointing details, finetuning recipes, per-machine job templates.

#### Key arguments

The argparser is the largest in the project (70+ flags across `get_args`, `get_model_args`, `get_path_args`, `get_wrapper_args`). The most commonly-edited flags:

| Arg | Type | Default | Description |
|---|---|---|---|
| `--config_path` | str | None | Path to JSON config. |
| `--run_id` | str | None | Unique identifier for this run. |
| `--primary_data_path` | str | `'None'` | Path to primary HDF5 training dataset. |
| `--secondary_data_paths` | str (n+) | None | One or more secondary HDF5 dataset paths. |
| `--training_strategy` | str | `auto` | One of `auto`, `primary_only`, `combine`. |
| `--start_secondary` | str | `'False'` | `'True'`/`'False'`. Phase transition: load primary weights, then train on combined data. |
| `--epochs` | int | 1 | Used in `primary_only` mode. |
| `--max_steps` | int | 100000 | Used in `combine` mode. |
| `--val_check_interval` | int | 10000 | Steps between validations (step-based mode). |
| `--limit_val_batches` | float | 200 | Cap val batches per check. See [Choosing limit_val_batches](stage3_training.md#choosing-limit_val_batches-and-limit_train_batches). |
| `--limit_train_batches` | float | None | Same convention as above. |
| `--batch_size` | int | 16 | Per-device batch size. |
| `--lr` | float | 3e-4 | Base learning rate. |
| `--scale_learning_rate` | str | `'True'` | Scale LR by world size. |
| `--precision` | str | `no` | One of `no`, `fp16`, `bf16`, `32`. |
| `--device` | str | `cuda` | One of `cpu`, `cuda`, `xpu`. |
| `--gpu_devices` | int | 1 | GPUs/tiles per node. |
| `--num_nodes` | int | 1 | Nodes. |
| `--resume_from_checkpoint` | str | `'None'` | Path to a Lightning `.ckpt` to resume from. |
| `--pretrained_weights` | str | `'None'` | Path to raw weights to load before training. |
| `--finetune` | str | `'False'` | `'True'`/`'False'`. Enable finetuning mode. |
| `--finetune_last_n_blocks` | int | -2 | -1 = all, 0 = none, N = last N blocks. |
| `--finetune_last_n_layers` | int | -2 | Same convention as blocks. |
| `--finetune_output_layers` | str | `"True"` | Whether to unfreeze the transformer output layers. |
| `--checkpoint_every_n_steps` / `--checkpoint_every_n_epochs` | int | None | Periodic snapshot cadence (orthogonal to best-metric saves). |
| `--checkpoint_monitors` | JSON | None | List of `{metric, mode}` dicts for multi-metric checkpointing. |
| `--early_stopping_metric` | str | None | Metric to monitor (`'val_loss'`, etc.). None disables. |
| `--save_metrics_history` | str | `'True'` | Save MetricsHistoryCallback JSONL. |
| `--wandb` | str | `'False'` | Enable wandb (requires `WANDB_API_KEY`). |
| `--output_root` | str | None | Base output directory. |

Run `biom3_pretrain_stage3 --help` for the full list, including model-architecture flags (`--diffusion_steps`, `--transformer_blocks`, etc.) and benchmark flags (`--save_benchmark`, `--benchmark_per_step`).

#### Example: pretrain from scratch

```bash
biom3_pretrain_stage3 \
    --config_path configs/stage3_training/pretrain_scratch_v2.json \
    --run_id my_run_001 \
    --epochs 100
```

#### Example: finetune

```bash
biom3_pretrain_stage3 \
    --config_path configs/stage3_training/finetune_v1.json \
    --run_id finetune_001 \
    --finetune True \
    --pretrained_weights /path/to/state_dict.best.pth \
    --finetune_last_n_blocks 4 \
    --finetune_last_n_layers -1 \
    --finetune_output_layers True
```

See [stage3_training.md](stage3_training.md) for resumption, secondary-data continuation, and per-machine submission examples.

---

## Wandb handling (training entrypoints)

All three training entrypoints accept `--wandb True|False` (stringified bool). When invoked through the HPC wrappers ([scripts/stage{1,2,3}_train_{single,multi}node.sh](../scripts/)), the resolution rules in [scripts/_wandb_resolve.sh](../scripts/_wandb_resolve.sh) apply:

| Input | Result |
|---|---|
| `--wandb False` | wandb OFF (always honored) |
| `--wandb True` + `WANDB_API_KEY` set | wandb ON |
| `--wandb True` + no `WANDB_API_KEY` | wrapper errors out before exec |
| no `--wandb` + `WANDB_API_KEY` set | defaults ON |
| no `--wandb` + no `WANDB_API_KEY` | defaults OFF (warns) |

In job templates, edit the `use_wandb=True` variable near `epochs=` to flip wandb per job.

---

## Inspecting argparser output

The most reliable, always-current reference is the entrypoint's own help:

```bash
biom3_PenCL_inference --help
biom3_Facilitator_sample --help
biom3_ProteoScribe_sample --help
biom3_embedding_pipeline --help
biom3_pretrain_stage1 --help
biom3_pretrain_stage2 --help
biom3_pretrain_stage3 --help
```

If any table in this document drifts from `--help` output, the argparser is the source of truth.

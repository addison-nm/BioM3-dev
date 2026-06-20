# Cluster-aware train/val/test splits

By default, Stage 3 training splits each input HDF5 file into 80% train / 20%
validation with a random per-row shuffle ([`HDF5DataModule.split_indices`](../src/biom3/Stage3/PL_wrapper.py)).
Random splitting can leak information: near-duplicate or homologous sequences
land on both sides of the split, inflating validation metrics.

`biom3_cluster_split` produces a **curated split** instead. It clusters
sequences by similarity (MMseqs2) and assigns **whole clusters** to train, val,
or test, so no cluster is ever split across two partitions. Stage 3 then
consumes the resulting *split manifest* in place of the random split.

## Pipeline

```
HDF5 input(s)
  → biom3_cluster_split   (mmseqs easy-cluster → pack whole clusters → manifest)
  → split_manifest.json
  → biom3_train_stage3 --split_manifest_path split_manifest.json
```

Clustering runs over the **pooled** sequences from all input files at once, so a
cluster that straddles the primary/secondary file boundary still lands entirely
in one split. Exact-duplicate sequences collapse into a single cluster (100%
identity) and therefore cannot leak either.

The **test** split is recorded in the manifest but **never loaded by the
training loop** — it is held out for downstream generation/evaluation.

## Building a manifest

```bash
biom3_cluster_split \
    --primary_data_path data/Stage2_MMD_swissprot_embedding.hdf5 \
    --facilitator MMD \
    --train_frac 0.8 --val_frac 0.1 --test_frac 0.1 \
    --min_seq_id 0.3 --coverage 0.8 \
    -o data/split_manifest.json
```

Requires the `mmseqs` executable on `PATH`; it raises if absent. A config file
may supply any argument via `--config_path` (CLI args override it); see
[`configs/split/cluster_split_v1.json`](../configs/split/cluster_split_v1.json).

Key options:

| Option | Default | Meaning |
| --- | --- | --- |
| `--train_frac` / `--val_frac` / `--test_frac` | 0.8 / 0.1 / 0.1 | Target split fractions. Set `--test_frac 0` for train/val only. |
| `--min_seq_id` | 0.3 | mmseqs `--min-seq-id` identity threshold. |
| `--coverage` | 0.8 | mmseqs `-c` alignment coverage. |
| `--cov_mode` / `--cluster_mode` | 0 / 0 | mmseqs coverage / clustering modes. |
| `--diffusion_steps` | 1024 | Derives the length filter (`min_seq_length = diffusion_steps - 2`), matching Stage 3, so reported ratios describe usable data. `--no_length_filter` disables it. |
| `--clusters_tsv` | — | Use a precomputed `rep<TAB>member` clustering and skip mmseqs entirely (see below). |

Outputs `split_manifest.json` plus `split_manifest.stats.md` (per-split counts
and achieved-vs-target ratios). Because clusters are packed whole, a single
cluster larger than a split's target will overshoot that ratio — the run logs a
warning when a split deviates from its target by more than 0.05.

## Training with a manifest

```bash
biom3_train_stage3 \
    --config_path configs/stage3_training/finetune_v1.json \
    --primary_data_path data/Stage2_MMD_swissprot_embedding.hdf5 \
    --split_manifest_path data/split_manifest.json
```

The DataModule validates each file against the manifest's row count and content
fingerprint and **fails loud** if the dataset changed since the manifest was
built — regenerate the manifest in that case. The resolved indices (including
the held-out test rows) are persisted to `artifacts/dataset_splits.pt`.

## Bring-your-own clustering

`--clusters_tsv` accepts any clustering in the mmseqs `createtsv` format
(`representative<TAB>member`, one pair per line). Member ids must follow the
`"<file_index>-<row>"` scheme, where `file_index` is the 0-based position in
`[primary, *secondary]` and `row` is the HDF5 row index. This lets you swap in
cd-hit or a precomputed clustering, and is what the test suite uses to exercise
the pipeline without the mmseqs binary.

## Module layout

```
src/biom3/split/
  pack.py        # greedy whole-cluster → split packing (leakage-free core)
  manifest.py    # manifest schema, read/write, fingerprint validation
  cluster.py     # mmseqs seam: run_mmseqs_easy_cluster(), parse_cluster_tsv()
  run_split.py   # end-to-end orchestration + CLI (biom3_cluster_split)
```

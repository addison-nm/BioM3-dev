"""Source-stratified, cluster-aware train/val/test split for JSONL records.

A second splitter (distinct from :mod:`biom3.split.run_split`, which reads
Stage-3 HDF5 files) for the generalized dataloader's cleaned JSONL records. It
clusters ALL sequences globally with MMseqs2 (or a precomputed ``--clusters_tsv``)
so any similar pair — within or across sources — lands in one cluster and hence
one split (no leakage), then packs whole clusters into train/val/test
*stratified by a per-record source field*, so every source (e.g. pfam,
swissprot, supplemental) is distributed ~train/val/test rather than one source
being lumped into a single split.

Records are read via :func:`biom3.core.dataloaders.read_jsonl_records`, so the
manifest's row indices and fingerprint align exactly with what
``GeneralizedRecordDataset`` sees at train time.

    biom3_stratified_cluster_split \\
        --data_path data/SH3_caption_fields_all.jsonl \\
        --source_key source \\
        --min_seq_id 0.5 \\
        --train_frac 0.7 --val_frac 0.2 --test_frac 0.1 \\
        -o data/SH3_split_manifest.json
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile

import numpy as np

from biom3.backend.device import setup_logger
from biom3.core.dataloaders import read_jsonl_records
from biom3.core.helpers import load_json_config
from biom3.split import cluster as clu
from biom3.split import manifest as mf
from biom3.split.pack import SPLITS
from biom3.split.pack_stratified import pack_clusters_stratified

logger = setup_logger(__name__)

RATIO_WARN_TOL = 0.05
_FILE_INDEX = 0  # single JSONL input


def _member_id(row):
    return f"{_FILE_INDEX}-{row}"


def build_stratified_split_manifest(
    *,
    data_path,
    sequence_key="sequence",
    source_key="source",
    length_field="sequence_length",
    ratios,
    seed=0,
    min_seq_id=0.3,
    coverage=0.8,
    cov_mode=0,
    cluster_mode=0,
    threads=None,
    mmseqs_extra=None,
    min_seq_length=None,
    clusters_tsv=None,
    tmp_dir=None,
):
    """Cluster JSONL sequences globally and pack them into a stratified manifest."""
    records = read_jsonl_records(data_path)
    n_rows = len(records)
    sequences = [str(rec[sequence_key]) for rec in records]

    def _row_length(rec, seq):
        length = rec.get(length_field)
        if length is None:
            length = len(seq.replace("-", ""))
        return int(length)

    member_source = {}   # (file_index, row) -> source
    eligible = {}        # member_id -> (file_index, row)
    fasta_records = []   # (member_id, sequence)
    for row, (rec, seq) in enumerate(zip(records, sequences)):
        if min_seq_length is not None and _row_length(rec, seq) > min_seq_length:
            continue
        mid = _member_id(row)
        key = (_FILE_INDEX, row)
        eligible[mid] = key
        member_source[key] = rec.get(source_key) or "unknown"
        fasta_records.append((mid, seq))

    if not eligible:
        raise ValueError("no eligible sequences to cluster (check path / length filter)")
    logger.info("Read %s (%d rows, %d eligible after length filter)",
                data_path, n_rows, len(eligible))

    if clusters_tsv is not None:
        logger.info("Using precomputed clustering from %s", clusters_tsv)
        raw_clusters = clu.parse_cluster_tsv(clusters_tsv)
    else:
        work_dir = tmp_dir or tempfile.mkdtemp(prefix="biom3_stratsplit_")
        os.makedirs(work_dir, exist_ok=True)
        fasta_path = os.path.join(work_dir, "pooled.fasta")
        clu.write_pooled_fasta(fasta_path, fasta_records)
        tsv = clu.run_mmseqs_easy_cluster(
            fasta_path, work_dir,
            min_seq_id=min_seq_id, coverage=coverage,
            cov_mode=cov_mode, cluster_mode=cluster_mode,
            threads=threads, extra_args=mmseqs_extra,
        )
        raw_clusters = clu.parse_cluster_tsv(tsv)

    clusters = []
    seen = set()
    unknown = 0
    for raw in raw_clusters:
        members = []
        for mid in raw:
            key = eligible.get(mid)
            if key is None:
                unknown += 1
                continue
            members.append(key)
            seen.add(mid)
        if members:
            clusters.append(members)
    if unknown:
        logger.warning("%d clustered members did not match any eligible row "
                       "and were ignored", unknown)

    missing = [mid for mid in eligible if mid not in seen]
    if missing:
        logger.warning("%d eligible rows were absent from the clustering; "
                       "treating each as a singleton cluster", len(missing))
        for mid in missing:
            clusters.append([eligible[mid]])

    result = pack_clusters_stratified(clusters, member_source, ratios, seed=seed)

    files_meta = [{
        "path": data_path,
        "group": None,
        "n_rows": n_rows,
        "fingerprint": mf.compute_fingerprint(sequences),
    }]
    tool = {
        "name": "precomputed" if clusters_tsv is not None else "mmseqs",
        "clustering": "global",
        "stratified_by": source_key,
        "min_seq_id": min_seq_id,
        "coverage": coverage,
        "cov_mode": cov_mode,
        "cluster_mode": cluster_mode,
        "min_seq_length": min_seq_length,
    }
    manifest = mf.build_manifest(
        files=files_meta, pack_result=result, ratios_target=ratios,
        seed=seed, n_clusters=len(clusters), tool=tool,
    )
    manifest["per_source_counts"] = result.per_source_counts
    manifest["per_source_achieved"] = result.per_source_achieved
    return manifest, result


def _format_stats(manifest, result):
    lines = ["# Stratified split manifest stats", ""]
    lines.append(f"- clusters: {manifest['n_clusters']}")
    lines.append(f"- seed: {manifest['seed']}")
    lines.append(f"- tool: {manifest['tool'].get('name')} "
                 f"(global clustering, stratified by {manifest['tool'].get('stratified_by')})")
    lines.append("")
    lines.append("## Overall")
    lines.append("| split | target | achieved | count |")
    lines.append("| --- | --- | --- | --- |")
    for s in SPLITS:
        if s in result.counts:
            lines.append(
                f"| {s} | {manifest['ratios_target'].get(s, 0):.3f} | "
                f"{manifest['ratios_achieved'][s]:.3f} | {result.counts[s]} |"
            )
    lines.append("")
    lines.append("## Per source (achieved fraction / count)")
    header = "| source | " + " | ".join(SPLITS) + " |"
    lines.append(header)
    lines.append("| --- | " + " | ".join("---" for _ in SPLITS) + " |")
    for src in sorted(result.per_source_achieved):
        cells = []
        for s in SPLITS:
            frac = result.per_source_achieved[src].get(s)
            cnt = result.per_source_counts[src].get(s)
            cells.append(f"{frac:.3f} ({cnt})" if frac is not None else "-")
        lines.append(f"| {src} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main(args):
    if not args.data_path or str(args.data_path).lower() == "none":
        raise ValueError("--data_path (JSONL) is required")

    ratios = {"train": args.train_frac, "val": args.val_frac, "test": args.test_frac}
    min_seq_length = None if args.no_length_filter else args.diffusion_steps - 2

    manifest, result = build_stratified_split_manifest(
        data_path=args.data_path,
        sequence_key=args.sequence_key,
        source_key=args.source_key,
        length_field=args.length_field,
        ratios=ratios,
        seed=args.seed,
        min_seq_id=args.min_seq_id,
        coverage=args.coverage,
        cov_mode=args.cov_mode,
        cluster_mode=args.cluster_mode,
        threads=args.threads,
        mmseqs_extra=args.mmseqs_extra,
        min_seq_length=min_seq_length,
        clusters_tsv=args.clusters_tsv,
        tmp_dir=args.tmp_dir,
    )

    out = args.output
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    mf.write_manifest(out, manifest)
    stats_path = os.path.splitext(out)[0] + ".stats.md"
    with open(stats_path, "w") as fh:
        fh.write(_format_stats(manifest, result))

    logger.info("Wrote stratified split manifest to %s", out)
    for s in SPLITS:
        if s in result.counts:
            target = manifest["ratios_target"].get(s, 0.0)
            achieved = manifest["ratios_achieved"][s]
            logger.info("  %-5s target=%.3f achieved=%.3f (n=%d)",
                        s, target, achieved, result.counts[s])
    for src in sorted(result.per_source_achieved):
        for s in SPLITS:
            if s in result.counts:
                target = manifest["ratios_target"].get(s, 0.0)
                achieved = result.per_source_achieved[src][s]
                if target > 0 and abs(achieved - target) > RATIO_WARN_TOL:
                    logger.warning("  source %s split %s deviates from target by "
                                   "%.3f (cluster larger than the source's split)",
                                   src, s, abs(achieved - target))


def parse_arguments(argv):
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config_path", "-c", type=str, default=None)
    pre_args, _ = pre_parser.parse_known_args(argv)

    parser = argparse.ArgumentParser(
        description="Source-stratified, cluster-aware train/val/test split for "
                    "JSONL records (global clustering + per-source packing).",
    )
    parser.add_argument("--config_path", "-c", type=str, default=None,
                        help="Path to JSON config file. CLI args override it.")
    parser.add_argument("--data_path", type=str, default=None,
                        help="JSONL of cleaned records to split.")
    parser.add_argument("--sequence_key", type=str, default="sequence",
                        help="Record key holding the protein sequence.")
    parser.add_argument("--source_key", type=str, default="source",
                        help="Record key holding the data source to stratify by.")
    parser.add_argument("--length_field", type=str, default="sequence_length",
                        help="Record key with the precomputed (ungapped) length; "
                             "computed from the sequence when absent.")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Output path for the split manifest JSON.")
    parser.add_argument("--train_frac", type=float, default=0.8)
    parser.add_argument("--val_frac", type=float, default=0.1)
    parser.add_argument("--test_frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0,
                        help="Deterministic packing seed (ties among equal-size clusters).")
    parser.add_argument("--min_seq_id", type=float, default=0.3,
                        help="mmseqs --min-seq-id (sequence identity threshold).")
    parser.add_argument("--coverage", type=float, default=0.8,
                        help="mmseqs -c (alignment coverage).")
    parser.add_argument("--cov_mode", type=int, default=0, help="mmseqs --cov-mode.")
    parser.add_argument("--cluster_mode", type=int, default=0, help="mmseqs --cluster-mode.")
    parser.add_argument("--threads", type=int, default=None, help="mmseqs --threads.")
    parser.add_argument("--mmseqs_extra", type=str, nargs="+", default=None,
                        help="Extra args passed verbatim to mmseqs easy-cluster.")
    parser.add_argument("--diffusion_steps", type=int, default=1024,
                        help="Length filter is min_seq_length = diffusion_steps - 2.")
    parser.add_argument("--no_length_filter", action="store_true",
                        help="Disable the sequence-length filter.")
    parser.add_argument("--clusters_tsv", type=str, default=None,
                        help="Precomputed rep<TAB>member TSV (member ids as "
                             "'0-<row>'). Skips mmseqs when provided.")
    parser.add_argument("--tmp_dir", type=str, default=None,
                        help="Working directory for FASTA/mmseqs scratch.")

    if pre_args.config_path is not None:
        json_config = load_json_config(pre_args.config_path)
        parser.set_defaults(**json_config)

    args = parser.parse_args(argv)
    if not args.output:
        parser.error("-o/--output is required")
    return args


if __name__ == "__main__":
    main(parse_arguments(sys.argv[1:]))

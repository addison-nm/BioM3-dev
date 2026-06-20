"""End-to-end cluster-aware train/val/test split for Stage 3 HDF5 inputs.

Pools the sequences from one or more Stage-3 HDF5 files, clusters them with
MMseqs2 (or a precomputed clustering supplied via ``--clusters_tsv``), packs
whole clusters into train/val/test, and writes a split manifest that Stage 3
training consumes in place of its random split.

Clustering runs over the *pooled* sequences from all files at once, so a cluster
that straddles the primary/secondary file boundary still lands entirely in one
split — no cross-file leakage. Exact-duplicate sequences collapse into a single
cluster (100% identity) and therefore cannot leak either.

    biom3_cluster_split \\
        --primary_data_path data/train.hdf5 \\
        --facilitator MMD \\
        --train_frac 0.8 --val_frac 0.1 --test_frac 0.1 \\
        --min_seq_id 0.3 \\
        -o data/split_manifest.json
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile

import h5py
import numpy as np

from biom3.backend.device import setup_logger
from biom3.core.helpers import load_json_config
from biom3.split import cluster as clu
from biom3.split import manifest as mf
from biom3.split.pack import SPLITS, pack_clusters

logger = setup_logger(__name__)

RATIO_WARN_TOL = 0.05


def _decode(seq):
    return seq.decode() if isinstance(seq, (bytes, bytearray)) else str(seq)


def _read_hdf5_file(path, group_name):
    """Read sequences and lengths from a Stage-3 HDF5 file."""
    with h5py.File(path, "r") as f:
        group = f[group_name]
        sequences = [_decode(s) for s in group["sequence"][:]]
        if "sequence_length" in group:
            lengths = np.asarray(group["sequence_length"][:])
        else:
            lengths = np.array([len(s) for s in sequences], dtype=np.int64)
    return sequences, lengths


def _member_id(file_index, row):
    return f"{file_index}-{row}"


def _parse_member_id(member_id):
    fi, row = member_id.split("-")
    return int(fi), int(row)


def build_split_manifest(
    *,
    file_paths,
    group_name,
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
    """Cluster pooled sequences and pack them into a split manifest dict.

    Args:
        file_paths: ordered list of HDF5 paths. The list order defines the
            ``file_index`` recorded in member keys and the manifest.
        group_name: HDF5 group to read (e.g. ``"MMD_data"``).
        ratios: mapping of split name -> target fraction.
        min_seq_length: if set, rows whose ``sequence_length`` exceeds this are
            excluded from every split (mirrors Stage 3's length filter so the
            reported ratios describe data that will actually be used).
        clusters_tsv: optional path to a precomputed ``rep<TAB>member`` TSV whose
            member ids follow the ``"{file_index}-{row}"`` scheme. When given,
            MMseqs2 is not invoked.
    """
    files_meta = []
    eligible = {}  # member_id -> (file_index, row)
    records = []   # (member_id, sequence) for the pooled FASTA

    for fi, path in enumerate(file_paths):
        sequences, lengths = _read_hdf5_file(path, group_name)
        n_rows = len(sequences)
        files_meta.append({
            "path": path,
            "group": group_name,
            "n_rows": n_rows,
            "fingerprint": mf.compute_fingerprint(sequences),
        })
        for row in range(n_rows):
            if min_seq_length is not None and lengths[row] > min_seq_length:
                continue
            mid = _member_id(fi, row)
            eligible[mid] = (fi, row)
            records.append((mid, sequences[row]))
        logger.info("Read %s (%d rows, %d eligible after length filter)",
                    path, n_rows, sum(1 for k in eligible if k.startswith(f"{fi}-")))

    if not eligible:
        raise ValueError("no eligible sequences to cluster (check paths / length filter)")

    # Obtain clustering as a list of member-id strings per cluster.
    if clusters_tsv is not None:
        logger.info("Using precomputed clustering from %s", clusters_tsv)
        raw_clusters = clu.parse_cluster_tsv(clusters_tsv)
    else:
        work_dir = tmp_dir or tempfile.mkdtemp(prefix="biom3_split_")
        os.makedirs(work_dir, exist_ok=True)
        fasta_path = os.path.join(work_dir, "pooled.fasta")
        clu.write_pooled_fasta(fasta_path, records)
        tsv = clu.run_mmseqs_easy_cluster(
            fasta_path, work_dir,
            min_seq_id=min_seq_id, coverage=coverage,
            cov_mode=cov_mode, cluster_mode=cluster_mode,
            threads=threads, extra_args=mmseqs_extra,
        )
        raw_clusters = clu.parse_cluster_tsv(tsv)

    # Map member ids back to (file_index, row), dropping anything not eligible.
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

    # Any eligible row absent from the clustering becomes its own singleton, so
    # nothing is silently dropped from the split.
    missing = [mid for mid in eligible if mid not in seen]
    if missing:
        logger.warning("%d eligible rows were absent from the clustering; "
                       "treating each as a singleton cluster", len(missing))
        for mid in missing:
            clusters.append([eligible[mid]])

    result = pack_clusters(clusters, ratios, seed=seed)

    tool = {
        "name": "precomputed" if clusters_tsv is not None else "mmseqs",
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
    return manifest, result


def _format_stats(manifest, result):
    lines = ["# Split manifest stats", ""]
    lines.append(f"- clusters: {manifest['n_clusters']}")
    lines.append(f"- seed: {manifest['seed']}")
    lines.append(f"- tool: {manifest['tool'].get('name')}")
    lines.append("")
    lines.append("| split | target | achieved | count |")
    lines.append("| --- | --- | --- | --- |")
    for s in SPLITS:
        if s in result.counts:
            lines.append(
                f"| {s} | {manifest['ratios_target'].get(s, 0):.3f} | "
                f"{manifest['ratios_achieved'][s]:.3f} | {result.counts[s]} |"
            )
    return "\n".join(lines) + "\n"


def main(args):
    primary = getattr(args, "primary_data_path", None)
    if not primary or str(primary).lower() == "none":
        raise ValueError("--primary_data_path is required")
    file_paths = [primary]
    if args.secondary_data_paths:
        file_paths.extend(args.secondary_data_paths)

    ratios = {"train": args.train_frac, "val": args.val_frac, "test": args.test_frac}
    min_seq_length = None if args.no_length_filter else args.diffusion_steps - 2

    manifest, result = build_split_manifest(
        file_paths=file_paths,
        group_name=args.facilitator + "_data",
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

    logger.info("Wrote split manifest to %s", out)
    for s in SPLITS:
        if s in result.counts:
            target = manifest["ratios_target"].get(s, 0.0)
            achieved = manifest["ratios_achieved"][s]
            logger.info("  %-5s target=%.3f achieved=%.3f (n=%d)",
                        s, target, achieved, result.counts[s])
            if target > 0 and abs(achieved - target) > RATIO_WARN_TOL:
                logger.warning("  %s ratio deviates from target by %.3f "
                               "(likely a cluster larger than the target split)",
                               s, abs(achieved - target))


def parse_arguments(argv):
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config_path", "-c", type=str, default=None)
    pre_args, _ = pre_parser.parse_known_args(argv)

    parser = argparse.ArgumentParser(
        description="Cluster-aware train/val/test split for Stage 3 HDF5 inputs.",
    )
    parser.add_argument("--config_path", "-c", type=str, default=None,
                        help="Path to JSON config file. CLI args override it.")
    parser.add_argument("--primary_data_path", type=str, default=None,
                        help="Primary Stage-3 HDF5 dataset.")
    parser.add_argument("--secondary_data_paths", type=str, nargs="+", default=None,
                        help="Additional HDF5 datasets pooled into the same clustering.")
    parser.add_argument("--facilitator", type=str, default="MMD",
                        help="Facilitator name; HDF5 group is '<facilitator>_data'.")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Output path for the split manifest JSON.")
    parser.add_argument("--train_frac", type=float, default=0.8)
    parser.add_argument("--val_frac", type=float, default=0.1)
    parser.add_argument("--test_frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0,
                        help="Deterministic packing seed (only breaks ties among "
                             "equal-size clusters).")
    parser.add_argument("--min_seq_id", type=float, default=0.3,
                        help="mmseqs --min-seq-id (sequence identity threshold).")
    parser.add_argument("--coverage", type=float, default=0.8,
                        help="mmseqs -c (alignment coverage).")
    parser.add_argument("--cov_mode", type=int, default=0,
                        help="mmseqs --cov-mode.")
    parser.add_argument("--cluster_mode", type=int, default=0,
                        help="mmseqs --cluster-mode.")
    parser.add_argument("--threads", type=int, default=None,
                        help="mmseqs --threads.")
    parser.add_argument("--mmseqs_extra", type=str, nargs="+", default=None,
                        help="Extra args passed verbatim to mmseqs easy-cluster.")
    parser.add_argument("--diffusion_steps", type=int, default=1024,
                        help="Used to derive the length filter (min_seq_length = "
                             "diffusion_steps - 2), matching Stage 3.")
    parser.add_argument("--no_length_filter", action="store_true",
                        help="Disable the sequence-length filter.")
    parser.add_argument("--clusters_tsv", type=str, default=None,
                        help="Precomputed rep<TAB>member TSV (member ids as "
                             "'<file_index>-<row>'). Skips mmseqs when provided.")
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

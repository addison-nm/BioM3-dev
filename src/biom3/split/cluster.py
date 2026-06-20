"""MMseqs2 clustering seam.

Isolates the only external-binary dependency. The rest of the pipeline talks to
clustering exclusively through a list-of-clusters representation, so any tool
that can emit an mmseqs-style ``rep<TAB>member`` TSV can be substituted via
:func:`parse_cluster_tsv` (e.g. the ``--clusters_tsv`` override).
"""

from __future__ import annotations

import os
import shutil
import subprocess

from biom3.backend.device import setup_logger

logger = setup_logger(__name__)


def write_pooled_fasta(path, records):
    """Write a FASTA file from ``(id, sequence)`` pairs.

    The id encodes the originating ``(file_index, row)`` so cluster membership
    can be mapped back to specific HDF5 rows.
    """
    with open(path, "w") as fh:
        for rec_id, seq in records:
            fh.write(f">{rec_id}\n{seq}\n")


def parse_cluster_tsv(path):
    """Parse an mmseqs ``*_cluster.tsv`` into a list of clusters.

    Each line is ``representative<TAB>member``; the representative itself also
    appears as one of its members. Returns a list of clusters, each a list of
    member-id strings, with cluster order following first appearance of each
    representative.
    """
    groups = {}
    order = []
    with open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                raise ValueError(f"malformed cluster TSV line (expected rep<TAB>member): {line!r}")
            rep, member = parts[0], parts[1]
            if rep not in groups:
                groups[rep] = []
                order.append(rep)
            groups[rep].append(member)
    return [groups[rep] for rep in order]


def run_mmseqs_easy_cluster(
    fasta_path, tmp_dir, *,
    min_seq_id=0.3,
    coverage=0.8,
    cov_mode=0,
    cluster_mode=0,
    threads=None,
    extra_args=None,
):
    """Run ``mmseqs easy-cluster`` and return the path to its cluster TSV.

    Raises:
        RuntimeError: if the ``mmseqs`` binary is not on PATH or no cluster TSV
            is produced.
    """
    exe = shutil.which("mmseqs")
    if exe is None:
        raise RuntimeError(
            "mmseqs not found on PATH. Install MMseqs2 and ensure the 'mmseqs' "
            "executable is available, or pass a precomputed clustering via "
            "--clusters_tsv."
        )

    prefix = os.path.join(tmp_dir, "clu")
    cmd = [
        exe, "easy-cluster", fasta_path, prefix, tmp_dir,
        "--min-seq-id", str(min_seq_id),
        "-c", str(coverage),
        "--cov-mode", str(cov_mode),
        "--cluster-mode", str(cluster_mode),
    ]
    if threads is not None:
        cmd += ["--threads", str(threads)]
    if extra_args:
        cmd += list(extra_args)

    logger.info("Running: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)

    tsv = prefix + "_cluster.tsv"
    if not os.path.exists(tsv):
        raise RuntimeError(f"mmseqs did not produce expected cluster TSV at {tsv}")
    return tsv

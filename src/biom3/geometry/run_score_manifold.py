"""Score query embeddings against a fitted latent manifold.

Reads an ``(N, D)`` matrix of query vectors and a manifold written by
:mod:`biom3.geometry.run_fit_manifold`, and writes one score per query. Lower
means closer to the manifold; the fitting method is read from the manifold file.

    biom3_score_manifold \\
        --manifold manifold_sho1.npz \\
        --queries zp_designs.npy \\
        --ids design_ids.txt \\
        -o scores.csv

The output is a CSV (``id``, ``score``, ``band``) unless ``-o`` ends in
``.npy``, in which case the raw ``(N,)`` score array is written.
"""

from __future__ import annotations

import argparse
import csv
import sys

import numpy as np

from biom3.backend.device import setup_logger
from biom3.geometry.io import load_manifold, load_vectors
from biom3.geometry.gaussian import band_position
from biom3.geometry.manifold import manifold_score

logger = setup_logger(__name__)


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(
        description="Score query embeddings against a fitted latent manifold.",
    )
    parser.add_argument(
        "--manifold", type=str, required=True,
        help="path to a manifold written by biom3_fit_manifold (.npz)",
    )
    parser.add_argument(
        "--queries", type=str, required=True,
        help="path to an (N, D) query matrix (.npy or .npz)",
    )
    parser.add_argument(
        "--queries_key", type=str, default=None,
        help="array name to read when --queries is a multi-array .npz",
    )
    parser.add_argument(
        "--ids", type=str, default=None,
        help="optional text file of query identifiers, one per line, in row order",
    )
    parser.add_argument(
        "--no_norm_check", action="store_true",
        help="skip the query-vs-reference mean-norm sanity check (only for "
             "methods that perform one)",
    )
    parser.add_argument(
        "-o", "--output", type=str, required=True,
        help="path to write scores (.csv, or .npy for the raw array)",
    )
    return parser.parse_args(argv)


def _read_ids(path, n_expected):
    with open(path) as fh:
        ids = [line.strip() for line in fh if line.strip()]
    if len(ids) != n_expected:
        raise ValueError(
            f"{path} holds {len(ids)} ids but there are {n_expected} query vectors"
        )
    return ids


def _write_csv(path, ids, scores, positions):
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["id", "score"] + (["band"] if positions is not None else []))
        for i, (name, score) in enumerate(zip(ids, scores)):
            row = [name, f"{score:.6f}"]
            if positions is not None:
                row.append(positions[i])
            writer.writerow(row)


def main(args):
    manifold = load_manifold(args.manifold)
    queries = load_vectors(args.queries, key=args.queries_key)
    logger.info("loaded queries %s with shape %s", args.queries, queries.shape)
    logger.info("scoring against %s", manifold)

    score_kwargs = {"check_norm": False} if args.no_norm_check else {}
    scores = manifold_score(queries, manifold, **score_kwargs)

    band = getattr(manifold, "band", None)
    positions = band_position(scores, manifold) if band is not None else None

    if args.output.endswith(".npy"):
        np.save(args.output, scores)
    else:
        ids = (
            _read_ids(args.ids, len(scores))
            if args.ids
            else [str(i) for i in range(len(scores))]
        )
        _write_csv(args.output, ids, scores, positions)

    if positions is None:
        logger.info(
            "%d queries: median score %.4f",
            len(scores), float(np.median(scores)),
        )
    else:
        logger.info(
            "%d queries: median %.4f (reference band p5 %.4f / median %.4f / "
            "p95 %.4f), %d within band, %d below, %d above",
            len(scores), float(np.median(scores)),
            band["p5"], band["median"], band["p95"],
            int(np.count_nonzero(positions == "within")),
            int(np.count_nonzero(positions == "below")),
            int(np.count_nonzero(positions == "above")),
        )
    logger.info("wrote %s", args.output)
    return scores


if __name__ == "__main__":
    main(parse_arguments(sys.argv[1:]))

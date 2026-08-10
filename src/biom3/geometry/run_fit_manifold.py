"""Fit a latent manifold to a reference set of embeddings.

Reads an ``(M, D)`` matrix of reference vectors, fits a manifold to them, and
writes the result as a single ``.npz`` that
:mod:`biom3.geometry.run_score_manifold` scores against. The file records which
method produced it, so scoring never has to be told.

    biom3_fit_manifold \\
        --reference zp_naturals.npy \\
        --label "PenCL z_p run1_trackC_step187000 / Sho1 naturals" \\
        -o manifold_sho1.npz

The reference vectors must come from the same embedding path as the queries
that will later be scored -- the fit records their mean L2 norm so that a
mismatch is at least warned about at score time.
"""

from __future__ import annotations

import argparse
import sys

from biom3.backend.device import setup_logger
from biom3.geometry.base import available_methods
from biom3.geometry.io import load_vectors, save_manifold
from biom3.geometry.manifold import DEFAULT_METHOD, fit_manifold

logger = setup_logger(__name__)


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(
        description="Fit a latent manifold to a reference set of embeddings.",
    )
    parser.add_argument(
        "--reference", type=str, required=True,
        help="path to an (M, D) reference matrix (.npy or .npz)",
    )
    parser.add_argument(
        "--reference_key", type=str, default=None,
        help="array name to read when --reference is a multi-array .npz",
    )
    parser.add_argument(
        "--method", type=str, default=DEFAULT_METHOD, choices=available_methods(),
        help="manifold fitting method",
    )
    parser.add_argument(
        "--label", type=str, default="",
        help="free-form provenance string stored on the fitted manifold "
             "(embedding checkpoint, reference set name, ...)",
    )
    parser.add_argument(
        "-o", "--output", type=str, required=True,
        help="path to write the fitted manifold (.npz)",
    )
    return parser.parse_args(argv)


def main(args):
    reference = load_vectors(args.reference, key=args.reference_key)
    logger.info(
        "loaded reference %s with shape %s", args.reference, reference.shape
    )

    manifold = fit_manifold(reference, method=args.method, label=args.label)

    output = args.output
    if not output.endswith(".npz"):
        output += ".npz"
    save_manifold(output, manifold)

    logger.info("fitted manifold: %s", manifold)
    if len(reference) < manifold.n_dim:
        logger.info(
            "reference set is smaller than the dimension (%d < %d); a covariance "
            "over it is singular and only shrinkage makes the inverse defined",
            len(reference), manifold.n_dim,
        )
    logger.info("wrote %s", output)
    return manifold


if __name__ == "__main__":
    main(parse_arguments(sys.argv[1:]))

#!/usr/bin/env python3
"""Build a synthetic train/test TSV for the GRPO surrogate-in-the-loop demo.

Subsamples ``--n`` rows from a BioM3 prompt CSV (default
``data/datasets/SH3/FINAL_SH3_all_dataset_with_prompts.csv``), computes a
closed-form ``functional_score`` from each row's sequence, splits into
train and test, and writes two TSV files. The synthetic score stands in
for a wet-lab functional assay so we can validate the
surrogate-regressor + GRPO pipeline end-to-end without lab data.

Schema in / out
---------------

Input columns (BioM3 prompt CSV, header required):
    primary_Accession, protein_sequence, [final]text_caption, pfam_label

Output columns (TSV):
    primary_Accession, protein_sequence, [final]text_caption,
    functional_score

The ``functional_score`` is ``AAFractionReward(target_aa, target_fraction)``
applied to ``protein_sequence``, plus optional Gaussian noise (σ controlled
by ``--noise_std``) to make the surrogate's regression task non-trivial.

Example
-------
    python scripts/make_grpo_synthetic_eval.py \\
        --input data/datasets/SH3/FINAL_SH3_all_dataset_with_prompts.csv \\
        --output_dir data/datasets/SH3/ \\
        --n 2000 --n_test 200 --seed 42
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

from biom3.rl.rewards import AAFractionReward


_REQUIRED_COLS = ("primary_Accession", "protein_sequence", "[final]text_caption")


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--input", required=True,
                   help="Path to source CSV (BioM3 prompt CSV).")
    p.add_argument("--output_dir", required=True,
                   help="Directory to write synthetic_train.tsv + synthetic_test.tsv.")
    p.add_argument("--n", type=int, default=2000,
                   help="Total rows to subsample (train + test).")
    p.add_argument("--n_test", type=int, default=200,
                   help="Test rows held out from the subsample.")
    p.add_argument("--seed", type=int, default=42)

    # Synthetic ground-truth shape
    p.add_argument("--target_aa", default="A", help="AA whose fraction defines the score.")
    p.add_argument("--target_fraction", type=float, default=0.4,
                   help="Peak score at this fraction of target_aa.")
    p.add_argument("--scale", type=float, default=100.0,
                   help="Score scale (peak value).")
    p.add_argument("--noise_std", type=float, default=2.0,
                   help="Gaussian noise σ added to the score (on the same scale).")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    df = pd.read_csv(args.input)
    missing = [c for c in _REQUIRED_COLS if c not in df.columns]
    if missing:
        raise KeyError(
            f"Input CSV missing required columns {missing}; got {list(df.columns)}"
        )
    total = len(df)
    print(f"[info] Loaded {total} rows from {args.input}", flush=True)
    if args.n > total:
        raise ValueError(f"--n {args.n} > available {total}")
    if args.n_test >= args.n:
        raise ValueError(f"--n_test {args.n_test} must be < --n {args.n}")

    idx = rng.choice(total, size=args.n, replace=False)
    sub = df.iloc[idx].reset_index(drop=True)[list(_REQUIRED_COLS)]

    reward = AAFractionReward(
        target_aa=args.target_aa,
        target_fraction=args.target_fraction,
        scale=args.scale,
    )
    base_scores = np.array(reward(sub["protein_sequence"].tolist()), dtype=np.float64)
    noise = rng.normal(0.0, args.noise_std, size=base_scores.shape)
    scores = base_scores + noise
    sub["functional_score"] = scores

    # Length stats for the build log
    lens = sub["protein_sequence"].str.len()
    print(
        f"[info] Sampled {args.n} rows | seq_len mean={lens.mean():.1f} "
        f"min={lens.min()} max={lens.max()} | "
        f"score (pre-noise) mean={base_scores.mean():.2f} std={base_scores.std():.2f}",
        flush=True,
    )

    # Test split: take the LAST n_test rows after a deterministic shuffle
    perm = rng.permutation(len(sub))
    sub = sub.iloc[perm].reset_index(drop=True)
    test = sub.iloc[: args.n_test]
    train = sub.iloc[args.n_test:]

    train_path = os.path.join(args.output_dir, "synthetic_train.tsv")
    test_path = os.path.join(args.output_dir, "synthetic_test.tsv")
    train.to_csv(train_path, sep="\t", index=False)
    test.to_csv(test_path, sep="\t", index=False)
    print(f"[info] Wrote {len(train)} train rows -> {train_path}", flush=True)
    print(f"[info] Wrote {len(test)} test  rows -> {test_path}", flush=True)

    log_path = os.path.join(args.output_dir, "synthetic_build.log")
    with open(log_path, "w") as f:
        f.write(
            f"source={args.input}\n"
            f"n_total_source={total}\n"
            f"n_sampled={args.n}\n"
            f"n_test={args.n_test}\n"
            f"seed={args.seed}\n"
            f"target_aa={args.target_aa}\n"
            f"target_fraction={args.target_fraction}\n"
            f"scale={args.scale}\n"
            f"noise_std={args.noise_std}\n"
            f"score_mean_before_noise={base_scores.mean():.4f}\n"
            f"score_std_before_noise={base_scores.std():.4f}\n"
        )
    print(f"[info] Build log -> {log_path}", flush=True)


if __name__ == "__main__":
    sys.exit(main())

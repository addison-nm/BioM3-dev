#!/usr/bin/env python3
"""Sample a small Stage 3 benchmark HDF5 from the full training file.

Used to produce the tiny datasets consumed by
``biom3_benchmark_stage3_training`` so each sweep cell has fast,
deterministic epochs.

The source file is one of the Stage 2 MMD embedding outputs (e.g.
``Stage2_MMD_swissprot_embedding_last_ckpt_all.hdf5``) containing a
group ``{facilitator}_data`` with datasets ``sequence``,
``text_to_protein_embedding``, and (optionally) ``sequence_length``.

Example
-------
    python scripts/make_stage3_benchmark_dataset.py \\
        --input  data/Stage2_MMD_swissprot_embedding_last_ckpt_all.hdf5 \\
        --output data/benchmark/Stage2_MMD_swissprot_embedding_bench_4096.hdf5 \\
        --n 4096 --seed 0
"""
import argparse

import h5py
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--n", type=int, default=4096,
                   help="Number of sequences to sample (default: 4096)")
    p.add_argument("--group", default="MMD_data",
                   help="HDF5 group name (default: MMD_data)")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    with h5py.File(args.input, "r") as fin:
        if args.group not in fin:
            raise KeyError(
                f"group '{args.group}' not in {args.input}; "
                f"available: {list(fin.keys())}"
            )
        gin = fin[args.group]
        total = len(gin["sequence"])
        if args.n > total:
            raise ValueError(f"--n {args.n} > available {total}")
        idx = np.sort(rng.choice(total, size=args.n, replace=False))

        with h5py.File(args.output, "w") as fout:
            gout = fout.create_group(args.group)
            gout.create_dataset("sequence", data=gin["sequence"][idx])
            gout.create_dataset(
                "text_to_protein_embedding",
                data=gin["text_to_protein_embedding"][idx],
            )
            if "sequence_length" in gin:
                gout.create_dataset(
                    "sequence_length", data=gin["sequence_length"][idx],
                )
            gout.attrs["sampled_from"] = args.input
            gout.attrs["source_total"] = total
            gout.attrs["sample_size"] = args.n
            gout.attrs["sample_seed"] = args.seed

    print(f"Wrote {args.n} sequences (from {total}) to {args.output}")


if __name__ == "__main__":
    main()

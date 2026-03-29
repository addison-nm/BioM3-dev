"""
End-to-end embedding pipeline: CSV → Stage 1 → Stage 2 → compiled HDF5.

Python replacement for scripts/embedding_pipeline.sh. Runs PenCL inference,
Facilitator sampling, and HDF5 compilation in sequence, constructing the
intermediate file paths automatically from --output_dir and --prefix.
"""

import argparse
import os
import sys
from argparse import Namespace


def parse_arguments(args):
    parser = argparse.ArgumentParser(
        description="BioM3 Embedding Pipeline (Stage 1 → Stage 2 → HDF5)"
    )
    # Required paths
    parser.add_argument(
        "-i", "--input_data_path", type=str, required=True,
        help="Path to input CSV (sequences + text prompts)"
    )
    parser.add_argument(
        "-o", "--output_dir", type=str, required=True,
        help="Directory for all output files"
    )
    parser.add_argument(
        "--pencl_weights", type=str, required=True,
        help="Path to PenCL model weights or checkpoint"
    )
    parser.add_argument(
        "--facilitator_weights", type=str, required=True,
        help="Path to Facilitator model weights or checkpoint"
    )
    parser.add_argument(
        "--pencl_config", type=str, required=True,
        help="Path to Stage 1 JSON config (stage1_config_PenCL_inference.json)"
    )
    parser.add_argument(
        "--facilitator_config", type=str, required=True,
        help="Path to Stage 2 JSON config (stage2_config_Facilitator_sample.json)"
    )
    parser.add_argument(
        "--prefix", type=str, required=True,
        help="Filename prefix for intermediate and final output files"
    )

    # Optional overrides
    parser.add_argument(
        "--device", type=str, default="cuda",
        choices=["cpu", "cuda", "xpu"],
        help="Device for inference (default: cuda)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=256,
        help="Batch size for Stage 1 PenCL inference (default: 256)"
    )
    parser.add_argument(
        "--num_workers", type=int, default=0,
        help="Number of dataloader workers for Stage 1 (default: 0)"
    )
    parser.add_argument(
        "--mmd_sample_limit", type=int, default=1000,
        help="Sample limit for MMD computation in Stage 2 (default: 1000)"
    )
    parser.add_argument(
        "--dataset_key", type=str, default="MMD_data",
        help="HDF5 group name for compiled output (default: MMD_data)"
    )
    return parser.parse_args(args)


def main(args):
    from biom3.Stage1.run_PenCL_inference import (
        parse_arguments as parse_stage1_args,
        main as run_stage1,
    )
    from biom3.Stage2.run_Facilitator_sample import (
        parse_arguments as parse_stage2_args,
        main as run_stage2,
    )
    from biom3.data_prep.compile_stage2_data_to_hdf5 import (
        parse_arguments as parse_compile_args,
        main as run_compile,
    )

    os.makedirs(args.output_dir, exist_ok=True)

    # Intermediate file paths
    pencl_output = os.path.join(args.output_dir, f"{args.prefix}.PenCL_emb.pt")
    facilitator_output = os.path.join(args.output_dir, f"{args.prefix}.Facilitator_emb.pt")
    hdf5_output = os.path.join(args.output_dir, f"{args.prefix}.compiled_emb.hdf5")

    # --- Stage 1: PenCL inference ---
    print("=" * 60)
    print("Stage 1: PenCL inference")
    print("=" * 60)
    stage1_args = parse_stage1_args([
        "-i", args.input_data_path,
        "-c", args.pencl_config,
        "-m", args.pencl_weights,
        "-o", pencl_output,
        "--device", args.device,
        "--batch_size", str(args.batch_size),
        "--num_workers", str(args.num_workers),
    ])
    run_stage1(stage1_args)

    # --- Stage 2: Facilitator sampling ---
    print("=" * 60)
    print("Stage 2: Facilitator sampling")
    print("=" * 60)
    stage2_args = parse_stage2_args([
        "-i", pencl_output,
        "-c", args.facilitator_config,
        "-m", args.facilitator_weights,
        "-o", facilitator_output,
        "--device", args.device,
        "--mmd_sample_limit", str(args.mmd_sample_limit),
    ])
    run_stage2(stage2_args)

    # --- Compile to HDF5 ---
    print("=" * 60)
    print("Compiling Stage 2 output to HDF5")
    print("=" * 60)
    compile_args = parse_compile_args([
        "-i", facilitator_output,
        "-o", hdf5_output,
        "--dataset_key", args.dataset_key,
    ])
    run_compile(compile_args)

    print("=" * 60)
    print(f"Pipeline complete. Output: {hdf5_output}")
    print("=" * 60)

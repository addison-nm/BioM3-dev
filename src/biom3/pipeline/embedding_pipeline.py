"""
End-to-end embedding pipeline: CSV → Stage 1 → Stage 2 → compiled HDF5.

Python replacement for scripts/embedding_pipeline.sh. Runs PenCL inference,
Facilitator sampling, and HDF5 compilation in sequence, constructing the
intermediate file paths automatically from --output_dir and --prefix.

With --generate the terminal step is ProteoScribe sampling instead of HDF5
compilation, giving CSV → Stage 1 → Stage 2 → Stage 3 generated sequences.
Multi-GPU/multi-node sampling comes from running this entrypoint under
scripts/launchers/container_{single,multi}node.sh: stages 1-2 are
deterministic, so every rank computes identical embeddings, and the Stage 3
sampler shards by rank with only rank 0 writing.
"""

import argparse
import os
import sys
from argparse import Namespace
from datetime import datetime

from biom3.backend.device import setup_logger
from biom3.core.helpers import load_json_config
from biom3.core.run_utils import (
    get_biom3_version,
    get_git_hash,
    setup_file_logging,
    teardown_file_logging,
    write_manifest,
)

logger = setup_logger(__name__)


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
        "--weight_set", type=str, default=None,
        help="Path to a weight-set bundle JSON (e.g. configs/weights/run1_base.json) "
             "providing pencl/facilitator weights. Explicit --*_weights override it."
    )
    parser.add_argument(
        "--pencl_weights", type=str, default=None,
        help="Path to PenCL model weights or checkpoint (overrides --weight_set)"
    )
    parser.add_argument(
        "--facilitator_weights", type=str, default=None,
        help="Path to Facilitator model weights or checkpoint (overrides --weight_set)"
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

    # Stage 3 generation (terminal step instead of HDF5 compilation)
    parser.add_argument(
        "--generate", action="store_true", default=False,
        help="Run Stage 3 ProteoScribe sampling after Stage 2 instead of "
             "compiling to HDF5"
    )
    parser.add_argument(
        "--proteoscribe_weights", type=str, default=None,
        help="Path to ProteoScribe weights or checkpoint (overrides --weight_set); "
             "required with --generate"
    )
    parser.add_argument(
        "--proteoscribe_config", type=str, default=None,
        help="Path to Stage 3 JSON config; required with --generate"
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Stage 3 sampling seed (default: 0)"
    )
    parser.add_argument(
        "--no_fasta", dest="fasta", action="store_false", default=True,
        help="Do not write generated sequences as FASTA (Stage 3; FASTA is on by default)"
    )
    parser.add_argument(
        "--token_strategy", type=str, default=None,
        help="Stage 3 token strategy (passed through to the sampler)"
    )
    parser.add_argument(
        "--unmasking_order", type=str, default=None,
        help="Stage 3 unmasking order (passed through to the sampler)"
    )
    parsed = parser.parse_args(args)

    weight_keys = ["pencl_weights", "facilitator_weights"]
    if parsed.generate:
        weight_keys.append("proteoscribe_weights")

    from biom3.core.weight_sets import merge_weight_set
    merge_weight_set(parsed, parsed.weight_set, keys=tuple(weight_keys))
    missing = [k for k in weight_keys if not getattr(parsed, k)]
    if missing:
        parser.error(
            "missing weight path(s): " + ", ".join("--" + m for m in missing)
            + " (provide them directly or via --weight_set)"
        )
    if parsed.generate and not parsed.proteoscribe_config:
        parser.error("--proteoscribe_config is required with --generate")
    return parsed


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

    # Set up dual logging (console + file)
    log_path, file_handler = setup_file_logging(args.output_dir)
    start_time = datetime.now()
    logger.info("=" * 60)
    logger.info(
        "Embedding pipeline (Stage 1 -> Stage 2 -> %s)",
        "Stage 3" if args.generate else "HDF5",
    )
    logger.info("biom3 version: %s (git: %s)", get_biom3_version(), get_git_hash())
    logger.info("Command:     %s", " ".join(sys.argv))
    logger.info("=" * 60)

    # Load config contents for manifest
    pencl_config_contents = load_json_config(args.pencl_config)
    facilitator_config_contents = load_json_config(args.facilitator_config)

    # Intermediate file paths
    pencl_output = os.path.join(args.output_dir, f"{args.prefix}.PenCL_emb.pt")
    facilitator_output = os.path.join(args.output_dir, f"{args.prefix}.Facilitator_emb.pt")
    hdf5_output = os.path.join(args.output_dir, f"{args.prefix}.compiled_emb.hdf5")
    generated_output = os.path.join(args.output_dir, f"{args.prefix}.generated.pt")

    # --- Stage 1: PenCL inference ---
    logger.info("=" * 60)
    logger.info("Stage 1: PenCL inference")
    logger.info("=" * 60)
    stage1_args = parse_stage1_args([
        "-i", args.input_data_path,
        "-c", args.pencl_config,
        "-m", args.pencl_weights,
        "-o", pencl_output,
        "--device", args.device,
        "--batch_size", str(args.batch_size),
        "--num_workers", str(args.num_workers),
    ])
    run_stage1(stage1_args, _setup_logging=False)

    # --- Stage 2: Facilitator sampling ---
    logger.info("=" * 60)
    logger.info("Stage 2: Facilitator sampling")
    logger.info("=" * 60)
    stage2_args = parse_stage2_args([
        "-i", pencl_output,
        "-c", args.facilitator_config,
        "-m", args.facilitator_weights,
        "-o", facilitator_output,
        "--device", args.device,
        "--mmd_sample_limit", str(args.mmd_sample_limit),
    ])
    run_stage2(stage2_args, _setup_logging=False)

    if args.generate:
        # --- Stage 3: ProteoScribe sampling ---
        from biom3.Stage3.run_ProteoScribe_sample import (
            parse_arguments as parse_stage3_args,
            main as run_stage3,
        )

        logger.info("=" * 60)
        logger.info("Stage 3: ProteoScribe sampling")
        logger.info("=" * 60)
        stage3_argv = [
            "-i", facilitator_output,
            "-c", args.proteoscribe_config,
            "-m", args.proteoscribe_weights,
            "-o", generated_output,
            "--device", args.device,
            "--seed", str(args.seed),
        ]
        if args.fasta:
            stage3_argv.append("--fasta")
        if args.token_strategy:
            stage3_argv += ["--token_strategy", args.token_strategy]
        if args.unmasking_order:
            stage3_argv += ["--unmasking_order", args.unmasking_order]
        run_stage3(parse_stage3_args(stage3_argv), _setup_logging=False)
        final_output = generated_output
    else:
        # --- Compile to HDF5 ---
        logger.info("=" * 60)
        logger.info("Compiling Stage 2 output to HDF5")
        logger.info("=" * 60)
        compile_args = parse_compile_args([
            "-i", facilitator_output,
            "-o", hdf5_output,
            "--dataset_key", args.dataset_key,
        ])
        run_compile(compile_args, _setup_logging=False)
        final_output = hdf5_output

    logger.info("=" * 60)
    logger.info("Pipeline complete. Output: %s", final_output)
    logger.info("=" * 60)

    # Write manifest and clean up logging
    elapsed = datetime.now() - start_time
    outputs = {
        "pencl_output": os.path.abspath(pencl_output),
        "facilitator_output": os.path.abspath(facilitator_output),
    }
    resolved_paths = {
        "input_data_path": os.path.abspath(args.input_data_path),
        "weight_set": os.path.abspath(args.weight_set) if args.weight_set else None,
        "pencl_weights": os.path.abspath(args.pencl_weights),
        "facilitator_weights": os.path.abspath(args.facilitator_weights),
        "pencl_config": os.path.abspath(args.pencl_config),
        "facilitator_config": os.path.abspath(args.facilitator_config),
    }
    config_contents = {
        "pencl": pencl_config_contents,
        "facilitator": facilitator_config_contents,
    }
    if args.generate:
        outputs["generated_output"] = os.path.abspath(generated_output)
        resolved_paths["proteoscribe_weights"] = os.path.abspath(args.proteoscribe_weights)
        resolved_paths["proteoscribe_config"] = os.path.abspath(args.proteoscribe_config)
        config_contents["proteoscribe"] = load_json_config(args.proteoscribe_config)
    else:
        outputs["hdf5_output"] = os.path.abspath(hdf5_output)

    write_manifest(
        args, args.output_dir, start_time, elapsed,
        outputs=outputs,
        resolved_paths=resolved_paths,
        config_contents=config_contents,
    )
    logger.info("Done in %s", elapsed)
    teardown_file_logging("biom3", file_handler)

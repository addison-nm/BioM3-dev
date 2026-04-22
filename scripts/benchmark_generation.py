#!/usr/bin/env python
"""Benchmark ProteoScribe sequence generation across sweep configurations.

Measures total wall-clock time, per-batch time, and peak device memory for
each ``(token_strategy, N, P, B)`` triple in the sweep. Writes results to
``<output_root>/<UTC timestamp>/`` with:

- ``config.json`` — the input benchmark config, verbatim
- ``env.json``    — device/backend/host info
- ``results.json``— list of per-run records
- ``results.npz`` — dense arrays (T_total_s, T_per_batch_s, peak_alloc_bytes,
                    peak_reserved_bytes, plus sweep-axis arrays)
- ``run.log``     — captured stdout/stderr

Usage:

    python scripts/benchmark_generation.py \\
        --config configs/benchmark/stage3_generation_example.json

See ``configs/benchmark/stage3_generation_example.json`` for the config
schema.
"""

import argparse
import copy
import itertools
import json
import logging
import os
import shutil
import string
import subprocess
import sys
import time
from datetime import datetime, timezone

import numpy as np
import torch

from biom3.core.helpers import load_json_config, convert_to_namespace
from biom3.backend.device import (
    device_sync,
    get_device_info,
    get_peak_memory_stats,
    reset_peak_memory_stats,
)
from biom3.Stage3.io import prepare_model_ProteoScribe
from biom3.Stage3.run_ProteoScribe_sample import batch_stage3_generate_sequences


def _parse_cli(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True,
                        help="Path to benchmark config JSON")
    return parser.parse_args(argv)


def _validate_sweep(cfg):
    sweep = cfg["sweep"]
    for key in ("N", "P", "B"):
        if key not in sweep:
            raise ValueError(f"sweep.{key} is required")
    for n in sweep["N"]:
        for p in sweep["P"]:
            if n % p != 0:
                raise ValueError(
                    f"P={p} does not divide N={n}; R must be an integer"
                )


def _synthetic_prompts(num_prompts: int, length: int) -> list:
    """Generate distinct dummy text strings: 'AAAA', 'AAAB', 'AAAC', ..."""
    if num_prompts <= 0:
        raise ValueError("num_prompts must be positive")
    alphabet = string.ascii_uppercase
    max_slots = len(alphabet) ** length
    if num_prompts > max_slots:
        raise ValueError(
            f"Cannot generate {num_prompts} distinct length-{length} strings "
            f"(max {max_slots})"
        )
    prompts = []
    for idx in range(num_prompts):
        chars = []
        v = idx
        for _ in range(length):
            chars.append(alphabet[v % len(alphabet)])
            v //= len(alphabet)
        prompts.append("".join(reversed(chars)))
    return prompts


def _write_prompts_csv(path: str, prompts: list):
    """Write a CSV compatible with biom3_PenCL_inference.

    Columns: primary_Accession, [final]text_caption, sequence.
    The sequence column is a dummy 10-AA placeholder (unused downstream for
    benchmarking — only z_c is consumed by Stage 3).
    """
    with open(path, "w") as fh:
        fh.write("primary_Accession,[final]text_caption,protein_sequence\n")
        for i, text in enumerate(prompts):
            fh.write(f"BENCH{i:04d},{text},MAAAAAAAAA\n")


def _embed_prompts(cfg, prompts, work_dir, logger):
    """Run Stage 1 + Stage 2 via CLI to produce z_c for the given prompts.

    Returns the ``z_c`` tensor of shape ``[num_prompts, embed_dim]``.
    """
    csv_path = os.path.join(work_dir, "prompts.csv")
    _write_prompts_csv(csv_path, prompts)

    pencl_out = os.path.join(work_dir, "pencl_embeddings.pt")
    facilitator_out = os.path.join(work_dir, "facilitator_embeddings.pt")

    subprocess_env = {**os.environ, "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "true"}

    pencl_cmd = [
        "biom3_PenCL_inference",
        "-i", csv_path,
        "-c", cfg["stage1_config_path"],
        "-m", cfg["stage1_model_path"],
        "-o", pencl_out,
        "--device", cfg["device"],
    ]
    logger.info("Running Stage 1: %s", " ".join(pencl_cmd))
    subprocess.run(pencl_cmd, check=True, env=subprocess_env)

    facilitator_cmd = [
        "biom3_Facilitator_sample",
        "-i", pencl_out,
        "-c", cfg["stage2_config_path"],
        "-m", cfg["stage2_model_path"],
        "-o", facilitator_out,
        "--device", cfg["device"],
    ]
    logger.info("Running Stage 2: %s", " ".join(facilitator_cmd))
    subprocess.run(facilitator_cmd, check=True, env=subprocess_env)

    payload = torch.load(facilitator_out, map_location="cpu")
    z_c = payload["z_c"]
    logger.info("Embedded %d prompts into z_c shape %s",
                len(prompts), tuple(z_c.shape))
    return z_c


def _setup_logger(log_path):
    logger = logging.getLogger("biom3.benchmark_generation")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    fh = logging.FileHandler(log_path)
    sh = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                            datefmt="%H:%M:%S")
    fh.setFormatter(fmt)
    sh.setFormatter(fmt)
    logger.handlers = [fh, sh]
    return logger


def _build_generation_args(cfg, model_config, token_strategy, batch_size,
                           num_replicas):
    """Construct the Namespace consumed by batch_stage3_generate_sequences."""
    args = convert_to_namespace(copy.deepcopy(model_config))
    args.device = cfg["device"]
    args.num_replicas = num_replicas
    args.batch_size_sample = batch_size
    args.token_strategy = token_strategy
    args.unmasking_order = cfg["model_kwargs"].get("unmasking_order", "random")

    pre_cfg = cfg.get("pre_unmask", {})
    args.sequence_length = args.diffusion_steps
    args.pre_unmask = bool(pre_cfg.get("enabled", False))
    if args.pre_unmask:
        budget = cfg["model_kwargs"]["diffusion_steps"]
        if budget > args.sequence_length:
            raise ValueError(
                f"pre_unmask diffusion_budget ({budget}) must be <= "
                f"sequence_length ({args.sequence_length})"
            )
        args.diffusion_steps = budget
        args.pre_unmask_strategy = pre_cfg.get("strategy", "last_k")
        args.pre_unmask_fill_with = pre_cfg.get("fill_with", "PAD")
    return args


def _run_single(args, model, z_c_slice, logger):
    """Execute one generation run and return timing + memory metrics."""
    num_batches = (z_c_slice.size(0) * args.num_replicas
                   + args.batch_size_sample - 1) // args.batch_size_sample
    reset_peak_memory_stats()
    device_sync()
    t0 = time.perf_counter()
    batch_stage3_generate_sequences(args=args, model=model, z_t=z_c_slice)
    device_sync()
    T = time.perf_counter() - t0
    peak_alloc, peak_reserved = get_peak_memory_stats()
    logger.info(
        "  T=%.4fs  T/batch=%.4fs  peak_alloc=%s  peak_reserved=%s",
        T, T / max(num_batches, 1),
        peak_alloc, peak_reserved,
    )
    return {
        "num_batches": num_batches,
        "T_total_s": T,
        "T_per_batch_s": T / max(num_batches, 1),
        "peak_alloc_bytes": peak_alloc,
        "peak_reserved_bytes": peak_reserved,
    }


def _records_to_npz(records, axes_order, out_path):
    """Write per-run records as a dense npz with matching sweep-axis arrays."""
    arrays = {
        ax: np.array([r[ax] for r in records])
        for ax in axes_order
    }
    for metric in ("num_batches", "T_total_s", "T_per_batch_s",
                   "peak_alloc_bytes", "peak_reserved_bytes"):
        arrays[metric] = np.array([
            (r[metric] if r[metric] is not None else np.nan)
            for r in records
        ])
    arrays["_axes_order"] = np.array(axes_order, dtype=object)
    np.savez(out_path, **arrays)


def main(argv=None):
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "true")
    ns = _parse_cli(argv or sys.argv[1:])
    cfg = load_json_config(ns.config)
    _validate_sweep(cfg)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = os.path.join(cfg["output_root"], timestamp)
    os.makedirs(run_dir, exist_ok=True)
    log_path = os.path.join(run_dir, "run.log")
    logger = _setup_logger(log_path)

    logger.info("Benchmark run dir: %s", run_dir)
    logger.info("Architecture ID:   %s", cfg["arch_id"])

    with open(os.path.join(run_dir, "config.json"), "w") as fh:
        json.dump(cfg, fh, indent=2)

    env = {
        "arch_id": cfg["arch_id"],
        "timestamp_utc": timestamp,
        **get_device_info(),
    }
    with open(os.path.join(run_dir, "env.json"), "w") as fh:
        json.dump(env, fh, indent=2)
    logger.info("Device: %s (%s)", env["device_name"], env["backend"])

    prompts_dir = os.path.join(run_dir, "prompts")
    os.makedirs(prompts_dir, exist_ok=True)

    max_P = max(cfg["sweep"]["P"])
    prompt_length = cfg.get("prompts", {}).get("length", 4)
    prompts = _synthetic_prompts(max_P, prompt_length)
    logger.info("Generated %d prompts: %s", len(prompts), prompts)
    z_c_full = _embed_prompts(cfg, prompts, prompts_dir, logger).cpu()

    model_config = load_json_config(cfg["model_config_path"])
    model_args_ns = convert_to_namespace(copy.deepcopy(model_config))
    model_args_ns.device = cfg["device"]
    model_args_ns.seed = 0
    logger.info("Loading ProteoScribe model from %s", cfg["model_path"])
    model = prepare_model_ProteoScribe(
        config_args=model_args_ns,
        model_fpath=cfg["model_path"],
        device=cfg["device"],
        strict=True,
        eval=True,
        attempt_correction=True,
    )

    token_strategies = cfg["model_kwargs"].get("token_strategy", ["sample"])
    if isinstance(token_strategies, str):
        token_strategies = [token_strategies]

    sweep = cfg["sweep"]

    # Burn-in pass: one untimed generation so per-sweep measurements aren't
    # contaminated by first-call costs (kernel JIT, autocast init, allocator
    # first-touch, Gumbel buffer alloc). Uses the smallest config in the
    # sweep so it's cheap; warming with the first token strategy and
    # smallest B/P is enough to settle the dominant kernels.
    warmup_strategy = token_strategies[0]
    warmup_B = min(sweep["B"])
    warmup_P = min(sweep["P"])
    warmup_R = max(1, warmup_B // warmup_P)
    warmup_args = _build_generation_args(
        cfg, model_config, warmup_strategy, warmup_B, warmup_R,
    )
    warmup_z = z_c_full[:warmup_P].to(cfg["device"])
    logger.info(
        "Warmup: token_strategy=%s B=%d P=%d R=%d D=%d (untimed)",
        warmup_strategy, warmup_B, warmup_P, warmup_R, warmup_args.diffusion_steps,
    )
    device_sync()
    t0 = time.perf_counter()
    batch_stage3_generate_sequences(args=warmup_args, model=model, z_t=warmup_z)
    device_sync()
    logger.info("Warmup complete in %.2fs", time.perf_counter() - t0)

    records = []
    axes_order = ["token_strategy", "N", "P", "R", "B", "D"]

    for token_strategy, N, P, B in itertools.product(
        token_strategies, sweep["N"], sweep["P"], sweep["B"],
    ):
        R = N // P
        if B > N:
            logger.info(
                "Skipping B=%d > N=%d (batch larger than total sequences)",
                B, N,
            )
            continue
        args = _build_generation_args(cfg, model_config, token_strategy, B, R)
        z_c_slice = z_c_full[:P].to(cfg["device"])
        logger.info(
            "Run: token_strategy=%s N=%d P=%d R=%d B=%d D=%d",
            token_strategy, N, P, R, B, args.diffusion_steps,
        )
        metrics = _run_single(args, model, z_c_slice, logger)
        records.append({
            "token_strategy": token_strategy,
            "arch_id": cfg["arch_id"],
            "N": N, "P": P, "R": R, "B": B,
            "D": args.diffusion_steps,
            **metrics,
        })

    with open(os.path.join(run_dir, "results.json"), "w") as fh:
        json.dump(records, fh, indent=2)
    _records_to_npz(records, axes_order,
                    os.path.join(run_dir, "results.npz"))
    logger.info("Wrote %d records to %s", len(records), run_dir)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Evaluate a Stage 3 checkpoint by sampling sequences from prompts and
scoring them with a configurable reward.

Designed for the GRPO surrogate-in-the-loop demo: run once on the base
Stage 3 weights (baseline), once on the GRPO-fine-tuned weights, diff
the summary means. Always also computes the ground-truth
``AAFractionReward`` so an apples-to-apples comparison stays available
even when the configured reward is a surrogate.

Outputs
-------
``--output_dir/<run_id>/results.csv``  one row per (prompt, replica) generation
``--output_dir/<run_id>/summary.json``  config + per-prompt + overall stats

Required args mirror the GRPO trainer's: ``--config_path`` (the same
GRPO base config), ``--checkpoint`` (Stage 3 weights to evaluate),
``--prompts_tsv`` (column ``[final]text_caption`` by default).

Example
-------
    python scripts/eval_grpo_checkpoint.py \\
        --config_path configs/grpo/example_grpo.json \\
        --checkpoint  ./weights/ProteoScribe/ProteoScribe_SH3_epoch52.ckpt \\
        --prompts_tsv data/datasets/SH3/synthetic_test.tsv \\
        --n_per_prompt 4 \\
        --reward aa_fraction \\
        --output_dir outputs/grpo/eval/baseline
"""
import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch

from biom3.backend.device import get_device, setup_logger
from biom3.core.helpers import convert_to_namespace, load_json_config
from biom3.rl.grpo import _PromptEncoder
from biom3.rl.io import (
    load_facilitator_frozen,
    load_pencl_frozen,
    load_proteoscribe_trainable,
)
from biom3.rl.rewards import AAFractionReward, build_reward
from biom3.Stage3.run_ProteoScribe_sample import batch_stage3_generate_sequences

logger = setup_logger(__name__)


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--config_path", required=True,
                   help="GRPO base config (provides stage{1,2,3}_config + weights).")
    p.add_argument("--checkpoint", required=True,
                   help="Stage 3 weights to evaluate (raw .pt, .ckpt, or DeepSpeed dir).")
    p.add_argument("--prompts_tsv", required=True,
                   help="TSV with a prompt column.")
    p.add_argument("--prompt_column", default="[final]text_caption")

    p.add_argument("--n_per_prompt", type=int, default=4,
                   help="K sequences sampled per prompt.")
    p.add_argument("--max_prompts", type=int, default=None,
                   help="Optional cap on number of prompts to evaluate (default: all).")

    # Reward selection
    p.add_argument("--reward", default="aa_fraction",
                   choices=["aa_fraction", "esmfold_plddt", "stub", "surrogate"])
    p.add_argument("--surrogate_path", default=None,
                   help="Path to a fitted surrogate joblib (only used when --reward surrogate).")

    # Ground-truth (always computed alongside)
    p.add_argument("--target_aa", default="A")
    p.add_argument("--target_fraction", type=float, default=0.4)
    p.add_argument("--gt_scale", type=float, default=100.0)

    p.add_argument("--output_dir", required=True)
    p.add_argument("--run_id", default=None,
                   help="Run id; defaults to <reward>_<timestamp>.")
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _build_reward(args, device):
    if args.reward == "surrogate":
        if not args.surrogate_path:
            raise ValueError("--surrogate_path required when --reward surrogate")
        import joblib
        from biom3.rl.featurizers import build_featurizer
        from biom3.rl.rewards import SurrogateReward

        sidecar = args.surrogate_path + ".config.json"
        with open(sidecar) as f:
            sc = json.load(f)
        predictor = joblib.load(args.surrogate_path)
        featurizer = build_featurizer(sc["featurizer"], **sc["featurizer_kwargs"])
        return SurrogateReward(predictor=predictor, featurizer=featurizer)
    return build_reward(args.reward, device=device)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device) if args.device else get_device()
    logger.info("Device: %s", device)

    cfg = convert_to_namespace(load_json_config(args.config_path))
    cfg1 = convert_to_namespace(load_json_config(cfg.stage1_config))
    cfg2 = convert_to_namespace(load_json_config(cfg.stage2_config))
    cfg3 = convert_to_namespace(load_json_config(cfg.stage3_config))
    cfg3.device = str(device)
    cfg3.num_replicas = args.n_per_prompt

    # Load prompts
    df = pd.read_csv(args.prompts_tsv, sep="\t")
    if args.prompt_column not in df.columns:
        raise KeyError(
            f"prompt_column {args.prompt_column!r} not in {list(df.columns)}"
        )
    prompts = df[args.prompt_column].astype(str).tolist()
    if args.max_prompts is not None:
        prompts = prompts[: args.max_prompts]
    logger.info("Loaded %d prompts from %s", len(prompts), args.prompts_tsv)
    if not prompts:
        raise ValueError("No prompts to evaluate.")

    # Stage 1+2 frozen for prompt encoding; Stage 3 in eval mode with the
    # checkpoint we're evaluating.
    logger.info("Loading Stage 1 (PenCL)...")
    s1 = load_pencl_frozen(cfg1, getattr(cfg, "stage1_weights", None), device=str(device))
    logger.info("Loading Stage 2 (Facilitator)...")
    s2 = load_facilitator_frozen(cfg2, getattr(cfg, "stage2_weights", None), device=str(device))
    logger.info("Loading Stage 3 (ProteoScribe) from %s ...", args.checkpoint)
    s3 = load_proteoscribe_trainable(cfg3, args.checkpoint, device=str(device))
    s3.eval()

    encode_prompt = _PromptEncoder(s1, s2, cfg1, device)

    logger.info("Encoding %d prompts ...", len(prompts))
    t0 = time.time()
    z_cs = torch.cat([encode_prompt(p) for p in prompts], dim=0)  # (P, emb)
    logger.info("Encoded in %.1fs | z_cs=%s", time.time() - t0, tuple(z_cs.shape))

    # Stage 3 batch sample. num_replicas baked into cfg3 above.
    logger.info(
        "Sampling %d sequences/prompt = %d total ...",
        args.n_per_prompt, args.n_per_prompt * len(prompts),
    )
    t0 = time.time()
    design_dict, _ = batch_stage3_generate_sequences(args=cfg3, model=s3, z_t=z_cs)
    logger.info("Sampled in %.1fs", time.time() - t0)

    # Flatten generations: rows in (prompt_idx, replica_idx, sequence) order.
    rows = []
    for p_idx in range(len(prompts)):
        seqs = design_dict[f"prompt_{p_idx}"]
        for r_idx, seq in enumerate(seqs):
            rows.append({
                "prompt_idx": p_idx,
                "replica_idx": r_idx,
                "prompt": prompts[p_idx],
                "sequence": seq,
            })

    seqs_flat = [r["sequence"] for r in rows]

    # Configured reward
    reward_fn = _build_reward(args, device)
    cfg_scores = reward_fn(seqs_flat)
    # Ground-truth (always computed)
    gt_fn = AAFractionReward(
        target_aa=args.target_aa,
        target_fraction=args.target_fraction,
        scale=args.gt_scale,
    )
    gt_scores = gt_fn(seqs_flat)

    for row, cs, gs in zip(rows, cfg_scores, gt_scores):
        row["reward_score"] = float(cs)
        row["gt_score"] = float(gs)

    # Output
    run_id = args.run_id or f"{args.reward}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir = os.path.join(args.output_dir, run_id)
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["prompt_idx", "replica_idx", "prompt", "sequence",
                           "reward_score", "gt_score"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    logger.info("Wrote %s (%d rows)", csv_path, len(rows))

    cfg_arr = np.array([r["reward_score"] for r in rows], dtype=np.float64)
    gt_arr = np.array([r["gt_score"] for r in rows], dtype=np.float64)
    # Per-prompt means
    per_prompt = []
    for p_idx in range(len(prompts)):
        sl = slice(p_idx * args.n_per_prompt, (p_idx + 1) * args.n_per_prompt)
        per_prompt.append({
            "prompt_idx": p_idx,
            "reward_mean": float(cfg_arr[sl].mean()),
            "gt_mean": float(gt_arr[sl].mean()),
        })

    summary = {
        "config_path": args.config_path,
        "checkpoint": args.checkpoint,
        "prompts_tsv": args.prompts_tsv,
        "n_prompts": len(prompts),
        "n_per_prompt": args.n_per_prompt,
        "reward": args.reward,
        "surrogate_path": args.surrogate_path,
        "reward_mean": float(cfg_arr.mean()),
        "reward_std": float(cfg_arr.std()),
        "gt_mean": float(gt_arr.mean()),
        "gt_std": float(gt_arr.std()),
        "per_prompt": per_prompt,
    }
    sum_path = os.path.join(out_dir, "summary.json")
    with open(sum_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Wrote %s", sum_path)

    print(
        f"\n[summary] reward={args.reward}  "
        f"reward mean={summary['reward_mean']:.3f}  "
        f"GT mean={summary['gt_mean']:.3f}  "
        f"GT std={summary['gt_std']:.3f}",
        flush=True,
    )


if __name__ == "__main__":
    sys.exit(main())

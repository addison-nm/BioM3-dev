#!/usr/bin/env python3
"""BioM3 Stage 3 GDPO fine-tuning runner.

Single-GPU GDPO training. Loads three configs (Stage 1, Stage 2, Stage 3)
plus a GDPO config, builds the policy + frozen reference, and runs the
GDPO loop in ``biom3.rl.gdpo.gdpo_train``.

Configuration precedence (high → low):
    CLI args  >  --config_path JSON  >  argparse defaults

Example:

biom3_gdpo_train \\
    --config_path configs/grpo/example_gdpo.json \\
    --run_id gdpo_001 \\
    --steps 100 --num_generations 4 --n_quadrature 3
"""

import argparse
import os
import sys
from typing import List, Optional

import torch

from biom3.backend.device import get_device, setup_logger
from biom3.core.helpers import convert_to_namespace, load_json_config
from biom3.rl.gdpo import GDPOConfig, gdpo_train
from biom3.rl.grpo import load_prompts
from biom3.rl.rewards import build_reward

logger = setup_logger(__name__)


def _parse_float_list(s: Optional[str]) -> Optional[List[float]]:
    if s is None:
        return None
    if isinstance(s, list):
        return [float(x) for x in s]
    s = s.strip()
    if not s:
        return None
    return [float(x) for x in s.split(",")]


def _parse_device_list(s) -> Optional[List[str]]:
    """Parse ``--rollout_devices`` into a list of device strings.

    Accepts ``None``, an already-parsed list (when set via JSON config),
    a single token like ``"auto"``, or a comma-separated string.
    """
    if s is None:
        return None
    if isinstance(s, list):
        return [str(x) for x in s]
    s = str(s).strip()
    if not s:
        return None
    return [tok.strip() for tok in s.split(",") if tok.strip()]


def get_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--config_path", type=str, default=None,
                        help="GDPO experiment config JSON.")
    parser.add_argument("--run_id", type=str, default="gdpo_run")
    parser.add_argument("--output_root", type=str, default="./outputs/gdpo")

    # Conditioning + policy configs
    parser.add_argument("--stage1_config", type=str, required=False)
    parser.add_argument("--stage2_config", type=str, required=False)
    parser.add_argument("--stage3_config", type=str, required=False)

    # Weights
    parser.add_argument("--stage1_weights", type=str, default=None)
    parser.add_argument("--stage2_weights", type=str, default=None)
    parser.add_argument("--stage3_init_weights", type=str, default=None)

    # Prompts
    parser.add_argument("--prompts_path", type=str, required=False)

    # GDPO hyperparameters (shared with GRPO)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--num_generations", type=int, default=4,
                        help="G — sequences per prompt (group size).")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Prompts per gradient update.")
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--beta", type=float, default=0.01,
                        help="KL coefficient.")
    parser.add_argument("--eps", type=float, default=0.20,
                        help="PPO clip epsilon.")
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--save_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)

    # GDPO-specific: SDMC quadrature
    parser.add_argument("--n_quadrature", type=int, default=3,
                        help="Number of quadrature time-points N (paper uses 3).")
    parser.add_argument("--quadrature_grid", type=str, default="uniform",
                        choices=["uniform", "explicit"])
    parser.add_argument("--quadrature_points", type=str, default=None,
                        help="Comma-separated t_n in (0,1] for --quadrature_grid=explicit.")
    parser.add_argument("--quadrature_weights", type=str, default=None,
                        help="Comma-separated w_n; defaults to uniform 1/N.")
    parser.add_argument("--inner_mc", type=int, default=1,
                        help="MC mask samples per quadrature point.")
    parser.add_argument("--eps_t", type=float, default=1e-3,
                        help="Lower clamp on t_n in the 1/t SDMC factor.")
    parser.add_argument("--kl_estimator", type=str, default="tokenwise_k3",
                        choices=["tokenwise_k3", "sdmc"])
    parser.add_argument("--no_old_policy_snapshot", dest="use_old_policy_snapshot",
                        action="store_false",
                        help="Reuse π_ref as π_old (GRPO-style). Default: snapshot per step.")
    parser.set_defaults(use_old_policy_snapshot=True)
    parser.add_argument("--advantage_normalize", action="store_true",
                        help="Divide advantages by std (classic GRPO). Paper default: off.")
    parser.add_argument("--no_debug_log", dest="debug_log", action="store_false",
                        help="Disable per-step debug.out dump (sequences, masks, ELBOs).")
    parser.set_defaults(debug_log=True)
    parser.add_argument("--no_gradient_checkpoint", dest="gradient_checkpoint",
                        action="store_false",
                        help="Disable activation checkpointing on the trainable-policy ELBO. "
                             "Only safe for short L or small N — otherwise OOMs.")
    parser.set_defaults(gradient_checkpoint=True)

    # Pre-unmask: only diffuse over the first D positions; pre-fill the
    # rest with PAD. Mirrors biom3_ProteoScribe_sample.
    parser.add_argument("--pre_unmask", action="store_true", default=False,
                        help="Diffuse only over the first D positions; pre-fill "
                             "[D, sequence_length) with the configured fill token.")
    parser.add_argument("--pre_unmask_config", type=str, default=None,
                        help="Path to JSON with {strategy, fill_with, diffusion_budget}.")

    # Multi-device rollout (single process, threaded). Gradient updates
    # stay on the master device; replicas exist only for parallel
    # diffusion rollout. ``auto`` picks all visible XPU/CUDA tiles.
    parser.add_argument("--rollout_devices", type=str, default=None,
                        help="Comma-separated devices for parallel rollout "
                             "(e.g. 'xpu:0,xpu:1,xpu:2,xpu:3,xpu:4,xpu:5'), "
                             "or 'auto' for all visible tiles. Default: single device.")

    # Reward
    parser.add_argument("--reward", type=str, default="esmfold_plddt",
                        choices=["esmfold_plddt", "stub"])

    # Device
    parser.add_argument("--device", type=str, default=None,
                        help="Device override; defaults to backend.get_device().")

    return parser


def parse_arguments(argv):
    parser = argparse.ArgumentParser(
        description="BioM3 Stage 3 GDPO fine-tuning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser = get_args(parser)
    pre_args, _ = parser.parse_known_args(argv)
    if pre_args.config_path is not None:
        json_config = load_json_config(pre_args.config_path)
        parser.set_defaults(**json_config)
    return parser.parse_args(argv)


def _required(value, name):
    if value is None:
        raise ValueError(f"Missing required arg: --{name} (or set in --config_path JSON)")
    return value


def main(args):
    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = get_device()
    logger.info("Device: %s", device)

    cfg1 = convert_to_namespace(load_json_config(_required(args.stage1_config, "stage1_config")))
    cfg2 = convert_to_namespace(load_json_config(_required(args.stage2_config, "stage2_config")))
    cfg3 = convert_to_namespace(load_json_config(_required(args.stage3_config, "stage3_config")))

    prompts = load_prompts(_required(args.prompts_path, "prompts_path"))
    logger.info("Loaded %d prompts from %s", len(prompts), args.prompts_path)
    if not prompts:
        raise ValueError(f"No prompts found in {args.prompts_path}")

    output_dir = os.path.join(args.output_root, args.run_id)

    gdpo_cfg = GDPOConfig(
        output_dir=output_dir,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        beta=args.beta,
        eps=args.eps,
        num_generations=args.num_generations,
        batch_size=args.batch_size,
        steps=args.steps,
        save_steps=args.save_steps,
        max_grad_norm=args.max_grad_norm,
        seed=args.seed,
        n_quadrature=args.n_quadrature,
        quadrature_grid=args.quadrature_grid,
        quadrature_points=_parse_float_list(args.quadrature_points),
        quadrature_weights=_parse_float_list(args.quadrature_weights),
        inner_mc=args.inner_mc,
        eps_t=args.eps_t,
        kl_estimator=args.kl_estimator,
        use_old_policy_snapshot=args.use_old_policy_snapshot,
        advantage_normalize=args.advantage_normalize,
        debug_log=args.debug_log,
        gradient_checkpoint=args.gradient_checkpoint,
        pre_unmask=args.pre_unmask,
        pre_unmask_config=args.pre_unmask_config,
        rollout_devices=_parse_device_list(args.rollout_devices),
    )

    reward_fn = build_reward(args.reward, device=device)

    gdpo_train(
        gdpo_cfg=gdpo_cfg,
        cfg1=cfg1,
        cfg2=cfg2,
        cfg3=cfg3,
        prompts=prompts,
        reward_fn=reward_fn,
        device=device,
        stage1_weights=args.stage1_weights,
        stage2_weights=args.stage2_weights,
        stage3_init_weights=args.stage3_init_weights,
    )


if __name__ == "__main__":
    main(parse_arguments(sys.argv[1:]))

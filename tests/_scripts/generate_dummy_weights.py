#!/usr/bin/env python3
"""Generate small dummy weight files for testing.

Builds a model from a config, extracts its state_dict, and saves two
versions simulating weights saved with different versions of the
axial-positional-embedding package:

  - *_v2.bin: v0.2.x key style (underscore: weights_0, weights_1)
  - *_v3.bin: v0.3.x key style (dot: weights.0, weights.1)

See docs/bug_reports/axial_positional_embedding_keys.md for background.

Usage:
    python tests/_scripts/generate_dummy_weights.py \
        --config tests/_data/models/stage3/configs/origkeys_mini.json \
        --output-dir tests/_data/models/stage3/weights

This produces:
    origkeys_mini_v2.bin  (v0.2.x keys: weights_0, weights_1)
    origkeys_mini_v3.bin  (v0.3.x keys: weights.0, weights.1)
"""

import argparse
import os
from collections import OrderedDict

import torch

from biom3.core.helpers import load_json_config, convert_to_namespace
from biom3.Stage3.io import build_model_ProteoScribe


# Substitutions to convert from v0.3.x key style (dot) to v0.2.x (underscore).
_V3_TO_V2_SUBS = {
    "axial_pos_emb.weights.0": "axial_pos_emb.weights_0",
    "axial_pos_emb.weights.1": "axial_pos_emb.weights_1",
}


def generate_dummy_weights(config_path: str, output_dir: str, seed: int = 42):
    """Generate dummy weight files from a model config."""
    torch.manual_seed(seed)

    # Build model and extract state dict.
    # With axial-positional-embedding>=0.3, this produces v0.3 key style.
    config_dict = load_json_config(config_path)
    config_args = convert_to_namespace(config_dict)
    model = build_model_ProteoScribe(config_args)
    v3_sd = model.state_dict()

    # Create the v0.2.x key style (weights_0 / weights_1)
    v2_sd = OrderedDict()
    for key, val in v3_sd.items():
        new_key = key
        for old, new in _V3_TO_V2_SUBS.items():
            new_key = new_key.replace(old, new)
        v2_sd[new_key] = val

    # Derive output filenames from the config filename
    config_name = os.path.splitext(os.path.basename(config_path))[0]
    v2_path = os.path.join(output_dir, f"{config_name}_v2.bin")
    v3_path = os.path.join(output_dir, f"{config_name}_v3.bin")

    torch.save(v2_sd, v2_path)
    torch.save(v3_sd, v3_path)

    # Print summary
    v2_size = os.path.getsize(v2_path)
    v3_size = os.path.getsize(v3_path)
    print(f"Config:  {config_path}")
    print(f"Keys:    {len(v3_sd)}")
    print(f"Params:  {sum(v.numel() for v in v3_sd.values())}")
    print(f"Saved:   {v2_path} ({v2_size / 1024:.1f} KB) — v0.2.x keys (weights_0)")
    print(f"Saved:   {v3_path} ({v3_size / 1024:.1f} KB) — v0.3.x keys (weights.0)")


def main():
    parser = argparse.ArgumentParser(
        description="Generate small dummy weight files for testing."
    )
    parser.add_argument(
        "--config", required=True,
        help="Path to a model config JSON file."
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="Directory to write the dummy weight files."
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)."
    )
    args = parser.parse_args()
    generate_dummy_weights(args.config, args.output_dir, args.seed)


if __name__ == "__main__":
    main()

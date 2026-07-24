#!/usr/bin/env python
"""Assemble the BioM3 Run 1 base weights bundle for publication to GHCR.

The bundle pairs the Run 1 frozen weights with architecture-only partial configs,
so a consumer can pull one versioned object and know which model graph the weights
belong to. Lightning checkpoints are flattened to core weights here rather than
shipped whole, which drops roughly 880 MB of optimizer state.

Layout produced under <output_dir>:

    biom3-weights-run1_base/
      MANIFEST.json
      README.md
      configs/     _base_PenCL.json  _base_Facilitator.json  _base_ProteoScribe.json
      weights/     LLMs/  PenCL/  Facilitator/  ProteoScribe/

The weights/ subtree mirrors the repo's own weights/ directory so that
scripts/link_weights.sh can symlink it straight in, leaving every existing config
path valid.

Usage:

    python scripts/weights_bundle/build_bundle.py --output_dir /scratch/bundles
    python scripts/weights_bundle/build_bundle.py --output_dir /scratch/bundles --dry_run
"""

import argparse
import json
import os
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

from biom3.core.io import load_state_dict_unwrap_pl
from biom3.core.helpers import load_json_config
from biom3.core.run_utils import get_biom3_version, get_git_hash
from biom3.backend.device import setup_logger

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from verify_bundle import sha256_file  # noqa: E402

logger = setup_logger(__name__)

BUNDLE_NAME = "biom3-weights-run1_base"

FROZEN_DIR = "weights/Run1_frozen_ckpts"
FROZEN_MANIFEST = "FROZEN_MANIFEST.md"
LLM_DIR = "weights/LLMs"
BIOMEDBERT_DIR = "BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"

# Lightning checkpoints -> flat state dicts.
FLATTEN_SOURCES = [
    ("Run1_TrackC_step_187000_3b7e39f9.ckpt", "PenCL/BioM3_PenCL_run1_base.bin"),
    ("v3_mmd_full49M_final.ckpt", "Facilitator/BioM3_Facilitator_run1_base.bin"),
]

# Already flat. Its keys are `model.transformer.*`, which is what the Stage 3
# loader expects, so it is copied byte-for-byte rather than re-keyed.
COPY_SOURCES = [
    ("ep_mmd_ep353_base.bin", "ProteoScribe/BioM3_ProteoScribe_run1_base.bin"),
]

# esm.pretrained.load_model_and_alphabet_local() loads the sibling
# -contact-regression.pt unconditionally for any non-esm1v/esm_if model, so the
# small file is mandatory, not optional.
ESM_FILES = [
    "esm2_t33_650M_UR50D.pt",
    "esm2_t33_650M_UR50D-contact-regression.pt",
]

# The checked-out BiomedBERT directory is 1.7 GB; these are the files the code
# reads. flax_model.msgpack and .git/ (with its LFS duplicates) are excluded.
BIOMEDBERT_FILES = [
    "config.json",
    "pytorch_model.bin",
    "tokenizer_config.json",
    "vocab.txt",
    "LICENSE.md",
]

# Architecture and loading keys only. Excluded from PenCL: trainable_seq,
# trainable_text and the *_n_layers_to_finetune pair (they only set
# requires_grad) and temperature (loss-only).
CONFIG_SPECS = [
    (
        "_base_PenCL.json",
        "configs/inference/models/_base_PenCL.json",
        [
            "seq_model_path",
            "pretrained_seq",
            "rep_layer",
            "protein_encoder_embedding",
            "text_model_path",
            "pretrained_text",
            "text_encoder_embedding",
            "text_max_length",
            "proj_embedding_dim",
            "dropout",
        ],
    ),
    (
        "_base_Facilitator.json",
        "configs/inference/models/_base_Facilitator.json",
        ["emb_dim", "hid_dim", "dropout"],
    ),
    (
        "_base_ProteoScribe.json",
        "configs/stage3_training/models/_base_ProteoScribe_1block.json",
        [
            "model_option",
            "num_classes",
            "num_y_class_labels",
            "diffusion_steps",
            "image_size",
            "num_steps",
            "actnorm",
            "perm_channel",
            "perm_length",
            "input_dp_rate",
            "transformer_dim",
            "transformer_heads",
            "transformer_depth",
            "transformer_blocks",
            "transformer_dropout",
            "transformer_reversible",
            "transformer_local_heads",
            "transformer_local_size",
            "text_emb_dim",
            "facilitator",
        ],
    ),
]

MANIFEST_ROW = re.compile(
    r"\|[^|]*\|\s*`([^`]+)`\s*\|\s*(\d+)\s*\|\s*`([0-9a-f]{64})`\s*\|"
)


def parse_frozen_manifest(manifest_path):
    """Pull {filename: (bytes, sha256)} out of the frozen manifest's table."""
    with open(manifest_path) as fh:
        text = fh.read()
    entries = {
        name: (int(size), digest)
        for name, size, digest in MANIFEST_ROW.findall(text)
    }
    if not entries:
        raise ValueError(
            f"No checksum rows parsed from {manifest_path}. The frozen manifest "
            "is the source of truth for source integrity; refusing to build."
        )
    return entries


def verify_sources(frozen_dir, expected):
    """Re-verify every frozen source against the manifest before building."""
    failures = []
    for name, (exp_size, exp_sha) in sorted(expected.items()):
        path = os.path.join(frozen_dir, name)
        if not os.path.isfile(path):
            failures.append(f"{name}: missing")
            continue
        size = os.path.getsize(path)
        digest = sha256_file(path)
        if size != exp_size or digest != exp_sha:
            failures.append(
                f"{name}: size {size} (want {exp_size}), sha256 {digest} (want {exp_sha})"
            )
        else:
            logger.info("  OK   %s", name)
    if failures:
        raise SystemExit(
            "Frozen source verification failed:\n  " + "\n  ".join(failures)
        )


def tensor_bytes(state_dict):
    return sum(
        v.numel() * v.element_size()
        for v in state_dict.values()
        if torch.is_tensor(v)
    )


def flatten_checkpoint(src_path, dst_path):
    """Unwrap a Lightning checkpoint to a flat state_dict and save it."""
    state_dict = load_state_dict_unwrap_pl(src_path)
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    torch.save(state_dict, dst_path)

    n_params = sum(v.numel() for v in state_dict.values() if torch.is_tensor(v))
    on_disk = os.path.getsize(dst_path)
    raw = tensor_bytes(state_dict)
    logger.info(
        "  %s: %d tensors, %.2f M params, %.3f GB on disk (tensors %.3f GB)",
        os.path.basename(dst_path), len(state_dict), n_params / 1e6,
        on_disk / 1e9, raw / 1e9,
    )
    if on_disk > raw * 1.05:
        logger.warning(
            "  %s is larger than its tensor payload — possible unshared storage "
            "views. Inspect before publishing.", os.path.basename(dst_path)
        )
    return {"tensors": len(state_dict), "params": n_params}


def copy_file(src_path, dst_path):
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    shutil.copyfile(src_path, dst_path)


def build_configs(repo_root, config_dir):
    """Emit architecture-only partials derived from the repo's own configs."""
    os.makedirs(config_dir, exist_ok=True)
    written = []
    for out_name, src_rel, keys in CONFIG_SPECS:
        src_path = os.path.join(repo_root, src_rel)
        merged = load_json_config(src_path)
        missing = [k for k in keys if k not in merged]
        if missing:
            raise SystemExit(
                f"{src_rel} is missing expected architecture keys: {missing}. "
                "The bundle's partials are generated from the repo configs; "
                "update CONFIG_SPECS if the schema changed."
            )
        partial = {k: merged[k] for k in keys}
        dst_path = os.path.join(config_dir, out_name)
        with open(dst_path, "w") as fh:
            json.dump(partial, fh, indent=2)
            fh.write("\n")
        written.append((out_name, src_rel, len(partial)))
        logger.info("  %s (%d keys) <- %s", out_name, len(partial), src_rel)
    return written


BUNDLE_README = """\
# {bundle}

BioM3 Run 1 base weights (Stages 1-3) with the architecture-only partial configs
that describe them. Built from the frozen checkpoints in
`weights/Run1_frozen_ckpts/` of BioM3-dev at git `{git_sha}`.

## Install

The `weights/` subtree here mirrors BioM3-dev's own `weights/` layout, so the
repo's existing linker populates a checkout directly:

```bash
cd /path/to/BioM3-dev
./scripts/link_weights.sh {bundle_abs}/weights weights
ln -s {bundle_abs}/configs configs/bundles/run1_base
```

Point `link_weights.sh` at `<bundle>/weights`, not at the bundle root.

Existing configs keep working unchanged: they reference `./weights/LLMs/...`,
and those paths are now symlinks into this bundle.

## Files

| Path | Role |
|---|---|
| `weights/LLMs/esm2_t33_650M_UR50D.pt` | ESM-2 backbone, needed to construct PenCL |
| `weights/LLMs/esm2_t33_650M_UR50D-contact-regression.pt` | Required sibling of the above |
| `weights/LLMs/{biomedbert}/` | BiomedBERT backbone + tokenizer |
| `weights/PenCL/BioM3_PenCL_run1_base.bin` | Stage 1 |
| `weights/Facilitator/BioM3_Facilitator_run1_base.bin` | Stage 2 |
| `weights/ProteoScribe/BioM3_ProteoScribe_run1_base.bin` | Stage 3 |
| `configs/_base_*.json` | Architecture keys pinned to these weights |

`configs/_base_ProteoScribe.json` pins `transformer_blocks: 1`. The Stage 3
weights are a 1-block model; the 16-block base config will not load them.

Every file's sha256 is recorded in `MANIFEST.json`.
"""


def write_bundle_readme(bundle_dir, git_sha):
    text = BUNDLE_README.format(
        bundle=BUNDLE_NAME,
        bundle_abs=os.path.join("<dest>", BUNDLE_NAME),
        git_sha=git_sha,
        biomedbert=BIOMEDBERT_DIR,
    )
    with open(os.path.join(bundle_dir, "README.md"), "w") as fh:
        fh.write(text)


def collect_files(bundle_dir):
    """sha256 + size for every file in the bundle, excluding the manifest."""
    records = []
    for root, _, names in os.walk(bundle_dir):
        for name in sorted(names):
            path = os.path.join(root, name)
            rel = os.path.relpath(path, bundle_dir)
            if rel == "MANIFEST.json":
                continue
            records.append({
                "path": rel,
                "bytes": os.path.getsize(path),
                "sha256": sha256_file(path),
            })
    return sorted(records, key=lambda r: r["path"])


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Build the BioM3 Run 1 base weights bundle."
    )
    parser.add_argument("-o", "--output_dir", type=str, required=True,
                        help="Directory to create the bundle in.")
    parser.add_argument("--repo_root", type=str, default=None,
                        help="BioM3-dev root (default: inferred from this script).")
    parser.add_argument("--dry_run", action="store_true",
                        help="Verify sources and report the plan without writing.")
    parser.add_argument("--skip_llms", action="store_true",
                        help="Omit the LLM backbones. Produces an incomplete bundle; "
                             "useful for a small end-to-end push rehearsal.")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite an existing bundle directory.")
    args = parser.parse_args(argv)

    repo_root = args.repo_root or str(Path(__file__).resolve().parents[2])
    frozen_dir = os.path.join(repo_root, FROZEN_DIR)
    llm_dir = os.path.join(repo_root, LLM_DIR)
    bundle_dir = os.path.join(os.path.abspath(args.output_dir), BUNDLE_NAME)

    logger.info("=" * 60)
    logger.info("Building %s", BUNDLE_NAME)
    logger.info("repo root:  %s", repo_root)
    logger.info("output:     %s", bundle_dir)
    logger.info("biom3:      %s (git %s)", get_biom3_version(), get_git_hash())
    logger.info("=" * 60)

    logger.info("Verifying frozen sources against %s", FROZEN_MANIFEST)
    expected = parse_frozen_manifest(os.path.join(frozen_dir, FROZEN_MANIFEST))
    verify_sources(frozen_dir, expected)

    if args.dry_run:
        logger.info("Dry run: sources verified, nothing written.")
        return 0

    if os.path.exists(bundle_dir):
        if not args.force:
            raise SystemExit(
                f"{bundle_dir} already exists. Pass --force to overwrite."
            )
        shutil.rmtree(bundle_dir)
    os.makedirs(os.path.join(bundle_dir, "weights"))

    weights_out = os.path.join(bundle_dir, "weights")

    logger.info("Flattening Lightning checkpoints")
    derivations = {}
    for src_name, dst_rel in FLATTEN_SOURCES:
        stats = flatten_checkpoint(
            os.path.join(frozen_dir, src_name),
            os.path.join(weights_out, dst_rel),
        )
        derivations[os.path.join("weights", dst_rel)] = {
            "derivation": "flattened from Lightning checkpoint",
            "source": f"{FROZEN_DIR}/{src_name}",
            **stats,
        }

    logger.info("Copying flat weights")
    for src_name, dst_rel in COPY_SOURCES:
        copy_file(
            os.path.join(frozen_dir, src_name),
            os.path.join(weights_out, dst_rel),
        )
        logger.info("  %s", dst_rel)
        derivations[os.path.join("weights", dst_rel)] = {
            "derivation": "copied verbatim",
            "source": f"{FROZEN_DIR}/{src_name}",
        }

    if args.skip_llms:
        logger.warning("--skip_llms: bundle will NOT be usable for Stage 1.")
    else:
        logger.info("Copying LLM backbones")
        for name in ESM_FILES:
            copy_file(
                os.path.join(llm_dir, name),
                os.path.join(weights_out, "LLMs", name),
            )
            logger.info("  LLMs/%s", name)
        for name in BIOMEDBERT_FILES:
            copy_file(
                os.path.join(llm_dir, BIOMEDBERT_DIR, name),
                os.path.join(weights_out, "LLMs", BIOMEDBERT_DIR, name),
            )
            logger.info("  LLMs/%s/%s", BIOMEDBERT_DIR, name)

    logger.info("Generating partial configs")
    config_sources = build_configs(repo_root, os.path.join(bundle_dir, "configs"))

    write_bundle_readme(bundle_dir, get_git_hash())

    logger.info("Hashing bundle contents")
    files = collect_files(bundle_dir)
    for record in files:
        record.update(derivations.get(record["path"], {}))

    manifest = {
        "bundle": BUNDLE_NAME,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "biom3_version": get_biom3_version(),
        "source_repo_git_sha": get_git_hash(),
        "source_manifest": f"{FROZEN_DIR}/{FROZEN_MANIFEST}",
        "frozen_sources": {
            name: {"bytes": size, "sha256": digest}
            for name, (size, digest) in sorted(expected.items())
        },
        "config_sources": [
            {"path": f"configs/{out}", "generated_from": src, "keys": n}
            for out, src, n in config_sources
        ],
        "incomplete": bool(args.skip_llms),
        "files": files,
    }
    with open(os.path.join(bundle_dir, "MANIFEST.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
        fh.write("\n")

    total = sum(r["bytes"] for r in files)
    logger.info("=" * 60)
    logger.info("Bundle complete: %s", bundle_dir)
    logger.info("%d files, %.3f GB", len(files), total / 1e9)
    logger.info("Next: scripts/weights_bundle/push_bundle.sh %s", bundle_dir)
    logger.info("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())

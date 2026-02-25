"""
I/O module for Stage 3 ProteoScribe

"""

import torch.nn as nn
import argparse

from biom3.core.io import load_and_prepare_model
import biom3.Stage3.cond_diff_transformer_layer as mod
import biom3.Stage3.PL_wrapper as PL_mod


def build_model_ProteoScribe(
        config_args: argparse.Namespace
) -> nn.Module:
    return mod.get_model(
        args=config_args,
        data_shape=(config_args.image_size, config_args.image_size),
        num_classes=config_args.num_classes,
    )


def prepare_model_ProteoScribe(
    config_args: argparse.Namespace,
    weights_fpath=None,
    device=None,
    strict=True,
    eval=False,
    attempt_correction=False,
    verbosity=2,
) -> nn.Module:
    
    model = build_model_ProteoScribe(config_args)

    model = load_and_prepare_model(
        model=model,
        weights_path=weights_fpath,
        device=device,
        strict=strict,
        eval_mode=eval,
        attempt_correction=attempt_correction,
        verbosity=verbosity,
    )

    return model
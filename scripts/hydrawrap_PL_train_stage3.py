#!/usr/bin/env python3

"""Training script for BioM3 Stage 3

Wraps the PL_train_stage3 script with Hydra config functionality. This wrapper
should handle any conversion between the structured config file and the argparse
Namespace.

"""

import os
import hydra
from omegaconf import DictConfig, OmegaConf
from lightning.pytorch.utilities import rank_zero_only

from argparse import Namespace


from PL_train_stage3 import main as wrapped_main


def flatten_dict(d, parent_key="", sep="_", join=True):
    items = {}
    for k, v in d.items():
        if join and parent_key:
            new_key = f"{parent_key}{sep}{k}"
        else:
            new_key = k

        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep=sep, join=join))
        else:
            items[new_key] = v
    return items


def cfg_to_namespace(cfg_dict, join, sep="."):
    """Convert an omegaconf container to an argparse Namespace"""
    flat = flatten_dict(cfg_dict, join=join, sep=sep)
    return Namespace(**flat)


def print_config(cfg):
    for k in sorted(list(cfg.keys())):
        if hasattr(cfg[k], "keys"):
            print(f"{k}:")
            for kk in sorted(list(cfg[k].keys())):
                print(f"\t{kk}: {cfg[k][kk]}")
        else:
            print(f"{k}: {cfg[k]}")


# @rank_zero_only
def execute_run(cfg: DictConfig):
    # ----- Process config parameters -----
    cfg_dict = OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )
    print_config(cfg_dict)

    # Retrieve any non-argparse processed configurations
    if "deepspeed" in cfg_dict:
        ds_config = cfg_dict["deepspeed"]
        del cfg_dict["deepspeed"]
    else:
        ds_config = None

    args = cfg_to_namespace(cfg_dict, join=False)
    wrapped_main(args, use_hydra=True, ds_config=ds_config)


@hydra.main(config_path="../configs", config_name="default", version_base=None)
def main(cfg: DictConfig):
    
    # ----- Retrieve MPI environment -----
    local_rank = os.environ.get("LOCAL_RANK")
    print("LOCAL_RANK:", local_rank)

    # ----- Execute the run on rank 0 -----
    execute_run(cfg)
    

if __name__ == '__main__':
    main()

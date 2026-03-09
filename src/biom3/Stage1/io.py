"""
I/O module for Stage 1 PenCL

"""

import torch
import torch.nn as nn

import biom3.Stage1.model as mod


def prepare_model_pfam_PenCL(
        config_args, 
        state_dict_fpath=None, 
        strict=True,
        device=None,
        eval=False,
) -> nn.Module:
    """TODO: implement"""
    # # TODO: Need to test loading using strict=False. Possible that weights 
    # # are not being properly loaded.
    # model = mod.pfam_PEN_CL(args=config_args)
    # if state_dict_fpath:
    #     model.load_state_dict(
    #         torch.load(state_dict_fpath, map_location=device), 
    #         strict=strict,
    #     )
    # if device:
    #     model.to(device)
    # if eval:
    #     model.eval()
    # print("Model loaded successfully with weights!")
    # return model

    raise NotImplementedError()
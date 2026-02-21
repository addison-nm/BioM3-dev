from torch import nn
import torch
from deepspeed.runtime.zero import GatheredParameters
import deepspeed
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

def _z3_params_to_fetch(param_list):
    return [
        p for p in param_list
        if hasattr(p, 'ds_id') and p.ds_status == ZeroParamStatus.NOT_AVAILABLE
    ]


def moving_average(model, model_ema, beta=0.9999, device=None, zero_stage=3):
    zero_stage_3 = (zero_stage == 3)
    with torch.no_grad():
        for param, param_ema in zip(model.parameters(),
                                    model_ema.parameters()):
            # TODO: use prefiltering for efficiency
            params_to_fetch = _z3_params_to_fetch([param, param_ema
                                                   ]) if zero_stage_3 else []
            should_gather_param = len(params_to_fetch) > 0
            with deepspeed.zero.GatheredParameters(
                    params_to_fetch, enabled=should_gather_param):
                data = param.data
                if device is not None:
                    data = data.to(device)
                #print('real model',data.shape, data)
                #print('ema model',param_ema.shape, param_ema.data)
                param_ema.data.copy_(torch.lerp(data, param_ema.data, beta))
                #print('after ema copy',param_ema.shape, param_ema.data)


def clone_zero_model(src_model, dst_model, zero_stage=0):
    zero_stage_3 = (zero_stage == 3)
    with torch.no_grad():
        for src_param, dst_param in zip(src_model.parameters(), dst_model.parameters()):
            # TODO: use prefiltering for efficiency
            params_to_fetch = _z3_params_to_fetch([src_param, dst_param
                                                   ]) if zero_stage_3 else []
            should_gather_param = len(params_to_fetch) > 0
            with deepspeed.zero.GatheredParameters(params_to_fetch, enabled=should_gather_param):
                dst_param.data.copy_(src_param.data)
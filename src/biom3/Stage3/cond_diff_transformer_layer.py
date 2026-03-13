import torch

if torch.cuda.is_available():
    from .cond_diff_transformer_layer_cuda import *
elif hasattr(torch, "xpu") and torch.xpu.is_available():
    from .cond_diff_transformer_layer_xpu import *
else:
    # Default to cuda version for CPU instances?  # TODO: consider alternative
    from .cond_diff_transformer_layer_cuda import *

"""Import tests

"""

import pytest


def test_core_imports():
    import numpy as np
    import torch
    print("Success!")


def test_pytorch_lightning_imports():
    import pytorch_lightning as pl
    from pytorch_lightning import Trainer
    from pytorch_lightning.strategies import DeepSpeedStrategy
    from pytorch_lightning.loggers import TensorBoardLogger
    from pytorch_lightning.loggers import WandbLogger
    from pytorch_lightning.utilities.deepspeed import convert_zero_checkpoint_to_fp32_state_dict
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning.plugins.environments import ClusterEnvironment
    print("Success!")


def test_stage3_imports():
    import Stage3_source.preprocess as prep
    import Stage3_source.cond_diff_transformer_layer as mod
    import Stage3_source.helper_funcs as help_tools
    import Stage3_source.PL_wrapper as PL_mod
    print("Success!")

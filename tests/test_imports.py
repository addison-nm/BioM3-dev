"""Import tests

"""

import pytest


def test_core_imports():
    import numpy as np
    import torch


def test_pytorch_lightning_imports():
    import pytorch_lightning as pl
    from pytorch_lightning import Trainer
    from pytorch_lightning.strategies import DeepSpeedStrategy
    from pytorch_lightning.loggers import TensorBoardLogger
    from pytorch_lightning.loggers import WandbLogger
    from pytorch_lightning.utilities.deepspeed import convert_zero_checkpoint_to_fp32_state_dict
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning.plugins.environments import ClusterEnvironment


def test_stage3_imports():
    import biom3.Stage3.preprocess
    import biom3.Stage3.cond_diff_transformer_layer
    import biom3.Stage3.PL_wrapper


def test_dbio_imports():
    import biom3.dbio
    import biom3.dbio.config
    import biom3.dbio.base
    import biom3.dbio.swissprot
    import biom3.dbio.pfam
    import biom3.dbio.taxonomy
    import biom3.dbio.enrich
    import biom3.dbio.uniprot_client
    import biom3.dbio.build_dataset

"""
Train the z_p text decoder on the swissprot PenCL embeddings
"""
#make sure it's on intel xpus 
try:
    import intel_extension_for_pytorch as ipex
    _HAS_IPEX = True
except ImportError:
    _HAS_IPEX = False

import argparse
import os
import sys
from datetime import datetime
from functools import partial
import torch
from torch.utils.data import DataLoader
from transformers import BioGptTokenizer
from biom3.decode.model import ZpDecoder
from biom3.decode.PL_wrapper import PL_ZpDecoder

from biom3.backend.device import BACKEND_NAME, _XPU, setup_logger, get_rank
from biom3.core.helpers import load_json_config, convert_to_namespace
from biom3.core.run_utils import get_biom3_version, get_git_hash, setup_file_logging, teardown_file_logging, write_manifest
import biom3.decode.preprocess as prep

if BACKEND_NAME == _XPU:
    import lightning as pl
    from lightning import Trainer, seed_everything
    from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
    from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger

else:
    import pytorch_lightning as pl
    from pytorch_lightning import Trainer, seed_everything
    from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
    from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger

logger = setup_logger(__name__)

def parse_arguments(args):
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config_path", "-c", type=str, default=None)
    pre_args, _ = pre_parser.parse_known_args(args)

    parser = argparse.ArgumentParser(description="Train z_p decoder probe")
    parser.add_argument("--config_path", "-c", type=str, default=None)
    get_args(parser)

    if pre_args.config_path is not None:
        json_config = load_json_config(pre_args.config_path)
        parser.set_defaults(**json_config)

    return parser.parse_args(args)

def get_args(parser):
    parser.add_argument('--embedding_path', type=str, required=True,
                        help='path to PenCL embedding (pt) file')
    parser.add_argument('--split_path', type=str, required=True,
                        help='path to train, val, test split CSV')
    parser.add_argument('--output_root', type=str, required=True,
                        help='base directory for all training outputs')
    parser.add_argument('--run_id', type=str, default=None,
                        help='identified for this training run')
    parser.add_argument("--lm_path", type=str, default="microsoft/biogpt",
                        help="language model identife. default is microsoft/biogpt")
    parser.add_argument("--z_dim", type=int, default=512,
                        help="dimentionality of z_p embeddings")
    parser.add_argument("--prefix_length", type=int, default=10,
                        help="number of prefix tokens that the MLP head produces for LM")
    parser.add_argument("--hidden_dim", type=int, default=2048,
                        help="hidden width of the MLP head in the intermediate linear layer")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="dropout probability within MLP head")
    parser.add_argument("--max_length", type=int, default=1024,
                        help="maximum number of caption tokens (length of BioGptTokenizer)")
    parser.add_argument("--lr", type=float, default=5e-4,
                        help="learning rate for the MLP")
    parser.add_argument("--weight_decay", type=float, default=1e-2,
                        help="AdamW weight decay")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="number of z_p-caption pairs per training step")
    parser.add_argument("--num_workers", type=int, default=2,
                        help="number of DataLoader worker processes for tokenization in collate_fn")
    parser.add_argument("--epochs", type=int, default=10,
                        help="number of training epochs")
    parser.add_argument("--seed", type=int, default=0,
                        help="random seed for shuffling, weight init, and dropout")
    parser.add_argument("--device", type=str, default="xpu",
                        choices=["cpu", "cuda", "xpu"],
                        help="backend compute")
    parser.add_argument("--precision", type=str, default="bf16-mixed",
                        help="trainer precision")
    parser.add_argument("--val_check_interval", type=float, default=1.0,
                        help="how often validation runs per epoch")
    parser.add_argument("--early_stopping_patience", type=int, default=3,
                        help="stop training if validation loss does not improve for x epochs")
    parser.add_argument("--num_devices", type=int, default=1,
                        help="number of devives being used from one node")
    parser.add_argument("--num_nodes", type=int, default=1,
                        help="number of nodes being utilized")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="path to a Lightning .ckpt to resume training from "
                             "(omit to start fresh)")
    parser.add_argument("--checkpoint_every_n_epochs", type=int, default=10,
                    help="save an checkpoint every N epochs")


    return parser
    

def main(args):
    #1. build the run and the output directories 
    if args.run_id is None: 
        args.run_id = f"decode_V{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = os.path.join(args.output_root, "runs", args.run_id)
    artifacts_dir = os.path.join(run_dir, "artifacts")
    ckpt_dir = os.path.join(args.output_root, "checkpoints", args.run_id)
    if get_rank() == 0:
        os.makedirs(artifacts_dir, exist_ok=True)
        os.makedirs(ckpt_dir, exist_ok=True)
        
    #2. logging files 
    log_path, file_handler = setup_file_logging(artifacts_dir)
    start_time = datetime.now()
    logger.info("=" * 60)
    logger.info("Decoder training")
    logger.info("biom3 version: %s (git: %s)", get_biom3_version(), get_git_hash())
    logger.info("Command: %s", " ".join(sys.argv))
    logger.info("=" * 60)
    logger.info("torch.xpu.is_available: %s", torch.xpu.is_available() if hasattr(torch, "xpu") else False)
    logger.info("torch.xpu.device_count: %s", torch.xpu.device_count() if hasattr(torch, "xpu") and torch.xpu.is_available() else 0)
    logger.info("torch_xla loaded: %s", "torch_xla" in sys.modules)
    logger.info("ipex loaded: %s", _HAS_IPEX)
    
    #3. set seed and build tokenizer 
    seed_everything(args.seed, workers=True)
    tokenizer = BioGptTokenizer.from_pretrained(args.lm_path)

    #4. build dataset
    train_ds = prep.ZpEmbeddingDataset(args.embedding_path, args.split_path, "train")
    val_ds = prep.ZpEmbeddingDataset(args.embedding_path, args.split_path, "val")
    logger.info("train: %d, val: %d", len(train_ds), len(val_ds))

    #5. build dataloader
    collate = partial(prep.collate_fn, tokenizer=tokenizer, max_length=args.max_length)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle = True,
                                num_workers=args.num_workers,  collate_fn=collate, pin_memory =True)

    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle = False,
                            num_workers=args.num_workers,  collate_fn=collate, pin_memory =True)
    
    #6. build model 
    decoder = ZpDecoder(args)
    pl_module = PL_ZpDecoder(args=args, model=decoder)

    #7. build callbacks and trainer 
    callbacks = [
        ModelCheckpoint(
            dirpath=ckpt_dir, filename="best",
            monitor="val_loss", mode="min", save_top_k=1, save_last=True
        ),
        ModelCheckpoint(
            dirpath=os.path.join(ckpt_dir, "periodic"),
            filename="epoch-{epoch:03d}",
            every_n_epochs=args.checkpoint_every_n_epochs,
            save_top_k=-1,
            save_on_train_epoch_end=True,
        ),
        EarlyStopping( monitor="val_loss", patience=args.early_stopping_patience, mode="min"),
    ]

    accelerator = {"cpu": "cpu", "cuda": "gpu", "xpu": "xpu"}[args.device]
    trainer = Trainer(
        max_epochs = args.epochs, 
        accelerator = accelerator, 
        devices=args.num_devices,
        num_nodes=args.num_nodes,
        strategy="ddp_find_unused_parameters_true",
        precision=args.precision,
        default_root_dir=run_dir,
        callbacks=callbacks,
        val_check_interval=args.val_check_interval,
        log_every_n_steps=50,
        logger=[
            TensorBoardLogger(save_dir=run_dir, name="lightning_logs"),
            CSVLogger(save_dir=run_dir, name="csv_logs"),
        ],
    )

    #8. train 
    ckpt_path = args.resume_from_checkpoint
    if isinstance(ckpt_path, str) and ckpt_path.lower() == "none":
        ckpt_path = None
    if ckpt_path is not None:
        logger.info("Resuming from checkpoint: %s", ckpt_path)
    trainer.fit(pl_module, train_loader, val_loader, ckpt_path=ckpt_path)

    #9. manifest and clean-up 
    if get_rank() == 0:
        elapsed = datetime.now() - start_time
        write_manifest(args, artifacts_dir, start_time, elapsed)
    teardown_file_logging("biom3", file_handler)





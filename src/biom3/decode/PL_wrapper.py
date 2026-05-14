# pytorch fucntions
import torch
from torch import nn, optim
from torch.nn import functional as F
import torch.distributed as dist

# PL functions
from biom3.backend.device import BACKEND_NAME, _XPU

if BACKEND_NAME == _XPU:
    import lightning as pl
    from lightning import Trainer, seed_everything
else:
    import pytorch_lightning as pl
    from pytorch_lightning import Trainer, seed_everything

def _safe_barrier():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()

class PL_ZpDecoder(pl.LightningModule):

    def __init__(
            self,
            args: any,
            model: nn.Module
        ):

        super().__init__()

        self.script_args = args
        self.model = model
    
    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}

    #stop cpu fallback 
    def on_train_start(self):
        p = next(self.parameters())
        if not p.device.type.startswith(("xpu", "cuda")):
            raise RuntimeError(f"Expected params on accelerator, found {p.device}")
        x = torch.randn(2, 4, device=p.device)
        y = torch.nn.functional.layer_norm(x, (4,))
        if y.device != p.device:
            raise RuntimeError(f"layer_norm dispatched off-device: {y.device}")

    def training_step(
            self,
            batch: torch.Tensor,
            batch_idx: any,
        ) -> dict:
        
        device = next(self.model.parameters()).device
        z_p = batch["z_p"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        with torch.amp.autocast(device_type=self.script_args.device, dtype=torch.bfloat16):
            out = self.model(z_p, input_ids, attention_mask, labels)
        _safe_barrier()

    
        self.log('train_loss', out.loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train_perplexity", torch.exp(out.loss.detach()), prog_bar=True, on_epoch=True, sync_dist=True)

        return {'loss': out.loss}
    
    def validation_step(
            self,
            batch: list,
            batch_idx: any
        ) -> dict:
        
        device = next(self.model.parameters()).device
        z_p = batch["z_p"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        with torch.amp.autocast(device_type=self.script_args.device, dtype=torch.bfloat16):
            out = self.model(z_p, input_ids, attention_mask, labels)
        _safe_barrier()

     
        self.log('val_loss', out.loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("val_perplexity", torch.exp(out.loss.detach()), prog_bar=True, on_epoch=True, sync_dist=True)

        return {'val_loss': out.loss}
    
    def configure_optimizers(self,):
        if self.script_args.device == "xpu":
            target_device = self.trainer.strategy.root_device
            self.model = self.model.to(target_device)

        optimizer = torch.optim.AdamW(self.parameters(), lr=self.script_args.lr, weight_decay = self.script_args.weight_decay)

        if self.script_args.device == "xpu":
            import intel_extension_for_pytorch as ipex
            self.model, optimizer = ipex.optimize(self.model, optimizer = optimizer, dtype=torch.bfloat16, inplace=True,)

        return {"optimizer": optimizer}







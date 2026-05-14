"""
query the decoder: AA sequence -> predicted text caption 

python:
    from biom3.decode_llama.query import caption 
    print(caption("<SEQUENCE>"))

terminal: 
    biom3_decode_caption <SEQUENCE>
"""
from __future__ import annotations
import os 
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "true")
import argparse
import sys
from functools import lru_cache
from pathlib import Path 
import torch 
from transformers import AutoTokenizer

from biom3.backend.device import get_device
from biom3.core.helpers import convert_to_namespace, load_json_config
from biom3.Stage1.run_PenCL_inference import prepare_model_from_raw_weights
from biom3.decode_llama.model import ZpDecoder
from biom3.decode_llama.PL_wrapper import PL_ZpDecoder

REPO = Path(__file__).resolve().parents[3]
DEFAULT_PENCL_CONFIG = REPO / "configs/inference/stage1_PenCL.json"
DEFAULT_PENCL_WEIGHTS = REPO / "weights/PenCL/BioM3_PenCL_epoch20.bin"
DEFAULT_DECODER_CKPT = REPO / "src/biom3/decode_llama/outputs/train/checkpoints/llama32_1b_v1/best.ckpt"
DEFAULT_LM_PATH = str(REPO / "weights/LLMs/Llama-3.2-1B")
DECODER_HPARAMS = {
    "lm_path": DEFAULT_LM_PATH,
    "z_dim": 512,
    "prefix_length": 10,
    "hidden_dim": 2048,
    "dropout": 0.1,
}

class Captioner: 
    def __init__(
            self,
            pencl_config: str | Path = DEFAULT_PENCL_CONFIG,
            pencl_weights: str | Path = DEFAULT_PENCL_WEIGHTS,
            decoder_ckpt: str | Path = DEFAULT_DECODER_CKPT,
            device: str | torch.device | None = None 
    ):
        self.device = torch.device(device) if device is not None else get_device()

        pencl_cnfg = convert_to_namespace(load_json_config(str(pencl_config)))
        self.pencl = prepare_model_from_raw_weights(pencl_cnfg, str(pencl_weights), self.device).eval()
        self._batch_converter = self.pencl.protein_encoder.alphabet.get_batch_converter()

        decoder_args = convert_to_namespace(dict(DECODER_HPARAMS)) 
        inner = ZpDecoder(decoder_args)
        pl_mod = PL_ZpDecoder.load_from_checkpoint(
            str(decoder_ckpt),
            args = decoder_args,
            model = inner, 
            map_location = self.device
        )
        self.decoder = pl_mod.model.eval().to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(DEFAULT_LM_PATH)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    @torch.no_grad()
    def encode(self, sequence: str) -> torch.Tensor: 
        _, _, toks = self._batch_converter([("query", sequence)])
        toks = toks.to(self.device).unsqueeze(1)
        z_s = self.pencl.protein_encoder(toks)
        z_p = self.pencl.protein_projection(z_s)

        return z_p 

    @torch.no_grad()
    def __call__(
        self, 
        sequence: str, 
        *,
        max_new_tokens: int = 200,
        do_sample: bool = False, 
        num_beams: int = 4,
        temperature: float = 1.0, 
        top_p: float = 1.0, 
        num_return_sequences: int = 1,
    ) -> str | list[str]:
        
        z_p = self.encode(sequence)
        prefix = self.decoder.head(z_p)
        bos = torch.tensor([[self.tokenizer.bos_token_id]], device=self.device)
        bos_emb = self.decoder.model.get_input_embeddings()(bos)
        inputs_embeds = torch.cat([prefix, bos_emb], dim = 1)
        attn = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=self.device)

        out_ids = self.decoder.model.generate(
            inputs_embeds = inputs_embeds, 
            attention_mask = attn, 
            max_new_tokens = max_new_tokens, 
            do_sample = do_sample, 
            num_beams = 1 if do_sample else num_beams,
            temperature = temperature, 
            top_p = top_p, 
            num_return_sequences = num_return_sequences, 
            pad_token_id = self.tokenizer.pad_token_id,
        )
        texts = [self.tokenizer.decode(ids, skip_special_tokens = True) for ids in out_ids]
        
        return texts[0] if num_return_sequences == 1 else texts 
    
@lru_cache(maxsize= 1)
def _default_captioner() -> Captioner:
    return Captioner()
    
def caption(sequence: str, **gen_kwargs) -> str | list[str]:
    return _default_captioner()(sequence, **gen_kwargs)

def _read_sequence(args: argparse.Namespace) -> str:
    if args.seq_file:
        return Path(args.seq_file).read_text().strip()
    if args.sequence: 
        return args.sequence
    if not sys.stdin.isatty():
        return sys.stdin.read().strip()
    return ""
    
def _cli(argv: list[str] | None = None) -> None: 
    p = argparse.ArgumentParser(description="caption for a protein sequence")
    p.add_argument("sequence", nargs="?", help="string of amino acids")
    p.add_argument("--seq-file", help="path to file containing sequences")
    p.add_argument("--ckpt", default=str(DEFAULT_DECODER_CKPT), help="decoder ckpt path")
    p.add_argument("--pencl-config", default=str(DEFAULT_PENCL_CONFIG), help="path to pencl config")
    p.add_argument("--pencl-weights", default=str(DEFAULT_PENCL_WEIGHTS), help="path to pencl weights")
    p.add_argument("--sample", action="store_true", help="top-p sampling instead of beam search")
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--num-beams", type=int, default=4)
    p.add_argument("-n", "--num-return-sequences", type=int, default=1)
    p.add_argument("--max-new-tokens", type=int, default=200)
    args = p.parse_args(argv)

    seq = _read_sequence(args)
    if not seq:
        p.error("provide a sequence (or a --seq-file)")
        
    cap = Captioner(
        pencl_config=args.pencl_config,
        pencl_weights=args.pencl_weights,
        decoder_ckpt=args.ckpt
    )

    out = cap(
        seq, 
        do_sample = args.sample, 
        num_beams = args.num_beams,
        temperature = args.temperature,
        top_p = args.top_p if args.sample else 1.0,
        num_return_sequences = args.num_return_sequences,
        max_new_tokens = args.max_new_tokens,
    )
    if isinstance(out, str):
        print(out)
    else:
        for i, text in enumerate(out, 1):
            print(f"[{i}] {text}")


if __name__ == "__main__":
    _cli()


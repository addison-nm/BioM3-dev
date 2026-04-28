"""GRPO single-GPU trainer for BioM3 Stage 3.

Group Relative Policy Optimization: for each prompt, sample K sequences,
compute group-normalized advantages from a per-sequence reward, and update
the policy with PPO-clip + KL against a frozen reference snapshot.

Ported from biom3-grpo-rl/biom3_grpo.py to use the packaged biom3.* API.
"""

import copy
import json
import os
import time
from argparse import Namespace
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import biom3.Stage1.preprocess as S1prep
import biom3.Stage3.animation_tools as S3ani
import biom3.Stage3.sampling_analysis as S3sample
from biom3.backend.device import setup_logger
from biom3.rl.io import (
    load_facilitator_frozen,
    load_pencl_frozen,
    load_proteoscribe_trainable,
)

logger = setup_logger(__name__)


# Token vocabulary — must match the order baked into Stage 3 weights.
TOKENS = [
    "-", "<START>", "A", "C", "D", "E", "F", "G", "H", "I",
    "K", "L", "M", "N", "P", "Q", "R", "S", "T", "V",
    "W", "Y", "<END>", "<PAD>", "X", "U", "Z", "B", "O",
]
TOK2ID = {t: i for i, t in enumerate(TOKENS)}
PAD_ID = TOK2ID["<PAD>"]
START_ID = TOK2ID["<START>"]
END_ID = TOK2ID["<END>"]
NUM_CLASSES = len(TOKENS)
VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")


@dataclass
class GRPOConfig:
    output_dir: str = "./grpo_output"
    learning_rate: float = 1e-5
    weight_decay: float = 1e-6
    beta: float = 0.01           # KL coefficient
    eps: float = 0.20            # PPO clip
    num_generations: int = 4     # K
    batch_size: int = 1          # prompts per gradient step
    steps: int = 200
    save_steps: int = 50
    max_grad_norm: float = 1.0
    seed: int = 42


def load_prompts(path: str) -> List[str]:
    """Read prompts from a text file. Supports comments (#) and `name|text`."""
    prompts: List[str] = []
    with open(path) as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "|" in line:
                _, text = line.split("|", 1)
                prompts.append(text.strip())
            else:
                prompts.append(line)
    return prompts


class _PromptEncoder:
    """Wraps Stage 1 + Stage 2 to turn a text prompt into a conditioning
    vector ``z_c`` of shape ``(1, emb_dim)``.

    The padding/length conventions in
    ``Stage1.preprocess.TextSeqPairing_Dataset`` (``padding='max_length'``,
    ``max_length = cfg1.text_max_length``) match Stage 1 training — see
    docs/bug_reports/bert_embedding_mismatch.md for why this matters.
    """

    def __init__(self, s1, s2, cfg1: Namespace, device: torch.device):
        self.s1 = s1
        self.s2 = s2
        self.cfg1 = cfg1
        self.device = device
        self._dataset = None

    def _build_dataset(self) -> S1prep.TextSeqPairing_Dataset:
        tmp_df = pd.DataFrame({
            self.cfg1.sequence_keyword: ["A" * 50],
            "[final]text_caption": ["placeholder"],
            self.cfg1.id_keyword: ["GRPO_PROMPT"],
        })
        return S1prep.TextSeqPairing_Dataset(args=self.cfg1, df=tmp_df)

    @torch.no_grad()
    def __call__(self, prompt: str) -> torch.Tensor:
        if self._dataset is None:
            self._dataset = self._build_dataset()
        ds = self._dataset
        tmp_df = pd.DataFrame({
            self.cfg1.sequence_keyword: ["A" * 50],
            "[final]text_caption": [prompt],
            self.cfg1.id_keyword: ["GRPO_PROMPT"],
        })
        ds.df = tmp_df
        ds.text_captions_list = tmp_df["[final]text_caption"].tolist()
        ds.protein_sequence_list = tmp_df[self.cfg1.sequence_keyword].tolist()
        ds.accession_id_list = tmp_df[self.cfg1.id_keyword].tolist()

        x_t, x_p = ds[0]
        x_t = x_t.to(self.device)
        x_p = x_p.to(self.device)
        out = self.s1(x_t, x_p, compute_masked_logits=False)
        z_t = out["text_joint_latent"]
        return self.s2(z_t).to(self.device)


def decode_tokens(tok_tensor: torch.Tensor) -> str:
    """Token ID tensor → clean amino-acid string (special tokens stripped)."""
    raw = S3ani.convert_num_to_char(TOKENS, tok_tensor)
    for special in ("<START>", "<END>", "<PAD>", "<MASK>", "<UNK>", "-"):
        raw = raw.replace(special, "")
    return raw


@torch.no_grad()
def diffusion_rollout(
    s3: torch.nn.Module,
    cfg3: Namespace,
    z_c: torch.Tensor,
    K: int,
    device: torch.device,
) -> torch.Tensor:
    """Sample ``K`` sequences from the absorbing-state diffusion model
    conditioned on ``z_c`` (shape ``(1, emb_dim)``).

    Returns ``(K, L)`` int64 token IDs, where ``L = cfg3.diffusion_steps``.
    """
    L = cfg3.diffusion_steps
    z_rep = z_c.repeat(K, 1)
    perms = torch.stack([torch.randperm(L, device=device) for _ in range(K)])

    mask_list, _, _ = S3sample.batch_generate_denoised_sampled(
        args=cfg3,
        model=s3,
        extract_digit_samples=torch.zeros(K, L, device=device),
        extract_time=torch.zeros(K, dtype=torch.long, device=device),
        extract_digit_label=z_rep,
        sampling_path=perms,
    )
    final = mask_list[-1]   # (K, 1, L)
    out = torch.empty(K, L, dtype=torch.long, device=device)
    for k in range(K):
        out[k] = torch.as_tensor(final[k, 0], dtype=torch.long, device=device)
    return out


def _policy_logprobs(
    s3: torch.nn.Module,
    ids: torch.Tensor,             # (BK, L)
    z_c_rep: torch.Tensor,         # (BK, emb_dim)
    pad_id: int,
) -> torch.Tensor:
    """Per-token log-probs at the chosen ids under ``s3`` with input fully
    masked at ``t = 0``. Returns shape ``(BK, L)``."""
    BK, L = ids.shape
    x_masked = torch.full_like(ids, pad_id)
    t_steps = torch.zeros(BK, dtype=torch.long, device=ids.device)
    logits = s3(x_masked, t_steps, z_c_rep).float().permute(0, 2, 1)  # (BK, L, V)
    lp = F.log_softmax(logits, dim=-1)
    return lp.gather(-1, ids.unsqueeze(-1)).squeeze(-1)


def grpo_train(
    grpo_cfg: GRPOConfig,
    cfg1: Namespace,
    cfg2: Namespace,
    cfg3: Namespace,
    prompts: List[str],
    reward_fn,
    device: torch.device,
    stage1_weights: Optional[str] = None,
    stage2_weights: Optional[str] = None,
    stage3_init_weights: Optional[str] = None,
):
    cfg3.device = str(device)

    torch.manual_seed(grpo_cfg.seed)
    np.random.seed(grpo_cfg.seed)

    logger.info("Loading Stage 1 (PenCL)...")
    s1 = load_pencl_frozen(cfg1, stage1_weights, device=str(device))
    logger.info("Loading Stage 2 (Facilitator)...")
    s2 = load_facilitator_frozen(cfg2, stage2_weights, device=str(device))
    logger.info("Loading Stage 3 (ProteoScribe, trainable)...")
    s3 = load_proteoscribe_trainable(cfg3, stage3_init_weights, device=str(device))
    s3.train()

    logger.info("Snapshotting frozen reference policy...")
    ref_s3 = copy.deepcopy(s3).eval()
    for p in ref_s3.parameters():
        p.requires_grad_(False)

    encode_prompt = _PromptEncoder(s1, s2, cfg1, device)

    optimizer = torch.optim.AdamW(
        s3.parameters(),
        lr=grpo_cfg.learning_rate,
        weight_decay=grpo_cfg.weight_decay,
    )

    os.makedirs(grpo_cfg.output_dir, exist_ok=True)
    log_rows = []
    L = cfg3.diffusion_steps
    eps = grpo_cfg.eps
    beta = grpo_cfg.beta
    B = grpo_cfg.batch_size
    K = grpo_cfg.num_generations
    BK = B * K

    logger.info(
        "GRPO start: steps=%d K=%d batch=%d beta=%.4f eps=%.2f",
        grpo_cfg.steps, K, B, beta, eps,
    )

    for step in range(1, grpo_cfg.steps + 1):
        t0 = time.time()
        batch_prompts = [
            prompts[np.random.randint(len(prompts))] for _ in range(B)
        ]

        s3.eval()
        with torch.no_grad():
            z_cs = torch.cat([encode_prompt(p) for p in batch_prompts], dim=0)
            z_cs_rep = z_cs.repeat_interleave(K, dim=0)             # (BK, emb)

            ids_per_prompt = [
                diffusion_rollout(s3, cfg3, z_cs[i:i + 1], K, device)
                for i in range(B)
            ]
            ids_all = torch.cat(ids_per_prompt, dim=0)              # (BK, L)
            seqs = [decode_tokens(ids_all[i]) for i in range(BK)]

            lp_ref_tok = _policy_logprobs(ref_s3, ids_all, z_cs_rep, PAD_ID)

        rewards_raw = reward_fn(seqs)
        R = torch.tensor(rewards_raw, dtype=torch.float32, device=device)
        Rg = R.view(B, K)
        adv = (
            (Rg - Rg.mean(dim=-1, keepdim=True))
            / (Rg.std(dim=-1, keepdim=True).clamp(min=1e-8))
        ).view(BK)

        s3.train()
        lp_new_tok = _policy_logprobs(s3, ids_all, z_cs_rep, PAD_ID)

        valid = (ids_all != PAD_ID).float()
        log_ratio = lp_new_tok - lp_ref_tok.detach()
        ratio = torch.exp(log_ratio)
        adv_2d = adv.unsqueeze(1).expand_as(valid)
        pg1 = -adv_2d * ratio
        pg2 = -adv_2d * ratio.clamp(1 - eps, 1 + eps)
        pg_loss = (torch.max(pg1, pg2) * valid).sum() / (valid.sum() + 1e-8)

        kl_tok = (
            torch.exp(lp_ref_tok.detach() - lp_new_tok)
            - (lp_ref_tok.detach() - lp_new_tok)
            - 1.0
        )
        kl_loss = (kl_tok * valid).sum() / (valid.sum() + 1e-8)

        loss = pg_loss + beta * kl_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(s3.parameters(), grpo_cfg.max_grad_norm)
        optimizer.step()

        dt = time.time() - t0
        clip_frac = ((ratio - 1.0).abs() > eps).float().mean().item()
        mean_reward = R.mean().item()
        avg_len = float(np.mean([len(s) for s in seqs]))

        logger.info(
            "step=%4d | reward=%5.2f | loss=%.4f | pg=%.4f | kl=%.4f "
            "| clip=%.2f | len=%.0f | %.1fs",
            step, mean_reward, loss.item(), pg_loss.item(), kl_loss.item(),
            clip_frac, avg_len, dt,
        )
        log_rows.append({
            "step": step,
            "reward": round(mean_reward, 3),
            "loss": round(loss.item(), 5),
            "pg": round(pg_loss.item(), 5),
            "kl": round(kl_loss.item(), 5),
            "clip_frac": round(clip_frac, 4),
            "avg_len": round(avg_len, 1),
        })

        if step % grpo_cfg.save_steps == 0:
            ckpt_path = os.path.join(grpo_cfg.output_dir, f"step{step}.pt")
            torch.save({"step": step, "model_state": s3.state_dict()}, ckpt_path)
            logger.info("Saved checkpoint: %s", ckpt_path)

    final_path = os.path.join(grpo_cfg.output_dir, "final.pt")
    torch.save({"step": grpo_cfg.steps, "model_state": s3.state_dict()}, final_path)
    log_path = os.path.join(grpo_cfg.output_dir, "train_log.json")
    with open(log_path, "w") as f:
        json.dump(log_rows, f, indent=2)
    logger.info("GRPO done. Final checkpoint: %s", final_path)
    return log_rows

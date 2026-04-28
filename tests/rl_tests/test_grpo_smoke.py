"""CPU smoke tests for biom3.rl.grpo.

Exercises the GRPO building blocks without requiring real Stage 1/Stage 2
weights, ESMFold, or a GPU. Uses the mini Stage 3 fixture under
tests/_data/models/stage3/.
"""

import copy
import os

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from tests.conftest import DATDIR
from biom3.core.helpers import convert_to_namespace, load_json_config
from biom3.Stage3.io import build_model_ProteoScribe
from biom3.rl.grpo import (
    NUM_CLASSES,
    PAD_ID,
    TOKENS,
    _policy_logprobs,
    decode_tokens,
    diffusion_rollout,
    load_prompts,
)
from biom3.rl.rewards import StubReward, build_reward


MINI_WEIGHTS = os.path.join(DATDIR, "models/stage3/weights/minimodel1_ds128_weights1.pth")
MINI_CONFIG = os.path.join(DATDIR, "configs/test_stage3_config_v2.json")


@pytest.fixture
def mini_s3():
    cfg = convert_to_namespace(load_json_config(MINI_CONFIG))
    cfg.device = "cpu"
    model = build_model_ProteoScribe(cfg)
    sd = torch.load(MINI_WEIGHTS, map_location="cpu")
    model.load_state_dict(sd, strict=True)
    return model, cfg


def test_token_constants():
    assert NUM_CLASSES == 29
    assert TOKENS[PAD_ID] == "<PAD>"


def test_load_prompts_handles_comments_and_name_pipe(tmp_path):
    p = tmp_path / "prompts.txt"
    p.write_text(
        "# header comment\n"
        "\n"
        "first prompt\n"
        "namedone|second prompt\n"
        "  # indented comment is kept (only leading-#-after-strip is dropped)\n"
        "third prompt\n"
    )
    prompts = load_prompts(str(p))
    assert prompts == ["first prompt", "second prompt", "third prompt"]


def test_decode_tokens_strips_specials():
    tok_ids = torch.tensor([
        TOKENS.index("<START>"),
        TOKENS.index("A"),
        TOKENS.index("L"),
        TOKENS.index("<END>"),
        PAD_ID,
        PAD_ID,
    ])
    out = decode_tokens(tok_ids)
    assert out == "AL"


def test_stub_reward_in_range():
    rfn = StubReward(target_length=20)
    rewards = rfn(["AAAAAAAAAAAAAAAAAAAA", "ACDEFGHIKLMNPQRSTVWY", ""])
    assert all(0.0 <= r <= 100.0 for r in rewards)


def test_build_reward_dispatch():
    assert isinstance(build_reward("stub", device=torch.device("cpu")), StubReward)
    with pytest.raises(ValueError):
        build_reward("nonsense", device=torch.device("cpu"))


def test_diffusion_rollout_shape_and_dtype(mini_s3):
    s3, cfg = mini_s3
    s3.eval()
    z_c = torch.randn(1, cfg.text_emb_dim)
    K = 2
    ids = diffusion_rollout(s3, cfg, z_c, K, torch.device("cpu"))
    assert ids.shape == (K, cfg.diffusion_steps)
    assert ids.dtype == torch.int64
    assert int(ids.min()) >= 0
    assert int(ids.max()) < cfg.num_classes


def test_policy_logprobs_shape_and_negativity(mini_s3):
    s3, cfg = mini_s3
    s3.eval()
    BK, L = 3, cfg.diffusion_steps
    ids = torch.randint(0, cfg.num_classes, (BK, L), dtype=torch.int64)
    z_c = torch.randn(BK, cfg.text_emb_dim)
    lp = _policy_logprobs(s3, ids, z_c, PAD_ID)
    assert lp.shape == (BK, L)
    assert torch.isfinite(lp).all()
    assert (lp <= 0).all()


def test_grpo_inner_step_updates_params(mini_s3):
    s3, cfg = mini_s3
    cfg.device = "cpu"
    device = torch.device("cpu")

    ref_s3 = copy.deepcopy(s3).eval()
    for p in ref_s3.parameters():
        p.requires_grad_(False)

    s3.train()
    optimizer = torch.optim.AdamW(s3.parameters(), lr=1e-3)
    reward_fn = StubReward(target_length=30)

    B, K = 1, 4
    BK = B * K
    z_cs = torch.randn(B, cfg.text_emb_dim)
    z_cs_rep = z_cs.repeat_interleave(K, dim=0)

    with torch.no_grad():
        ids_all = diffusion_rollout(s3, cfg, z_cs[0:1], K, device)
        seqs = [decode_tokens(ids_all[i]) for i in range(BK)]
        lp_ref_tok = _policy_logprobs(ref_s3, ids_all, z_cs_rep, PAD_ID)

    R = torch.tensor(reward_fn(seqs), dtype=torch.float32)
    Rg = R.view(B, K)
    adv = (
        (Rg - Rg.mean(dim=-1, keepdim=True))
        / (Rg.std(dim=-1, keepdim=True).clamp(min=1e-8))
    ).view(BK)

    lp_new_tok = _policy_logprobs(s3, ids_all, z_cs_rep, PAD_ID)
    valid = (ids_all != PAD_ID).float()
    log_ratio = lp_new_tok - lp_ref_tok.detach()
    ratio = torch.exp(log_ratio)
    adv_2d = adv.unsqueeze(1).expand_as(valid)
    pg = (torch.max(-adv_2d * ratio, -adv_2d * ratio.clamp(0.8, 1.2)) * valid).sum() \
         / (valid.sum() + 1e-8)
    kl = (
        (torch.exp(lp_ref_tok.detach() - lp_new_tok)
         - (lp_ref_tok.detach() - lp_new_tok) - 1.0) * valid
    ).sum() / (valid.sum() + 1e-8)

    loss = pg + 0.01 * kl
    assert torch.isfinite(loss)

    p0 = next(s3.parameters()).detach().clone()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    p1 = next(s3.parameters()).detach()
    assert not torch.allclose(p0, p1), "Optimizer step did not modify Stage 3 params"

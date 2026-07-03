"""CPU tests for biom3.Stage3.lora (LoRA finetuning for ProteoScribe).

Uses the mini Stage 3 model fixture (random init is fine — these are structural
and numerical-equivalence checks, no weights needed).
"""

import copy
import os

import pytest
import torch
import torch.nn as nn

from tests.conftest import DATDIR
from biom3.core.helpers import convert_to_namespace, load_json_config
from biom3.Stage3.io import build_model_ProteoScribe
from biom3.Stage3.PL_wrapper import PL_ProtARDM
from biom3.Stage3.lora import (
    DEFAULT_TARGET_PATTERNS,
    LoRALinear,
    apply_lora_finetuning,
    export_lora_finetuned,
    freeze_and_inject_lora,
    inject_lora,
    load_lora_weights,
    merge_lora,
    save_lora_weights,
)

MINI_CONFIG = os.path.join(DATDIR, "configs/test_stage3_config_v2.json")
MINI_WEIGHTS = os.path.join(DATDIR, "models/stage3/weights/minimodel1_ds128_weights1.pth")


def _base_with_weights():
    """A mini ProteoScribe loaded with the ds128 weights (deterministic base)."""
    cfg = convert_to_namespace({**load_json_config(MINI_CONFIG), "device": "cpu"})
    m = build_model_ProteoScribe(cfg)
    m.load_state_dict(torch.load(MINI_WEIGHTS, map_location="cpu"), strict=True)
    return m, cfg


@pytest.fixture
def mini_model():
    cfg = convert_to_namespace(load_json_config(MINI_CONFIG))
    cfg.device = "cpu"
    return build_model_ProteoScribe(cfg), cfg


def _inputs(cfg, B=2):
    L = cfg.diffusion_steps
    x = torch.randint(1, cfg.num_classes, (B, L))
    t = torch.zeros(B, dtype=torch.long)
    z_c = torch.randn(B, cfg.text_emb_dim)
    return x, t, z_c


def _n_qv_targets(model):
    return sum(1 for n, m in model.named_modules()
               if isinstance(m, nn.Linear)
               and any(p in n for p in DEFAULT_TARGET_PATTERNS))


def test_inject_targets_qv_only(mini_model):
    model, _ = mini_model
    expected = _n_qv_targets(model)
    assert expected > 0
    model, injected = inject_lora(model, r=4, alpha=8, dropout=0.0)
    assert len(injected) == expected
    assert all(".fn.to_q" in n or ".fn.to_v" in n for n in injected)
    n_lora = sum(1 for m in model.modules() if isinstance(m, LoRALinear))
    assert n_lora == expected


def test_inject_no_match_raises(mini_model):
    model, _ = mini_model
    with pytest.raises(ValueError):
        inject_lora(model, target_patterns=(".does_not_exist",))


def test_only_lora_and_ymlp_trainable(mini_model):
    model, _ = mini_model
    model, summary = freeze_and_inject_lora(model, r=4, alpha=8, dropout=0.0,
                                            unfreeze_y_mlp=True)
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert trainable, "nothing trainable"
    for n in trainable:
        assert ("lora_" in n) or ("y_mlp" in n), f"unexpected trainable param: {n}"
    # base attention weights are frozen
    frozen = [p for n, p in model.named_parameters()
              if "base_layer" in n and "weight" in n]
    assert frozen and all(not p.requires_grad for p in frozen)
    assert summary["n_lora_layers"] == _n_qv_targets(build_model_ProteoScribe(
        convert_to_namespace({**load_json_config(MINI_CONFIG), "device": "cpu"})))
    assert summary["trainable_params"] == summary["lora_params"] + summary["y_mlp_params"]
    # Base is frozen (attention out/k, feed-forward, embeddings, output head), so
    # only a strict subset is trainable. (The exact fraction depends on model size;
    # it is tiny on the real 86M model but large on this mini fixture.)
    assert 0 < summary["trainable_params"] < summary["total_params"]


def test_zero_delta_at_init_matches_base(mini_model):
    model, cfg = mini_model
    model.eval()
    x, t, z_c = _inputs(cfg)
    with torch.no_grad():
        base_out = model(x, t, z_c).clone()
    model, _ = freeze_and_inject_lora(model, r=4, alpha=8, dropout=0.0)
    model.eval()
    with torch.no_grad():
        lora_out = model(x, t, z_c)
    # lora_B is zero-initialized -> ΔW = 0 -> identical output.
    assert torch.allclose(base_out, lora_out, atol=1e-5)


def test_forward_shape_and_grad_flow(mini_model):
    model, cfg = mini_model
    model, _ = freeze_and_inject_lora(model, r=4, alpha=8, dropout=0.0)
    model.train()
    x, t, z_c = _inputs(cfg)
    out = model(x, t, z_c)
    assert out.shape == (x.shape[0], cfg.num_classes, cfg.diffusion_steps)
    out.sum().backward()
    got_lora = got_ymlp = 0
    for n, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"trainable param without grad: {n}"
            got_lora += "lora_" in n
            got_ymlp += "y_mlp" in n
        else:
            assert p.grad is None, f"frozen param received grad: {n}"
    assert got_lora > 0 and got_ymlp > 0


def test_merge_equivalence_and_downstream_loadable(mini_model):
    model, cfg = mini_model
    model, _ = freeze_and_inject_lora(model, r=4, alpha=8, dropout=0.0)
    # Simulate a trained adapter (lora_B is zero at init, so perturb it).
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, LoRALinear):
                m.lora_B.weight.normal_(std=0.02)
    model.eval()
    x, t, z_c = _inputs(cfg)
    with torch.no_grad():
        lora_out = model(x, t, z_c).clone()

    merged = copy.deepcopy(model)
    merged, n = merge_lora(merged)
    merged.eval()
    with torch.no_grad():
        merged_out = merged(x, t, z_c)

    assert n == _n_qv_targets(merged)  # every LoRALinear folded away
    assert not any(isinstance(m, LoRALinear) for m in merged.modules())
    assert torch.allclose(lora_out, merged_out, atol=1e-5)

    # Merged model is a plain ProteoScribe again: loads strict into a fresh one.
    fresh = build_model_ProteoScribe(
        convert_to_namespace({**load_json_config(MINI_CONFIG), "device": "cpu"}))
    missing, unexpected = fresh.load_state_dict(merged.state_dict(), strict=False)
    assert not missing and not unexpected


def test_lora_ft_job_saves_loadable_weights_only_qv_ymlp_change(mini_model):
    """End-to-end LoRA finetune: run real training steps, save + reload the
    merged weights, and verify ONLY the expected params (attention Q/V + y_mlp)
    differ from the original base — the LoRA analog of test_finetuning's
    expected-changes check.
    """
    model, cfg = mini_model
    # Training hyperparams the PL wrapper / loss path reads off script_args.
    cfg.lr = 1e-2
    cfg.weight_decay = 0.0
    base_sd = {k: v.detach().clone() for k, v in model.state_dict().items()}

    pl_model = PL_ProtARDM(args=cfg, model=model)
    pl_model = apply_lora_finetuning(pl_model, r=4, alpha=8, dropout=0.0)
    pl_model.train()

    optimizer = torch.optim.AdamW(
        [p for p in pl_model.parameters() if p.requires_grad], lr=cfg.lr)
    torch.manual_seed(0)
    L = cfg.diffusion_steps
    for _ in range(3):  # a few real diffusion-loss steps
        # cond_elbo_objective is the loss common_step runs; call it directly to
        # avoid the PL Trainer-bound self.log() in common_step.
        realization = torch.randint(1, cfg.num_classes, (2, 1, L)).long()
        z_c = torch.randn(2, cfg.text_emb_dim)
        train_tuple = pl_model.cond_elbo_objective(
            realization=realization, y_c=z_c, realization_idx=0, stage="train")
        loss = train_tuple[0]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # "Save": fold adapters back into the base and export a plain state_dict.
    merge_lora(pl_model.model)
    saved = {k: v.detach().clone() for k, v in pl_model.model.state_dict().items()}

    # Reloadable by a fresh (plain) ProteoScribe with no key drift.
    fresh = build_model_ProteoScribe(
        convert_to_namespace({**load_json_config(MINI_CONFIG), "device": "cpu"}))
    missing, unexpected = fresh.load_state_dict(saved, strict=False)
    assert not missing and not unexpected, (missing[:3], unexpected[:3])

    # Only attention Q/V (LoRA delta folded in) and y_mlp may differ from base.
    def _expected_to_change(name):
        return (".fn.to_q" in name) or (".fn.to_v" in name) or ("y_mlp" in name)

    errors, changed_expected = [], 0
    assert set(saved) == set(base_sd), "param name set drifted after merge"
    for name, w0 in base_sd.items():
        w1 = saved[name]
        differs = not torch.equal(w0, w1)
        if _expected_to_change(name):
            changed_expected += int(differs)
        elif differs:
            errors.append(f"unexpected change in frozen param {name} "
                          f"(max|Δ|={ (w1 - w0).abs().max():.3e})")
    assert not errors, "LoRA FT changed params it should not have:\n" + "\n".join(errors)
    # Sanity: the adapters actually moved the Q/V + y_mlp weights.
    assert changed_expected > 0, "no expected param changed — LoRA had no effect"


def test_save_load_lora_delta_roundtrip(tmp_path):
    """The small adapter+y_mlp delta file reconstructs the family model when
    loaded back onto the same base."""
    m1, cfg = _base_with_weights()
    m1, _ = freeze_and_inject_lora(m1, r=4, alpha=8, dropout=0.0)
    with torch.no_grad():  # simulate a trained adapter (+ moved y_mlp)
        for mod in m1.modules():
            if isinstance(mod, LoRALinear):
                mod.lora_A.weight.normal_(std=0.02)
                mod.lora_B.weight.normal_(std=0.02)
        for n, p in m1.named_parameters():
            if "y_mlp" in n:
                p.add_(0.01 * torch.randn_like(p))
    m1.eval()
    x, t, z_c = _inputs(cfg)
    with torch.no_grad():
        out1 = m1(x, t, z_c).clone()

    path = os.path.join(tmp_path, "lora.pt")
    n = save_lora_weights(m1, path)
    ckpt = torch.load(path, map_location="cpu")
    keys = list(ckpt["lora_state_dict"])
    assert keys and all(("lora_" in k) or ("y_mlp" in k) for k in keys)
    assert ckpt["config"]["r"] == 4 and ckpt["config"]["alpha"] == 8

    m2, loaded = load_lora_weights(_base_with_weights()[0], path)
    assert loaded == n
    m2.eval()
    with torch.no_grad():
        out2 = m2(x, t, z_c)
    assert torch.allclose(out1, out2, atol=1e-5)


def test_export_lora_writes_both_artifacts(tmp_path):
    """A LoRA run emits BOTH a small adapter file and a full merged checkpoint;
    both reconstruct the same model and the adapter file is much smaller."""
    m, cfg = _base_with_weights()
    m, _ = freeze_and_inject_lora(m, r=4, alpha=8, dropout=0.0)
    with torch.no_grad():
        for mod in m.modules():
            if isinstance(mod, LoRALinear):
                mod.lora_B.weight.normal_(std=0.02)
    m.eval()
    x, t, z_c = _inputs(cfg)
    with torch.no_grad():
        out_before = m(x, t, z_c).clone()

    lora_p = os.path.join(tmp_path, "lora_weights.pt")
    merged_p = os.path.join(tmp_path, "state_dict.merged.pth")
    export_lora_finetuned(m, lora_p, merged_p)  # merges m in place
    assert os.path.exists(lora_p) and os.path.exists(merged_p)

    # Merged: plain ProteoScribe, strict-loadable, reproduces output.
    fresh = build_model_ProteoScribe(cfg)
    miss, unexp = fresh.load_state_dict(torch.load(merged_p, map_location="cpu"), strict=False)
    assert not miss and not unexp
    fresh.eval()
    with torch.no_grad():
        assert torch.allclose(out_before, fresh(x, t, z_c), atol=1e-5)

    # Adapter file: reconstructs the same model on a fresh base, and is smaller.
    m2, _ = load_lora_weights(_base_with_weights()[0], lora_p)
    m2.eval()
    with torch.no_grad():
        assert torch.allclose(out_before, m2(x, t, z_c), atol=1e-5)
    assert os.path.getsize(lora_p) < os.path.getsize(merged_p)

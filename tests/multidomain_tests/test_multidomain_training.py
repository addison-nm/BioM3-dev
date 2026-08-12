"""Objective, conditioning and optimizer-group tests for the multidomain module.

The invariants that matter: one shared diffusion timestep across canvases with
independent unmasking paths, per-domain independent alpha, prior references that
stay out of the checkpoint, and an optimizer grouping the pre-flight audit will
accept.
"""

import argparse

import pytest
import torch

from biom3.Stage3.multidomain.audit import (
    AuditFailure,
    audit_trainable_parameters,
    enforce_audit,
)
from biom3.Stage3.multidomain.io import MultiDomainSpec
from biom3.Stage3.multidomain.model import PAD_ID
from biom3.Stage3.multidomain.PL_wrapper import (
    PRIOR_GENERATION,
    PL_ProtARDM_Multidomain,
)

from .conftest import EMB_DIM, NUM_DOMAINS, SEQ_LEN, make_composed


BATCH = 2


class _FakeEmbedder(torch.nn.Module):
    """Stand-in for the frozen PenCL text branch + Facilitator."""

    def __init__(self, emb_dim=EMB_DIM):
        super().__init__()
        self.emb_dim = emb_dim
        self.linear = torch.nn.Linear(4, emb_dim)
        self.calls = []

    def forward(self, input_ids):
        self.calls.append(tuple(input_ids.shape))
        # Deterministic in the token ids, so identical captions give identical z_c.
        base = input_ids.float().mean(dim=-1, keepdim=True)
        return base.expand(-1, self.emb_dim).contiguous()


def _args(**overrides):
    base = dict(task="proteins", lr=1e-4, weight_decay=1e-6)
    base.update(overrides)
    return argparse.Namespace(**base)


def _module(tiny_args, *, expert_prior_lambda=0.0, train_alpha=0.0,
            zp_lookup=None, spec=None, prior_mode="weight", model=None):
    model = model or make_composed(tiny_args)
    return PL_ProtARDM_Multidomain(
        _args(), model, _FakeEmbedder(), spec=spec, zp_lookup=zp_lookup,
        train_alpha=train_alpha, expert_prior_lambda=expert_prior_lambda,
        prior_mode=prior_mode,
    )


def _batch(num_domains=NUM_DOMAINS, batch=BATCH, n_real=20, text_len=4):
    torch.manual_seed(5)
    num_seqs = torch.full((batch, num_domains, SEQ_LEN), float(PAD_ID))
    for d in range(num_domains):
        length = n_real - 4 * d
        num_seqs[:, d, :length] = torch.randint(1, 23, (batch, length)).float()
    input_ids = torch.randint(0, 100, (batch, num_domains, text_len))
    return [num_seqs, input_ids]


# ── conditioning ──────────────────────────────────────────────────────────


def test_batch_transfer_produces_per_domain_conditioning(tiny_args):
    module = _module(tiny_args)
    num_seqs, y_c = module.on_after_batch_transfer(_batch(), 0)
    assert y_c.shape == (BATCH, NUM_DOMAINS, EMB_DIM)
    # The embedder saw the flattened rows, not a [B, K, T] tensor.
    assert module.embedder.calls[0] == (BATCH * NUM_DOMAINS, 4)


def test_embedder_is_frozen_and_excluded_from_parameters(tiny_args):
    module = _module(tiny_args)
    assert not any(p.requires_grad for p in module.embedder.parameters())
    embedder_ids = {id(p) for p in module.embedder.parameters()}
    assert not any(id(p) in embedder_ids for p in module.parameters())
    assert not any("embedder" in k for k in module.state_dict())


def test_train_alpha_is_drawn_per_example_and_domain(tiny_args):
    """A batch may pair a protein-faithful domain with a text-driven one."""
    module = _module(tiny_args, train_alpha="blend")
    module.train()
    torch.manual_seed(0)
    alpha = module._train_alpha(BATCH * NUM_DOMAINS, torch.device("cpu"))
    assert alpha.shape == (BATCH * NUM_DOMAINS, 1)
    assert len(set(alpha.flatten().tolist())) > 1


def test_eval_alpha_is_deterministic_per_domain_sequence(tiny_args):
    module = _module(tiny_args)
    sequences = ["ACDEF", "GHIKL", "ACDEF"]
    first = module._eval_alpha(sequences, torch.device("cpu"))
    second = module._eval_alpha(sequences, torch.device("cpu"))
    assert torch.equal(first, second)
    assert first[0].item() == first[2].item()
    assert first[0].item() != first[1].item()


def test_zp_blend_requires_the_sequence_list(tiny_args):
    module = _module(tiny_args, zp_lookup={}, train_alpha=1.0)
    with pytest.raises(RuntimeError, match="needs the raw domain sequences"):
        module.on_after_batch_transfer(_batch(), 0)


def test_zp_blend_rejects_a_misaligned_sequence_list(tiny_args):
    module = _module(tiny_args, zp_lookup={}, train_alpha=1.0)
    batch = _batch() + [["ACDEF"]]
    with pytest.raises(RuntimeError, match="one sequence per caption"):
        module.on_after_batch_transfer(batch, 0)


def test_zp_blend_at_alpha_one_returns_zp(tiny_args):
    sequences = [f"SEQ{i}" for i in range(BATCH * NUM_DOMAINS)]
    lookup = {s: torch.full((EMB_DIM,), float(i))
              for i, s in enumerate(sequences)}
    module = _module(tiny_args, zp_lookup=lookup, train_alpha=1.0)
    module.train()
    _, y_c = module.on_after_batch_transfer(_batch() + [sequences], 0)
    expected = torch.stack([lookup[s] for s in sequences]).reshape(
        BATCH, NUM_DOMAINS, EMB_DIM)
    assert torch.allclose(y_c, expected)


# ── objective ─────────────────────────────────────────────────────────────


def test_training_step_returns_a_finite_loss(tiny_args):
    module = _module(tiny_args)
    batch = module.on_after_batch_transfer(_batch(), 0)
    out = module.common_step(batch, 0, stage="train")
    assert torch.isfinite(out["loss"])


def test_timestep_is_shared_but_paths_are_independent(tiny_args, monkeypatch):
    """One idx for the assembly; a separate unmasking path per canvas."""
    import biom3.Stage3.transformer_training_helper as helper

    seen_idx, seen_paths = [], []
    real_index = helper.sample_random_index_for_sampling
    real_path = helper.sample_random_path

    def spy_index(*a, **k):
        idx = real_index(*a, **k)
        seen_idx.append(idx)
        return idx

    def spy_path(*a, **k):
        path = real_path(*a, **k)
        seen_paths.append(path)
        return path

    monkeypatch.setattr(helper, "sample_random_index_for_sampling", spy_index)
    monkeypatch.setattr(helper, "sample_random_path", spy_path)

    module = _module(tiny_args)
    batch = module.on_after_batch_transfer(_batch(), 0)
    module.common_step(batch, 0, stage="train")

    assert len(seen_idx) == 1, "idx must be drawn once for the whole assembly"
    assert len(seen_paths) == NUM_DOMAINS, "each canvas needs its own path"
    assert not torch.equal(seen_paths[0], seen_paths[1])


def test_loss_sums_over_canvases(tiny_args):
    """Dropping a domain must change the loss — no canvas is ignored."""
    module = _module(tiny_args)
    batch = module.on_after_batch_transfer(_batch(), 0)
    torch.manual_seed(0)
    both = module.common_step(batch, 0, stage="train")["loss"]

    single = _module(tiny_args, model=make_composed(tiny_args, num_domains=2))
    torch.manual_seed(0)
    perturbed = list(batch)
    perturbed[0] = batch[0].clone()
    perturbed[0][:, 1, :10] = 7.0
    changed = single.common_step(perturbed, 0, stage="train")["loss"]
    assert not torch.equal(both, changed)


def test_rejects_a_domain_count_mismatch(tiny_args):
    module = _module(tiny_args)
    batch = module.on_after_batch_transfer(_batch(num_domains=NUM_DOMAINS), 0)
    batch[0] = batch[0][:, :1]
    with pytest.raises(ValueError, match="batch carries 1 domains"):
        module.common_step(batch, 0, stage="train")


def test_gradients_reach_the_coupling(tiny_args):
    module = _module(tiny_args)
    with torch.no_grad():
        for param in module.model.coupling.parameters():
            param.add_(0.01)
    batch = module.on_after_batch_transfer(_batch(), 0)
    module.common_step(batch, 0, stage="train")["loss"].backward()
    grads = [p.grad for p in module.model.coupling.parameters() if p.grad is not None]
    assert grads and any(g.abs().sum() > 0 for g in grads)


# ── Component B ───────────────────────────────────────────────────────────


def test_prior_references_stay_out_of_the_checkpoint(tiny_args):
    module = _module(tiny_args, expert_prior_lambda=1e-5)
    assert module._prior_refs is not None
    assert not any("_prior_ref_" in key for key in module.state_dict())


def test_prior_penalty_is_zero_at_init(tiny_args):
    module = _module(tiny_args, expert_prior_lambda=1e-5)
    penalty = module._prior_penalty(None, None, None)
    assert penalty.item() == pytest.approx(0.0, abs=1e-12)


def test_prior_penalty_grows_with_drift(tiny_args):
    module = _module(tiny_args, expert_prior_lambda=1e-5)
    with torch.no_grad():
        next(module.model.experts[0].parameters()).add_(0.5)
    assert module._prior_penalty(None, None, None).item() > 0


def test_expert_delta_norms_track_each_expert(tiny_args):
    module = _module(tiny_args, expert_prior_lambda=1e-5)
    assert all(v == pytest.approx(0.0, abs=1e-12)
               for v in module.expert_delta_norms().values())
    with torch.no_grad():
        for param in module.model.experts[1].parameters():
            param.mul_(1.01)
    deltas = module.expert_delta_norms()
    assert deltas[0] == pytest.approx(0.0, abs=1e-12)
    assert deltas[1] == pytest.approx(0.01, rel=1e-3)


def test_no_prior_references_when_disabled(tiny_args):
    module = _module(tiny_args, expert_prior_lambda=0.0)
    assert module._prior_refs is None
    assert module.expert_delta_norms() == {}


def test_generation_prior_restores_expert_weights(tiny_args):
    """The reference forward must not leave the experts perturbed."""
    module = _module(tiny_args, expert_prior_lambda=1e-5,
                     prior_mode=PRIOR_GENERATION)
    with torch.no_grad():
        next(module.model.experts[0].parameters()).add_(0.25)
    before = [p.detach().clone() for p in module.model.experts[0].parameters()]

    batch = module.on_after_batch_transfer(_batch(), 0)
    module.common_step(batch, 0, stage="train")

    after = list(module.model.experts[0].parameters())
    assert all(torch.equal(a, b) for a, b in zip(before, after))


# ── optimizer groups ──────────────────────────────────────────────────────


def test_prior_regularized_experts_get_zero_weight_decay(tiny_args):
    module = _module(tiny_args, expert_prior_lambda=1e-5)
    optimizer = module.configure_optimizers()
    expert_ids = {id(p) for e in module.model.experts for p in e.parameters()}
    for group in optimizer.param_groups:
        if any(id(p) in expert_ids for p in group["params"]):
            assert group["weight_decay"] == 0.0


def test_optimizer_passes_the_preflight_audit(tiny_args):
    module = _module(tiny_args, expert_prior_lambda=1e-5)
    optimizer = module.configure_optimizers()
    report = audit_trainable_parameters(
        module.model, train_experts=True, optimizer=optimizer)
    assert enforce_audit(report)


def test_frozen_experts_leave_only_the_coupling_trainable(tiny_args):
    model = make_composed(tiny_args)
    for expert in model.experts:
        for param in expert.parameters():
            param.requires_grad_(False)
    module = _module(tiny_args, model=model)
    optimizer = module.configure_optimizers()
    trainable = {id(p) for p in model.parameters() if p.requires_grad}
    in_optimizer = {id(p) for g in optimizer.param_groups for p in g["params"]}
    assert in_optimizer == trainable

    report = audit_trainable_parameters(
        model, train_experts=False, optimizer=optimizer)
    assert enforce_audit(report)


# ── checkpoint spec ───────────────────────────────────────────────────────


def test_spec_is_saved_into_hyperparameters(tiny_args):
    spec = MultiDomainSpec.from_args(tiny_args, NUM_DOMAINS,
                                     domain_ids=("PF00501", "PF13193"))
    module = _module(tiny_args, spec=spec)
    assert module.hparams["multidomain_spec"] == spec.to_dict()
    assert module.hparams["multidomain_fingerprint"]


def test_checkpoint_round_trips_through_the_real_saver(tmp_path, tiny_args):
    """The spec written at train time must satisfy the reader in io.py.

    Until now the round-trip test built its checkpoint by hand; this closes the
    loop against a checkpoint produced the way training produces one.
    """
    from biom3.Stage3.multidomain.io import build_multidomain_from_checkpoint

    spec = MultiDomainSpec.from_args(tiny_args, NUM_DOMAINS,
                                     domain_ids=("PF00501", "PF13193"))
    module = _module(tiny_args, spec=spec)
    with torch.no_grad():
        for param in module.model.coupling.parameters():
            param.add_(0.05)

    path = tmp_path / "trained.ckpt"
    torch.save({
        "state_dict": module.state_dict(),
        "hyper_parameters": dict(module.hparams),
    }, path)

    restored, restored_spec = build_multidomain_from_checkpoint(
        path, template_args=tiny_args)
    assert restored_spec == spec

    module.model.eval()
    restored.eval()
    torch.manual_seed(1)
    x = torch.randint(0, 29, (2, NUM_DOMAINS, SEQ_LEN))
    t = torch.randint(0, SEQ_LEN, (2,)).float()
    y_c = torch.randn(2, NUM_DOMAINS, EMB_DIM)
    with torch.no_grad():
        assert torch.equal(module.model(x, t, y_c), restored(x, t, y_c))

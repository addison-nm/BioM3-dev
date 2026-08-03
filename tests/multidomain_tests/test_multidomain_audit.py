"""Pre-flight gate tests: additive null, freeze inventory, expert drift."""

import copy

import pytest
import torch

from biom3.Stage3.multidomain.audit import (
    AuditFailure,
    assert_additive_null,
    audit_trainable_parameters,
    enforce_audit,
    expert_delta_norms,
)

from .conftest import NUM_DOMAINS, make_composed


def _freeze_experts(model):
    for expert in model.experts:
        for param in expert.parameters():
            param.requires_grad_(False)


def _references(model):
    return [[p.detach().clone() for p in expert.parameters()]
            for expert in model.experts]


# ── additive null ─────────────────────────────────────────────────────────


def test_assert_additive_null_passes_at_init(composed):
    assert assert_additive_null(composed, batch=2)


def test_assert_additive_null_rejects_non_null_coupling(composed):
    with torch.no_grad():
        for param in composed.coupling.out_proj["0_1"]:
            param.weight.fill_(0.01)
    with pytest.raises(AuditFailure, match="not at the additive null"):
        assert_additive_null(composed, batch=2)


def test_assert_additive_null_rejects_a_disconnected_coupling(composed, monkeypatch):
    """A coupling never wired into the forward pass must not pass the gate.

    Such a model satisfies every equality check trivially — it is the converse
    probe, not the equalities, that catches it.
    """
    original_forward = type(composed).forward

    def uncoupled_forward(self, x, t, y_c, real_masks=None, couple=True):
        return original_forward(self, x, t, y_c, real_masks=real_masks, couple=False)

    monkeypatch.setattr(type(composed), "forward", uncoupled_forward)
    with pytest.raises(AuditFailure, match="not wired into the forward pass"):
        assert_additive_null(composed, batch=2)


def test_assert_additive_null_restores_the_probed_weight(composed):
    before = composed.coupling.out_proj["0_1"][0].weight.detach().clone()
    assert_additive_null(composed, batch=2)
    after = composed.coupling.out_proj["0_1"][0].weight.detach()
    assert torch.equal(before, after)
    assert composed.coupling.is_additive_null(tol=0.0)


def test_assert_additive_null_restores_training_mode(composed):
    composed.train()
    assert_additive_null(composed, batch=2)
    assert composed.training


# ── trainable inventory ───────────────────────────────────────────────────


def test_audit_passes_for_coupling_only(composed):
    _freeze_experts(composed)
    report = audit_trainable_parameters(composed, train_experts=False)
    assert report["experts_trainable"] == 0
    assert report["coupling_trainable"] > 0
    assert report["other_trainable"] == 0
    assert enforce_audit(report)


def test_audit_passes_for_component_ab(composed):
    report = audit_trainable_parameters(composed, train_experts=True)
    assert report["experts_trainable"] > 0
    assert enforce_audit(report)


def test_audit_rejects_trainable_experts_when_frozen_is_declared(composed):
    report = audit_trainable_parameters(composed, train_experts=False)
    with pytest.raises(AuditFailure, match="train_experts is off but"):
        enforce_audit(report)


def test_audit_rejects_frozen_experts_when_training_is_declared(composed):
    _freeze_experts(composed)
    report = audit_trainable_parameters(composed, train_experts=True)
    with pytest.raises(AuditFailure, match="no expert parameter is trainable"):
        enforce_audit(report)


def test_audit_rejects_a_frozen_coupling(composed):
    _freeze_experts(composed)
    for param in composed.coupling.parameters():
        param.requires_grad_(False)
    report = audit_trainable_parameters(composed, train_experts=False)
    with pytest.raises(AuditFailure, match="coupling has no trainable parameters"):
        enforce_audit(report)


def test_audit_rejects_a_stray_trainable_tensor(composed):
    _freeze_experts(composed)
    composed.register_parameter("stray", torch.nn.Parameter(torch.zeros(3)))
    report = audit_trainable_parameters(composed, train_experts=False)
    assert "stray" in report["other_trainable_names"]
    with pytest.raises(AuditFailure, match="outside the coupling"):
        enforce_audit(report)


def test_audit_rejects_a_mismatched_declared_count(composed):
    _freeze_experts(composed)
    report = audit_trainable_parameters(
        composed, train_experts=False, expected_trainable=1)
    with pytest.raises(AuditFailure, match="!= the 1 declared"):
        enforce_audit(report)


def test_audit_can_require_a_null_coupling(composed):
    _freeze_experts(composed)
    with torch.no_grad():
        composed.coupling.out_proj["0_1"][0].weight.fill_(0.01)
    report = audit_trainable_parameters(composed, train_experts=False)
    with pytest.raises(AuditFailure, match="not at the additive null"):
        enforce_audit(report, require_null_coupling=True)
    assert enforce_audit(report, require_null_coupling=False)


# ── optimizer groups ──────────────────────────────────────────────────────


def _optimizer(model, expert_weight_decay):
    expert_params = [p for e in model.experts for p in e.parameters() if p.requires_grad]
    coupling_params = [p for p in model.coupling.parameters() if p.requires_grad]
    return torch.optim.AdamW([
        {"params": coupling_params, "weight_decay": 1e-6},
        {"params": expert_params, "weight_decay": expert_weight_decay},
    ])


def test_audit_rejects_weight_decay_on_prior_regularized_experts(composed):
    """Decay toward zero and a prior toward W_ref are conflicting priors."""
    optimizer = _optimizer(composed, expert_weight_decay=1e-6)
    report = audit_trainable_parameters(
        composed, train_experts=True, optimizer=optimizer)
    assert report["optimizer"]["prior_regularized_with_weight_decay"] > 0
    with pytest.raises(AuditFailure, match="weight_decay > 0 group"):
        enforce_audit(report)


def test_audit_accepts_zero_decay_on_prior_regularized_experts(composed):
    optimizer = _optimizer(composed, expert_weight_decay=0.0)
    report = audit_trainable_parameters(
        composed, train_experts=True, optimizer=optimizer)
    assert report["optimizer"]["prior_regularized_with_weight_decay"] == 0
    assert enforce_audit(report)


def test_audit_rejects_a_trainable_parameter_missing_from_the_optimizer(composed):
    coupling_params = [p for p in composed.coupling.parameters()]
    optimizer = torch.optim.AdamW(coupling_params[:-1], weight_decay=0.0)
    _freeze_experts(composed)
    report = audit_trainable_parameters(
        composed, train_experts=False, optimizer=optimizer)
    assert report["optimizer"]["missing_from_optimizer"] > 0
    with pytest.raises(AuditFailure, match="not in the optimizer"):
        enforce_audit(report)


def test_audit_rejects_a_frozen_parameter_inside_the_optimizer(composed):
    _freeze_experts(composed)
    frozen = next(composed.experts[0].parameters())
    optimizer = torch.optim.AdamW(
        [{"params": list(composed.coupling.parameters()) + [frozen],
          "weight_decay": 0.0}])
    report = audit_trainable_parameters(
        composed, train_experts=False, optimizer=optimizer)
    assert report["optimizer"]["frozen_in_optimizer"] > 0
    with pytest.raises(AuditFailure, match="frozen parameter"):
        enforce_audit(report)


# ── expert drift ──────────────────────────────────────────────────────────


def test_expert_delta_is_zero_at_init(composed):
    deltas = expert_delta_norms(composed, _references(composed))
    assert set(deltas) == set(range(NUM_DOMAINS))
    for value in deltas.values():
        assert value == pytest.approx(0.0, abs=1e-12)


def test_expert_delta_grows_after_a_weight_change(composed):
    references = _references(composed)
    with torch.no_grad():
        next(composed.experts[0].parameters()).add_(0.5)
    deltas = expert_delta_norms(composed, references)
    assert deltas[0] > 0
    assert deltas[1] == pytest.approx(0.0, abs=1e-12)


def test_expert_delta_is_relative_to_its_own_reference(tiny_args):
    """Drift is measured per domain, never pooled across experts."""
    model = make_composed(tiny_args)
    references = _references(model)
    with torch.no_grad():
        for param in model.experts[1].parameters():
            param.mul_(1.01)
    deltas = expert_delta_norms(model, references)
    assert deltas[0] == pytest.approx(0.0, abs=1e-12)
    assert deltas[1] == pytest.approx(0.01, rel=1e-3)

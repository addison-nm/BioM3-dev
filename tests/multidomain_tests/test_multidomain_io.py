"""Checkpoint round-trip and load-safety gates.

``test_checkpoint_round_trip`` is the regression for the failure this module
exists to prevent: building a different model than the one that was trained,
loading it non-strictly, and never inspecting ``unexpected_keys`` — which
silently discards every trained tensor and leaves a randomly-initialised model
behind. Asserting that the reloaded model reproduces the original's *logits*
catches that; counting keys does not.
"""

import argparse

import pytest
import torch

from biom3.Stage3.multidomain.coupling import AllPairsCoupling
from biom3.Stage3.multidomain.io import (
    ALL_PAIRS,
    MultiDomainSpec,
    build_from_spec,
    build_multidomain_from_checkpoint,
    extract_composed_state_dict,
    load_composed_state_dict,
    read_spec,
    state_dict_fingerprint,
)
from biom3.Stage3.multidomain.model import MultiDomainProteoScribe

from .conftest import NUM_DOMAINS, TINY_CONFIG, make_batch, make_composed


@pytest.fixture
def spec(tiny_args):
    return MultiDomainSpec.from_args(
        tiny_args, NUM_DOMAINS, domain_ids=("PF00501", "PF13193"))


def _randomize(model, seed=5):
    """Move every parameter away from init, coupling and experts alike."""
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for param in model.parameters():
            param.copy_(torch.randn(param.shape, generator=generator) * 0.1)


def _write_checkpoint(tmp_path, model, spec, name="composed.ckpt"):
    path = tmp_path / name
    torch.save(
        {
            "state_dict": {f"model.{k}": v for k, v in model.state_dict().items()},
            "hyper_parameters": {
                "multidomain_spec": spec.to_dict(),
                "multidomain_fingerprint": state_dict_fingerprint(model),
            },
        },
        path,
    )
    return path


# ── the round-trip gate ───────────────────────────────────────────────────


def test_checkpoint_round_trip(tmp_path, tiny_args, spec):
    """Rebuild from the stored spec alone and reproduce the original exactly."""
    original = make_composed(tiny_args)
    _randomize(original)
    original.eval()
    path = _write_checkpoint(tmp_path, original, spec)

    restored, restored_spec = build_multidomain_from_checkpoint(
        path, template_args=tiny_args)
    restored.eval()

    assert restored_spec == spec

    original_state = original.state_dict()
    restored_state = restored.state_dict()
    assert set(original_state) == set(restored_state)
    for key, tensor in original_state.items():
        assert torch.equal(tensor, restored_state[key]), f"tensor {key} differs"

    x, t, y_c = make_batch()
    with torch.no_grad():
        assert torch.equal(original(x, t, y_c), restored(x, t, y_c))


def test_round_trip_survives_a_trained_non_null_coupling(tmp_path, tiny_args, spec):
    """The realistic case: a coupling that has moved off the additive null."""
    original = make_composed(tiny_args)
    generator = torch.Generator().manual_seed(9)
    with torch.no_grad():
        for param in original.coupling.parameters():
            param.add_(torch.randn(param.shape, generator=generator) * 0.2)
    original.eval()
    assert not original.coupling.is_additive_null(tol=0.0)

    path = _write_checkpoint(tmp_path, original, spec)
    restored, _ = build_multidomain_from_checkpoint(path, template_args=tiny_args)
    restored.eval()

    assert not restored.coupling.is_additive_null(tol=0.0)
    x, t, y_c = make_batch()
    with torch.no_grad():
        assert torch.equal(original(x, t, y_c), restored(x, t, y_c))


# ── load safety ───────────────────────────────────────────────────────────


def test_load_rejects_unexpected_keys(tiny_args, spec):
    """A state dict trained against a different coupling must not load silently.

    Its tensors land in ``unexpected_keys``, which ``strict=False`` discards
    without a word. This is the exact shape of the defect being guarded against.
    """
    model = build_from_spec(spec, template_args=tiny_args)
    state = model.state_dict()
    renamed = {
        (k.replace("coupling.out_proj", "coupling.o_NC") if "coupling.out_proj" in k
         else k): v
        for k, v in state.items()
    }
    with pytest.raises(ValueError, match="matched no parameter"):
        load_composed_state_dict(model, renamed)


def test_load_rejects_missing_keys(tiny_args, spec):
    model = build_from_spec(spec, template_args=tiny_args)
    state = {k: v for k, v in model.state_dict().items()
             if not k.startswith("coupling.q_proj")}
    with pytest.raises(ValueError, match="were not populated"):
        load_composed_state_dict(model, state)


def test_load_accepts_an_exact_match(tiny_args, spec):
    model = build_from_spec(spec, template_args=tiny_args)
    load_composed_state_dict(model, model.state_dict())


def test_load_rejects_domain_count_mismatch(tiny_args, spec):
    """A K=2 state dict against a K=3 model fails in both directions at once."""
    two = build_from_spec(spec, template_args=tiny_args)
    three_spec = MultiDomainSpec.from_args(tiny_args, 3)
    three = build_from_spec(three_spec, template_args=tiny_args)
    with pytest.raises(ValueError):
        load_composed_state_dict(three, two.state_dict())


def test_fingerprint_detects_topology_change(tiny_args, spec):
    two = build_from_spec(spec, template_args=tiny_args)
    three = build_from_spec(MultiDomainSpec.from_args(tiny_args, 3),
                            template_args=tiny_args)
    assert state_dict_fingerprint(two) != state_dict_fingerprint(three)
    assert state_dict_fingerprint(two) == state_dict_fingerprint(
        build_from_spec(spec, template_args=tiny_args))


def test_checkpoint_with_mismatched_fingerprint_raises(tmp_path, tiny_args, spec):
    """A spec that does not describe the stored weights fails loudly."""
    model = make_composed(tiny_args)
    path = tmp_path / "bad.ckpt"
    torch.save(
        {
            "state_dict": {f"model.{k}": v for k, v in model.state_dict().items()},
            "hyper_parameters": {
                "multidomain_spec": spec.to_dict(),
                "multidomain_fingerprint": "0" * 40,
            },
        },
        path,
    )
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        build_multidomain_from_checkpoint(path, template_args=tiny_args)


def test_checkpoint_without_spec_raises(tmp_path, tiny_args):
    model = make_composed(tiny_args)
    path = tmp_path / "nospec.ckpt"
    torch.save({"state_dict": model.state_dict()}, path)
    with pytest.raises(ValueError, match="carries no multidomain_spec"):
        build_multidomain_from_checkpoint(path, template_args=tiny_args)


# ── spec ──────────────────────────────────────────────────────────────────


def test_spec_dict_round_trip(spec):
    assert MultiDomainSpec.from_dict(spec.to_dict()) == spec


def test_spec_rejects_unknown_fields(spec):
    payload = spec.to_dict()
    payload["mystery"] = 1
    with pytest.raises(ValueError, match="unknown MultiDomainSpec fields"):
        MultiDomainSpec.from_dict(payload)


def test_spec_overrides_template_architecture(tiny_args, spec):
    """The spec wins over a config that has since moved on."""
    drifted = argparse.Namespace(**{**TINY_CONFIG, "transformer_dim": 64})
    model_args = spec.to_model_args(drifted)
    assert model_args.transformer_dim == TINY_CONFIG["transformer_dim"]
    assert model_args.transformer_dropout == drifted.transformer_dropout


def test_build_from_spec_rejects_unknown_topology(tiny_args, spec):
    bad = MultiDomainSpec.from_dict({**spec.to_dict(), "coupling_topology": "pairwise"})
    with pytest.raises(ValueError, match="unknown coupling_topology"):
        build_from_spec(bad, template_args=tiny_args)


def test_extract_state_dict_handles_both_prefixes(tiny_args):
    model = make_composed(tiny_args)
    raw = model.state_dict()
    prefixed = {f"model.{k}": v for k, v in raw.items()}
    assert set(extract_composed_state_dict({"state_dict": prefixed})) == set(raw)
    assert set(extract_composed_state_dict({"state_dict": raw})) == set(raw)


def test_load_experts_rejects_wrong_count(tiny_args, spec):
    from biom3.Stage3.multidomain.io import load_experts
    with pytest.raises(ValueError, match="expert weight paths"):
        load_experts(spec, ["only-one.bin"], template_args=tiny_args)


def test_load_experts_rejects_sha_mismatch(tmp_path, tiny_args, spec):
    from biom3.Stage3.io import build_model_ProteoScribe
    from biom3.Stage3.multidomain.io import load_experts

    paths = []
    for d in range(NUM_DOMAINS):
        path = tmp_path / f"expert{d}.bin"
        torch.save(build_model_ProteoScribe(tiny_args).state_dict(), path)
        paths.append(str(path))

    with pytest.raises(ValueError, match="sha256 mismatch"):
        load_experts(spec, paths, template_args=tiny_args,
                     expert_sha256=["f" * 64, "f" * 64])


def test_load_experts_is_strict(tmp_path, tiny_args, spec):
    """A truncated expert state dict must raise, not load 19 random tensors."""
    from biom3.Stage3.io import build_model_ProteoScribe
    from biom3.Stage3.multidomain.io import load_experts

    paths = []
    for d in range(NUM_DOMAINS):
        state = build_model_ProteoScribe(tiny_args).state_dict()
        if d == 1:
            state = {k: v for k, v in state.items() if "y_mlp" not in k}
        path = tmp_path / f"expert{d}.bin"
        torch.save(state, path)
        paths.append(str(path))

    with pytest.raises(Exception):
        load_experts(spec, paths, template_args=tiny_args)


def test_loaded_experts_are_distinct_instances(tmp_path, tiny_args, spec):
    """Each domain gets its own expert module, never a shared reference."""
    from biom3.Stage3.io import build_model_ProteoScribe
    from biom3.Stage3.multidomain.io import load_experts

    paths = []
    for d in range(NUM_DOMAINS):
        torch.manual_seed(d)
        path = tmp_path / f"expert{d}.bin"
        torch.save(build_model_ProteoScribe(tiny_args).state_dict(), path)
        paths.append(str(path))

    experts = load_experts(spec, paths, template_args=tiny_args)
    assert experts[0] is not experts[1]
    first = dict(experts[0].named_parameters())
    second = dict(experts[1].named_parameters())
    assert not torch.equal(first["transformer.x_emb_NN.weight"],
                           second["transformer.x_emb_NN.weight"])

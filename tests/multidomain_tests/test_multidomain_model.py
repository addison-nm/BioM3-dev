"""Correctness gates for the composed multidomain decoder.

The two that matter most:

``test_composed_matches_shared_forward`` pins the composed forward pass to the
shared, unmodified single-domain decoder. The composed model reimplements that
forward so it can inject a cross term between layers; without this test the copy
would drift the moment the shared decoder changed, which is exactly how the
reference implementation this design is based on went wrong.

``test_additive_null_is_bit_exact`` asserts the property the whole warm-start
argument rests on: with the coupling's output projections zeroed, the composed
model reproduces the independent experts exactly.
"""

import pytest
import torch

from biom3.Stage3.multidomain import build_multidomain_model
from biom3.Stage3.multidomain.coupling import AllPairsCoupling
from biom3.Stage3.multidomain.model import MultiDomainProteoScribe, PAD_ID

from .conftest import NUM_DOMAINS, SEQ_LEN, make_batch, make_composed, make_experts


def _perturb_module(module, scale=0.05, seed=3):
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for param in module.parameters():
            param.add_(torch.randn(param.shape, generator=generator) * scale)


def _perturb_coupling(composed, scale=0.05, seed=3):
    """Move the coupling off the additive null so it actually contributes."""
    _perturb_module(composed.coupling, scale=scale, seed=seed)


# ── the drift guard ───────────────────────────────────────────────────────


def test_composed_matches_shared_forward(tiny_args):
    """At the additive null each canvas equals the shared decoder's own forward.

    ``experts[d](...)`` runs the unmodified
    ``LinearAttentionTransformerEmbedding.forward``; the composed model runs its
    own copy of that loop. They must agree bit-for-bit.
    """
    experts = make_experts(tiny_args)
    composed = MultiDomainProteoScribe(
        experts,
        AllPairsCoupling(NUM_DOMAINS, tiny_args.transformer_depth,
                         tiny_args.transformer_dim, tiny_args.transformer_heads),
    )
    composed.eval()
    x, t, y_c = make_batch()

    with torch.no_grad():
        composed_logits = composed(x, t, y_c)
        for d in range(NUM_DOMAINS):
            shared_logits = experts[d](x[:, d], t, y_c[:, d])
            assert torch.equal(composed_logits[:, d], shared_logits), (
                f"domain {d}: composed forward diverged from the shared decoder"
            )


def test_standalone_logits_match_shared_forward(tiny_args):
    experts = make_experts(tiny_args)
    composed = MultiDomainProteoScribe(
        experts,
        AllPairsCoupling(NUM_DOMAINS, tiny_args.transformer_depth,
                         tiny_args.transformer_dim, tiny_args.transformer_heads),
    )
    composed.eval()
    x, t, y_c = make_batch()

    with torch.no_grad():
        for d in range(NUM_DOMAINS):
            assert torch.equal(
                composed.standalone_logits(d, x[:, d], t, y_c[:, d]),
                experts[d](x[:, d], t, y_c[:, d]),
            )


# ── additive null ─────────────────────────────────────────────────────────


def test_coupling_starts_at_additive_null(composed):
    assert composed.coupling.is_additive_null(tol=0.0)


def test_additive_null_is_bit_exact(composed, batch):
    """couple=True, couple=False and the standalone experts all agree exactly."""
    x, t, y_c = batch
    with torch.no_grad():
        coupled = composed(x, t, y_c, couple=True)
        uncoupled = composed(x, t, y_c, couple=False)
        for d in range(NUM_DOMAINS):
            standalone = composed.standalone_logits(d, x[:, d], t, y_c[:, d])
            assert torch.equal(coupled[:, d], uncoupled[:, d])
            assert torch.equal(coupled[:, d], standalone)


def test_coupling_perturbation_moves_logits(composed, batch):
    """The converse check: a non-null coupling must actually change the output.

    Without this, a coupling that was never wired into the forward pass would
    pass every additive-null assertion.
    """
    x, t, y_c = batch
    with torch.no_grad():
        before = composed(x, t, y_c, couple=True)
    _perturb_coupling(composed)
    with torch.no_grad():
        after = composed(x, t, y_c, couple=True)

    assert not composed.coupling.is_additive_null(tol=0.0)
    assert not torch.equal(before, after)
    for d in range(NUM_DOMAINS):
        assert not torch.equal(before[:, d], after[:, d]), (
            f"domain {d} unaffected by the coupling — its cross term is not wired in"
        )


def test_uncoupled_forward_ignores_perturbed_coupling(composed, batch):
    """couple=False is a genuine ablation, not merely the un-perturbed path."""
    x, t, y_c = batch
    with torch.no_grad():
        before = composed(x, t, y_c, couple=False)
    _perturb_coupling(composed)
    with torch.no_grad():
        after = composed(x, t, y_c, couple=False)
    assert torch.equal(before, after)


# ── cross-domain wiring ───────────────────────────────────────────────────


def test_cross_term_ignores_masked_partner_positions(tiny_args):
    """The coupling must read only its partner's real positions.

    Tested on the coupling directly rather than through the model: perturbing a
    partner canvas's PAD *tokens* also shifts that partner's real-position hidden
    states via its own self-attention, so a model-level check cannot isolate the
    mask. Here the partner's hidden state is edited only inside the masked region.
    """
    coupling = AllPairsCoupling(NUM_DOMAINS, tiny_args.transformer_depth,
                                tiny_args.transformer_dim, tiny_args.transformer_heads)
    _perturb_module(coupling)

    torch.manual_seed(11)
    dim = tiny_args.transformer_dim
    layer_inputs = [torch.randn(2, SEQ_LEN, dim) for _ in range(NUM_DOMAINS)]
    real_masks = [torch.zeros(2, SEQ_LEN, dtype=torch.bool) for _ in range(NUM_DOMAINS)]
    for mask in real_masks:
        mask[:, :16] = True

    with torch.no_grad():
        baseline = coupling.cross_terms(0, layer_inputs, real_masks)

        perturbed = [t.clone() for t in layer_inputs]
        perturbed[1][:, 16:] += torch.randn(2, SEQ_LEN - 16, dim) * 10.0
        masked_out = coupling.cross_terms(0, perturbed, real_masks)

    assert torch.equal(baseline[0], masked_out[0]), (
        "domain 0's cross term read domain 1's masked-out positions"
    )

    with torch.no_grad():
        perturbed_real = [t.clone() for t in layer_inputs]
        perturbed_real[1][:, :16] += torch.randn(2, 16, dim)
        changed = coupling.cross_terms(0, perturbed_real, real_masks)
    assert not torch.equal(baseline[0], changed[0])


def test_partner_real_positions_do_affect_logits(composed, batch):
    """The complement of the PAD test: real partner residues must matter."""
    _perturb_coupling(composed)
    x, t, y_c = batch
    real_masks = x != PAD_ID

    mutated = x.clone()
    mutated[:, 1, :5] = (mutated[:, 1, :5] % 20) + 2

    with torch.no_grad():
        original = composed(x, t, y_c, real_masks=real_masks)
        changed = composed(mutated, t, y_c, real_masks=real_masks)

    assert not torch.equal(original[:, 0], changed[:, 0])


def test_per_domain_timesteps_are_routed_independently(composed, batch):
    """t may be [B] (shared) or [B, K]; the [B, K] form must reach each canvas."""
    x, t, y_c = batch
    shared = t.view(-1, 1).expand(-1, NUM_DOMAINS).contiguous()
    per_domain = shared.clone()
    per_domain[:, 1] = (per_domain[:, 1] + 7) % SEQ_LEN

    with torch.no_grad():
        from_vector = composed(x, t, y_c)
        from_matrix = composed(x, shared, y_c)
        altered = composed(x, per_domain, y_c)

    assert torch.equal(from_vector, from_matrix)
    assert torch.equal(from_vector[:, 0], altered[:, 0])
    assert not torch.equal(from_vector[:, 1], altered[:, 1])


# ── module hygiene ────────────────────────────────────────────────────────


def test_no_duplicate_state_dict_entries(composed, tiny_args):
    """Each expert tensor is stored once — no aliased re-registration."""
    state = composed.state_dict()
    pointers = [tensor.data_ptr() for tensor in state.values()]
    assert len(pointers) == len(set(pointers))

    expert_keys = [k for k in state if k.startswith("experts.")]
    single = len(build_multidomain_model(tiny_args, NUM_DOMAINS)
                 .experts[0].state_dict())
    assert len(expert_keys) == NUM_DOMAINS * single


def test_all_parameters_receive_gradients(composed, batch):
    """No unused parameters, so DDP never needs find_unused_parameters=True."""
    _perturb_coupling(composed)
    composed.train()
    x, t, y_c = batch
    composed(x, t, y_c).sum().backward()

    missing = [name for name, param in composed.named_parameters() if param.grad is None]
    assert missing == [], f"parameters received no gradient: {missing[:5]}"


def test_shape_contract(composed, batch):
    x, t, y_c = batch
    with torch.no_grad():
        logits = composed(x, t, y_c)
    assert logits.shape == (x.size(0), NUM_DOMAINS,
                            composed.experts[0].transformer.out.out_features, SEQ_LEN)


def test_rejects_mismatched_domain_count(composed, batch):
    x, t, y_c = batch
    with pytest.raises(ValueError, match="expected x of shape"):
        composed(x[:, :1], t, y_c)
    with pytest.raises(ValueError, match="expected y_c of shape"):
        composed(x, t, y_c[:, :1])


# ── general K ─────────────────────────────────────────────────────────────


def test_k3_ordered_pairs_and_additive_null(tiny_args):
    composed = make_composed(tiny_args, num_domains=3)
    assert len(composed.coupling.pairs) == 6
    assert composed.coupling.is_additive_null(tol=0.0)

    x, t, y_c = make_batch(num_domains=3)
    with torch.no_grad():
        coupled = composed(x, t, y_c, couple=True)
        for d in range(3):
            assert torch.equal(
                coupled[:, d], composed.standalone_logits(d, x[:, d], t, y_c[:, d]))


def test_k2_coupling_parameter_count_matches_reference_shape(tiny_args):
    """K=2 has 2 ordered pairs; the count scales as K*(K-1)."""
    two = AllPairsCoupling(2, tiny_args.transformer_depth,
                           tiny_args.transformer_dim, tiny_args.transformer_heads)
    three = AllPairsCoupling(3, tiny_args.transformer_depth,
                             tiny_args.transformer_dim, tiny_args.transformer_heads)
    assert len(two.pairs) == 2
    assert len(three.pairs) == 6

    def projection_params(coupling):
        return sum(p.numel() for name, p in coupling.named_parameters()
                   if not name.startswith("norms."))

    assert projection_params(three) == 3 * projection_params(two)


def test_coupling_rejects_invalid_shapes(tiny_args):
    with pytest.raises(ValueError, match="num_domains must be >= 2"):
        AllPairsCoupling(1, 2, 32, 4)
    with pytest.raises(ValueError, match="not divisible by heads"):
        AllPairsCoupling(2, 2, 32, 7)


def test_model_rejects_expert_coupling_mismatch(tiny_args):
    experts = make_experts(tiny_args, num_domains=2)
    with pytest.raises(ValueError, match="but the coupling is built for"):
        MultiDomainProteoScribe(
            experts,
            AllPairsCoupling(3, tiny_args.transformer_depth,
                             tiny_args.transformer_dim, tiny_args.transformer_heads),
        )

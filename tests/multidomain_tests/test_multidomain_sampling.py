"""Composed generation tests.

``test_timestep_follows_the_revealed_count`` is the one that matters. Feeding the
raw step counter instead of ``seq_len - (n - step)`` runs the model far outside
the timestep range it was trained on, and the output still looks like a protein —
there is no crash and no obviously wrong shape, so only an explicit assertion
catches it.
"""

import pytest
import torch

from biom3.Stage3.inpaint import END_ID, PAD_ID, RUNTIME_TOKENS, START_ID
from biom3.Stage3.multidomain.sampling import (
    assemble_domains,
    build_domain_canvases,
    decode_assemblies,
    decode_domain,
    generate_multidomain,
)

from .conftest import EMB_DIM, NUM_DOMAINS, SEQ_LEN, make_composed


LENGTHS = [[12, 7]]


class _RecordingModel(torch.nn.Module):
    """Stands in for the composed decoder, recording what it was called with."""

    def __init__(self, num_domains=NUM_DOMAINS, num_classes=len(RUNTIME_TOKENS),
                 seq_len=SEQ_LEN):
        super().__init__()
        self.num_domains = num_domains
        self.num_classes = num_classes
        self.seq_len = seq_len
        self.calls = []
        self._probe = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x, t, y_c, real_masks=None, couple=True):
        self.calls.append({
            "clock": t.clone(),
            "state": x.clone(),
            "real_masks": None if real_masks is None else real_masks.clone(),
            "couple": couple,
        })
        batch = x.size(0)
        return torch.zeros(batch, self.num_domains, self.num_classes, self.seq_len)


# ── canvases ──────────────────────────────────────────────────────────────


def test_canvases_place_structural_tokens():
    states, paths, offsets = build_domain_canvases([12, 7], SEQ_LEN)
    assert states.shape == (2, SEQ_LEN)
    for d, length in enumerate([12, 7]):
        assert states[d, 0].item() == START_ID
        assert states[d, length + 1].item() == END_ID
        assert torch.all(states[d, length + 2:] == PAD_ID)
        # Structural positions are never scheduled for unmasking.
        assert paths[d, 0].item() == -1
        assert paths[d, length + 1].item() == -1
        assert torch.all(paths[d, length + 2:] == -1)


def test_canvas_offsets_are_seq_len_minus_length():
    _, _, offsets = build_domain_canvases([12, 7], SEQ_LEN)
    assert offsets.tolist() == [SEQ_LEN - 12, SEQ_LEN - 7]


def test_path_is_a_permutation_over_the_generated_span():
    _, paths, offsets = build_domain_canvases([12, 7], SEQ_LEN)
    for d, length in enumerate([12, 7]):
        values = sorted(v for v in paths[d].tolist() if v != -1)
        offset = int(offsets[d])
        assert values == list(range(offset, offset + length))


def test_canvas_rejects_a_domain_too_long():
    with pytest.raises(ValueError, match="canvas holds"):
        build_domain_canvases([SEQ_LEN], SEQ_LEN)


def test_canvas_rejects_an_empty_domain():
    with pytest.raises(ValueError, match="length must be >= 1"):
        build_domain_canvases([0], SEQ_LEN)


# ── the diffusion clock ───────────────────────────────────────────────────


def test_timestep_follows_the_revealed_count():
    """t must be seq_len - (n_d - step), not the bare step counter."""
    model = _RecordingModel()
    y_c = torch.zeros(1, NUM_DOMAINS, EMB_DIM)
    generate_multidomain(model, y_c, LENGTHS, seq_len=SEQ_LEN, device="cpu",
                         token_strategy="argmax")

    n0, n1 = LENGTHS[0]
    assert len(model.calls) == max(n0, n1)
    for step, call in enumerate(model.calls):
        clock = call["clock"]
        assert clock.shape == (1, NUM_DOMAINS)
        assert clock[0, 0].item() == SEQ_LEN - (n0 - step)
        # The shorter canvas finishes first and then holds at seq_len.
        assert clock[0, 1].item() == min(SEQ_LEN - (n1 - step), SEQ_LEN)


def test_finished_canvas_clock_never_exceeds_seq_len():
    model = _RecordingModel()
    y_c = torch.zeros(1, NUM_DOMAINS, EMB_DIM)
    generate_multidomain(model, y_c, LENGTHS, seq_len=SEQ_LEN, device="cpu",
                         token_strategy="argmax")
    assert all(int(c["clock"].max()) <= SEQ_LEN for c in model.calls)


def test_step_count_is_the_longest_domain():
    model = _RecordingModel()
    y_c = torch.zeros(1, NUM_DOMAINS, EMB_DIM)
    generate_multidomain(model, y_c, [[5, 20]], seq_len=SEQ_LEN, device="cpu",
                         token_strategy="argmax")
    assert len(model.calls) == 20


# ── structural tokens are never overwritten ───────────────────────────────


def test_structural_positions_survive_the_trajectory():
    """<START>, <END> and the PAD tail must be intact at every step.

    A finished canvas has no path entry matching its clock; without an explicit
    guard the position lookup falls through to index 0 and quietly overwrites
    <START>.
    """
    model = _RecordingModel()
    y_c = torch.zeros(1, NUM_DOMAINS, EMB_DIM)
    states, _ = generate_multidomain(
        model, y_c, LENGTHS, seq_len=SEQ_LEN, device="cpu",
        token_strategy="argmax")

    for call in model.calls:
        seen = call["state"]
        for d, length in enumerate(LENGTHS[0]):
            assert seen[0, d, 0].item() == START_ID
            assert seen[0, d, length + 1].item() == END_ID
            assert torch.all(seen[0, d, length + 2:] == PAD_ID)

    for d, length in enumerate(LENGTHS[0]):
        assert states[0, d, 0].item() == START_ID
        assert states[0, d, length + 1].item() == END_ID
        assert torch.all(states[0, d, length + 2:] == PAD_ID)


def test_real_masks_cover_start_residues_and_end_only():
    model = _RecordingModel()
    y_c = torch.zeros(1, NUM_DOMAINS, EMB_DIM)
    generate_multidomain(model, y_c, LENGTHS, seq_len=SEQ_LEN, device="cpu",
                         token_strategy="argmax")
    masks = model.calls[0]["real_masks"]
    for d, length in enumerate(LENGTHS[0]):
        assert torch.all(masks[0, d, :length + 2])
        assert not torch.any(masks[0, d, length + 2:])


def test_real_masks_are_constant_across_the_trajectory():
    model = _RecordingModel()
    y_c = torch.zeros(1, NUM_DOMAINS, EMB_DIM)
    generate_multidomain(model, y_c, LENGTHS, seq_len=SEQ_LEN, device="cpu",
                         token_strategy="argmax")
    first = model.calls[0]["real_masks"]
    assert all(torch.equal(first, c["real_masks"]) for c in model.calls)


# ── generated content ─────────────────────────────────────────────────────


def test_every_generated_position_is_filled(tiny_args):
    composed = make_composed(tiny_args)
    y_c = torch.randn(1, NUM_DOMAINS, EMB_DIM)
    states, _ = generate_multidomain(
        composed, y_c, LENGTHS, seq_len=SEQ_LEN, device="cpu",
        token_strategy="argmax")
    from biom3.Stage3.inpaint import MASK_ID
    for d, length in enumerate(LENGTHS[0]):
        assert not torch.any(states[0, d, 1:length + 1] == MASK_ID)


def test_restrict_to_residues_excludes_structural_tokens(tiny_args):
    composed = make_composed(tiny_args)
    y_c = torch.randn(2, NUM_DOMAINS, EMB_DIM)
    states, _ = generate_multidomain(
        composed, y_c, [[12, 7], [10, 9]], seq_len=SEQ_LEN, device="cpu",
        token_strategy="argmax", restrict_to_residues=True)
    structural = {START_ID, END_ID, PAD_ID, 0}
    for b, lengths in enumerate([[12, 7], [10, 9]]):
        for d, length in enumerate(lengths):
            interior = states[b, d, 1:length + 1].tolist()
            assert not (set(interior) & structural)


def test_couple_false_is_forwarded(tiny_args):
    model = _RecordingModel()
    y_c = torch.zeros(1, NUM_DOMAINS, EMB_DIM)
    generate_multidomain(model, y_c, LENGTHS, seq_len=SEQ_LEN, device="cpu",
                         token_strategy="argmax", couple=False)
    assert all(c["couple"] is False for c in model.calls)


def test_coupling_changes_the_output(tiny_args):
    """The zero-coupling ablation must actually differ once trained."""
    composed = make_composed(tiny_args)
    gen = torch.Generator().manual_seed(7)
    with torch.no_grad():
        for param in composed.coupling.parameters():
            param.add_(torch.randn(param.shape, generator=gen) * 0.5)

    y_c = torch.randn(1, NUM_DOMAINS, EMB_DIM)
    kwargs = dict(seq_len=SEQ_LEN, device="cpu", token_strategy="argmax")
    coupled, _ = generate_multidomain(
        composed, y_c, LENGTHS, generator=torch.Generator().manual_seed(1), **kwargs)
    uncoupled, _ = generate_multidomain(
        composed, y_c, LENGTHS, generator=torch.Generator().manual_seed(1),
        couple=False, **kwargs)
    assert not torch.equal(coupled, uncoupled)


# ── determinism ───────────────────────────────────────────────────────────


def test_argmax_generation_is_deterministic(tiny_args):
    composed = make_composed(tiny_args)
    y_c = torch.randn(1, NUM_DOMAINS, EMB_DIM)
    kwargs = dict(seq_len=SEQ_LEN, device="cpu", token_strategy="argmax")
    first, _ = generate_multidomain(
        composed, y_c, LENGTHS, generator=torch.Generator().manual_seed(3), **kwargs)
    second, _ = generate_multidomain(
        composed, y_c, LENGTHS, generator=torch.Generator().manual_seed(3), **kwargs)
    assert torch.equal(first, second)


def test_sample_seeds_make_sampling_reproducible(tiny_args):
    composed = make_composed(tiny_args)
    y_c = torch.randn(2, NUM_DOMAINS, EMB_DIM)
    lengths = [[12, 7], [12, 7]]
    kwargs = dict(seq_len=SEQ_LEN, device="cpu", token_strategy="sample")
    first, _ = generate_multidomain(
        composed, y_c, lengths, sample_seeds=[11, 22],
        generator=torch.Generator().manual_seed(4), **kwargs)
    second, _ = generate_multidomain(
        composed, y_c, lengths, sample_seeds=[11, 22],
        generator=torch.Generator().manual_seed(4), **kwargs)
    assert torch.equal(first, second)


def test_different_seeds_give_different_sequences(tiny_args):
    composed = make_composed(tiny_args)
    y_c = torch.randn(1, NUM_DOMAINS, EMB_DIM)
    kwargs = dict(seq_len=SEQ_LEN, device="cpu", token_strategy="sample",
                  generator=torch.Generator().manual_seed(5))
    first, _ = generate_multidomain(composed, y_c, LENGTHS, sample_seeds=[1],
                                    **dict(kwargs, generator=torch.Generator().manual_seed(5)))
    second, _ = generate_multidomain(composed, y_c, LENGTHS, sample_seeds=[999],
                                     **dict(kwargs, generator=torch.Generator().manual_seed(5)))
    assert not torch.equal(first, second)


# ── decoding and assembly ─────────────────────────────────────────────────


def test_decode_stops_at_end():
    tokens = torch.tensor([START_ID] + [RUNTIME_TOKENS.index("A")] * 3
                          + [END_ID, PAD_ID, PAD_ID])
    assert decode_domain(tokens) == "AAA"


def test_decode_assemblies_shape(tiny_args):
    composed = make_composed(tiny_args)
    y_c = torch.randn(2, NUM_DOMAINS, EMB_DIM)
    states, _ = generate_multidomain(
        composed, y_c, [[12, 7], [10, 9]], seq_len=SEQ_LEN, device="cpu",
        token_strategy="argmax")
    decoded = decode_assemblies(states)
    assert len(decoded) == 2
    assert [len(d) for d in decoded] == [NUM_DOMAINS, NUM_DOMAINS]
    assert [len(s) for s in decoded[0]] == [12, 7]
    assert [len(s) for s in decoded[1]] == [10, 9]


def test_assemble_is_concatenation():
    assert assemble_domains(["AAA", "CCC"]) == "AAACCC"
    assert assemble_domains(["AAA", "CCC"], linker="GG") == "AAAGGCCC"


def test_generated_lengths_match_the_request(tiny_args):
    composed = make_composed(tiny_args)
    y_c = torch.randn(1, NUM_DOMAINS, EMB_DIM)
    states, _ = generate_multidomain(
        composed, y_c, LENGTHS, seq_len=SEQ_LEN, device="cpu",
        token_strategy="argmax")
    decoded = decode_assemblies(states)[0]
    assert [len(s) for s in decoded] == LENGTHS[0]
    assert len(assemble_domains(decoded)) == sum(LENGTHS[0])


# ── input validation ──────────────────────────────────────────────────────


def test_rejects_domain_count_mismatch(tiny_args):
    composed = make_composed(tiny_args)
    y_c = torch.randn(1, NUM_DOMAINS, EMB_DIM)
    with pytest.raises(ValueError, match="lengths has 3 domains"):
        generate_multidomain(composed, y_c, [[5, 5, 5]], seq_len=SEQ_LEN,
                             device="cpu")


def test_rejects_mismatched_y_c(tiny_args):
    composed = make_composed(tiny_args)
    with pytest.raises(ValueError, match="y_c must be"):
        generate_multidomain(composed, torch.randn(2, NUM_DOMAINS, EMB_DIM),
                             LENGTHS, seq_len=SEQ_LEN, device="cpu")


def test_rejects_one_dimensional_lengths(tiny_args):
    composed = make_composed(tiny_args)
    y_c = torch.randn(1, NUM_DOMAINS, EMB_DIM)
    with pytest.raises(ValueError, match=r"lengths must be \[B, K\]"):
        generate_multidomain(composed, y_c, [12, 7], seq_len=SEQ_LEN, device="cpu")

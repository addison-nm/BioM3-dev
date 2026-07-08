"""Unit tests for the ProteoScribe likelihood estimator (Stage3/tools.py).

Uses a tiny randomly-initialised model on CPU — no weights required. The
tests target the estimator's accounting and invariants (position roles,
determinism, quadrature reduction) rather than absolute likelihood values,
which are meaningless for random weights.
"""

import argparse

import pytest
import torch

from biom3.Stage3.tools import (
    LikelihoodConfig,
    ProteoScribeLikelihoodEstimator,
    _build_corruptions,
    _build_masked_levels,
    _classify_sequence,
    _CONTEXT,
    _QUERY,
    _UNKNOWN,
)
from biom3.rl.grpo import END_ID, MASK_ID, PAD_ID, START_ID, TOK2ID


TINY_CONFIG = {
    "model_option": "transformer",
    "num_classes": 29,
    "num_y_class_labels": 6,
    "diffusion_steps": 64,
    "image_size": 8,
    "num_steps": 1,
    "actnorm": False,
    "perm_channel": "none",
    "perm_length": "reverse",
    "input_dp_rate": 0.0,
    "transformer_dim": 32,
    "transformer_heads": 4,
    "transformer_depth": 2,
    "transformer_blocks": 1,
    "transformer_dropout": 0.0,
    "transformer_reversible": False,
    "transformer_local_heads": 2,
    "transformer_local_size": 16,
    "text_emb_dim": 32,
    "facilitator": "MMD",
}
SEQ_LEN = TINY_CONFIG["diffusion_steps"]
EMB_DIM = TINY_CONFIG["text_emb_dim"]


@pytest.fixture(scope="module")
def estimator():
    return ProteoScribeLikelihoodEstimator.from_weights(
        config=dict(TINY_CONFIG), weights_path=None, device="cpu"
    )


@pytest.fixture
def z_c():
    torch.manual_seed(0)
    return torch.randn(EMB_DIM)


# ── classification ────────────────────────────────────────────────────────


def test_classify_wraps_special_tokens_and_pads():
    cfg = LikelihoodConfig()
    ids, roles = _classify_sequence("ACD", SEQ_LEN, None, cfg, torch.device("cpu"))
    assert ids.shape == (SEQ_LEN,)
    assert ids[0].item() == START_ID and roles[0].item() == _CONTEXT
    assert [ids[i].item() for i in (1, 2, 3)] == [TOK2ID[c] for c in "ACD"]
    assert all(roles[i].item() == _QUERY for i in (1, 2, 3))
    assert ids[4].item() == END_ID and roles[4].item() == _QUERY
    assert torch.all(ids[5:] == PAD_ID)
    # PAD tail is fixed context by default (score_padding=False).
    assert torch.all(roles[5:] == _CONTEXT)


def test_classify_mask_char_is_unknown():
    cfg = LikelihoodConfig()
    ids, roles = _classify_sequence("A#D", SEQ_LEN, None, cfg, torch.device("cpu"))
    assert roles[2].item() == _UNKNOWN
    assert ids[2].item() == MASK_ID


def test_classify_context_mask_pins_positions():
    cfg = LikelihoodConfig()
    ctx = [True, False, False]
    ids, roles = _classify_sequence("ACD", SEQ_LEN, ctx, cfg, torch.device("cpu"))
    assert roles[1].item() == _CONTEXT   # first residue pinned
    assert roles[2].item() == _QUERY
    assert roles[3].item() == _QUERY


def test_score_end_false_holds_end_as_context():
    cfg = LikelihoodConfig(score_end=False)
    ids, roles = _classify_sequence("ACD", SEQ_LEN, None, cfg, torch.device("cpu"))
    assert ids[4].item() == END_ID
    assert roles[4].item() == _CONTEXT          # END not scored
    assert all(roles[i].item() == _QUERY for i in (1, 2, 3))


def test_score_start_true_scores_start():
    cfg = LikelihoodConfig(score_start=True)
    ids, roles = _classify_sequence("ACD", SEQ_LEN, None, cfg, torch.device("cpu"))
    assert ids[0].item() == START_ID
    assert roles[0].item() == _QUERY            # START scored


def test_score_end_false_excludes_end_from_query_count(estimator, z_c):
    cfg = LikelihoodConfig(n_quadrature=6, n_repeats=1, seed=7)
    with_end = estimator.estimate("MKTAYIAK", z_c, config=cfg)
    cfg_no_end = LikelihoodConfig(n_quadrature=6, n_repeats=1, seed=7, score_end=False)
    no_end = estimator.estimate("MKTAYIAK", z_c, config=cfg_no_end)
    assert with_end.n_query == 9                # 8 residues + END
    assert no_end.n_query == 8                  # END dropped from the query


def test_loop_only_no_end(estimator, z_c):
    """The motivating case: score only an interior loop, END held as context."""
    cfg = LikelihoodConfig(n_quadrature=6, n_repeats=1, seed=7, score_end=False)
    ctx = [True] * 8
    for j in (3, 4, 5):
        ctx[j] = False
    res = estimator.estimate("MKTAYIAK", z_c, context_mask=ctx, config=cfg)
    assert res.n_query == 3                     # exactly the 3 loop residues


def test_classify_score_padding_promotes_pad_to_query():
    cfg = LikelihoodConfig(score_padding=True)
    _, roles = _classify_sequence("ACD", SEQ_LEN, None, cfg, torch.device("cpu"))
    assert torch.any(roles[5:] == _QUERY)


def test_classify_rejects_unknown_residue():
    cfg = LikelihoodConfig()
    with pytest.raises(ValueError, match="unknown residue"):
        _classify_sequence("AC1", SEQ_LEN, None, cfg, torch.device("cpu"))


# ── masked-count levels ───────────────────────────────────────────────────


def test_levels_are_integer_masked_counts_summing_to_one():
    cfg = LikelihoodConfig(n_quadrature=8)
    m, w = _build_masked_levels(cfg, Q=20, device=torch.device("cpu"))
    assert m.numel() == 8
    assert torch.all((m >= 1) & (m <= 20))
    assert m.dtype == torch.long
    assert pytest.approx(1.0, abs=1e-6) == float(w.sum())


def test_levels_use_all_when_nquadrature_exceeds_Q():
    cfg = LikelihoodConfig(n_quadrature=64)
    m, w = _build_masked_levels(cfg, Q=5, device=torch.device("cpu"))
    assert m.tolist() == [1, 2, 3, 4, 5]     # every level, exact


def test_explicit_grid_requires_points():
    cfg = LikelihoodConfig(quadrature_grid="explicit", quadrature_points=None)
    with pytest.raises(ValueError, match="requires quadrature_points"):
        _build_masked_levels(cfg, Q=10, device=torch.device("cpu"))


# ── end-to-end estimation ─────────────────────────────────────────────────


def test_estimate_basic_fields(estimator, z_c):
    cfg = LikelihoodConfig(n_quadrature=8, n_repeats=2, seed=1)
    res = estimator.estimate("MKTAYIAK", z_c, config=cfg)
    # query = 8 residues + <END>; <START> and PAD tail are context.
    assert res.n_query == 9
    assert res.n_unknown == 0
    assert res.n_context == SEQ_LEN - res.n_query
    assert res.log_likelihood < 0                      # log-prob is negative
    assert res.log_likelihood_std >= 0
    assert res.bits_per_residue > 0
    assert res.perplexity > 1.0


def test_estimate_is_deterministic_with_seed(estimator, z_c):
    cfg = LikelihoodConfig(n_quadrature=6, n_repeats=3, seed=123)
    a = estimator.estimate("MKTAYIAK", z_c, config=cfg)
    b = estimator.estimate("MKTAYIAK", z_c, config=cfg)
    assert a.log_likelihood == pytest.approx(b.log_likelihood, abs=1e-5)


def test_unknown_positions_reduce_query_count(estimator, z_c):
    cfg = LikelihoodConfig(n_quadrature=6, n_repeats=1, seed=7)
    full = estimator.estimate("MKTAYIAK", z_c, config=cfg)
    masked = estimator.estimate("MKT##IAK", z_c, config=cfg)
    assert masked.n_unknown == 2
    assert masked.n_query == full.n_query - 2


def test_context_mask_reduces_query_count(estimator, z_c):
    cfg = LikelihoodConfig(n_quadrature=6, n_repeats=1, seed=7)
    ctx = [True, True, False, False, False, False, False, False]
    res = estimator.estimate("MKTAYIAK", z_c, context_mask=ctx, config=cfg)
    # 2 pinned residues drop from the 9-position query (8 residues + END).
    assert res.n_query == 7


def test_id_tensor_input_matches_string(estimator, z_c):
    cfg = LikelihoodConfig(n_quadrature=6, n_repeats=1, seed=42)
    ids, _ = _classify_sequence("MKTAYIAK", SEQ_LEN, None, cfg, torch.device("cpu"))
    from_str = estimator.estimate("MKTAYIAK", z_c, config=cfg)
    from_ids = estimator.estimate(ids, z_c, config=cfg)
    assert from_ids.n_query == from_str.n_query
    assert from_ids.log_likelihood == pytest.approx(from_str.log_likelihood, abs=1e-5)


def test_empty_query_raises(estimator, z_c):
    cfg = LikelihoodConfig()
    ctx = [True, True, True]
    with pytest.raises(ValueError, match="no QUERY positions"):
        estimator.estimate("ACD", z_c, context_mask=ctx,
                            config=LikelihoodConfig(add_special_tokens=False))


# ── corruptions: context stays concrete (regression for tools.py bug) ──────


def test_corruptions_keep_context_positions_concrete():
    """Regression: _build_corruptions must never mask CONTEXT positions.

    <START>, the PAD tail, and any pinned context must keep their concrete
    token in x_t (the time index claims they are revealed). The original code
    seeded the mask from ``roles != _QUERY``, masking CONTEXT too — this test
    fails on that code and passes on the fix.
    """
    cfg = LikelihoodConfig(n_quadrature=6, n_repeats=2, seed=3)
    dev = torch.device("cpu")
    # "MK#TA": pin the first residue as context; '#' is an UNKNOWN position,
    # so all three role types are present (plus <START>/<END>/PAD).
    ctx = [True, False, False, False, False]
    ids, roles = _classify_sequence("MK#TA", SEQ_LEN, ctx, cfg, dev)
    Q = int((roles == _QUERY).sum())
    m, w = _build_masked_levels(cfg, Q, dev)
    gen = torch.Generator(device=dev).manual_seed(0)
    corruptions, _ = _build_corruptions(ids, roles, m, w, cfg, gen, dev)

    context_pos = roles == _CONTEXT
    unknown_pos = roles == _UNKNOWN
    query_pos = roles == _QUERY
    assert corruptions and context_pos.any() and unknown_pos.any() and query_pos.any()
    for c in corruptions:
        x_t = c["x_t"]
        # CONTEXT positions keep their concrete token (the bug masked these).
        assert torch.equal(x_t[context_pos], ids[context_pos])
        # UNKNOWN positions are always masked.
        assert torch.all(x_t[unknown_pos] == MASK_ID)
        # Every query position is either its concrete token (revealed) or MASK.
        revealed_q = query_pos & (x_t == ids)
        masked_q = query_pos & (x_t == MASK_ID)
        assert torch.all(revealed_q | masked_q | ~query_pos)
        # Only currently-masked query positions are scored.
        assert torch.equal(c["score_mask"], masked_q)


def test_no_empty_corruptions():
    """Every corruption masks >= 1 query position (the masked-count grid can't
    produce the degenerate 'reveal everything, score nothing' levels the old
    continuous-t grid did for small Q)."""
    cfg = LikelihoodConfig(n_quadrature=32, n_repeats=2, seed=0)
    dev = torch.device("cpu")
    ids, roles = _classify_sequence("AKQ", SEQ_LEN, None, cfg, dev)   # tiny Q
    Q = int((roles == _QUERY).sum())
    m, w = _build_masked_levels(cfg, Q, dev)
    corruptions, _ = _build_corruptions(ids, roles, m, w, cfg,
                                        torch.Generator(device=dev).manual_seed(0), dev)
    for c in corruptions:
        assert int(c["score_mask"].sum()) >= 1


# ── weighting correctness: uniform model → exactly log2(V) bits/residue ─────


class _UniformModel(torch.nn.Module):
    """Context-independent model: equal logits over all V classes."""

    def __init__(self, v):
        super().__init__()
        self.v = v

    def forward(self, x, t, y_c):
        return torch.zeros(x.shape[0], self.v, x.shape[1])


def test_uniform_model_gives_exact_bits_per_residue():
    """A uniform model's ELBO is tight and equals -Q·log V, i.e. exactly
    log2(V) bits/residue. This validates the Q·w/m weighting end-to-end;
    the old 1/t weighting failed this by ~0.2 bits and did not converge."""
    import math

    est = ProteoScribeLikelihoodEstimator.from_weights(
        config=dict(TINY_CONFIG), weights_path=None, device="cpu"
    )
    est.model = _UniformModel(TINY_CONFIG["num_classes"]).eval()
    z = torch.zeros(EMB_DIM)
    expected = math.log2(TINY_CONFIG["num_classes"])
    for nq in (4, 16, 64):
        res = est.estimate("MKTAYIAKQR", z,
                           config=LikelihoodConfig(n_quadrature=nq, n_repeats=1, seed=0))
        assert res.bits_per_residue == pytest.approx(expected, abs=1e-4)
        assert res.log_likelihood == pytest.approx(-res.n_query * math.log(TINY_CONFIG["num_classes"]), abs=1e-3)

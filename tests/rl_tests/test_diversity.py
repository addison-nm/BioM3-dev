"""Unit tests for biom3.rl.diversity and biom3.rl.rewards.diversity.

Covers:

- The BioPython-aligner-backed ``pairwise_identity`` and ``diversity_stats``
  helpers (alignment of equal- and unequal-length sequences, group
  splitting, edge cases).
- The ``DiversityReward`` modes and ``last_components()`` integration.
- Back-compat of ``from biom3.rl.rewards import ...`` after the
  ``rewards.py`` → ``rewards/`` subpackage refactor.
"""

import inspect

import numpy as np
import pytest

biopython = pytest.importorskip("Bio.Align")  # diversity needs biopython

from biom3.rl.diversity import diversity_stats, pairwise_identity
from biom3.rl.rewards import (
    AAFractionReward,
    CompositeReward,
    DiversityReward,
    ESMFoldReward,
    Reward,
    StubReward,
    SurrogateReward,
    TsvLookupReward,
    build_reward,
)


# ─────────────────────────────────────────────────────────────────────────────
# pairwise_identity
# ─────────────────────────────────────────────────────────────────────────────


def test_pairwise_identity_identical():
    seqs = ["AAAA", "AAAA"]
    m = pairwise_identity(seqs)
    assert m.shape == (2, 2)
    assert pytest.approx(m[0, 1], abs=1e-9) == 1.0
    assert pytest.approx(m[1, 0], abs=1e-9) == 1.0
    assert m[0, 0] == 1.0 and m[1, 1] == 1.0


def test_pairwise_identity_disjoint_alphabets():
    # AAAA vs CCCC should align trivially with zero matches.
    m = pairwise_identity(["AAAA", "CCCC"])
    assert pytest.approx(m[0, 1], abs=1e-9) == 0.0


def test_pairwise_identity_length_mismatch_gap_counts_as_difference():
    # ACG (length 3) vs ACGT (length 4). Global alignment with
    # match=1, mismatch=0, gap=-1 produces:
    #   ACG-          alignment_length = 4
    #   ACGT          matches = 3 (A,C,G) → identity = 3/4 = 0.75.
    # User-spec semantics: gap columns count toward the denominator
    # and reduce identity to 0.75 (divergence 0.25). Encodes the
    # AA-B vs AABB example from the design doc with canonical AA
    # letters (B is not in the canonical 20 — A,C,G,T are).
    m = pairwise_identity(["ACG", "ACGT"])
    assert pytest.approx(m[0, 1], abs=1e-6) == 0.75


def test_pairwise_identity_symmetric_and_unit_diag():
    seqs = ["AAAA", "AACC", "ACAC"]
    m = pairwise_identity(seqs)
    assert m.shape == (3, 3)
    # symmetry
    assert np.allclose(m, m.T)
    # unit diagonal
    assert np.allclose(np.diag(m), 1.0)
    # all entries in [0, 1]
    assert (m >= 0).all() and (m <= 1).all()


def test_pairwise_identity_strips_non_canonical_chars():
    # _clean_aa drops anything outside the canonical 20 AAs. Spaces,
    # digits, and punctuation get stripped, leaving "AAAA" — but the
    # canonical letters that happen to spell English (e.g. <START>'s
    # S, T, A, R, T) are kept, since the cleaner has no notion of
    # special tokens, only of canonical AA membership. This test pins
    # the behavior with chars guaranteed to be stripped.
    m = pairwise_identity(["AAAA", "A 1?A:A;A"])
    assert pytest.approx(m[0, 1], abs=1e-9) == 1.0


# ─────────────────────────────────────────────────────────────────────────────
# diversity_stats
# ─────────────────────────────────────────────────────────────────────────────


def test_diversity_stats_group_size_1_zero_diversity():
    stats = diversity_stats(["AAAA", "CCCC", "GGGG"], group_size=1)
    assert stats["diversity_mean"] == 0.0
    assert all(d == 0.0 for d in stats["per_replica_diversity"])
    assert stats["unique_count"] == 3


def test_diversity_stats_all_identical_group_zero():
    stats = diversity_stats(["AAAA", "AAAA", "AAAA", "AAAA"], group_size=4)
    assert stats["diversity_mean"] == 0.0
    assert pytest.approx(stats["diversity_min_pair"], abs=1e-9) == 1.0
    assert all(d == 0.0 for d in stats["per_replica_diversity"])


def test_diversity_stats_distinct_group_positive():
    stats = diversity_stats(["AAAA", "CCCC", "GGGG", "TTTT"], group_size=4)
    assert stats["diversity_mean"] > 0.9
    assert all(d > 0.9 for d in stats["per_replica_diversity"])
    assert stats["unique_count"] == 4


def test_diversity_stats_two_groups_per_group_mean_not_pooled():
    """Diversity mean is averaged over groups, not over all pairs.

    Group 1 (tight): all identical → group divergence ≈ 0.
    Group 2 (diverse): all distinct → group divergence ≈ 1.
    Per-group mean = (0 + 1) / 2 = 0.5. Pooled (all 6 pairs) would
    give a different answer. This test pins the spec.
    """
    seqs = [
        "AAAA", "AAAA", "AAAA",      # group 1: tight
        "AAAA", "CCCC", "GGGG",      # group 2: diverse
    ]
    stats = diversity_stats(seqs, group_size=3)
    # Group 1 contributes ~0; group 2 contributes ~1.
    assert stats["diversity_mean"] == pytest.approx(0.5, abs=0.05)
    # Worst pair (max within-group identity) is 1.0 (group 1 has identical pairs).
    assert pytest.approx(stats["diversity_min_pair"], abs=1e-9) == 1.0


def test_diversity_stats_rejects_uneven_split():
    with pytest.raises(ValueError):
        diversity_stats(["A", "A", "A"], group_size=2)


def test_diversity_stats_returns_per_replica_length_BG():
    seqs = ["AAAA", "AACC", "ACAC", "GGGG", "GGAA", "AAAA"]
    stats = diversity_stats(seqs, group_size=3)
    assert len(stats["per_replica_diversity"]) == len(seqs)


# ─────────────────────────────────────────────────────────────────────────────
# DiversityReward
# ─────────────────────────────────────────────────────────────────────────────


def test_diversity_reward_all_identical_zero():
    r = DiversityReward(group_size=4, scale=100.0)
    out = r(["AAAA"] * 4)
    assert all(x == 0.0 for x in out)


def test_diversity_reward_all_distinct_positive():
    r = DiversityReward(group_size=4, scale=100.0)
    out = r(["AAAA", "CCCC", "GGGG", "TTTT"])
    assert all(x > 90.0 for x in out)


def test_diversity_reward_targeted_peaks_at_target():
    target = 0.5
    r = DiversityReward(group_size=4, scale=100.0, mode="targeted", target=target)
    # All identical → diversity 0 → reward = 100*(1-2*0.5) = 0.
    r_zero = r(["AAAA"] * 4)
    # All distinct → diversity ≈ 1 → reward = 100*(1-2*0.5) = 0.
    r_one = r(["AAAA", "CCCC", "GGGG", "TTTT"])
    # Mixed (~half identical) → diversity around 0.5 → near peak.
    # Hand-tune: AAAA, AAAA, CCCC, GGGG. Replica 0&1 share ↦ 0; 2,3
    # diverge ↦ ~1. Per-replica avg ≈ {0.66, 0.66, 1.0, 1.0} mean ≈ 0.83.
    # Use a tighter mid case to land near 0.5.
    r_mid = r(["AAAA", "AAAA", "ACGT", "ACGT"])
    assert max(r_mid) > max(r_zero)
    assert max(r_mid) > max(r_one)


def test_diversity_reward_last_components_length_BG():
    G = 4
    r = DiversityReward(group_size=G, scale=100.0)
    seqs = ["AAAA", "CCCC", "GGGG", "TTTT"] * 2  # B=2 groups
    _ = r(seqs)
    comps = r.last_components()
    assert "diversity" in comps
    assert len(comps["diversity"]) == len(seqs)


def test_diversity_reward_rejects_bad_args():
    with pytest.raises(ValueError):
        DiversityReward(group_size=0)
    with pytest.raises(ValueError):
        DiversityReward(group_size=4, mode="bogus")
    with pytest.raises(ValueError):
        DiversityReward(group_size=4, target=1.5)


def test_diversity_reward_via_composite_reward_last_components():
    """Composite-reward pipeline surfaces both base and diversity components."""
    G = 4
    base = StubReward(target_length=4)
    div = DiversityReward(group_size=G, scale=100.0)
    composite = CompositeReward(
        {"stub": (base, 1.0), "diversity": (div, 0.5)},
        reduction="weighted_sum",
    )
    seqs = ["AAAA", "CCCC", "GGGG", "TTTT"]
    out = composite(seqs)
    assert len(out) == len(seqs)
    last = composite.last_components()
    assert set(last.keys()) == {"stub", "diversity"}
    assert len(last["stub"]) == len(seqs)
    assert len(last["diversity"]) == len(seqs)


# ─────────────────────────────────────────────────────────────────────────────
# Back-compat after the rewards.py → rewards/ subpackage refactor
# ─────────────────────────────────────────────────────────────────────────────


def test_rewards_back_compat_imports_resolve():
    # Symbols every existing importer relies on must remain importable
    # from `biom3.rl.rewards` after the subpackage refactor.
    assert StubReward is not None
    assert ESMFoldReward is not None
    assert AAFractionReward is not None
    assert SurrogateReward is not None
    assert TsvLookupReward is not None
    assert CompositeReward is not None
    assert DiversityReward is not None
    assert build_reward is not None
    # `Reward` is a Protocol class; it should at least look like one.
    assert inspect.isclass(Reward)


def test_build_reward_diversity_dispatch():
    import torch
    r = build_reward("diversity", device=torch.device("cpu"), group_size=4)
    assert isinstance(r, DiversityReward)
    assert r.group_size == 4


def test_build_reward_unknown_lists_diversity_in_error():
    import torch
    with pytest.raises(ValueError) as ei:
        build_reward("nope", device=torch.device("cpu"))
    msg = str(ei.value)
    # Both old options and the new diversity name appear in the help.
    assert "diversity" in msg
    assert "esmfold_plddt" in msg

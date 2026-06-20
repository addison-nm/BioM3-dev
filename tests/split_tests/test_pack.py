"""Tests for greedy whole-cluster packing.

The central property under test: a cluster never spans two splits, so packing
cannot leak similar sequences across train/val/test.
"""

import pytest

from biom3.split.pack import pack_clusters


def _make_clusters(sizes):
    """Build clusters of the requested sizes with globally unique members."""
    clusters = []
    nxt = 0
    for size in sizes:
        clusters.append([(0, nxt + i) for i in range(size)])
        nxt += size
    return clusters


def test_no_cluster_spans_two_splits():
    clusters = _make_clusters([10, 7, 5, 5, 3, 2, 2, 1, 1, 1])
    result = pack_clusters(clusters, {"train": 0.8, "val": 0.1, "test": 0.1}, seed=0)

    member_to_split = {}
    for split, members in result.members.items():
        for m in members:
            assert m not in member_to_split, "member assigned to two splits"
            member_to_split[m] = split

    for cluster in clusters:
        splits = {member_to_split[m] for m in cluster}
        assert len(splits) == 1, "a cluster was split across multiple splits"


def test_all_members_assigned_once():
    clusters = _make_clusters([4, 3, 3, 2, 2, 1])
    result = pack_clusters(clusters, {"train": 0.7, "val": 0.3}, seed=1)
    total = sum(len(c) for c in clusters)
    assigned = sum(len(v) for v in result.members.values())
    assert assigned == total
    assert sum(result.counts.values()) == total


def test_ratios_approximately_honored_with_many_small_clusters():
    clusters = _make_clusters([1] * 1000)
    result = pack_clusters(clusters, {"train": 0.8, "val": 0.1, "test": 0.1}, seed=0)
    assert result.achieved["train"] == pytest.approx(0.8, abs=0.02)
    assert result.achieved["val"] == pytest.approx(0.1, abs=0.02)
    assert result.achieved["test"] == pytest.approx(0.1, abs=0.02)


def test_determinism_same_seed():
    clusters = _make_clusters([5, 5, 5, 3, 3, 2, 2, 1, 1])
    r1 = pack_clusters(clusters, {"train": 0.8, "val": 0.2}, seed=7)
    r2 = pack_clusters(clusters, {"train": 0.8, "val": 0.2}, seed=7)
    assert r1.cluster_splits == r2.cluster_splits


def test_zero_target_split_gets_nothing():
    clusters = _make_clusters([3, 3, 2, 2, 1, 1])
    result = pack_clusters(clusters, {"train": 0.8, "val": 0.2, "test": 0.0}, seed=0)
    assert "test" not in result.members
    assert result.counts.get("test", 0) == 0


def test_invalid_ratios_raise():
    clusters = _make_clusters([1, 1])
    with pytest.raises(ValueError):
        pack_clusters(clusters, {"train": 0.5, "val": 0.2}, seed=0)
    with pytest.raises(ValueError):
        pack_clusters(clusters, {"train": 1.0, "bogus": 0.0}, seed=0)


def test_giant_cluster_overshoots_and_is_reported():
    # One cluster bigger than the val target forces an overshoot; packing must
    # still keep it whole rather than split it.
    clusters = _make_clusters([50, 1, 1, 1, 1, 1])
    result = pack_clusters(clusters, {"train": 0.8, "val": 0.2}, seed=0)
    # The 50-member cluster lands whole in one split.
    big = clusters[0][0]
    big_split = next(s for s, ms in result.members.items() if big in ms)
    assert all(m in result.members[big_split] for m in clusters[0])

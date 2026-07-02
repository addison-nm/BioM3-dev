"""Tests for source-stratified whole-cluster packing.

Guarantees under test: whole clusters never span two splits (leakage-safe), and
each source is distributed across splits close to the target ratios.
"""

import pytest

from biom3.split.pack_stratified import pack_clusters_stratified


def _singletons(source, start, n):
    clusters = [[(0, start + i)] for i in range(n)]
    src = {(0, start + i): source for i in range(n)}
    return clusters, src


def _pool(*groups):
    clusters, src = [], {}
    for g_clusters, g_src in groups:
        clusters.extend(g_clusters)
        src.update(g_src)
    return clusters, src


def test_no_cluster_spans_two_splits():
    clusters, src = _pool(_singletons("pfam", 0, 30), _singletons("sp", 100, 30))
    clusters.append([(0, 200), (0, 201), (0, 202)])
    src.update({(0, 200): "pfam", (0, 201): "sp", (0, 202): "pfam"})
    result = pack_clusters_stratified(
        clusters, src, {"train": 0.7, "val": 0.2, "test": 0.1}, seed=0
    )
    split_of = {}
    for split, members in result.members.items():
        for m in members:
            assert m not in split_of
            split_of[m] = split
    for cluster in clusters:
        assert len({split_of[m] for m in cluster}) == 1


def test_all_members_assigned_once():
    clusters, src = _pool(_singletons("pfam", 0, 20), _singletons("sp", 100, 10))
    result = pack_clusters_stratified(clusters, src, {"train": 0.7, "val": 0.3}, seed=1)
    total = sum(len(c) for c in clusters)
    assert sum(len(v) for v in result.members.values()) == total
    assert sum(result.counts.values()) == total


def test_each_source_hits_target_ratios():
    clusters, src = _pool(_singletons("pfam", 0, 500), _singletons("sp", 1000, 100))
    result = pack_clusters_stratified(
        clusters, src, {"train": 0.7, "val": 0.2, "test": 0.1}, seed=0
    )
    for source in ("pfam", "sp"):
        ach = result.per_source_achieved[source]
        assert ach["train"] == pytest.approx(0.7, abs=0.05)
        assert ach["val"] == pytest.approx(0.2, abs=0.05)
        assert ach["test"] == pytest.approx(0.1, abs=0.05)


def test_small_source_not_lumped_into_one_split():
    # Without stratification a 100-row source could land entirely in train.
    clusters, src = _pool(_singletons("pfam", 0, 2000), _singletons("sp", 5000, 100))
    result = pack_clusters_stratified(
        clusters, src, {"train": 0.8, "val": 0.2}, seed=0
    )
    sp = result.per_source_counts["sp"]
    assert sp["train"] > 0 and sp["val"] > 0
    assert result.per_source_achieved["sp"]["val"] == pytest.approx(0.2, abs=0.05)


def test_cross_source_cluster_pinned_to_one_split():
    clusters = [[(0, 0), (0, 1), (0, 2)]]  # pfam+pfam+swissprot mixed cluster
    src = {(0, 0): "pfam", (0, 1): "pfam", (0, 2): "sp"}
    fill, fsrc = _singletons("pfam", 10, 100)
    clusters += fill
    src.update(fsrc)
    result = pack_clusters_stratified(clusters, src, {"train": 0.7, "val": 0.3}, seed=0)
    split_of = {m: sp for sp, members in result.members.items() for m in members}
    assert split_of[(0, 0)] == split_of[(0, 1)] == split_of[(0, 2)]


def test_determinism_same_seed():
    clusters, src = _pool(_singletons("pfam", 0, 40), _singletons("sp", 100, 20))
    r1 = pack_clusters_stratified(clusters, src, {"train": 0.8, "val": 0.2}, seed=7)
    r2 = pack_clusters_stratified(clusters, src, {"train": 0.8, "val": 0.2}, seed=7)
    assert r1.cluster_splits == r2.cluster_splits

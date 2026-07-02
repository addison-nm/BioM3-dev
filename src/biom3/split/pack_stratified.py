"""Source-stratified whole-cluster packing.

Like :func:`biom3.split.pack.pack_clusters`, but targets the split ratios
*within each data source* (e.g. pfam, swissprot, supplemental) simultaneously,
so no single source lands entirely in one split. Whole clusters are still never
split across train/val/test, so the sequence-similarity leakage guarantee is
unchanged. A cluster that spans multiple sources (produced by global clustering
of cross-source homologs) pins all its members — from every source it contains —
into the same split, which is exactly what prevents cross-source leakage.

Packing is greedy-by-fractional-deficit: clusters are visited largest first and
each is placed in the split that currently has the largest per-source fractional
deficit, weighted by how many members of each source the cluster carries.
Normalizing by each source's own target keeps small sources from being drowned
out by large ones. Whole-cluster granularity means a cluster larger than a
source's split target will overshoot; per-source achieved ratios are returned so
the caller can surface deviations.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from biom3.split.pack import SPLITS, PackResult, _normalize_ratios


@dataclass
class StratifiedPackResult(PackResult):
    """PackResult plus per-source breakdown.

    ``per_source_counts`` maps source -> {split: member count}; and
    ``per_source_achieved`` maps source -> {split: realized fraction}.
    """
    per_source_counts: dict = field(default_factory=dict)
    per_source_achieved: dict = field(default_factory=dict)


def pack_clusters_stratified(clusters, member_source, ratios, seed=0):
    """Assign whole clusters to splits, stratified by source.

    Args:
        clusters: iterable of clusters, each a sequence of hashable member keys.
        member_source: mapping member key -> source label.
        ratios: mapping split name -> target fraction (applied within each
            source). Splits with a non-positive fraction are excluded.
        seed: tie-break seed; only affects ordering among equal-size clusters.

    Returns:
        StratifiedPackResult.
    """
    ratios = _normalize_ratios(ratios)
    clusters = [list(c) for c in clusters]

    cluster_src_counts = []
    source_totals = {}
    for cluster in clusters:
        src_counts = {}
        for key in cluster:
            src = member_source[key]
            src_counts[src] = src_counts.get(src, 0) + 1
            source_totals[src] = source_totals.get(src, 0) + 1
        cluster_src_counts.append(src_counts)

    targets = {
        src: {sp: ratios[sp] * total for sp in ratios}
        for src, total in source_totals.items()
    }
    counts = {src: {sp: 0 for sp in ratios} for src in source_totals}

    members = {sp: [] for sp in ratios}
    split_counts = {sp: 0 for sp in ratios}
    cluster_splits = [None] * len(clusters)

    # Shuffle then stable sort by size desc, so equal-size clusters break ties
    # deterministically by seed (mirrors pack_clusters).
    order = list(range(len(clusters)))
    random.Random(seed).shuffle(order)
    order.sort(key=lambda i: len(clusters[i]), reverse=True)

    def _score(src_counts, split):
        total = 0.0
        for src, n in src_counts.items():
            target = targets[src][split]
            if target > 0:
                total += n * (target - counts[src][split]) / target
        return total

    for i in order:
        src_counts = cluster_src_counts[i]
        best = max(
            ratios,
            key=lambda sp: (_score(src_counts, sp), -SPLITS.index(sp)),
        )
        members[best].extend(clusters[i])
        for src, n in src_counts.items():
            counts[src][best] += n
        split_counts[best] += len(clusters[i])
        cluster_splits[i] = best

    total_members = sum(len(c) for c in clusters)
    achieved = {
        sp: (split_counts[sp] / total_members if total_members else 0.0)
        for sp in ratios
    }
    per_source_achieved = {
        src: {
            sp: (counts[src][sp] / total if total else 0.0) for sp in ratios
        }
        for src, total in source_totals.items()
    }
    return StratifiedPackResult(
        members=members,
        cluster_splits=cluster_splits,
        counts=split_counts,
        achieved=achieved,
        per_source_counts=counts,
        per_source_achieved=per_source_achieved,
    )

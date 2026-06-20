"""Greedy whole-cluster packing into train/val/test splits.

The single correctness guarantee provided here: every cluster is assigned to
exactly one split, so no cluster ever spans two splits.  This is what prevents
sequence-similarity leakage between train, validation, and test.

Packing is greedy-by-deficit: clusters are visited largest first, and each is
placed in whichever split is currently furthest below its target count.  This
keeps the achieved ratios as close to target as whole-cluster granularity
allows; a single cluster larger than a split's target will necessarily
overshoot, which the caller is expected to surface.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

SPLITS = ("train", "val", "test")


@dataclass
class PackResult:
    """Outcome of packing clusters into splits.

    ``members`` maps each split name to the list of member keys assigned to it.
    ``cluster_splits`` records, per input cluster (by index), which split it
    landed in.  ``counts`` and ``achieved`` give the member counts and realized
    fractions per split.
    """
    members: dict = field(default_factory=dict)
    cluster_splits: list = field(default_factory=list)
    counts: dict = field(default_factory=dict)
    achieved: dict = field(default_factory=dict)


def _normalize_ratios(ratios):
    unknown = set(ratios) - set(SPLITS)
    if unknown:
        raise ValueError(f"unknown split name(s): {sorted(unknown)}; allowed: {SPLITS}")
    active = {k: float(v) for k, v in ratios.items() if float(v) > 0.0}
    if not active:
        raise ValueError("ratios must contain at least one split with a positive fraction")
    total = sum(active.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"ratios must sum to 1.0 (got {total})")
    # Preserve canonical split ordering for deterministic tie-breaks.
    return {k: active[k] for k in SPLITS if k in active}


def pack_clusters(clusters, ratios, seed=0):
    """Assign whole clusters to splits to approximate target ratios.

    Args:
        clusters: iterable of clusters, each a sequence of hashable member keys.
        ratios: mapping of split name -> target fraction. Splits with a
            non-positive fraction are excluded entirely (no members assigned).
        seed: tie-break seed; only affects ordering among equal-size clusters.

    Returns:
        PackResult.
    """
    ratios = _normalize_ratios(ratios)
    clusters = [list(c) for c in clusters]

    total_members = sum(len(c) for c in clusters)
    targets = {s: ratios[s] * total_members for s in ratios}

    members = {s: [] for s in ratios}
    counts = {s: 0 for s in ratios}
    cluster_splits = [None] * len(clusters)

    # Shuffle first so equal-size clusters are not biased by input order, then
    # stable-sort by size descending. The stable sort preserves the shuffled
    # order within each size group, making tie-breaks seed-deterministic.
    order = list(range(len(clusters)))
    random.Random(seed).shuffle(order)
    order.sort(key=lambda i: len(clusters[i]), reverse=True)

    for i in order:
        cluster = clusters[i]
        # Largest remaining deficit wins; ties broken by canonical split order.
        best = max(ratios, key=lambda s: (targets[s] - counts[s], -SPLITS.index(s)))
        members[best].extend(cluster)
        counts[best] += len(cluster)
        cluster_splits[i] = best

    achieved = {
        s: (counts[s] / total_members if total_members else 0.0) for s in ratios
    }
    return PackResult(
        members=members,
        cluster_splits=cluster_splits,
        counts=counts,
        achieved=achieved,
    )

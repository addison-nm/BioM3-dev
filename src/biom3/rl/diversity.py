"""Within-batch sequence diversity utilities for GRPO/GDPO.

The first GDPO production run mode-collapsed within ~100 steps:
within-batch pairwise sequence identity rose from 65.6% → 91.9%, reward
std fell from 1.75 → 0.11, and the group-relative advantage signal died.
This module surfaces that failure mode as a per-step metric and provides
the helper used by ``DiversityReward`` to penalize collapse explicitly.

Identity is computed via ``Bio.Align.PairwiseAligner`` in global
(Needleman-Wunsch) mode with simple match/mismatch/gap scoring
(match=1, mismatch=0, gap=-1). The user-facing semantics is

    identity = matches / alignment_length

so gap columns count toward the denominator but never the numerator —
``AA-B`` aligned against ``AABB`` yields 3 matches over 4 columns
(0.75 identity, 0.25 divergence). This deliberately differs from
substitution-matrix scoring (e.g. BLOSUM62 in
``Stage3.eval_metrics``); BLOSUM gives partial credit for similar
substitutions, which would mask collapse.

Pair-identity lookups are LRU-cached on the unordered pair of cleaned
sequences. Mode collapse repeatedly emits the same string, so the cache
collapses to near-zero recompute exactly when the diagnostic is most
useful.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, List

import numpy as np

from biom3.backend.device import setup_logger

logger = setup_logger(__name__)


# Inlined VALID_AA + cleaner to avoid a circular import with
# biom3.rl.rewards.base (which imports DiversityReward via the rewards
# package __init__, which imports back from this module). The 21-glyph
# canonical alphabet here matches biom3.rl.rewards.base.VALID_AA — the
# base file remains the single source of truth for the *reward* code
# path; this duplicate is private to the diversity helper.
_DIV_VALID_AA = frozenset("ACDEFGHIKLMNPQRSTVWY")


def _clean_aa(seq: str) -> str:
    return "".join(c for c in seq if c in _DIV_VALID_AA)


# Lazy import: biopython is in the [grpo] extra. We construct a single
# module-level aligner because PairwiseAligner is stateful (scoring
# parameters) but its align() calls are independent — safe to reuse.
_aligner = None


def _get_aligner():
    global _aligner
    if _aligner is not None:
        return _aligner
    try:
        from Bio.Align import PairwiseAligner
    except ImportError as e:
        raise ImportError(
            "biom3.rl.diversity requires biopython. Install with: "
            "pip install -e '.[grpo]'"
        ) from e
    a = PairwiseAligner()
    a.mode = "global"
    a.match_score = 1.0
    a.mismatch_score = 0.0
    a.open_gap_score = -1.0
    a.extend_gap_score = -1.0
    _aligner = a
    return _aligner


@lru_cache(maxsize=4096)
def _pair_identity_cached(a: str, b: str) -> float:
    """Identity for a single ordered pair of cleaned sequences.

    Cached on the cleaned strings so callers should pass the canonical
    form (see ``_clean_pair`` below). Order is normalized by the caller
    via ``frozenset``-style sorting before this is invoked, keeping the
    cache symmetric without a wrapper layer.
    """
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    aligner = _get_aligner()
    alignments = aligner.align(a, b)
    # Take the top-scoring alignment. Under our match=1/mismatch=0/gap=-1
    # scoring (no substitution matrix), identity is invariant across
    # tied alignments with the same number of matches and gapped length,
    # so any deterministic pick is fine.
    aln = alignments[0]
    # ``aln[0]`` and ``aln[1]`` are the gapped strings for the first and
    # second sequence respectively. Their length equals the alignment
    # length (matches + mismatches + gap columns); ``aln.length`` in
    # this BioPython version returns the count of *aligned residues*,
    # not the gapped length, so we read the gapped string directly.
    gapped_a = str(aln[0])
    gapped_b = str(aln[1])
    aln_len = len(gapped_a)
    if aln_len == 0:
        return 0.0
    matches = sum(
        1 for ca, cb in zip(gapped_a, gapped_b)
        if ca != "-" and cb != "-" and ca == cb
    )
    return matches / aln_len


def _pair_identity(a: str, b: str) -> float:
    """Symmetric pair identity (delegates to the cached ordered helper)."""
    if a > b:
        a, b = b, a
    return _pair_identity_cached(a, b)


def pairwise_identity(seqs: List[str]) -> np.ndarray:
    """Symmetric `(N, N)` matrix of pairwise sequence identity.

    Identity is in `[0, 1]`; the diagonal is 1.0. Inputs are cleaned to
    canonical AA via ``_clean_aa`` before alignment. Empty cleaned
    strings get identity 0 against any non-empty peer (and 1 against
    other empty strings).
    """
    n = len(seqs)
    cleaned = [_clean_aa(s) for s in seqs]
    out = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        out[i, i] = 1.0
    for i in range(n):
        for j in range(i + 1, n):
            ident = _pair_identity(cleaned[i], cleaned[j])
            out[i, j] = ident
            out[j, i] = ident
    return out


def diversity_stats(seqs: List[str], group_size: int) -> Dict[str, Any]:
    """Per-batch diversity diagnostics for GRPO/GDPO logging.

    ``seqs`` is the flat ``B*G`` rollout (one prompt's replicas
    contiguous, then the next prompt's, etc.). The function splits it
    into ``B = len(seqs) / group_size`` groups and reports:

    - ``diversity_mean``        : mean over groups of mean off-diagonal
                                  divergence (1 - identity).
    - ``diversity_min_pair``    : worst (highest) within-group pairwise
                                  identity across groups — sensitive to
                                  early collapse.
    - ``unique_count``          : sum over groups of distinct cleaned
                                  sequences.
    - ``per_replica_diversity`` : flat list, length ``len(seqs)``;
                                  replica i gets the mean over peers in
                                  its own group of (1 - identity).

    Singleton groups (``group_size == 1``) have no peers; per-replica
    diversity is 0.0 and group-mean diversity is 0.0 by definition.
    """
    if group_size <= 0:
        raise ValueError(f"group_size must be positive, got {group_size}")
    n = len(seqs)
    if n == 0:
        return {
            "diversity_mean": 0.0,
            "diversity_min_pair": 0.0,
            "unique_count": 0,
            "per_replica_diversity": [],
        }
    if n % group_size != 0:
        raise ValueError(
            f"len(seqs)={n} not divisible by group_size={group_size}"
        )
    n_groups = n // group_size
    per_replica: List[float] = [0.0] * n
    group_div_means: List[float] = []
    group_max_idents: List[float] = []
    unique_total = 0

    for g in range(n_groups):
        start = g * group_size
        end = start + group_size
        group = seqs[start:end]
        unique_total += len(set(_clean_aa(s) for s in group))
        if group_size == 1:
            group_div_means.append(0.0)
            group_max_idents.append(1.0)
            continue
        ident = pairwise_identity(group)            # (G, G)
        # Off-diagonal mean per replica
        off_mask = ~np.eye(group_size, dtype=bool)
        for i in range(group_size):
            row_off = ident[i][off_mask[i]]
            per_replica[start + i] = float((1.0 - row_off).mean())
        # Group-level scalars
        off = ident[off_mask]
        group_div_means.append(float((1.0 - off).mean()))
        group_max_idents.append(float(off.max()))

    diversity_mean = float(np.mean(group_div_means)) if group_div_means else 0.0
    # Worst-pair: report the largest within-group identity (most-similar pair).
    diversity_min_pair = (
        float(np.max(group_max_idents)) if group_max_idents else 0.0
    )
    return {
        "diversity_mean": diversity_mean,
        "diversity_min_pair": diversity_min_pair,
        "unique_count": int(unique_total),
        "per_replica_diversity": per_replica,
    }

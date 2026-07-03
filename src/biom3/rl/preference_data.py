"""Preference-data ingestion for BioM3 Stage 3 DPO.

Consumes the tidy scored-set CSVs produced by ``data/rl/convert_rl_data_to_csv.py``
(columns ``dataset, source, prompt, prompt_text, functional, sequence, score``)
and turns them into the two batch shapes the DPO trainer consumes:

  * **paired**  — an ordered ``(chosen, rejected)`` pair per example, for the
    Bradley-Terry (Paired) objective.
  * **weighted** — a set of ``K`` scored candidates per example, for the
    scalar-label (Weighted) objective (ProteinDPO eq. 15-17), which needs no
    binarization.

Both share one abstraction — a :class:`PreferenceGroup` (a set of sequences
that condition on the *same* ``z_c``). Grouping supports:

  * **Case B** (prompt available, e.g. BioM3 designs): one group per
    ``prompt_text``; the caption is that text.
  * **Case C** (no prompt, e.g. VAE designs): a single group whose caption is
    a configurable ``default_caption``.

Sequences are tokenized with :func:`biom3.Stage3.preprocess.encode_protein_sequence`
so the ids exactly match the Stage 3 training convention (``<START>`` + residues
+ ``<END>`` padded with ``-`` to ``image_size**2``).
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from biom3.Stage3.preprocess import encode_protein_sequence
from biom3.rl.grpo import PAD_ID, TOKENS

# Amino-acid / special letters the Stage 3 tokenizer understands (everything in
# TOKENS that is a single residue-like glyph). Sequences containing anything
# else (e.g. an ambiguity code 'J') are dropped rather than silently mis-encoded.
_VOCAB_AA = set(t for t in TOKENS if len(t) == 1 and t.isalpha())

DEFAULT_CAPTION_KEY = "<default>"


@dataclass
class PreferenceGroup:
    """A set of sequences sharing one conditioning caption."""
    caption_key: str          # dedup key for z_c caching (prompt text or DEFAULT_CAPTION_KEY)
    caption: str              # the text actually encoded into z_c
    seqs: List[str]
    scores: np.ndarray        # (n,)
    functional: Optional[np.ndarray]  # (n,) 0/1 or None


def _valid_seq(s: str, min_len: int, max_len: Optional[int]) -> bool:
    if not isinstance(s, str) or not s:
        return False
    if len(s) < min_len or (max_len is not None and len(s) > max_len):
        return False
    return set(s) <= _VOCAB_AA


def load_groups(
    csv_path: str,
    *,
    dataset: Optional[str] = None,
    group_by: str = "prompt_text",
    default_caption: str = "SH3 domain protein.",
    min_len: int = 1,
    max_len: Optional[int] = None,
    min_group_size: int = 2,
) -> List[PreferenceGroup]:
    """Load preference groups from a scored-set CSV.

    If ``group_by`` names a column with non-null values (e.g. ``prompt_text``),
    one group is built per distinct value (Case B). Otherwise all rows form a
    single group conditioned on ``default_caption`` (Case C).
    """
    df = pd.read_csv(csv_path)
    if dataset is not None:
        df = df[df["dataset"] == dataset]
    df = df[df["sequence"].map(lambda s: _valid_seq(s, min_len, max_len))].copy()
    if df.empty:
        raise ValueError(f"No valid sequences in {csv_path} (dataset={dataset}).")

    has_group = group_by in df.columns and df[group_by].notna().any()
    groups: List[PreferenceGroup] = []
    if has_group:
        sub = df[df[group_by].notna()]
        for caption, gdf in sub.groupby(group_by, sort=True):
            if len(gdf) < min_group_size:
                continue
            func = (gdf["functional"].to_numpy(dtype=float)
                    if "functional" in gdf and gdf["functional"].notna().all() else None)
            groups.append(PreferenceGroup(
                caption_key=str(caption),
                caption=str(caption),
                seqs=gdf["sequence"].tolist(),
                scores=gdf["score"].to_numpy(dtype=float),
                functional=func,
            ))
    else:
        groups.append(PreferenceGroup(
            caption_key=DEFAULT_CAPTION_KEY,
            caption=default_caption,
            seqs=df["sequence"].tolist(),
            scores=df["score"].to_numpy(dtype=float),
            functional=None,
        ))
    if not groups:
        raise ValueError(
            f"No groups with >= {min_group_size} sequences (group_by={group_by})."
        )
    return groups


class PreferenceSampler:
    """Draws paired / weighted training batches from a list of groups.

    Tokenization is cached per unique sequence. ``captions`` exposes the
    (caption_key, caption) pairs so the trainer can pre-encode each z_c once.
    """

    def __init__(self, groups: List[PreferenceGroup], image_size: int, seed: int = 0):
        if not groups:
            raise ValueError("PreferenceSampler needs at least one group.")
        self.groups = groups
        self.image_size = int(image_size)
        self.rng = np.random.default_rng(seed)
        self._tok_cache: Dict[str, torch.Tensor] = {}
        # Groups usable for label-based pairing (both classes present).
        self._label_groups = [
            i for i, g in enumerate(groups)
            if g.functional is not None
            and (g.functional == 1).any() and (g.functional == 0).any()
        ]

    @property
    def captions(self) -> List[Tuple[str, str]]:
        seen, out = set(), []
        for g in self.groups:
            if g.caption_key not in seen:
                seen.add(g.caption_key)
                out.append((g.caption_key, g.caption))
        return out

    def encode(self, seq: str) -> torch.Tensor:
        t = self._tok_cache.get(seq)
        if t is None:
            t = torch.tensor(encode_protein_sequence(seq, self.image_size), dtype=torch.long)
            self._tok_cache[seq] = t
        return t

    # ---- paired ---------------------------------------------------------
    def _pick_pair_margin(self, g: PreferenceGroup, gap_level: float, min_margin: float):
        """Rank-sort by score; pick (chosen, rejected) with a rank gap set by
        ``gap_level`` in [0, 1] (ProteinDPO's gap heuristic), enforcing a
        minimum score margin. Returns (chosen_idx, rejected_idx) or None."""
        n = len(g.scores)
        order = np.argsort(g.scores)  # ascending
        gap = max(1, int(round(gap_level * (n - 1))))
        for _ in range(8):
            lo = int(self.rng.integers(0, n - gap))
            hi = lo + gap
            ci, ri = order[hi], order[lo]  # higher score chosen
            if g.scores[ci] - g.scores[ri] >= min_margin:
                return int(ci), int(ri)
        return None

    def _pick_pair_label(self, g: PreferenceGroup):
        pos = np.flatnonzero(g.functional == 1)
        neg = np.flatnonzero(g.functional == 0)
        if pos.size == 0 or neg.size == 0:
            return None
        return int(self.rng.choice(pos)), int(self.rng.choice(neg))

    def sample_paired_batch(
        self, batch_size: int, *, pairing: str = "margin",
        gap_level: float = 0.5, min_margin: float = 0.0,
    ) -> Dict:
        caption_keys, w_ids, l_ids = [], [], []
        pool = self._label_groups if pairing == "label" else list(range(len(self.groups)))
        if not pool:
            raise ValueError(
                "pairing='label' but no group has both functional & nonfunctional members."
            )
        while len(caption_keys) < batch_size:
            g = self.groups[int(self.rng.choice(pool))]
            pick = (self._pick_pair_label(g) if pairing == "label"
                    else self._pick_pair_margin(g, gap_level, min_margin))
            if pick is None:
                continue
            ci, ri = pick
            caption_keys.append(g.caption_key)
            w_ids.append(self.encode(g.seqs[ci]))
            l_ids.append(self.encode(g.seqs[ri]))
        return {
            "caption_keys": caption_keys,
            "w_ids": torch.stack(w_ids),
            "l_ids": torch.stack(l_ids),
        }

    # ---- weighted -------------------------------------------------------
    def sample_weighted_batch(self, batch_size: int, K: int) -> Dict:
        caption_keys, ids, scores = [], [], []
        for _ in range(batch_size):
            g = self.groups[int(self.rng.integers(0, len(self.groups)))]
            n = len(g.seqs)
            replace = n < K
            sel = self.rng.choice(n, size=K, replace=replace)
            caption_keys.append(g.caption_key)
            ids.append(torch.stack([self.encode(g.seqs[j]) for j in sel]))
            scores.append(torch.tensor(g.scores[sel], dtype=torch.float32))
        return {
            "caption_keys": caption_keys,
            "ids": torch.stack(ids),          # (B, K, L)
            "scores": torch.stack(scores),    # (B, K)
        }


def valid_length(ids: torch.Tensor) -> torch.Tensor:
    """Per-sequence count of non-PAD tokens (content length incl START/END).

    Used to length-normalize the ELBO so the DPO temperature ``beta`` is
    comparable across sequences of different length.
    """
    return (ids != PAD_ID).sum(dim=-1).clamp(min=1).to(torch.float32)

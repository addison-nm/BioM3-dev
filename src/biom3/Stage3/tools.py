"""Standalone likelihood estimation for ProteoScribe (Stage 3).

ProteoScribe is an order-agnostic absorbing-state autoregressive diffusion
model: given a conditioning vector ``z_c`` and a partially-masked sequence it
returns position-wise amino-acid probabilities. This module answers the
inverse question — *how likely is the model to generate a particular
sequence?* — via the Semi-deterministic Monte-Carlo (SDMC) ELBO from GDPO
(Rojas et al., ICLR 2026; arXiv 2510.08554), the same estimator used inside
``biom3.rl.gdpo._elbo_sdmc`` during RL training.

The estimator here is a strict generalization of the training-time one: it
scores an arbitrary *query* set of positions conditioned on an arbitrary
*fixed-context* set, with any remaining positions held *unknown* (absorbing
and marginalized out). With no masks it reduces to the plain sequence
likelihood ``log P(sequence | z_c)``.

Position roles
--------------
Every position in the (length-``diffusion_steps``) tensor is one of:

- ``QUERY``   — a concrete residue whose likelihood is estimated. Diffused
  over (revealed/masked across the SDMC trajectory) and scored. ``<END>`` is
  QUERY by default (``score_end``).
- ``CONTEXT`` — a concrete residue that is *always revealed* and conditioned
  upon but never scored (the ``<START>`` anchor, the ``<PAD>`` tail, and any
  position the caller pins via ``context_mask``). Whether ``<START>``/``<END>``
  and the ``<PAD>`` tail are CONTEXT or QUERY is set by ``score_start`` /
  ``score_end`` / ``score_padding``.
- ``UNKNOWN`` — an absorbing/mask position (``#`` in an input string). Never
  revealed, never scored; marginalized out of the likelihood.

The model's time index equals the number of currently-revealed (non-absorbing)
positions, so for a corruption that reveals ``r`` of the ``Q`` query residues
the index fed to the model is ``n_context + r`` (see
``run_ProteoScribe_sample._pre_revealed_offset`` for the same convention).
"""

import math
from argparse import Namespace
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Union

import torch
import torch.nn.functional as F

from biom3.Stage3.io import prepare_model_ProteoScribe
from biom3.backend.device import get_device, setup_logger
from biom3.core.helpers import load_json_config
from biom3.rl.grpo import (
    END_ID,
    MASK_ID,
    PAD_ID,
    START_ID,
    TOK2ID,
    TOKENS,
)

logger = setup_logger(__name__)

# Placeholder glyph for an UNKNOWN (absorbing) position inside an AA string.
MASK_CHAR = "#"

# Position roles.
_QUERY = 0
_CONTEXT = 1
_UNKNOWN = 2


@dataclass
class LikelihoodConfig:
    """SDMC quadrature settings for a likelihood estimate.

    The defaults trade a little compute for a low-variance estimate; drop
    ``n_quadrature`` / ``n_repeats`` for a quick approximate score.
    """

    # SDMC levels over the integer masked-count m in {1..Q}. n_quadrature caps
    # the number of levels; >= Q uses all of them (exact any-order NLL).
    n_quadrature: int = 16
    quadrature_grid: str = "uniform"          # "uniform" spaced | "explicit"
    quadrature_points: Optional[List[float]] = None   # explicit masked fractions m/Q
    quadrature_weights: Optional[List[float]] = None  # for "explicit"; else uniform

    inner_mc: int = 1        # mask samples per level (per repeat)
    n_repeats: int = 4       # independent estimates → mean + std

    # Include the <PAD> tail (and any padding beyond <END>) in the query set.
    # Off by default: padding is typically likely given <END> and would inflate
    # the log-likelihood with trivially-predicted tokens.
    score_padding: bool = False

    # Wrap the raw AA string with <START> … <END> before scoring, then pad to
    # diffusion_steps. Turn off if the input already carries special tokens.
    add_special_tokens: bool = True

    # Whether the <START>/<END> anchors are scored (QUERY) or held as fixed
    # context (revealed, conditioned on, not scored). Governs these two tokens
    # authoritatively (over context_mask). <START> is deterministic, so scoring
    # it rarely helps. <END> encodes where the protein terminates and is scored
    # by default; set score_end=False to score, e.g., only an interior loop
    # given a fixed scaffold without also paying for the terminus. No effect on
    # inputs that lack these tokens (e.g. add_special_tokens=False).
    score_start: bool = False
    score_end: bool = True

    # Max corruptions per model forward pass. Lower it if a long sequence /
    # large grid overflows device memory.
    max_forward_batch: int = 64

    seed: Optional[int] = None


@dataclass
class LikelihoodResult:
    """Result of a conditional likelihood estimate.

    ``log_likelihood`` is the SDMC ELBO — a lower bound on
    ``log P(query | context, z_c)`` in nats. ``log_likelihood_std`` is the
    sample standard deviation across ``n_repeats`` independent estimates
    (0.0 when ``n_repeats == 1``).
    """

    log_likelihood: float
    log_likelihood_std: float
    perplexity: float
    bits_per_residue: float
    n_query: int
    n_context: int
    n_unknown: int
    sequence_length: int
    per_quadrature: List[dict] = field(default_factory=list)

    @property
    def probability(self) -> float:
        """``exp(log_likelihood)``. Underflows to 0.0 for realistic lengths —
        prefer ``log_likelihood`` / ``bits_per_residue`` for comparison."""
        try:
            return math.exp(self.log_likelihood)
        except OverflowError:
            return float("inf")


# ─────────────────────────────────────────────────────────────────────────────
# Sequence → (ids, roles)
# ─────────────────────────────────────────────────────────────────────────────


def _classify_sequence(
    sequence: Union[str, Sequence[int], torch.Tensor],
    seq_len: int,
    context_mask: Optional[Sequence[bool]],
    cfg: LikelihoodConfig,
    device: torch.device,
):
    """Turn a sequence into ``(ids, roles)`` of shape ``(seq_len,)``.

    ``ids`` holds the concrete target token at every QUERY/CONTEXT position
    (used as the gather target); UNKNOWN positions hold ``MASK_ID``. ``roles``
    holds one of ``_QUERY`` / ``_CONTEXT`` / ``_UNKNOWN`` per position.

    A string is tokenized residue-by-residue (``#`` → UNKNOWN) and, when
    ``add_special_tokens`` is set, wrapped as ``<START> … <END>`` and padded to
    ``seq_len`` with ``<PAD>``. A pre-built id tensor/list of length
    ``seq_len`` is classified from its token values instead.
    """
    if isinstance(sequence, str):
        residues = list(sequence.strip())
        n_res = len(residues)
        if context_mask is not None and len(context_mask) != n_res:
            raise ValueError(
                f"context_mask length {len(context_mask)} != number of residues {n_res}"
            )
        ids = torch.full((seq_len,), PAD_ID, dtype=torch.long)
        roles = torch.full((seq_len,), _CONTEXT, dtype=torch.long)  # PAD tail = CONTEXT

        pos = 0
        if cfg.add_special_tokens:
            ids[pos] = START_ID          # role set by the policy pass below
            pos += 1
        for j, ch in enumerate(residues):
            if pos >= seq_len:
                raise ValueError(
                    f"sequence (+special tokens) exceeds diffusion_steps={seq_len}"
                )
            if ch == MASK_CHAR:
                ids[pos], roles[pos] = MASK_ID, _UNKNOWN
            elif ch in TOK2ID:
                ids[pos] = TOK2ID[ch]
                is_ctx = bool(context_mask[j]) if context_mask is not None else False
                roles[pos] = _CONTEXT if is_ctx else _QUERY
            else:
                raise ValueError(f"unknown residue {ch!r} at position {j}")
            pos += 1
        if cfg.add_special_tokens:
            if pos >= seq_len:
                raise ValueError(
                    f"sequence (+special tokens) exceeds diffusion_steps={seq_len}"
                )
            ids[pos] = END_ID            # role set by the policy pass below
            pos += 1
    else:
        ids = torch.as_tensor(sequence, dtype=torch.long).reshape(-1).clone()
        if ids.numel() != seq_len:
            raise ValueError(
                f"id sequence length {ids.numel()} != diffusion_steps={seq_len}"
            )
        roles = torch.full((seq_len,), _QUERY, dtype=torch.long)
        roles[ids == MASK_ID] = _UNKNOWN
        if context_mask is not None:
            if len(context_mask) != seq_len:
                raise ValueError(
                    f"context_mask length {len(context_mask)} != seq_len {seq_len}"
                )
            cm = torch.as_tensor(list(context_mask), dtype=torch.bool)
            roles[cm & (roles == _QUERY)] = _CONTEXT

    # Special-token roles are set here (authoritative over context_mask), then
    # the PAD tail. START/END/PAD are uniquely-valued tokens, so keying on id
    # is unambiguous; residues never collide with them.
    roles[ids == START_ID] = _QUERY if cfg.score_start else _CONTEXT
    roles[ids == END_ID] = _QUERY if cfg.score_end else _CONTEXT
    roles[ids == PAD_ID] = _QUERY if cfg.score_padding else _CONTEXT
    return ids.to(device), roles.to(device)


# ─────────────────────────────────────────────────────────────────────────────
# SDMC quadrature
# ─────────────────────────────────────────────────────────────────────────────


def _build_masked_levels(cfg: LikelihoodConfig, Q: int, device: torch.device):
    """Build the SDMC levels over the integer masked-count ``m in {1..Q}``.

    Returns ``(masked_counts, weights)`` with ``weights`` summing to 1. At level
    ``m`` exactly ``m`` query positions are masked (``Q - m`` revealed) and the
    ``m`` masked positions are scored. The estimator sums
    ``coeff_i · Σ_{masked} log p`` with ``coeff_i = Q · w_i / m_i`` (assembled in
    ``_build_corruptions``); this is the any-order ARDM NLL (Hoogeboom et al.
    2021) and is *exact* for a context-consistent model when every level is used.

    Gridding over the integer count (rather than a continuous fraction ``t``
    mapped through ``round((1 - t)·Q)``) avoids two failure modes of the earlier
    scheme: degenerate levels that reveal everything (``m = 0``, zero
    contribution) and the ``1/t`` weighting bias where ``m/t ≠ Q`` after
    rounding.

    ``uniform``: evenly-spaced distinct levels in ``{1..Q}`` (all ``Q`` when
    ``n_quadrature >= Q``, i.e. exact). ``explicit``: ``quadrature_points`` are
    read as masked *fractions* ``m/Q in (0, 1]`` (with optional
    ``quadrature_weights``).
    """
    if Q < 1:
        raise ValueError(f"Q must be >= 1, got {Q}")
    if cfg.quadrature_grid == "uniform":
        K = min(max(1, int(cfg.n_quadrature)), Q)
        m = torch.linspace(1, Q, K, device=device).round().long().unique()
    elif cfg.quadrature_grid == "explicit":
        if not cfg.quadrature_points:
            raise ValueError("quadrature_grid='explicit' requires quadrature_points")
        f = torch.tensor(cfg.quadrature_points, dtype=torch.float32, device=device)
        if (f <= 0).any() or (f > 1).any():
            raise ValueError(f"explicit quadrature_points are masked fractions in (0, 1]; got {f.tolist()}")
        m = torch.clamp(torch.round(f * Q).long(), min=1, max=Q)
        if cfg.quadrature_weights:
            if len(cfg.quadrature_weights) != f.numel():
                raise ValueError("quadrature_weights length must match quadrature_points")
            w = torch.tensor(cfg.quadrature_weights, dtype=torch.float32, device=device)
            return m, w / w.sum()
    else:
        raise ValueError(
            f"quadrature_grid must be 'uniform' or 'explicit', got {cfg.quadrature_grid!r}"
        )
    weights = torch.full((m.numel(),), 1.0 / m.numel(), dtype=torch.float32, device=device)
    return m, weights


def _build_corruptions(
    ids: torch.Tensor,           # (L,) concrete target tokens; MASK at unknown
    roles: torch.Tensor,         # (L,) role per position
    masked_counts: torch.Tensor, # (K,) integer m per level
    weights: torch.Tensor,       # (K,) level weights summing to 1
    cfg: LikelihoodConfig,
    generator: Optional[torch.Generator],
    device: torch.device,
):
    """Sample masked corruptions, one bundle per (repeat, level, inner_mc).

    At level ``m`` exactly ``m`` query positions are masked (``Q - m`` revealed
    at random); UNKNOWN positions are always masked, CONTEXT positions stay at
    their concrete token, and only the ``m`` masked query positions are scored.

    Returns ``(corruptions, Q)`` where each corruption is a dict with:
        x_t        (L,) long  — model input (MASK at masked/unknown)
        idx        int        — model time index = n_context + (Q - m)
        coeff      float      — Q · w_i / (m · inner_mc)
        score_mask (L,) bool  — the m currently-masked query positions
        repeat     int        — which repeat this corruption belongs to
    """
    L = ids.numel()
    query_idx = torch.nonzero(roles == _QUERY, as_tuple=False).flatten()  # (Q,)
    Q = query_idx.numel()
    if Q == 0:
        raise ValueError("no QUERY positions to score (empty query set)")
    n_context = int((roles == _CONTEXT).sum().item())
    unknown = roles == _UNKNOWN  # masked base: UNKNOWN only — CONTEXT stays concrete

    inner = max(1, cfg.inner_mc)
    corruptions: List[dict] = []
    for rep in range(max(1, cfg.n_repeats)):
        for i in range(masked_counts.numel()):
            m = int(masked_counts[i].item())
            w_i = float(weights[i].item())
            r = Q - m                            # query positions revealed
            coeff = (Q * w_i) / (m * inner)      # 1/t_eff = Q/m, exact by level
            for _ in range(inner):
                perm = torch.randperm(Q, generator=generator, device=device)
                revealed_pos = query_idx[perm[:r]]

                # Masked = UNKNOWN positions plus query residues not revealed
                # this draw. CONTEXT positions stay at their concrete token.
                masked_flag = unknown.clone()
                masked_flag[query_idx] = True
                masked_flag[revealed_pos] = False

                x_t = torch.where(masked_flag, torch.full_like(ids, MASK_ID), ids)
                # Currently-masked *query* positions are the only scored ones.
                score_mask = torch.zeros(L, dtype=torch.bool, device=device)
                score_mask[query_idx] = True
                score_mask[revealed_pos] = False

                corruptions.append(
                    {
                        "x_t": x_t,
                        "idx": n_context + r,
                        "coeff": coeff,
                        "score_mask": score_mask,
                        "repeat": rep,
                    }
                )
    return corruptions, Q


@torch.no_grad()
def _score_corruptions(
    model: torch.nn.Module,
    ids: torch.Tensor,        # (L,) target tokens
    z_c: torch.Tensor,        # (emb_dim,)
    corruptions: List[dict],
    n_repeats: int,
    max_batch: int,
    device: torch.device,
):
    """Run the model over all corruptions and reduce to a per-repeat ELBO.

    Returns a 1-D tensor of length ``n_repeats`` holding each repeat's SDMC
    ELBO estimate. The corruption ``coeff`` already folds in the quadrature
    weight, the ``1/t`` factor and the ``1/inner_mc`` average, so each repeat's
    ELBO is just the coeff-weighted sum of masked-query log-probs.
    """
    L = ids.numel()
    target = ids.view(1, L)
    z_row = z_c.view(1, -1)
    elbo = torch.zeros(n_repeats, dtype=torch.float32, device=device)

    for start in range(0, len(corruptions), max_batch):
        chunk = corruptions[start : start + max_batch]
        B = len(chunk)
        x_t = torch.stack([c["x_t"] for c in chunk], dim=0)                  # (B, L)
        idx = torch.tensor([c["idx"] for c in chunk], dtype=torch.long, device=device)
        coeff = torch.tensor([c["coeff"] for c in chunk], dtype=torch.float32, device=device)
        score = torch.stack([c["score_mask"] for c in chunk], dim=0).to(torch.float32)
        reps = torch.tensor([c["repeat"] for c in chunk], dtype=torch.long, device=device)

        logits = model(x=x_t, t=idx, y_c=z_row.expand(B, -1))               # (B, V, L)
        log_probs = F.log_softmax(logits, dim=1)
        lp = log_probs.gather(1, target.expand(B, L).unsqueeze(1)).squeeze(1)  # (B, L)
        lp_sum = (lp * score).sum(dim=1)                                     # (B,)
        contrib = coeff * lp_sum                                            # (B,)
        elbo.index_add_(0, reps, contrib)

    return elbo


# ─────────────────────────────────────────────────────────────────────────────
# Estimator
# ─────────────────────────────────────────────────────────────────────────────


class ProteoScribeLikelihoodEstimator:
    """Estimate ``log P(sequence | z_c)`` under a trained ProteoScribe model.

    Load once, score many sequences::

        est = ProteoScribeLikelihoodEstimator.from_weights(
            config="configs/inference/stage3_ProteoScribe_sample.json",
            weights_path="weights/ProteoScribe.pt",
        )
        result = est.estimate("MKTAYIAKQR", z_c)     # z_c: (emb_dim,) or (1, emb_dim)
        print(result.log_likelihood, result.bits_per_residue)

    Conditional / partial scoring:
        - ``"MKT###QR"`` marginalizes over the three ``#`` positions.
        - ``context_mask=[True, True, False, ...]`` pins leading residues as
          fixed context so ``estimate`` returns ``log P(rest | those, z_c)``.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        config: Namespace,
        device: Optional[Union[str, torch.device]] = None,
    ):
        self.device = torch.device(device) if device is not None else get_device()
        self.model = model.to(self.device).eval()
        self.config = config
        self.seq_len = int(getattr(config, "diffusion_steps"))

    @classmethod
    def from_weights(
        cls,
        config: Union[str, dict, Namespace],
        weights_path: Optional[str] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> "ProteoScribeLikelihoodEstimator":
        """Build an estimator from a config (path/dict/Namespace) and weights.

        ``config`` accepts the same JSON files Stage 3 inference uses (resolved
        through ``load_json_config`` so ``_base_configs`` composition works).
        ``weights_path`` may be a raw ``.pt``/``.bin``, a Lightning ``.ckpt``,
        or a DeepSpeed sharded directory; ``None`` yields a random-init model
        (useful for tests).
        """
        cfg = _coerce_config(config)
        dev = torch.device(device) if device is not None else get_device()
        model = prepare_model_ProteoScribe(
            config_args=cfg,
            model_fpath=weights_path,
            device=str(dev),
            strict=False,
            eval=True,
            attempt_correction=True,
            verbosity=1,
        )
        return cls(model=model, config=cfg, device=dev)

    def estimate(
        self,
        sequence: Union[str, Sequence[int], torch.Tensor],
        z_c: torch.Tensor,
        *,
        context_mask: Optional[Sequence[bool]] = None,
        config: Optional[LikelihoodConfig] = None,
    ) -> LikelihoodResult:
        """Estimate the (conditional) log-likelihood of ``sequence`` given ``z_c``.

        ``sequence`` is an amino-acid string (``#`` marks unknown positions) or
        a length-``diffusion_steps`` tensor/list of token ids. ``z_c`` is the
        Stage 2 conditioning vector, shape ``(emb_dim,)`` or ``(1, emb_dim)``.
        ``context_mask`` (bool, one entry per residue for a string / per
        position for an id tensor) pins positions as fixed context.
        """
        cfg = config or LikelihoodConfig()
        z = torch.as_tensor(z_c, dtype=torch.float32, device=self.device).view(1, -1)

        generator = None
        if cfg.seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(int(cfg.seed))

        ids, roles = _classify_sequence(
            sequence, self.seq_len, context_mask, cfg, self.device
        )
        Q = int((roles == _QUERY).sum().item())
        if Q == 0:
            raise ValueError("no QUERY positions to score (empty query set)")
        masked_counts, weights = _build_masked_levels(cfg, Q, self.device)
        corruptions, Q = _build_corruptions(
            ids, roles, masked_counts, weights, cfg, generator, self.device
        )
        n_repeats = max(1, cfg.n_repeats)
        elbo_per_repeat = _score_corruptions(
            self.model, ids, z, corruptions, n_repeats,
            cfg.max_forward_batch, self.device,
        )

        ll = float(elbo_per_repeat.mean().item())
        ll_std = float(elbo_per_repeat.std(unbiased=False).item()) if n_repeats > 1 else 0.0
        n_context = int((roles == _CONTEXT).sum().item())
        n_unknown = int((roles == _UNKNOWN).sum().item())

        # Q >= 1 here: _build_corruptions raises on an empty query set.
        nats_per_res = -ll / Q
        perplexity = math.exp(nats_per_res)
        bits_per_res = nats_per_res / math.log(2)

        per_quadrature = [
            {"masked": int(m), "revealed": Q - int(m), "w": round(float(w), 6)}
            for m, w in zip(masked_counts.tolist(), weights.tolist())
        ]

        return LikelihoodResult(
            log_likelihood=ll,
            log_likelihood_std=ll_std,
            perplexity=perplexity,
            bits_per_residue=bits_per_res,
            n_query=Q,
            n_context=n_context,
            n_unknown=n_unknown,
            sequence_length=self.seq_len,
            per_quadrature=per_quadrature,
        )


def _coerce_config(config: Union[str, dict, Namespace]) -> Namespace:
    if isinstance(config, Namespace):
        return config
    if isinstance(config, str):
        loaded = load_json_config(config)
        return loaded if isinstance(loaded, Namespace) else Namespace(**loaded)
    if isinstance(config, dict):
        return Namespace(**config)
    raise TypeError(f"config must be a path, dict, or Namespace, got {type(config)}")

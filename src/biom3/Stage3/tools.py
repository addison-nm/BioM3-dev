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
  over (revealed/masked across the SDMC trajectory) and scored.
- ``CONTEXT`` — a concrete residue that is *always revealed* and conditioned
  upon but never scored (the ``<START>`` anchor, the ``<PAD>`` tail, and any
  position the caller pins via ``context_mask``).
- ``UNKNOWN`` — an absorbing/mask position (``#`` in an input string). Never
  revealed, never scored; marginalized out of the likelihood.

The model's time index equals the number of currently-revealed (non-absorbing)
positions, so for a corruption that reveals ``r`` of the ``Q`` query residues
the index fed to the model is ``n_context + r`` (see
``run_ProteoScribe_sample._pre_revealed_offset`` for the same convention).
"""

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

    # SDMC time grid over the query-masking fraction t in (0, 1].
    n_quadrature: int = 16
    quadrature_grid: str = "uniform"          # "uniform" midpoints | "explicit"
    quadrature_points: Optional[List[float]] = None   # for "explicit"
    quadrature_weights: Optional[List[float]] = None  # for "explicit"; else uniform

    inner_mc: int = 1        # mask samples per grid point (per repeat)
    n_repeats: int = 4       # independent estimates → mean + std
    eps_t: float = 1e-3      # clamp t away from 0 in the 1/t factor

    # Include the <PAD> tail (and any padding beyond <END>) in the query set.
    # Off by default: padding is deterministic given <END> and would inflate
    # the log-likelihood with trivially-predicted tokens.
    score_padding: bool = False

    # Wrap the raw AA string with <START>/<END> before scoring. The <START>
    # anchor is always CONTEXT; <END> is QUERY (the model must decide where
    # the protein terminates), unless the input already carries them.
    add_special_tokens: bool = True

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
        import math

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
            ids[pos], roles[pos] = START_ID, _CONTEXT
            pos += 1
        content_start = pos
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
            ids[pos], roles[pos] = END_ID, _QUERY
            pos += 1
        content_end = pos  # first PAD index
    else:
        ids = torch.as_tensor(sequence, dtype=torch.long).reshape(-1).clone()
        if ids.numel() != seq_len:
            raise ValueError(
                f"id sequence length {ids.numel()} != diffusion_steps={seq_len}"
            )
        roles = torch.empty(seq_len, dtype=torch.long)
        for i, tok in enumerate(ids.tolist()):
            if tok == MASK_ID:
                roles[i] = _UNKNOWN
            elif tok in (PAD_ID, START_ID):
                roles[i] = _CONTEXT
            else:
                roles[i] = _QUERY
        if context_mask is not None:
            if len(context_mask) != seq_len:
                raise ValueError(
                    f"context_mask length {len(context_mask)} != seq_len {seq_len}"
                )
            for i, flag in enumerate(context_mask):
                if flag and roles[i] == _QUERY:
                    roles[i] = _CONTEXT
        content_end = seq_len

    if not cfg.score_padding:
        # PAD tail already CONTEXT for the string path; enforce for the id path.
        roles[(ids == PAD_ID)] = _CONTEXT
    else:
        # Promote padding to QUERY so its prediction is scored too.
        pad_mask = ids == PAD_ID
        roles[pad_mask] = _QUERY

    return ids.to(device), roles.to(device)


# ─────────────────────────────────────────────────────────────────────────────
# SDMC quadrature
# ─────────────────────────────────────────────────────────────────────────────


def _build_query_grid(cfg: LikelihoodConfig, device: torch.device):
    """Build the SDMC grid over the query-masking fraction t in (0, 1].

    Returns ``(t_floats, weights)`` with ``weights`` summing to 1. ``t`` is the
    fraction of *query* residues that are masked; the number revealed at grid
    point ``n`` is ``round((1 - t_n) * Q)``.
    """
    if cfg.quadrature_grid == "uniform":
        N = max(1, int(cfg.n_quadrature))
        t_floats = torch.tensor(
            [(n - 0.5) / N for n in range(1, N + 1)], dtype=torch.float32, device=device
        )
        weights = torch.full((N,), 1.0 / N, dtype=torch.float32, device=device)
    elif cfg.quadrature_grid == "explicit":
        if not cfg.quadrature_points:
            raise ValueError("quadrature_grid='explicit' requires quadrature_points")
        t_floats = torch.tensor(cfg.quadrature_points, dtype=torch.float32, device=device)
        N = t_floats.numel()
        if cfg.quadrature_weights:
            if len(cfg.quadrature_weights) != N:
                raise ValueError("quadrature_weights length must match quadrature_points")
            weights = torch.tensor(
                cfg.quadrature_weights, dtype=torch.float32, device=device
            )
        else:
            weights = torch.full((N,), 1.0 / N, dtype=torch.float32, device=device)
    else:
        raise ValueError(
            f"quadrature_grid must be 'uniform' or 'explicit', got {cfg.quadrature_grid!r}"
        )
    if (t_floats <= 0).any() or (t_floats > 1).any():
        raise ValueError(f"quadrature t values must be in (0, 1]; got {t_floats.tolist()}")
    return t_floats, weights


def _build_corruptions(
    ids: torch.Tensor,        # (L,) concrete target tokens; MASK at unknown
    roles: torch.Tensor,      # (L,) role per position
    t_floats: torch.Tensor,   # (N,)
    weights: torch.Tensor,    # (N,)
    cfg: LikelihoodConfig,
    generator: Optional[torch.Generator],
    device: torch.device,
):
    """Sample masked corruptions, one bundle per (repeat, grid point, inner_mc).

    Each corruption reveals a random subset of the query residues, masks the
    rest (plus all UNKNOWN positions), keeps every CONTEXT position at its
    concrete token, and scores only the currently-masked query residues.

    Returns a list of dicts with:
        x_t        (L,) long  — model input (MASK at masked/unknown)
        idx        int        — model time index = n_context + r
        coeff      float      — w_n / (inner_mc * max(t_n, eps_t))
        score_mask (L,) bool  — currently-masked query positions
        repeat     int        — which repeat this corruption belongs to
    """
    L = ids.numel()
    query_idx = torch.nonzero(roles == _QUERY, as_tuple=False).flatten()  # (Q,)
    Q = query_idx.numel()
    if Q == 0:
        raise ValueError("no QUERY positions to score (empty query set)")
    n_context = int((roles == _CONTEXT).sum().item())
    context_and_unknown = roles != _QUERY  # positions never carrying a revealed query token

    inner = max(1, cfg.inner_mc)
    corruptions: List[dict] = []
    for rep in range(max(1, cfg.n_repeats)):
        for n in range(t_floats.numel()):
            t_n = float(t_floats[n].item())
            w_n = float(weights[n].item())
            r = int(round((1.0 - t_n) * Q))
            r = max(0, min(Q, r))
            coeff = w_n / (inner * max(t_n, cfg.eps_t))
            for _ in range(inner):
                perm = torch.randperm(Q, generator=generator, device=device)
                revealed_local = perm[:r]
                revealed_pos = query_idx[revealed_local]

                revealed_flag = torch.zeros(L, dtype=torch.bool, device=device)
                revealed_flag[revealed_pos] = True
                # Masked = every non-context position that is not a revealed query.
                masked_flag = context_and_unknown.clone()
                masked_flag[query_idx] = True
                masked_flag[revealed_pos] = False

                x_t = torch.where(
                    masked_flag, torch.full_like(ids, MASK_ID), ids
                )
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
        t_floats, weights = _build_query_grid(cfg, self.device)
        corruptions, Q = _build_corruptions(
            ids, roles, t_floats, weights, cfg, generator, self.device
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

        import math

        nats_per_res = -ll / Q if Q > 0 else float("nan")
        perplexity = math.exp(nats_per_res) if Q > 0 else float("nan")
        bits_per_res = nats_per_res / math.log(2) if Q > 0 else float("nan")

        per_quadrature = [
            {"t": round(float(t), 6), "w": round(float(w), 6)}
            for t, w in zip(t_floats.tolist(), weights.tolist())
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

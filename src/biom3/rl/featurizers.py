"""Sequence featurizers for the surrogate-reward pipeline.

A ``Featurizer`` turns a list of amino-acid sequences into a fixed-shape
``(N, D)`` ``np.ndarray`` suitable as input to a regressor. The trainer
script (``scripts/train_grpo_surrogate.py``) calls a featurizer once on
the full training set; the eval / inference path inside
``SurrogateReward`` calls it once per GRPO step on a small batch.

Three implementations:

- ``OneHotFeaturizer`` — pad/truncate to a fixed length, one-hot over
  the 20 canonical amino acids. CPU-only, no model deps.
- ``ESM2MeanPoolFeaturizer`` — load ESM-2 directly via
  ``esm.pretrained.load_model_and_alphabet`` (the same call Stage 1's
  ``ProteinEncoder.get_ESM_model`` makes — see
  src/biom3/Stage1/model.py:51-57), tokenize via the alphabet's
  batch converter, run forward at ``layer``, and mean-pool over residue
  positions. Returns ``(N, hidden_dim)`` — 1280 for the 33-layer 650M
  weights at ``weights/LLMs/esm2_t33_650M_UR50D.pt``.
- ``PenCLZpFeaturizer`` — PenCL's *joint-latent* protein embedding
  ``z_p``, i.e. the CLS token through the projection head. This is the
  space Stage 1 aligns against text and the space
  ``biom3.geometry`` manifolds are fitted in. Not interchangeable with
  ``ESM2MeanPoolFeaturizer``: that one mean-pools raw ESM-2 residues and
  never touches the projection head.
"""

import os
from typing import List, Optional, Protocol

import numpy as np
import torch

from biom3.backend.device import setup_logger

logger = setup_logger(__name__)


CANONICAL_AA = "ACDEFGHIKLMNPQRSTVWY"
_AA_TO_IDX = {a: i for i, a in enumerate(CANONICAL_AA)}


class Featurizer(Protocol):
    def __call__(self, sequences: List[str]) -> np.ndarray: ...


def _clean(seq: str) -> str:
    return "".join(c for c in seq if c in _AA_TO_IDX)


class OneHotFeaturizer:
    """Pad/truncate to ``max_length`` and one-hot over the 20 canonical AAs.

    Output: ``np.ndarray`` shape ``(N, max_length * 20)``, dtype float32.
    Sequences shorter than ``max_length`` are zero-padded; longer ones
    are truncated. Non-canonical characters are dropped before encoding
    (matches the behavior of decoded GRPO rollouts after
    ``decode_tokens``).
    """

    def __init__(self, max_length: int = 256):
        if max_length <= 0:
            raise ValueError(f"max_length must be positive, got {max_length}")
        self.max_length = max_length
        self.dim = max_length * len(CANONICAL_AA)

    def __call__(self, sequences: List[str]) -> np.ndarray:
        n_aa = len(CANONICAL_AA)
        out = np.zeros((len(sequences), self.max_length, n_aa), dtype=np.float32)
        for i, seq in enumerate(sequences):
            clean = _clean(seq)[: self.max_length]
            for j, c in enumerate(clean):
                out[i, j, _AA_TO_IDX[c]] = 1.0
        return out.reshape(len(sequences), self.dim)


class ESM2MeanPoolFeaturizer:
    """ESM-2 mean-pooled embedding for each sequence.

    Lazy-loads the model on first call (mirrors ``ESMFoldReward._load``)
    so module import stays cheap. Tokenizes via the alphabet's batch
    converter, runs a single forward at ``layer``, and mean-pools over
    valid residue positions only (excludes BOS / EOS / pad).
    """

    def __init__(
        self,
        model_path: str = "./weights/LLMs/esm2_t33_650M_UR50D.pt",
        layer: int = 33,
        device: Optional[str] = None,
        max_length: int = 256,
        batch_size: int = 16,
    ):
        self.model_path = os.path.expanduser(model_path)
        self.layer = layer
        self.device = device
        self.max_length = max_length
        self.batch_size = batch_size
        self._model = None
        self._alphabet = None
        self._batch_converter = None
        self._dim: Optional[int] = None

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._load()
        assert self._dim is not None
        return self._dim

    def _load(self):
        if self._model is not None:
            return
        try:
            import esm  # noqa: F401  (the load_model_and_alphabet call below imports as needed)
            from esm import pretrained
        except ImportError as e:
            raise ImportError(
                "ESM2MeanPoolFeaturizer needs `fair-esm`. Install via the "
                "[grpo] extras: pip install -e '.[grpo]'"
            ) from e
        logger.info("Loading ESM-2 from %s ...", self.model_path)
        model, alphabet = pretrained.load_model_and_alphabet(self.model_path)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        if self.device is not None:
            model = model.to(self.device)
        self._model = model
        self._alphabet = alphabet
        self._batch_converter = alphabet.get_batch_converter()
        # ESM-2 650M has 1280-dim hidden; use the actual model attr.
        self._dim = model.embed_dim if hasattr(model, "embed_dim") else 1280
        logger.info("ESM-2 loaded (layer=%d, dim=%d)", self.layer, self._dim)

    @torch.no_grad()
    def __call__(self, sequences: List[str]) -> np.ndarray:
        self._load()
        cleaned = [_clean(s)[: self.max_length] for s in sequences]
        out = np.zeros((len(cleaned), self._dim), dtype=np.float32)
        # Sequences with no canonical AA after cleaning get an all-zero embedding.
        active = [(i, s) for i, s in enumerate(cleaned) if s]
        if not active:
            return out

        for start in range(0, len(active), self.batch_size):
            chunk = active[start : start + self.batch_size]
            tokens_in = [(f"seq_{i}", s) for i, s in chunk]
            _, _, batch_tokens = self._batch_converter(tokens_in)
            if self.device is not None:
                batch_tokens = batch_tokens.to(self.device)
            results = self._model(batch_tokens, repr_layers=[self.layer], return_contacts=False)
            reps = results["representations"][self.layer]      # (B, L+2, D)
            # Build a residue-only mask: drop BOS, EOS, and pad. ESM-2's pad token
            # idx lives on the alphabet; everything that isn't BOS/EOS/pad is a
            # valid residue.
            pad_idx = self._alphabet.padding_idx
            bos_idx = self._alphabet.cls_idx
            eos_idx = self._alphabet.eos_idx
            mask = (
                (batch_tokens != pad_idx)
                & (batch_tokens != bos_idx)
                & (batch_tokens != eos_idx)
            ).float().unsqueeze(-1)                            # (B, L+2, 1)
            pooled = (reps * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
            pooled = pooled.float().cpu().numpy()
            for (i, _), vec in zip(chunk, pooled):
                out[i] = vec
        return out


class PenCLZpFeaturizer:
    """Stage 1 joint-latent protein embedding ``z_p`` for each sequence.

    Wraps an already-loaded ``pfam_PEN_CL`` (the one ``gdpo_train`` and
    ``gdpo_train_multinode`` already hold for prompt encoding, so this
    costs no extra weights) and reproduces the protein branch of its
    forward: CLS token out of ``protein_encoder``, then
    ``protein_projection``. Returns ``(N, emb_dim)`` float64 — float64
    because the consumer is a Mahalanobis quadratic form.

    Two properties are load-bearing when the output feeds
    ``biom3.geometry``; both were measured against the 911 reference
    vectors of ``manifold_sho1_911_run1.npz``:

    - **fp32 only.** Stage 1 inference runs bf16 autocast on XPU by
      default. At bf16 the cosine to the fp32 vector is still 0.99984,
      but d_E moves by up to 8.68 against a band spanning 4.14-17.46 —
      the Mahalanobis form amplifies exactly the low-variance directions
      reduced precision perturbs. Autocast is therefore explicitly
      disabled here rather than merely not enabled, so an ambient
      autocast in a caller cannot silently change the numbers.
    - **Padding mode is free.** Dynamic padding to the batch max and
      padding to a fixed 1024 agree to cosine 1.000000 (ESM-2 masks pads
      internally), so this uses dynamic padding — for SH3-length inputs
      that is a ~70-token forward instead of a 1024-token one.

    Sequences that clean to fewer than ``min_length`` residues get an
    all-zero row; ``ManifoldReward`` maps those to the worst reward
    rather than letting a degenerate rollout score well by accident.
    """

    def __init__(self, pencl, max_length: int = 1024, min_length: int = 1,
                 batch_size: int = 32):
        self.pencl = pencl
        self.max_length = max_length
        self.min_length = min_length
        self.batch_size = batch_size
        self._alphabet = pencl.protein_encoder.alphabet
        self._batch_converter = self._alphabet.get_batch_converter()
        self.device = next(pencl.parameters()).device
        self.dim = int(pencl.protein_projection.projection.out_features)

    @torch.no_grad()
    def __call__(self, sequences: List[str]) -> np.ndarray:
        out = np.zeros((len(sequences), self.dim), dtype=np.float64)
        cleaned = [_clean(s)[: self.max_length] for s in sequences]
        active = [(i, s) for i, s in enumerate(cleaned) if len(s) >= self.min_length]
        if not active:
            return out

        for start in range(0, len(active), self.batch_size):
            chunk = active[start : start + self.batch_size]
            _, _, tokens = self._batch_converter([(f"s{i}", s) for i, s in chunk])
            tokens = tokens.to(self.device)
            # enabled=False, not "no autocast context" — see class docstring.
            with torch.autocast(device_type=self.device.type, enabled=False):
                z = self.pencl.protein_projection(self.pencl.protein_encoder(tokens))
            z = z.float().cpu().numpy()
            for (i, _), vec in zip(chunk, z):
                out[i] = vec
        return out


def build_featurizer(name: str, **kwargs) -> Featurizer:
    if name == "onehot":
        return OneHotFeaturizer(**kwargs)
    if name == "esm2":
        return ESM2MeanPoolFeaturizer(**kwargs)
    if name == "pencl_zp":
        return PenCLZpFeaturizer(**kwargs)
    raise ValueError(f"Unknown featurizer '{name}'. Known: onehot, esm2, pencl_zp")

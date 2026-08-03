"""Cross-domain coupling for the composed multidomain decoder.

A K-domain protein is decoded on K parallel canvases, one per domain, each driven
by its own ProteoScribe expert. This module is the only place the canvases talk to
each other: for every ordered pair of domains ``(a, b)`` and every transformer
layer it holds a dedicated cross-attention projection set, so domain ``a`` reads a
summary of domain ``b``'s current hidden state.

Output projections are zero-initialised, which makes the coupling contribute
exactly zero at initialisation. The composed decoder therefore starts bit-exactly
equal to the K independent experts, and training opens the interface from there.
That property is asserted, not assumed — see :mod:`biom3.Stage3.multidomain.audit`.

Note that ``linear_attn`` normalizes keys over the sequence axis and contracts to a
per-head ``dim_head x dim_head`` context matrix, so the partner canvas enters as a
pooled summary rather than through position-to-position attention.
"""

import torch
import torch.nn as nn

from linear_attention_transformer.linear_attention_transformer import linear_attn


def _pair_key(a: int, b: int) -> str:
    return f"{a}_{b}"


class AllPairsCoupling(nn.Module):
    """Directed per-layer cross-attention between K domain canvases.

    Holds one q/k/v/out projection set per ordered pair ``(a, b)`` per layer, plus
    one input LayerNorm per domain per layer. ``K == 2`` gives the two directed
    couplings of the two-domain case; larger K adds every ordered pair, so the
    parameter count grows as ``K * (K - 1)``.

    Args:
        num_domains: number of canvases, K >= 2
        n_layers: transformer layers per block in each expert
        dim: expert embedding dimension
        heads: attention heads used to split ``dim`` for the cross term
    """

    def __init__(self, num_domains: int, n_layers: int, dim: int, heads: int):
        super().__init__()
        if num_domains < 2:
            raise ValueError(f"num_domains must be >= 2, got {num_domains}")
        if dim % heads != 0:
            raise ValueError(f"dim {dim} is not divisible by heads {heads}")

        self.num_domains = int(num_domains)
        self.n_layers = int(n_layers)
        self.dim = int(dim)
        self.heads = int(heads)
        self.dim_head = self.dim // self.heads
        self.pairs = [(a, b)
                      for a in range(self.num_domains)
                      for b in range(self.num_domains)
                      if a != b]

        def _projections(bias):
            return nn.ModuleDict({
                _pair_key(a, b): nn.ModuleList([
                    nn.Linear(self.dim, self.dim, bias=bias) for _ in range(self.n_layers)
                ])
                for a, b in self.pairs
            })

        self.q_proj = _projections(False)
        self.k_proj = _projections(False)
        self.v_proj = _projections(False)
        self.out_proj = _projections(True)

        self.norms = nn.ModuleList([
            nn.ModuleList([nn.LayerNorm(self.dim) for _ in range(self.n_layers)])
            for _ in range(self.num_domains)
        ])

        self.reset_to_additive_null()

    def reset_to_additive_null(self):
        """Zero every output projection so the coupling contributes nothing."""
        for a, b in self.pairs:
            out = self.out_proj[_pair_key(a, b)]
            for layer in out:
                nn.init.zeros_(layer.weight)
                nn.init.zeros_(layer.bias)

    def _split_heads(self, x):
        b, n, _ = x.shape
        return x.reshape(b, n, self.heads, self.dim_head).transpose(1, 2)

    def _merge_heads(self, x):
        b, _, n, _ = x.shape
        return x.transpose(1, 2).reshape(b, n, self.dim)

    def cross_terms(self, layer_idx: int, layer_inputs, real_masks):
        """Cross-domain contribution for each canvas at one transformer layer.

        Args:
            layer_idx: index of the layer within the block
            layer_inputs: K tensors of shape [B, L, dim] — each canvas's layer input
            real_masks: K bool tensors of shape [B, L] marking non-pad positions, so
                a canvas only attends to its partners' real residues

        Returns:
            list of K tensors of shape [B, L, dim], summed over that domain's partners
        """
        normed = [self.norms[d][layer_idx](layer_inputs[d])
                  for d in range(self.num_domains)]

        cross = [None] * self.num_domains
        for a, b in self.pairs:
            key = _pair_key(a, b)
            q = self._split_heads(self.q_proj[key][layer_idx](normed[a]))
            k = self._split_heads(self.k_proj[key][layer_idx](normed[b]))
            v = self._split_heads(self.v_proj[key][layer_idx](normed[b]))
            attended = linear_attn(q, k, v, kv_mask=real_masks[b])
            term = self.out_proj[key][layer_idx](self._merge_heads(attended))
            cross[a] = term if cross[a] is None else cross[a] + term
        return cross

    def is_additive_null(self, tol: float = 0.0) -> bool:
        """Whether every output projection is still zero within ``tol``."""
        for a, b in self.pairs:
            for layer in self.out_proj[_pair_key(a, b)]:
                if layer.weight.abs().max().item() > tol:
                    return False
                if layer.bias.abs().max().item() > tol:
                    return False
        return True

    def parameter_report(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "num_domains": self.num_domains,
            "ordered_pairs": len(self.pairs),
            "n_layers": self.n_layers,
            "dim": self.dim,
            "heads": self.heads,
            "total_params": total,
            "trainable_params": trainable,
            "additive_null": self.is_additive_null(),
        }

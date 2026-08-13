"""The composed multidomain ProteoScribe decoder.

K domain canvases are decoded in parallel, each by its own expert, with an
:class:`~biom3.Stage3.multidomain.coupling.AllPairsCoupling` term added to every
transformer layer's output so the canvases can condition on each other.

The per-domain forward pass reproduces
:meth:`biom3.Stage3.cond_diff_transformer_layer.LinearAttentionTransformerEmbedding.forward`
step by step, driving each expert's own submodules one layer at a time so the
cross term can be injected between layers. That duplication is deliberate — this
subpackage does not modify the shared decoder — and
``test_composed_matches_shared_forward`` pins the two together at the additive
null so they cannot silently diverge.
"""

import torch
import torch.nn as nn

from biom3.Stage3.cond_diff_transformer_layer import get_model
from biom3.Stage3.preprocess import create_num_seqs

# The shared vocabulary's gap token; read from create_num_seqs so the id can never
# drift from the tokenizer the canvases are built with.
PAD_ID = create_num_seqs(['-'])[0]


class MultiDomainProteoScribe(nn.Module):
    """K ProteoScribe experts on K canvases, coupled per transformer layer.

    Args:
        experts: K ``DiffTransformer`` modules, in N->C domain order
        coupling: the cross-domain coupling, sized to match the experts

    Each expert is registered exactly once, under ``experts.<d>``. Prior
    references for expert adaptation live outside the module tree so they stay
    out of ``state_dict()``.
    """

    def __init__(self, experts, coupling):
        super().__init__()
        experts = list(experts)
        if len(experts) != coupling.num_domains:
            raise ValueError(
                f"got {len(experts)} experts but the coupling is built for "
                f"{coupling.num_domains} domains"
            )
        reference = experts[0].transformer
        for d, expert in enumerate(experts):
            transformer = expert.transformer
            for attr in ("emb_dim", "n_blocks", "depth"):
                if getattr(transformer, attr) != getattr(reference, attr):
                    raise ValueError(
                        f"expert {d} has {attr}={getattr(transformer, attr)}, "
                        f"expected {getattr(reference, attr)} to match expert 0"
                    )
        if reference.emb_dim != coupling.dim:
            raise ValueError(
                f"expert embedding dim {reference.emb_dim} != coupling dim {coupling.dim}"
            )
        if reference.depth != coupling.n_layers:
            raise ValueError(
                f"expert depth {reference.depth} != coupling n_layers {coupling.n_layers}"
            )

        self.experts = nn.ModuleList(experts)
        self.coupling = coupling
        self.emb_dim = reference.emb_dim
        self.n_blocks = reference.n_blocks
        self.depth = reference.depth

    @property
    def num_domains(self) -> int:
        return len(self.experts)

    def _transformer(self, d):
        return self.experts[d].transformer

    def _embed_domain(self, d, x_d, t_d, y_d):
        """Token, time and conditioning embeddings for one canvas.

        Mirrors the embedding half of the shared decoder's forward pass.
        """
        transformer = self._transformer(d)
        batch_size = x_d.size(0)

        t_e = transformer.time_pos_emb(t_d).type(
            [p.dtype for p in transformer.mlp.parameters()][0])
        t_e = transformer.mlp(t_e)
        time_embed = t_e.reshape(
            batch_size, 1, transformer.emb_dim, transformer.n_blocks, transformer.depth)

        x_e = transformer.x_emb_NN(x_d.long())
        x_pos = transformer.axial_pos_emb(x_e).type(x_e.type())
        x_embed_axial = x_e + x_pos

        # y_d arrives as the precomputed fp32 conditioning from the dataloader,
        # so it needs the same cast the time embedding above gets: under
        # DeepSpeed the parameters are bf16 and F.linear refuses the mismatch.
        y_emb = transformer.y_mlp(
            y_d.type([p.dtype for p in transformer.y_mlp.parameters()][0])
        ).reshape(
            batch_size, 1, transformer.emb_dim, transformer.n_blocks, transformer.depth)

        return x_embed_axial, time_embed, y_emb

    def forward(self, x, t, y_c, real_masks=None, couple=True):
        """Decode all K canvases jointly.

        Args:
            x: token grids of shape [B, K, L]
            t: timesteps of shape [B] (shared across canvases) or [B, K]
            y_c: conditioning of shape [B, K, D]
            real_masks: bool [B, K, L] marking non-pad positions; derived from
                ``x != PAD_ID`` when omitted
            couple: set False to run the canvases independently, which is the
                zero-coupling ablation and the additive-null reference

        Returns:
            logits of shape [B, K, num_classes, L]
        """
        num_domains = self.num_domains
        if x.dim() != 3 or x.size(1) != num_domains:
            raise ValueError(
                f"expected x of shape [B, {num_domains}, L], got {tuple(x.shape)}")
        if y_c.dim() != 3 or y_c.size(1) != num_domains:
            raise ValueError(
                f"expected y_c of shape [B, {num_domains}, D], got {tuple(y_c.shape)}")

        t = t.view(-1, 1).expand(-1, num_domains) if t.dim() == 1 else t
        if real_masks is None:
            real_masks = x != PAD_ID

        embedded = [self._embed_domain(d, x[:, d], t[:, d].reshape(-1), y_c[:, d])
                    for d in range(num_domains)]
        masks = [real_masks[:, d] for d in range(num_domains)]

        hidden = [torch.zeros_like(x_embed_axial) for x_embed_axial, _, _ in embedded]
        for i in range(self.n_blocks):
            for d in range(num_domains):
                hidden[d] = hidden[d] + embedded[d][0]
            for j in range(self.depth):
                layer_inputs = [
                    hidden[d] + embedded[d][1][..., i, j] + embedded[d][2][..., i, j]
                    for d in range(num_domains)
                ]
                outputs = [
                    self._transformer(d).transformer_blocks[i][j](layer_inputs[d])
                    for d in range(num_domains)
                ]
                if couple:
                    cross = self.coupling.cross_terms(j, layer_inputs, masks)
                    outputs = [outputs[d] + cross[d] for d in range(num_domains)]
                hidden = outputs

        logits = []
        for d in range(num_domains):
            transformer = self._transformer(d)
            logits.append(transformer.out(transformer.norm(hidden[d])).permute(0, 2, 1))
        return torch.stack(logits, dim=1)

    def standalone_logits(self, d, x_d, t_d, y_d):
        """One canvas through its expert alone, with no cross term.

        The reference the additive-null gate compares against, and the forward
        used by the generation-space prior-preservation objective.
        """
        transformer = self._transformer(d)
        x_embed_axial, time_embed, y_emb = self._embed_domain(d, x_d, t_d, y_d)
        h = torch.zeros_like(x_embed_axial)
        for i in range(self.n_blocks):
            h = h + x_embed_axial
            for j in range(self.depth):
                h = transformer.transformer_blocks[i][j](
                    h + time_embed[..., i, j] + y_emb[..., i, j])
        return transformer.out(transformer.norm(h)).permute(0, 2, 1)


def build_multidomain_model(config_args, num_domains: int, experts=None):
    """Build a composed decoder, freshly initialised at the additive null.

    Args:
        config_args: ProteoScribe model args (``transformer_dim``, ``image_size``,
            ``num_classes`` and friends), shared by every expert
        num_domains: K
        experts: pre-built experts to compose; K fresh ones are built when omitted

    Returns:
        MultiDomainProteoScribe
    """
    from biom3.Stage3.multidomain.coupling import AllPairsCoupling

    if experts is None:
        experts = [
            get_model(
                args=config_args,
                data_shape=(config_args.image_size, config_args.image_size),
                num_classes=config_args.num_classes,
            )
            for _ in range(num_domains)
        ]
    reference = experts[0].transformer
    coupling = AllPairsCoupling(
        num_domains=num_domains,
        n_layers=reference.depth,
        dim=reference.emb_dim,
        heads=config_args.transformer_heads,
    )
    return MultiDomainProteoScribe(experts, coupling)

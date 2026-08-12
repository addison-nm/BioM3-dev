"""Lightning module for composed multidomain training.

The objective follows the reference design: **one diffusion timestep per example,
shared across all K canvases**, with an **independent unmasking path per canvas**,
and the per-canvas OA-ARDM reconstruction losses summed. Sharing the timestep is
what makes the assembly a single coupled trajectory — the alternative, drawing
``idx`` per canvas, would mask one canvas at step 40 while telling the model it is
at step 900 for its partner.

Component B (expert adaptation) adds ``lambda * sum_d ||W_d - W_ref_d||^2``,
pulling each expert toward the weights it started from. Its reference tensors are
non-persistent buffers, so they move with ``.to(device)`` but never enter
``state_dict()`` — a checkpoint stores each expert once.

Conditioning is ``y_d = alpha_d * z_p_d + (1 - alpha_d) * z_c_d``, with alpha drawn
**independently per (example, domain)** so a batch can pair a protein-faithful
domain with a text-driven one.
"""

import torch
import torch.nn.functional as F
from torch.distributions import OneHotCategorical

import biom3.Stage3.transformer_training_helper as trainer_tools
from biom3.backend.device import BACKEND_NAME, _XPU, setup_logger
from biom3.Stage3.multidomain.model import PAD_ID
# Reused rather than restated: these define the alpha vocabulary ("zc"/"zp"/
# "blend"/"spread"/constant) and the sequence-keyed deterministic eval alpha that
# the single-domain path is already tested against.
from biom3.Stage3.PL_wrapper import (
    ALPHA_BLEND,
    EVAL_SPREAD,
    deterministic_alpha,
    normalize_alpha_spec,
    resolve_eval_alpha,
)

if BACKEND_NAME == _XPU:
    import lightning as pl
else:
    import pytorch_lightning as pl

logger = setup_logger(__name__)

PRIOR_WEIGHT = "weight"
PRIOR_GENERATION = "generation"


class PL_ProtARDM_Multidomain(pl.LightningModule):
    """Composed multidomain diffusion training.

    Args:
        args: run configuration (``lr``, ``weight_decay``, ``choose_optim``, ...)
        model: a :class:`~biom3.Stage3.multidomain.model.MultiDomainProteoScribe`
        embedder: frozen text->z_c front-end, kept outside the module tree
        spec: the :class:`~biom3.Stage3.multidomain.io.MultiDomainSpec` this model
            was built from; stored in the checkpoint so it is self-describing
        zp_lookup: sequence -> z_p mapping, required when alpha weights z_p
        train_alpha / eval_alpha: conditioning blend specs
        expert_prior_lambda: Component B strength; 0 disables it entirely
        prior_mode: ``"weight"`` (cheap ||W - W_ref||^2) or ``"generation"``
            (KL between each expert's standalone logits and its reference's)
    """

    def __init__(self, args, model, embedder, *, spec=None, zp_lookup=None,
                 train_alpha=0.0, eval_alpha=EVAL_SPREAD,
                 expert_prior_lambda=0.0, prior_mode=PRIOR_WEIGHT,
                 metrics_all_domains=False):
        super().__init__()
        self.script_args = args
        self.model = model
        self.spec = spec

        embedder.eval()
        for param in embedder.parameters():
            param.requires_grad = False
        # One-element list keeps the frozen front-end out of self.parameters()
        # (so the optimizer and DeepSpeed never see it) and out of checkpoints.
        self._embedder_ref = [embedder]

        self.zp_lookup = zp_lookup
        self.train_alpha = normalize_alpha_spec(train_alpha)
        self.eval_alpha = resolve_eval_alpha(eval_alpha)
        self.expert_prior_lambda = float(expert_prior_lambda)
        self.prior_mode = prior_mode
        self.metrics_all_domains = metrics_all_domains

        if self.expert_prior_lambda > 0:
            self._register_prior_references()
        else:
            self._prior_refs = None

        if spec is not None:
            from biom3.Stage3.multidomain.io import state_dict_fingerprint
            self.save_hyperparameters({
                "multidomain_spec": spec.to_dict(),
                "multidomain_fingerprint": state_dict_fingerprint(model),
            })

    # ── Component B references ────────────────────────────────────────────

    def _register_prior_references(self):
        """Snapshot each expert's weights as non-persistent buffers.

        Non-persistent so they stay out of ``state_dict()``: the reference *is*
        the pretrained expert, which already exists on disk, and persisting it
        would double every checkpoint.
        """
        self._prior_refs = []
        for d, expert in enumerate(self.model.experts):
            names = []
            for i, (name, param) in enumerate(expert.named_parameters()):
                buffer_name = f"_prior_ref_{d}_{i}"
                self.register_buffer(buffer_name, param.detach().clone(),
                                     persistent=False)
                names.append(buffer_name)
            self._prior_refs.append(names)

    def _prior_penalty(self, masked, timesteps, y_c):
        if self.prior_mode == PRIOR_WEIGHT:
            penalty = 0.0
            for d, expert in enumerate(self.model.experts):
                for buffer_name, param in zip(self._prior_refs[d],
                                              expert.parameters()):
                    penalty = penalty + (param - getattr(self, buffer_name)).pow(2).sum()
            return penalty
        # Generation-space: keep each expert's *standalone* behaviour near its
        # reference, which is the quantity that matters for not erasing the fold.
        penalty = 0.0
        for d in range(self.model.num_domains):
            trained = self.model.standalone_logits(
                d, masked[d], timesteps[:, d], y_c[:, d])
            with torch.no_grad():
                reference = self._reference_standalone_logits(
                    d, masked[d], timesteps[:, d], y_c[:, d])
            penalty = penalty + F.kl_div(
                F.log_softmax(trained, dim=1), F.softmax(reference, dim=1),
                reduction="batchmean")
        return penalty

    def _reference_standalone_logits(self, d, x_d, t_d, y_d):
        """Standalone logits under the *reference* weights, restored afterwards."""
        expert = self.model.experts[d]
        saved = [param.detach().clone() for param in expert.parameters()]
        try:
            with torch.no_grad():
                for buffer_name, param in zip(self._prior_refs[d],
                                              expert.parameters()):
                    param.copy_(getattr(self, buffer_name))
            return self.model.standalone_logits(d, x_d, t_d, y_d)
        finally:
            with torch.no_grad():
                for param, original in zip(expert.parameters(), saved):
                    param.copy_(original)

    @torch.no_grad()
    def expert_delta_norms(self):
        """Relative drift per expert, for the Component B erasing monitor."""
        if self._prior_refs is None:
            return {}
        deltas = {}
        for d, expert in enumerate(self.model.experts):
            numerator = 0.0
            denominator = 0.0
            for buffer_name, param in zip(self._prior_refs[d], expert.parameters()):
                reference = getattr(self, buffer_name)
                numerator += (param - reference).pow(2).sum().item()
                denominator += reference.pow(2).sum().item()
            deltas[d] = (numerator ** 0.5) / (denominator ** 0.5 + 1e-12)
        return deltas

    # ── conditioning ──────────────────────────────────────────────────────

    @property
    def embedder(self):
        return self._embedder_ref[0]

    def _is_training_batch(self):
        trainer = getattr(self, "_trainer", None)
        return self.training if trainer is None else trainer.training

    def _train_alpha(self, n, device):
        if self.train_alpha != ALPHA_BLEND:
            return torch.full((n, 1), float(self.train_alpha), device=device)
        # {alpha=1: .25, alpha=0: .25, U(0,1): .5}, drawn per (example, domain).
        r = torch.rand(n, device=device)
        a = torch.rand(n, device=device)
        a = torch.where(r < 0.25, torch.ones_like(a), a)
        a = torch.where(r >= 0.75, torch.zeros_like(a), a)
        return a.view(n, 1)

    def _eval_alpha(self, sequences, device):
        if self.eval_alpha == EVAL_SPREAD:
            values = [deterministic_alpha(s) for s in sequences]
            return torch.tensor(values, dtype=torch.float32, device=device).view(-1, 1)
        return torch.full((len(sequences), 1), self.eval_alpha, device=device)

    def on_after_batch_transfer(self, batch, dataloader_idx):
        """Embed each domain's caption to z_c, optionally blending in z_p.

        ``input_ids`` arrives as ``[B, K, T]``; the front-end runs over the
        flattened ``B*K`` rows so every domain draws its own alpha, and the
        conditioning is returned as ``[B, K, D]``.
        """
        num_seqs, input_ids = batch[0], batch[1]
        embedder = self.embedder
        if next(embedder.parameters()).device != input_ids.device:
            embedder.to(input_ids.device)

        lead = input_ids.shape[:-1]
        flat_ids = input_ids.reshape(-1, input_ids.size(-1))
        with torch.no_grad():
            z_c = embedder(flat_ids)

        if self.zp_lookup is None:
            return [num_seqs, z_c.reshape(*lead, -1)]

        if len(batch) < 3:
            raise RuntimeError(
                "z_p blending needs the raw domain sequences in the batch; build "
                "the data module with needs_unique_sequences=True so the collate "
                "emits them"
            )
        sequences = batch[2]
        if len(sequences) != z_c.size(0):
            raise RuntimeError(
                f"batch carries {len(sequences)} domain sequences but "
                f"{z_c.size(0)} caption rows; the collate must emit one sequence "
                "per caption, flattened the same way"
            )
        z_p = torch.stack([self.zp_lookup[s] for s in sequences]).to(z_c)
        if self._is_training_batch():
            alpha = self._train_alpha(z_c.size(0), z_c.device)
        else:
            alpha = self._eval_alpha(sequences, z_c.device).to(z_c)
        y = alpha * z_p + (1.0 - alpha) * z_c
        return [num_seqs, y.reshape(*lead, -1)]

    # ── objective ─────────────────────────────────────────────────────────

    def forward(self, x, t, y_c, real_masks=None, couple=True):
        return self.model(x, t, y_c, real_masks=real_masks, couple=couple)

    def _mask_canvas(self, realization, idx, batch_size, seq_length):
        """Mask one canvas at a *given* shared diffusion step.

        The path ordering is drawn per canvas, so each domain reveals its
        residues in its own order while sitting at the same step.
        """
        path = trainer_tools.sample_random_path(
            batch_size, seq_length, device=self.device)
        path_mask = trainer_tools.create_mask_at_random_path_index(
            path, idx, batch_size, seq_length)
        real_tokens, _, _ = trainer_tools.create_token_labels(
            self.script_args, realization)
        return real_tokens, trainer_tools.mask_realizations(real_tokens, path_mask)

    def _recon_loss(self, logits, real_tokens, masked, idx, seq_length):
        conditional_prob = OneHotCategorical(logits=logits.permute(0, 2, 1))
        log_prob = trainer_tools.log_prob_of_realization(
            self.script_args, conditional_prob, real_tokens)
        unsampled = trainer_tools.log_prob_of_unsampled_locations(log_prob, masked)
        weighted = trainer_tools.weight_log_prob(unsampled, idx, seq_length)
        return trainer_tools.compute_average_loss_for_batch(weighted)

    def common_step(self, batch, batch_idx, stage):
        num_seqs, y_c = batch[0], batch[1]
        batch_size, num_domains, seq_length = num_seqs.size()
        if num_domains != self.model.num_domains:
            raise ValueError(
                f"batch carries {num_domains} domains but the model was built "
                f"for {self.model.num_domains}"
            )

        # One timestep for the whole assembly; independent paths per canvas.
        idx = trainer_tools.sample_random_index_for_sampling(
            batch_size, seq_length, device=self.device, option="random")

        real_tokens, masked = [], []
        for d in range(num_domains):
            realization = num_seqs[:, d].reshape(batch_size, 1, seq_length).long()
            tokens, masked_tokens = self._mask_canvas(
                realization, idx, batch_size, seq_length)
            real_tokens.append(tokens)
            masked.append(masked_tokens)

        x = torch.stack(masked, dim=1)
        real_masks = torch.stack(
            [tokens != PAD_ID for tokens in real_tokens], dim=1)
        timesteps = idx.view(-1, 1).expand(-1, num_domains)

        logits = self(x, idx.view(-1), y_c, real_masks=real_masks)
        logits = logits.float()

        recon = 0.0
        for d in range(num_domains):
            recon = recon + self._recon_loss(
                logits[:, d], real_tokens[d], masked[d], idx, seq_length)

        loss = recon
        is_val = "val" in stage
        log_kwargs = dict(on_step=not is_val, on_epoch=True, sync_dist=is_val)
        self.log(f"{stage}_recon", recon, prog_bar=True, **log_kwargs)

        if self.expert_prior_lambda > 0:
            penalty = self._prior_penalty(masked, timesteps, y_c)
            loss = loss + self.expert_prior_lambda * penalty
            self.log(f"{stage}_prior", penalty, **log_kwargs)

        self.log(f"{stage}_loss", loss, prog_bar=True, **log_kwargs)
        return {"loss": loss}

    def training_step(self, batch, batch_idx):
        return self.common_step(batch, batch_idx, stage="train")

    def validation_step(self, batch, batch_idx):
        self.common_step(batch, batch_idx, stage="val")

    def on_train_epoch_end(self):
        deltas = self.expert_delta_norms()
        for d, value in deltas.items():
            label = (self.spec.domain_ids[d]
                     if self.spec is not None and d < len(self.spec.domain_ids)
                     else str(d))
            self.log(f"expert_delta_norm/{label}", value, prog_bar=True)

    # ── optimization ──────────────────────────────────────────────────────

    def configure_optimizers(self):
        """AdamW over coupling and (if unfrozen) experts, in separate groups.

        Prior-regularized experts go in a ``weight_decay=0`` group: decay pulls
        them toward zero while the prior pulls them toward their reference, and
        two conflicting priors on one tensor is a silent misconfiguration. The
        pre-flight audit rejects it, so the grouping here is what lets the audit
        pass.
        """
        coupling_params = [p for p in self.model.coupling.parameters()
                           if p.requires_grad]
        expert_params = [p for expert in self.model.experts
                         for p in expert.parameters() if p.requires_grad]

        weight_decay = getattr(self.script_args, "weight_decay", 0.0)
        groups = [{"params": coupling_params, "weight_decay": weight_decay}]
        if expert_params:
            groups.append({
                "params": expert_params,
                "weight_decay": 0.0 if self.expert_prior_lambda > 0 else weight_decay,
            })

        return torch.optim.AdamW(groups, lr=self.script_args.lr)

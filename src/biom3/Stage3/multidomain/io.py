"""Building and loading composed multidomain decoders.

Every consumer of a multidomain checkpoint — training resume, sampling, and any
evaluation — must go through :func:`build_multidomain_from_checkpoint`. The
architecture is read from the spec stored *in* the checkpoint rather than guessed
from a config, and the state dict is then loaded with both directions checked:
a required parameter left unpopulated raises, and so does a saved tensor that
matches no parameter.

That rule exists because the reverse is a silent, total failure. Building a
different coupling class than the one that was trained, loading it with
``strict=False``, and never inspecting ``unexpected_keys`` discards every trained
tensor and leaves a freshly-initialised model behind — with no error and no
warning. Note that such a failure shows up in ``unexpected_keys``, not
``missing_keys``, so checking only for missing parameters does not catch it.
"""

import argparse
import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from typing import Optional, Sequence

import torch

from biom3.Stage3.io import prepare_model_ProteoScribe
from biom3.Stage3.multidomain.coupling import AllPairsCoupling
from biom3.Stage3.multidomain.model import MultiDomainProteoScribe
from biom3.backend.device import setup_logger

logger = setup_logger(__name__)

ALL_PAIRS = "all_pairs"
_COUPLING_TOPOLOGIES = {ALL_PAIRS: AllPairsCoupling}


@dataclass(frozen=True)
class MultiDomainSpec:
    """Everything needed to rebuild a composed decoder, stored with its weights.

    Persisted into the Lightning checkpoint's hyper-parameters and echoed into the
    run's build manifest, so a checkpoint is self-describing and a loader never has
    to infer the architecture from a config that may have moved on.
    """

    num_domains: int
    dim: int
    depth: int
    n_blocks: int
    heads: int
    num_classes: int
    text_emb_dim: int
    image_size: int
    diffusion_steps: int
    domain_ids: Sequence[str] = field(default_factory=tuple)
    coupling_topology: str = ALL_PAIRS
    train_experts: bool = False
    expert_sources: Sequence[str] = field(default_factory=tuple)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["domain_ids"] = list(self.domain_ids)
        data["expert_sources"] = list(self.expert_sources)
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "MultiDomainSpec":
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown MultiDomainSpec fields: {sorted(unknown)}")
        payload = dict(data)
        payload["domain_ids"] = tuple(payload.get("domain_ids", ()))
        payload["expert_sources"] = tuple(payload.get("expert_sources", ()))
        return cls(**payload)

    @classmethod
    def from_args(cls, config_args, num_domains, *, domain_ids=(),
                  coupling_topology=ALL_PAIRS, train_experts=False,
                  expert_sources=()) -> "MultiDomainSpec":
        return cls(
            num_domains=int(num_domains),
            dim=int(config_args.transformer_dim),
            depth=int(config_args.transformer_depth),
            n_blocks=int(config_args.transformer_blocks),
            heads=int(config_args.transformer_heads),
            num_classes=int(config_args.num_classes),
            text_emb_dim=int(config_args.text_emb_dim),
            image_size=int(config_args.image_size),
            diffusion_steps=int(config_args.diffusion_steps),
            domain_ids=tuple(domain_ids),
            coupling_topology=coupling_topology,
            train_experts=bool(train_experts),
            expert_sources=tuple(expert_sources),
        )

    def to_model_args(self, template=None) -> argparse.Namespace:
        """Model args for building one expert, taken from the spec.

        ``template`` supplies the fields the spec does not carry (dropout, local
        attention sizing and so on); its architecture-defining fields are
        overridden by the spec so the spec always wins.
        """
        base = dict(vars(template)) if template is not None else {
            "model_option": "transformer",
            "num_y_class_labels": 6,
            "num_steps": 1,
            "actnorm": False,
            "perm_channel": "none",
            "perm_length": "reverse",
            "input_dp_rate": 0.0,
            "transformer_dropout": 0.1,
            "transformer_reversible": False,
            "transformer_local_heads": 8,
            "transformer_local_size": 128,
        }
        base.update(
            transformer_dim=self.dim,
            transformer_depth=self.depth,
            transformer_blocks=self.n_blocks,
            transformer_heads=self.heads,
            num_classes=self.num_classes,
            text_emb_dim=self.text_emb_dim,
            image_size=self.image_size,
            diffusion_steps=self.diffusion_steps,
        )
        return argparse.Namespace(**base)


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_experts(spec: MultiDomainSpec, expert_weights, *, template_args=None,
                 device=None, expert_sha256=None):
    """Load K experts strictly, one per domain, in N->C order.

    Weight loading goes through :func:`prepare_model_ProteoScribe`, which handles
    raw ``.bin``/``.pt``, Lightning ``.ckpt``, DeepSpeed shard directories and the
    ``axial_pos_emb.weights_`` naming drift, and raises on any residual mismatch.

    ``expert_sha256`` optionally pins each file's digest. That is provenance, kept
    separate from the load: a missing pin never relaxes the strict load.
    """
    expert_weights = list(expert_weights)
    if len(expert_weights) != spec.num_domains:
        raise ValueError(
            f"spec declares {spec.num_domains} domains but {len(expert_weights)} "
            "expert weight paths were given"
        )
    if expert_sha256:
        expert_sha256 = list(expert_sha256)
        if len(expert_sha256) != len(expert_weights):
            raise ValueError(
                f"got {len(expert_sha256)} sha256 pins for "
                f"{len(expert_weights)} experts"
            )

    model_args = spec.to_model_args(template_args)
    experts = []
    for d, path in enumerate(expert_weights):
        if path is None:
            raise ValueError(f"expert {d} has no weights path")
        if expert_sha256:
            expected = expert_sha256[d]
            if expected:
                actual = _sha256_file(path)
                if actual != expected:
                    raise ValueError(
                        f"expert {d} sha256 mismatch for {path}: "
                        f"expected {expected}, got {actual}"
                    )
        logger.info("Loading expert %s/%s from %s", d + 1, spec.num_domains, path)
        experts.append(
            prepare_model_ProteoScribe(
                model_args, path, device=device,
                strict=True, attempt_correction=True,
            )
        )
    return experts


def build_from_spec(spec: MultiDomainSpec, *, experts=None, template_args=None,
                    expert_weights=None, device=None, expert_sha256=None):
    """Construct a composed decoder matching ``spec``."""
    if spec.coupling_topology not in _COUPLING_TOPOLOGIES:
        raise ValueError(
            f"unknown coupling_topology {spec.coupling_topology!r}; "
            f"known: {sorted(_COUPLING_TOPOLOGIES)}"
        )
    if experts is None:
        if expert_weights is None:
            model_args = spec.to_model_args(template_args)
            experts = [
                prepare_model_ProteoScribe(model_args, None, device=device)
                for _ in range(spec.num_domains)
            ]
        else:
            experts = load_experts(
                spec, expert_weights, template_args=template_args,
                device=device, expert_sha256=expert_sha256,
            )
    coupling = _COUPLING_TOPOLOGIES[spec.coupling_topology](
        num_domains=spec.num_domains, n_layers=spec.depth,
        dim=spec.dim, heads=spec.heads,
    )
    model = MultiDomainProteoScribe(experts, coupling)
    if device is not None:
        model = model.to(device)
    return model


def state_dict_fingerprint(state_dict) -> str:
    """Stable digest over key names and shapes.

    Recorded when a checkpoint is written and re-checked when it is read, so a
    topology change fails with a readable diff instead of loading garbage.
    """
    if hasattr(state_dict, "state_dict"):
        state_dict = state_dict.state_dict()
    payload = json.dumps(
        [[key, list(tensor.shape)] for key, tensor in sorted(state_dict.items())],
        separators=(",", ":"),
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def load_composed_state_dict(model, state_dict, *, label="composed checkpoint"):
    """Load a composed state dict, refusing anything that would silently no-op.

    Raises when a parameter or buffer the model needs is absent from the state
    dict, and equally when the state dict carries a tensor the model has no slot
    for. The second check is the one that matters: a state dict trained against a
    different coupling lands entirely in ``unexpected_keys``, which
    ``strict=False`` would discard without a word.
    """
    state_dict = {k: v for k, v in state_dict.items()}
    result = model.load_state_dict(state_dict, strict=False)

    problems = []
    if result.missing_keys:
        shown = result.missing_keys[:8]
        problems.append(
            f"{len(result.missing_keys)} parameter(s) were not populated by the "
            f"state dict, e.g. {shown}"
        )
    if result.unexpected_keys:
        shown = result.unexpected_keys[:8]
        problems.append(
            f"{len(result.unexpected_keys)} saved tensor(s) matched no parameter, "
            f"e.g. {shown} — the model built here does not match the one that was "
            "trained"
        )
    if problems:
        raise ValueError(f"{label}: " + "; ".join(problems))

    logger.info("%s: loaded %s tensors, all matched", label, len(state_dict))
    return model


def extract_composed_state_dict(checkpoint: dict) -> dict:
    """Pull the composed model's tensors out of a Lightning checkpoint.

    Lightning stores the wrapped model under a ``model.`` prefix; raw state dicts
    saved directly from the composed module carry no prefix.
    """
    state_dict = checkpoint.get("state_dict", checkpoint)
    prefix = "model."
    if any(key.startswith(prefix) for key in state_dict):
        return {key[len(prefix):]: value
                for key, value in state_dict.items() if key.startswith(prefix)}
    return dict(state_dict)


def read_spec(checkpoint: dict) -> MultiDomainSpec:
    """Read the multidomain spec a checkpoint was written with."""
    hparams = checkpoint.get("hyper_parameters", {}) or {}
    raw = hparams.get("multidomain_spec")
    if raw is None:
        raise ValueError(
            "checkpoint carries no multidomain_spec — it was not written by the "
            "multidomain trainer, so its architecture cannot be reconstructed"
        )
    return MultiDomainSpec.from_dict(dict(raw))


def build_multidomain_from_checkpoint(checkpoint_path, *, template_args=None,
                                      device=None, expect_fingerprint=None):
    """Rebuild a composed decoder from a checkpoint, architecture and all.

    The single entry point for every consumer of a trained multidomain model.

    Returns:
        tuple of (model, spec)
    """
    if os.path.isdir(checkpoint_path):
        raise ValueError(
            f"{checkpoint_path} is a directory; consolidate a sharded checkpoint "
            "to a single file before loading a composed model"
        )
    checkpoint = torch.load(checkpoint_path, map_location=device or "cpu",
                            weights_only=False)
    spec = read_spec(checkpoint)
    state_dict = extract_composed_state_dict(checkpoint)

    model = build_from_spec(spec, template_args=template_args, device=device)

    stored = checkpoint.get("hyper_parameters", {}).get("multidomain_fingerprint")
    expected = expect_fingerprint or stored
    if expected is not None:
        actual = state_dict_fingerprint(model)
        if actual != expected:
            raise ValueError(
                f"{checkpoint_path}: architecture fingerprint mismatch "
                f"(checkpoint {expected}, model built from spec {actual}); the "
                "stored spec does not describe the stored weights"
            )

    load_composed_state_dict(model, state_dict, label=str(checkpoint_path))
    if device is not None:
        model = model.to(device)
    return model, spec

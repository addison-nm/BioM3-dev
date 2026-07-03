"""LoRA (Low-Rank Adaptation) for ProteoScribe finetuning.

An alternative to block-freezing (`freeze_except_last_n_blocks_and_layers`) for
the generalized finetuning pipeline: freeze the whole base, inject low-rank
adapters on the attention Q/V projections of every transformer block, and
(optionally) unfreeze the ``y_mlp`` z_c-conditioning pathway. Only the adapters
+ y_mlp are trained.

The ProteoScribe attention linears live at
``transformer.transformer_blocks.0.{i}.layers.layers.0.0.fn.{to_q,to_v}`` — all
``nn.Linear(dim, dim, bias=False)`` — so the default target patterns
``('.fn.to_q', '.fn.to_v')`` match them by name.

Downstream tooling (generation, DPO) loads ProteoScribe with ``strict=True`` into
a plain model, which has no LoRA submodules. Use :func:`merge_lora` to fold the
adapters back into the base ``nn.Linear`` weights and unwrap them, yielding a
standard ProteoScribe state_dict again.
"""

import math
from typing import List, Sequence, Tuple

import torch
import torch.nn as nn

from biom3.backend.device import setup_logger

logger = setup_logger(__name__)

# ProteoScribe attention Q/V projections (matched as name substrings).
DEFAULT_TARGET_PATTERNS: Tuple[str, ...] = (".fn.to_q", ".fn.to_v")


class LoRALinear(nn.Module):
    """Low-rank adapter wrapping a frozen ``nn.Linear``.

    ``y = base(x) + scaling * lora_B(lora_A(dropout(x)))`` with ``scaling = alpha/r``.
    ``lora_B`` is zero-initialized so ΔW = 0 at construction (the wrapped model is
    numerically identical to the base until training moves the adapters).
    """

    def __init__(self, base_layer: nn.Linear, r: int = 16, alpha: int = 32,
                 dropout: float = 0.05):
        super().__init__()
        if r <= 0:
            raise ValueError(f"LoRA rank r must be > 0, got {r}")
        self.base_layer = base_layer
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        for p in base_layer.parameters():
            p.requires_grad = False

        self.lora_A = nn.Linear(base_layer.in_features, r, bias=False)
        self.lora_B = nn.Linear(r, base_layer.out_features, bias=False)
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        return self.base_layer(x) + self.scaling * self.lora_B(self.lora_A(self.lora_dropout(x)))

    @torch.no_grad()
    def merged_weight(self) -> torch.Tensor:
        """Return ``W_base + scaling * (W_B @ W_A)`` (shape out×in)."""
        delta = self.scaling * (self.lora_B.weight @ self.lora_A.weight)
        return self.base_layer.weight + delta.to(self.base_layer.weight.dtype)


def _parent_and_attr(model: nn.Module, dotted: str):
    parent = model
    parts = dotted.split(".")
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]


def inject_lora(model: nn.Module, r: int = 16, alpha: int = 32, dropout: float = 0.05,
                target_patterns: Sequence[str] = DEFAULT_TARGET_PATTERNS
                ) -> Tuple[nn.Module, List[str]]:
    """Wrap every ``nn.Linear`` whose qualified name contains a target pattern."""
    targets = [
        (name, mod) for name, mod in model.named_modules()
        if isinstance(mod, nn.Linear) and any(pat in name for pat in target_patterns)
    ]
    injected = []
    for name, mod in targets:
        parent, attr = _parent_and_attr(model, name)
        setattr(parent, attr, LoRALinear(mod, r=r, alpha=alpha, dropout=dropout))
        injected.append(name)
    if not injected:
        raise ValueError(
            f"inject_lora matched no nn.Linear for patterns {tuple(target_patterns)}. "
            "Check the module names of this model."
        )
    return model, injected


def freeze_and_inject_lora(model: nn.Module, r: int = 16, alpha: int = 32,
                           dropout: float = 0.05,
                           target_patterns: Sequence[str] = DEFAULT_TARGET_PATTERNS,
                           unfreeze_y_mlp: bool = True) -> Tuple[nn.Module, dict]:
    """Freeze the whole model, inject LoRA on ``target_patterns``, optionally
    unfreeze ``y_mlp``. Returns ``(model, summary)``."""
    for p in model.parameters():
        p.requires_grad = False

    model, injected = inject_lora(model, r=r, alpha=alpha, dropout=dropout,
                                  target_patterns=target_patterns)

    y_mlp_params = 0
    if unfreeze_y_mlp:
        for name, p in model.named_parameters():
            if "y_mlp" in name:
                p.requires_grad = True
                y_mlp_params += p.numel()

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    lora_params = sum(p.numel() for n, p in model.named_parameters()
                      if "lora_" in n and p.requires_grad)
    summary = {
        "total_params": total,
        "trainable_params": trainable,
        "trainable_fraction": trainable / total if total else 0.0,
        "lora_params": lora_params,
        "y_mlp_params": y_mlp_params,
        "n_lora_layers": len(injected),
        "lora_rank": r,
        "lora_alpha": alpha,
        "lora_dropout": dropout,
        "target_patterns": list(target_patterns),
    }
    return model, summary


@torch.no_grad()
def merge_lora(model: nn.Module) -> Tuple[nn.Module, int]:
    """Fold every :class:`LoRALinear` back into its base ``nn.Linear`` and unwrap it.

    After this the model has plain ``nn.Linear`` modules again and its
    ``state_dict`` matches a fresh ProteoScribe (downstream-loadable with
    ``strict=True``). Returns ``(model, n_merged)``.
    """
    lora_modules = [(name, mod) for name, mod in model.named_modules()
                    if isinstance(mod, LoRALinear)]
    for name, mod in lora_modules:
        parent, attr = _parent_and_attr(model, name)
        base = mod.base_layer
        base.weight.data.copy_(mod.merged_weight())
        for p in base.parameters():
            p.requires_grad = True
        setattr(parent, attr, base)
    return model, len(lora_modules)


def save_lora_weights(model: nn.Module, path: str,
                      target_patterns: Sequence[str] = DEFAULT_TARGET_PATTERNS) -> int:
    """Save only the trainable delta (LoRA adapters + y_mlp) plus the config
    needed to re-inject them onto a base. Small and family-portable.

    Returns the number of saved tensors.
    """
    lora_mods = [m for m in model.modules() if isinstance(m, LoRALinear)]
    if not lora_mods:
        raise ValueError("model has no LoRALinear modules; inject LoRA before saving")
    ref = lora_mods[0]
    drop = ref.lora_dropout.p if isinstance(ref.lora_dropout, nn.Dropout) else 0.0
    state = {n: p.detach().cpu().clone()
             for n, p in model.named_parameters() if p.requires_grad}
    torch.save({
        "lora_state_dict": state,
        "config": {"r": ref.r, "alpha": ref.alpha, "dropout": drop,
                   "target_patterns": list(target_patterns)},
    }, path)
    return len(state)


def load_lora_weights(base_model: nn.Module, path: str, map_location="cpu"
                      ) -> Tuple[nn.Module, int]:
    """Re-inject LoRA onto a fresh base and load saved adapter+y_mlp weights.

    ``base_model`` must be a plain ProteoScribe already holding the SAME base
    weights the adapters were trained on. Returns ``(model, n_loaded)``.
    """
    ckpt = torch.load(path, map_location=map_location)
    cfg = ckpt["config"]
    model, _ = freeze_and_inject_lora(
        base_model, r=cfg["r"], alpha=cfg["alpha"], dropout=cfg["dropout"],
        target_patterns=tuple(cfg["target_patterns"]), unfreeze_y_mlp=True)
    named = dict(model.named_parameters())
    loaded = 0
    with torch.no_grad():
        for k, v in ckpt["lora_state_dict"].items():
            if k in named:
                named[k].copy_(v.to(named[k].device))
                loaded += 1
            else:
                logger.warning("LoRA key %s not found in model; skipping", k)
    return model, loaded


def export_lora_finetuned(model: nn.Module, lora_path: str, merged_path: str,
                          target_patterns: Sequence[str] = DEFAULT_TARGET_PATTERNS):
    """Save BOTH the LoRA adapter delta and the full merged plain state_dict.

    Order matters: the adapters are saved first (they are unwrapped by the
    in-place merge). ``model`` is left merged (plain nn.Linear) afterwards.
    """
    n_lora = save_lora_weights(model, lora_path, target_patterns=target_patterns)
    _, n_merged = merge_lora(model)
    torch.save(model.state_dict(), merged_path)
    logger.info("LoRA export: %d adapter tensors -> %s | merged (%d folded) -> %s",
                n_lora, lora_path, n_merged, merged_path)
    return n_lora, n_merged


def apply_lora_finetuning(PL_model, r: int = 16, alpha: int = 32, dropout: float = 0.05,
                          target_patterns: Sequence[str] = DEFAULT_TARGET_PATTERNS,
                          unfreeze_y_mlp: bool = True):
    """Apply LoRA freezing to a PL finetune wrapper (operates on ``PL_model.model``).

    Mirrors ``freeze_except_last_n_blocks_and_layers`` as a drop-in alternative in
    the finetuning runner. Returns the (mutated) ``PL_model``.
    """
    _, summary = freeze_and_inject_lora(
        PL_model.model, r=r, alpha=alpha, dropout=dropout,
        target_patterns=target_patterns, unfreeze_y_mlp=unfreeze_y_mlp)
    logger.info(
        "LoRA finetuning: r=%d alpha=%d dropout=%.3f | %d adapter layers | "
        "trainable %s/%s (%.2f%%) [lora=%s, y_mlp=%s]",
        r, alpha, dropout, summary["n_lora_layers"],
        f"{summary['trainable_params']:,}", f"{summary['total_params']:,}",
        100 * summary["trainable_fraction"],
        f"{summary['lora_params']:,}", f"{summary['y_mlp_params']:,}",
    )
    return PL_model

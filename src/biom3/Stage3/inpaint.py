"""Template-based in-painting for ProteoScribe (Stage 3) sampling.

In-painting starts diffusion from a partially-filled template instead of an
all-masked tensor. The template is a string mixing fixed amino acids and the
mask symbol ``'-'``; fixed residues are frozen as context and only the masked
positions are generated. ``<START>``/``<END>`` are auto-added (toggleable) and
the tail is auto-filled with ``<PAD>`` up to the model's context window.

This module owns the canonical *runtime* token vocabulary used by Stage 3
sampling so that template parsing and the sampler agree on token ids.

Usage:
    from biom3.Stage3.inpaint import build_template_state, build_sampling_path_row

    state, mask_positions = build_template_state("MKA--GG", seq_len=1024)
    path = build_sampling_path_row(mask_positions, seq_len=1024)
"""

import json

import torch


# Canonical runtime token vocabulary for Stage 3 sampling. Index == token id.
# id 0 = MASK (absorbing state), id 1 = <START>, ids 2-21 = the 20 standard
# amino acids, id 22 = <END>, id 23 = <PAD>, ids 24-28 = rare/ambiguous amino
# acids. The model operates on these integer ids; the strings are the display
# mapping. This matches the (formerly inline) list in run_ProteoScribe_sample.
RUNTIME_TOKENS = [
    '-', '<START>', 'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M',
    'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', '<END>', '<PAD>',
    'X', 'U', 'Z', 'B', 'O',
]

MASK_ID = 0
START_ID = RUNTIME_TOKENS.index('<START>')
END_ID = RUNTIME_TOKENS.index('<END>')
PAD_ID = RUNTIME_TOKENS.index('<PAD>')

# Sentinel written into the sampling path at frozen positions. It must never
# equal any diffusion step index (which run over [0, D)).
_PATH_SENTINEL = -1


def _single_char_vocab(tokens):
    """Map single-character tokens (mask + amino acids) to their token ids.

    Multi-character tokens (``<START>``, ``<END>``, ``<PAD>``) are excluded:
    start/stop are auto-added and padding is auto-filled, so they are not
    written directly in a template string.
    """
    return {tok: i for i, tok in enumerate(tokens) if len(tok) == 1}


def build_template_state(
        template,
        seq_len,
        auto_add_start=True,
        auto_add_stop=True,
        tokens=RUNTIME_TOKENS,
    ):
    """Build the initial diffusion state from an in-painting template.

    Args:
        template: string of amino-acid characters and the mask symbol ``'-'``.
        seq_len: total context length of the assembled state.
        auto_add_start: prepend ``<START>`` before the template.
        auto_add_stop: append ``<END>`` after the template.
        tokens: token vocabulary (index == token id).

    Returns:
        ``(state, mask_positions)`` where ``state`` is a ``LongTensor[seq_len]``
        holding the assembled token ids (mask positions are ``MASK_ID``, the
        tail is ``<PAD>``) and ``mask_positions`` is a ``LongTensor[D]`` of the
        indices to be generated.

    Raises:
        ValueError: on an unknown template character, or if the assembled
            length (including any auto-added start/stop) exceeds ``seq_len``.
    """
    char2id = _single_char_vocab(tokens)
    start_id = tokens.index('<START>')
    end_id = tokens.index('<END>')
    pad_id = tokens.index('<PAD>')

    ids = []
    for ch in template:
        if ch not in char2id:
            raise ValueError(
                f"Invalid template character {ch!r}; valid characters: "
                f"{sorted(char2id)}"
            )
        ids.append(char2id[ch])

    if auto_add_start:
        ids = [start_id] + ids
    if auto_add_stop:
        ids = ids + [end_id]

    if len(ids) > seq_len:
        raise ValueError(
            f"Template length {len(ids)} (including auto-added start/stop) "
            f"exceeds sequence length {seq_len}"
        )

    state = torch.full((seq_len,), pad_id, dtype=torch.long)
    state[:len(ids)] = torch.tensor(ids, dtype=torch.long)
    mask_positions = (state == MASK_ID).nonzero(as_tuple=False).squeeze(-1)
    return state, mask_positions


def build_sampling_path_row(mask_positions, seq_len, generator=None):
    """Build a per-item sampling path for in-painting.

    The returned row has shape ``[seq_len]``; masked positions hold a random
    permutation of ``range(offset, offset + D)`` where ``offset = seq_len - D``
    is the number of frozen residues (the order in which masked positions are
    unmasked), and all frozen positions hold the sentinel ``-1`` so they are
    never selected by the random-unmask loop.

    The offset shifts the path values so they equal the true revealed count at
    each step: the ``offset`` frozen residues are already revealed when
    diffusion starts. Paired with ``extract_time = offset`` in the sampler,
    this keeps the model's time index correct.

    Args:
        mask_positions: 1-D tensor of positions to generate.
        seq_len: total context length.
        generator: optional ``torch.Generator`` for reproducible ordering.
    """
    mask_positions = torch.as_tensor(mask_positions, dtype=torch.long)
    D = mask_positions.numel()
    path = torch.full((seq_len,), _PATH_SENTINEL, dtype=torch.long)
    offset = seq_len - D
    order = torch.randperm(D, generator=generator) + offset
    path[mask_positions] = order
    return path


_INPAINT_ALLOWED_KEYS = {"template", "per_prompt", "auto_add_start", "auto_add_stop"}


def load_inpaint_config(config_path):
    """Load and validate an in-painting config JSON file.

    Allowed keys: ``template`` (str), ``per_prompt`` (dict of prompt-index
    string -> template str), ``auto_add_start`` (bool, default True),
    ``auto_add_stop`` (bool, default True). At least one of ``template`` /
    ``per_prompt`` must be present. Unknown keys raise.
    """
    if config_path is None:
        raise ValueError("--inpaint requires --inpaint_config")
    with open(config_path) as fh:
        cfg = json.load(fh)

    unknown = set(cfg) - _INPAINT_ALLOWED_KEYS
    if unknown:
        raise ValueError(f"inpaint_config has unknown keys: {sorted(unknown)}")
    if "template" not in cfg and "per_prompt" not in cfg:
        raise ValueError(
            "inpaint_config must define 'template' and/or 'per_prompt'"
        )
    if "template" in cfg and not isinstance(cfg["template"], str):
        raise ValueError("inpaint_config 'template' must be a string")
    if "per_prompt" in cfg:
        if not isinstance(cfg["per_prompt"], dict):
            raise ValueError("inpaint_config 'per_prompt' must be an object")
        for key, val in cfg["per_prompt"].items():
            if not isinstance(val, str):
                raise ValueError(
                    f"inpaint_config per_prompt[{key!r}] must be a string"
                )
    for key in ("auto_add_start", "auto_add_stop"):
        if key in cfg and not isinstance(cfg[key], bool):
            raise ValueError(f"inpaint_config '{key}' must be a boolean")

    cfg.setdefault("auto_add_start", True)
    cfg.setdefault("auto_add_stop", True)
    return cfg


def resolve_template(prompt_idx, cfg):
    """Return the template for a prompt: per-prompt override else shared default."""
    per_prompt = cfg.get("per_prompt") or {}
    if str(prompt_idx) in per_prompt:
        return per_prompt[str(prompt_idx)]
    if cfg.get("template") is not None:
        return cfg["template"]
    raise ValueError(
        f"No template for prompt index {prompt_idx} and no shared "
        f"'template' default in inpaint_config"
    )

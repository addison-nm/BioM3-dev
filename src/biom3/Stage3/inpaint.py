"""Template-based in-painting for ProteoScribe (Stage 3) sampling.

In-painting starts diffusion from a partially-filled template instead of an
all-masked tensor. The template is a string mixing fixed amino acids, a *mask*
symbol (positions to generate) and a *pad* symbol (frozen padding). Fixed
residues and pad are frozen as context; only mask positions are generated.
``<START>``/``<END>`` are auto-added (toggleable) and the tail is auto-filled
with ``<PAD>`` up to the model's context window.

Symbols. The mask and pad symbols are configurable. Their defaults mirror the
*training* vocabulary in ``Stage3/preprocess.py`` (``create_num_seqs``), where
``'*'`` is the mask/absorbing token (id 0) and ``'-'`` is the pad token
(id 23). The model only ever sees integer ids; these strings are the
user-facing convention. The *runtime* vocabulary (``RUNTIME_TOKENS``) happens
to spell the same ids differently (``'-'`` for mask, ``'<PAD>'`` for pad), so
template parsing converts the user's chosen symbols to the canonical ids rather
than mapping characters straight through the runtime vocabulary.

Run-length notation. A segment ``(<pattern>:<n>)`` expands to ``pattern``
repeated ``n`` times, e.g. ``(*:50)`` is fifty masks and ``A(GS:3)C`` is
``AGSGSGSC``. Expansion happens before symbol resolution.

This module owns the canonical *runtime* token vocabulary used by Stage 3
sampling so that template parsing and the sampler agree on token ids.

Usage:
    from biom3.Stage3.inpaint import build_template_state, build_sampling_path_row

    # default symbols: '*' = mask, '-' = pad
    state, mask_positions = build_template_state("MKA(*:5)GG", seq_len=1024)
    path = build_sampling_path_row(mask_positions, seq_len=1024)
"""

import json
import re

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

# Default template symbols, mirroring the TRAINING vocabulary in
# Stage3/preprocess.py (create_num_seqs): '*' is the mask/absorbing token and
# '-' is the pad token. Used when a config does not specify mask_symbol /
# pad_symbol. See module docstring for the id-vs-string conversion rationale.
DEFAULT_MASK_SYMBOL = '*'
DEFAULT_PAD_SYMBOL = '-'

# Characters that may never be used as a mask/pad symbol: the run-length
# grammar delimiters, digits, and the special-token brackets.
_RESERVED_SYMBOL_CHARS = set("():<>") | set("0123456789")

# Sentinel written into the sampling path at frozen positions. It must never
# equal any diffusion step index (which run over [0, D)).
_PATH_SENTINEL = -1

_RUNLENGTH_RE = re.compile(r"\(([^():]*):(\d+)\)")


def _expand_runlength(template):
    """Expand ``(<pattern>:<n>)`` run-length segments into a flat string.

    ``pattern`` is any run of non-delimiter characters and is repeated ``n``
    times (``n >= 0``). Expansion is a single non-nested pass. Raises
    ``ValueError`` on an empty pattern or on parentheses that don't form a
    valid group (which would otherwise survive into character mapping).
    """
    def _repl(m):
        pattern, count = m.group(1), int(m.group(2))
        if pattern == "":
            raise ValueError(
                f"Empty run-length pattern in segment {m.group(0)!r}; "
                f"use '(<symbols>:<count>)'"
            )
        return pattern * count

    expanded = _RUNLENGTH_RE.sub(_repl, template)
    if "(" in expanded or ")" in expanded:
        raise ValueError(
            f"Malformed run-length notation in template {template!r}: "
            f"unbalanced or invalid parentheses. Use '(<symbols>:<count>)'."
        )
    return expanded


def _amino_acid_vocab(tokens):
    """Single-character amino-acid tokens mapped to ids.

    Excludes the runtime mask string (``tokens[MASK_ID]``) so the mask/pad
    symbols are resolved by config, not by the runtime vocabulary's spelling.
    Multi-character tokens (``<START>``/``<END>``/``<PAD>``) are excluded too.
    """
    return {
        tok: i for i, tok in enumerate(tokens)
        if len(tok) == 1 and i != MASK_ID
    }


def _validate_symbol(name, symbol, aa_vocab):
    """Validate a single mask/pad symbol against the amino-acid vocabulary."""
    if not isinstance(symbol, str) or len(symbol) != 1:
        raise ValueError(f"{name} must be a single character, got {symbol!r}")
    if symbol in aa_vocab:
        raise ValueError(
            f"{name}={symbol!r} collides with an amino-acid token; "
            f"choose a non-amino-acid symbol"
        )
    if symbol in _RESERVED_SYMBOL_CHARS:
        raise ValueError(
            f"{name}={symbol!r} is reserved (run-length grammar / digits / "
            f"special-token brackets); choose another symbol"
        )


def _resolve_symbol_vocab(tokens, mask_symbol, pad_symbol):
    """Build the char->id map for template parsing.

    Amino acids resolve via the runtime vocabulary; ``mask_symbol`` resolves to
    ``MASK_ID`` and ``pad_symbol`` (if not ``None``) to ``PAD_ID``. Raises on an
    invalid symbol or a mask/pad clash.
    """
    aa_vocab = _amino_acid_vocab(tokens)
    _validate_symbol("mask_symbol", mask_symbol, aa_vocab)
    char2id = dict(aa_vocab)
    char2id[mask_symbol] = MASK_ID
    if pad_symbol is not None:
        _validate_symbol("pad_symbol", pad_symbol, aa_vocab)
        if pad_symbol == mask_symbol:
            raise ValueError(
                f"mask_symbol and pad_symbol must differ (both {mask_symbol!r})"
            )
        char2id[pad_symbol] = PAD_ID
    return char2id


def build_template_state(
        template,
        seq_len,
        auto_add_start=True,
        auto_add_stop=True,
        tokens=RUNTIME_TOKENS,
        mask_symbol=DEFAULT_MASK_SYMBOL,
        pad_symbol=DEFAULT_PAD_SYMBOL,
    ):
    """Build the initial diffusion state from an in-painting template.

    Args:
        template: string of amino-acid characters, the ``mask_symbol`` and
            (optionally) the ``pad_symbol``. Run-length segments
            ``(<pattern>:<n>)`` are expanded first.
        seq_len: total context length of the assembled state.
        auto_add_start: prepend ``<START>`` before the template.
        auto_add_stop: append ``<END>`` after the template.
        tokens: token vocabulary (index == token id).
        mask_symbol: character denoting a position to generate (-> ``MASK_ID``).
        pad_symbol: character denoting frozen padding (-> ``PAD_ID``); ``None``
            disables in-template padding (the tail is still auto-padded).

    Returns:
        ``(state, mask_positions)`` where ``state`` is a ``LongTensor[seq_len]``
        holding the assembled token ids (mask positions are ``MASK_ID``, the
        tail is ``<PAD>``) and ``mask_positions`` is a ``LongTensor[D]`` of the
        indices to be generated.

    Raises:
        ValueError: on an invalid symbol, malformed run-length notation, an
            unknown template character, or if the assembled length (including
            any auto-added start/stop) exceeds ``seq_len``.
    """
    char2id = _resolve_symbol_vocab(tokens, mask_symbol, pad_symbol)
    start_id = tokens.index('<START>')
    end_id = tokens.index('<END>')
    pad_id = tokens.index('<PAD>')

    ids = []
    for ch in _expand_runlength(template):
        if ch not in char2id:
            raise ValueError(
                f"Invalid template character {ch!r}; valid characters: "
                f"mask={mask_symbol!r}, pad={pad_symbol!r}, amino acids "
                f"{sorted(_amino_acid_vocab(tokens))}"
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


_INPAINT_ALLOWED_KEYS = {
    "template", "per_prompt", "auto_add_start", "auto_add_stop",
    "mask_symbol", "pad_symbol",
}


def load_inpaint_config(config_path):
    """Load and validate an in-painting config JSON file.

    Allowed keys: ``template`` (str), ``per_prompt`` (dict of prompt-index
    string -> template str), ``auto_add_start`` (bool, default True),
    ``auto_add_stop`` (bool, default True), ``mask_symbol`` (single char,
    default ``'*'``), ``pad_symbol`` (single char or ``null``, default ``'-'``).
    At least one of ``template`` / ``per_prompt`` must be present. Unknown keys
    raise. Symbols are validated against the amino-acid vocabulary so a clash is
    caught at config-load time rather than during template parsing.
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
    cfg.setdefault("mask_symbol", DEFAULT_MASK_SYMBOL)
    cfg.setdefault("pad_symbol", DEFAULT_PAD_SYMBOL)

    # Validate symbols up front (mask/pad clash, amino-acid collision, reserved
    # characters). Mirrors what build_template_state would enforce per prompt.
    _resolve_symbol_vocab(RUNTIME_TOKENS, cfg["mask_symbol"], cfg["pad_symbol"])
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

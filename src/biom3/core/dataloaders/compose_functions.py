"""Composition functions: cleaned record -> a single composed output value.

A *compose function* has the signature ``fn(obj, args, rng) -> value`` where

    obj   the full cleaned record (a dict), e.g.
          ``{"sequence": "MSII...", "fields": {"lineage": "...", ...}}``
    args  a JSON-serializable dict of options for this composition
    rng   a ``random.Random`` (or the ``random`` module) used for any
          stochastic step, so callers control reproducibility

Functions are *pure* and *standalone*: they take a record and return a value,
independent of any Dataset. That lets the same function drive both live,
per-epoch augmentation (inside :class:`GeneralizedRecordDataset`) and an
offline pass that expands records into a finite variant set for z_c precompute.

The pre-built functions are registered by name via :func:`register_compose`
and looked up with :func:`get_compose_function`, so a schema can reference one
with a ``(name, args)`` pair instead of importing a Python callable.

Note on dropout semantics: ``dropout_rates``/``default_dropout`` are *removal*
probabilities (the complement of a *retention* probability) — an item with a
removal rate of 0.5 is kept half the time.
"""

import random

_COMPOSE_REGISTRY = {}


def register_compose(name):
    """Decorator registering a compose function under ``name``."""

    def _decorator(fn):
        if name in _COMPOSE_REGISTRY:
            raise ValueError(f"compose function {name!r} is already registered")
        _COMPOSE_REGISTRY[name] = fn
        return fn

    return _decorator


def get_compose_function(name):
    """Look up a registered compose function by name."""
    try:
        return _COMPOSE_REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"unknown compose function {name!r}; "
            f"registered: {list_compose_functions()}"
        ) from None


def list_compose_functions():
    """Return the sorted names of all registered compose functions."""
    return sorted(_COMPOSE_REGISTRY)


#########################################################
# Primitive ops (reusable building blocks for custom    #
# user-written compose functions)                       #
#########################################################

def fields_to_items(fields):
    """Normalize a fields mapping to an ordered list of ``(key, value)`` pairs.

    Accepts a dict (insertion order preserved) or any iterable of pairs.
    """
    if isinstance(fields, dict):
        return list(fields.items())
    return [(key, value) for key, value in fields]


def normalize_items(items, list_separator=", "):
    """Coerce item values to non-empty strings, dropping empties.

    List/tuple values are joined with ``list_separator`` (skipping empty
    elements); scalar values pass through ``str``. Items whose value is ``None``
    or empty after coercion are dropped, so a missing/blank field never yields a
    bare ``KEY:`` fragment.
    """
    normalized = []
    for key, value in items:
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            value = list_separator.join(
                str(v) for v in value if v is not None and str(v) != ""
            )
        else:
            value = str(value)
        if value:
            normalized.append((key, value))
    return normalized


def dropout_items(items, rates=None, default=0.0, rng=random):
    """Drop ``(key, value)`` items by per-key *removal* probability.

    An item is kept when ``rng.random() >= rate``; ``rates`` maps a key to its
    removal probability and ``default`` applies to keys absent from ``rates``.
    """
    rates = rates or {}
    kept = []
    for key, value in items:
        drop_p = rates.get(key, default)
        if rng.random() >= drop_p:
            kept.append((key, value))
    return kept


def shuffle_items(items, rng=random):
    """Return a shuffled copy of ``items`` (input list is not mutated)."""
    items = list(items)
    rng.shuffle(items)
    return items


_KEY_TRANSFORMS = {
    "none": lambda key: key,
    "upper": lambda key: key.upper(),
    "lower": lambda key: key.lower(),
    "title": lambda key: key.title(),
    "upper_spaced": lambda key: key.replace("_", " ").upper(),
    "title_spaced": lambda key: key.replace("_", " ").title(),
}


def _resolve_key_transform(key_transform):
    if callable(key_transform):
        return key_transform
    try:
        return _KEY_TRANSFORMS[key_transform]
    except KeyError:
        raise ValueError(
            f"unknown key_transform {key_transform!r}; "
            f"choose from {sorted(_KEY_TRANSFORMS)} or pass a callable"
        ) from None


def add_labels(items, label_format="{key}: {value}", key_transform="upper"):
    """Render ``(key, value)`` items into labeled strings.

    Each item becomes ``label_format.format(key=transform(key), value=value)``.
    ``key_transform`` is a name in ``{none, upper, lower, title}`` or a callable.
    """
    transform = _resolve_key_transform(key_transform)
    return [
        label_format.format(key=transform(key), value=value)
        for key, value in items
    ]


def concatenate(values, separator=". ", trailing_period=True):
    """Join non-empty string values, optionally ensuring a trailing period.

    When the separator is period-based (e.g. ``". "``), a fragment that already
    ends in a period has it stripped before joining so the separator supplies the
    single period — otherwise a value like ``"...protein."`` followed by another
    field would yield a doubled ``"protein.. NEXT"``.
    """
    sep_is_period = separator.lstrip().startswith(".")
    cleaned = []
    for value in values:
        if not value:
            continue
        value = value.rstrip()
        if sep_is_period and value.endswith("."):
            value = value[:-1].rstrip()
        if value:
            cleaned.append(value)
    text = separator.join(cleaned)
    if trailing_period and text and not text.endswith("."):
        text += "."
    return text


#########################################################
# Pre-built compose functions                           #
#########################################################

@register_compose("fields_to_caption")
def fields_to_caption(obj, args=None, rng=random):
    """Compose a text caption from a record's field fragments.

    Pipeline: list the items under ``fields_key``, coerce values to non-empty
    strings (joining lists, dropping blanks), apply per-key dropout, optionally
    shuffle, optionally prepend a field label, then concatenate into one caption.

    args keys (all optional):
        fields_key       record key holding the field mapping (default "fields")
        list_separator   join string for list-valued fields (default ", ")
        dropout_rates    {field_key: removal_prob}
        default_dropout  removal prob for keys absent from dropout_rates (0.0)
        shuffle          shuffle surviving fields before joining (False)
        add_label        prepend a field label to each value (True)
        label_format     label template (default "{key}: {value}")
        key_transform    label key casing: none/upper/lower/title/upper_spaced/
                         title_spaced or a callable
        separator        fragment separator (default ". ")
        trailing_period  ensure a non-empty caption ends with "." (True)
    """
    args = args or {}
    fields = obj[args.get("fields_key", "fields")]
    items = fields_to_items(fields)
    items = normalize_items(items, list_separator=args.get("list_separator", ", "))
    items = dropout_items(
        items,
        rates=args.get("dropout_rates"),
        default=args.get("default_dropout", 0.0),
        rng=rng,
    )
    if args.get("shuffle", False):
        items = shuffle_items(items, rng=rng)
    if args.get("add_label", True):
        values = add_labels(
            items,
            label_format=args.get("label_format", "{key}: {value}"),
            key_transform=args.get("key_transform", "upper"),
        )
    else:
        values = [value for _, value in items]
    return concatenate(
        values,
        separator=args.get("separator", ". "),
        trailing_period=args.get("trailing_period", True),
    )


def reduce_list_field(items, policy, default_max_item_chars=None):
    """Reduce a field's list of raw items to a single string per ``policy``.

    ``items`` is a list of strings (one per raw annotation comment / term).
    ``policy`` keys (all optional):
        keep: "first" (default) | "all" | "all_but_last" | int (first N items)
        join: separator joining kept items (default ", ")
        max_item_chars: per-field override of the char cutoff (only for "first")

    For ``keep == "first"`` (the comment-field default), items longer than the
    char cutoff are dropped first, then the first survivor is returned; if every
    item exceeds the cutoff the field is dropped (returns ""). All character
    comparisons are on raw ``len(str)`` — no tokenization — so this is cheap
    enough to run per access. Other ``keep`` modes are not char-filtered.
    """
    items = [str(x).strip() for x in items if x is not None and str(x).strip()]
    if not items:
        return ""
    keep = policy.get("keep", "first")
    join = policy.get("join", ", ")
    if keep == "all":
        return join.join(items)
    if keep == "all_but_last":
        return join.join(items[:-1] if len(items) > 1 else items)
    if isinstance(keep, int):
        return join.join(items[:keep])
    # keep == "first": drop over-long items, take the first survivor.
    cap = policy.get("max_item_chars", default_max_item_chars)
    if cap is not None:
        items = [x for x in items if len(x) <= cap]
    return items[0] if items else ""


@register_compose("list_fields_to_caption")
def list_fields_to_caption(obj, args=None, rng=random):
    """Compose a caption from raw list-valued fields with per-field selection.

    Unlike :func:`fields_to_caption` (which joins the already-cleaned strings in
    ``caption_fields``), this composes from the raw ``fields`` lists, reducing
    each field to a single string via :func:`reduce_list_field` (default: drop
    items longer than ``max_item_chars`` and take the first survivor). Fields can
    be dropped for specific sources (e.g. gene_ontology for swissprot). The
    reduced fields then get the usual per-key dropout, optional shuffle,
    label-adding, and concatenation.

    args keys (all optional unless noted):
        fields_key       record key holding {field: [items]} (default "fields")
        source_key       record key holding the data source (default "source")
        max_item_chars   global per-item char cutoff for the "first" policy
        field_policies   {field: policy} overrides (see reduce_list_field); a
                         policy may also set ``exclude_sources`` (list) to drop
                         that field for the given sources
        dropout_rates    {field: removal_prob}
        default_dropout  removal prob for keys absent from dropout_rates (0.0)
        shuffle, add_label, label_format, key_transform, separator,
        trailing_period  -- as in fields_to_caption
    """
    args = args or {}
    fields = obj[args.get("fields_key", "fields")]
    source = obj.get(args.get("source_key", "source"))
    policies = args.get("field_policies", {})
    default_cap = args.get("max_item_chars")

    items = []
    for key, value in fields_to_items(fields):
        policy = policies.get(key, {})
        if source is not None and source in policy.get("exclude_sources", ()):
            continue
        value = value if isinstance(value, list) else [value]
        reduced = reduce_list_field(value, policy, default_cap)
        if reduced:
            items.append((key, reduced))

    items = dropout_items(
        items,
        rates=args.get("dropout_rates"),
        default=args.get("default_dropout", 0.0),
        rng=rng,
    )
    if args.get("shuffle", False):
        items = shuffle_items(items, rng=rng)
    if args.get("add_label", True):
        values = add_labels(
            items,
            label_format=args.get("label_format", "{key}: {value}"),
            key_transform=args.get("key_transform", "upper"),
        )
    else:
        values = [value for _, value in items]
    return concatenate(
        values,
        separator=args.get("separator", ". "),
        trailing_period=args.get("trailing_period", True),
    )

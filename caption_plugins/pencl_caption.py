"""External compose-function plugin: PenCL-faithful caption composition.

This file lives *outside* the biom3 package. It is loaded at finetune time via
the ``compose_plugins`` config key (Option A), which imports it before the
record_schema is resolved so the ``@register_compose`` below populates the
shared registry. The schema then references it with ``"compose": "pencl_caption"``.

Why a plugin rather than the built-in ``list_fields_to_caption``: reproducing
the exact caption distribution PenCL/Facilitator were trained on needs two
value/label-level transforms the mechanical composer deliberately does not do:

  1. LINEAGE rendered as ``The organism lineage is <comma-joined ranks>`` (the
     jsonl stores a raw ``;``/``,``-delimited lineage; the "cellular organisms"
     root is dropped and the delimiter normalized to ", ").
  2. The Pfam ``family_name`` / ``family_description`` pair collapsed into a
     single ``FAMILY NAMES: Family names are <names>`` field — matching PenCL's
     vocabulary, which never contained ``FAMILY NAME`` (singular),
     ``FAMILY DESCRIPTION`` or ``GENE ONTOLOGY``.

Fields are emitted in a fixed canonical order (PROTEIN NAME first), and only
keys in CANONICAL_ORDER are kept — so out-of-vocabulary labels are dropped by
construction. Everything else (per-key dropout, label casing, concatenation) is
delegated to the shared primitives so behaviour stays consistent with the rest
of the pipeline.
"""

import random
import re

from biom3.core.dataloaders.compose_functions import (
    add_labels,
    concatenate,
    dropout_items,
    fields_to_items,
    normalize_items,
    register_compose,
    shuffle_items,
)

# Canonical field order matching PenCL's SwissProt/Pfam keyword captions. Acts as
# a whitelist: keys absent here (gene_ontology, family_name, family_description)
# are dropped, keeping the emitted label vocabulary in-distribution.
CANONICAL_ORDER = [
    "protein_name",
    "function",
    "catalytic_activity",
    "cofactor",
    "activity_regulation",
    "biophysicochemical_properties",
    "pathway",
    "subunit",
    "subcellular_location",
    "tissue_specificity",
    "induction",
    "developmental_stage",
    "ptm",
    "domain",
    "miscellaneous",
    "similarity",
    "lineage",
    "sh3_paralog_name",
    "paralog_function",
    "family_names",
]


def _lineage_to_natural(value):
    parts = [p.strip() for p in re.split(r"[;,]", value) if p.strip()]
    if parts and parts[0].lower() == "cellular organisms":
        parts = parts[1:]
    if not parts:
        return ""
    return "The organism lineage is " + ", ".join(parts)


def _family_names_value(fields):
    names = fields.get("family_names") or fields.get("family_name")
    if not names:
        return ""
    return "Family names are " + names


@register_compose("pencl_caption")
def pencl_caption(obj, args=None, rng=random):
    """Compose a PenCL-distribution caption from a cleaned record's fields.

    Reuses the same dropout / label / concatenate primitives as the built-in
    composers; adds a lineage value-transform, family-field fusion, and a fixed
    canonical field order. Honors the usual args: ``fields_key``, ``dropout_rates``,
    ``default_dropout``, ``shuffle``, ``label_format``, ``key_transform``,
    ``separator``, ``trailing_period``, ``list_separator``.
    """
    args = args or {}
    raw_items = fields_to_items(obj[args.get("fields_key", "fields")])
    norm = dict(normalize_items(
        raw_items, list_separator=args.get("list_separator", ", ")))

    built = dict(norm)
    if "lineage" in built:
        built["lineage"] = _lineage_to_natural(built["lineage"])
    fam = _family_names_value(norm)
    if fam:
        built["family_names"] = fam

    ordered = [(k, built[k]) for k in CANONICAL_ORDER if built.get(k)]
    ordered = dropout_items(
        ordered,
        rates=args.get("dropout_rates"),
        default=args.get("default_dropout", 0.0),
        rng=rng,
    )
    if args.get("shuffle", False):
        ordered = shuffle_items(ordered, rng=rng)
    values = add_labels(
        ordered,
        label_format=args.get("label_format", "{key}: {value}"),
        key_transform=args.get("key_transform", "upper_spaced"),
    )
    return concatenate(
        values,
        separator=args.get("separator", ". "),
        trailing_period=args.get("trailing_period", True),
    )

"""Configurable caption composition for dataset building.

Provides CaptionSpec to control which annotation fields are included,
their labels, ordering, formatting, and post-processing (PubMed/evidence
tag removal). This allows different dataset recipes to produce different
caption formats without modifying library code.

Usage:
    from biom3.dbio.caption import CaptionSpec, compose_row_caption

    spec = CaptionSpec(
        fields=[("PROTEIN NAME", "annot_protein_name"), ("FUNCTION", "annot_function")],
        strip_pubmed=True,
    )
    caption = compose_row_caption(annotations, spec)
"""

import re
from dataclasses import dataclass, field

from biom3.dbio.enrich import ANNOTATION_FIELDS

_PUBMED_RE = re.compile(r"\s*\(PubMed:\d+(?:,\s*PubMed:\d+)*\)")
_EVIDENCE_RE = re.compile(r"\s*\{ECO:\d+.*?\}")
_MULTI_DOT_RE = re.compile(r"\.(\s*\.)+")
_MULTI_SPACE_RE = re.compile(r"  +")


def strip_pubmed_refs(text):
    """Remove (PubMed:NNNNN) references and clean up resulting artifacts."""
    text = _PUBMED_RE.sub("", text)
    text = _MULTI_DOT_RE.sub(".", text)
    text = _MULTI_SPACE_RE.sub(" ", text).strip()
    return text


def strip_evidence_tags(text):
    """Remove {ECO:...} evidence tags."""
    return _EVIDENCE_RE.sub("", text).strip()


@dataclass
class CaptionSpec:
    """Configures how text captions are assembled from annotation dicts.

    Attributes:
        fields: list of (label, annotation_key) tuples defining which
            fields to include and in what order.
        field_template: format string with {label} and {value} placeholders.
        separator: string inserted between fields.
        strip_pubmed: whether to remove PubMed references from values.
        strip_evidence: whether to remove {ECO:...} evidence tags.
        family_names_label: if set, a trailing field is appended with Pfam
            family names. Set to None to omit.
        family_names_template: format string with {names} placeholder for
            the comma-separated family names.
        trailing_period: whether to ensure the caption ends with a period.
    """
    fields: list = field(default_factory=lambda: list(ANNOTATION_FIELDS))
    field_template: str = "{label}: {value}"
    separator: str = ". "
    strip_pubmed: bool = True
    strip_evidence: bool = True
    family_names_label: str | None = None
    family_names_template: str = "Family names are {names}"
    trailing_period: bool = True


# Ordered list of NCBI taxonomic ranks from broadest to most specific.
TAXONOMY_RANKS = [
    "superkingdom", "kingdom", "phylum", "class",
    "order", "family", "genus", "species",
]


def build_lineage_string(lineage_dict, ranks=None):
    """Build a lineage string from selected taxonomic ranks.

    Args:
        lineage_dict: dict from TaxonomyTree.get_lineage(tax_id),
            with keys like 'superkingdom', 'phylum', 'class', etc.
        ranks: list of rank names to include, e.g. ["superkingdom", "phylum"].
            If None, includes all non-empty ranks in TAXONOMY_RANKS order.

    Returns:
        Formatted string like "The organism lineage is Bacteria, Pseudomonadota"
        or None if no ranks have values.
    """
    if ranks is None:
        ranks = TAXONOMY_RANKS

    parts = [lineage_dict.get(r, "") for r in ranks]
    parts = [p for p in parts if p]
    if not parts:
        return None
    return "The organism lineage is " + ", ".join(parts)


def compose_row_caption(annotations, spec, pfam_family_names=None):
    """Compose a single caption string from an annotation dict.

    Args:
        annotations: dict with annotation keys (e.g. annot_protein_name).
        spec: CaptionSpec controlling format and field selection.
        pfam_family_names: optional list of Pfam family name strings.

    Returns:
        Assembled caption string.
    """
    parts = []
    for label, key in spec.fields:
        val = annotations.get(key)
        if not val:
            continue

        if spec.strip_evidence:
            val = strip_evidence_tags(val)
        if spec.strip_pubmed:
            val = strip_pubmed_refs(val)

        val = val.rstrip(".")
        if val:
            parts.append(spec.field_template.format(label=label, value=val))

    if spec.family_names_label and pfam_family_names:
        names_str = ", ".join(pfam_family_names)
        family_val = spec.family_names_template.format(names=names_str)
        parts.append(spec.field_template.format(
            label=spec.family_names_label, value=family_val,
        ))

    caption = spec.separator.join(parts)
    if spec.trailing_period and caption and not caption.endswith("."):
        caption += "."
    return caption

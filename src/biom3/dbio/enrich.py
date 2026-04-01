"""Caption enrichment from UniProt annotations and local NCBI taxonomy.

Two-step workflow:
1. enrich_dataframe() populates individual annotation columns (annot_protein_name,
   annot_function, annot_lineage, etc.) from UniProt API and/or local NCBI taxonomy.
2. compose_caption() assembles [final]text_caption from those columns using the
   BioM3 ALL-CAPS field label format.

This separation keeps the raw annotation data inspectable and the caption format
customizable.
"""

import pandas as pd

from biom3.backend.device import setup_logger

logger = setup_logger(__name__)

# Canonical field ordering for caption composition, matching the BioM3 paper.
# Each entry is (LABEL, column_name) where LABEL is the ALL-CAPS prefix used in
# the composed caption and column_name is the DataFrame column holding the value.
ANNOTATION_FIELDS = [
    ("FAMILY NAME",                "annot_family_name"),
    ("FAMILY DESCRIPTION",         "annot_family_description"),
    ("PROTEIN NAME",               "annot_protein_name"),
    ("FUNCTION",                   "annot_function"),
    ("CATALYTIC ACTIVITY",         "annot_catalytic_activity"),
    ("COFACTOR",                   "annot_cofactor"),
    ("ACTIVITY REGULATION",        "annot_activity_regulation"),
    ("BIOPHYSICOCHEMICAL PROPERTIES", "annot_biophysicochemical_properties"),
    ("PATHWAY",                    "annot_pathway"),
    ("SUBUNIT",                    "annot_subunit"),
    ("SUBCELLULAR LOCATION",       "annot_subcellular_location"),
    ("PTM",                        "annot_ptm"),
    ("SIMILARITY",                 "annot_similarity"),
    ("DOMAIN",                     "annot_domain"),
    ("MISCELLANEOUS",              "annot_miscellaneous"),
    ("INDUCTION",                  "annot_induction"),
    ("GENE ONTOLOGY",              "annot_gene_ontology"),
    ("LINEAGE",                    "annot_lineage"),
]

ANNOTATION_COLUMNS = [col for _, col in ANNOTATION_FIELDS]


# ---------------------------------------------------------------------------
# UniProt JSON parsers
# ---------------------------------------------------------------------------

def parse_protein_name(entry):
    pd_field = entry.get("proteinDescription", {})
    rec = pd_field.get("recommendedName")
    if rec:
        val = rec.get("fullName", {}).get("value")
        if val:
            return val
    subs = pd_field.get("submissionNames", [])
    if subs:
        val = subs[0].get("fullName", {}).get("value")
        if val:
            return val
    return None


def parse_gene_ontology(entry):
    go_terms = []
    for xref in entry.get("uniProtKBCrossReferences", []):
        if xref.get("database") != "GO":
            continue
        for prop in xref.get("properties", []):
            if prop.get("key") == "GoTerm":
                val = prop.get("value", "")
                if len(val) > 2 and val[1] == ":":
                    val = val[2:]
                go_terms.append(val)
    return ", ".join(go_terms) if go_terms else ""


def parse_lineage(entry):
    lineage = entry.get("organism", {}).get("lineage", [])
    if not lineage:
        return None
    return "The organism lineage is " + ", ".join(lineage)


def parse_texts_comment(entry, comment_type):
    for comment in entry.get("comments", []):
        if comment.get("commentType") == comment_type:
            texts = comment.get("texts", [])
            if texts:
                return texts[0].get("value")
    return None


def parse_catalytic_activity(entry):
    parts = []
    for comment in entry.get("comments", []):
        if comment.get("commentType") == "CATALYTIC ACTIVITY":
            reaction = comment.get("reaction", {})
            name = reaction.get("name")
            if name:
                parts.append(f"Reaction={name}")
    return ". ".join(parts) if parts else None


def parse_cofactor(entry):
    for comment in entry.get("comments", []):
        if comment.get("commentType") == "COFACTOR":
            cofactors = comment.get("cofactors", [])
            names = [c.get("name") for c in cofactors if c.get("name")]
            if names:
                return ", ".join(names)
    return None


def parse_subcellular_location(entry):
    for comment in entry.get("comments", []):
        if comment.get("commentType") == "SUBCELLULAR LOCATION":
            locs = comment.get("subcellularLocations", [])
            values = []
            for loc in locs:
                val = loc.get("location", {}).get("value")
                if val:
                    values.append(val)
            if values:
                return ", ".join(values)
    return None


def extract_annotations(entry):
    """Parse a full UniProt JSON entry into a dict of column_name -> text."""
    annotations = {}

    val = parse_protein_name(entry)
    if val:
        annotations["annot_protein_name"] = val

    val = parse_texts_comment(entry, "FUNCTION")
    if val:
        annotations["annot_function"] = val

    val = parse_catalytic_activity(entry)
    if val:
        annotations["annot_catalytic_activity"] = val

    val = parse_cofactor(entry)
    if val:
        annotations["annot_cofactor"] = val

    val = parse_texts_comment(entry, "ACTIVITY REGULATION")
    if val:
        annotations["annot_activity_regulation"] = val

    val = parse_texts_comment(entry, "BIOPHYSICOCHEMICAL PROPERTIES")
    if val:
        annotations["annot_biophysicochemical_properties"] = val

    val = parse_texts_comment(entry, "PATHWAY")
    if val:
        annotations["annot_pathway"] = val

    val = parse_texts_comment(entry, "SUBUNIT")
    if val:
        annotations["annot_subunit"] = val

    val = parse_subcellular_location(entry)
    if val:
        annotations["annot_subcellular_location"] = val

    val = parse_texts_comment(entry, "PTM")
    if val:
        annotations["annot_ptm"] = val

    val = parse_texts_comment(entry, "SIMILARITY")
    if val:
        annotations["annot_similarity"] = val

    val = parse_texts_comment(entry, "DOMAIN")
    if val:
        annotations["annot_domain"] = val

    val = parse_texts_comment(entry, "MISCELLANEOUS")
    if val:
        annotations["annot_miscellaneous"] = val

    val = parse_texts_comment(entry, "INDUCTION")
    if val:
        annotations["annot_induction"] = val

    go = parse_gene_ontology(entry)
    if go:
        annotations["annot_gene_ontology"] = go

    val = parse_lineage(entry)
    if val:
        annotations["annot_lineage"] = val

    return annotations


# ---------------------------------------------------------------------------
# DataFrame enrichment (Step 1: populate annotation columns)
# ---------------------------------------------------------------------------

def enrich_dataframe(df, local_annotations=None, uniprot_data=None,
                     taxonomy_tree=None, accession_taxid_map=None):
    """Populate individual annotation columns from local .dat, UniProt API,
    and/or NCBI taxonomy.

    Adds columns named annot_family_name, annot_family_description,
    annot_protein_name, annot_function, annot_lineage, etc. Each column
    holds the raw annotation text (or NaN if unavailable).

    Does NOT compose [final]text_caption — call compose_caption() for that.

    Args:
        df: DataFrame with primary_Accession column. Optionally family_name
            and family_description columns (for Pfam-sourced rows).
        local_annotations: optional dict mapping accession -> annotation dict
            (annot_* keys), as returned by SwissProtDatParser.parse().
        uniprot_data: optional dict mapping accession -> UniProt JSON entry.
        taxonomy_tree: optional TaxonomyTree instance.
        accession_taxid_map: optional dict mapping accession -> tax_id (int).

    Returns:
        DataFrame with annotation columns added.
    """
    df = df.copy()

    # Initialize annotation columns with NaN
    for col in ANNOTATION_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA

    # Copy family-level columns if present (Pfam rows)
    if "family_name" in df.columns:
        df["annot_family_name"] = df["family_name"]
    if "family_description" in df.columns:
        df["annot_family_description"] = df["family_description"]

    enriched_count = 0

    # Local .dat annotations (preferred — no API needed)
    if local_annotations:
        for idx, row in df.iterrows():
            acc = str(row.get("primary_Accession", ""))
            annots = local_annotations.get(acc)
            if annots:
                for col, val in annots.items():
                    df.at[idx, col] = val
                enriched_count += 1

    # UniProt API annotations (fallback when --use-api is set)
    if uniprot_data:
        for idx, row in df.iterrows():
            acc = str(row.get("primary_Accession", ""))
            entry = uniprot_data.get(acc)
            if entry:
                annotations = extract_annotations(entry)
                for col, val in annotations.items():
                    df.at[idx, col] = val
                enriched_count += 1

    if taxonomy_tree and accession_taxid_map:
        for idx, row in df.iterrows():
            acc = str(row.get("primary_Accession", ""))
            tax_id = accession_taxid_map.get(acc)
            if tax_id is not None:
                lineage_str = taxonomy_tree.get_lineage_string(tax_id)
                if lineage_str:
                    df.at[idx, "annot_lineage"] = lineage_str
                    if not (uniprot_data or local_annotations):
                        enriched_count += 1

    logger.info("Enriched %s/%s rows with annotation columns",
                f"{enriched_count:,}", f"{len(df):,}")
    return df


# ---------------------------------------------------------------------------
# Caption composition (Step 2: assemble [final]text_caption from columns)
# ---------------------------------------------------------------------------

def compose_caption(df, fields=None):
    """Assemble [final]text_caption from annotation columns.

    Concatenates non-empty annotation columns using the BioM3 ALL-CAPS
    field label format: "LABEL: value. LABEL: value. ..."

    Args:
        df: DataFrame with annotation columns (annot_*).
        fields: optional list of (label, column_name) tuples specifying
            which fields to include and in what order. Defaults to
            ANNOTATION_FIELDS (all 18 fields in canonical order).

    Returns:
        DataFrame with [final]text_caption column added/replaced.
    """
    if fields is None:
        fields = ANNOTATION_FIELDS

    captions = []
    for _, row in df.iterrows():
        parts = []
        for label, col in fields:
            val = row.get(col)
            if pd.notna(val) and str(val).strip():
                parts.append(f"{label}: {val}.")
        captions.append(" ".join(parts) if parts else "")

    df = df.copy()
    df["[final]text_caption"] = captions
    return df

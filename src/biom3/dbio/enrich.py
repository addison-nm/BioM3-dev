"""Caption enrichment from UniProt annotations and local NCBI taxonomy.

Two-step workflow:
1. enrich_dataframe() populates individual annotation columns (annot_protein_name,
   annot_function, annot_lineage, etc.) from UniProt API and/or local NCBI taxonomy.
   Optionally joins ExPASy, BRENDA, and SMART source CSVs to add EC-based and
   domain-based annotations.
2. compose_caption() assembles [final]text_caption from those columns using the
   BioM3 ALL-CAPS field label format.

This separation keeps the raw annotation data inspectable and the caption format
customizable.
"""

import re

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
    # Fields populated by the source-CSV join layer (opt-in via
    # --use_expasy / --use_brenda / --use_smart). Kept together at the
    # end so legacy captions don't shift.
    ("EC NAMES",                   "annot_ec_names"),
    ("EC DESCRIPTION",             "annot_ec_description"),
    ("SMART DOMAINS",              "annot_smart_domains"),
    ("BRENDA SUBSTRATES",          "annot_brenda_substrates"),
    ("BRENDA KM VALUES",           "annot_brenda_km_values"),
    ("BRENDA PH OPTIMUM",          "annot_brenda_ph_optimum"),
    ("BRENDA TEMPERATURE OPTIMUM", "annot_brenda_temperature_optimum"),
]

ANNOTATION_COLUMNS = [col for _, col in ANNOTATION_FIELDS]

# Annotation columns that aren't part of the caption schema but are still
# populated by source builders and consumed by the join layer. Initialized
# to pd.NA by enrich_dataframe so downstream reads are consistent.
EXTRA_ANNOTATION_COLUMNS = ["annot_ec_numbers"]
ALL_ANNOTATION_COLUMNS = ANNOTATION_COLUMNS + EXTRA_ANNOTATION_COLUMNS

# Regex for extracting EC numbers from UniProt catalytic_activity text.
# Matches patterns like "EC=1.2.3.4" (JSON API format), "EC 1.2.3.4"
# (prose), and bare "1.2.3.4" when embedded in reaction strings. The
# trailing dash forms ("1.2.-.-", "1.-.-.-") denote partial EC numbers.
_EC_NUMBER_RE = re.compile(
    r"(?<![\w.])(?:EC[\s=:]*)?(\d+\.(?:\d+|-)\.(?:\d+|-)\.(?:\d+|-))(?![\d-])"
)


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
# Source-CSV lookup loaders (used by enrich_dataframe to join ExPASy/BRENDA/SMART)
# ---------------------------------------------------------------------------

def load_expasy_lookup(csv_path):
    """Load expasy_enzyme.csv into a dict keyed by EC number.

    Each value is a dict with keys: annot_name, annot_alternative_names,
    annot_catalytic_activity, annot_comments, annot_uniprot_accessions.
    Obsolete (Transferred, Deleted) entries are included so legacy EC
    numbers resolve to a forwarding record.
    """
    logger.info("Loading ExPASy lookup from %s", csv_path)
    df = pd.read_csv(csv_path, dtype=str).fillna("")
    lookup = {}
    for _, row in df.iterrows():
        ec = row["ec"].strip()
        if not ec:
            continue
        lookup[ec] = {
            "name": row.get("annot_name", ""),
            "alternative_names": row.get("annot_alternative_names", ""),
            "catalytic_activity": row.get("annot_catalytic_activity", ""),
            "comments": row.get("annot_comments", ""),
            "uniprot_accessions": row.get("annot_uniprot_accessions", ""),
            "transferred_to": row.get("transferred_to", ""),
            "deleted": row.get("deleted", "False") == "True",
        }
    logger.info("Loaded %s ExPASy entries", f"{len(lookup):,}")
    return lookup


def load_brenda_lookup(csv_path):
    """Load brenda_kinetics.csv into a nested dict keyed by (EC, organism_lower).

    Each value is a dict with keys: substrates_products, km_values,
    ph_optimum, temperature_optimum. EC-level fields (recommended_name,
    synonyms) are stored under the "_shared" key per-EC for quick lookup
    when no organism match is found.
    """
    logger.info("Loading BRENDA lookup from %s", csv_path)
    df = pd.read_csv(csv_path, dtype=str).fillna("")
    by_ec_org = {}
    shared_by_ec = {}
    for _, row in df.iterrows():
        ec = row["ec"].strip()
        organism = row["organism"].strip()
        if not ec:
            continue
        key = (ec, organism.lower())
        by_ec_org[key] = {
            "substrates_products": row.get("annot_substrates_products", ""),
            "km_values": row.get("annot_km_values", ""),
            "ph_optimum": row.get("annot_ph_optimum", ""),
            "temperature_optimum": row.get("annot_temperature_optimum", ""),
        }
        if ec not in shared_by_ec:
            shared_by_ec[ec] = {
                "recommended_name": row.get("annot_recommended_name", ""),
                "systematic_name": row.get("annot_systematic_name", ""),
                "synonyms": row.get("annot_synonyms", ""),
                "reactions": row.get("annot_reactions", ""),
            }
    logger.info("Loaded %s BRENDA (EC, organism) records across %s ECs",
                f"{len(by_ec_org):,}", f"{len(shared_by_ec):,}")
    return {"by_ec_org": by_ec_org, "shared_by_ec": shared_by_ec}


def load_smart_lookup(csv_path):
    """Load smart_domains.csv into a dict keyed by SMART accession (SMxxxxx)."""
    logger.info("Loading SMART lookup from %s", csv_path)
    df = pd.read_csv(csv_path, dtype=str).fillna("")
    lookup = {}
    for _, row in df.iterrows():
        smart_id = row["domain_id"].strip()
        if not smart_id:
            continue
        lookup[smart_id] = {
            "name": row.get("annot_domain_name", ""),
            "definition": row.get("annot_definition", ""),
            "description": row.get("annot_description", ""),
        }
    logger.info("Loaded %s SMART domains", f"{len(lookup):,}")
    return lookup


def extract_ec_numbers(catalytic_activity_text):
    """Extract EC numbers from a UniProt catalytic_activity annotation.

    Returns a de-duplicated list of strings like ["1.1.1.1", "2.7.11.1"].
    """
    if catalytic_activity_text is None:
        return []
    try:
        if pd.isna(catalytic_activity_text):
            return []
    except (TypeError, ValueError):
        pass
    text = str(catalytic_activity_text)
    if not text:
        return []
    matches = _EC_NUMBER_RE.findall(text)
    seen = []
    for m in matches:
        if m not in seen:
            seen.append(m)
    return seen


# ---------------------------------------------------------------------------
# Join helpers (applied inside enrich_dataframe when lookups are provided)
# ---------------------------------------------------------------------------

def _is_missing(value):
    """Robust is-missing check that tolerates pd.NA, np.nan, None, '', etc."""
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    return False


def _strip_lineage_prefix(lineage_str):
    prefix = "The organism lineage is "
    if isinstance(lineage_str, str) and lineage_str.startswith(prefix):
        return lineage_str[len(prefix):]
    if _is_missing(lineage_str):
        return ""
    return str(lineage_str)


def _extract_row_ec_numbers(row):
    """Extract ECs for a row, checking three sources in precedence order:

    1. ``annot_ec_numbers`` column — the canonical structured field
       populated by SwissProt/TrEMBL source builders from .dat-level
       EC xrefs.
    2. ``annot_catalytic_activity`` — prose containing embedded EC
       references (rare in cleanly-built source CSVs since the xrefs
       are usually split out, but covers other data paths).
    3. ``[final]text_caption`` — last-resort scan of the composed
       caption for EC numbers.
    """
    ec_numbers_text = row.get("annot_ec_numbers")
    if not _is_missing(ec_numbers_text):
        ecs = [e.strip() for e in str(ec_numbers_text).split(",") if e.strip()]
        if ecs:
            return ecs
    ecs = extract_ec_numbers(row.get("annot_catalytic_activity"))
    if ecs:
        return ecs
    return extract_ec_numbers(row.get("[final]text_caption"))


def _row_organism_candidates(row):
    """Derive organism names for a row from annot_lineage, newest-first.

    Species are typically the last taxon in a UniProt lineage; we also try
    the last two tokens to catch genus+species binomials that BRENDA may
    store in a slightly different form.
    """
    lineage = row.get("annot_lineage")
    if _is_missing(lineage):
        return []
    parts = [p.strip() for p in _strip_lineage_prefix(lineage).split(",") if p.strip()]
    candidates = []
    if parts:
        candidates.append(parts[-1])
        if len(parts) >= 2:
            candidates.append(f"{parts[-2]} {parts[-1]}")
    return candidates


def _join_expasy(df, expasy_lookup):
    """Add annot_ec_names and annot_ec_description from ExPASy lookup.

    Returns a (df, stats) tuple where stats is a dict of hit counters.
    """
    ec_extracted = 0
    expasy_matched = 0

    def _row_expasy(row):
        nonlocal ec_extracted, expasy_matched
        ecs = _extract_row_ec_numbers(row)
        if ecs:
            ec_extracted += 1
        names = []
        descriptions = []
        matched_this_row = False
        for ec in ecs:
            entry = expasy_lookup.get(ec)
            if entry and entry["name"]:
                names.append(f"EC {ec}: {entry['name']}")
                if entry["comments"]:
                    descriptions.append(entry["comments"])
                matched_this_row = True
        if matched_this_row:
            expasy_matched += 1
        # Write ecs back to annot_ec_numbers only when the row didn't already
        # carry a source-supplied value — don't clobber the canonical column.
        existing = row.get("annot_ec_numbers")
        ec_cell = existing if not _is_missing(existing) and str(existing) else (
            ", ".join(ecs) if ecs else pd.NA
        )
        return pd.Series({
            "annot_ec_numbers": ec_cell,
            "annot_ec_names": "; ".join(names) if names else pd.NA,
            "annot_ec_description": " | ".join(descriptions) if descriptions else pd.NA,
        })

    enriched = df.apply(_row_expasy, axis=1)
    for col in ["annot_ec_numbers", "annot_ec_names", "annot_ec_description"]:
        df[col] = enriched[col]

    total = len(df)
    return df, {
        "ec_extraction_rate": ec_extracted / total if total else 0.0,
        "expasy_hit_rate": expasy_matched / total if total else 0.0,
    }


def _join_brenda(df, brenda_lookup, organism_match="strict"):
    """Add BRENDA per-(EC, organism) annotations.

    organism_match:
        - "strict": species-level match (last lineage taxon).
        - "relaxed": genus-level fallback if strict misses.
        - "ec_only": fall back to EC-level data when organism doesn't match.
    """
    by_ec_org = brenda_lookup["by_ec_org"]
    shared_by_ec = brenda_lookup["shared_by_ec"]

    hit_strict = 0
    hit_relaxed = 0
    hit_ec_only = 0

    def _row_brenda(row):
        nonlocal hit_strict, hit_relaxed, hit_ec_only
        ec_numbers_text = row.get("annot_ec_numbers")
        ecs = []
        if not _is_missing(ec_numbers_text) and str(ec_numbers_text):
            ecs = [e.strip() for e in str(ec_numbers_text).split(",") if e.strip()]
        if not ecs:
            ecs = _extract_row_ec_numbers(row)

        organisms = _row_organism_candidates(row)
        substrates = []
        kms = []
        phs = []
        temps = []
        matched_strict = False
        matched_relaxed = False
        matched_ec_only = False

        for ec in ecs:
            entry = None
            for org in organisms:
                entry = by_ec_org.get((ec, org.lower()))
                if entry:
                    matched_strict = True
                    break
            if not entry and organism_match == "relaxed" and organisms:
                genus = organisms[0].split()[0].lower() if organisms[0] else ""
                if genus:
                    for (cand_ec, cand_org), rec in by_ec_org.items():
                        if cand_ec == ec and cand_org.startswith(genus):
                            entry = rec
                            matched_relaxed = True
                            break
            if not entry and organism_match in ("strict", "relaxed", "ec_only"):
                if ec in shared_by_ec and organism_match == "ec_only":
                    matched_ec_only = True
                    continue
            if entry:
                if entry["substrates_products"]:
                    substrates.append(entry["substrates_products"])
                if entry["km_values"]:
                    kms.append(entry["km_values"])
                if entry["ph_optimum"]:
                    phs.append(entry["ph_optimum"])
                if entry["temperature_optimum"]:
                    temps.append(entry["temperature_optimum"])

        if matched_strict:
            hit_strict += 1
        elif matched_relaxed:
            hit_relaxed += 1
        elif matched_ec_only:
            hit_ec_only += 1

        return pd.Series({
            "annot_brenda_substrates": " | ".join(substrates) if substrates else pd.NA,
            "annot_brenda_km_values": " | ".join(kms) if kms else pd.NA,
            "annot_brenda_ph_optimum": " | ".join(phs) if phs else pd.NA,
            "annot_brenda_temperature_optimum": " | ".join(temps) if temps else pd.NA,
        })

    enriched = df.apply(_row_brenda, axis=1)
    for col in [
        "annot_brenda_substrates", "annot_brenda_km_values",
        "annot_brenda_ph_optimum", "annot_brenda_temperature_optimum",
    ]:
        df[col] = enriched[col]

    total = len(df)
    hit_any = hit_strict + hit_relaxed + hit_ec_only
    return df, {
        "brenda_hit_rate": hit_any / total if total else 0.0,
        "brenda_strict_hits": hit_strict,
        "brenda_relaxed_hits": hit_relaxed,
        "brenda_ec_only_hits": hit_ec_only,
    }


def _join_smart(df, smart_lookup):
    """Add annot_smart_domains from the xref_smart_ids list column."""
    if "xref_smart_ids" not in df.columns:
        df["annot_smart_domains"] = pd.NA
        return df, {"smart_hit_rate": 0.0}

    hit = 0

    def _row_smart(row):
        nonlocal hit
        ids = row.get("xref_smart_ids")
        if _is_missing(ids):
            return pd.NA
        if hasattr(ids, "__len__") and len(ids) == 0:
            return pd.NA
        parts = []
        for sid in ids:
            entry = smart_lookup.get(str(sid).strip())
            if entry and entry["definition"]:
                parts.append(f"{sid}: {entry['definition']}")
        if parts:
            hit += 1
            return "; ".join(parts)
        return pd.NA

    df["annot_smart_domains"] = df.apply(_row_smart, axis=1)
    total = len(df)
    return df, {"smart_hit_rate": hit / total if total else 0.0}


# ---------------------------------------------------------------------------
# DataFrame enrichment (Step 1: populate annotation columns)
# ---------------------------------------------------------------------------

def enrich_dataframe(df, local_annotations=None, uniprot_data=None,
                     taxonomy_tree=None, accession_taxid_map=None,
                     expasy_lookup=None, brenda_lookup=None,
                     smart_lookup=None, organism_match="strict"):
    """Populate individual annotation columns from local .dat, UniProt API,
    and/or NCBI taxonomy, and optionally join ExPASy/BRENDA/SMART source CSVs.

    Adds columns named annot_family_name, annot_family_description,
    annot_protein_name, annot_function, annot_lineage, etc. Each column
    holds the raw annotation text (or NaN if unavailable).

    Does NOT compose [final]text_caption — call compose_caption() for that.

    Args:
        df: DataFrame with primary_Accession column. Optionally family_name
            and family_description columns (for Pfam-sourced rows).
        local_annotations: optional dict mapping accession -> annotation dict
            (annot_* keys plus optional xref_smart_ids/xref_interpro_ids/
            xref_pdb_ids lists), as returned by SwissProtDatParser.parse().
        uniprot_data: optional dict mapping accession -> UniProt JSON entry.
        taxonomy_tree: optional TaxonomyTree instance.
        accession_taxid_map: optional dict mapping accession -> tax_id (int).
        expasy_lookup: dict from load_expasy_lookup(); enables EC-based join.
        brenda_lookup: dict from load_brenda_lookup(); enables (EC, organism)
            kinetics join.
        smart_lookup: dict from load_smart_lookup(); enables SMART domain join.
        organism_match: 'strict' | 'relaxed' | 'ec_only' for BRENDA.

    Returns:
        (df, join_stats) tuple where join_stats is a dict with hit-rate
        counters (empty if no lookups supplied).
    """
    df = df.copy()

    # Initialize annotation columns with NaN
    for col in ALL_ANNOTATION_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA

    # Copy family-level columns if present (Pfam rows)
    if "family_name" in df.columns:
        df["annot_family_name"] = df["family_name"]
    if "family_description" in df.columns:
        df["annot_family_description"] = df["family_description"]

    def _ensure_object_column(col):
        """Make sure *col* exists in df as an object-dtype column so list
        values can be stored per-cell via .at[]."""
        if col not in df.columns:
            df[col] = pd.Series([None] * len(df), dtype=object, index=df.index)
        elif df[col].dtype != object:
            df[col] = df[col].astype(object)

    def _apply_annotations(idx, annots):
        for col, val in annots.items():
            if isinstance(val, (list, tuple, set)):
                _ensure_object_column(col)
            df.at[idx, col] = val

    enriched_count = 0

    # Local .dat annotations (preferred — no API needed)
    if local_annotations:
        for idx, row in df.iterrows():
            acc = str(row.get("primary_Accession", ""))
            annots = local_annotations.get(acc)
            if annots:
                _apply_annotations(idx, annots)
                enriched_count += 1

    # UniProt API annotations (fallback when --use-api is set)
    if uniprot_data:
        for idx, row in df.iterrows():
            acc = str(row.get("primary_Accession", ""))
            entry = uniprot_data.get(acc)
            if entry:
                annotations = extract_annotations(entry)
                _apply_annotations(idx, annotations)
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

    join_stats = {}
    if expasy_lookup is not None:
        df, stats = _join_expasy(df, expasy_lookup)
        join_stats.update(stats)
        logger.info(
            "ExPASy join: EC extraction %.1f%%, hit rate %.1f%%",
            100 * stats.get("ec_extraction_rate", 0.0),
            100 * stats.get("expasy_hit_rate", 0.0),
        )
    if brenda_lookup is not None:
        df, stats = _join_brenda(df, brenda_lookup, organism_match=organism_match)
        join_stats.update(stats)
        logger.info(
            "BRENDA join (%s): hit rate %.1f%% (strict=%s, relaxed=%s, ec_only=%s)",
            organism_match,
            100 * stats.get("brenda_hit_rate", 0.0),
            stats.get("brenda_strict_hits", 0),
            stats.get("brenda_relaxed_hits", 0),
            stats.get("brenda_ec_only_hits", 0),
        )
    if smart_lookup is not None:
        df, stats = _join_smart(df, smart_lookup)
        join_stats.update(stats)
        logger.info(
            "SMART join: hit rate %.1f%%",
            100 * stats.get("smart_hit_rate", 0.0),
        )

    return df, join_stats


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

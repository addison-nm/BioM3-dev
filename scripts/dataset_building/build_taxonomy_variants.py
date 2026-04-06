#!/usr/bin/env python3
"""Build dataset variants with different levels of taxonomic information.

For a given Pfam family, builds multiple finetuning-ready CSVs that share
the same proteins but vary in how much taxonomy is included in the text
caption. This enables studying how taxonomy depth impacts model performance.

Usage:
    python scripts/dataset_building/build_taxonomy_variants.py \
        --pfam PF01817 --name CM \
        -o outputs/taxonomy_study/CM/

    python scripts/dataset_building/build_taxonomy_variants.py \
        --pfam PF00018 --name SH3 \
        -o outputs/taxonomy_study/SH3/

Prerequisites:
    - pip install -e .
    - Raw databases in data/databases/ (swissprot, pfam, ncbi_taxonomy)
    - Pre-built source CSVs (or will build from raw databases)
"""

import argparse
import csv
import os
import sys
from copy import deepcopy
from datetime import datetime

from biom3.backend.device import setup_logger
from biom3.dbio.caption import (
    CaptionSpec,
    TAXONOMY_RANKS,
    build_lineage_string,
    compose_row_caption,
)
from biom3.dbio.pfam_metadata import PfamMetadataParser
from biom3.dbio.swissprot_dat import SwissProtDatParser
from biom3.dbio.taxonomy import TaxonomyTree, AccessionTaxidMapper
from biom3.dbio.build_source_pfam import iter_pfam_fasta, _parse_fasta_header

logger = setup_logger(__name__)

# ---------------------------------------------------------------------------
# Variant definitions
# ---------------------------------------------------------------------------

# Base caption fields (everything except LINEAGE)
_BASE_FIELDS = [
    ("PROTEIN NAME",                  "annot_protein_name"),
    ("FUNCTION",                      "annot_function"),
    ("CATALYTIC ACTIVITY",            "annot_catalytic_activity"),
    ("COFACTOR",                      "annot_cofactor"),
    ("ACTIVITY REGULATION",           "annot_activity_regulation"),
    ("BIOPHYSICOCHEMICAL PROPERTIES", "annot_biophysicochemical_properties"),
    ("PATHWAY",                       "annot_pathway"),
    ("SUBUNIT",                       "annot_subunit"),
    ("SUBCELLULAR LOCATION",          "annot_subcellular_location"),
    ("TISSUE SPECIFICITY",            "annot_tissue_specificity"),
    ("DOMAIN",                        "annot_domain"),
    ("PTM",                           "annot_ptm"),
    ("SIMILARITY",                    "annot_similarity"),
    ("MISCELLANEOUS",                 "annot_miscellaneous"),
    ("INDUCTION",                     "annot_induction"),
    ("GENE ONTOLOGY",                 "annot_gene_ontology"),
]

VARIANTS = {
    "no_taxonomy": {
        "description": "No taxonomic information",
        "ranks": None,
        "include_lineage": False,
    },
    "domain_only": {
        "description": "Superkingdom only (Bacteria/Eukaryota/Archaea/Viruses)",
        "ranks": ["superkingdom"],
        "include_lineage": True,
    },
    "shallow": {
        "description": "Superkingdom + Phylum",
        "ranks": ["superkingdom", "phylum"],
        "include_lineage": True,
    },
    "medium": {
        "description": "Superkingdom through Order",
        "ranks": ["superkingdom", "phylum", "class", "order"],
        "include_lineage": True,
    },
    "full": {
        "description": "All NCBI ranks (rank-structured)",
        "ranks": None,  # all ranks
        "include_lineage": True,
    },
    "oc_lineage": {
        "description": "Raw OC lineage from .dat file (legacy format, unranked)",
        "ranks": "oc",  # sentinel: use OC lines instead of TaxonomyTree
        "include_lineage": True,
    },
}

OUTPUT_COLUMNS = [
    "primary_Accession",
    "protein_sequence",
    "[final]text_caption",
    "pfam_label",
]


def _make_spec(variant_name):
    """Build a CaptionSpec for a given variant."""
    v = VARIANTS[variant_name]
    fields = list(_BASE_FIELDS)
    if v["include_lineage"]:
        fields.append(("LINEAGE", "annot_lineage"))
    return CaptionSpec(
        fields=fields,
        strip_pubmed=True,
        strip_evidence=True,
        family_names_label="FAMILY NAMES",
        family_names_template="Family names are {names}",
    )


def _resolve_lineage(entry, variant_name, taxonomy_tree):
    """Compute annot_lineage for an entry based on the variant."""
    v = VARIANTS[variant_name]
    if not v["include_lineage"]:
        return None

    if v["ranks"] == "oc":
        # Use raw OC lineage already in annotations
        return entry["annotations"].get("annot_lineage")

    tax_id = entry.get("tax_id")
    if not tax_id or taxonomy_tree is None:
        return entry["annotations"].get("annot_lineage")

    lineage_dict = taxonomy_tree.get_lineage(tax_id)
    if not lineage_dict:
        return entry["annotations"].get("annot_lineage")

    return build_lineage_string(lineage_dict, ranks=v["ranks"])


def build_variant(variant_name, pfam_ids, dat_path, pfam_fasta_path,
                  pfam_metadata, taxonomy_tree, accession_taxid_map,
                  output_dir):
    """Build a single dataset variant for a set of Pfam IDs.

    Args:
        variant_name: key into VARIANTS dict
        pfam_ids: list of Pfam accessions to include
        dat_path: path to uniprot_sprot.dat.gz
        pfam_fasta_path: path to Pfam-A.fasta.gz
        pfam_metadata: dict from PfamMetadataParser.parse()
        taxonomy_tree: TaxonomyTree instance (loaded)
        accession_taxid_map: dict mapping accession -> tax_id (for Pfam rows)
        output_dir: directory for this variant's output
    """
    os.makedirs(output_dir, exist_ok=True)
    spec = _make_spec(variant_name)
    pfam_set = set(pfam_ids)

    rows = []

    # --- SwissProt entries ---
    logger.info("[%s] Scanning SwissProt .dat for Pfam IDs: %s",
                variant_name, pfam_ids)
    parser = SwissProtDatParser(dat_path)
    for accession, entry in parser.parse_all(require_pfam=True):
        entry_pfams = set(entry["pfam_ids"])
        if not entry_pfams & pfam_set:
            continue

        annotations = dict(entry["annotations"])
        lineage = _resolve_lineage(entry, variant_name, taxonomy_tree)
        if lineage:
            annotations["annot_lineage"] = lineage
        else:
            annotations.pop("annot_lineage", None)

        family_names = []
        for pid in entry["pfam_ids"]:
            meta = pfam_metadata.get(pid)
            family_names.append(meta["family_name"] if meta else pid)

        caption = compose_row_caption(annotations, spec,
                                      pfam_family_names=family_names)
        rows.append([
            accession,
            entry["sequence"],
            caption,
            repr(entry["pfam_ids"]),
        ])

    sp_count = len(rows)
    logger.info("[%s] SwissProt: %s rows", variant_name, f"{sp_count:,}")

    # --- Pfam FASTA entries ---
    logger.info("[%s] Scanning Pfam FASTA for Pfam IDs: %s",
                variant_name, pfam_ids)
    for header, sequence in iter_pfam_fasta(pfam_fasta_path):
        parsed = _parse_fasta_header(header)
        if parsed is None:
            continue
        if parsed["pfam_label"] not in pfam_set:
            continue

        meta = pfam_metadata.get(parsed["pfam_label"], {})
        family_name = meta.get("family_name", "")
        family_description = meta.get("family_description", "")

        annotations = {}
        if family_name:
            annotations["annot_family_name"] = family_name
        if family_description:
            annotations["annot_family_description"] = family_description

        # Add lineage for Pfam rows via accession → taxid → tree
        v = VARIANTS[variant_name]
        if v["include_lineage"] and v["ranks"] != "oc":
            acc = parsed["id"]
            tax_id = accession_taxid_map.get(acc)
            if tax_id and taxonomy_tree:
                lineage_dict = taxonomy_tree.get_lineage(tax_id)
                if lineage_dict:
                    lineage = build_lineage_string(lineage_dict, ranks=v["ranks"])
                    if lineage:
                        annotations["annot_lineage"] = lineage

        # Pfam rows use FAMILY NAME/DESCRIPTION fields instead of protein-level
        pfam_spec = CaptionSpec(
            fields=[
                ("FAMILY NAME", "annot_family_name"),
                ("FAMILY DESCRIPTION", "annot_family_description"),
            ] + ([("LINEAGE", "annot_lineage")] if "annot_lineage" in annotations else []),
            strip_pubmed=False,
            strip_evidence=False,
            family_names_label=None,
        )
        caption = compose_row_caption(annotations, pfam_spec)

        rows.append([
            parsed["id"],
            sequence,
            caption,
            parsed["pfam_label"],
        ])

    pfam_count = len(rows) - sp_count
    logger.info("[%s] Pfam: %s rows", variant_name, f"{pfam_count:,}")

    # --- Write output ---
    output_path = os.path.join(output_dir, "dataset.csv")
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(OUTPUT_COLUMNS)
        writer.writerows(rows)

    logger.info("[%s] Written %s total rows to %s",
                variant_name, f"{len(rows):,}", output_path)
    return {
        "swissprot_rows": sp_count,
        "pfam_rows": pfam_count,
        "total_rows": len(rows),
        "sample_caption": rows[0][2][:300] if rows else "",
    }


def _write_readme(output_dir, family_name, pfam_ids, variant_results):
    """Generate a README.md documenting the built variants."""
    readme_path = os.path.join(output_dir, "README.md")
    with open(readme_path, "w") as f:
        f.write(f"# Taxonomy Depth Variants: {family_name}\n\n")
        f.write(f"**Pfam IDs:** {', '.join(pfam_ids)}\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        f.write("## Variants\n\n")
        f.write("| Variant | Description | SwissProt | Pfam | Total |\n")
        f.write("|---------|-------------|-----------|------|-------|\n")
        for vname in VARIANTS:
            if vname in variant_results:
                r = variant_results[vname]
                desc = VARIANTS[vname]["description"]
                f.write(f"| `{vname}` | {desc} | "
                        f"{r['swissprot_rows']:,} | {r['pfam_rows']:,} | "
                        f"{r['total_rows']:,} |\n")
        f.write("\n## Caption Samples\n\n")
        for vname in VARIANTS:
            if vname in variant_results:
                r = variant_results[vname]
                f.write(f"### {vname}\n\n")
                f.write(f"```\n{r['sample_caption']}\n```\n\n")
        f.write("## Usage\n\n")
        f.write("Each variant directory contains a `dataset.csv` with columns:\n")
        f.write("`primary_Accession`, `protein_sequence`, "
                "`[final]text_caption`, `pfam_label`\n\n")
        f.write("These are ready for the embedding pipeline:\n")
        f.write("```bash\n")
        f.write(f"./scripts/embedding_pipeline.sh "
                f"{family_name}_<variant>/dataset.csv ...\n")
        f.write("```\n")


def main():
    parser = argparse.ArgumentParser(
        description="Build dataset variants with different taxonomy depths."
    )
    parser.add_argument(
        "--pfam", type=str, nargs="+", required=True,
        help="Pfam ID(s) to include (e.g., PF01817)",
    )
    parser.add_argument(
        "--name", type=str, required=True,
        help="Short name for the family (e.g., CM, SH3)",
    )
    parser.add_argument(
        "-o", "--output_dir", type=str, required=True,
        help="Output directory for all variants",
    )
    parser.add_argument(
        "--dat", type=str,
        default="data/databases/swissprot/uniprot_sprot.dat.gz",
        help="Path to uniprot_sprot.dat.gz",
    )
    parser.add_argument(
        "--pfam_fasta", type=str,
        default="data/databases/pfam/Pfam-A.fasta.gz",
        help="Path to Pfam-A.fasta.gz",
    )
    parser.add_argument(
        "--pfam_metadata", type=str,
        default="data/databases/pfam/Pfam-A.full.gz",
        help="Path to Pfam-A.full.gz for family metadata",
    )
    parser.add_argument(
        "--taxonomy_dir", type=str,
        default="data/databases/ncbi_taxonomy",
        help="Path to NCBI taxonomy directory",
    )
    parser.add_argument(
        "--taxid_index", type=str, default=None,
        help="Path to accession2taxid.sqlite (for Pfam row lineage)",
    )
    parser.add_argument(
        "--variants", type=str, nargs="*", default=None,
        help="Specific variants to build (default: all). "
             f"Choices: {', '.join(VARIANTS.keys())}",
    )
    args = parser.parse_args()

    pfam_ids = args.pfam
    variants_to_build = args.variants or list(VARIANTS.keys())

    # Validate variant names
    for v in variants_to_build:
        if v not in VARIANTS:
            parser.error(f"Unknown variant: {v}. Choices: {', '.join(VARIANTS.keys())}")

    # --- Load shared resources ---
    logger.info("Loading Pfam family metadata...")
    pfam_metadata = PfamMetadataParser(args.pfam_metadata).parse()

    logger.info("Loading NCBI taxonomy tree...")
    taxonomy_tree = TaxonomyTree(args.taxonomy_dir)
    taxonomy_tree.load()

    # Load accession-to-taxid map for Pfam rows (if SQLite index available)
    accession_taxid_map = {}
    taxid_index = args.taxid_index
    if taxid_index is None:
        candidate = os.path.join(args.taxonomy_dir, "accession2taxid.sqlite")
        if os.path.exists(candidate):
            taxid_index = candidate

    if taxid_index and os.path.exists(taxid_index):
        logger.info("Loading accession-to-taxid SQLite index: %s", taxid_index)
        # We'll do lookups on-the-fly per variant, but preload mapper
        accession_taxid_mapper = AccessionTaxidMapper(
            os.path.join(args.taxonomy_dir, "prot.accession2taxid.gz")
        )
    else:
        accession_taxid_mapper = None
        logger.warning("No accession2taxid index found. "
                       "Pfam rows will not have taxonomy lineage.")

    # Pre-collect Pfam accessions for bulk taxid lookup
    if accession_taxid_mapper:
        logger.info("Collecting Pfam accessions for taxonomy lookup...")
        pfam_accessions = set()
        pfam_set = set(pfam_ids)
        for header, _ in iter_pfam_fasta(args.pfam_fasta):
            parsed = _parse_fasta_header(header)
            if parsed and parsed["pfam_label"] in pfam_set:
                pfam_accessions.add(parsed["id"])
        logger.info("Found %s Pfam accessions to look up", f"{len(pfam_accessions):,}")

        if pfam_accessions:
            accession_taxid_map = accession_taxid_mapper.lookup(pfam_accessions)
            logger.info("Mapped %s/%s accessions to tax IDs",
                         f"{len(accession_taxid_map):,}",
                         f"{len(pfam_accessions):,}")

    # --- Build each variant ---
    os.makedirs(args.output_dir, exist_ok=True)
    variant_results = {}

    for vname in variants_to_build:
        logger.info("=" * 60)
        logger.info("Building variant: %s — %s", vname, VARIANTS[vname]["description"])
        logger.info("=" * 60)
        result = build_variant(
            variant_name=vname,
            pfam_ids=pfam_ids,
            dat_path=args.dat,
            pfam_fasta_path=args.pfam_fasta,
            pfam_metadata=pfam_metadata,
            taxonomy_tree=taxonomy_tree,
            accession_taxid_map=accession_taxid_map,
            output_dir=os.path.join(args.output_dir, vname),
        )
        variant_results[vname] = result

    # --- Write README ---
    _write_readme(args.output_dir, args.name, pfam_ids, variant_results)
    logger.info("README written to %s/README.md", args.output_dir)
    logger.info("Done. All variants built in %s/", args.output_dir)


if __name__ == "__main__":
    main()

"""Pipeline orchestrator for building fine-tuning datasets."""

import argparse
import csv
import logging
import os
import sys
from datetime import datetime

import pandas as pd

from biom3.backend.device import setup_logger
from biom3.core.run_utils import (
    get_biom3_version,
    get_git_hash,
    get_file_metadata,
    setup_file_logging,
    teardown_file_logging,
    write_manifest,
)
from biom3.dbio.config import (
    get_database_path,
    get_training_data_path,
)
from biom3.dbio.swissprot import SwissProtReader, OUTPUT_COLS
from biom3.dbio.pfam import PfamReader
from biom3.dbio.enrich import compose_caption

logger = setup_logger(__name__)


def _read_release_version(filepath, pattern):
    """Read a release file and extract a version string matching *pattern*."""
    import re
    try:
        with open(filepath) as f:
            text = f.read(2048)
        match = re.search(pattern, text)
        return match.group(1).strip() if match else None
    except Exception:
        return None


def _get_database_versions(args):
    """Collect version and file metadata for databases used in this build."""
    versions = {}

    swissprot_path = _resolve_swissprot_path(args)
    pfam_path = _resolve_pfam_path(args)

    versions["swissprot_csv"] = get_file_metadata(swissprot_path)
    versions["pfam_csv"] = get_file_metadata(pfam_path)

    # UniProt release version from reldate.txt
    try:
        db_root = str(get_database_path("swissprot", args.config))
        reldate_path = os.path.join(db_root, "reldate.txt")
        versions["uniprot_release"] = _read_release_version(
            reldate_path, r"Release\s+(\S+)",
        )
    except Exception:
        pass

    # Pfam release version from relnotes.txt
    try:
        db_root = str(get_database_path("pfam", args.config))
        relnotes_path = os.path.join(db_root, "relnotes.txt")
        versions["pfam_release"] = _read_release_version(
            relnotes_path, r"RELEASE\s+(\S+)",
        )
    except Exception:
        pass

    # NCBI taxonomy dump metadata
    if args.add_taxonomy:
        try:
            db_root = str(get_database_path("ncbi_taxonomy", args.config))
            rankedlineage = os.path.join(db_root, "rankedlineage.dmp")
            versions["ncbi_taxonomy"] = get_file_metadata(rankedlineage)
        except Exception:
            pass

    # Provenance TSV (download log with timestamps and MD5s)
    try:
        db_root = str(get_database_path("swissprot", args.config))
        provenance_path = os.path.join(os.path.dirname(db_root), "provenance.tsv")
        if os.path.exists(provenance_path):
            versions["provenance_tsv"] = os.path.realpath(provenance_path)
    except Exception:
        pass

    return versions


def parse_arguments(args):
    parser = argparse.ArgumentParser(
        description="Build a fine-tuning dataset for specified Pfam families."
    )
    parser.add_argument(
        "-p", "--pfam_ids", type=str, nargs="+", required=True,
        help="One or more Pfam IDs to extract (e.g. PF00018 PF00169)",
    )
    parser.add_argument(
        "-o", "--outdir", type=str, required=True,
        help="Output directory (will be created if it doesn't exist)",
    )
    parser.add_argument(
        "--swissprot", type=str, default=None,
        help="Path to fully_annotated_swiss_prot.csv (default: from config)",
    )
    parser.add_argument(
        "--pfam", type=str, default=None,
        help="Path to Pfam_protein_text_dataset.csv (default: from config)",
    )
    parser.add_argument(
        "--databases_root", type=str, default=None,
        help="Override database root path",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to dbio config JSON (default: configs/dbio_config.json)",
    )
    parser.add_argument(
        "--chunk_size", type=int, default=500_000,
        help="Chunk size for reading the Pfam CSV (default: 500000)",
    )
    parser.add_argument(
        "--enrich_pfam", action="store_true", default=False,
        help="Enrich Pfam captions with UniProt annotations (API by default)",
    )
    parser.add_argument(
        "--uniprot_dat", type=str, nargs="+", default=None,
        metavar="PATH",
        help="Use local UniProt .dat.gz file(s) instead of API for enrichment. "
             "Accepts one or more paths (e.g. uniprot_sprot.dat.gz uniprot_trembl.dat.gz). "
             "For full Pfam coverage, include the TrEMBL file.",
    )
    parser.add_argument(
        "--annotation_cache", type=str, nargs="+", default=None,
        metavar="PATH",
        help="Pre-built annotation Parquet cache(s) for fast enrichment "
             "(built via biom3_build_annotation_cache). Checked before "
             "--uniprot_dat; .dat files are only parsed for accessions "
             "not found in the cache.",
    )
    parser.add_argument(
        "--add_taxonomy", action="store_true", default=False,
        help="Add NCBI taxonomy lineage (local, no API needed)",
    )
    parser.add_argument(
        "--taxonomy_filter", type=str, nargs="*", default=None,
        help='Filter by taxonomy rank (e.g. "superkingdom=Bacteria")',
    )
    parser.add_argument(
        "--taxid_index", type=str, default=None,
        help="Path to pre-built SQLite accession2taxid index (built via biom3_build_taxid_index)",
    )
    parser.add_argument(
        "--output_filename", type=str, default="dataset.csv",
        help="Filename for the output dataset CSV (default: dataset.csv). "
             "The annotations file will be named with an '_annotations' suffix.",
    )
    parser.add_argument(
        "--uniprot_cache_dir", type=str, default=".uniprot_cache",
        help="Directory for caching UniProt API responses",
    )
    parser.add_argument(
        "--uniprot_batch_size", type=int, default=100,
        help="Batch size for UniProt API requests",
    )

    # Source-CSV join layer (opt-in). Each flag enables an additional
    # join against a per-database source CSV produced by the matching
    # biom3_build_source_* builder.
    parser.add_argument(
        "--use_expasy", action="store_true", default=False,
        help="Join ExPASy enzyme data on EC numbers extracted from "
             "annot_catalytic_activity. Adds annot_ec_names and "
             "annot_ec_description columns.",
    )
    parser.add_argument(
        "--expasy_csv", type=str, default=None,
        help="Path to expasy_enzyme.csv (required if --use_expasy).",
    )
    parser.add_argument(
        "--use_brenda", action="store_true", default=False,
        help="Join BRENDA per-organism kinetics on (EC, organism). Adds "
             "annot_brenda_substrates / km_values / ph_optimum / "
             "temperature_optimum columns.",
    )
    parser.add_argument(
        "--brenda_csv", type=str, default=None,
        help="Path to brenda_kinetics.csv (required if --use_brenda).",
    )
    parser.add_argument(
        "--organism_match", choices=["strict", "relaxed", "ec_only"],
        default="strict",
        help="BRENDA organism matching strictness. 'strict' (default) "
             "requires species-level match; 'relaxed' falls back to "
             "genus; 'ec_only' accepts any EC-level BRENDA record.",
    )
    parser.add_argument(
        "--use_smart", action="store_true", default=False,
        help="Join SMART domain descriptions on UniProt DR SMART "
             "cross-references. Adds annot_smart_domains column.",
    )
    parser.add_argument(
        "--smart_csv", type=str, default=None,
        help="Path to smart_domains.csv (required if --use_smart).",
    )
    return parser.parse_args(args)


def _resolve_swissprot_path(args):
    if args.swissprot:
        return args.swissprot
    return str(get_training_data_path("swissprot_csv", args.config))


def _resolve_pfam_path(args):
    if args.pfam:
        return args.pfam
    return str(get_training_data_path("pfam_csv", args.config))




def _parse_taxonomy_filters(filter_strs):
    """Parse "rank=value" strings into (rank, include_set) pairs."""
    filters = []
    for s in filter_strs:
        if "=" not in s:
            raise ValueError(f"Invalid taxonomy filter: {s!r}. Expected 'rank=value'.")
        rank, value = s.split("=", 1)
        filters.append((rank.strip(), value.strip()))
    return filters


def main(args):
    os.makedirs(args.outdir, exist_ok=True)

    # Set up dual logging (console + file)
    log_path, file_handler = setup_file_logging(
        args.outdir, logger_prefix="biom3.dbio", log_filename="build.log",
    )

    start_time = datetime.now()
    logger.info("=" * 60)
    logger.info("Build fine-tuning dataset")
    logger.info("biom3 version: %s (git: %s)", get_biom3_version(), get_git_hash())
    logger.info("Command:     %s", " ".join(sys.argv))
    logger.info("Pfam IDs:    %s", " ".join(args.pfam_ids))
    logger.info("Output dir:  %s", os.path.abspath(args.outdir))
    logger.info("=" * 60)

    # Resolve paths
    swissprot_path = _resolve_swissprot_path(args)
    pfam_path = _resolve_pfam_path(args)

    # Extract from SwissProt
    logger.info("Extracting from SwissProt...")
    sp_reader = SwissProtReader(swissprot_path)
    df_sp = sp_reader.query_by_pfam(args.pfam_ids)

    # Extract from Pfam (always keep family columns for annotation/caption)
    logger.info("Extracting from Pfam (chunked)...")
    pfam_reader = PfamReader(pfam_path, chunk_size=args.chunk_size)
    df_pfam = pfam_reader.query_by_pfam(
        args.pfam_ids, keep_family_cols=True,
    )

    # Step 1: Populate annotation columns on Pfam rows
    from biom3.dbio.enrich import enrich_dataframe

    local_annotations = None
    uniprot_data = None
    taxonomy_tree = None
    accession_taxid_map = None

    if args.enrich_pfam or args.add_taxonomy:
        accessions = df_pfam["primary_Accession"].dropna().unique().tolist()
        accession_set = set(accessions)

        if args.enrich_pfam:
            local_annotations = {}

            # Priority 1: Parquet annotation cache (instant lookup)
            if args.annotation_cache:
                from biom3.dbio.build_annotation_cache import load_annotation_cache

                local_annotations = load_annotation_cache(
                    args.annotation_cache, accession_set,
                )
                logger.info("Annotation cache: %s/%s accessions found",
                            f"{len(local_annotations):,}",
                            f"{len(accessions):,}")

            # Priority 2: Raw .dat file parsing (remaining accessions)
            if args.uniprot_dat:
                remaining = accession_set - set(local_annotations.keys())
                if remaining:
                    from biom3.dbio.swissprot_dat import SwissProtDatParser

                    for dat_path in args.uniprot_dat:
                        logger.info("Parsing local .dat file: %s", dat_path)
                        parser = SwissProtDatParser(dat_path)
                        still_remaining = accession_set - set(local_annotations.keys())
                        if not still_remaining:
                            logger.info("All accessions already found, skipping %s", dat_path)
                            break
                        local_annotations.update(parser.parse(still_remaining))
                    logger.info("Local enrichment total: %s/%s accessions found",
                                f"{len(local_annotations):,}",
                                f"{len(accessions):,}")

            # Priority 3: UniProt REST API (fallback)
            if not local_annotations and not args.uniprot_dat and not args.annotation_cache:
                from biom3.dbio.uniprot_client import UniProtClient

                logger.info("Enriching Pfam rows via UniProt REST API...")
                logger.info("Fetching annotations for %s unique accessions",
                             f"{len(accessions):,}")
                client = UniProtClient(
                    cache_dir=args.uniprot_cache_dir, use_cache=True,
                )
                uniprot_data = client.fetch_all(
                    accessions, batch_size=args.uniprot_batch_size,
                )

            local_annotations = local_annotations or None

        if args.add_taxonomy:
            taxonomy_tree, accession_taxid_map = _load_taxonomy(
                args, accessions,
            )

    # Load source-CSV lookups for the join layer (opt-in)
    expasy_lookup = None
    brenda_lookup = None
    smart_lookup = None

    if args.use_expasy:
        if not args.expasy_csv:
            raise ValueError("--use_expasy requires --expasy_csv")
        from biom3.dbio.enrich import load_expasy_lookup
        expasy_lookup = load_expasy_lookup(args.expasy_csv)
    if args.use_brenda:
        if not args.brenda_csv:
            raise ValueError("--use_brenda requires --brenda_csv")
        from biom3.dbio.enrich import load_brenda_lookup
        brenda_lookup = load_brenda_lookup(args.brenda_csv)
    if args.use_smart:
        if not args.smart_csv:
            raise ValueError("--use_smart requires --smart_csv")
        from biom3.dbio.enrich import load_smart_lookup
        smart_lookup = load_smart_lookup(args.smart_csv)

    # Always run enrich_dataframe to copy family columns into annot_* columns
    df_pfam, join_stats = enrich_dataframe(
        df_pfam,
        local_annotations=local_annotations,
        uniprot_data=uniprot_data,
        taxonomy_tree=taxonomy_tree,
        accession_taxid_map=accession_taxid_map,
        expasy_lookup=expasy_lookup,
        brenda_lookup=brenda_lookup,
        smart_lookup=smart_lookup,
        organism_match=args.organism_match,
    )

    # Step 2: Compose [final]text_caption from annotation columns (Pfam only).
    # SwissProt rows already have ALL-CAPS captions from the source CSV.
    df_pfam = compose_caption(df_pfam)

    # Combine
    df_combined = pd.concat([df_sp, df_pfam], ignore_index=True)
    logger.info("Combined dataset: %s rows", f"{len(df_combined):,}")
    logger.info("  SwissProt: %s", f"{len(df_sp):,}")
    logger.info("  Pfam:      %s", f"{len(df_pfam):,}")

    # Apply taxonomy filters
    if args.taxonomy_filter:
        df_combined = _apply_taxonomy_filters(
            df_combined, args,
        )

    # Derive annotations filename from output filename
    stem, ext = os.path.splitext(args.output_filename)
    annotations_filename = f"{stem}_annotations{ext}"

    # Save intermediate CSV with annotation columns (all columns preserved)
    annotations_path = os.path.join(args.outdir, annotations_filename)
    df_combined.to_csv(annotations_path, index=False, quoting=csv.QUOTE_NONNUMERIC)
    logger.info("Saved annotated dataset to %s", annotations_path)

    # Save final dataset (standard output columns only)
    out_path = os.path.join(args.outdir, args.output_filename)
    df_combined[OUTPUT_COLS].to_csv(out_path, index=False, quoting=csv.QUOTE_NONNUMERIC)
    logger.info("Saved dataset to %s", out_path)

    # Save metadata
    pfam_ids_path = os.path.join(args.outdir, "pfam_ids.csv")
    pd.DataFrame({"pfam_id": args.pfam_ids}).to_csv(pfam_ids_path, index=False)

    elapsed = datetime.now() - start_time
    logger.info("Done in %s", elapsed)
    logger.info("Log saved to %s", log_path)

    # Write reproducibility manifest
    row_counts = {
        "swissprot": len(df_sp),
        "pfam": len(df_pfam),
        "combined": len(df_combined),
    }
    resolved_paths = {
        "swissprot_csv": os.path.abspath(_resolve_swissprot_path(args)),
        "pfam_csv": os.path.abspath(_resolve_pfam_path(args)),
    }
    if args.use_expasy and args.expasy_csv:
        resolved_paths["expasy_csv"] = os.path.abspath(args.expasy_csv)
    if args.use_brenda and args.brenda_csv:
        resolved_paths["brenda_csv"] = os.path.abspath(args.brenda_csv)
    if args.use_smart and args.smart_csv:
        resolved_paths["smart_csv"] = os.path.abspath(args.smart_csv)

    database_versions = _get_database_versions(args)
    if args.use_expasy and args.expasy_csv:
        database_versions["expasy_csv"] = get_file_metadata(args.expasy_csv)
    if args.use_brenda and args.brenda_csv:
        database_versions["brenda_csv"] = get_file_metadata(args.brenda_csv)
    if args.use_smart and args.smart_csv:
        database_versions["smart_csv"] = get_file_metadata(args.smart_csv)

    outputs = {"row_counts": row_counts}
    if join_stats:
        outputs["join_stats"] = join_stats

    manifest_path = write_manifest(
        args, args.outdir, start_time, elapsed,
        outputs=outputs,
        resolved_paths=resolved_paths,
        database_versions=database_versions,
    )
    logger.info("Saved build manifest to %s", manifest_path)

    # Clean up file handler
    teardown_file_logging("biom3.dbio", file_handler)


def _load_taxonomy(args, accessions):
    """Load taxonomy tree and look up accessions."""
    from biom3.dbio.taxonomy import TaxonomyTree, AccessionTaxidMapper

    taxonomy_dir = str(get_database_path("ncbi_taxonomy", args.config))
    taxonomy_tree = TaxonomyTree(taxonomy_dir)
    taxonomy_tree.load()

    accession2taxid_path = os.path.join(taxonomy_dir, "prot.accession2taxid.gz")
    mapper = AccessionTaxidMapper(accession2taxid_path)

    if args.taxid_index:
        accession_taxid_map = mapper.lookup_sqlite(accessions, args.taxid_index)
    else:
        accession_taxid_map = mapper.lookup(accessions)

    return taxonomy_tree, accession_taxid_map


def _parse_annot_lineage(lineage_str):
    """Parse an annot_lineage string into a list of taxon names.

    Example input:
        "The organism lineage is Eukaryota, Metazoa, Chordata, ..."
    Returns:
        ["Eukaryota", "Metazoa", "Chordata", ...]
    """
    prefix = "The organism lineage is "
    if lineage_str.startswith(prefix):
        lineage_str = lineage_str[len(prefix):]
    return [t.strip() for t in lineage_str.split(",") if t.strip()]


def _apply_taxonomy_filters(df, args):
    """Filter the combined DataFrame by taxonomy rank constraints.

    Uses two sources for taxonomy data:
    1. NCBI prot.accession2taxid lookup (structured rank->value via TaxonomyTree)
    2. The annot_lineage column populated by UniProt enrichment (flat lineage list)

    Accessions found in the NCBI index use the structured approach (exact rank
    matching). Accessions not in the NCBI index fall back to checking whether
    the filter value appears anywhere in the annot_lineage string.
    """
    from biom3.dbio.taxonomy import TaxonomyTree, AccessionTaxidMapper

    filters = _parse_taxonomy_filters(args.taxonomy_filter)
    taxonomy_dir = str(get_database_path("ncbi_taxonomy", args.config))
    taxonomy_tree = TaxonomyTree(taxonomy_dir)
    taxonomy_tree.load()

    accession2taxid_path = os.path.join(taxonomy_dir, "prot.accession2taxid.gz")
    mapper = AccessionTaxidMapper(accession2taxid_path)

    accessions = df["primary_Accession"].dropna().unique().tolist()
    if args.taxid_index:
        acc_to_taxid = mapper.lookup_sqlite(accessions, args.taxid_index)
    else:
        acc_to_taxid = mapper.lookup(accessions)

    # --- Path 1: NCBI structured lookup ---
    taxid_to_accs = {}
    for acc, tid in acc_to_taxid.items():
        taxid_to_accs.setdefault(tid, set()).add(acc)

    all_taxids = set(acc_to_taxid.values())

    for rank, value in filters:
        logger.info("Applying taxonomy filter: %s=%s", rank, value)
        kept_taxids = taxonomy_tree.filter_by_rank(
            all_taxids, rank, include={value},
        )
        all_taxids = kept_taxids

    kept_accs = set()
    for tid in all_taxids:
        kept_accs.update(taxid_to_accs.get(tid, set()))
    ncbi_kept = len(kept_accs)

    # --- Path 2: fallback to annot_lineage for unmapped accessions ---
    ncbi_mapped = set(acc_to_taxid.keys())
    has_lineage_col = "annot_lineage" in df.columns
    fallback_kept = 0

    if has_lineage_col:
        for _, row in df.iterrows():
            acc = row.get("primary_Accession")
            if pd.isna(acc) or acc in ncbi_mapped:
                continue
            lineage_str = row.get("annot_lineage")
            if pd.isna(lineage_str) or not lineage_str:
                continue
            lineage_terms = _parse_annot_lineage(str(lineage_str))
            passes = all(value in lineage_terms for _, value in filters)
            if passes:
                kept_accs.add(acc)
                fallback_kept += 1

    if has_lineage_col:
        logger.info("Taxonomy filter matched: %s via NCBI index, %s via annot_lineage",
                     f"{ncbi_kept:,}", f"{fallback_kept:,}")

    before = len(df)
    df = df[df["primary_Accession"].isin(kept_accs)].copy()
    logger.info("Taxonomy filter: %s -> %s rows", f"{before:,}", f"{len(df):,}")
    return df

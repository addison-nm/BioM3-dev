"""Pipeline orchestrator for building fine-tuning datasets."""

import argparse
import csv
import json
import logging
import os
import subprocess
import sys
from datetime import datetime

import pandas as pd

from biom3.backend.device import setup_logger
from biom3.dbio.config import (
    get_database_path,
    get_training_data_path,
)
from biom3.dbio.swissprot import SwissProtReader, OUTPUT_COLS
from biom3.dbio.pfam import PfamReader
from biom3.dbio.enrich import compose_caption

logger = setup_logger(__name__)


def _get_biom3_version():
    try:
        from importlib.metadata import version
        return version("biom3")
    except Exception:
        return "unknown"


def _get_git_hash():
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def _setup_file_logging(outdir):
    """Add a FileHandler to all biom3.dbio.* loggers so they log to both
    console and a file in the output directory."""
    log_path = os.path.join(outdir, "build.log")
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    file_handler.setLevel(logging.INFO)
    # Add to all existing biom3.dbio.* loggers (propagate=False prevents
    # inheritance, so we must attach directly)
    for name, lg in logging.Logger.manager.loggerDict.items():
        if isinstance(lg, logging.Logger) and name.startswith("biom3.dbio"):
            lg.addHandler(file_handler)
    return log_path, file_handler


def _write_manifest(args, outdir, start_time, elapsed, row_counts):
    """Write build_manifest.json with reproduction info."""
    manifest = {
        "biom3_version": _get_biom3_version(),
        "git_hash": _get_git_hash(),
        "timestamp": start_time.isoformat(),
        "elapsed_seconds": elapsed.total_seconds(),
        "command": " ".join(sys.argv),
        "args": {k: v for k, v in vars(args).items()},
        "resolved_paths": {
            "swissprot_csv": os.path.abspath(_resolve_swissprot_path(args)),
            "pfam_csv": os.path.abspath(_resolve_pfam_path(args)),
        },
        "row_counts": row_counts,
        "python_version": sys.version,
    }
    manifest_path = os.path.join(outdir, "build_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    logger.info("Saved build manifest to %s", manifest_path)


def parse_arguments(args):
    parser = argparse.ArgumentParser(
        description="Build a fine-tuning dataset for specified Pfam families."
    )
    parser.add_argument(
        "-p", "--pfam-ids", type=str, nargs="+", required=True,
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
        "--databases-root", type=str, default=None,
        help="Override database root path",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to dbio config JSON (default: configs/dbio_config.json)",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=500_000,
        help="Chunk size for reading the Pfam CSV (default: 500000)",
    )
    parser.add_argument(
        "--enrich-pfam", action="store_true", default=False,
        help="Enrich Pfam captions with UniProt annotations (API by default)",
    )
    parser.add_argument(
        "--uniprot-dat", type=str, nargs="+", default=None,
        metavar="PATH",
        help="Use local UniProt .dat.gz file(s) instead of API for enrichment. "
             "Accepts one or more paths (e.g. uniprot_sprot.dat.gz uniprot_trembl.dat.gz). "
             "For full Pfam coverage, include the TrEMBL file.",
    )
    parser.add_argument(
        "--add-taxonomy", action="store_true", default=False,
        help="Add NCBI taxonomy lineage (local, no API needed)",
    )
    parser.add_argument(
        "--taxonomy-filter", type=str, nargs="*", default=None,
        help='Filter by taxonomy rank (e.g. "superkingdom=Bacteria")',
    )
    parser.add_argument(
        "--taxid-index", type=str, default=None,
        help="Path to pre-built SQLite accession2taxid index (built via biom3_build_taxid_index)",
    )
    parser.add_argument(
        "--uniprot-cache-dir", type=str, default=".uniprot_cache",
        help="Directory for caching UniProt API responses",
    )
    parser.add_argument(
        "--uniprot-batch-size", type=int, default=100,
        help="Batch size for UniProt API requests",
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
    log_path, file_handler = _setup_file_logging(args.outdir)

    start_time = datetime.now()
    logger.info("=" * 60)
    logger.info("Build fine-tuning dataset")
    logger.info("biom3 version: %s (git: %s)", _get_biom3_version(), _get_git_hash())
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
            if args.uniprot_dat:
                from biom3.dbio.swissprot_dat import SwissProtDatParser

                local_annotations = {}
                for dat_path in args.uniprot_dat:
                    logger.info("Parsing local .dat file: %s", dat_path)
                    parser = SwissProtDatParser(dat_path)
                    remaining = accession_set - set(local_annotations.keys())
                    if not remaining:
                        logger.info("All accessions already found, skipping %s", dat_path)
                        break
                    local_annotations.update(parser.parse(remaining))
                logger.info("Local .dat enrichment: %s/%s accessions found",
                            f"{len(local_annotations):,}",
                            f"{len(accessions):,}")
            else:
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

        if args.add_taxonomy:
            taxonomy_tree, accession_taxid_map = _load_taxonomy(
                args, accessions,
            )

    # Always run enrich_dataframe to copy family columns into annot_* columns
    df_pfam = enrich_dataframe(
        df_pfam,
        local_annotations=local_annotations,
        uniprot_data=uniprot_data,
        taxonomy_tree=taxonomy_tree,
        accession_taxid_map=accession_taxid_map,
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

    # Save intermediate CSV with annotation columns (all columns preserved)
    annotations_path = os.path.join(args.outdir, "dataset_annotations.csv")
    df_combined.to_csv(annotations_path, index=False, quoting=csv.QUOTE_NONNUMERIC)
    logger.info("Saved annotated dataset to %s", annotations_path)

    # Save final dataset (standard output columns only)
    out_path = os.path.join(args.outdir, "dataset.csv")
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
    _write_manifest(args, args.outdir, start_time, elapsed, row_counts)

    # Clean up file handler
    for name, lg in logging.Logger.manager.loggerDict.items():
        if isinstance(lg, logging.Logger) and name.startswith("biom3.dbio"):
            lg.removeHandler(file_handler)
    file_handler.close()


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


def _apply_taxonomy_filters(df, args):
    """Filter the combined DataFrame by taxonomy rank constraints."""
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

    # Build taxid -> set of accessions mapping
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

    # Collect accessions that survived filtering
    kept_accs = set()
    for tid in all_taxids:
        kept_accs.update(taxid_to_accs.get(tid, set()))

    before = len(df)
    df = df[df["primary_Accession"].isin(kept_accs)].copy()
    logger.info("Taxonomy filter: %s -> %s rows", f"{before:,}", f"{len(df):,}")
    return df

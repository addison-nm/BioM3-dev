#!/usr/bin/env python
"""Demo: Build an SH3 domain dataset using the dbio module.

This script demonstrates how to use BioM3's database I/O layer to query
protein databases by Pfam family, enrich Pfam rows with UniProt annotations,
and assemble a fine-tuning dataset with ALL-CAPS caption formatting.

SH3 domains (PF00018) are small protein interaction modules (~60 residues)
that bind proline-rich peptide sequences. They are one of the most common
domain families in signaling proteins.

Usage (with mini test data — no database download required):

    python demo/build_sh3_dataset.py

Usage (with real databases + UniProt enrichment):

    python demo/build_sh3_dataset.py \
        --swissprot ../BioM3-data-share/data/datasets/fully_annotated_swiss_prot.csv \
        --pfam ../BioM3-data-share/data/datasets/Pfam_protein_text_dataset.csv \
        --enrich-pfam
"""

import argparse
import os

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MINI_SWISSPROT = os.path.join(REPO_ROOT, "tests", "_data", "dbio", "mini_swissprot.csv")
MINI_PFAM = os.path.join(REPO_ROOT, "tests", "_data", "dbio", "mini_pfam.csv")
DEFAULT_OUTDIR = os.path.join(REPO_ROOT, "demo", "_output", "sh3_dataset")

SH3_PFAM_ID = "PF00018"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--swissprot", default=MINI_SWISSPROT,
                        help="Path to SwissProt CSV (default: mini test data)")
    parser.add_argument("--pfam", default=MINI_PFAM,
                        help="Path to Pfam CSV (default: mini test data)")
    parser.add_argument("-o", "--outdir", default=DEFAULT_OUTDIR,
                        help="Output directory")
    parser.add_argument("--enrich-pfam", action="store_true", default=False,
                        help="Enrich Pfam captions with UniProt annotations (API by default)")
    parser.add_argument("--uniprot-dat", type=str, nargs="+", default=None,
                        metavar="PATH",
                        help="Use local .dat.gz file(s) instead of API for enrichment")
    parser.add_argument("--uniprot-cache-dir", default=".uniprot_cache",
                        help="Directory for caching UniProt API responses")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("  BioM3 Demo: Build SH3 Domain Dataset")
    print("=" * 60)
    print()

    # ------------------------------------------------------------------
    # Step 1: Query SwissProt for SH3-containing proteins
    # ------------------------------------------------------------------
    print(f"[1/5] Querying SwissProt for {SH3_PFAM_ID} (SH3 domain)...")
    from biom3.dbio.swissprot import SwissProtReader, OUTPUT_COLS

    sp_reader = SwissProtReader(args.swissprot)
    df_sp = sp_reader.query_by_pfam([SH3_PFAM_ID])

    print(f"      Found {len(df_sp)} proteins in SwissProt")
    if len(df_sp) > 0:
        print()
        print("      Sample entries:")
        for _, row in df_sp.head(3).iterrows():
            seq_preview = row["protein_sequence"][:20] + "..."
            print(f"        {row['primary_Accession']:12s}  {seq_preview}  {row['[final]text_caption'][:50]}")
    print()

    # ------------------------------------------------------------------
    # Step 2: Query Pfam for SH3 domain sequences
    # ------------------------------------------------------------------
    print(f"[2/5] Querying Pfam for {SH3_PFAM_ID}...")
    from biom3.dbio.pfam import PfamReader

    pfam_reader = PfamReader(args.pfam, chunk_size=100_000)
    df_pfam = pfam_reader.query_by_pfam([SH3_PFAM_ID], keep_family_cols=True)

    print(f"      Found {len(df_pfam)} domain sequences in Pfam")
    print()

    # ------------------------------------------------------------------
    # Step 3: Enrich Pfam rows with annotation columns
    # ------------------------------------------------------------------
    from biom3.dbio.enrich import enrich_dataframe, compose_caption

    if args.enrich_pfam:
        accessions = df_pfam["primary_Accession"].dropna().unique().tolist()

        if args.uniprot_dat:
            from biom3.dbio.swissprot_dat import SwissProtDatParser

            local_annotations = {}
            for dat_path in args.uniprot_dat:
                print(f"[3/5] Parsing local .dat file: {dat_path}")
                parser = SwissProtDatParser(dat_path)
                remaining = set(accessions) - set(local_annotations.keys())
                if not remaining:
                    print("      All accessions already found, skipping")
                    break
                local_annotations.update(parser.parse(remaining))
            print(f"      Local enrichment: {len(local_annotations):,}/{len(accessions):,} accessions found")
            df_pfam = enrich_dataframe(df_pfam, local_annotations=local_annotations)
        else:
            print("[3/5] Enriching Pfam rows via UniProt REST API...")
            from biom3.dbio.uniprot_client import UniProtClient

            print(f"      Fetching annotations for {len(accessions):,} unique accessions...")
            client = UniProtClient(cache_dir=args.uniprot_cache_dir, use_cache=True)
            uniprot_data = client.fetch_all(accessions, batch_size=25)
            print(f"      Fetched {len(uniprot_data):,} entries")
            df_pfam = enrich_dataframe(df_pfam, uniprot_data=uniprot_data)
    else:
        print("[3/5] Skipping enrichment (use --enrich-pfam to enable)")
        df_pfam = enrich_dataframe(df_pfam)
    print()

    # ------------------------------------------------------------------
    # Step 4: Compose captions and combine
    # ------------------------------------------------------------------
    print("[4/5] Composing captions and combining...")
    df_pfam = compose_caption(df_pfam)

    df_combined = pd.concat(
        [df_sp, df_pfam[OUTPUT_COLS].copy()], ignore_index=True,
    )
    n_unique = df_combined["primary_Accession"].nunique()
    print(f"      Combined: {len(df_combined)} rows ({n_unique} unique accessions)")
    print(f"        SwissProt: {len(df_sp)}")
    print(f"        Pfam:      {len(df_pfam)}")
    print()

    # ------------------------------------------------------------------
    # Step 5: Save the dataset
    # ------------------------------------------------------------------
    os.makedirs(args.outdir, exist_ok=True)
    out_path = os.path.join(args.outdir, "dataset.csv")
    df_combined.to_csv(out_path, index=False)

    pfam_ids_path = os.path.join(args.outdir, "pfam_ids.csv")
    pd.DataFrame({"pfam_id": [SH3_PFAM_ID]}).to_csv(pfam_ids_path, index=False)

    print(f"[5/5] Saved dataset to {out_path}")
    print(f"      Saved Pfam IDs to {pfam_ids_path}")
    print()

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("=" * 60)
    print("  Dataset Summary")
    print("=" * 60)
    print(f"  Pfam family:       {SH3_PFAM_ID} (SH3 domain)")
    print(f"  Total rows:        {len(df_combined)}")
    print(f"  Unique accessions: {n_unique}")
    print(f"  Enriched:          {'yes (UniProt)' if args.enrich_pfam else 'no'}")
    print(f"  Output columns:    {OUTPUT_COLS}")
    print()

    # Preview the output
    def _print_rows(df_slice):
        for _, row in df_slice.iterrows():
            seq = row["protein_sequence"][:20] + "..."
            cap = row["[final]text_caption"][:80] + "..."
            print(f"    {row['primary_Accession']:12s}  {seq:24s}  {cap}")

    print("  Preview (first 5 rows):")
    _print_rows(df_combined.head(5))
    print("  ...")
    print("  Preview (last 5 rows):")
    _print_rows(df_combined.tail(5))
    print()


if __name__ == "__main__":
    main()

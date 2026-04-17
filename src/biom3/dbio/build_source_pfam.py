"""Build Pfam_protein_text_dataset.csv from Pfam-A.fasta.gz + family metadata.

This produces a CSV that is a drop-in replacement for the legacy
Pfam_protein_text_dataset.csv used by build_dataset.py and PfamReader.

Usage:
    biom3_build_source_pfam \\
        --fasta data/databases/pfam/Pfam-A.fasta.gz \\
        --pfam_metadata data/databases/pfam/Pfam-A.full.gz \\
        -o data/datasets/Pfam_protein_text_dataset.csv
"""

import argparse
import csv
import gzip
import os
import re
import sys
from datetime import datetime

from tqdm import tqdm

from biom3.backend.device import setup_logger
from biom3.core.run_utils import (
    get_file_metadata,
    write_manifest,
)
from biom3.dbio.caption import CaptionSpec, compose_row_caption
from biom3.dbio.pfam_metadata import PfamMetadataParser

logger = setup_logger(__name__)

# Default caption spec matching the original Pfam CSV format.
# Uses lowercase labels ("Protein name:", "Family description:").
PFAM_SPEC = CaptionSpec(
    fields=[
        ("Protein name", "family_name"),
        ("Family description", "family_description"),
    ],
    strip_pubmed=False,
    strip_evidence=False,
    family_names_label=None,
    trailing_period=False,
)

OUTPUT_COLUMNS = [
    "id",
    "range",
    "description",
    "pfam_label",
    "sequence",
    "family_name",
    "family_description",
    "[final]text_caption",
]


def _parse_fasta_header(header):
    """Parse a Pfam-A.fasta header line (without leading '>').

    Format: A0A067SRH6_GALM3/383-505 A0A067SRH6.1 PF26733.1;03009_C;

    Returns:
        dict with keys: id, range, description, pfam_label
        or None if header can't be parsed.
    """
    parts = header.split()
    if len(parts) < 3:
        return None

    name_range = parts[0]
    acc_versioned = parts[1]
    pfam_field = parts[2]

    range_part = name_range.split("/")[-1] if "/" in name_range else ""
    accession = acc_versioned.split(".")[0]
    pfam_label = pfam_field.split(".")[0]
    description = " ".join(parts[1:])

    return {
        "id": accession,
        "range": range_part,
        "description": description,
        "pfam_label": pfam_label,
    }


def iter_pfam_fasta(fasta_path):
    """Yield (header, sequence) tuples from a FASTA file (plain or gzipped).

    This is a public utility for recipe scripts that need to iterate
    over Pfam FASTA entries without going through the full CSV builder.
    """
    is_gzipped = fasta_path.endswith(".gz")
    opener = gzip.open if is_gzipped else open

    header = None
    seq_parts = []

    with opener(fasta_path, "rt") as f:
        for line in tqdm(f, desc="Parsing Pfam FASTA", unit=" lines"):
            line = line.rstrip("\n")
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(seq_parts)
                header = line[1:]
                seq_parts = []
            else:
                seq_parts.append(line)

    if header is not None:
        yield header, "".join(seq_parts)


def build_pfam_csv(fasta_path, pfam_metadata, output_path,
                   caption_spec=None, chunk_size=100_000):
    """Build Pfam_protein_text_dataset.csv from Pfam-A.fasta.gz.

    Args:
        fasta_path: path to Pfam-A.fasta.gz
        pfam_metadata: dict from PfamMetadataParser.parse()
        output_path: output CSV path
        caption_spec: CaptionSpec controlling caption format. Defaults to
            PFAM_SPEC (original Pfam CSV format).
        chunk_size: rows to buffer before writing

    Returns:
        int: number of rows written.
    """
    if caption_spec is None:
        caption_spec = PFAM_SPEC

    buffer = []
    total = 0
    skipped = 0

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(OUTPUT_COLUMNS)

        for header, sequence in iter_pfam_fasta(fasta_path):
            parsed = _parse_fasta_header(header)
            if parsed is None:
                skipped += 1
                continue

            pfam_label = parsed["pfam_label"]
            meta = pfam_metadata.get(pfam_label, {})
            family_name = meta.get("family_name", "")
            family_description = meta.get("family_description", "")

            caption = compose_row_caption(
                {"family_name": family_name, "family_description": family_description},
                caption_spec,
            )

            buffer.append([
                parsed["id"],
                parsed["range"],
                parsed["description"],
                pfam_label,
                sequence,
                family_name,
                family_description,
                caption,
            ])

            if len(buffer) >= chunk_size:
                writer.writerows(buffer)
                total += len(buffer)
                buffer = []
                logger.info("Written %s rows", f"{total:,}")

        if buffer:
            writer.writerows(buffer)
            total += len(buffer)

    if skipped:
        logger.warning("Skipped %s unparseable FASTA headers", f"{skipped:,}")
    logger.info("Build complete: %s rows written to %s", f"{total:,}", output_path)
    return total


def _read_release_version(filepath, pattern):
    """Read a release file and extract a version string matching *pattern*."""
    try:
        with open(filepath) as f:
            text = f.read(2048)
        match = re.search(pattern, text)
        return match.group(1).strip() if match else None
    except Exception:
        return None


def _collect_database_versions(fasta_path, pfam_metadata_path):
    """Collect provenance metadata for the Pfam FASTA and metadata files."""
    versions = {"pfam_fasta": get_file_metadata(fasta_path),
                "pfam_metadata": get_file_metadata(pfam_metadata_path)}

    pfam_dir = os.path.dirname(os.path.abspath(pfam_metadata_path))
    relnotes_path = os.path.join(pfam_dir, "relnotes.txt")
    if os.path.exists(relnotes_path):
        versions["pfam_release"] = _read_release_version(
            relnotes_path, r"RELEASE\s+(\S+)",
        )

    return versions


def parse_arguments(args):
    parser = argparse.ArgumentParser(
        description="Build Pfam_protein_text_dataset.csv from Pfam-A.fasta.gz."
    )
    parser.add_argument(
        "--fasta", type=str, required=True,
        help="Path to Pfam-A.fasta.gz",
    )
    parser.add_argument(
        "--pfam_metadata", type=str, required=True,
        help="Path to Pfam-A.full.gz or Pfam-A.hmm.gz for family metadata",
    )
    parser.add_argument(
        "-o", "--output", type=str, required=True,
        help="Output CSV path",
    )
    parser.add_argument(
        "--chunk_size", type=int, default=100_000,
        help="Rows to buffer before writing (default: 100000)",
    )
    return parser.parse_args(args)


def main(args):
    start_time = datetime.now()

    logger.info("Loading Pfam family metadata from %s", args.pfam_metadata)
    pfam_metadata = PfamMetadataParser(args.pfam_metadata).parse()

    row_count = build_pfam_csv(
        fasta_path=args.fasta,
        pfam_metadata=pfam_metadata,
        output_path=args.output,
        chunk_size=args.chunk_size,
    )

    elapsed = datetime.now() - start_time

    outdir = os.path.dirname(os.path.abspath(args.output))
    stem = os.path.splitext(os.path.basename(args.output))[0]
    database_versions = _collect_database_versions(args.fasta, args.pfam_metadata)
    resolved_paths = {
        "fasta": os.path.abspath(args.fasta),
        "pfam_metadata": os.path.abspath(args.pfam_metadata),
        "output": os.path.abspath(args.output),
    }
    manifest_path = write_manifest(
        args, outdir, start_time, elapsed,
        outputs={"row_counts": {"pfam": row_count}},
        resolved_paths=resolved_paths,
        database_versions=database_versions,
        manifest_filename=f"{stem}.build_manifest.json",
    )
    logger.info("Saved build manifest to %s", manifest_path)


if __name__ == "__main__":
    main(parse_arguments(sys.argv[1:]))

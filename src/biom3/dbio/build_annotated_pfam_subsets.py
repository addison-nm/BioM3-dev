"""Build annotated per-family CSVs from Pfam-A.full.gz (the non-NR source).

Unlike biom3_build_source_pfam which reads the 90%-non-redundant
Pfam-A.fasta.gz and produces a whole-DB CSV, this extractor streams the
full Stockholm alignment file (Pfam-A.full.gz) and emits only the
requested Pfam families. For PF00018 (SH3) this is ~176K rows — 6.7x
more than Pfam-A.fasta's 26K — because Pfam-A.fasta is clustered at 90%
identity while Pfam-A.full contains every reference-proteome hit.

Usage:
    biom3_build_annotated_pfam_subsets \\
        -p PF00018 PF07714 \\
        --pfam_full data/databases/pfam/Pfam-A.full.gz \\
        -o outputs/SH3_kinase_full.csv
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
from biom3.dbio.build_source_pfam import PFAM_SPEC
from biom3.dbio.caption import compose_row_caption
from biom3.dbio.pfam_metadata import PfamMetadataParser
from biom3.dbio.stats import IncrementalStatsBuilder, write_stats_markdown

logger = setup_logger(__name__)

OUTPUT_COLUMNS = [
    "id",
    "range",
    "pfam_label",
    "sequence",
    "family_name",
    "family_description",
    "family_type",
    "family_clan",
    "family_wikipedia",
    "family_references",
    "[final]text_caption",
]

_GAP_TRANS = str.maketrans("", "", "-.")


def _clean_sequence(raw):
    """Strip Stockholm gap chars (`.` and `-`) and uppercase the residues."""
    return raw.translate(_GAP_TRANS).upper()


def iter_annotated_pfam_rows(full_gz_path, target_pfam_ids):
    """Yield row dicts for sequences in target families from Pfam-A.full.gz.

    Args:
        full_gz_path: path to Pfam-A.full.gz (Stockholm format, gzipped).
        target_pfam_ids: iterable of unversioned Pfam IDs (e.g. ["PF00018"]).

    Yields:
        dict with keys matching OUTPUT_COLUMNS except `[final]text_caption`
        (the caller composes that). Per-row keys: id, range, pfam_label,
        sequence, family_name, family_description, family_type, family_clan,
        family_wikipedia, family_references.
    """
    target_set = {str(p).strip() for p in target_pfam_ids if str(p).strip()}
    remaining = set(target_set)
    hits = {p: 0 for p in target_set}

    opener = gzip.open if full_gz_path.endswith(".gz") else open

    state = PfamMetadataParser._new_state()
    current_pfam_id = None
    in_target = False
    gs_to_acc = {}
    pending_sequences = []

    with opener(full_gz_path, "rt") as f:
        for line in tqdm(f, desc="Scanning Pfam-A.full", unit=" lines"):
            if line.startswith("#=GF "):
                tag = line[5:10].rstrip()
                value = line[10:].strip() if len(line) > 10 else ""

                if tag == "ID":
                    state["short_id"] = value
                elif tag == "AC":
                    state["accession"] = value.rstrip(";").strip()
                    current_pfam_id = state["accession"].split(".")[0]
                    in_target = current_pfam_id in target_set
                elif tag == "DE":
                    state["description"] = value
                elif tag == "CC":
                    state["cc_lines"].append(value)
                elif tag == "TP":
                    state["family_type"] = value
                elif tag == "CL":
                    state["family_clan"] = value.rstrip(";").strip()
                elif tag == "WK":
                    state["family_wikipedia"] = value.rstrip(";").strip()
                elif tag == "RT":
                    state["rt_lines"].append(value)
                continue

            if line.startswith("//"):
                if in_target and current_pfam_id in remaining:
                    meta_bucket = {}
                    PfamMetadataParser._finalize_family(meta_bucket, state)
                    meta = meta_bucket.get(current_pfam_id, {})
                    for entry_name_range, raw_seq in pending_sequences:
                        _, _, range_part = entry_name_range.partition("/")
                        accession = gs_to_acc.get(entry_name_range, "")
                        yield {
                            "id": accession,
                            "range": range_part,
                            "pfam_label": current_pfam_id,
                            "sequence": _clean_sequence(raw_seq),
                            "family_name": meta.get("family_name", ""),
                            "family_description": meta.get("family_description", ""),
                            "family_type": meta.get("family_type", ""),
                            "family_clan": meta.get("family_clan", ""),
                            "family_wikipedia": meta.get("family_wikipedia", ""),
                            "family_references": meta.get("family_references", ""),
                        }
                        hits[current_pfam_id] += 1
                    remaining.discard(current_pfam_id)

                state = PfamMetadataParser._new_state()
                current_pfam_id = None
                in_target = False
                gs_to_acc = {}
                pending_sequences = []

                if not remaining:
                    break
                continue

            if not in_target:
                continue

            if line.startswith("#=GS "):
                parts = line[5:].split(None, 2)
                if len(parts) >= 3 and parts[1] == "AC":
                    gs_to_acc[parts[0]] = parts[2].split(".")[0].rstrip(";").strip()
                continue

            if line.startswith("#"):
                continue

            stripped = line.rstrip("\n").rstrip()
            if not stripped:
                continue
            parts = stripped.split(None, 1)
            if len(parts) == 2:
                pending_sequences.append((parts[0], parts[1]))

    for pfam_id, count in hits.items():
        if count == 0:
            logger.warning("No rows extracted for requested Pfam ID %s "
                           "(missing from %s or typo?)", pfam_id, full_gz_path)


def build_annotated_pfam_subsets_csv(full_gz_path, target_pfam_ids,
                                     output_path, caption_spec=None,
                                     chunk_size=10_000):
    """Build the annotated per-family CSV and return (row_count, stats_builder)."""
    if caption_spec is None:
        caption_spec = PFAM_SPEC

    annotation_fields = [field for (_, field) in caption_spec.fields]
    stats_builder = IncrementalStatsBuilder(
        annotation_fields=annotation_fields,
        seq_field="sequence",
        pfam_field="pfam_ids",
        caption_field="caption",
    )

    buffer = []
    total = 0

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(OUTPUT_COLUMNS)

        for row in iter_annotated_pfam_rows(full_gz_path, target_pfam_ids):
            caption = compose_row_caption(
                {
                    "family_name": row["family_name"],
                    "family_description": row["family_description"],
                },
                caption_spec,
            )

            buffer.append([
                row["id"],
                row["range"],
                row["pfam_label"],
                row["sequence"],
                row["family_name"],
                row["family_description"],
                row["family_type"],
                row["family_clan"],
                row["family_wikipedia"],
                row["family_references"],
                caption,
            ])

            stats_builder.update({
                "sequence": row["sequence"],
                "pfam_ids": [row["pfam_label"]] if row["pfam_label"] else [],
                "caption": caption,
                **{f: row.get(f, "") for f in annotation_fields},
            })

            if len(buffer) >= chunk_size:
                writer.writerows(buffer)
                total += len(buffer)
                buffer = []
                logger.info("Written %s rows", f"{total:,}")

        if buffer:
            writer.writerows(buffer)
            total += len(buffer)

    logger.info("Build complete: %s rows written to %s", f"{total:,}", output_path)
    return total, stats_builder


def _read_release_version(filepath, pattern):
    try:
        with open(filepath) as f:
            text = f.read(2048)
        match = re.search(pattern, text)
        return match.group(1).strip() if match else None
    except Exception:
        return None


def _collect_database_versions(full_gz_path):
    versions = {"pfam_full": get_file_metadata(full_gz_path)}
    pfam_dir = os.path.dirname(os.path.abspath(full_gz_path))
    relnotes_path = os.path.join(pfam_dir, "relnotes.txt")
    if os.path.exists(relnotes_path):
        versions["pfam_release"] = _read_release_version(
            relnotes_path, r"RELEASE\s+(\S+)",
        )
    return versions


def parse_arguments(args):
    parser = argparse.ArgumentParser(
        description="Build annotated per-family CSVs from Pfam-A.full.gz."
    )
    parser.add_argument(
        "-p", "--pfam_ids", nargs="+", required=True,
        help="One or more Pfam IDs to extract (e.g. PF00018 PF07714)",
    )
    parser.add_argument(
        "--pfam_full", type=str, required=True,
        help="Path to Pfam-A.full.gz (Stockholm format, gzipped)",
    )
    parser.add_argument(
        "-o", "--output", type=str, required=True,
        help="Output CSV path",
    )
    parser.add_argument(
        "--chunk_size", type=int, default=10_000,
        help="Rows to buffer before writing (default: 10000)",
    )
    return parser.parse_args(args)


def main(args):
    start_time = datetime.now()

    target_ids = [str(p).strip() for p in args.pfam_ids if str(p).strip()]
    logger.info("Extracting %s Pfam families from %s", len(target_ids), args.pfam_full)

    row_count, stats_builder = build_annotated_pfam_subsets_csv(
        full_gz_path=args.pfam_full,
        target_pfam_ids=target_ids,
        output_path=args.output,
        chunk_size=args.chunk_size,
    )

    elapsed = datetime.now() - start_time
    stats = stats_builder.finalize()

    outdir = os.path.dirname(os.path.abspath(args.output))
    stem = os.path.splitext(os.path.basename(args.output))[0]

    stats_path = os.path.join(outdir, f"{stem}.stats.md")
    write_stats_markdown(stats, stats_path, title=f"{stem} — coverage stats")
    logger.info("Saved stats report to %s", stats_path)

    database_versions = _collect_database_versions(args.pfam_full)
    resolved_paths = {
        "pfam_full": os.path.abspath(args.pfam_full),
        "output": os.path.abspath(args.output),
        "stats_markdown": stats_path,
    }
    manifest_path = write_manifest(
        args, outdir, start_time, elapsed,
        outputs={
            "row_counts": {"pfam_annotated_subsets": row_count},
            "pfam_ids": target_ids,
        },
        resolved_paths=resolved_paths,
        database_versions=database_versions,
        stats=stats,
        manifest_filename=f"{stem}.build_manifest.json",
    )
    logger.info("Saved build manifest to %s", manifest_path)


if __name__ == "__main__":
    main(parse_arguments(sys.argv[1:]))

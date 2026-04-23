"""Build brenda_kinetics.csv from the BRENDA flatfile.

Emits one row per (EC, organism) with EC-level fields (name, synonyms,
reactions) shared across rows and per-organism fields (substrates,
kinetics, pH, temperature) bucketed.

Usage:
    biom3_build_source_brenda \\
        --input data/databases/brenda/brenda_2026_1.txt \\
        -o data/datasets/brenda_kinetics.csv
"""

import argparse
import csv
import os
import re
import sys
from datetime import datetime

import pandas as pd

from biom3.backend.device import setup_logger
from biom3.core.run_utils import get_file_metadata, write_manifest
from biom3.dbio.brenda import BrendaParser
from biom3.dbio.caption import CaptionSpec, compose_row_caption
from biom3.dbio.stats import compute_coverage_stats, write_stats_markdown

logger = setup_logger(__name__)

BRENDA_SPEC = CaptionSpec(
    fields=[
        ("ENZYME NAME",             "annot_recommended_name"),
        ("SYSTEMATIC NAME",         "annot_systematic_name"),
        ("SYNONYMS",                "annot_synonyms"),
        ("ORGANISM",                "annot_organism"),
        ("REACTIONS",               "annot_reactions"),
        ("SUBSTRATES AND PRODUCTS", "annot_substrates_products"),
        ("KM VALUES",               "annot_km_values"),
        ("PH OPTIMUM",              "annot_ph_optimum"),
        ("TEMPERATURE OPTIMUM",     "annot_temperature_optimum"),
    ],
    strip_pubmed=False,
    strip_evidence=False,
    family_names_label=None,
)

OUTPUT_COLUMNS = [
    "ec",
    "organism",
    "annot_organism_details",
    "annot_recommended_name",
    "annot_systematic_name",
    "annot_synonyms",
    "annot_reactions",
    "annot_substrates_products",
    "annot_km_values",
    "annot_ph_optimum",
    "annot_temperature_optimum",
    "[final]text_caption",
]

# Per-field caps control how much BRENDA text ends up in a single
# (EC, organism) row. Two limits: max number of records and max total
# characters. Captions are for BioBERT tokenization (~512 tokens), not
# exhaustive data dumps, so these stay tight by default.
DEFAULT_CAPS = {
    "reactions": {"count": 3, "chars": 2000},
    "synonyms": {"count": 15, "chars": 800},
    "substrates_products": {"count": 10, "chars": 1500},
    "km_values": {"count": 10, "chars": 1000},
    "ph_optimum": {"count": 5, "chars": 400},
    "temperature_optimum": {"count": 5, "chars": 400},
}


def _truncate_join(values, count_cap=10, char_cap=1500, sep=" | "):
    """Join *values* with *sep*, capped both by count and cumulative length.

    Individual records longer than *char_cap* are truncated mid-string so a
    single huge record can't blow past the budget.
    """
    if not values:
        return ""
    kept = []
    running = 0
    for v in values[:count_cap]:
        s = str(v)
        prefix_len = len(sep) if kept else 0
        remaining = char_cap - running - prefix_len
        if remaining <= 0:
            break
        if len(s) > remaining:
            kept.append(s[:remaining] + "…")
            running = char_cap
            break
        kept.append(s)
        running += prefix_len + len(s)
    omitted = len(values) - len(kept)
    text = sep.join(kept)
    if omitted > 0:
        suffix = f"{sep}... ({omitted} more)"
        text = text[: char_cap - len(suffix)] + suffix
    return text


def _cap(field):
    c = DEFAULT_CAPS[field]
    return {"count_cap": c["count"], "char_cap": c["chars"]}


def build_brenda_csv(input_path, output_path, caption_spec=None,
                     chunk_size=10_000):
    """Parse BRENDA flatfile and write brenda_kinetics.csv.

    Returns:
        int: number of rows written.
    """
    if caption_spec is None:
        caption_spec = BRENDA_SPEC

    parser = BrendaParser(input_path)
    total = 0
    buffer = []
    all_rows = []

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(OUTPUT_COLUMNS)

        for entry in parser.iter_entries():
            reactions_text = _truncate_join(entry.reactions, **_cap("reactions"))
            synonyms_text = _truncate_join(
                entry.synonyms, sep="; ", **_cap("synonyms"),
            )

            for org_num, org in entry.organisms.items():
                annotations = {
                    "annot_recommended_name": entry.recommended_name,
                    "annot_systematic_name": entry.systematic_name,
                    "annot_synonyms": synonyms_text,
                    "annot_organism": org.name,
                    "annot_reactions": reactions_text,
                    "annot_substrates_products": _truncate_join(
                        entry.substrates_products.get(org_num, []),
                        **_cap("substrates_products"),
                    ),
                    "annot_km_values": _truncate_join(
                        entry.km_values.get(org_num, []),
                        **_cap("km_values"),
                    ),
                    "annot_ph_optimum": _truncate_join(
                        entry.ph_optimum.get(org_num, []),
                        **_cap("ph_optimum"),
                    ),
                    "annot_temperature_optimum": _truncate_join(
                        entry.temperature_optimum.get(org_num, []),
                        **_cap("temperature_optimum"),
                    ),
                }
                caption = compose_row_caption(annotations, caption_spec)
                row_dict = {
                    "ec": entry.ec,
                    "organism": org.name,
                    "annot_organism_details": org.details,
                    **{k: v for k, v in annotations.items() if k != "annot_organism"},
                    "[final]text_caption": caption,
                }
                row = [row_dict[c] for c in OUTPUT_COLUMNS]
                buffer.append(row)
                all_rows.append(row_dict)

            if len(buffer) >= chunk_size:
                writer.writerows(buffer)
                total += len(buffer)
                buffer = []
                logger.info("Written %s rows", f"{total:,}")

        if buffer:
            writer.writerows(buffer)
            total += len(buffer)

    logger.info("Build complete: %s rows written to %s", f"{total:,}", output_path)
    df = pd.DataFrame(all_rows, columns=OUTPUT_COLUMNS)
    return total, df


def _read_release_version(input_path):
    with open(input_path) as fh:
        head = fh.read(200)
    match = re.search(r"^BR\s+(\S+)", head, re.MULTILINE)
    return match.group(1).strip() if match else None


def parse_arguments(args):
    parser = argparse.ArgumentParser(
        description="Build brenda_kinetics.csv from BRENDA flatfile."
    )
    parser.add_argument(
        "--input", type=str, required=True,
        help="Path to data/databases/brenda/brenda_<version>.txt",
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

    row_count, df = build_brenda_csv(
        input_path=args.input,
        output_path=args.output,
        chunk_size=args.chunk_size,
    )

    elapsed = datetime.now() - start_time

    stats = compute_coverage_stats(df)
    stats["brenda"] = {
        "distinct_ecs": int(df["ec"].nunique()) if len(df) else 0,
        "distinct_organisms": int(df["organism"].nunique()) if len(df) else 0,
        "mean_organisms_per_ec": round(
            len(df) / max(1, df["ec"].nunique()), 2
        ) if len(df) else 0.0,
    }

    outdir = os.path.dirname(os.path.abspath(args.output))
    stem = os.path.splitext(os.path.basename(args.output))[0]

    stats_path = os.path.join(outdir, f"{stem}.stats.md")
    write_stats_markdown(stats, stats_path, title=f"{stem} — coverage stats")
    logger.info("Saved stats report to %s", stats_path)

    resolved_paths = {
        "input": os.path.abspath(args.input),
        "output": os.path.abspath(args.output),
        "stats_markdown": stats_path,
    }
    manifest_path = write_manifest(
        args, outdir, start_time, elapsed,
        outputs={"row_counts": {"brenda": row_count}},
        resolved_paths=resolved_paths,
        database_versions={
            "brenda_flatfile": get_file_metadata(args.input),
            "brenda_release": _read_release_version(args.input),
        },
        stats=stats,
        manifest_filename=f"{stem}.build_manifest.json",
    )
    logger.info("Saved build manifest to %s", manifest_path)


if __name__ == "__main__":
    main(parse_arguments(sys.argv[1:]))

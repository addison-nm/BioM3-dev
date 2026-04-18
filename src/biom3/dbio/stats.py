"""Coverage statistics for dbio-built CSVs.

Every source builder and build_dataset invocation calls this module to compute
a structured stats dict, which is:

1. Pretty-printed to `<stem>.stats.md` beside the output CSV.
2. Embedded in `<stem>.build_manifest.json` under a `stats` key.

The dict shape is stable (append-only) so downstream tooling can diff stats
across builds to spot regressions in annotation coverage.
"""

from __future__ import annotations

import ast
import statistics
from collections import Counter


_EMPTY_SENTINELS = {"", "nan", "none", "null", "[]", "['nan']"}


def _is_populated(value):
    """Return True if *value* represents a meaningful, non-empty annotation."""
    if value is None:
        return False
    try:
        if value != value:
            return False
    except TypeError:
        pass
    s = str(value).strip().lower()
    if not s:
        return False
    return s not in _EMPTY_SENTINELS


def _sequence_length_stats(df, seq_col="protein_sequence"):
    if seq_col not in df.columns:
        return None
    lengths = [len(s) for s in df[seq_col].dropna().astype(str) if s]
    if not lengths:
        return None
    lengths_sorted = sorted(lengths)
    n = len(lengths_sorted)
    return {
        "min": lengths_sorted[0],
        "mean": round(statistics.mean(lengths_sorted), 1),
        "median": int(statistics.median(lengths_sorted)),
        "p95": lengths_sorted[min(n - 1, int(0.95 * n))],
        "max": lengths_sorted[-1],
    }


def _annotation_coverage(df, prefix="annot_"):
    """Per-column coverage for every `annot_*` column in *df*."""
    cols = [c for c in df.columns if c.startswith(prefix)]
    total = len(df)
    result = {}
    for col in cols:
        populated_mask = df[col].map(_is_populated)
        populated_values = df.loc[populated_mask, col].astype(str)
        populated_count = int(populated_mask.sum())
        if populated_count:
            char_lens = populated_values.map(len)
            mean_chars = round(float(char_lens.mean()), 1)
            max_chars = int(char_lens.max())
        else:
            mean_chars = 0.0
            max_chars = 0
        result[col] = {
            "populated": populated_count,
            "percentage": round(100 * populated_count / total, 2) if total else 0.0,
            "mean_chars": mean_chars,
            "max_chars": max_chars,
        }
    return result


def _parse_pfam_label(raw):
    """Extract Pfam IDs from a row's pfam_label cell (single ID, list string, or list)."""
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(x) for x in raw if _is_populated(x)]
    s = str(raw).strip()
    if not s or s.lower() in _EMPTY_SENTINELS:
        return []
    if s.startswith("[") and s.endswith("]"):
        try:
            parsed = ast.literal_eval(s)
        except (ValueError, SyntaxError):
            return [s]
        if isinstance(parsed, (list, tuple)):
            return [str(x) for x in parsed if _is_populated(x)]
        return [str(parsed)]
    return [s]


def _pfam_stats(df, pfam_col="pfam_label", top_n=10):
    if pfam_col not in df.columns:
        return None
    counter = Counter()
    rows_with_pfam = 0
    for raw in df[pfam_col]:
        ids = _parse_pfam_label(raw)
        if ids:
            rows_with_pfam += 1
            counter.update(ids)
    return {
        "rows_with_pfam": rows_with_pfam,
        "distinct_families": len(counter),
        "top": [
            {"pfam_id": pid, "rows": n}
            for pid, n in counter.most_common(top_n)
        ],
    }


def _source_breakdown(df, source_col):
    if source_col not in df.columns:
        return None
    counts = df[source_col].value_counts().to_dict()
    return {str(k): int(v) for k, v in counts.items()}


def compute_coverage_stats(df, source_col=None, join_metadata=None,
                           seq_col="protein_sequence", pfam_col="pfam_label"):
    """Return a structured stats dict describing coverage of *df*.

    Args:
        df: pandas DataFrame with dbio-canonical columns.
        source_col: if given, column whose values identify the source DB
            (e.g. "swissprot" vs "pfam" vs "trembl"). Used to break row
            counts down by source in the aggregate manifest.
        join_metadata: if given, dict of already-computed join hit-rate
            stats from enrich.py, merged verbatim under `joins`.
        seq_col: column holding the protein sequence (default "protein_sequence").
        pfam_col: column holding Pfam labels (default "pfam_label").

    Returns:
        dict with keys: row_count, sequence_length, annotation_coverage,
        pfam_label, sources (optional), joins (optional).
    """
    stats = {
        "row_count": int(len(df)),
        "sequence_length": _sequence_length_stats(df, seq_col=seq_col),
        "annotation_coverage": _annotation_coverage(df),
        "pfam_label": _pfam_stats(df, pfam_col=pfam_col),
    }
    sources = _source_breakdown(df, source_col) if source_col else None
    if sources is not None:
        stats["sources"] = sources
    if join_metadata:
        stats["joins"] = join_metadata
    return stats


def format_stats_markdown(stats, title):
    """Render *stats* as a human-readable markdown string."""
    lines = [f"# {title}", ""]
    lines.append(f"**Rows:** {stats['row_count']:,}")
    lines.append("")

    seq = stats.get("sequence_length")
    if seq:
        lines.append("## Sequence length (residues)")
        lines.append("")
        lines.append("| min | mean | median | p95 | max |")
        lines.append("|-----|------|--------|-----|-----|")
        lines.append(
            f"| {seq['min']:,} | {seq['mean']:,} | {seq['median']:,} "
            f"| {seq['p95']:,} | {seq['max']:,} |"
        )
        lines.append("")

    sources = stats.get("sources")
    if sources:
        lines.append("## Rows by source")
        lines.append("")
        lines.append("| source | rows |")
        lines.append("|--------|------|")
        for src, count in sorted(sources.items(), key=lambda kv: -kv[1]):
            lines.append(f"| {src} | {count:,} |")
        lines.append("")

    caption = stats.get("caption_coverage")
    if caption:
        lines.append("## Caption coverage")
        lines.append("")
        lines.append(
            f"- Populated: {caption['populated']:,} "
            f"({caption['percentage']:.1f}%)"
        )
        lines.append(f"- Mean caption length: {caption['mean_chars']:,} chars")
        lines.append("")

    coverage = stats.get("annotation_coverage") or {}
    if coverage:
        lines.append("## Annotation coverage")
        lines.append("")
        lines.append("| column | populated | % | mean chars | max chars |")
        lines.append("|--------|-----------|---|------------|-----------|")
        for col, info in sorted(coverage.items(), key=lambda kv: -kv[1]["percentage"]):
            lines.append(
                f"| `{col}` | {info['populated']:,} | {info['percentage']:.1f}% "
                f"| {info['mean_chars']:,} | {info['max_chars']:,} |"
            )
        lines.append("")

    pfam = stats.get("pfam_label")
    if pfam and pfam.get("distinct_families") is not None:
        lines.append("## Pfam family distribution")
        lines.append("")
        lines.append(f"- Rows with Pfam label: {pfam['rows_with_pfam']:,}")
        lines.append(f"- Distinct families: {pfam['distinct_families']:,}")
        if pfam["top"]:
            lines.append("")
            lines.append("Top families by row count:")
            lines.append("")
            lines.append("| pfam_id | rows |")
            lines.append("|---------|------|")
            for entry in pfam["top"]:
                lines.append(f"| {entry['pfam_id']} | {entry['rows']:,} |")
        lines.append("")

    joins = stats.get("joins")
    if joins:
        lines.append("## Enrichment join hit rates")
        lines.append("")
        lines.append("| join | hit rate |")
        lines.append("|------|----------|")
        for name, rate in joins.items():
            if isinstance(rate, (int, float)):
                lines.append(f"| {name} | {rate:.1%} |")
            else:
                lines.append(f"| {name} | {rate} |")
        lines.append("")

    # Render any remaining dict-valued top-level keys (builder-specific extras
    # like uniprot_crosslinks, obsolete, etc.) as generic key-value sections.
    known_keys = {
        "row_count", "sequence_length", "annotation_coverage",
        "caption_coverage", "pfam_label", "sources", "joins",
    }
    for key, value in stats.items():
        if key in known_keys or not isinstance(value, dict) or not value:
            continue
        lines.append(f"## {key.replace('_', ' ').title()}")
        lines.append("")
        lines.append("| field | value |")
        lines.append("|-------|-------|")
        for k, v in value.items():
            if isinstance(v, float):
                lines.append(f"| {k} | {v:,.2f} |")
            elif isinstance(v, int):
                lines.append(f"| {k} | {v:,} |")
            else:
                lines.append(f"| {k} | {v} |")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def write_stats_markdown(stats, output_path, title):
    """Write `format_stats_markdown(stats, title)` to *output_path*."""
    content = format_stats_markdown(stats, title)
    with open(output_path, "w") as f:
        f.write(content)
    return output_path


class IncrementalStatsBuilder:
    """Accumulate coverage stats one row at a time.

    Used by streaming source builders whose output CSVs are too large to
    re-read with pandas. Call `update()` per row and `finalize()` at the end
    to obtain the same dict shape as `compute_coverage_stats()`.
    """

    def __init__(self, annotation_fields=(), seq_field="sequence",
                 pfam_field="pfam_ids", caption_field="caption"):
        self.annotation_fields = list(annotation_fields)
        self.seq_field = seq_field
        self.pfam_field = pfam_field
        self.caption_field = caption_field

        self.row_count = 0
        self.seq_lengths = []
        self.populated = {f: 0 for f in self.annotation_fields}
        self.char_sums = {f: 0 for f in self.annotation_fields}
        self.char_maxes = {f: 0 for f in self.annotation_fields}
        self.pfam_counter = Counter()
        self.rows_with_pfam = 0
        self.caption_populated = 0
        self.caption_char_sum = 0

    def update(self, row):
        """Ingest one row (dict-like). Keys are field names; values may be None."""
        self.row_count += 1

        seq = row.get(self.seq_field)
        if seq:
            self.seq_lengths.append(len(str(seq)))

        for field in self.annotation_fields:
            val = row.get(field)
            if _is_populated(val):
                self.populated[field] += 1
                vlen = len(str(val))
                self.char_sums[field] += vlen
                if vlen > self.char_maxes[field]:
                    self.char_maxes[field] = vlen

        pfam_ids = row.get(self.pfam_field)
        if isinstance(pfam_ids, (list, tuple)):
            ids = [str(p) for p in pfam_ids if _is_populated(p)]
        else:
            ids = _parse_pfam_label(pfam_ids)
        if ids:
            self.rows_with_pfam += 1
            self.pfam_counter.update(ids)

        caption = row.get(self.caption_field)
        if _is_populated(caption):
            self.caption_populated += 1
            self.caption_char_sum += len(str(caption))

    def finalize(self, top_n=10, annotation_prefix="annot_"):
        """Return the structured stats dict."""
        total = self.row_count

        if self.seq_lengths:
            lengths_sorted = sorted(self.seq_lengths)
            n = len(lengths_sorted)
            seq_stats = {
                "min": lengths_sorted[0],
                "mean": round(statistics.mean(lengths_sorted), 1),
                "median": int(statistics.median(lengths_sorted)),
                "p95": lengths_sorted[min(n - 1, int(0.95 * n))],
                "max": lengths_sorted[-1],
            }
        else:
            seq_stats = None

        annotation_coverage = {}
        for field in self.annotation_fields:
            pop = self.populated[field]
            col_name = field if field.startswith(annotation_prefix) else annotation_prefix + field
            annotation_coverage[col_name] = {
                "populated": pop,
                "percentage": round(100 * pop / total, 2) if total else 0.0,
                "mean_chars": round(self.char_sums[field] / pop, 1) if pop else 0.0,
                "max_chars": self.char_maxes[field],
            }

        pfam_stats = {
            "rows_with_pfam": self.rows_with_pfam,
            "distinct_families": len(self.pfam_counter),
            "top": [
                {"pfam_id": pid, "rows": n}
                for pid, n in self.pfam_counter.most_common(top_n)
            ],
        }

        caption_coverage = {
            "populated": self.caption_populated,
            "percentage": round(100 * self.caption_populated / total, 2) if total else 0.0,
            "mean_chars": round(self.caption_char_sum / self.caption_populated, 1)
                if self.caption_populated else 0.0,
        }

        return {
            "row_count": total,
            "sequence_length": seq_stats,
            "annotation_coverage": annotation_coverage,
            "caption_coverage": caption_coverage,
            "pfam_label": pfam_stats,
        }

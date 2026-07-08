# `biom3.dbio` fragility audit

**Date:** 2026-05-17
**Scope:** the full `biom3.dbio` subpackage — parsers, builders,
readers, the `build_dataset` orchestrator, enrichment/caption layer,
stats, and supporting taxonomy/cross-reference code.
**Reviewer:** adversarial read for *robustness*, not structure.

## How this differs from the existing audit

[dbio_package_audit.md](dbio_package_audit.md) is a *structural* map:
what each file does, how the layers stack, where the
backward-compat surfaces are. It is accurate and worth keeping, but it
is written from the inside and is largely affirmative ("clean
layering", "provenance everywhere", "backward-compat surfaces are
explicit"). It does not stress-test the data path.

This document is the complement: where does the pipeline silently
produce wrong or inconsistent data, and why. The conclusion up front:

> The layer **separation** is genuinely good and is not the problem.
> The fragility is concentrated in two places the structural audit
> does not examine: **(1) the on-disk data contract between layers is
> stringly-typed CSV with hand-rolled, non-equivalent encoders and
> decoders, and (2) every parser treats its input as well-formed and
> has no fail-loud path** — so corruption, format drift, and schema
> mismatch all surface as plausible-looking data with a clean
> provenance manifest, not as errors.

Severity legend:

- **Critical** — silently produces wrong training data that looks
  correct and ships with a clean manifest.
- **High** — silent inconsistency or correctness-by-accident; works
  today only because of an unstated assumption.
- **Medium** — real fragility that will bite on format drift, scale,
  or refactor, but is currently contained.
- **Low** — code rot / asymmetry worth fixing but low blast radius.

---

## Critical

### C1. `pigz` decompression failures are undetectable

[`parsers/swissprot_dat.py:47-70`](../../src/biom3/dbio/parsers/swissprot_dat.py#L47-L70)

`_open_gz()` shells out to `pigz -dc` with
`stderr=subprocess.DEVNULL`, returns `proc.stdout`, and **discards the
`Popen` handle**. Callers (`parse()`, `parse_all()`) iterate
`proc.stdout` and `finally: fh.close()`. Nothing anywhere can check
`proc.returncode` — the handle is gone — and pigz's own error text is
routed to `/dev/null`.

If `pigz` dies mid-stream (corrupt `.gz`, disk full, OOM kill — all
plausible on the 161 GB TrEMBL file this optimization exists for), the
pipe closes at EOF and iteration ends **normally**. The parser reports
"Bulk parse complete: N entries yielded" with a short N, the builder
writes a truncated CSV / Parquet cache, `write_manifest` stamps it
with a git hash and row count, and `stats.md` reports clean coverage
on the rows that *did* parse. There is no signal that the dataset is
partial.

Note the asymmetry: the `gzip.open` fallback path *would* raise
`BadGzipFile`/`EOFError` on a truncated stream. The performance
optimization is what introduced the silent-failure mode. A truncated
training set that looks complete is the single highest-consequence
failure in the package because every downstream stage trusts the
manifest.

**Failure scenario:** TrEMBL annotation-cache build is OOM-killed at
60% on a busy node. pigz is reaped; Python sees EOF. A 60%-complete
`trembl_annotations.parquet` ships. Every later `--annotation_cache`
run gets silent 40% enrichment misses, indistinguishable from
"protein genuinely has no annotation."

---

## High

### H1. `pfam_label` has two on-disk encodings and four decoders

This is the central ad-hoc data-modeling problem; everything else in
the storage layer is downstream of it.

**Producers disagree on the encoding of the same column name:**

| Builder | `pfam_label` cell | Source |
|---|---|---|
| SwissProt | `"['PF00018', 'PF00169']"` (Python `repr` of a list); `"['nan']"` for none | [`source_swissprot.py:97-107`](../../src/biom3/dbio/builders/source_swissprot.py#L97-L107) |
| Pfam (fasta) | `"PF00018"` (bare scalar) | [`source_pfam.py:78`](../../src/biom3/dbio/builders/source_pfam.py#L78) |
| Pfam subsets | `"PF00018"` (bare scalar) | [`pfam_subsets.py:124`](../../src/biom3/dbio/builders/pfam_subsets.py#L124) |

**Consumers each decode it a different, non-equivalent way:**

1. [`swissprot_csv.py:79-82`](../../src/biom3/dbio/readers/swissprot_csv.py#L79-L82)
   — `df["pfam_label"].str.contains("|".join(pfam_ids))`. Unescaped
   regex **substring** match against the list-repr string. Querying
   `PF0001` would substring-match `PF00018`, `PF00019`, …; it works
   *only* because Pfam IDs are fixed-width `PF\d{5}` so no requested ID
   is a prefix of another. Correctness by accident of the identifier
   format. Also a latent regex-injection surface if an ID ever
   contained a metacharacter.
2. [`pfam_csv.py:106`](../../src/biom3/dbio/readers/pfam_csv.py#L106) /
   [`pfam_csv.py:81-84`](../../src/biom3/dbio/readers/pfam_csv.py#L81-L84)
   — exact `.isin()` / Parquet `("pfam_label","in",…)`. Assumes a
   single bare scalar per cell. Would never match a list-repr cell.
3. [`build_dataset.py:39-48`](../../src/biom3/dbio/pipelines/build_dataset.py#L39-L48)
   — `re.findall(r"PF\d{5}")`. A third semantics, used only on the
   `--per_pfam_output` split path.
4. [`stats.py:80-97`](../../src/biom3/dbio/stats.py#L80-L97)
   `_parse_pfam_label` — the *most* tolerant: handles scalar,
   list-repr (via `ast.literal_eval`), and real list. **The existence
   of this fourth, robust decoder in the reporting layer is the smoking
   gun: the team already knows the encoding is ambiguous and wrote a
   tolerant parser — but only for stats, never for the read path.**

**Concrete consequence, not theoretical:** `build_dataset.main()`
concatenates `df_sp` (list-repr `pfam_label`) and `df_pfam` (bare
scalar `pfam_label`) into one frame
([`build_dataset.py:499-502`](../../src/biom3/dbio/pipelines/build_dataset.py#L499-L502)).
The shipped `dataset.csv` therefore has a `pfam_label` column with
**two different encodings interleaved by row**. The non-`per_pfam`
path never normalizes it. Any downstream consumer that does the
obvious thing (`== "PF00018"` or `.isin`) silently sees only the Pfam
rows and drops every SwissProt row for that family.

**Bonus:** the null sentinel is triple-encoded: `None` → `["nan"]` →
`repr` → the literal string `"['nan']"`, which then needs a dedicated
entry in `stats._EMPTY_SENTINELS`
([`stats.py:20`](../../src/biom3/dbio/stats.py#L20)). A stringified
list containing the string `"nan"` is a recognized empty value. That
sentinel leaks across three layers (builder → CSV → stats).

### H2. The two `.dat` entry parsers extract different annotation sets

[`parsers/swissprot_dat.py:209-357`](../../src/biom3/dbio/parsers/swissprot_dat.py#L209-L357)
(`_parse_entry`, the targeted/enrichment path) vs
[`:460-628`](../../src/biom3/dbio/parsers/swissprot_dat.py#L460-L628)
(`_parse_entry_full`, the bulk/build path).

These are ~120 lines of hand-duplicated CC-block logic that must be
kept in sync manually, and **they are already out of sync**:
`_parse_entry` does not map `TISSUE SPECIFICITY`,
`DEVELOPMENTAL STAGE`, or `BIOTECHNOLOGY`
([`:317-329`](../../src/biom3/dbio/parsers/swissprot_dat.py#L317-L329)),
while `_parse_entry_full` does
([`:597-604`](../../src/biom3/dbio/parsers/swissprot_dat.py#L597-L604)).

Effect: the *same protein* gets a different annotation set depending
on which path touched it. A row enriched at orchestrator time via
`--uniprot_dat` (→ `parse()` → `_parse_entry`) is missing three fields
that the same accession would have if it had come through
`biom3_build_source_swissprot` (→ `parse_all()` → `_parse_entry_full`)
or the annotation cache (built via `_parse_entry_full`,
[`annotation_cache.py:82`](../../src/biom3/dbio/helpers/annotation_cache.py#L82)).
Enrichment quality is a function of *which CLI path* you took, not of
the data. This is silent and untested.

The same duplication produces the EC-merge dance three times
([`:339-346`](../../src/biom3/dbio/parsers/swissprot_dat.py#L339-L346),
[`:419-428`](../../src/biom3/dbio/parsers/swissprot_dat.py#L419-L428),
[`:611-618`](../../src/biom3/dbio/parsers/swissprot_dat.py#L611-L618)).

### H3. `parse()` only matches the first accession token; misses are invisible

[`parsers/swissprot_dat.py:129-134`](../../src/biom3/dbio/parsers/swissprot_dat.py#L129-L134)

Targeted enrichment matches an entry by `raw_line[5:].split(b";",1)[0]`
of the **first** `AC` line only, gated by `if current_acc is None`.
UniProt entries routinely carry secondary accessions and multiple `AC`
lines. A request for a secondary accession never matches; a request
for an accession that appears only on the second `AC` line never
matches. The miss is reported only as an aggregate count in a log line
([`:155-157`](../../src/biom3/dbio/parsers/swissprot_dat.py#L155-L157)),
and downstream `enrich_dataframe` treats "accession absent from the
result dict" as "protein has no annotations"
([`enrich.py:487-493`](../../src/biom3/dbio/enrich.py#L487-L493)) —
indistinguishable from a genuinely unannotated protein. Coverage looks
lower; no error is raised; the stats report it as a normal coverage
percentage.

### H4. CSV seams have no schema; a column rename becomes silent data loss

The interchange format between every layer is CSV with `csv.writer` on
the producer side and `pd.read_csv` on the consumer side, with column
agreement enforced **nowhere**. The enrichment loaders are written
defensively *because* of this:
[`enrich.py:87-101`](../../src/biom3/dbio/enrich.py#L87-L101),
[`:115-136`](../../src/biom3/dbio/enrich.py#L115-L136),
[`:145-155`](../../src/biom3/dbio/enrich.py#L145-L155) all do
`pd.read_csv(dtype=str).fillna("")` then `row.get("annot_name", "")`.

The `.get(col, "")` default is the trap: rename a column in
`source_expasy.py` (or ship an older source CSV), and the join
produces *all-empty* annotations with **no error**. That empty result
then flows into `_join_expasy`/`_join_brenda`/`_join_smart`, which
faithfully compute a 0% hit rate and write it to `stats["joins"]`
([`enrich.py:291-295`](../../src/biom3/dbio/enrich.py#L291-L295)).
A schema mismatch is thus *recorded as a coverage statistic* rather
than raised. The same pattern lets the four-vs-five-vs-six-vs-seven
column `OUTPUT_COLUMNS*` matrix in
[`source_swissprot.py:62-94`](../../src/biom3/dbio/builders/source_swissprot.py#L62-L94)
plus `OPTIONAL_OUTPUT_COLS`
([`swissprot_csv.py:23-25`](../../src/biom3/dbio/readers/swissprot_csv.py#L23-L25))
"work": readers infer capability by `if c in df.columns`. There is no
schema version stamped *in the data* — only the header row and the
out-of-band manifest. Schema is also literally sniffed from a column
name elsewhere (the Parquet converter detects "the Pfam CSV" by the
presence of `family_description`, per the structural audit). Every new
option multiplies the matrix.

The two source CSVs do not even agree on the primary-key column name:
SwissProt uses `primary_Accession`/`protein_sequence`, Pfam uses
`id`/`sequence`, reconciled only by `COLUMN_MAP` inside one reader
([`pfam_csv.py:20`](../../src/biom3/dbio/readers/pfam_csv.py#L20)).
The canonical name for "the accession" depends on which file you open.

---

## Medium

### M1. Two caption composers that can silently diverge

[`enrich.py:544-573`](../../src/biom3/dbio/enrich.py#L544-L573)
(`compose_caption`, used for Pfam rows in `build_dataset`) builds
`f"{label}: {val}."` joined by `" "` with no PubMed stripping and no
multi-period collapse. [`caption.py:99-135`](../../src/biom3/dbio/caption.py#L99-L135)
(`compose_row_caption`, used by every source builder) applies
`strip_pubmed`/`strip_evidence`, `rstrip(".")`, and a configurable
separator/template. SwissProt rows get a caption from path B baked
into the source CSV; Pfam rows get a caption from path A at
orchestrator time; the two are concatenated into one training file.
The "ALL-CAPS labels matching the paper" invariant is maintained in
two implementations with no shared test asserting they produce
equivalent output for equivalent input. Any future edit to one path
silently desyncs caption formatting *within a single dataset*.

### M2. BRENDA "relaxed" join is quadratic and unguarded

[`enrich.py:338-345`](../../src/biom3/dbio/enrich.py#L338-L345): the
genus fallback iterates the entire `by_ec_org` dict **inside** a
`df.apply(_row_brenda, axis=1)`
([`:374`](../../src/biom3/dbio/enrich.py#L374)) — O(rows ×
brenda_records) Python-level. All three joins (`_join_expasy`,
`_join_brenda`, `_join_smart`) are per-row `df.apply` returning
`pd.Series`; `enrich_dataframe` additionally does two full
`df.iterrows()` passes with scalar `df.at[idx, col]=` assignment
([`enrich.py:487-504`](../../src/biom3/dbio/enrich.py#L487-L504)), the
pandas-pathological access pattern. Contained today only because
`build_dataset` enriches a Pfam *subset*, not the 44–63M-row whole DB —
but `--organism_match relaxed` on a large family (the
`pfam_subsets`-fed path can be ~176K rows for one family) will appear
to hang with no progress signal and no guard.

### M3. Cross-module reach into private parser internals

[`pfam_subsets.py:80,115,135`](../../src/biom3/dbio/builders/pfam_subsets.py#L80-L135)
calls `PfamMetadataParser._new_state()` and
`PfamMetadataParser._finalize_family()` — `@staticmethod` *private*
methods of another module — from a free function, and re-implements
the Stockholm `#=GF` scan inline
([`:88-110`](../../src/biom3/dbio/builders/pfam_subsets.py#L88-L110))
parallel to the canonical one in
[`pfam_stockholm.py:46-89`](../../src/biom3/dbio/parsers/pfam_stockholm.py#L46-L89).
The Stockholm format is thus parsed in two places coupled by the
undocumented shape of a `_new_state()` dict. Any change to that dict
silently breaks `pfam_subsets` with no type contract and (per
`--quick`) likely no failing test until a real `.full.gz` is run.

### M4. Parsing keyed on landmark prose and exhaustive header sets

The `.dat` CC-section terminator is detected by
`text.startswith("-----")` or `startswith("Copyrighted")`
([`swissprot_dat.py:257`](../../src/biom3/dbio/parsers/swissprot_dat.py#L257),
[`:524`](../../src/biom3/dbio/parsers/swissprot_dat.py#L524)) — i.e.
matching UniProt's copyright-footer wording. The BRENDA parser
recognises sections only if the bare line is in a hard-coded
38-entry `SECTION_HEADERS` set
([`brenda_flatfile.py:31-70`](../../src/biom3/dbio/parsers/brenda_flatfile.py#L31-L70));
an unknown/renamed header is silently treated as record continuation,
folding the next section's text into the previous record. None of
these formats are version-pinned or validated against the release
file the manifest already records. Format drift degrades data quietly
rather than failing.

### M5. ExPASy parser drops the final entry on a truncated file

[`expasy_dat.py:64-149`](../../src/biom3/dbio/parsers/expasy_dat.py#L64-L149):
an `EnzymeEntry` is only `yield`ed when a `//` terminator is seen
([`:127-149`](../../src/biom3/dbio/parsers/expasy_dat.py#L127-L149));
there is no post-loop flush. Compare `BrendaParser.iter_entries`,
which *does* flush a trailing entry
([`brenda_flatfile.py:232-234`](../../src/biom3/dbio/parsers/brenda_flatfile.py#L232-L234)).
On a truncated `enzyme.dat` the last record vanishes with no warning;
the count just looks one short. The terminator is also handled as the
last `elif` in a code-dispatch chain rather than checked explicitly,
so the structure obscures the omission.

---

## Low

- **Dead code signalling rot.**
  [`taxonomy.py:173-175`](../../src/biom3/dbio/readers/taxonomy.py#L173-L175)
  computes `opener`/`mode` that are never used (the actual read uses
  `pd.read_csv(..., compression="gzip")`). Harmless, but indicates the
  streaming path has been edited without a close read.
- **`rankedlineage.dmp` load has no per-line resilience.**
  [`taxonomy.py:50-62`](../../src/biom3/dbio/readers/taxonomy.py#L50-L62)
  does `int(fields[0])` with no try/skip; one malformed line aborts the
  whole 2.7M-row load with no row context. (At least this one *does*
  fail loudly — the opposite problem from C1.)
- **Three independent copies of `_read_release_version`**
  (`build_dataset.py`, `source_swissprot.py`, `source_pfam.py`,
  `pfam_subsets.py`), each `except Exception: return None`. Provenance
  version capture fails silently and is duplicated four ways.
- **Unescaped regex from user input.** The `pfam_ids` interpolation in
  [`swissprot_csv.py:79`](../../src/biom3/dbio/readers/swissprot_csv.py#L79)
  (`"|".join(pfam_ids)`) is safe only because of the `PF\d{5}` format
  contract; it is not defended.

---

## Test-coverage gaps that let the above persist

The structural audit notes "245 tests pass under `--quick`." The
fragility above survives that suite because the tests exercise each
layer in isolation on well-formed fixtures. Specifically missing:

- No round-trip test asserting `pfam_label` written by *any* builder
  is decoded equivalently by *every* reader and by
  `stats._parse_pfam_label` (would catch H1).
- No test asserting `_parse_entry` and `_parse_entry_full` yield the
  same annotation keys for the same entry (would catch H2).
- No truncated-input / corrupt-`.gz` fixture for any parser (would
  catch C1, M5).
- No cross-composer equivalence test for `compose_caption` vs
  `compose_row_caption` (would catch M1).
- No schema-mismatch fixture (source CSV with a renamed/missing
  column) asserting the join *raises* rather than reporting 0%
  (would catch H4).

These are the highest-leverage tests to add regardless of whether the
deeper refactor happens, because they convert every "silent" failure
above into a red test.

---

## What is *not* broken (so remediation stays targeted)

- The four-layer separation (`parsers/` → `readers/` → `builders/` →
  `pipelines/`) is sound and should be preserved.
- The provenance sidecar discipline (`build_manifest.json` +
  `stats.md` per output) is genuinely valuable — the problem is that
  it certifies *that a build ran*, not *that the data is complete or
  consistent*. It is a strong foundation to attach validation to, not
  something to replace.
- `config.py` path resolution and the streaming/SQLite taxonomy
  strategy split are reasonable and not implicated.

## Root cause, in one sentence

Every seam between the clean layers is an untyped CSV string with a
bespoke encoder on one side and one-or-more bespoke, non-equivalent
decoders on the other, and no layer is allowed to fail — so format
drift, corruption, and schema mismatch all degrade the data into
something that still parses and still gets a clean manifest.

## Suggested remediation directions (not a plan)

Pointers only — a sequenced plan is a separate deliverable:

1. **Make C1 fail loud first** — it is the cheapest fix with the
   highest consequence: retain the `Popen`, check `returncode` on
   close, surface pigz `stderr`. One function.
2. **One `pfam_label` codec.** A single `encode_pfam_label` /
   `decode_pfam_label` pair (or a normalize-on-read in the base
   reader) used by every producer and consumer, replacing the four
   decoders and the `"['nan']"` sentinel. Pick one on-disk form.
3. **Collapse the two `.dat` parsers** into one entry parser with a
   `want_sequence: bool` flag, eliminating H2's drift class entirely.
4. **A declared schema per artifact** (even a frozen
   `dataclass`/`pa.schema` + a `validate_columns()` that raises) so
   H4's renames become errors, and stamp a schema version into the
   data, not just the manifest.
5. **Add the five regression tests above** before/with any refactor.

## Related docs

- [dbio_package_audit.md](dbio_package_audit.md) — structural map
  (complementary; not superseded).
- [building_datasets_with_dbio.md](building_datasets_with_dbio.md) —
  user-facing CLI usage.
- [training_csv_provenance.md](training_csv_provenance.md) —
  column-level source mapping.
</content>
</invoke>

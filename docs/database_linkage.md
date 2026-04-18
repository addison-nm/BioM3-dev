# Database Linkage — `biom3.dbio` cross-reference reference

This document is the authoritative reference for how every raw database used by `biom3.dbio` connects to every other. Every new source builder, enrichment join, and dataset recipe should consult this file before inventing ad-hoc join logic.

## Scope

Covers two groups:

- **Integrated** — raw files under `data/databases/` with a `build_source_*` script (or concrete plan for one).
- **Planned** — raw files not yet on disk, but whose linkage scheme is locked so future integration is straightforward.

Out of scope: downloaded-only artifacts (BLAST indices, FASTA-only mirrors) that carry no annotation beyond sequences. See [docs/setup_databases.md](setup_databases.md) for download instructions.

## Canonical identifiers

Every join in `biom3.dbio` uses one of these identifier families. Tags match the `annot_*` column naming.

| Identifier family | Format | Primary producer | Also appears in |
|---|---|---|---|
| **UniProt accession** | `[OPQ][0-9][A-Z0-9]{3}[0-9]` or `[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2}` | SwissProt, TrEMBL (`ID` line / `AC` line) | Pfam `.fasta` headers, ExPASy `DR` lines, PDB `DBREF`, SMART search output |
| **Pfam family ID** | `PF[0-9]{5}` (version-stripped) | Pfam (`#=GF AC` in Stockholm) | UniProt `DR Pfam;` lines |
| **EC number** | `N.N.N.N` (may end with `-` for partial) | ExPASy `ID` lines | UniProt `DE EC=`, UniProt `CC CATALYTIC ACTIVITY`, BRENDA `ID` lines |
| **SMART domain accession** | `SM[0-9]{5}` | SMART `SMART_domains.txt` `ACC` column | UniProt `DR SMART;` lines |
| **InterPro accession** | `IPR[0-9]{6}` | InterPro (planned) | UniProt `DR InterPro;` lines |
| **NCBI tax ID** | integer | NCBI taxonomy `nodes.dmp` | UniProt `OX` lines, `prot.accession2taxid.gz` |
| **PDB ID** | `[0-9][A-Z0-9]{3}` + chain (e.g. `1ABC`, `1ABC_A`) | PDB (planned) | UniProt `DR PDB;`, SCOPe / CATH entries |
| **SCOPe concise classifier** | `a.1.1.1`-style hierarchy | SCOPe (planned) | Linked to PDB ID + chain |
| **CATH superfamily code** | `N.N.N.N` (homologous superfamily) | CATH (planned) | Linked to PDB ID + chain |

---

## Integrated databases

### SwissProt (UniProtKB reviewed)

- **Raw file:** `data/databases/swissprot/uniprot_sprot.dat.gz`
- **Version:** `reldate.txt` in same directory; regex `Release\s+(\S+)` via [_collect_database_versions()](src/biom3/dbio/build_source_swissprot.py#L223-L242)
- **Reader / parser:** [SwissProtDatParser](src/biom3/dbio/swissprot_dat.py)
- **Built CSV:** `fully_annotated_swiss_prot.csv` + `fully_annotated_swiss_prot.build_manifest.json`
- **Primary key:** `primary_Accession` (UniProt accession)
- **Cross-refs captured today:**
  - `DR Pfam;` → `pfam_ids` (list column)
  - `DR GO;` → `annot_gene_ontology` (flattened string)
  - `OX NCBI_TaxID=…` → `tax_id` (used for taxonomy-tree lineage lookup)
- **Cross-refs present in raw `.dat` but NOT yet captured** (future work for this plan):
  - `DR SMART;` — required for SMART join
  - `DR InterPro;` — required for planned InterPro join
  - `DR PDB;` — required for planned PDB join
  - `DR KEGG;`, `DR Reactome;` — future
  - `DE EC=` / `CC CATALYTIC ACTIVITY` EC numbers — extracted via `_extract_ec_numbers()` in enrichment layer (string-parse from `annot_catalytic_activity`)

### TrEMBL (UniProtKB unreviewed) — planned Phase 1

- **Raw file:** `data/databases/trembl/uniprot_trembl.dat.gz` (~199 GB)
- **Version:** `reldate.txt` in same directory (shared release line with SwissProt)
- **Reader / parser:** reuses [SwissProtDatParser](src/biom3/dbio/swissprot_dat.py) unchanged
- **Built CSV:** `fully_annotated_trembl.csv` (auto-convert to Parquet when > 10 GB)
- **Primary key:** `primary_Accession`
- **Quality caveat:** TrEMBL annotations are largely auto-propagated. The builder defaults to `--evidence_filter experimental` to keep only records with manual or experimental evidence codes.

### Pfam (protein families)

- **Raw files:**
  - `data/databases/pfam/Pfam-A.fasta.gz` (44.8 M sequences, family-filtered)
  - `data/databases/pfam/Pfam-A.full.gz` (Stockholm-format alignments; source of family name/description)
  - `data/databases/pfam/Pfam-A.hmm.gz` (HMM profiles; alternative metadata source)
  - `data/databases/pfam/relnotes.txt` (version)
- **Readers / parsers:** [PfamReader](src/biom3/dbio/pfam.py), [PfamMetadataParser](src/biom3/dbio/pfam_metadata.py)
- **Built CSV:** `Pfam_protein_text_dataset.csv` + `Pfam_protein_text_dataset.build_manifest.json`
- **Primary keys:** `id` (UniProt accession) + `range` (domain span) + `pfam_label` (PF ID)
- **Cross-refs out:** UniProt accession, Pfam family ID. Stockholm metadata carries InterPro equivalents (`#=GF DR INTERPRO`) and SMART equivalents (`#=GF DR SMART`) — not currently captured, candidate for future.

### NCBI Taxonomy

- **Raw files:** `data/databases/ncbi_taxonomy/` (14 `.dmp` files + `prot.accession2taxid.gz`)
- **Version:** mtime of `new_taxdump.tar.gz`; no formal release string
- **Readers:** [TaxonomyTree](src/biom3/dbio/taxonomy.py) (in-memory ~2.7 M nodes), [AccessionTaxidMapper](src/biom3/dbio/taxonomy.py) (SQLite-backed)
- **Built index:** `accession2taxid.sqlite` via `biom3_build_taxid_index`
- **Primary keys:** `tax_id` (node-keyed); accession-keyed via `prot.accession2taxid.gz`
- **Cross-refs in:** UniProt `OX NCBI_TaxID=…`, UniProt accession via `prot.accession2taxid.gz`

### ExPASy Enzyme Nomenclature — planned Phase 2

- **Raw file:** `data/databases/expasy/enzyme.dat` (~9 MB)
- **Version:** header line `CC Release of <DD-Mon-YYYY>`
- **Parser:** `ExPASyEnzymeParser` (new, [src/biom3/dbio/expasy.py](src/biom3/dbio/expasy.py))
- **Built CSV:** `expasy_enzyme.csv` + `expasy_enzyme.build_manifest.json`
- **Primary key:** `ec` (EC number)
- **Format:** section-delimited flatfile; `ID`=EC, `DE`=description, `AN`=alternative names, `CA`=catalytic activity (reaction), `CF`=cofactor, `DR`=UniProt accession cross-refs, `//`=record separator
- **Cross-refs out:** UniProt accessions (via `DR` lines) — **ExPASy is the canonical EC↔UniProt bridge**

### SMART (domain architectures) — planned Phase 3

- **Raw file:** `data/databases/smart/SMART_domains.txt` (~350 KB; TSV)
- **Version:** file mtime (no formal release string)
- **Parser:** `SmartReader` (new, [src/biom3/dbio/smart.py](src/biom3/dbio/smart.py))
- **Built CSV:** `smart_domains.csv` + `smart_domains.build_manifest.json`
- **Primary key:** `domain_id` (SMART accession, `SMxxxxx`)
- **Format:** header `DOMAIN\tACC\tDEFINITION\tDESCRIPTION`, one row per domain
- **Cross-refs in:** UniProt `DR SMART;` lines (requires extending [SwissProtDatParser](src/biom3/dbio/swissprot_dat.py) to capture these — see Phase 5)

### BRENDA (enzyme kinetics) — planned Phase 4

- **Raw file:** `data/databases/brenda/brenda_2026_1.txt` (~278 MB)
- **Version:** header `BR\t<version>` on line 1 (e.g. `BR\t2026.1`)
- **Parser:** `BrendaParser` (new, [src/biom3/dbio/brenda.py](src/biom3/dbio/brenda.py))
- **Built CSV:** `brenda_kinetics.csv` + `brenda_kinetics.build_manifest.json`
- **Primary keys:** `ec` + `organism` (one row per enzyme-organism pair)
- **Format:** section-delimited flatfile; each entry starts with `ID\t<EC>`, contains sections `PROTEIN` (per-organism refs), `RECOMMENDED_NAME` (`RN`), `SYSTEMATIC_NAME` (`SN`), `REACTION` (`RE`), `KM_VALUE`, `TURNOVER_NUMBER`, etc. Multi-line records use indented continuation; `///` separates entries.
- **Cross-refs in:** EC number → ExPASy `ec` column → UniProt accession via ExPASy `DR` lines (**three-hop join**)
- **Organism matching:** BRENDA records organism strings; joining against UniProt requires name normalization. `--organism_match strict` (species match) vs `relaxed` (genus match).

---

## Cross-reference cheat sheet

Every join used (or planned) in [enrich.py](src/biom3/dbio/enrich.py). Read these as one-hop or multi-hop keys.

```
# UniProt accession ↔ Pfam
SwissProt.primary_Accession  ─DR Pfam;──►  Pfam.id (fasta)  ─►  Pfam.pfam_label

# UniProt accession ↔ NCBI taxonomy (two hops)
SwissProt.primary_Accession  ─OX NCBI_TaxID=──►  taxonomy.tax_id  ─►  lineage (kingdom/phylum/…)
# OR bulk:
prot.accession2taxid.gz      ─accession,taxid──►  taxonomy.tax_id

# UniProt ↔ ExPASy ↔ BRENDA  (EC is the bridge)
SwissProt.annot_catalytic_activity  ─regex EC=N.N.N.N──►  ExPASy.ec  ─DR──►  UniProt accession set
                                                         │
                                                         └──►  BRENDA.ec  (× organism)

# UniProt ↔ SMART
SwissProt.DR SMART;   ─SMxxxxx──►  SmartReader.domain_id  ─►  description

# UniProt ↔ PDB  (planned)
SwissProt.DR PDB;     ─4-char ID──►  PDB.pdb_id  ─chain──►  SCOPe / CATH classification
```

Note: `DR SMART;`, `DR InterPro;`, `DR PDB;` are **present in the raw UniProt `.dat`** but not captured by the current [SwissProtDatParser](src/biom3/dbio/swissprot_dat.py#L445-L460). Extending the parser is part of Phase 5 in the integration plan.

---

## Canonical CSV contract

All CSVs emitted by `biom3.dbio` obey one contract. Downstream consumers (Stage 1 preprocess, training-time augmentation) should target the structured columns, not the composed string.

| Column | Status | Purpose |
|---|---|---|
| `primary_Accession` | required | Join key for UniProt-rooted data |
| `protein_sequence` | required | Raw AA string |
| `pfam_label` | required (may be `['nan']`) | Pfam family membership, list-stringified |
| `annot_*` (structured) | **ground truth** | Per-field annotation values; one column per field (e.g. `annot_function`, `annot_cofactor`, `annot_ec_names`). **Canonical — always present, always parseable.** |
| `[final]text_caption` | **convenience only** | A deterministic pre-composition of the `annot_*` columns via [CaptionSpec](src/biom3/dbio/caption.py). Legacy consumers (Stage 1 today) read this directly. New consumers doing caption augmentation should read `annot_*` and compose per-epoch. |
| `[clean]text_caption`, `text_caption` | opt-in audit | Present only with `--keep_intermediate_captions`; shows the raw → evidence-stripped → final pipeline for inspection. |

**Guarantees:**
1. `annot_*` columns will remain the structured source of truth. Composed captions may be reformatted across versions; `annot_*` schemas are append-only (new fields added; existing fields never renamed without a migration).
2. `[final]text_caption` will remain present and named as-is — Stage 1 reads it at [Stage1/preprocess.py:39](src/biom3/Stage1/preprocess.py#L39).
3. Every output CSV is accompanied by:
   - `<stem>.build_manifest.json` — input paths, mtimes, sizes, upstream release versions, row counts, elapsed wall time, and a structured `stats` dict. See [build_source_swissprot.py:312-318](src/biom3/dbio/build_source_swissprot.py#L312-L318).
   - `<stem>.stats.md` — human-readable coverage report showing row count, sequence-length distribution, per-`annot_*` column coverage (% populated, mean character length), `pfam_label` distinct-family count with top-N, and (for `build_dataset`) per-source row breakdown + per-join hit rates.

---

## Planned databases

These are next in line. They will follow the same `build_source_*` + `<stem>.build_manifest.json` + linkage-via-UniProt pattern. No implementation lives here yet; this section locks the linkage scheme so future integration doesn't stall on schema debate.

### PDB (structural)

- **Expected raw file:** something like `data/databases/pdb/pdb_seqres.txt.gz` or mmCIF mirror; not yet on disk.
- **Primary key:** 4-char PDB ID (`1ABC`) optionally with chain (`1ABC_A`)
- **Linkage in:** UniProt `DR PDB;` (currently unparsed — see Phase 5 parser extension)
- **Annotation value:** method, resolution, chain → UniProt range mapping. Feeds structural quality signals into captions (e.g. `annot_pdb_structures: "1ABC (X-ray 2.0 Å), 2DEF (cryo-EM 3.1 Å)"`).

### SCOPe (structural classification)

- **Expected raw file:** SCOPe `dir.cla.scope*.txt` or SCOPe-Provis dumps.
- **Primary key:** SCOPe concise classifier (e.g. `a.1.1.1`) keyed by PDB ID + chain
- **Linkage in:** via PDB ID → SCOPe domain → hierarchy label
- **Annotation value:** fold / superfamily / family names (four-level hierarchy). Adds high-level structural annotation orthogonal to Pfam.

### CATH (structural classification)

- **Expected raw file:** `cath-domain-list.txt` or the `CathDomainFunFam` derivative.
- **Primary key:** CATH superfamily code (`N.N.N.N`) keyed by PDB ID + chain
- **Linkage in:** via PDB ID → CATH domain → hierarchy label
- **Annotation value:** class / architecture / topology / homologous superfamily names.

All three share the **PDB ID as the required bridge**. Integrating any one of them requires, at minimum, the `DR PDB;` parser extension in [SwissProtDatParser](src/biom3/dbio/swissprot_dat.py).

---

## When to revise this doc

- When a new `build_source_*` script lands, add its database section here.
- When [SwissProtDatParser](src/biom3/dbio/swissprot_dat.py) starts capturing a new `DR` cross-ref type, update the "Cross-refs captured today" list for SwissProt.
- When a join helper is added to [enrich.py](src/biom3/dbio/enrich.py), add the hop to the cheat sheet.
- When a planned database gets downloaded and a parser starts, move its section up from "Planned" into "Integrated".
- When the canonical CSV contract changes, update the table and bump a version note on the consumer side (Stage 1 preprocess).

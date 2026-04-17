# Training CSV Provenance

The two training CSV files used by `biom3.dbio` for dataset construction were built
by parsing raw protein database files. This document traces each CSV column back to
its source.

Both CSVs can now be regenerated in-package via
[`biom3_build_source_swissprot`](../src/biom3/dbio/build_source_swissprot.py) and
[`biom3_build_source_pfam`](../src/biom3/dbio/build_source_pfam.py) — see
[building_datasets_with_dbio.md](building_datasets_with_dbio.md#rebuilding-the-source-csvs-from-raw-databases)
for CLI usage. The column mappings below describe what those builders (and the
legacy CSVs they replace) produce.

## Swiss-Prot CSV: `fully_annotated_swiss_prot.csv`

**Source files**:
- `uniprot_sprot.dat.gz` — UniProt/Swiss-Prot flat file (~661 MB compressed)
- `Pfam-A.full.gz` or `Pfam-A.hmm.gz` — for Pfam family-name lookup (used only
  when a `FAMILY NAMES` caption field is emitted)

**Rows**: ~570K (one per Swiss-Prot entry). By default the builder keeps
entries without any `DR Pfam;` cross-reference and stamps their `pfam_label`
as `['nan']` for legacy parity. Pass `--require_pfam` to filter them out
(~31K entries removed).

### Column mapping (4 columns)

| CSV Column | Source | Derivation |
|---|---|---|
| `primary_Accession` | `AC` line in `.dat` | First accession before the semicolon (e.g., `AC   A0A009IHW8;` → `A0A009IHW8`) |
| `protein_sequence` | `SQ` block in `.dat` | Full amino acid sequence (whitespace stripped) |
| `pfam_label` | `DR   Pfam;` lines in `.dat` | All Pfam accessions as a stringified Python list (e.g., `"['PF13676']"`). Entries with no `DR Pfam;` lines get `"['nan']"` unless filtered via `--require_pfam`. |
| `[final]text_caption` | `DE`, `CC`, `DR GO`, `OC`/`OX` lines + Pfam metadata | Composed by `biom3.dbio.caption.compose_row_caption` using `SWISSPROT_SPEC` (ALL-CAPS labels, PubMed + `{ECO:…}` tags stripped) |

Legacy distributions of this CSV also shipped with intermediate `text_caption`
(raw, with `(PubMed:…)` refs and `{ECO:…}` tags) and `[clean]text_caption`
(evidence-stripped) columns showing progressive cleanup. By default the
in-package builder emits only `[final]text_caption` — the cleaning is applied
directly by `CaptionSpec` and the intermediate strings are not persisted. Pass
`--keep_intermediate_captions` to recover the legacy 6-column schema
(`text_caption` / `[clean]text_caption` / `[final]text_caption`).

### Caption field sources

`[final]text_caption` is composed in the field order defined by `SWISSPROT_SPEC`
(see [build_source_swissprot.py](../src/biom3/dbio/build_source_swissprot.py)),
with ALL-CAPS labels as prefixes:

| Caption Field | `annot_*` column | `.dat` Line Code | Parsing Notes |
|---|---|---|---|
| `PROTEIN NAME` | `annot_protein_name` | `DE   RecName: Full=...` | Extract `Full=` value |
| `FUNCTION` | `annot_function` | `CC   -!- FUNCTION:` | Multi-line block, continuation lines joined |
| `CATALYTIC ACTIVITY` | `annot_catalytic_activity` | `CC   -!- CATALYTIC ACTIVITY:` | Extract `Reaction=` value |
| `COFACTOR` | `annot_cofactor` | `CC   -!- COFACTOR:` | Cofactor names |
| `ACTIVITY REGULATION` | `annot_activity_regulation` | `CC   -!- ACTIVITY REGULATION:` | Text block |
| `BIOPHYSICOCHEMICAL PROPERTIES` | `annot_biophysicochemical_properties` | `CC   -!- BIOPHYSICOCHEMICAL PROPERTIES:` | KM, kcat, pH data |
| `PATHWAY` | `annot_pathway` | `CC   -!- PATHWAY:` | Text block |
| `SUBUNIT` | `annot_subunit` | `CC   -!- SUBUNIT:` | Text block |
| `SUBCELLULAR LOCATION` | `annot_subcellular_location` | `CC   -!- SUBCELLULAR LOCATION:` | Location values extracted |
| `TISSUE SPECIFICITY` | `annot_tissue_specificity` | `CC   -!- TISSUE SPECIFICITY:` | Text block |
| `DOMAIN` | `annot_domain` | `CC   -!- DOMAIN:` | Text block |
| `PTM` | `annot_ptm` | `CC   -!- PTM:` | Text block |
| `SIMILARITY` | `annot_similarity` | `CC   -!- SIMILARITY:` | Text block |
| `MISCELLANEOUS` | `annot_miscellaneous` | `CC   -!- MISCELLANEOUS:` | Text block |
| `INDUCTION` | `annot_induction` | `CC   -!- INDUCTION:` | Text block |
| `DEVELOPMENTAL STAGE` | `annot_developmental_stage` | `CC   -!- DEVELOPMENTAL STAGE:` | Text block |
| `BIOTECHNOLOGY` | `annot_biotechnology` | `CC   -!- BIOTECHNOLOGY:` | Text block |
| `GENE ONTOLOGY` | `annot_gene_ontology` | `DR   GO; GO:nnnnnnn; X:term_name; ...` | GO terms extracted, `F:`/`P:`/`C:` aspect prefixes stripped |
| `LINEAGE` | `annot_lineage` | `OC` lines (or NCBI `OX` tax_id, if a `TaxonomyTree` is passed) | Formatted as `"The organism lineage is …"` |
| `FAMILY NAMES` | (derived per-row) | Pfam family names joined from `pfam_metadata` | Template: `"Family names are {names}"` |

`SWISSPROT_SPEC` sets `strip_pubmed=True` and `strip_evidence=True`, which remove
`(PubMed:\d+)` citations and `{ECO:…}` qualifier tags during caption composition.

---

## Pfam CSV: `Pfam_protein_text_dataset.csv`

**Source files**:
- `Pfam-A.fasta.gz` — Ungapped per-domain sequences (one FASTA record per
  domain occurrence). This is the primary source parsed row-by-row.
- `Pfam-A.full.gz` *(preferred)* or `Pfam-A.hmm.gz` — Family metadata only
  (`family_name`, `family_description`), parsed once into an in-memory
  `PF_ID → metadata` lookup by
  [`PfamMetadataParser`](../src/biom3/dbio/pfam_metadata.py). The Stockholm
  file is preferred because it carries family-description `#=GF CC` lines,
  which the HMM file lacks.

**Rows**: one per FASTA record (one domain occurrence in a protein).

### Column mapping (8 columns)

Each `Pfam-A.fasta` header has the form:

```
>A0A067SRH6_GALM3/383-505 A0A067SRH6.1 PF26733.1;03009_C;
```

— three whitespace-separated parts, which the builder splits as follows:

| CSV Column | Source | Derivation |
|---|---|---|
| `id` | `parts[1]` of FASTA header | UniProt accession, version stripped (e.g., `A0A067SRH6.1` → `A0A067SRH6`) |
| `range` | `parts[0]` of header, after `/` | Domain boundary residue range (e.g., `383-505`) |
| `description` | `" ".join(parts[1:])` | Full tail of the header line (e.g., `"A0A067SRH6.1 PF26733.1;03009_C;"`) |
| `pfam_label` | `parts[2]` of header, version stripped | Single Pfam accession (e.g., `PF26733.1` → `PF26733`) |
| `sequence` | FASTA body | Ungapped amino acid sequence (concatenated lines between headers) |
| `family_name` | `pfam_metadata[pfam_label]["family_name"]` | From Stockholm `#=GF DE` or HMM `DESC` |
| `family_description` | `pfam_metadata[pfam_label]["family_description"]` | From Stockholm `#=GF CC` lines (empty when HMM is used) |
| `[final]text_caption` | `family_name` + `family_description` | Composed by `PFAM_SPEC` as `"Protein name: {family_name}. Family description: {family_description}"` (lowercase labels, no trailing period) |

Unlike the SwissProt builder, `PFAM_SPEC` keeps `strip_pubmed=False` /
`strip_evidence=False` — Pfam family metadata doesn't carry PubMed/ECO tags,
so no scrubbing is needed.

### Key observation: Pfam accessions are TrEMBL

The `id` column contains UniProt accessions that are overwhelmingly from TrEMBL
(unreviewed), not Swiss-Prot (reviewed). This is because Pfam full alignments
include all matching sequences from UniProtKB, and TrEMBL vastly outnumbers
Swiss-Prot (~250M vs ~568K entries). This has implications for enrichment — the
local `uniprot_sprot.dat.gz` covers <0.01% of Pfam accessions.

---

## Regeneration

Both CSVs can be regenerated from raw database files with no API dependency via
the `biom3_build_source_*` CLIs:

```bash
biom3_build_source_swissprot \
    --dat data/databases/swissprot/uniprot_sprot.dat.gz \
    --pfam_metadata data/databases/pfam/Pfam-A.full.gz \
    -o data/datasets/fully_annotated_swiss_prot.csv

biom3_build_source_pfam \
    --fasta data/databases/pfam/Pfam-A.fasta.gz \
    --pfam_metadata data/databases/pfam/Pfam-A.full.gz \
    -o data/datasets/Pfam_protein_text_dataset.csv
```

Caption formatting for either CSV is controlled by a `CaptionSpec`
(`SWISSPROT_SPEC` / `PFAM_SPEC`). See
[building_datasets_with_dbio.md](building_datasets_with_dbio.md#rebuilding-the-source-csvs-from-raw-databases)
for details and [demos/custom_caption_format.py](../demos/custom_caption_format.py)
for overriding the defaults.

# Training CSV Provenance

The two training CSV files used by `biom3.dbio` for dataset construction were built
by parsing raw protein database files. This document traces each CSV column back to
its source, to enable future regeneration from updated database releases.

## Swiss-Prot CSV: `fully_annotated_swiss_prot.csv`

**Source file**: `uniprot_sprot.dat.gz` (UniProt/Swiss-Prot flat file, ~661 MB compressed)

**Rows**: ~570K (one per reviewed Swiss-Prot entry)

### Column mapping

| CSV Column | Source | Derivation |
|---|---|---|
| `primary_Accession` | `AC` line in `.dat` | First accession before the semicolon (e.g., `AC   A0A009IHW8;` → `A0A009IHW8`) |
| `protein_sequence` | `SQ` block in `.dat` | Full amino acid sequence (whitespace stripped) |
| `pfam_label` | `DR   Pfam;` lines in `.dat` | All Pfam accessions collected into a stringified Python list (e.g., `"['PF13676']"`) |
| `text_caption` | `DE`, `CC`, `DR GO` lines | Parsed and concatenated with ALL-CAPS field labels. **Includes PubMed evidence tags** like `(PubMed:29395922)` |
| `[clean]text_caption` | `text_caption` | PubMed/evidence references stripped (e.g., `(PubMed:29395922)` removed) |
| `[final]text_caption` | `[clean]text_caption` | Identical to `[clean]` — no further processing applied |

### Caption field sources

The `text_caption` column concatenates annotations from these `.dat` file sections
using ALL-CAPS labels as prefixes:

| Caption Field | `.dat` Line Code | Parsing Notes |
|---|---|---|
| `PROTEIN NAME` | `DE   RecName: Full=...` | Extract `Full=` value, strip `{ECO:...}` evidence tags |
| `FUNCTION` | `CC   -!- FUNCTION: ...` | Multi-line text block, join continuation lines |
| `CATALYTIC ACTIVITY` | `CC   -!- CATALYTIC ACTIVITY:` | Extract `Reaction=` value from sub-block |
| `COFACTOR` | `CC   -!- COFACTOR:` | Extract cofactor names |
| `ACTIVITY REGULATION` | `CC   -!- ACTIVITY REGULATION:` | Text block |
| `BIOPHYSICOCHEMICAL PROPERTIES` | `CC   -!- BIOPHYSICOCHEMICAL PROPERTIES:` | KM, kcat, pH data |
| `PATHWAY` | `CC   -!- PATHWAY:` | Text block |
| `SUBUNIT` | `CC   -!- SUBUNIT:` | Text block |
| `SUBCELLULAR LOCATION` | `CC   -!- SUBCELLULAR LOCATION:` | Location values extracted |
| `PTM` | `CC   -!- PTM:` | Text block |
| `SIMILARITY` | `CC   -!- SIMILARITY:` | Text block |
| `DOMAIN` | `CC   -!- DOMAIN:` | Text block |
| `MISCELLANEOUS` | `CC   -!- MISCELLANEOUS:` | Text block |
| `INDUCTION` | `CC   -!- INDUCTION:` | Text block |
| `GENE ONTOLOGY` | `DR   GO; GO:nnnnnnn; X:term_name; ...` | GO term names extracted, `F:`/`P:`/`C:` aspect prefixes stripped, comma-joined |
| `LINEAGE` | `OC` lines | Organism classification taxa, formatted as `"The organism lineage is ..."` |

### Cleaning pipeline

The three caption columns represent a progressive cleaning:

1. **`text_caption`**: Raw parsed text including PubMed evidence citations
   - Example: `"...nicotinamide (PubMed:29395922). In addition..."`
2. **`[clean]text_caption`**: Citations stripped via regex
   - Example: `"...nicotinamide. In addition..."`
3. **`[final]text_caption`**: Identical to `[clean]` in this dataset

The cleaning regex likely matches patterns like `(PubMed:\d+)`, `(ECO:\d+|...)`,
`{ECO:...}`, and `(By similarity)` / `(Probable)` qualifiers.

---

## Pfam CSV: `Pfam_protein_text_dataset.csv`

**Source files**:
- `Pfam-A.full.gz` — Full Pfam alignments in Stockholm format (~23 GB compressed)
- `Pfam-A.hmm.gz` — Profile HMMs with family metadata (~367 MB compressed)

**Rows**: ~44.8M (one per domain occurrence in a protein)

### Column mapping

| CSV Column | Source | Derivation |
|---|---|---|
| `id` | Stockholm alignment `#=GS ... AC` annotation | UniProt accession (e.g., `A0A1I4YJU4`) |
| `range` | Sequence ID in alignment (e.g., `A0A1I4YJU4_9GAMM/160-195`) | Domain boundary residue range (`160-195`) |
| `sequence` | Alignment row in `Pfam-A.full.gz` | Gap characters (`.`, `-`) stripped from the aligned sequence |
| `pfam_label` | Stockholm `#=GF AC` header per alignment block | Single Pfam accession (e.g., `PF10417`) |
| `family_name` | `NAME` field in `Pfam-A.hmm.gz` | E.g., `"C-terminal domain of 1-Cys peroxiredoxin"` |
| `family_description` | `DESC` field in `Pfam-A.hmm.gz` | Family description text from the HMM |
| `description` | Stockholm sequence ID + accession | E.g., `"A0A1I4YJU4.1 PF10417.12;1-cysPrx_C;"` |
| `[final]text_caption` | `family_name` + `family_description` | `"Protein name: {family_name}. Family description: {family_description}"` |

### Stockholm alignment format

Each family in `Pfam-A.full.gz` is a Stockholm-format alignment block:

```
# STOCKHOLM 1.0
#=GF ID   1-cysPrx_C
#=GF AC   PF10417.15
#=GF DE   C-terminal domain of 1-Cys peroxiredoxin
...
#=GS A0A1I4YJU4_9GAMM/160-195  AC A0A1I4YJU4.1
A0A1I4YJU4_9GAMM/160-195       ..ALQFH.E..E.....H..G...
//
```

The CSV is built by iterating over all alignment blocks, extracting each sequence
entry with its accession, range, and gap-stripped sequence, then joining with family
metadata from the corresponding HMM entry.

### Key observation: Pfam accessions are TrEMBL

The `id` column contains UniProt accessions that are overwhelmingly from TrEMBL
(unreviewed), not Swiss-Prot (reviewed). This is because Pfam full alignments
include all matching sequences from UniProtKB, and TrEMBL vastly outnumbers
Swiss-Prot (~250M vs ~568K entries). This has implications for enrichment — the
local `uniprot_sprot.dat.gz` covers <0.01% of Pfam accessions.

---

## Regeneration

Both CSVs can be regenerated from raw database files without any API dependency:

1. **Swiss-Prot CSV**: Parse `uniprot_sprot.dat.gz` entry by entry, extracting
   DE/CC/DR/OC/SQ blocks, composing captions, and stripping evidence tags.

2. **Pfam CSV**: Parse `Pfam-A.full.gz` Stockholm blocks, extracting per-sequence
   entries with gap-stripped sequences. Join with `Pfam-A.hmm.gz` for family
   NAME/DESC metadata.

A regeneration pipeline could live in the `BioM3-data-share` repository alongside
the download scripts, since it operates on the raw database files stored there.
The `biom3.dbio.swissprot_dat.SwissProtDatParser` already implements much of the
Swiss-Prot parsing logic and could serve as a starting point.

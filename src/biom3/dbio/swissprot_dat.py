"""Local parser for UniProt Swiss-Prot .dat flat files.

Replaces the UniProt REST API for annotation extraction. Parses
uniprot_sprot.dat.gz and produces the same annotation dict as
enrich.extract_annotations(), keyed by accession.

Two usage modes:
  1. parse(accessions) — targeted extraction for a set of accessions
     (used by the enrichment pipeline).
  2. parse_all() — bulk generator over every entry in the file
     (used to build fully_annotated_swiss_prot.csv from scratch).

The .dat format is documented at:
https://web.expasy.org/docs/userman.html
"""

import gzip
import os
import re
import shutil
import subprocess

from tqdm import tqdm

from biom3.backend.device import setup_logger

logger = setup_logger(__name__)


# Matches UniProt EC-number cross-references in the .dat flat-file form
# "EC=N.N.N.N". Used to harvest ECs from DE lines and CC CATALYTIC ACTIVITY
# blocks into a structured annot_ec_numbers column.
_EC_IN_DAT_RE = re.compile(r"EC=(\d+\.\d+\.\d+\.\d+)")


def _extract_ec_numbers_from_lines(lines):
    """Return a de-duplicated list of EC=N.N.N.N numbers found in *lines*."""
    ecs = []
    for line in lines:
        for match in _EC_IN_DAT_RE.finditer(line):
            ec = match.group(1)
            if ec not in ecs:
                ecs.append(ec)
    return ecs


def _open_gz(path):
    """Open a .gz file using pigz (parallel) if available, else gzip.

    Returns a file-like object yielding raw bytes (binary mode).
    When pigz is on PATH, decompression is parallelised across cores,
    which is 4-8x faster than Python's single-threaded gzip module on
    large files like uniprot_trembl.dat.gz.

    For non-.gz files, returns a plain binary file handle.
    """
    if not path.endswith(".gz"):
        return open(path, "rb")

    pigz = shutil.which("pigz")
    if pigz:
        proc = subprocess.Popen(
            [pigz, "-dc", path],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        logger.info("Using pigz for parallel decompression")
        return proc.stdout

    return gzip.open(path, "rb")

# Evidence tags in curly braces that we strip from text
_EVIDENCE_RE = re.compile(r"\s*\{ECO:\d+.*?\}")


def _strip_evidence(text):
    """Remove {ECO:...} evidence tags from text."""
    return _EVIDENCE_RE.sub("", text).strip()


def _clean_text(text):
    """Strip evidence tags and trailing periods/semicolons."""
    text = _strip_evidence(text)
    text = text.rstrip(";").rstrip(".").strip()
    return text


class SwissProtDatParser:
    """Parses uniprot_sprot.dat.gz to extract annotations, sequences, and
    Pfam cross-references.

    Two modes:
      - parse(accessions): targeted extraction (enrichment pipeline)
      - parse_all(): bulk generator over all entries (source dataset builder)
    """

    def __init__(self, dat_path):
        self.dat_path = dat_path

    def parse(self, accessions):
        """Parse annotations for a set of accessions.

        Args:
            accessions: iterable of accession strings to extract.

        Returns:
            dict mapping accession -> annotation dict with annot_* keys
            (same format as enrich.extract_annotations).
        """
        accession_set = set(accessions)
        accession_set_bytes = {a.encode() for a in accession_set}
        results = {}

        logger.info("Parsing Swiss-Prot .dat file for %s accessions: %s",
                     f"{len(accession_set):,}", self.dat_path)

        entry_lines = []
        current_acc = None
        matched = False
        parsed_count = 0
        skipped_count = 0

        AC_PREFIX = b"AC   "
        ENTRY_END = b"//"

        fh = _open_gz(self.dat_path)
        try:
            for raw_line in tqdm(fh, desc="Parsing .dat", unit=" lines"):
                if raw_line[:5] == AC_PREFIX:
                    if current_acc is None:
                        acc_bytes = raw_line[5:].split(b";", 1)[0].strip()
                        if acc_bytes in accession_set_bytes:
                            current_acc = acc_bytes.decode()
                            matched = True

                if matched:
                    entry_lines.append(raw_line.decode())

                if raw_line[:2] == ENTRY_END:
                    if matched and current_acc:
                        annotations = _parse_entry(entry_lines)
                        results[current_acc] = annotations
                        parsed_count += 1
                    else:
                        skipped_count += 1
                    entry_lines = []
                    current_acc = None
                    matched = False

                    if len(results) == len(accession_set):
                        break
        finally:
            fh.close()

        logger.info("Parsed %s/%s accessions (%s entries skipped)",
                     f"{parsed_count:,}", f"{len(accession_set):,}",
                     f"{skipped_count:,}")
        return results

    def parse_all(self, require_pfam=False):
        """Yield (accession, entry_dict) for every entry in the .dat file.

        Args:
            require_pfam: if True, skip entries without Pfam DR cross-refs.

        Yields:
            (accession, dict) where dict has keys:
                'sequence': str — full protein sequence
                'pfam_ids': list[str] — Pfam accessions (e.g. ['PF04947'])
                'annotations': dict — annot_* keys (same as _parse_entry)
        """
        logger.info("Bulk-parsing Swiss-Prot .dat file: %s", self.dat_path)

        entry_lines = []
        current_acc = None
        parsed_count = 0
        skipped_count = 0

        AC_PREFIX = b"AC   "
        ENTRY_END = b"//"

        fh = _open_gz(self.dat_path)
        try:
            for raw_line in tqdm(fh, desc="Parsing .dat (bulk)", unit=" lines"):
                line = raw_line.decode()

                if raw_line[:5] == AC_PREFIX and current_acc is None:
                    current_acc = line[5:].split(";")[0].strip()

                entry_lines.append(line)

                if raw_line[:2] == ENTRY_END:
                    if current_acc:
                        result = _parse_entry_full(entry_lines)
                        if require_pfam and not result["pfam_ids"]:
                            skipped_count += 1
                        else:
                            parsed_count += 1
                            yield current_acc, result
                    entry_lines = []
                    current_acc = None
        finally:
            fh.close()

        logger.info("Bulk parse complete: %s entries yielded, %s skipped",
                     f"{parsed_count:,}", f"{skipped_count:,}")


def _parse_entry(lines):
    """Parse a single .dat entry into an annotation dict with annot_* keys.

    Also attaches SMART/InterPro/PDB cross-reference lists under the keys
    ``xref_smart_ids``, ``xref_interpro_ids``, ``xref_pdb_ids`` when present.
    These side-channel entries flow through ``enrich_dataframe`` into their
    own DataFrame columns for downstream join functions.
    """
    annotations = {}

    # Collect line groups
    de_lines = []
    oc_lines = []
    cc_blocks = {}
    go_terms = []
    smart_ids = []
    interpro_ids = []
    pdb_ids = []

    current_cc_topic = None
    current_cc_text = []

    for line in lines:
        code = line[:2]

        if code == "DE":
            de_lines.append(line[5:].rstrip("\n"))

        elif code == "OC":
            oc_lines.append(line[5:].rstrip("\n"))

        elif code == "CC":
            text = line[5:].rstrip("\n")
            if text.startswith("-!- "):
                # Save previous CC block
                if current_cc_topic and current_cc_text:
                    cc_blocks.setdefault(current_cc_topic, []).append(
                        " ".join(current_cc_text)
                    )
                # Start new topic
                topic_text = text[4:]
                if ":" in topic_text:
                    topic, rest = topic_text.split(":", 1)
                    current_cc_topic = topic.strip()
                    current_cc_text = [rest.strip()] if rest.strip() else []
                else:
                    current_cc_topic = topic_text.strip()
                    current_cc_text = []
            elif text.startswith("-----") or text.startswith("Copyrighted"):
                # End of CC blocks — save last one
                if current_cc_topic and current_cc_text:
                    cc_blocks.setdefault(current_cc_topic, []).append(
                        " ".join(current_cc_text)
                    )
                current_cc_topic = None
                current_cc_text = []
            elif current_cc_topic:
                current_cc_text.append(text.strip())

        elif code == "DR":
            text = line[5:].rstrip("\n")
            if text.startswith("GO;"):
                # Format: GO; GO:0005884; C:actin filament; IEA:Ensembl.
                parts = text.split(";")
                if len(parts) >= 3:
                    go_desc = parts[2].strip().rstrip(".")
                    # Strip aspect prefix (C:, F:, P:)
                    if len(go_desc) > 2 and go_desc[1] == ":":
                        go_desc = go_desc[2:]
                    go_terms.append(go_desc)
            elif text.startswith("SMART;"):
                parts = text.split(";")
                if len(parts) >= 2:
                    smart_acc = parts[1].strip()
                    if smart_acc and smart_acc not in smart_ids:
                        smart_ids.append(smart_acc)
            elif text.startswith("InterPro;"):
                parts = text.split(";")
                if len(parts) >= 2:
                    ipr_acc = parts[1].strip()
                    if ipr_acc and ipr_acc not in interpro_ids:
                        interpro_ids.append(ipr_acc)
            elif text.startswith("PDB;"):
                parts = text.split(";")
                if len(parts) >= 2:
                    pdb_acc = parts[1].strip()
                    if pdb_acc and pdb_acc not in pdb_ids:
                        pdb_ids.append(pdb_acc)

    # Save last CC block if not already saved
    if current_cc_topic and current_cc_text:
        cc_blocks.setdefault(current_cc_topic, []).append(
            " ".join(current_cc_text)
        )

    # --- Build annotation dict ---

    # PROTEIN NAME from DE lines
    protein_name = _parse_de_lines(de_lines)
    if protein_name:
        annotations["annot_protein_name"] = protein_name

    # LINEAGE from OC lines
    lineage = _parse_oc_lines(oc_lines)
    if lineage:
        annotations["annot_lineage"] = lineage

    # CC-based fields
    _map_cc_field(cc_blocks, "FUNCTION", "annot_function", annotations)
    _map_cc_catalytic_activity(cc_blocks, annotations)
    _map_cc_field(cc_blocks, "COFACTOR", "annot_cofactor", annotations)
    _map_cc_field(cc_blocks, "ACTIVITY REGULATION", "annot_activity_regulation", annotations)
    _map_cc_biophysicochemical(cc_blocks, annotations)
    _map_cc_field(cc_blocks, "PATHWAY", "annot_pathway", annotations)
    _map_cc_field(cc_blocks, "SUBUNIT", "annot_subunit", annotations)
    _map_cc_subcellular_location(cc_blocks, annotations)
    _map_cc_field(cc_blocks, "PTM", "annot_ptm", annotations)
    _map_cc_field(cc_blocks, "SIMILARITY", "annot_similarity", annotations)
    _map_cc_field(cc_blocks, "DOMAIN", "annot_domain", annotations)
    _map_cc_field(cc_blocks, "MISCELLANEOUS", "annot_miscellaneous", annotations)
    _map_cc_field(cc_blocks, "INDUCTION", "annot_induction", annotations)

    # GENE ONTOLOGY
    if go_terms:
        annotations["annot_gene_ontology"] = ", ".join(go_terms)

    # Merge EC numbers from DE lines with any already extracted from
    # CC CATALYTIC ACTIVITY. The DE block frequently carries an explicit
    # "EC=N.N.N.N" line that CC blocks may lack when the entry has no
    # catalytic_activity annotation (e.g., sub-classified enzymes).
    de_ecs = _extract_ec_numbers_from_lines(de_lines)
    if de_ecs:
        existing = annotations.get("annot_ec_numbers", "")
        merged = [e.strip() for e in str(existing).split(",") if e.strip()]
        for ec in de_ecs:
            if ec not in merged:
                merged.append(ec)
        annotations["annot_ec_numbers"] = ", ".join(merged)

    # Cross-reference side-channel — flow through enrich_dataframe into
    # distinct DataFrame columns for downstream join functions.
    if smart_ids:
        annotations["xref_smart_ids"] = smart_ids
    if interpro_ids:
        annotations["xref_interpro_ids"] = interpro_ids
    if pdb_ids:
        annotations["xref_pdb_ids"] = pdb_ids

    return annotations


def _parse_de_lines(de_lines):
    """Extract the recommended full protein name from DE lines."""
    for line in de_lines:
        line = line.strip()
        if line.startswith("RecName: Full="):
            name = line[len("RecName: Full="):]
            return _clean_text(name)
        if line.startswith("SubName: Full="):
            name = line[len("SubName: Full="):]
            return _clean_text(name)
    return None


def _parse_oc_lines(oc_lines):
    """Build lineage string from OC (Organism Classification) lines."""
    full = " ".join(oc_lines)
    # Remove trailing period
    full = full.rstrip(".")
    taxa = [t.strip() for t in full.split(";") if t.strip()]
    if taxa:
        return "The organism lineage is " + ", ".join(taxa)
    return None


def _map_cc_field(cc_blocks, topic, annot_key, annotations):
    """Map a simple CC topic to an annotation key, stripping evidence tags."""
    texts = cc_blocks.get(topic)
    if texts:
        # Take the first block (non-isoform-specific)
        val = _strip_evidence(texts[0])
        if val:
            annotations[annot_key] = val


def _map_cc_catalytic_activity(cc_blocks, annotations):
    """Parse CATALYTIC ACTIVITY blocks.

    Extracts two things in parallel:
    - Reaction= prose (caption-safe, stored in ``annot_catalytic_activity``)
    - Any EC=N.N.N.N xrefs (stored in ``annot_ec_numbers`` for downstream
      joins; never added to captions)
    """
    texts = cc_blocks.get("CATALYTIC ACTIVITY")
    if not texts:
        return
    parts = []
    ecs = []
    for text in texts:
        match = re.search(r"Reaction=([^;]+)", text)
        if match:
            reaction = _strip_evidence(match.group(1)).strip()
            if reaction:
                parts.append(reaction)
        for match in _EC_IN_DAT_RE.finditer(text):
            ec = match.group(1)
            if ec not in ecs:
                ecs.append(ec)
    if parts:
        annotations["annot_catalytic_activity"] = ". ".join(parts)
    if ecs:
        existing = annotations.get("annot_ec_numbers", "")
        if existing:
            merged = [e.strip() for e in str(existing).split(",") if e.strip()]
            for ec in ecs:
                if ec not in merged:
                    merged.append(ec)
            annotations["annot_ec_numbers"] = ", ".join(merged)
        else:
            annotations["annot_ec_numbers"] = ", ".join(ecs)


def _map_cc_biophysicochemical(cc_blocks, annotations):
    """Parse BIOPHYSICOCHEMICAL PROPERTIES — extract Note= lines and key data."""
    texts = cc_blocks.get("BIOPHYSICOCHEMICAL PROPERTIES")
    if not texts:
        return
    # Combine all blocks and clean
    combined = " ".join(texts)
    val = _strip_evidence(combined)
    if val:
        annotations["annot_biophysicochemical_properties"] = val


def _map_cc_subcellular_location(cc_blocks, annotations):
    """Parse SUBCELLULAR LOCATION blocks."""
    texts = cc_blocks.get("SUBCELLULAR LOCATION")
    if not texts:
        return
    # The first block typically has the location; strip isoform tags and evidence
    val = _strip_evidence(texts[0])
    # Clean up common suffixes
    val = val.split("Note=")[0].strip().rstrip(".")
    if val:
        annotations["annot_subcellular_location"] = val


# ---------------------------------------------------------------------------
# Full entry parser (annotations + sequence + Pfam cross-refs)
# ---------------------------------------------------------------------------

def _parse_entry_full(lines):
    """Parse a .dat entry into a dict with annotations, sequence, and Pfam IDs.

    Returns:
        dict with keys:
            'annotations': dict of annot_* keys (superset of _parse_entry output)
            'sequence': str, the full protein sequence
            'pfam_ids': list of str, Pfam accessions without version
    """
    annotations = {}

    de_lines = []
    oc_lines = []
    cc_blocks = {}
    go_terms = []
    pfam_ids = []
    smart_ids = []
    interpro_ids = []
    pdb_ids = []
    seq_lines = []
    tax_id = None
    in_sq = False

    current_cc_topic = None
    current_cc_text = []

    for line in lines:
        code = line[:2]

        if in_sq:
            if code == "//":
                in_sq = False
            else:
                seq_lines.append(line.strip().replace(" ", ""))
            continue

        if code == "DE":
            de_lines.append(line[5:].rstrip("\n"))

        elif code == "OC":
            oc_lines.append(line[5:].rstrip("\n"))

        elif code == "OX":
            # OX   NCBI_TaxID=654924;
            text = line[5:].rstrip("\n")
            match = re.search(r"NCBI_TaxID=(\d+)", text)
            if match:
                tax_id = int(match.group(1))

        elif code == "CC":
            text = line[5:].rstrip("\n")
            if text.startswith("-!- "):
                if current_cc_topic and current_cc_text:
                    cc_blocks.setdefault(current_cc_topic, []).append(
                        " ".join(current_cc_text)
                    )
                topic_text = text[4:]
                if ":" in topic_text:
                    topic, rest = topic_text.split(":", 1)
                    current_cc_topic = topic.strip()
                    current_cc_text = [rest.strip()] if rest.strip() else []
                else:
                    current_cc_topic = topic_text.strip()
                    current_cc_text = []
            elif text.startswith("-----") or text.startswith("Copyrighted"):
                if current_cc_topic and current_cc_text:
                    cc_blocks.setdefault(current_cc_topic, []).append(
                        " ".join(current_cc_text)
                    )
                current_cc_topic = None
                current_cc_text = []
            elif current_cc_topic:
                current_cc_text.append(text.strip())

        elif code == "DR":
            text = line[5:].rstrip("\n")
            if text.startswith("Pfam;"):
                parts = text.split(";")
                if len(parts) >= 2:
                    pfam_acc = parts[1].strip().split(".")[0]
                    if pfam_acc not in pfam_ids:
                        pfam_ids.append(pfam_acc)
            elif text.startswith("GO;"):
                parts = text.split(";")
                if len(parts) >= 3:
                    go_desc = parts[2].strip().rstrip(".")
                    if len(go_desc) > 2 and go_desc[1] == ":":
                        go_desc = go_desc[2:]
                    go_terms.append(go_desc)
            elif text.startswith("SMART;"):
                parts = text.split(";")
                if len(parts) >= 2:
                    smart_acc = parts[1].strip()
                    if smart_acc and smart_acc not in smart_ids:
                        smart_ids.append(smart_acc)
            elif text.startswith("InterPro;"):
                parts = text.split(";")
                if len(parts) >= 2:
                    ipr_acc = parts[1].strip()
                    if ipr_acc and ipr_acc not in interpro_ids:
                        interpro_ids.append(ipr_acc)
            elif text.startswith("PDB;"):
                parts = text.split(";")
                if len(parts) >= 2:
                    pdb_acc = parts[1].strip()
                    if pdb_acc and pdb_acc not in pdb_ids:
                        pdb_ids.append(pdb_acc)

        elif code == "SQ":
            in_sq = True

    # Save last CC block
    if current_cc_topic and current_cc_text:
        cc_blocks.setdefault(current_cc_topic, []).append(
            " ".join(current_cc_text)
        )

    # Build sequence (strip trailing line counts like "  128")
    sequence = re.sub(r"[^A-Z]", "", "".join(seq_lines))

    # Build annotations
    protein_name = _parse_de_lines(de_lines)
    if protein_name:
        annotations["annot_protein_name"] = protein_name

    lineage = _parse_oc_lines(oc_lines)
    if lineage:
        annotations["annot_lineage"] = lineage

    _map_cc_field(cc_blocks, "FUNCTION", "annot_function", annotations)
    _map_cc_catalytic_activity(cc_blocks, annotations)
    _map_cc_field(cc_blocks, "COFACTOR", "annot_cofactor", annotations)
    _map_cc_field(cc_blocks, "ACTIVITY REGULATION", "annot_activity_regulation", annotations)
    _map_cc_biophysicochemical(cc_blocks, annotations)
    _map_cc_field(cc_blocks, "PATHWAY", "annot_pathway", annotations)
    _map_cc_field(cc_blocks, "SUBUNIT", "annot_subunit", annotations)
    _map_cc_subcellular_location(cc_blocks, annotations)
    _map_cc_field(cc_blocks, "TISSUE SPECIFICITY", "annot_tissue_specificity", annotations)
    _map_cc_field(cc_blocks, "PTM", "annot_ptm", annotations)
    _map_cc_field(cc_blocks, "SIMILARITY", "annot_similarity", annotations)
    _map_cc_field(cc_blocks, "DOMAIN", "annot_domain", annotations)
    _map_cc_field(cc_blocks, "MISCELLANEOUS", "annot_miscellaneous", annotations)
    _map_cc_field(cc_blocks, "INDUCTION", "annot_induction", annotations)
    _map_cc_field(cc_blocks, "DEVELOPMENTAL STAGE", "annot_developmental_stage", annotations)
    _map_cc_field(cc_blocks, "BIOTECHNOLOGY", "annot_biotechnology", annotations)

    if go_terms:
        annotations["annot_gene_ontology"] = ", ".join(go_terms)

    # Merge DE-line EC numbers with any already harvested from
    # CC CATALYTIC ACTIVITY (see _map_cc_catalytic_activity).
    de_ecs = _extract_ec_numbers_from_lines(de_lines)
    if de_ecs:
        existing = annotations.get("annot_ec_numbers", "")
        merged = [e.strip() for e in str(existing).split(",") if e.strip()]
        for ec in de_ecs:
            if ec not in merged:
                merged.append(ec)
        annotations["annot_ec_numbers"] = ", ".join(merged)

    return {
        "annotations": annotations,
        "sequence": sequence,
        "pfam_ids": pfam_ids,
        "smart_ids": smart_ids,
        "interpro_ids": interpro_ids,
        "pdb_ids": pdb_ids,
        "tax_id": tax_id,
    }

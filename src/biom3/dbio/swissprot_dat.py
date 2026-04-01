"""Local parser for UniProt Swiss-Prot .dat flat files.

Replaces the UniProt REST API for annotation extraction. Parses
uniprot_sprot.dat.gz and produces the same annotation dict as
enrich.extract_annotations(), keyed by accession.

The .dat format is documented at:
https://web.expasy.org/docs/userman.html
"""

import gzip
import re

from tqdm import tqdm

from biom3.backend.device import setup_logger

logger = setup_logger(__name__)

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
    """Parses uniprot_sprot.dat.gz to extract annotations for a set of accessions.

    This replaces the UniProt REST API for enrichment. The .dat file
    (~661 MB compressed) is streamed line by line, and only entries
    matching the requested accessions are parsed in detail.
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
        results = {}
        is_gzipped = self.dat_path.endswith(".gz")
        opener = gzip.open if is_gzipped else open

        logger.info("Parsing Swiss-Prot .dat file for %s accessions: %s",
                     f"{len(accession_set):,}", self.dat_path)

        entry_lines = []
        current_acc = None
        matched = False
        parsed_count = 0
        skipped_count = 0

        with opener(self.dat_path, "rt") as f:
            for line in tqdm(f, desc="Parsing Swiss-Prot .dat", unit=" lines"):
                if line.startswith("AC   "):
                    # First AC line of an entry — extract primary accession
                    if current_acc is None:
                        acc = line[5:].split(";")[0].strip()
                        current_acc = acc
                        matched = acc in accession_set

                if matched:
                    entry_lines.append(line)

                if line.startswith("//"):
                    if matched and current_acc:
                        annotations = _parse_entry(entry_lines)
                        results[current_acc] = annotations
                        parsed_count += 1
                    else:
                        skipped_count += 1
                    entry_lines = []
                    current_acc = None
                    matched = False

                    # Early exit if all found
                    if len(results) == len(accession_set):
                        break

        logger.info("Parsed %s/%s accessions (%s entries skipped)",
                     f"{parsed_count:,}", f"{len(accession_set):,}",
                     f"{skipped_count:,}")
        return results


def _parse_entry(lines):
    """Parse a single .dat entry into an annotation dict with annot_* keys."""
    annotations = {}

    # Collect line groups
    de_lines = []
    oc_lines = []
    cc_blocks = {}
    go_terms = []

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
    """Parse CATALYTIC ACTIVITY blocks to extract Reaction= names."""
    texts = cc_blocks.get("CATALYTIC ACTIVITY")
    if not texts:
        return
    parts = []
    for text in texts:
        # Extract Reaction=...; portion
        match = re.search(r"Reaction=([^;]+)", text)
        if match:
            reaction = _strip_evidence(match.group(1)).strip()
            if reaction:
                parts.append(reaction)
    if parts:
        annotations["annot_catalytic_activity"] = ". ".join(parts)


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

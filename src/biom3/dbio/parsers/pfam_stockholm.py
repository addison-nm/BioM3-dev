"""Parse Pfam family metadata (name, description) from Stockholm or HMM files.

Provides a PF_ID -> {short_id, family_name, family_description} lookup dict
needed by both the SwissProt and Pfam source dataset builders.
"""

import gzip
import re

from tqdm import tqdm

from biom3.backend.device import setup_logger

logger = setup_logger(__name__)


class PfamMetadataParser:
    """Extract family-level metadata from Pfam-A.full.gz (Stockholm) or
    Pfam-A.hmm.gz.

    Stockholm is preferred because it contains #=GF CC comment lines
    (family_description). The HMM file only provides NAME and DESC.
    """

    def __init__(self, source_path):
        self.source_path = source_path

    def parse(self):
        """Parse all family metadata from the source file.

        Returns:
            dict mapping pfam_id (e.g. 'PF04947') -> {
                'short_id': str,            # e.g. 'Pox_VLTF3'
                'family_name': str,         # from DE/DESC line
                'family_description': str,  # from CC lines (empty if HMM)
                'family_type': str,         # from TP (empty if HMM or absent)
                'family_clan': str,         # from CL (empty if clanless)
                'family_wikipedia': str,    # from WK (empty if absent)
                'family_references': str,   # joined RT lines (empty if absent)
            }
        """
        if self.source_path.endswith(".hmm.gz") or self.source_path.endswith(".hmm"):
            return self._parse_hmm()
        return self._parse_stockholm()

    def _parse_stockholm(self):
        """Parse #=GF headers from Stockholm format.

        Reads line-by-line, extracting GF ID/AC/DE/CC/TP/CL/WK/RT per family
        block. Alignment data and GS/GR lines are skipped.
        """
        is_gzipped = self.source_path.endswith(".gz")
        opener = gzip.open if is_gzipped else open

        results = {}
        state = self._new_state()

        logger.info("Parsing Pfam family metadata from: %s", self.source_path)

        with opener(self.source_path, "rt") as f:
            for line in tqdm(f, desc="Scanning Pfam families", unit=" lines"):
                if not line.startswith("#=GF "):
                    if line.startswith("//"):
                        self._finalize_family(results, state)
                        state = self._new_state()
                    continue

                tag = line[5:10].rstrip()
                value = line[10:].strip() if len(line) > 10 else ""

                if tag == "ID":
                    state["short_id"] = value
                elif tag == "AC":
                    state["accession"] = value.rstrip(";").strip()
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

        logger.info("Parsed metadata for %s Pfam families", f"{len(results):,}")
        return results

    @staticmethod
    def _new_state():
        return {
            "short_id": None,
            "accession": None,
            "description": None,
            "cc_lines": [],
            "family_type": "",
            "family_clan": "",
            "family_wikipedia": "",
            "rt_lines": [],
        }

    @staticmethod
    def _finalize_family(results, state):
        accession = state["accession"]
        short_id = state["short_id"]
        if not (accession and short_id):
            return
        pfam_id = accession.split(".")[0]
        family_desc = " ".join(state["cc_lines"])
        family_refs = " ".join(
            re.sub(r"\s+", " ", line).strip()
            for line in state["rt_lines"]
            if line.strip()
        ).strip()
        results[pfam_id] = {
            "short_id": short_id,
            "family_name": state["description"] or short_id,
            "family_description": family_desc,
            "family_type": state["family_type"],
            "family_clan": state["family_clan"],
            "family_wikipedia": state["family_wikipedia"],
            "family_references": family_refs,
        }

    def _parse_hmm(self):
        """Parse NAME/ACC/DESC from HMM format.

        Faster but family_description / family_type / clan / wikipedia /
        references are all empty — the HMM header doesn't carry them.
        """
        is_gzipped = self.source_path.endswith(".gz")
        opener = gzip.open if is_gzipped else open

        results = {}
        short_id = None
        accession = None
        description = None

        logger.info("Parsing Pfam family metadata from HMM: %s", self.source_path)

        with opener(self.source_path, "rt") as f:
            for line in tqdm(f, desc="Scanning HMM families", unit=" lines"):
                if line.startswith("NAME  "):
                    short_id = line[6:].strip()
                elif line.startswith("ACC   "):
                    accession = line[6:].strip()
                elif line.startswith("DESC  "):
                    description = line[6:].strip()
                elif line.startswith("//"):
                    if accession and short_id:
                        pfam_id = accession.split(".")[0]
                        results[pfam_id] = {
                            "short_id": short_id,
                            "family_name": description or short_id,
                            "family_description": "",
                            "family_type": "",
                            "family_clan": "",
                            "family_wikipedia": "",
                            "family_references": "",
                        }
                    short_id = None
                    accession = None
                    description = None

        logger.info("Parsed metadata for %s Pfam families", f"{len(results):,}")
        return results

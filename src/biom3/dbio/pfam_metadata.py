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
            }
        """
        if self.source_path.endswith(".hmm.gz") or self.source_path.endswith(".hmm"):
            return self._parse_hmm()
        return self._parse_stockholm()

    def _parse_stockholm(self):
        """Parse #=GF headers from Stockholm format.

        Reads line-by-line, extracting only GF ID/AC/DE/CC per family block.
        Alignment data and GS/GR lines are skipped.
        """
        is_gzipped = self.source_path.endswith(".gz")
        opener = gzip.open if is_gzipped else open

        results = {}
        short_id = None
        accession = None
        description = None
        cc_lines = []

        logger.info("Parsing Pfam family metadata from: %s", self.source_path)

        with opener(self.source_path, "rt") as f:
            for line in tqdm(f, desc="Scanning Pfam families", unit=" lines"):
                if not line.startswith("#=GF "):
                    if line.startswith("//"):
                        if accession and short_id:
                            pfam_id = accession.split(".")[0]
                            family_desc = " ".join(cc_lines)
                            results[pfam_id] = {
                                "short_id": short_id,
                                "family_name": description or short_id,
                                "family_description": family_desc,
                            }
                        short_id = None
                        accession = None
                        description = None
                        cc_lines = []
                    continue

                tag = line[5:10].rstrip()
                value = line[10:].strip() if len(line) > 10 else ""

                if tag == "ID":
                    short_id = value
                elif tag == "AC":
                    accession = value.rstrip(";").strip()
                elif tag == "DE":
                    description = value
                elif tag == "CC":
                    cc_lines.append(value)

        logger.info("Parsed metadata for %s Pfam families", f"{len(results):,}")
        return results

    def _parse_hmm(self):
        """Parse NAME/ACC/DESC from HMM format.

        Faster but family_description will be empty.
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
                        }
                    short_id = None
                    accession = None
                    description = None

        logger.info("Parsed metadata for %s Pfam families", f"{len(results):,}")
        return results

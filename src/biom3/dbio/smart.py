"""Parser for the SMART `SMART_domains.txt` TSV.

Format:
    DOMAIN<TAB>ACC<TAB>DEFINITION<TAB>DESCRIPTION
    --------------------------------------------------------------------------------
    14_3_3<TAB>SM00101<TAB>14-3-3 homologues<TAB>14-3-3 homologues mediates ...
    ...
"""

from __future__ import annotations


class SmartReader:
    """Streaming reader for SMART_domains.txt."""

    def __init__(self, path):
        self.path = path

    def iter_domains(self):
        """Yield dicts with keys: domain_name, accession, definition, description."""
        with open(self.path, encoding="utf-8") as fh:
            header_seen = False
            for raw in fh:
                line = raw.rstrip("\n")
                if not line:
                    continue
                if not header_seen:
                    if line.startswith("DOMAIN"):
                        header_seen = True
                    continue
                if line.startswith("-" * 10):
                    continue
                parts = line.split("\t")
                while len(parts) < 4:
                    parts.append("")
                domain_name, accession, definition, description = parts[:4]
                if not accession.strip():
                    continue
                yield {
                    "domain_name": domain_name.strip(),
                    "accession": accession.strip(),
                    "definition": definition.strip(),
                    "description": description.strip(),
                }

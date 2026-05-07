"""Parser for the ExPASy Enzyme Nomenclature `enzyme.dat` flatfile.

The file is section-delimited with `//` between entries and uses a 2-char
line-code format. An entry spans from its `ID` line to the next `//`.

Line codes:
    ID  EC number (primary key; one per entry)
    DE  Description / recommended name (may mark entry as deleted or transferred)
    AN  Alternative names
    CA  Catalytic activity (reaction, optionally numbered (1), (2), ...)
    CF  Cofactor
    CC  Free-text comments (bullets prefixed with '-!-')
    DR  UniProt cross-references: "<accession>, <id_name>;" pairs

The leading header block (before the first `ID` line) is metadata about the
release and is skipped.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


TRANSFERRED_RE = re.compile(
    r"^Transferred entry:\s*(.+?)\.?\s*$", re.IGNORECASE
)


@dataclass
class EnzymeEntry:
    ec: str
    name: str = ""
    alternative_names: list = field(default_factory=list)
    catalytic_activities: list = field(default_factory=list)
    cofactors: list = field(default_factory=list)
    comments: list = field(default_factory=list)
    uniprot_accessions: list = field(default_factory=list)
    transferred_to: list = field(default_factory=list)
    deleted: bool = False


def _finalize_continuation(buffer):
    return " ".join(part.strip() for part in buffer).strip()


def _parse_dr_line(line, accessions):
    body = line.rstrip("\n")
    for pair in body.split(";"):
        pair = pair.strip()
        if not pair:
            continue
        acc = pair.split(",", 1)[0].strip()
        if acc:
            accessions.append(acc)


class ExPASyEnzymeParser:
    """Streaming parser for ExPASy `enzyme.dat`."""

    def __init__(self, path):
        self.path = path

    def iter_entries(self):
        """Yield `EnzymeEntry` instances for each record in the file."""
        entry = None
        de_buf = []
        ca_buf = []
        cf_buf = []
        cc_buf = []
        current_cc = []

        def flush_ca():
            if ca_buf:
                entry.catalytic_activities.append(_finalize_continuation(ca_buf))
                ca_buf.clear()

        def flush_cf():
            if cf_buf:
                entry.cofactors.append(_finalize_continuation(cf_buf))
                cf_buf.clear()

        def flush_cc():
            if current_cc:
                cc_buf.append(_finalize_continuation(current_cc))
                current_cc.clear()

        with open(self.path) as fh:
            for raw in fh:
                code = raw[:2]
                body = raw[5:].rstrip("\n") if len(raw) > 5 else ""

                if code == "ID":
                    entry = EnzymeEntry(ec=body.strip())
                    de_buf = []
                    ca_buf = []
                    cf_buf = []
                    cc_buf = []
                    current_cc = []
                    continue

                if entry is None:
                    continue

                if code == "DE":
                    de_buf.append(body.strip())
                elif code == "AN":
                    entry.alternative_names.append(body.strip().rstrip("."))
                elif code == "CA":
                    text = body.strip()
                    if re.match(r"^\(\d+\)", text):
                        flush_ca()
                        text = re.sub(r"^\(\d+\)\s*", "", text)
                    ca_buf.append(text)
                elif code == "CF":
                    text = body.strip()
                    cf_buf.append(text)
                elif code == "CC":
                    text = body.rstrip()
                    if text.startswith("-!-"):
                        flush_cc()
                        current_cc.append(text[3:].strip())
                    else:
                        current_cc.append(text.strip())
                elif code == "DR":
                    _parse_dr_line(body, entry.uniprot_accessions)
                elif raw.startswith("//"):
                    flush_ca()
                    flush_cf()
                    flush_cc()

                    description = _finalize_continuation(de_buf).rstrip(".").strip()
                    entry.name = description

                    lowered = description.lower()
                    if lowered.startswith("transferred entry"):
                        match = TRANSFERRED_RE.match(description)
                        if match:
                            targets = re.split(r"\s*(?:and|,)\s*", match.group(1))
                            entry.transferred_to = [
                                t.strip().rstrip(".") for t in targets if t.strip()
                            ]
                    elif "deleted entry" in lowered:
                        entry.deleted = True

                    entry.comments = cc_buf
                    yield entry
                    entry = None

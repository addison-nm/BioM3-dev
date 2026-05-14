"""Parser for the BRENDA flatfile (v1 — high-value sections only).

Format overview:
    BR\t<version>
    ID\t<EC>
    <SECTION_HEADER>        (all-caps bare line)
    <code>\t<record>        (new record)
    \t<continuation>        (continues prior record)
    ...
    ///                     (end of entry)

Each EC entry has a PROTEIN block mapping organism numbers (`#N#`) to
organism names, plus many named sections (RECOMMENDED_NAME, REACTION,
KM_VALUE, PH_OPTIMUM, ...). Per-organism records start with `#N#` or
`#N,M,...#`. Reference tags (`<3,7,12>`) and notes (`(...)`) close each
record.

This parser only captures the sections listed in `TRACKED_CODES`; others
are silently skipped to keep scope manageable. Extending the parser is a
matter of adding new codes + section-header mappings.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


TRACKED_CODES = {"PR", "RN", "SN", "SY", "RE", "SP", "KM", "PHO", "TO"}

SECTION_HEADERS = {
    "PROTEIN",
    "RECOMMENDED_NAME",
    "SYSTEMATIC_NAME",
    "SYNONYMS",
    "REACTION",
    "REACTION_TYPE",
    "NATURAL_SUBSTRATE_PRODUCT",
    "SUBSTRATE_PRODUCT",
    "TURNOVER_NUMBER",
    "KM_VALUE",
    "KI_VALUE",
    "COFACTOR",
    "ACTIVATING_COMPOUND",
    "INHIBITORS",
    "METALS_IONS",
    "PH_OPTIMUM",
    "PH_RANGE",
    "PH_STABILITY",
    "TEMPERATURE_OPTIMUM",
    "TEMPERATURE_RANGE",
    "TEMPERATURE_STABILITY",
    "LOCALIZATION",
    "SOURCE_TISSUE",
    "MOLECULAR_WEIGHT",
    "SUBUNITS",
    "POSTTRANSLATIONAL_MODIFICATION",
    "CRYSTALLIZATION",
    "CLONED",
    "ENGINEERING",
    "APPLICATION",
    "GENERAL_STABILITY",
    "STORAGE_STABILITY",
    "OXIDATION_STABILITY",
    "ORGANIC_SOLVENT_STABILITY",
    "PURIFICATION",
    "RENATURED",
    "GENERAL_INFORMATION",
    "REFERENCE",
}

ORGS_RE = re.compile(r"^#([\d,\s]+)#")
REF_TAIL_RE = re.compile(r"\s*<[\d,\s]+>\s*$")


@dataclass
class BrendaOrganism:
    number: int
    name: str
    details: str = ""


@dataclass
class BrendaEntry:
    ec: str
    organisms: dict = field(default_factory=dict)
    recommended_name: str = ""
    systematic_name: str = ""
    synonyms: list = field(default_factory=list)
    reactions: list = field(default_factory=list)
    # Per-organism buckets: {org_num: [str, str, ...]}
    substrates_products: dict = field(default_factory=dict)
    km_values: dict = field(default_factory=dict)
    ph_optimum: dict = field(default_factory=dict)
    temperature_optimum: dict = field(default_factory=dict)


def _split_orgs(text):
    """Extract leading #N,M,...# and return (org_nums, remainder)."""
    m = ORGS_RE.match(text)
    if not m:
        return None, text
    nums = [int(n.strip()) for n in m.group(1).split(",") if n.strip()]
    return nums, text[m.end():].strip()


def _strip_refs(text):
    return REF_TAIL_RE.sub("", text).strip()


def _parse_protein(record, entry):
    nums, body = _split_orgs(record)
    if not nums:
        return
    body = _strip_refs(body)
    name = body
    details = ""
    paren_start = body.find(" (")
    if paren_start != -1 and body.endswith(")"):
        name = body[:paren_start].strip()
        details = body[paren_start + 2:-1].strip()
    for num in nums:
        entry.organisms[num] = BrendaOrganism(num, name, details)


def _parse_per_org_record(record, bucket):
    nums, body = _split_orgs(record)
    if not nums:
        return
    value = _strip_refs(body)
    if not value:
        return
    for num in nums:
        bucket.setdefault(num, []).append(value)


def _finalize_record(code, record, entry):
    record = record.strip()
    if not record:
        return
    if code == "PR":
        _parse_protein(record, entry)
    elif code == "RN":
        if not entry.recommended_name:
            entry.recommended_name = _strip_refs(record)
    elif code == "SN":
        if not entry.systematic_name:
            entry.systematic_name = _strip_refs(record)
    elif code == "SY":
        entry.synonyms.append(_strip_refs(record))
    elif code == "RE":
        entry.reactions.append(_strip_refs(record))
    elif code == "SP":
        _parse_per_org_record(record, entry.substrates_products)
    elif code == "KM":
        _parse_per_org_record(record, entry.km_values)
    elif code == "PHO":
        _parse_per_org_record(record, entry.ph_optimum)
    elif code == "TO":
        _parse_per_org_record(record, entry.temperature_optimum)


class BrendaParser:
    """Streaming parser for BRENDA flatfiles."""

    def __init__(self, path):
        self.path = path

    def iter_entries(self):
        entry = None
        current_code = None
        current_record = []

        def flush():
            nonlocal current_record
            if current_code and current_record:
                _finalize_record(current_code, " ".join(current_record), entry)
            current_record = []

        with open(self.path, encoding="utf-8", errors="replace") as fh:
            for raw in fh:
                line = raw.rstrip("\n")

                if line.startswith("ID\t"):
                    if entry is not None:
                        flush()
                        yield entry
                    ec = line[3:].strip()
                    entry = BrendaEntry(ec=ec) if self._is_valid_ec(ec) else None
                    current_code = None
                    current_record = []
                    continue

                if entry is None:
                    continue

                if line == "///":
                    flush()
                    yield entry
                    entry = None
                    current_code = None
                    current_record = []
                    continue

                if line in SECTION_HEADERS:
                    flush()
                    current_code = None
                    current_record = []
                    continue

                if not line:
                    flush()
                    continue

                if line.startswith("\t"):
                    if current_code is not None:
                        current_record.append(line.strip())
                    continue

                if "\t" in line:
                    code, _, rest = line.partition("\t")
                    code = code.strip()
                    if code in TRACKED_CODES:
                        flush()
                        current_code = code
                        current_record = [rest]
                    else:
                        flush()
                        current_code = None
                        current_record = []

            if entry is not None:
                flush()
                yield entry

    @staticmethod
    def _is_valid_ec(ec):
        parts = ec.split(".")
        if len(parts) != 4:
            return False
        return all(p == "-" or p.isdigit() for p in parts)

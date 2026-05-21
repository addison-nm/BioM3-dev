"""Parser for the UniProtKB flat-file format (Swiss-Prot and TrEMBL .dat).

A parser for both Swiss-Prot and TrEMBL, sharing a common format; the
ID-line STATUS word (``Reviewed`` vs ``Unreviewed``) distinguishes SwissProt
and TrEMBL entries and is captured faithfully so callers can branch
appropriately. Entries are ``//``-delimited; the line-code reference is
available at https://web.expasy.org/docs/userman.html.

Contract: ``iter_records(path) -> Iterator[UniProtRecord]``. Capture the
fields the source provides without BioM3-prose and minimal modification
or reshaping. Verifies sequence length.

Real mini fixtures for tests:
  tests/_data/dbio_v2/uniprot_sprot_mini.dat   (100 real Reviewed entries)
  tests/_data/dbio_v2/uniprot_trembl_mini.dat  (100 real Unreviewed entries)

Structure
---------
``iter_records`` scans the file line by line, groups each ``ID`` … ``//``
block into one multi-line entry, and hands it to ``parse_entry``.
``parse_entry`` walks the entry line by line and dispatches on the
2-letter line code to a per-code helper that updates an accumulator;
helpers follow the UniProt conventions documented at
https://web.expasy.org/docs/userman.html.

UniProtRecord fields (``record_type="uniprot"``; the dataclass-style
annotations on the class are the executable copy of this list)
The bracketed two-letter code is the source flat-file line tag.
--------------------------------------------------------------------
- ``entry_name``        [ID]    : str, ID-line mnemonic (e.g. ``001R_FRG3G``)
- ``status``            [ID]    : str, ``"Reviewed"`` (Swiss-Prot) or
                                  ``"Unreviewed"`` (TrEMBL)
- ``accessions``        [AC]    : list[str], all AC numbers in file order
- ``primary_accession`` [AC]    : str, first AC (stable identifier)
- ``dates``             [DT]    : list[str], raw DT lines (3 lines expected)
- ``description``       [DE]    : dict, structured per userman.html "The DE
                                  line": ``{rec_name, sub_name, alt_names,
                                  includes, contains, flags}`` where each name
                                  block carries ``full / short / ec_numbers /
                                  evidence`` (AltName variants Allergen/
                                  Biotech/CD_antigen/INN replace ``full`` with
                                  the variant key). Trailing ``;`` is stripped
                                  from every value (terminal ``.`` is kept);
                                  evidence lists are deduplicated in
                                  insertion order.
- ``gene_names``        [GN]    : list[str], one element per ``Key=value``
                                  sub-token (GN lines are ``;``-separated, e.g.
                                  ``Name=GRF4; OrderedLocusNames=At1g35160;``)
- ``organism``          [OS]    : str, OS text (continuation lines joined)
- ``organelle``         [OG]    : list[str], raw OG lines
- ``lineage``           [OC]    : list[str], OC taxa, top-down, period stripped
- ``taxon_id``          [OX]    : int | None, NCBI TaxID from OX
- ``hosts``             [OH]    : list[str], raw OH lines (viruses)
- ``references``        [RN]    : list[dict], one dict per RN block mapping
                                  each continuation R-code (RP/RC/RX/RG/RA/
                                  RT/RL) to its text
- ``comments``          [CC]    : list[dict], ``{"topic": str, "text": str}``
                                  per CC ``-!-`` block; ECO evidence tags kept
                                  intact; the copyright footer is not a block
- ``cross_references``  [DR]    : dict[str, list[str]], grouped by database
                                  name. Each value is a list of the raw DR
                                  line bodies (cols 6+) verbatim — the
                                  database token, ``;``-separated fields,
                                  terminal ``.``, and any optional isoform
                                  tag ``[P12345-N]`` are all preserved as
                                  written.
- ``features``          [FT]    : list[dict], ``{"type", "location",
                                  "qualifiers": {name: str}}`` from FT
- ``keywords``          [KW]    : list[str], KW terms, period stripped
- ``protein_existence`` [PE]    : str | None, raw PE text
- ``sequence``          [SQ]    : str, assembled residues (whitespace removed)
- ``sequence_length``   [ID/SQ] : int, validated equal on ID and SQ lines
- ``molecular_weight``  [SQ]    : int, from the SQ line
- ``crc64``             [SQ]    : str, CRC64 from the SQ line (as written)
- ``unparsed``          [*]     : dict[str, list[str]], any line code without
                                  a dedicated helper — captured, never dropped

Fail-loud (raises ``ParseError``)
---------------------------------
- compressed-stream truncation (via ``base.iter_lines``)
- a non-empty file not starting at an ``ID`` line, or an entry with no
  closing ``//`` at EOF (plain-file truncation)
- missing/"malformed" ``SQ`` line, or no sequence
- assembled length != ID-line length != SQ-line length
- ``Bio.SeqUtils.CheckSum.crc64(sequence)`` != SQ CRC64
"""

from __future__ import annotations

import re
from typing import ClassVar, Iterator

from Bio.SeqUtils.CheckSum import crc64

from biom3.dbio_v2.parsers.base import ParseError, Record, iter_lines

RECORD_TYPE = "uniprot"


# ===========================================================================
# Public API
# ===========================================================================

class UniProtRecord(Record):
    """One UniProtKB entry. The annotations below are the single
    executable schema (Pylance-typed); ``Record.__init__`` enforces that
    ``parse_entry`` emits exactly these keys. Storage is the field-bag.
    """

    record_type: ClassVar[str] = RECORD_TYPE

    entry_name: str
    status: str
    accessions: list[str]
    primary_accession: str | None
    dates: list[str]
    description: dict
    gene_names: list[str]
    organism: str
    organelle: list[str]
    lineage: list[str]
    taxon_id: int | None
    hosts: list[str]
    references: list[dict]
    comments: list[dict]
    cross_references: dict[str, list[str]]
    features: list[dict]
    keywords: list[str]
    protein_existence: str | None
    sequence: str
    sequence_length: int
    molecular_weight: int
    crc64: str
    unparsed: dict[str, list[str]]


def iter_records(path: str) -> Iterator[UniProtRecord]:
    """Yield one :class:`UniProtRecord` per UniProt entry in *path*.

    Reads through :func:`base.iter_lines` (fail-loud on a truncated
    compressed stream), groups each ``ID`` … ``//`` block, and delegates
    to :func:`parse_entry`. Raises ``ParseError`` if the file does not
    start at an ``ID`` line or an entry is not ``//``-terminated at EOF.
    """
    entry: list[str] = []
    seen_id = False

    for line in iter_lines(path):
        if not seen_id:
            if line.strip() == "":
                continue
            if not line.startswith("ID   "):
                raise ParseError(
                    f"{path}: expected an ID line at start of entry, got {line!r}"
                )
            seen_id = True

        entry.append(line)

        if line.rstrip("\n") == "//":
            yield parse_entry(entry)
            entry = []
            seen_id = False

    if entry:
        raise ParseError(
            f"{path}: file ended mid-entry (no closing '//') — truncated input"
        )


def parse_entry(lines: list[str]) -> UniProtRecord:
    """Parse one ``ID`` … ``//`` block into a :class:`UniProtRecord`.

    Walks the entry line by line, dispatching on the 2-letter line code,
    then validates the assembled sequence against the ID- and SQ-line
    lengths and the SQ CRC64 (raises ``ParseError`` on any mismatch).
    """
    acc = _new_accumulator()

    for line in lines:
        code = line[:2]
        if code == "//":
            break
        data = _text(line)

        if acc["_in_sq"] and code == "  ":
            acc["_seq_chunks"].append(data)
            continue

        if code == "RN":
            _h_rn(data, acc)
            continue
        if code in _REF_CONT_CODES:
            _h_ref(code, data, acc)
            continue

        handler = _DISPATCH.get(code)
        if handler is None:
            if code.strip():
                acc["unparsed"].setdefault(code, []).append(data)
            continue
        handler(data, acc)

    # Finalize buffered multi-line fields.
    _flush_cc(acc)
    acc["description"] = parse_description(acc.pop("_de_lines"))
    for ref in acc["references"]:
        for k in ("RC", "RG", "RX", "RA", "RT"):
            if k in ref:
                ref[k] = ref[k].rstrip(";")
    acc["primary_accession"] = acc["accessions"][0] if acc["accessions"] else None
    acc["organism"] = _join_wrap(acc.pop("_os")).strip()
    lineage_text = " ".join(acc.pop("_oc")).strip().rstrip(".")
    acc["lineage"] = [t.strip() for t in lineage_text.split(";") if t.strip()]
    sequence = re.sub(r"\s", "", "".join(acc.pop("_seq_chunks")))

    _validate(acc, sequence)
    acc["sequence"] = sequence

    # Drop internal bookkeeping keys before building the Record.
    id_length = acc.pop("_id_length")
    acc.pop("_cc_topic", None)
    acc.pop("_cc_text", None)
    acc.pop("_ft_qual", None)
    acc.pop("_in_sq", None)
    acc["sequence_length"] = id_length  # validated == SQ length below

    return UniProtRecord(**acc)


# ===========================================================================
# Implementation details
# ===========================================================================

_ID_RE = re.compile(r"^(?P<name>\S+)\s+(?P<status>\w+);\s+(?P<length>\d+)\s+AA\.")
_SQ_RE = re.compile(
    r"SEQUENCE\s+(?P<length>\d+)\s+AA;\s+(?P<mw>\d+)\s+MW;\s+"
    r"(?P<crc>[0-9A-Fa-f]+)\s+CRC64;"
)
_TAXID_RE = re.compile(r"NCBI_TaxID=(\d+)")
_FT_FEATURE_RE = re.compile(r"^(?P<type>\S+)\s+(?P<location>\S.*?)\s*$")

#: R-codes that continue the most recent RN reference block. The set is
#: the empirical modern UniProt vocabulary (verified against the
#: 500k-line Swiss-Prot head and both mini fixtures). Legacy ``RM``
#: (MEDLINE-ID) is deliberately excluded — modern UniProt encodes the
#: MEDLINE UI inside ``RX   MEDLINE=...;``.
_REF_CONT_CODES = {"RP", "RC", "RX", "RG", "RA", "RT", "RL"}


def _text(line: str) -> str:
    """Return the data portion of a flat-file line (columns 6+)."""
    return line[5:].rstrip("\n")


def _join_wrap(parts) -> str:
    """Join wrapped UniProt continuation lines with one space, except
    when the previous chunk ends with ``-``. UniProt wraps mid-word on a
    hyphen with no separator (e.g. ``'Ser-`` on one CC line continued by
    ``241'`` on the next is a single token ``'Ser-241'``), so naive
    space-join silently introduces wrong spaces.
    """
    if not parts:
        return ""
    out = parts[0]
    for p in parts[1:]:
        sep = "" if out.endswith("-") else " "
        out = out + sep + p
    return out


def _new_accumulator() -> dict:
    return {
        "entry_name": None, "status": None, "_id_length": None,
        "accessions": [], "dates": [], "_de_lines": [], "gene_names": [],
        "_os": [], "organelle": [], "_oc": [], "taxon_id": None, "hosts": [],
        "references": [], "comments": [], "_cc_topic": None, "_cc_text": [],
        "cross_references": {}, "features": [], "_ft_qual": None,
        "keywords": [], "protein_existence": None,
        "sequence_length": None, "molecular_weight": None, "crc64": None,
        "_seq_chunks": [], "_in_sq": False, "unparsed": {},
    }


def _validate(acc, sequence: str) -> None:
    accession = acc["accessions"][0] if acc["accessions"] else acc["entry_name"]
    if acc["entry_name"] is None:
        raise ParseError("entry has no ID line")
    if acc["crc64"] is None or acc["sequence_length"] is None:
        raise ParseError(f"{accession}: entry has no SQ line / sequence header")
    if not sequence:
        raise ParseError(f"{accession}: entry has no sequence")
    assembled = len(sequence)
    if assembled != acc["sequence_length"]:
        raise ParseError(
            f"{accession}: assembled length {assembled} != SQ length "
            f"{acc['sequence_length']}"
        )
    if acc["_id_length"] is not None and assembled != acc["_id_length"]:
        raise ParseError(
            f"{accession}: assembled length {assembled} != ID length "
            f"{acc['_id_length']}"
        )
    # Biopython returns the checksum prefixed with "CRC-"; the SQ line
    # carries the bare 16-hex value.
    computed = crc64(sequence).upper()
    if computed.startswith("CRC-"):
        computed = computed[4:]
    if computed != acc["crc64"].upper():
        raise ParseError(
            f"{accession}: CRC64 mismatch (computed {computed} != "
            f"declared {acc['crc64'].upper()})"
        )


# --- DE description grammar parser (userman.html "The DE line") ----------

_KEY_RE = re.compile(r"([A-Za-z_]+)=([^;{]+?)(?:\s*\{([^}]+)\})?\s*;")
_HEAD_RE = re.compile(r"^(RecName|SubName|AltName|Includes|Contains|Flags):\s*(.*)$")
_ALT_VARIANT = {"Allergen": "allergen", "Biotech": "biotech",
                "CD_antigen": "cd_antigen", "INN": "inn"}


def _new_de_block() -> dict:
    return {"rec_name": None, "sub_name": None, "alt_names": [],
            "includes": [], "contains": [], "flags": []}


def _new_de_standard() -> dict:
    return {"full": None, "short": [], "ec_numbers": [], "evidence": []}


def _de_segments(text: str):
    out = []
    for m in _KEY_RE.finditer(text):
        value = m.group(2).strip().rstrip(";")
        ev_text = m.group(3)
        evidence = [e.strip() for e in ev_text.split(",")] if ev_text else []
        evidence = [e for e in evidence if e]
        out.append((m.group(1), value, evidence))
    return out


def _de_extend_unique(target, items):
    for x in items:
        if x not in target:
            target.append(x)


def _de_apply_standard(name, segments):
    for key, val, ev in segments:
        if key == "Full":
            name["full"] = val
        elif key == "Short":
            name["short"].append(val)
        elif key == "EC":
            name["ec_numbers"].append(val)
        _de_extend_unique(name["evidence"], ev)


def _de_alt_from_segments(segments):
    if not segments:
        return None
    first_key, first_val, first_ev = segments[0]
    if first_key == "Full":
        alt = _new_de_standard()
        alt["full"] = first_val
        _de_extend_unique(alt["evidence"], first_ev)
        _de_apply_standard(alt, segments[1:])
        return alt
    if first_key in _ALT_VARIANT:
        alt = {_ALT_VARIANT[first_key]: first_val, "evidence": []}
        _de_extend_unique(alt["evidence"], first_ev)
        for _, _, ev in segments[1:]:
            _de_extend_unique(alt["evidence"], ev)
        return alt
    alt = _new_de_standard()
    _de_apply_standard(alt, segments)
    return alt


def parse_description(de_lines) -> dict:
    """Parse a UniProt DE block (raw lines with indentation intact) into
    a structured dict per the "The DE line" grammar at
    https://web.expasy.org/docs/userman.html.

    Output shape (always all top-level keys present)::

        {"rec_name": <NameBlock or None>,
         "sub_name": <NameBlock or None>,         # TrEMBL substitute for RecName
         "alt_names": [<AltItem>, ...],
         "includes": [<DescBlock>, ...],          # recursive — same shape
         "contains": [<DescBlock>, ...],          # recursive — same shape
         "flags":    ["Precursor" | "Fragment" | ...]}

    A standard NameBlock is ``{full, short[], ec_numbers[], evidence[]}``.
    AltName variants ``Allergen=`` / ``Biotech=`` / ``CD_antigen=`` /
    ``INN=`` are polymorphic: a single value key (``allergen``/``biotech``/
    ``cd_antigen``/``inn``) plus ``evidence`` (no ``full``/``short``/
    ``ec_numbers``). Trailing ``;`` is stripped from every captured value
    (terminal ``.`` is preserved); evidence is deduplicated in
    first-seen order.
    """
    top = _new_de_block()
    current_subsection = None
    current_name = None

    for line in de_lines:
        leading = len(line) - len(line.lstrip())
        stripped = line.strip()
        if not stripped:
            continue

        container = current_subsection if (leading > 0 and current_subsection is not None) else top
        if leading == 0:
            current_subsection = None

        head = _HEAD_RE.match(stripped)
        if head:
            kind, rest = head.group(1), head.group(2).strip()
            if kind == "Contains":
                grp = _new_de_block()
                top["contains"].append(grp)
                current_subsection = grp
                current_name = None
                continue
            if kind == "Includes":
                grp = _new_de_block()
                top["includes"].append(grp)
                current_subsection = grp
                current_name = None
                continue
            if kind == "Flags":
                container["flags"].extend(
                    f.strip().rstrip(";")
                    for f in rest.split(";") if f.strip()
                )
                continue
            if kind == "RecName":
                rec = _new_de_standard()
                container["rec_name"] = rec
                current_name = rec
                _de_apply_standard(rec, _de_segments(rest))
                continue
            if kind == "SubName":
                sub = _new_de_standard()
                container["sub_name"] = sub
                current_name = sub
                _de_apply_standard(sub, _de_segments(rest))
                continue
            if kind == "AltName":
                alt = _de_alt_from_segments(_de_segments(rest))
                if alt is not None:
                    container["alt_names"].append(alt)
                    current_name = alt
                continue
        elif current_name is not None and "full" in current_name:
            _de_apply_standard(current_name, _de_segments(stripped))
    return top


# --- per-line-code helpers -------------------------------------------------
# Each helper takes (data, acc) where `data` is the column-6+ text and
# `acc` is the entry accumulator dict, and records its field(s) faithfully.

def _h_id(data, acc):
    m = _ID_RE.match(data)
    if not m:
        raise ParseError(f"malformed ID line: {data!r}")
    acc["entry_name"] = m.group("name")
    acc["status"] = m.group("status")
    acc["_id_length"] = int(m.group("length"))


def _h_ac(data, acc):
    for token in data.split(";"):
        token = token.strip()
        if token:
            acc["accessions"].append(token)


def _h_dt(data, acc):
    acc["dates"].append(data.strip())


def _h_de(data, acc):
    acc["_de_lines"].append(data)


def _h_gn(data, acc):
    # GN lines are `Key=value; Key=value; ...;` — one element per sub-token
    # (per the spec at https://web.expasy.org/docs/userman.html, "The GN line").
    for token in data.split(";"):
        token = token.strip()
        if token:
            acc["gene_names"].append(token)


def _h_os(data, acc):
    acc["_os"].append(data.strip())


def _h_og(data, acc):
    acc["organelle"].append(data.strip())


def _h_oc(data, acc):
    acc["_oc"].append(data.strip())


def _h_ox(data, acc):
    m = _TAXID_RE.search(data)
    if m:
        acc["taxon_id"] = int(m.group(1))


def _h_oh(data, acc):
    acc["hosts"].append(data.strip())


def _h_rn(data, acc):
    acc["references"].append({"RN": data.strip()})


def _h_ref(code, data, acc):
    if not acc["references"]:
        acc["references"].append({})
    ref = acc["references"][-1]
    chunk = data.strip()
    if code in ref:
        ref[code] = _join_wrap([ref[code], chunk]).strip()
    else:
        ref[code] = chunk


def _h_cc(data, acc):
    if data.startswith("-!- "):
        _flush_cc(acc)
        topic_text = data[4:]
        if ":" in topic_text:
            topic, rest = topic_text.split(":", 1)
            acc["_cc_topic"] = topic.strip()
            acc["_cc_text"] = [rest.strip()] if rest.strip() else []
        else:
            acc["_cc_topic"] = topic_text.strip()
            acc["_cc_text"] = []
    elif data.startswith("---") or data.startswith("Copyrighted by"):
        # License footer, not an entry comment block.
        _flush_cc(acc)
        acc["_cc_topic"] = None
        acc["_cc_text"] = []
    elif acc["_cc_topic"] is not None:
        acc["_cc_text"].append(data.strip())


def _flush_cc(acc):
    if acc["_cc_topic"] is not None and acc["_cc_text"]:
        text = _join_wrap(acc["_cc_text"]).strip()
        acc["comments"].append({"topic": acc["_cc_topic"], "text": text})
    acc["_cc_text"] = []


def _h_dr(data, acc):
    # Preserve each DR line verbatim (cols 6+, trailing `.` and any optional
    # isoform tag intact). Group by database name (first ``;``-delimited
    # token) so callers can still look up by DB without losing source detail.
    line = data.strip()
    semi = line.find(";")
    if semi <= 0:
        return
    database = line[:semi].strip()
    if not database:
        return
    acc["cross_references"].setdefault(database, []).append(line)


def _h_ft(data, acc):
    if data[:1] != " ":
        m = _FT_FEATURE_RE.match(data)
        if not m:
            raise ParseError(f"malformed FT feature line: {data!r}")
        acc["features"].append(
            {
                "type": m.group("type"),
                "location": m.group("location").strip(),
                "qualifiers": {},
            }
        )
        acc["_ft_qual"] = None
        return
    if not acc["features"]:
        raise ParseError(f"FT continuation with no open feature: {data!r}")
    stripped = data.strip()
    feature = acc["features"][-1]
    if stripped.startswith("/"):
        if "=" in stripped:
            name, value = stripped[1:].split("=", 1)
            value = value.strip()
            if value.startswith('"'):
                value = value[1:]
                closed = value.endswith('"')
                if closed:
                    value = value[:-1]
                feature["qualifiers"][name] = value
                acc["_ft_qual"] = None if closed else name
            else:
                feature["qualifiers"][name] = value
                acc["_ft_qual"] = None
        else:
            feature["qualifiers"][stripped[1:]] = True
            acc["_ft_qual"] = None
    elif acc["_ft_qual"] is not None:
        name = acc["_ft_qual"]
        chunk = stripped
        closed = chunk.endswith('"')
        if closed:
            chunk = chunk[:-1]
        feature["qualifiers"][name] = _join_wrap([feature["qualifiers"][name], chunk]).strip()
        if closed:
            acc["_ft_qual"] = None


def _h_kw(data, acc):
    for token in data.rstrip().rstrip(".").split(";"):
        token = token.strip()
        if token:
            acc["keywords"].append(token)


def _h_pe(data, acc):
    acc["protein_existence"] = data.strip().rstrip(";")


def _h_sq(data, acc):
    m = _SQ_RE.search(data)
    if not m:
        raise ParseError(f"malformed SQ line: {data!r}")
    acc["sequence_length"] = int(m.group("length"))
    acc["molecular_weight"] = int(m.group("mw"))
    acc["crc64"] = m.group("crc")
    acc["_in_sq"] = True


#: Single-code handlers. RN and the RN-continuation codes are routed
#: explicitly in parse_entry (they need ordered, stateful grouping).
_DISPATCH = {
    "ID": _h_id, "AC": _h_ac, "DT": _h_dt, "DE": _h_de, "GN": _h_gn,
    "OS": _h_os, "OG": _h_og, "OC": _h_oc, "OX": _h_ox, "OH": _h_oh,
    "CC": _h_cc, "DR": _h_dr, "FT": _h_ft, "KW": _h_kw,
    "PE": _h_pe, "SQ": _h_sq,
}

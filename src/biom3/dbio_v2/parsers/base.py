"""Shared parser primitives.

Parsers in ``biom3.dbio_v2.parsers`` iterate over database entries and yield 
``Record`` instances. Parsers signal truncated or corrupt input by raising
``ParseError``. File access goes through ``iter_lines`` so a truncated or
failed (de)compression stream raises an error rather than returning a silently
short or corrupted result.

``Record`` is a generic, concrete field-bag: a dict-backed storage with
item access, ``.get``/``.keys``/``.as_dict``, attribute read, and
iteration. Each parser subclasses it (e.g. ``UniProtRecord``) and
promises a fixed field set by declaring class-level annotations::

    class UniProtRecord(Record):
        record_type: ClassVar[str] = "uniprot"
        entry_name: str
        status: str
        ...

The subclass is the executable schema for that database, and 
``Record.__init__`` enforces it — the constructed bag's
keys must exactly equal the promised set, else ``SchemaError`` is
raised (loud). A plain ``Record(...)`` with no annotations keeps working
as a pure bag (no check).
"""

from __future__ import annotations

import gzip
import json
from contextlib import contextmanager
from typing import ClassVar, IO, Iterable, Iterator, get_origin


__all__ = [
    "Record", "ParseError", "SchemaError",
    "iter_lines", "open_text",
    "dump_jsonl",
]


class ParseError(Exception):
    """Raised on any truncated, corrupt, or structurally invalid input.

    Parsers raise this (rather than returning a short/partial result) when
    a compressed stream ends early, an entry fails an integrity check
    (e.g. UniProt ``SQ`` length / CRC64 mismatch), or the file does not
    conform to the expected record structure.
    """


class SchemaError(ParseError):
    """A subclass promised a field set its constructed bag didn't match.

    Subclass of ``ParseError`` so fail-loud handlers still catch it, but
    distinguishable: this signals a parser-contract bug (wrong/forgotten
    keys), not malformed input.
    """


def _is_classvar(typ) -> bool:
    """True if an annotation denotes ``typing.ClassVar`` (object or string)."""
    if isinstance(typ, str):
        return typ.startswith("ClassVar") or typ.startswith("typing.ClassVar")
    return typ is ClassVar or get_origin(typ) is ClassVar


class Record:
    """A generic, concrete field-bag for one parsed source entry.

    Plain use is an open bag::

        rec = Record("uniprot", primary_accession="P12345",
                      status="Reviewed", sequence="MAA...")

    Parsers instead **subclass** it and promise a fixed field set with
    class-level annotations only (see the module docstring). Fields are
    reachable by item access (``rec["status"]``), ``.get``, attribute
    read (``rec.status``), and key iteration. ``record_type`` is a
    reserved constructor argument / ``ClassVar`` and is not stored as a
    field.
    """

    #: Overridden as a ``ClassVar`` on each parser subclass; an explicit
    #: ``record_type=`` constructor arg overrides it per-instance.
    record_type: ClassVar[str | None] = None

    def __init__(self, record_type: str | None = None, **fields):
        # object.__setattr__ so __getattr__ can never recurse during
        # construction/unpickling. Only set an instance override when
        # explicitly passed, otherwise the subclass ClassVar serves.
        if record_type is not None:
            object.__setattr__(self, "record_type", record_type)
        object.__setattr__(self, "_fields", dict(fields))

        promised = type(self)._promised_fields()
        if promised:
            declared = set(promised)
            got = set(self._fields)
            unknown = got - declared
            missing = declared - got
            if unknown or missing:
                raise SchemaError(
                    f"{type(self).__name__} field mismatch: "
                    f"unknown={sorted(unknown)} missing={sorted(missing)} "
                    f"(promised {list(promised)})"
                )

    @classmethod
    def _promised_fields(cls) -> tuple[str, ...]:
        """Field names this class promises, in declaration order.

        Walks the MRO collecting class-level annotations from every
        proper ``Record`` subclass (``Record`` itself and non-Record
        bases excluded), skipping ``ClassVar``-typed and ``_``-prefixed
        names. Handles both forms: a real ``typing.ClassVar`` object
        (module without ``from __future__ import annotations``) and the
        stringized annotation (module with it).
        """
        promised: list[str] = []
        seen: set[str] = set()
        for klass in cls.__mro__:
            if klass is Record or not issubclass(klass, Record):
                continue
            annotations = klass.__dict__.get("__annotations__", {})
            for name, typ in annotations.items():
                if name.startswith("_") or name in seen:
                    continue
                if _is_classvar(typ):
                    continue
                seen.add(name)
                promised.append(name)
        return tuple(promised)

    # --- mapping-style access ------------------------------------------

    def __getitem__(self, key: str):
        return self._fields[key]

    def __setitem__(self, key: str, value) -> None:
        self._fields[key] = value

    def __contains__(self, key: str) -> bool:
        return key in self._fields

    def __iter__(self) -> Iterator[str]:
        return iter(self._fields)

    def __len__(self) -> int:
        return len(self._fields)

    def get(self, key: str, default=None):
        return self._fields.get(key, default)

    def keys(self):
        return self._fields.keys()

    def values(self):
        return self._fields.values()

    def items(self):
        return self._fields.items()

    def as_dict(self) -> dict:
        """Return a shallow copy of the stored fields as a plain dict."""
        return dict(self._fields)

    def to_json(
        self,
        *,
        indent: int | None = None,
        ensure_ascii: bool = False,
        sort_keys: bool = False,
    ) -> str:
        """Serialize the record's fields to a JSON string.

        Mirrors :meth:`as_dict` (``record_type`` is a class-level tag and
        is not emitted). Default ``indent=None`` yields compact one-line
        output suitable for JSONL streaming; pass ``indent=2`` or ``4``
        for pretty-printing. The defaults match what the expectation
        JSONs under ``tests/_data/dbio_v2/parser_expectations/`` use, so
        ``json.loads(rec.to_json())`` round-trips back to ``as_dict``.
        """
        return json.dumps(
            self.as_dict(),
            indent=indent,
            ensure_ascii=ensure_ascii,
            sort_keys=sort_keys,
        )

    # --- attribute-style read access -----------------------------------

    def __getattr__(self, name: str):
        # Only invoked when normal attribute lookup fails. The guard
        # prevents recursion if _fields is not yet set (e.g. unpickling).
        if name in ("_fields", "record_type"):
            raise AttributeError(name)
        try:
            return self._fields[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    # --- dunder niceties (dataclass-like) ------------------------------

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, Record)
            and self.record_type == other.record_type
            and self._fields == other._fields
        )

    def __repr__(self) -> str:
        inner = ", ".join(f"{k}={v!r}" for k, v in self._fields.items())
        sep = ", " if inner else ""
        return f"Record(record_type={self.record_type!r}{sep}{inner})"


# ---------------------------------------------------------------------------
# Fail-loud file access
# ---------------------------------------------------------------------------

def iter_lines(path: str, encoding: str = "utf-8") -> Iterator[str]:
    """Yield text lines from a plain or gzipped file, failing loud.

    Transparently handles ``.gz`` via the stdlib ``gzip`` module, which
    verifies the trailing CRC32 and ISIZE on the way out: a truncated or
    corrupt compressed stream raises (``EOFError`` / ``gzip.BadGzipFile``
    / ``OSError``) when the consumer reads to the end. Those are
    re-raised as ``ParseError`` so callers get one well-typed failure
    instead of a silently short iteration.

    Plain (uncompressed) text has no integrity marker, so generic
    truncation of a ``.dat`` file cannot be detected here — that is the
    job of the per-parser structural/integrity checks (e.g. UniProt's
    missing ``//`` terminator and ``SQ`` length / CRC64 validation).

    Lines are yielded verbatim (newline retained); the parser is
    responsible for any stripping. Stdlib only — no ``pigz``/``zcat``
    subprocess, which is what made the legacy reader silently truncate.
    """
    opener = gzip.open if path.endswith(".gz") else open
    try:
        with opener(path, "rt", encoding=encoding) as handle:
            for line in handle:
                yield line
    except (EOFError, gzip.BadGzipFile, OSError) as exc:
        raise ParseError(f"failed reading {path}: {exc}") from exc


@contextmanager
def open_text(path: str, encoding: str = "utf-8"):
    """Context-manager variant of :func:`iter_lines` for ad-hoc reads.

    Yields a text handle (gzip-transparent). Decompression errors raised
    while the caller iterates the handle are re-raised as ``ParseError``.
    Prefer :func:`iter_lines` in parsers.
    """
    opener = gzip.open if path.endswith(".gz") else open
    try:
        with opener(path, "rt", encoding=encoding) as handle:
            yield handle
    except (EOFError, gzip.BadGzipFile, OSError) as exc:
        raise ParseError(f"failed reading {path}: {exc}") from exc


# ---------------------------------------------------------------------------
# JSON / JSONL emission
# ---------------------------------------------------------------------------

def dump_jsonl(
    records: Iterable[Record],
    stream: IO[str],
    *,
    ensure_ascii: bool = False,
) -> int:
    """Write a stream of :class:`Record` instances to *stream* as JSONL —
    one compact JSON object per line. Returns the number of records
    written so callers can verify the count without re-iterating.

    Pairs with any ``iter_records`` parser::

        from biom3.dbio_v2.parsers import dump_jsonl
        from biom3.dbio_v2.parsers.uniprot_dat import iter_records

        with open("out.jsonl", "w") as f:
            n = dump_jsonl(iter_records("uniprot_sprot.dat.gz"), f)
    """
    count = 0
    for rec in records:
        stream.write(rec.to_json(ensure_ascii=ensure_ascii))
        stream.write("\n")
        count += 1
    return count

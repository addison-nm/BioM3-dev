import gzip
import io
import json
from typing import ClassVar

import pytest

from biom3.dbio_v2.parsers import (
    ParseError,
    Record,
    SchemaError,
    dump_jsonl,
    iter_lines,
    open_text,
)


class TestRecordFieldBag:

    def test_plain_record_is_open_bag(self):
        r = Record("misc", a=1, b="two")
        assert r.record_type == "misc"
        assert r.a == 1
        assert r["b"] == "two"
        assert "a" in r and "z" not in r
        assert r.get("z", 9) == 9
        assert r.as_dict() == {"a": 1, "b": "two"}
        assert set(r.keys()) == {"a", "b"}

    def test_plain_record_no_record_type(self):
        assert Record(x=1).record_type is None

    def test_missing_attribute_raises_attribute_error(self):
        with pytest.raises(AttributeError):
            _ = Record(a=1).nope


class _Promised(Record):
    record_type: ClassVar[str] = "promised"
    alpha: str
    beta: int


class TestSchemaPromise:

    def test_exact_fields_construct(self):
        r = _Promised(alpha="x", beta=2)
        assert r.alpha == "x" and r.beta == 2
        assert r.record_type == "promised"
        assert _Promised._promised_fields() == ("alpha", "beta")

    def test_missing_field_raises_schema_error(self):
        with pytest.raises(SchemaError, match="missing"):
            _Promised(alpha="x")

    def test_unknown_field_raises_schema_error(self):
        with pytest.raises(SchemaError, match="unknown"):
            _Promised(alpha="x", beta=1, extra=9)

    def test_schema_error_is_parse_error(self):
        assert issubclass(SchemaError, ParseError)

    def test_explicit_record_type_overrides_classvar(self):
        assert _Promised(alpha="a", beta=1, record_type="override").record_type == "override"


class TestFailLoudReader:

    def test_iter_lines_plain(self, tmp_path):
        p = tmp_path / "x.txt"
        p.write_text("a\nb\nc\n")
        assert [ln.rstrip("\n") for ln in iter_lines(str(p))] == ["a", "b", "c"]

    def test_iter_lines_gzip_roundtrip(self, tmp_path):
        p = tmp_path / "x.txt.gz"
        with gzip.open(p, "wt") as fh:
            fh.write("one\ntwo\n")
        assert [ln.rstrip("\n") for ln in iter_lines(str(p))] == ["one", "two"]

    def test_truncated_gzip_raises_parse_error(self, tmp_path):
        raw = gzip.compress(b"line\n" * 5000)
        p = tmp_path / "bad.txt.gz"
        p.write_bytes(raw[: len(raw) // 2])
        with pytest.raises(ParseError):
            list(iter_lines(str(p)))

    def test_open_text_truncated_gzip_raises(self, tmp_path):
        raw = gzip.compress(b"line\n" * 5000)
        p = tmp_path / "bad2.txt.gz"
        p.write_bytes(raw[: len(raw) // 2])
        with pytest.raises(ParseError):
            with open_text(str(p)) as fh:
                fh.read()


class TestJsonEmission:

    def test_to_json_round_trips_as_dict(self):
        r = Record("misc", a=1, b="two", c=[1, 2, {"x": None}])
        assert json.loads(r.to_json()) == r.as_dict()

    def test_to_json_excludes_record_type(self):
        # record_type is a ClassVar / construct-time tag, not a field;
        # to_json mirrors as_dict and must not leak it into the payload.
        assert json.loads(_Promised(alpha="x", beta=1).to_json()) == {
            "alpha": "x", "beta": 1,
        }

    def test_to_json_compact_default_is_single_line(self):
        s = Record(a=1, b=2).to_json()
        assert "\n" not in s

    def test_to_json_indent_pretty(self):
        s = Record(a=1).to_json(indent=2)
        assert s == '{\n  "a": 1\n}'

    def test_dump_jsonl_writes_one_line_per_record_and_returns_count(self):
        recs = [Record(a=i, b="v") for i in range(3)]
        buf = io.StringIO()
        n = dump_jsonl(recs, buf)
        lines = buf.getvalue().splitlines()
        assert n == 3 and len(lines) == 3
        assert [json.loads(ln) for ln in lines] == [r.as_dict() for r in recs]

    def test_dump_jsonl_empty_stream(self):
        buf = io.StringIO()
        assert dump_jsonl(iter([]), buf) == 0
        assert buf.getvalue() == ""

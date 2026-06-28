import json
import pickle
import random

import pytest
from torch.utils.data import DataLoader

from biom3.core.dataloaders.generalized_dataloader import (
    GeneralizedRecordDataset,
    JsonlRecordStore,
    collate_to_lists,
    read_jsonl_records,
)


def _write_jsonl(path, records, blank_lines=False):
    lines = []
    for rec in records:
        lines.append(json.dumps(rec))
        if blank_lines:
            lines.append("")
    path.write_text("\n".join(lines) + "\n")
    return str(path)


RECORDS = [
    {
        "sequence": "MSIIGATRLQ",
        "fields": {"description": "protein A", "keywords": "membrane"},
    },
    {
        "sequence": "ACDEFGHIKL",
        "fields": {"description": "protein B", "keywords": "binding"},
    },
]

SCHEMA = {
    "sequence": {"from": "sequence"},
    "caption": {"compose": "fields_to_caption", "args": {"shuffle": False}},
}


class TestReadJsonl:

    def test_reads_records_and_skips_blank_lines(self, tmp_path):
        path = tmp_path / "data.jsonl"
        path.write_text(
            json.dumps(RECORDS[0]) + "\n\n" + json.dumps(RECORDS[1]) + "\n"
        )
        records = read_jsonl_records(str(path))
        assert records == RECORDS

    def test_invalid_json_reports_line_number(self, tmp_path):
        path = tmp_path / "bad.jsonl"
        path.write_text(json.dumps(RECORDS[0]) + "\n{not json}\n")
        with pytest.raises(ValueError, match=":2:"):
            read_jsonl_records(str(path))


class TestJsonlRecordStore:

    def test_len_and_getitem(self, tmp_path):
        path = _write_jsonl(tmp_path / "d.jsonl", RECORDS)
        store = JsonlRecordStore(path)
        assert len(store) == 2
        assert store[0] == RECORDS[0]
        assert store[1] == RECORDS[1]

    def test_equivalent_to_eager_reader(self, tmp_path):
        path = _write_jsonl(tmp_path / "d.jsonl", RECORDS, blank_lines=True)
        store = JsonlRecordStore(path)
        eager = read_jsonl_records(path)
        assert [store[i] for i in range(len(store))] == eager

    def test_skips_blank_lines(self, tmp_path):
        path = _write_jsonl(tmp_path / "d.jsonl", RECORDS, blank_lines=True)
        assert len(JsonlRecordStore(path)) == 2

    def test_negative_index(self, tmp_path):
        path = _write_jsonl(tmp_path / "d.jsonl", RECORDS)
        store = JsonlRecordStore(path)
        assert store[-1] == RECORDS[1]

    def test_out_of_range_raises_indexerror(self, tmp_path):
        path = _write_jsonl(tmp_path / "d.jsonl", RECORDS)
        store = JsonlRecordStore(path)
        with pytest.raises(IndexError):
            store[5]

    def test_slice_returns_list(self, tmp_path):
        path = _write_jsonl(tmp_path / "d.jsonl", RECORDS)
        store = JsonlRecordStore(path)
        assert store[:] == RECORDS

    def test_invalid_json_raises_valueerror(self, tmp_path):
        path = tmp_path / "bad.jsonl"
        path.write_text(json.dumps(RECORDS[0]) + "\n{nope}\n")
        store = JsonlRecordStore(str(path))
        with pytest.raises(ValueError):
            store[1]

    def test_scalar_fields_capture_during_scan(self, tmp_path):
        recs = [
            {"sequence": "AC", "sequence_length": 2},
            {"sequence": "DEFG", "sequence_length": 4},
        ]
        path = _write_jsonl(tmp_path / "d.jsonl", recs)
        store = JsonlRecordStore(path, scalar_fields=["sequence_length"])
        assert store.get_scalar("sequence_length") == [2, 4]
        # equivalence to reading the field per record
        assert store.get_scalar("sequence_length") == [r["sequence_length"] for r in recs]

    def test_scalar_fields_missing_value_is_none(self, tmp_path):
        recs = [{"sequence": "AC", "sequence_length": 2}, {"sequence": "DEFG"}]
        path = _write_jsonl(tmp_path / "d.jsonl", recs)
        store = JsonlRecordStore(path, scalar_fields=["sequence_length"])
        assert store.get_scalar("sequence_length") == [2, None]

    def test_get_scalar_uncaptured_field_raises(self, tmp_path):
        path = _write_jsonl(tmp_path / "d.jsonl", RECORDS)
        store = JsonlRecordStore(path)  # no scalar_fields
        with pytest.raises(KeyError):
            store.get_scalar("sequence_length")

    def test_scalar_fields_survive_pickle(self, tmp_path):
        recs = [{"sequence": "AC", "sequence_length": 2}]
        path = _write_jsonl(tmp_path / "d.jsonl", recs)
        store = JsonlRecordStore(path, scalar_fields=["sequence_length"])
        revived = pickle.loads(pickle.dumps(store))
        assert revived.get_scalar("sequence_length") == [2]

    def test_pickle_roundtrip_drops_handle(self, tmp_path):
        # Surrogate for spawn-based workers: the store must survive pickling
        # (the open file handle is not picklable) and still read afterward.
        path = _write_jsonl(tmp_path / "d.jsonl", RECORDS)
        store = JsonlRecordStore(path)
        _ = store[0]  # force a handle open
        revived = pickle.loads(pickle.dumps(store))
        assert revived._fh is None
        assert revived[1] == RECORDS[1]

    def test_backs_generalized_dataset(self, tmp_path):
        path = _write_jsonl(tmp_path / "d.jsonl", RECORDS)
        store = JsonlRecordStore(path)
        ds = GeneralizedRecordDataset(store, SCHEMA, rng=random.Random(0))
        assert len(ds) == 2
        assert ds[0]["sequence"] == "MSIIGATRLQ"
        assert ds[0]["caption"] == "DESCRIPTION: protein A. KEYWORDS: membrane."

    def test_multiworker_dataloader_reads_all_records(self, tmp_path):
        # Exercises fork-safety: each worker must use its own file handle.
        records = [
            {"sequence": f"SEQ{i}", "fields": {"description": f"protein {i}"}}
            for i in range(12)
        ]
        path = _write_jsonl(tmp_path / "d.jsonl", records)
        store = JsonlRecordStore(path)
        ds = GeneralizedRecordDataset(
            store, {"sequence": "sequence"}, rng=random.Random(0)
        )
        loader = DataLoader(
            ds, batch_size=4, num_workers=2, collate_fn=collate_to_lists
        )
        seen = []
        for batch in loader:
            seen.extend(batch["sequence"])
        assert sorted(seen) == sorted(r["sequence"] for r in records)


class TestSchemaResolution:

    def test_passthrough_and_compose(self):
        ds = GeneralizedRecordDataset(RECORDS, SCHEMA, rng=random.Random(0))
        sample = ds[0]
        assert sample["sequence"] == "MSIIGATRLQ"
        assert sample["caption"] == "DESCRIPTION: protein A. KEYWORDS: membrane."

    def test_bare_string_spec_is_passthrough(self):
        ds = GeneralizedRecordDataset(RECORDS, {"seq": "sequence"})
        assert ds[1]["seq"] == "ACDEFGHIKL"

    def test_tuple_spec_resolves_registered_name(self):
        schema = {"caption": ("fields_to_caption", {"add_label": False})}
        ds = GeneralizedRecordDataset(RECORDS, schema, rng=random.Random(0))
        assert ds[0]["caption"] == "protein A. membrane."

    def test_custom_callable_compose(self):
        def first_word(obj, args, rng):
            return obj[args["key"]].split()[0]

        schema = {"head": {"compose": first_word, "args": {"key": "sequence"}}}
        ds = GeneralizedRecordDataset(RECORDS, schema)
        assert ds[0]["head"] == "MSIIGATRLQ"

    def test_empty_schema_raises(self):
        with pytest.raises(ValueError):
            GeneralizedRecordDataset(RECORDS, {})

    def test_dict_spec_with_both_keys_raises(self):
        with pytest.raises(ValueError):
            GeneralizedRecordDataset(RECORDS, {"x": {"from": "sequence", "compose": "f"}})

    def test_dict_spec_with_neither_key_raises(self):
        with pytest.raises(ValueError):
            GeneralizedRecordDataset(RECORDS, {"x": {"args": {}}})

    def test_unknown_compose_name_raises_at_construction(self):
        with pytest.raises(KeyError):
            GeneralizedRecordDataset(RECORDS, {"x": {"compose": "nope"}})

    def test_bad_tuple_length_raises(self):
        with pytest.raises(ValueError):
            GeneralizedRecordDataset(RECORDS, {"x": ("fields_to_caption", {}, "extra")})


class TestDatasetProtocol:

    def test_len(self):
        ds = GeneralizedRecordDataset(RECORDS, SCHEMA)
        assert len(ds) == 2

    def test_collect(self):
        ds = GeneralizedRecordDataset(RECORDS, SCHEMA)
        assert ds.collect("sequence") == ["MSIIGATRLQ", "ACDEFGHIKL"]

    def test_explicit_rng_is_deterministic(self):
        schema = {"caption": ("fields_to_caption", {"shuffle": True})}
        ds_a = GeneralizedRecordDataset(RECORDS, schema, rng=random.Random(3))
        ds_b = GeneralizedRecordDataset(RECORDS, schema, rng=random.Random(3))
        assert ds_a[0] == ds_b[0]

    def test_rerolls_across_accesses(self):
        schema = {
            "caption": ("fields_to_caption", {"dropout_rates": {"keywords": 0.5,
                                                                "description": 0.5}}),
        }
        ds = GeneralizedRecordDataset(RECORDS, schema, rng=random.Random(1))
        outputs = {ds[0]["caption"] for _ in range(20)}
        assert len(outputs) > 1


class TestCollateAndDataLoader:

    def test_collate_to_lists(self):
        batch = [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]
        assert collate_to_lists(batch) == {"a": [1, 2], "b": ["x", "y"]}

    def test_collate_empty(self):
        assert collate_to_lists([]) == {}

    def test_dataloader_end_to_end(self):
        ds = GeneralizedRecordDataset(RECORDS, SCHEMA, rng=random.Random(0))
        loader = DataLoader(ds, batch_size=2, collate_fn=collate_to_lists)
        batch = next(iter(loader))
        assert batch["sequence"] == ["MSIIGATRLQ", "ACDEFGHIKL"]
        assert all("DESCRIPTION:" in cap for cap in batch["caption"])

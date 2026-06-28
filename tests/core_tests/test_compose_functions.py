import random

import pytest

from biom3.core.dataloaders import compose_functions as cf


RECORD = {
    "sequence": "MSIIGATRLQ",
    "fields": {
        "description": "Uncharacterized protein 002L",
        "keywords": "Host membrane, Membrane",
        "subcellular location": "Host membrane; Single-pass membrane protein.",
    },
}


class TestRegistry:

    def test_fields_to_caption_is_registered(self):
        assert "fields_to_caption" in cf.list_compose_functions()
        assert cf.get_compose_function("fields_to_caption") is cf.fields_to_caption

    def test_unknown_name_raises_keyerror(self):
        with pytest.raises(KeyError):
            cf.get_compose_function("does_not_exist")

    def test_duplicate_registration_raises(self):
        with pytest.raises(ValueError):
            @cf.register_compose("fields_to_caption")
            def _dupe(obj, args, rng):
                return ""


class TestPrimitives:

    def test_fields_to_items_from_dict_preserves_order(self):
        items = cf.fields_to_items({"a": "1", "b": "2"})
        assert items == [("a", "1"), ("b", "2")]

    def test_fields_to_items_from_pairs(self):
        assert cf.fields_to_items([("a", "1")]) == [("a", "1")]

    def test_dropout_default_keeps_all(self):
        items = [("a", "1"), ("b", "2")]
        assert cf.dropout_items(items, rng=random.Random(0)) == items

    def test_dropout_rate_one_drops_all(self):
        items = [("a", "1"), ("b", "2")]
        out = cf.dropout_items(items, default=1.0, rng=random.Random(0))
        assert out == []

    def test_dropout_per_key_rate(self):
        items = [("keep", "1"), ("drop", "2")]
        out = cf.dropout_items(items, rates={"drop": 1.0}, rng=random.Random(0))
        assert out == [("keep", "1")]

    def test_dropout_does_not_mutate_input(self):
        items = [("a", "1"), ("b", "2")]
        cf.dropout_items(items, default=1.0, rng=random.Random(0))
        assert items == [("a", "1"), ("b", "2")]

    def test_shuffle_reproducible_with_seed(self):
        items = [(f"k{i}", str(i)) for i in range(8)]
        a = cf.shuffle_items(items, rng=random.Random(1))
        b = cf.shuffle_items(items, rng=random.Random(1))
        assert a == b

    def test_shuffle_does_not_mutate_input(self):
        items = [(f"k{i}", str(i)) for i in range(8)]
        cf.shuffle_items(items, rng=random.Random(1))
        assert items == [(f"k{i}", str(i)) for i in range(8)]

    def test_normalize_joins_list_values(self):
        out = cf.normalize_items([("go", ["a", "b", "c"])])
        assert out == [("go", "a, b, c")]

    def test_normalize_custom_list_separator(self):
        out = cf.normalize_items([("go", ["a", "b"])], list_separator=" | ")
        assert out == [("go", "a | b")]

    def test_normalize_drops_none_and_empty(self):
        out = cf.normalize_items([("a", None), ("b", ""), ("c", []), ("d", "x")])
        assert out == [("d", "x")]

    def test_normalize_stringifies_scalars(self):
        out = cf.normalize_items([("n", 320), ("f", 1.5)])
        assert out == [("n", "320"), ("f", "1.5")]

    def test_add_labels_uppercases_key_by_default(self):
        out = cf.add_labels([("subcellular location", "host membrane")])
        assert out == ["SUBCELLULAR LOCATION: host membrane"]

    def test_add_labels_upper_spaced_transform(self):
        out = cf.add_labels([("sh3_paralog_name", "SLA1")], key_transform="upper_spaced")
        assert out == ["SH3 PARALOG NAME: SLA1"]

    def test_add_labels_title_spaced_transform(self):
        out = cf.add_labels([("sh3_paralog_name", "SLA1")], key_transform="title_spaced")
        assert out == ["Sh3 Paralog Name: SLA1"]

    def test_add_labels_custom_format_and_transform(self):
        out = cf.add_labels(
            [("name", "SH3")], label_format="[{key}] {value}", key_transform="title"
        )
        assert out == ["[Name] SH3"]

    def test_add_labels_callable_transform(self):
        out = cf.add_labels([("name", "v")], key_transform=lambda k: k[::-1])
        assert out == ["eman: v"]

    def test_add_labels_unknown_transform_raises(self):
        with pytest.raises(ValueError):
            cf.add_labels([("a", "b")], key_transform="bogus")

    def test_concatenate_adds_trailing_period(self):
        assert cf.concatenate(["a", "b"]) == "a. b."

    def test_concatenate_skips_empty_values(self):
        assert cf.concatenate(["a", "", "b"]) == "a. b."

    def test_concatenate_no_trailing_period(self):
        assert cf.concatenate(["a"], trailing_period=False) == "a"

    def test_concatenate_empty_is_empty(self):
        assert cf.concatenate([]) == ""

    def test_concatenate_no_double_period_when_value_ends_in_period(self):
        assert cf.concatenate(["host membrane.", "binding"]) == "host membrane. binding."

    def test_concatenate_value_ending_in_period_at_end(self):
        assert cf.concatenate(["a", "binding."]) == "a. binding."

    def test_concatenate_non_period_separator_keeps_value_periods(self):
        assert cf.concatenate(["a.", "b"], separator=" | ") == "a. | b."


class TestFieldsToCaption:

    def test_all_kept_with_labels_default(self):
        caption = cf.fields_to_caption(RECORD, {}, random.Random(0))
        assert caption == (
            "DESCRIPTION: Uncharacterized protein 002L. "
            "KEYWORDS: Host membrane, Membrane. "
            "SUBCELLULAR LOCATION: Host membrane; Single-pass membrane protein."
        )

    def test_add_label_false_uses_raw_values(self):
        caption = cf.fields_to_caption(RECORD, {"add_label": False}, random.Random(0))
        assert caption.startswith("Uncharacterized protein 002L. ")
        assert "DESCRIPTION" not in caption

    def test_dropout_removes_field(self):
        caption = cf.fields_to_caption(
            RECORD, {"dropout_rates": {"keywords": 1.0}}, random.Random(0)
        )
        assert "KEYWORDS" not in caption
        assert "DESCRIPTION" in caption

    def test_custom_fields_key(self):
        obj = {"annotations": {"name": "SH3"}}
        caption = cf.fields_to_caption(obj, {"fields_key": "annotations"}, random.Random(0))
        assert caption == "NAME: SH3."

    def test_shuffle_can_reorder(self):
        obj = {"fields": {f"k{i}": f"v{i}" for i in range(8)}}
        unshuffled = cf.fields_to_caption(obj, {"shuffle": False}, random.Random(0))
        outputs = {
            cf.fields_to_caption(obj, {"shuffle": True}, random.Random(s))
            for s in range(50)
        }
        assert any(o != unshuffled for o in outputs)

    def test_field_value_ending_in_period_not_doubled(self):
        obj = {"fields": {
            "location": "Host membrane; Single-pass membrane protein.",
            "description": "Uncharacterized protein 002L",
        }}
        caption = cf.fields_to_caption(obj, {"shuffle": False}, random.Random(0))
        assert ".." not in caption
        assert caption == (
            "LOCATION: Host membrane; Single-pass membrane protein. "
            "DESCRIPTION: Uncharacterized protein 002L."
        )

    def test_reproducible_with_seeded_rng(self):
        args = {"dropout_rates": {"keywords": 0.5}, "shuffle": True}
        a = cf.fields_to_caption(RECORD, args, random.Random(7))
        b = cf.fields_to_caption(RECORD, args, random.Random(7))
        assert a == b

    def test_list_valued_field_is_joined(self):
        obj = {"fields": {"gene_ontology": ["cytoplasm", "membrane"]}}
        caption = cf.fields_to_caption(obj, {"shuffle": False}, random.Random(0))
        assert caption == "GENE_ONTOLOGY: cytoplasm, membrane."

    def test_empty_field_skipped(self):
        obj = {"fields": {"protein_name": "SH3", "function": "", "subunit": None}}
        caption = cf.fields_to_caption(obj, {"shuffle": False}, random.Random(0))
        assert caption == "PROTEIN_NAME: SH3."

    def test_upper_spaced_labels_in_caption(self):
        obj = {"fields": {"sh3_paralog_name": "SLA1"}}
        caption = cf.fields_to_caption(
            obj, {"shuffle": False, "key_transform": "upper_spaced"}, random.Random(0)
        )
        assert caption == "SH3 PARALOG NAME: SLA1."

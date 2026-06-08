import random

import pytest

from biom3.core.prompt_assembly import RandomizedPromptConstructor


FRAGMENTS = {
    "name": "PROTEIN_NAME: SH3",
    "function": "FUNCTION: osmosensing",
}


class TestInit:

    def test_no_probs_defaults_empty(self):
        rpc = RandomizedPromptConstructor()
        assert rpc.probs == {}

    def test_prob_out_of_range_raises(self):
        with pytest.raises(ValueError):
            RandomizedPromptConstructor({"name": 1.5})
        with pytest.raises(ValueError):
            RandomizedPromptConstructor({"name": -0.1})

    def test_shuffle_flag_stored(self):
        assert RandomizedPromptConstructor(shuffle=True)._shuffle is True
        assert RandomizedPromptConstructor(shuffle=False)._shuffle is False


class TestShuffleMethod:

    def test_shuffle_method_sets_flag(self):
        rpc = RandomizedPromptConstructor()
        rpc.shuffle(True)
        assert rpc._shuffle is True
        rpc.shuffle(False)
        assert rpc._shuffle is False

    def test_shuffle_method_returns_self(self):
        rpc = RandomizedPromptConstructor()
        assert rpc.shuffle(True) is rpc


class TestBuildSingle:

    def test_all_retained_no_shuffle_preserves_order(self):
        rpc = RandomizedPromptConstructor({"name": 1.0, "function": 1.0})
        assert rpc.build(FRAGMENTS) == "PROTEIN_NAME: SH3. FUNCTION: osmosensing."

    def test_missing_prob_key_always_kept(self):
        rpc = RandomizedPromptConstructor({"function": 0.0})
        assert rpc.build(FRAGMENTS) == "PROTEIN_NAME: SH3."

    def test_zero_prob_dropped(self):
        rpc = RandomizedPromptConstructor({"name": 1.0, "function": 0.0})
        assert rpc.build(FRAGMENTS) == "PROTEIN_NAME: SH3."

    def test_all_dropped_returns_empty(self):
        rpc = RandomizedPromptConstructor({"name": 0.0, "function": 0.0})
        assert rpc.build(FRAGMENTS) == ""

    def test_probs_key_absent_from_fragments_ignored(self):
        rpc = RandomizedPromptConstructor({"name": 1.0, "bogus": 0.0})
        assert rpc.build(FRAGMENTS) == "PROTEIN_NAME: SH3. FUNCTION: osmosensing."

    def test_empty_fragments(self):
        rpc = RandomizedPromptConstructor()
        assert rpc.build({}) == ""

    def test_callable_alias(self):
        rpc = RandomizedPromptConstructor({"name": 1.0, "function": 1.0})
        assert rpc(FRAGMENTS) == rpc.build(FRAGMENTS)

    def test_custom_separator_and_no_trailing_period(self):
        rpc = RandomizedPromptConstructor(
            {"name": 1.0, "function": 1.0},
            separator=" | ", trailing_period=False,
        )
        assert rpc.build(FRAGMENTS) == "PROTEIN_NAME: SH3 | FUNCTION: osmosensing"


class TestBuildList:

    def test_list_returns_list(self):
        rpc = RandomizedPromptConstructor({"name": 1.0, "function": 1.0})
        out = rpc.build([FRAGMENTS, FRAGMENTS])
        assert isinstance(out, list)
        assert out == [
            "PROTEIN_NAME: SH3. FUNCTION: osmosensing.",
            "PROTEIN_NAME: SH3. FUNCTION: osmosensing.",
        ]

    def test_tuple_returns_list(self):
        rpc = RandomizedPromptConstructor({"name": 1.0, "function": 1.0})
        out = rpc.build((FRAGMENTS,))
        assert out == ["PROTEIN_NAME: SH3. FUNCTION: osmosensing."]

    def test_heterogeneous_dicts(self):
        rpc = RandomizedPromptConstructor({"function": 0.0})
        out = rpc.build([
            {"name": "PROTEIN_NAME: SH3"},
            {"name": "PROTEIN_NAME: PDZ", "function": "FUNCTION: binding"},
        ])
        assert out == ["PROTEIN_NAME: SH3.", "PROTEIN_NAME: PDZ."]


class TestShuffleBehavior:

    def test_shuffle_is_reproducible_with_seeded_rng(self):
        rpc = RandomizedPromptConstructor(shuffle=True)
        a = rpc.build(FRAGMENTS, rng=random.Random(0))
        b = rpc.build(FRAGMENTS, rng=random.Random(0))
        assert a == b

    def test_shuffle_can_reorder(self):
        frags = {f"k{i}": f"L{i}: v{i}" for i in range(8)}
        rpc = RandomizedPromptConstructor(shuffle=True)
        rng = random.Random(3)
        unshuffled = ". ".join(frags.values()) + "."
        outputs = {rpc.build(frags, rng=rng) for _ in range(50)}
        assert any(o != unshuffled for o in outputs)

    def test_no_shuffle_never_reorders(self):
        frags = {f"k{i}": f"L{i}: v{i}" for i in range(8)}
        rpc = RandomizedPromptConstructor(shuffle=False)
        rng = random.Random(3)
        expected = ". ".join(frags.values()) + "."
        for _ in range(50):
            assert rpc.build(frags, rng=rng) == expected

    def test_build_re_rolls_each_call(self):
        rpc = RandomizedPromptConstructor({f"k{i}": 0.5 for i in range(20)})
        frags = {f"k{i}": f"L{i}: v{i}" for i in range(20)}
        rng = random.Random(1)
        outputs = {rpc.build(frags, rng=rng) for _ in range(20)}
        assert len(outputs) > 1

"""Unit tests for template-based in-painting in Stage 3 ProteoScribe sampling."""

import json
import os

import numpy as np
import pytest
import torch

from tests.conftest import DATDIR
from biom3.Stage3.inpaint import (
    RUNTIME_TOKENS,
    MASK_ID,
    START_ID,
    END_ID,
    PAD_ID,
    build_template_state,
    build_sampling_path_row,
    load_inpaint_config,
    resolve_template,
)
from biom3.Stage3.run_ProteoScribe_sample import (
    load_json_config,
    convert_to_namespace,
    _resolve_inpaint_args,
)
from biom3.Stage3.io import build_model_ProteoScribe
import biom3.Stage3.sampling_analysis as Stage3_sample_tools


MINI_WEIGHTS = os.path.join(DATDIR, "models/stage3/weights/minimodel1_ds128_weights1.pth")
MINI_CONFIG = os.path.join(DATDIR, "configs/test_stage3_config_v2.json")


# ---------------------------------------------------------------------------
#  Vocabulary constants
# ---------------------------------------------------------------------------

def test_vocab_constants_match_runtime_tokens():
    assert RUNTIME_TOKENS[MASK_ID] == '-'
    assert RUNTIME_TOKENS[START_ID] == '<START>'
    assert RUNTIME_TOKENS[END_ID] == '<END>'
    assert RUNTIME_TOKENS[PAD_ID] == '<PAD>'
    assert MASK_ID == 0
    assert len(RUNTIME_TOKENS) == 29


# ---------------------------------------------------------------------------
#  build_template_state
# ---------------------------------------------------------------------------

class TestBuildTemplateState:

    def test_basic_with_start_stop(self):
        state, mask_positions = build_template_state("AC-G", seq_len=10)
        # [START, A, C, MASK, G, END, PAD, PAD, PAD, PAD]
        expected = [START_ID, 2, 3, MASK_ID, 7, END_ID, PAD_ID, PAD_ID, PAD_ID, PAD_ID]
        assert state.tolist() == expected
        assert mask_positions.tolist() == [3]

    def test_no_start_no_stop(self):
        state, mask_positions = build_template_state(
            "AC-G", seq_len=6, auto_add_start=False, auto_add_stop=False,
        )
        expected = [2, 3, MASK_ID, 7, PAD_ID, PAD_ID]
        assert state.tolist() == expected
        assert mask_positions.tolist() == [2]

    def test_start_only(self):
        state, _ = build_template_state(
            "A", seq_len=5, auto_add_start=True, auto_add_stop=False,
        )
        assert state.tolist() == [START_ID, 2, PAD_ID, PAD_ID, PAD_ID]

    def test_multiple_scattered_masks(self):
        state, mask_positions = build_template_state(
            "A--C--G", seq_len=12, auto_add_start=False, auto_add_stop=False,
        )
        # positions of '-' are 1,2,4,5
        assert mask_positions.tolist() == [1, 2, 4, 5]
        assert (state == MASK_ID).sum().item() == 4

    def test_all_masks(self):
        state, mask_positions = build_template_state(
            "----", seq_len=8, auto_add_start=False, auto_add_stop=False,
        )
        assert mask_positions.tolist() == [0, 1, 2, 3]

    def test_extra_amino_acids_frozen(self):
        # X, U, Z, B, O are valid (frozen) residues; only '-' is a mask
        state, mask_positions = build_template_state(
            "XUZBO-", seq_len=10, auto_add_start=False, auto_add_stop=False,
        )
        assert mask_positions.tolist() == [5]
        for i, ch in enumerate("XUZBO"):
            assert state[i].item() == RUNTIME_TOKENS.index(ch)

    def test_tail_is_padded(self):
        state, _ = build_template_state("A-", seq_len=20)
        # everything after the auto-added <END> is PAD
        assert torch.all(state[4:] == PAD_ID)

    def test_overflow_raises(self):
        # 9 chars + START + STOP = 11 > seq_len 10
        with pytest.raises(ValueError, match="exceeds sequence length"):
            build_template_state("A" * 9, seq_len=10)

    def test_overflow_counts_start_stop(self):
        # exactly seq_len without start/stop, but adding them overflows
        build_template_state("A" * 10, seq_len=10, auto_add_start=False, auto_add_stop=False)
        with pytest.raises(ValueError, match="exceeds sequence length"):
            build_template_state("A" * 10, seq_len=10)

    def test_unknown_char_raises(self):
        with pytest.raises(ValueError, match="Invalid template character"):
            build_template_state("AC1G", seq_len=10)

    def test_lowercase_char_raises(self):
        with pytest.raises(ValueError, match="Invalid template character"):
            build_template_state("acg", seq_len=10)

    def test_multichar_special_tokens_not_allowed(self):
        # '<' is not a valid single-char residue
        with pytest.raises(ValueError, match="Invalid template character"):
            build_template_state("A<PAD>", seq_len=20)


# ---------------------------------------------------------------------------
#  build_sampling_path_row
# ---------------------------------------------------------------------------

class TestBuildSamplingPathRow:

    def test_single_mask(self):
        mask_positions = torch.tensor([3])
        path = build_sampling_path_row(mask_positions, seq_len=10)
        assert path[3].item() == 0
        frozen = [i for i in range(10) if i != 3]
        assert torch.all(path[frozen] == -1)

    def test_scattered_masks_get_permutation(self):
        mask_positions = torch.tensor([2, 5, 7])
        path = build_sampling_path_row(mask_positions, seq_len=10)
        assert set(path[mask_positions].tolist()) == {0, 1, 2}
        frozen = [i for i in range(10) if i not in (2, 5, 7)]
        assert torch.all(path[frozen] == -1)

    def test_reproducible_with_generator(self):
        mask_positions = torch.tensor([0, 1, 2, 3, 4])
        g1 = torch.Generator().manual_seed(123)
        g2 = torch.Generator().manual_seed(123)
        a = build_sampling_path_row(mask_positions, seq_len=8, generator=g1)
        b = build_sampling_path_row(mask_positions, seq_len=8, generator=g2)
        assert torch.equal(a, b)


# ---------------------------------------------------------------------------
#  load_inpaint_config / resolve_template
# ---------------------------------------------------------------------------

class TestLoadInpaintConfig:

    def test_valid_shared(self, tmp_path):
        path = tmp_path / "inpaint.json"
        path.write_text(json.dumps({"template": "AC--G"}))
        cfg = load_inpaint_config(str(path))
        assert cfg["template"] == "AC--G"
        # booleans default to True
        assert cfg["auto_add_start"] is True
        assert cfg["auto_add_stop"] is True

    def test_valid_per_prompt(self, tmp_path):
        path = tmp_path / "inpaint.json"
        path.write_text(json.dumps({"per_prompt": {"0": "AA--", "2": "--GG"}}))
        cfg = load_inpaint_config(str(path))
        assert cfg["per_prompt"]["0"] == "AA--"

    def test_explicit_booleans_kept(self, tmp_path):
        path = tmp_path / "inpaint.json"
        path.write_text(json.dumps({
            "template": "A-", "auto_add_start": False, "auto_add_stop": False,
        }))
        cfg = load_inpaint_config(str(path))
        assert cfg["auto_add_start"] is False
        assert cfg["auto_add_stop"] is False

    def test_requires_path(self):
        with pytest.raises(ValueError, match="--inpaint_config"):
            load_inpaint_config(None)

    def test_unknown_key_raises(self, tmp_path):
        path = tmp_path / "inpaint.json"
        path.write_text(json.dumps({"template": "A-", "bogus": 1}))
        with pytest.raises(ValueError, match="unknown keys"):
            load_inpaint_config(str(path))

    def test_requires_template_or_per_prompt(self, tmp_path):
        path = tmp_path / "inpaint.json"
        path.write_text(json.dumps({"auto_add_start": True}))
        with pytest.raises(ValueError, match="template.*per_prompt|per_prompt.*template"):
            load_inpaint_config(str(path))

    def test_template_must_be_string(self, tmp_path):
        path = tmp_path / "inpaint.json"
        path.write_text(json.dumps({"template": 123}))
        with pytest.raises(ValueError, match="'template' must be a string"):
            load_inpaint_config(str(path))

    def test_bool_type_validated(self, tmp_path):
        path = tmp_path / "inpaint.json"
        path.write_text(json.dumps({"template": "A-", "auto_add_start": "yes"}))
        with pytest.raises(ValueError, match="must be a boolean"):
            load_inpaint_config(str(path))


class TestResolveInpaintArgs:

    def _parser_ns(self, **kwargs):
        import argparse
        defaults = {"inpaint": False, "inpaint_config": None}
        defaults.update(kwargs)
        return argparse.Namespace(**defaults)

    def test_disabled(self):
        import argparse
        config_args = argparse.Namespace(pre_unmask=False, sequence_length=1024)
        _resolve_inpaint_args(config_args, self._parser_ns())
        assert config_args.inpaint is False

    def test_mutually_exclusive_with_pre_unmask(self, tmp_path):
        import argparse
        path = tmp_path / "inpaint.json"
        path.write_text(json.dumps({"template": "A-"}))
        config_args = argparse.Namespace(pre_unmask=True, sequence_length=1024)
        parser_ns = self._parser_ns(inpaint=True, inpaint_config=str(path))
        with pytest.raises(ValueError, match="mutually exclusive"):
            _resolve_inpaint_args(config_args, parser_ns)

    def test_enabled_loads_config(self, tmp_path):
        import argparse
        path = tmp_path / "inpaint.json"
        path.write_text(json.dumps({"template": "AC--G"}))
        config_args = argparse.Namespace(pre_unmask=False, sequence_length=1024)
        parser_ns = self._parser_ns(inpaint=True, inpaint_config=str(path))
        _resolve_inpaint_args(config_args, parser_ns)
        assert config_args.inpaint is True
        assert config_args.inpaint_config["template"] == "AC--G"


class TestResolveTemplate:

    def test_per_prompt_override(self):
        cfg = {"template": "SHARED", "per_prompt": {"1": "OVERRIDE"}}
        assert resolve_template(1, cfg) == "OVERRIDE"

    def test_falls_back_to_shared(self):
        cfg = {"template": "SHARED", "per_prompt": {"1": "OVERRIDE"}}
        assert resolve_template(0, cfg) == "SHARED"

    def test_no_shared_no_override_raises(self):
        cfg = {"per_prompt": {"1": "X"}}
        with pytest.raises(ValueError, match="No template for prompt"):
            resolve_template(0, cfg)


# ---------------------------------------------------------------------------
#  End-to-end: frozen positions are preserved through diffusion sampling
# ---------------------------------------------------------------------------

@pytest.fixture
def mini_model_and_args():
    config_dict = load_json_config(MINI_CONFIG)
    config_args = convert_to_namespace(config_dict)
    config_args.device = "cpu"
    model = build_model_ProteoScribe(config_args)
    state_dict = torch.load(MINI_WEIGHTS, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model, config_args


# A template with frozen residues at both ends and a masked region in the middle.
_TEMPLATE = "ACDEFGHIK" + "-" * 8 + "LMNPQRST"
_SEQ_LEN = 128


@pytest.mark.parametrize("unmasking", ["random", "confidence"])
@pytest.mark.parametrize("token_strategy", ["sample", "argmax"])
def test_inpaint_preserves_frozen_positions(mini_model_and_args, unmasking, token_strategy):
    model, args = mini_model_and_args
    args = type(args)(**vars(args))
    args.token_strategy = token_strategy

    state, mask_positions = build_template_state(_TEMPLATE, seq_len=_SEQ_LEN)
    D = len(mask_positions)
    assert D == 8

    batch_size = 2
    args.diffusion_steps = D
    init = state.unsqueeze(0).repeat(batch_size, 1)
    cond = torch.randn(batch_size, args.text_emb_dim)

    torch.manual_seed(0)
    if unmasking == "confidence":
        mask_list, _, _ = Stage3_sample_tools.batch_generate_denoised_sampled_confidence(
            args=args,
            model=model,
            extract_digit_samples=init,
            extract_time=torch.zeros(batch_size).long(),
            extract_digit_label=cond,
        )
    else:
        perms = torch.stack([
            build_sampling_path_row(mask_positions, seq_len=_SEQ_LEN)
            for _ in range(batch_size)
        ])
        mask_list, _, _ = Stage3_sample_tools.batch_generate_denoised_sampled(
            args=args,
            model=model,
            extract_digit_samples=init,
            extract_time=torch.zeros(batch_size).long(),
            extract_digit_label=cond,
            sampling_path=perms,
        )

    final = mask_list[-1]  # [batch, 1, seq_len]
    init_np = init.numpy()
    frozen_mask = (state != MASK_ID).numpy()  # positions that must never change

    for b in range(batch_size):
        row = final[b, 0]
        # frozen positions are byte-identical to the template
        np.testing.assert_array_equal(
            row[frozen_mask], init_np[b][frozen_mask],
            err_msg="in-painting must not modify frozen template positions",
        )
        # the only positions that may differ from the template are mask positions
        changed = np.where(row != init_np[b])[0]
        assert set(changed.tolist()).issubset(set(mask_positions.tolist()))
        # all tokens remain in the valid range
        assert np.all(row >= 0) and np.all(row < args.num_classes)


def test_generate_inpaint_end_to_end(mini_model_and_args):
    """batch_stage3_generate_sequences with --inpaint: per-prompt templates,
    a D==0 (fully specified) prompt, frozen residues preserved in outputs.

    Exercises the rank-local results contract (single-rank: one shard merged
    via _merge_shards, as main() does for world_size == 1)."""
    from biom3.Stage3.run_ProteoScribe_sample import (
        batch_stage3_generate_sequences,
        _merge_shards,
    )

    model, args = mini_model_and_args
    args = type(args)(**vars(args))
    args.device = "cpu"
    args.num_replicas = 2
    args.unmasking_order = "random"
    args.token_strategy = "sample"
    args.sequence_length = 64
    args._rank = 0
    args._world_size = 1
    args._base_seed = 0
    args.inpaint = True
    args.inpaint_config = {
        "auto_add_start": True,
        "auto_add_stop": True,
        "per_prompt": {
            "0": "ACD--GH",   # masked interior, frozen ends -> D == 2
            "1": "MKL",        # fully specified -> D == 0
        },
    }

    torch.manual_seed(0)
    z_c = torch.randn(2, args.text_emb_dim)

    results = batch_stage3_generate_sequences(args=args, model=model, z_t=z_c)
    design_dict = _merge_shards(
        [results["rank_local_sequences"]], results["num_prompts"], args.num_replicas,
    )

    assert design_dict["_metadata"]["num_prompts"] == 2
    assert design_dict["_metadata"]["num_replicas"] == 2

    # Prompt 0: frozen prefix/suffix preserved around the two filled masks.
    for seq in design_dict["prompt_0"]:
        assert seq.startswith("ACD")
        assert seq.endswith("GH")

    # Prompt 1: no masks -> emitted verbatim for every replica.
    for seq in design_dict["prompt_1"]:
        assert seq == "MKL"

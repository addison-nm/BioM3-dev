"""Tests for ProteoScribe dict-prompt finetuning (CPU, no weights required).

Covers:
1. JSONL parsing (read_dict_prompt_jsonl) and its error handling.
2. DictPromptDataset: prompt randomization (dropout + shuffle) and sequence
   tokenization shapes / lengths.
3. dict_prompt_collate_fn: max_length padding and tensor shapes/dtypes.
4. DictPromptDataModule: deterministic split + length filtering.
5. PL_ProtARDM_Finetune: on-device z_c embedding wiring, and the design
   invariant that the frozen embedder stays out of self.parameters() and out
   of train() mode.

A real BioBERT tokenizer / weights are not needed; a fake tokenizer/embedder is
used so these run under `pytest --quick`.
"""

import json
import random
from argparse import Namespace

import numpy as np
import pytest
import torch
from torch import nn

import biom3.Stage3.preprocess as prep
from biom3.Stage3.preprocess import (
    DictPromptDataset,
    dict_prompt_collate_fn,
    encode_protein_sequence,
    read_dict_prompt_jsonl,
)
from biom3.core.prompt_assembly import RandomizedPromptConstructor


# ----------------------------- fixtures / fakes -----------------------------

def _write_jsonl(path, records, fields_key="fields", sequence_key="sequence"):
    with open(path, "w") as fh:
        for fields, seq in records:
            fh.write(json.dumps({fields_key: fields, sequence_key: seq}) + "\n")


_SAMPLE_RECORDS = [
    ({"protein_name": "PROTEIN NAME: SH3", "function": "FUNCTION: binding"}, "ACDEF"),
    ({"protein_name": "PROTEIN NAME: PDZ", "function": "FUNCTION: signaling"}, "GHIKLM"),
    ({"protein_name": "PROTEIN NAME: WW", "domain": "DOMAIN: WW"}, "NPQRS"),
    ({"protein_name": "PROTEIN NAME: SH2"}, "TVWY"),
]


class _FakeTokenizer:
    """Minimal stand-in for a HF tokenizer's batch_encode_plus."""

    def batch_encode_plus(self, prompts, *, truncation, max_length, padding,
                          return_tensors, return_attention_mask,
                          return_token_type_ids):
        assert padding == "max_length"
        assert return_tensors == "pt"
        ids = torch.zeros(len(prompts), max_length, dtype=torch.long)
        for i, p in enumerate(prompts):
            toks = [ord(c) % 100 + 1 for c in p[:max_length]]
            if toks:
                ids[i, : len(toks)] = torch.tensor(toks, dtype=torch.long)
        return {"input_ids": ids}


class _FakeAutoTokenizer:
    @staticmethod
    def from_pretrained(path):
        return _FakeTokenizer()


# ------------------------------ JSONL parsing -------------------------------

class TestReadJsonl:
    def test_parses_records(self, tmp_path):
        path = tmp_path / "d.jsonl"
        _write_jsonl(path, _SAMPLE_RECORDS)
        recs = read_dict_prompt_jsonl(str(path), fields_key="fields", sequence_key="sequence")
        assert len(recs) == len(_SAMPLE_RECORDS)
        fields, seq = recs[0]
        assert fields["protein_name"] == "PROTEIN NAME: SH3"
        assert seq == "ACDEF"

    def test_blank_lines_skipped(self, tmp_path):
        path = tmp_path / "d.jsonl"
        with open(path, "w") as fh:
            fh.write(json.dumps({"fields": {"a": "A: x"}, "sequence": "AC"}) + "\n")
            fh.write("\n")
            fh.write(json.dumps({"fields": {"a": "A: y"}, "sequence": "DE"}) + "\n")
        recs = read_dict_prompt_jsonl(str(path), fields_key="fields", sequence_key="sequence")
        assert len(recs) == 2

    def test_missing_keys_raise(self, tmp_path):
        path = tmp_path / "d.jsonl"
        with open(path, "w") as fh:
            fh.write(json.dumps({"sequence": "AC"}) + "\n")
        with pytest.raises(KeyError):
            read_dict_prompt_jsonl(str(path), fields_key="fields", sequence_key="sequence")

    def test_non_dict_fields_raise(self, tmp_path):
        path = tmp_path / "d.jsonl"
        with open(path, "w") as fh:
            fh.write(json.dumps({"fields": "not a dict", "sequence": "AC"}) + "\n")
        with pytest.raises(TypeError):
            read_dict_prompt_jsonl(str(path), fields_key="fields", sequence_key="sequence")


# --------------------------- sequence encoding ------------------------------

class TestEncodeSequence:
    def test_padded_to_image_size_squared(self):
        nums = encode_protein_sequence("ACDEF", image_size=4)
        assert len(nums) == 16  # 4*4
        # START token index is 1, END is 22 in the vocab
        assert nums[0] == 1
        assert nums[6] == 22  # <START> A C D E F <END> -> END at idx 6

    def test_gaps_stripped(self):
        assert encode_protein_sequence("AC-DE", image_size=4) == \
            encode_protein_sequence("ACDE", image_size=4)


# ----------------------------- dataset behavior -----------------------------

class TestDictPromptDataset:
    def test_shapes_and_lengths(self):
        ctor = RandomizedPromptConstructor(shuffle=False)
        ds = DictPromptDataset(_SAMPLE_RECORDS, ctor, image_size=4)
        assert len(ds) == 4
        num_seqs, prompt = ds[0]
        assert num_seqs.shape == (16,)
        assert num_seqs.dtype == torch.float32
        assert isinstance(prompt, str)
        lengths = ds.get_sequence_lengths()
        np.testing.assert_array_equal(lengths, [5, 6, 5, 4])

    def test_no_dropout_keeps_all_fragments_in_order(self):
        ctor = RandomizedPromptConstructor(shuffle=False)  # probs empty -> retain all
        ds = DictPromptDataset(_SAMPLE_RECORDS, ctor, image_size=4)
        _, prompt = ds[0]
        assert prompt == "PROTEIN NAME: SH3. FUNCTION: binding."

    def test_dropout_removes_fragments(self):
        # function retained 0% of the time -> never present
        ctor = RandomizedPromptConstructor({"function": 0.0}, shuffle=False)
        ds = DictPromptDataset(_SAMPLE_RECORDS, ctor, image_size=4)
        random.seed(0)
        for _ in range(20):
            _, prompt = ds[0]
            assert "FUNCTION" not in prompt
            assert "PROTEIN NAME: SH3" in prompt

    def test_shuffle_can_reorder(self):
        ctor = RandomizedPromptConstructor(shuffle=True)
        ds = DictPromptDataset(_SAMPLE_RECORDS, ctor, image_size=4)
        random.seed(1)
        seen = {ds[0][1] for _ in range(40)}
        # both orderings of the two retained fragments should appear
        assert len(seen) >= 2


# ------------------------------- collate fn ---------------------------------

class TestCollate:
    def test_padding_and_shapes(self):
        ctor = RandomizedPromptConstructor(shuffle=False)
        ds = DictPromptDataset(_SAMPLE_RECORDS, ctor, image_size=4)
        batch = [ds[i] for i in range(3)]
        num_seqs, input_ids = dict_prompt_collate_fn(
            batch, text_tokenizer=_FakeTokenizer(), text_max_length=32,
        )
        assert num_seqs.shape == (3, 16)
        assert num_seqs.dtype == torch.float32
        assert input_ids.shape == (3, 32)  # padded to max_length
        assert input_ids.dtype == torch.long


# ------------------------------ data module ---------------------------------

class TestDictPromptDataModule:
    def _make_dm(self, tmp_path, monkeypatch, **overrides):
        import biom3.Stage3.PL_wrapper as plw
        monkeypatch.setattr(plw, "AutoTokenizer", _FakeAutoTokenizer)
        path = tmp_path / "d.jsonl"
        _write_jsonl(path, _SAMPLE_RECORDS * 10)  # 40 records
        kwargs = dict(
            jsonl_path=str(path),
            text_model_path="unused",
            text_max_length=32,
            retention_probs={},
            shuffle_prompt=False,
            fields_key="fields",
            sequence_key="sequence",
            batch_size=4,
            num_workers=0,
            valid_size=0.25,
            seed=42,
            diffusion_steps=8,   # min_seq_length = 6
            image_size=4,
        )
        kwargs.update(overrides)
        dm = plw.DictPromptDataModule(**kwargs)
        dm.setup()
        return dm

    def test_split_sizes_and_filter(self, tmp_path, monkeypatch):
        dm = self._make_dm(tmp_path, monkeypatch)
        info = dm.split_info[0]
        # all sample sequences have raw length <= 6 == min_seq_length, none filtered
        assert len(info["train_indices"]) + len(info["val_indices"]) == 40
        assert len(info["val_indices"]) == 10  # 25% of 40

    def test_split_deterministic(self, tmp_path, monkeypatch):
        dm1 = self._make_dm(tmp_path, monkeypatch, seed=7)
        dm2 = self._make_dm(tmp_path, monkeypatch, seed=7)
        np.testing.assert_array_equal(
            dm1.split_info[0]["train_indices"],
            dm2.split_info[0]["train_indices"],
        )

    def test_dataloader_batch_shapes(self, tmp_path, monkeypatch):
        dm = self._make_dm(tmp_path, monkeypatch)
        num_seqs, input_ids = next(iter(dm.train_dataloader()))
        assert num_seqs.shape == (4, 16)
        assert input_ids.shape == (4, 32)


# --------------------------- finetune PL wrapper ----------------------------

class _FakeEmbedder(nn.Module):
    def __init__(self, text_emb_dim):
        super().__init__()
        self.lin = nn.Linear(1, text_emb_dim)

    def forward(self, input_ids):
        return self.lin(input_ids[:, :1].float())


class TestFinetuneWrapper:
    def _make_wrapper(self, text_emb_dim=8):
        from biom3.Stage3.PL_wrapper import PL_ProtARDM_Finetune
        args = Namespace()
        stub_model = nn.Linear(4, 4)
        embedder = _FakeEmbedder(text_emb_dim)
        return PL_ProtARDM_Finetune(args=args, model=stub_model, embedder=embedder), embedder

    def test_embedder_excluded_from_parameters(self):
        pl_model, embedder = self._make_wrapper()
        wrapper_param_ids = {id(p) for p in pl_model.parameters()}
        embedder_param_ids = {id(p) for p in embedder.parameters()}
        assert wrapper_param_ids.isdisjoint(embedder_param_ids)

    def test_embedder_frozen_and_eval(self):
        pl_model, embedder = self._make_wrapper()
        assert all(not p.requires_grad for p in embedder.parameters())
        pl_model.train()  # must NOT flip the frozen front-end into train mode
        assert not embedder.training

    def test_on_after_batch_transfer_produces_zc(self):
        pl_model, _ = self._make_wrapper(text_emb_dim=8)
        num_seqs = torch.zeros(3, 16)
        input_ids = torch.randint(1, 50, (3, 32))
        out = pl_model.on_after_batch_transfer((num_seqs, input_ids), 0)
        assert isinstance(out, list) and len(out) == 2
        ret_seqs, z_c = out
        assert ret_seqs is num_seqs
        assert z_c.shape == (3, 8)
        assert not z_c.requires_grad  # computed under no_grad with a frozen front-end

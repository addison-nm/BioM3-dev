"""z_t must not depend on padding length or on batch composition.

The requirement, stated concretely: take three short captions of different
lengths. Embed caption 1 in a batch with caption 2, then embed caption 1 again
in a batch with caption 3, where 3 is longer than 2. Caption 1's embedding must
be identical in both cases -- and identical again when embedded alone.

Why this can fail: captions are padded so every tensor in a batch has the same
shape. If BERT is called without an attention_mask it attends over the [PAD]
positions too, so the hidden state -- and therefore z_t -- depends on how much
padding a caption happened to receive, which under batch-relative padding means
it depends on which other captions it was batched with. The same text then
embeds differently from run to run depending only on shuffle order.

The path exercised here is the training path: the tokenizer call from
Pfam_TextSeqPairing_Dataset.caption_tokenizer, then TextEncoder -> ProjectionHead,
which is exactly how pfam_PEN_CL.forward produces z_t.

Two padding regimes are checked:
  max_length -- what training uses. Every caption pads to text_max_length, so
                batch composition cannot change the input at all.
  longest    -- padding varies with the batch. This is the regime that actually
                probes whether the mask is doing its job.
"""
import torch
from transformers import AutoTokenizer

from biom3.Stage1.model import TextEncoder, ProjectionHead

TEXT_MODEL = ("/flare/NLDesignProtein/ahowe/BioM3-dev-space/BioM3-dev/weights/LLMs/"
              "BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext")

# Three captions of deliberately different tokenized lengths.
C1 = "PROTEIN NAME: Serine protease."
C2 = "PROTEIN NAME: Serine protease. FUNCTION: Hydrolyses peptide bonds in the gut."
C3 = ("PROTEIN NAME: Serine protease. FUNCTION: Hydrolyses peptide bonds in the gut. "
      "SUBCELLULAR LOCATION: Secreted into the extracellular space. SIMILARITY: "
      "Belongs to the peptidase S1 family of trypsin-like enzymes.")


class _Args:
    text_model_path = TEXT_MODEL
    pretrained_text = True
    trainable_text = False
    bLM_n_layers_to_finetune = 0
    proj_embedding_dim = 512
    dropout = 0.1
    text_max_length = 512


def _build():
    args = _Args()
    tok = AutoTokenizer.from_pretrained(args.text_model_path)
    enc = TextEncoder(args=args).eval()
    proj = ProjectionHead(embedding_dim=768, args=args).eval()
    return args, tok, enc, proj


def _z_t(tok, enc, proj, captions, args, padding, use_mask=True):
    """Reproduce the training z_t path for a batch of captions."""
    batch = tok.batch_encode_plus(
        captions,
        truncation=True,
        max_length=args.text_max_length,
        padding=padding,
        return_tensors="pt",
        return_attention_mask=True,
        return_token_type_ids=False,
    )
    with torch.no_grad():
        h = enc(batch["input_ids"], compute_logits=False,
                attention_mask=batch["attention_mask"] if use_mask else None)
        return proj(h)


def _report(name, a, b):
    same = torch.equal(a, b)
    md = (a - b).abs().max().item()
    cos = torch.nn.functional.cosine_similarity(a, b, dim=-1).item()
    print(f"    {name:<34} exact={str(same):<5} max|diff|={md:.3e}  cos={cos:.8f}")
    return same, md


def run(verbose=True):
    args, tok, enc, proj = _build()
    lens = [len(tok(c)["input_ids"]) for c in (C1, C2, C3)]
    if verbose:
        print(f"  caption token lengths: C1={lens[0]}  C2={lens[1]}  C3={lens[2]}")
        assert lens[0] < lens[1] < lens[2], "captions must be strictly increasing in length"

    results = {}
    for padding in ("max_length", "longest"):
        if verbose:
            print(f"\n  padding='{padding}'  (with attention mask)")
        alone = _z_t(tok, enc, proj, [C1], args, padding)
        with12 = _z_t(tok, enc, proj, [C1, C2], args, padding)[0:1]
        with13 = _z_t(tok, enc, proj, [C1, C3], args, padding)[0:1]
        with123 = _z_t(tok, enc, proj, [C1, C2, C3], args, padding)[0:1]
        r = []
        r.append(_report("C1 in [C1,C2] vs [C1,C3]", with12, with13))
        r.append(_report("C1 in [C1,C2] vs alone", with12, alone))
        r.append(_report("C1 in [C1,C2,C3] vs [C1,C2]", with123, with12))
        results[padding] = r

    if verbose:
        print(f"\n  control: padding='longest' WITHOUT the mask")
        n12 = _z_t(tok, enc, proj, [C1, C2], args, "longest", use_mask=False)[0:1]
        n13 = _z_t(tok, enc, proj, [C1, C3], args, "longest", use_mask=False)[0:1]
        _report("C1 in [C1,C2] vs [C1,C3]", n12, n13)
    return results


def test_zt_is_padding_and_batch_invariant():
    results = run(verbose=False)
    # max_length is what training uses: every caption pads identically, so the
    # embedding must be bitwise identical.
    for same, md in results["max_length"]:
        assert same, f"max_length padding not bitwise invariant (max|diff|={md})"
    # longest: tensor shapes genuinely differ between batches, so allow float
    # reassociation but require agreement far below any meaningful scale.
    for same, md in results["longest"]:
        assert md < 1e-5, f"longest padding not invariant (max|diff|={md})"


if __name__ == "__main__":
    run()

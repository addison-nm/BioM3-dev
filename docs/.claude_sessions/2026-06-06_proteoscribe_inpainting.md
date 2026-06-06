# 2026-06-06 — ProteoScribe template-based in-painting

Two unrelated pieces of work in this chat. The first (a randomized prompt
constructor) landed on its own branch earlier; this note focuses on the main
feature: **template-based in-painting** for Stage 3 (ProteoScribe) sequence
generation. Branch: `feat/proteoscribe-inpainting` (off `main`).

## Context / why

Stage 3 generates sequences by masked diffusion: it starts from an all-MASK
tensor of length 1024 and unmasks one position per diffusion step. An earlier
`pre_unmask` ("post-padding") feature let a user cap output length by
pre-filling the tail `[D, 1024)` with PAD and only generating the first `D`
positions.

We wanted to generalize the *starting state*: accept a **template** mixing
fixed amino acids and the mask symbol `'-'` (e.g. `"MKA--GG--"`), freeze the
residues as context, and let ProteoScribe fill only the masked positions —
i.e. in-painting / scaffolding. Requested as a new, separate feature;
`pre_unmask` stays untouched.

## Token-vocabulary investigation (prompted by user)

Before building, the user flagged a worry: the runtime token list
(`run_ProteoScribe_sample.py`, id 0=`'-'`, id 23=`'<PAD>'`) differs from the
training list (`preprocess.py`/`eval_metrics.py`, id 0=`'*'`, id 23=`'-'`).
Traced it to the bottom — **no correctness bug**. The model operates purely on
integer ids, and those are consistent across training and inference:

- id 0 = MASK (training sets masked positions to `0` in
  `transformer_training_helper.py:453` "mask (absorbing state) → label 0";
  inference inits to 0 and the samplers test `state == 0`)
- id 1 = `<START>`, ids 2–21 = the 20 amino acids (identical order),
  id 22 = `<END>`, id 23 = PAD, ids 24–28 = X/U/Z/B/O
- `nn.Embedding(29)` row 0 was trained as the mask and inference uses row 0 as
  the mask — they match.

The only difference is the *display label* for ids 0 and 23 across three
duplicated list literals. Each decode site is internally self-consistent
(`eval_metrics` decodes training tokens only, same map on both sides). Decision
(user): centralize the *runtime* list only; leave training/eval lists alone.
**Known smell logged:** 3-way token-list duplication
(`preprocess.py:134`, `eval_metrics.py:442`, formerly `run_*sample.py:363`).

## Key mechanics that shaped the design

- The random-unmask loop `batch_generate_denoised_sampled` picks the next
  position via `current_location = (sampling_path == temp_idx).argmax(dim=-1)`.
  It already supports unmasking **arbitrary** scattered positions — feed it a
  `[batch, seq_len]` path whose masked positions hold `0..D-1` and whose frozen
  positions hold sentinel `-1` (never equals a `temp_idx ∈ [0, D)`).
- The confidence-unmask loop keys off `is_masked = (state == 0)` — already
  correct for scattered masks.
- **Consequence: the two samplers in `sampling_analysis.py` needed no changes.**
  All work was building the initial state + sampling path, wiring config/CLI,
  and batching items of equal mask-count `D`.

## What shipped

| Change | File | Notes |
|---|---|---|
| New module: vocab constant + template logic | `src/biom3/Stage3/inpaint.py` | `RUNTIME_TOKENS`, `MASK_ID/START_ID/END_ID/PAD_ID`; `build_template_state`, `build_sampling_path_row`, `load_inpaint_config`, `resolve_template`. |
| Runtime token list de-duplicated | `run_ProteoScribe_sample.py` | Now imports `RUNTIME_TOKENS` instead of the inline literal (identical content). |
| New CLI flags `--inpaint` / `--inpaint_config` | same | Mirror the `--pre_unmask` pair. |
| `_resolve_inpaint_args(config_args, parser)` | same | Loads/validates config; enforces `--inpaint` ⊥ `--pre_unmask`. Extracted from `main()` for testability. |
| `_generate_inpaint(...)` driver | same | Resolves per-prompt templates, **buckets work items by mask count `D`** so each batch shares a uniform diffusion budget; `D==0` (fully specified) emitted verbatim. |
| `batch_stage3_generate_sequences` refactor | same | Per-batch body factored into a `_process_batch` closure shared by the default/pre_unmask path and the inpaint path. `args.diffusion_steps` overridden per bucket and restored. Default + pre_unmask behaviour preserved. |
| Example config | `configs/inference/inpaint_example.json` | Shared template + `per_prompt` overrides. |
| Tests (35) | `tests/stage3_tests/test_inpaint.py` | TDD. |

## Behaviour / decisions (confirmed with user)

- Template symbols: `'-'` = MASK (id 0); amino acids (incl. extra X/U/Z/B/O)
  are frozen. No explicit interior PAD symbol — padding is auto-added.
- `auto_add_start` / `auto_add_stop`, **both default True**: wrap the template
  with `<START>`/`<END>`, then PAD the tail to the context window. "Everything
  after stop becomes padding."
- Output length = template active region (START/END/PAD stripped on output,
  exactly as before). No separate target length.
- Config supports a shared `template` plus optional `per_prompt` overrides.
- In-painting and `pre_unmask` are mutually exclusive at runtime.

## Tests

- Template parsing: `'-'`→mask placement, auto start/stop wrap + tail PAD,
  overflow (len+START+END > seq_len) raises, unknown/lowercase/multi-char
  chars raise, extra AAs frozen.
- Sampling path: `-1` sentinel on frozen, permutation of `0..D-1` on masks,
  reproducible under a seeded `torch.Generator`.
- Config: missing/unknown keys, boolean defaults, shared vs per-prompt resolve,
  `--inpaint` ⊥ `--pre_unmask` guard.
- End-to-end with the mini model fixture: frozen positions are **byte-identical**
  through both unmasking orders × both token strategies; only mask positions
  change; tokens stay in `[0, num_classes)`. Plus an integration test through
  `batch_stage3_generate_sequences` covering per-prompt overrides and `D==0`.

Results: Stage 3 **182 passed / 181 skipped**; project `--quick`
**509 passed / 216 skipped**. No regressions.

## Out of scope / follow-ups

- Training/eval token-list literals left untouched (different ordering, broader
  blast radius). Future: a single shared `TOKENS` source of truth across all
  three sites.
- Pre-existing, cosmetic: output stripping does not remove a stray id-0 (`'-'`)
  if the model ever predicts the mask token in a filled position (extremely
  unlikely after a full denoise).
- No mid-template extension / separate target length (user chose
  template = active region).
- Manual CLI smoke with real weights not run this session (entrypoint test is
  weight-gated and skipped); the integration test exercises the full
  `batch_stage3_generate_sequences` inpaint path on the mini model.

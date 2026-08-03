# Session: Multidomain Protein Finetuning — Review, Plan, Phase 1

**Date:** 2026-07-30
**Branch:** feat-multidomain-ft
**Commit:** `7badd64` — `feat: composed multidomain decoder with additive-null and load gates`
**Pre-session state:** `git checkout a9b34ba` (`docs: reorganize docs/ and correct outdated references`)

## Goal

Scope a new BioM3 feature: multidomain protein finetuning. Our collaborator Rama has an
implementation at `~/ramas-codebase/BioM3_multidomain_FT/` (single squashed commit `8253711`),
to be reviewed **skeptically** — it derives from older copies of our Stage 3 code and was
expected to carry no-op weight loads, CUDA-specific assumptions, and dynamic-padding issues.
Learn exactly what he does, then plan and begin our own implementation.

## Summary

His design is sound and worth reproducing; his code is not reusable. A K-domain protein is
decoded on **K parallel fixed-1024 canvases**, one per domain, each driven by its own
single-family ProteoScribe expert, with a per-layer cross-attention coupling whose output
projections are zero-initialised — so the composed model starts bit-exactly equal to the
independent experts.

The suspicions were all confirmed, and one is severe: **four of his five evaluation scripts
contain a silent total no-op weight load**, which makes every "coupling helps" number he has
identically zero by construction. His headline result is unverified.

Phase 1 of our implementation landed: a self-contained `biom3.Stage3.multidomain` subpackage
(coupling, composed model, checkpoint I/O, pre-flight audit) with 55 tests, all green on CPU
with no weights. The plan for phases 2–5 is at
`~/.claude/plans/the-goal-of-this-zany-cherny.md`.

## What Rama does

- **Canvas.** `<START> + domain residues + <END> + '-' pad` on a fixed 1024 grid. 1024 is
  structural — local attention asserts `L % 128 == 0`.
- **Component A — coupling.** Per-layer, per-ordered-pair directed cross linear-attention with
  its own q/k/v/out projections and per-side LayerNorm; output zero-init. 33.6M params at K=2,
  100M at K=3, 201M at K=4.
- **Component B — expert adaptation.** Experts unfrozen full-weight, regularized by
  `λ‖W − W_ref‖²` toward frozen deep copies, λ=1e-5.
- **Loss.** `Σ_d OA-ARDM_recon(d) + λ Σ_d ‖Δ_d‖²`. One shared timestep across canvases,
  independent unmasking paths per canvas.
- **Conditioning.** `y_d = α_d·z_p_d + (1−α_d)·z_c_d`, α drawn independently per domain as
  `{1: .25, 0: .25, U(0,1): .5}`.
- **Corpus.** Per-domain precomputed `z_c` `.pt` bundles intersected by UniProt accession.
- **Generation.** Fixed-length two-grid OA-ARDM; final protein is string concatenation of the
  per-canvas decodes.

There is **no** Stage 1/2 code, no BioBERT, no ESM-2 and no HuggingFace tokenizer in that repo —
`z_c` arrives precomputed. `src/source/` is a byte-identical vendored copy of our old
ProteoScribe package (verified by md5), with one file modified to guard a `pynvml` import.

## Defects found in the reference implementation

1. **Silent no-op weight load** (`eval_harness.py:39-52`, `eval_coupling_benefit.py:84-89`,
   `eval_domain_matching.py:75-79`, `eval_acceptance.py:112-116`). Each builds the legacy
   `CouplingG` (params `gamma`, `linker_emb`) while every checkpoint was trained with
   `CrossCoupling` (q/k/v/o/ln). With `strict=False` the ~200 trained tensors land in
   `unexpected_keys`, which is never inspected; only the all-zero `linker_emb` transfers. They
   also pass `train_experts=False`, so Component-B adapted experts never load either.
   **Consequence: every reported "trained_g vs g0_ablation" delta is identically zero.** Note
   the failure surfaces in `unexpected_keys`, *not* `missing_keys` — a missing-key check would
   not catch it. `blend_generate_both.py:22-42` is the only loader that is correct.
2. Wrong generation timestep `t = s` instead of `t = GRID − (L − s)` (`eval_harness.py:79`).
3. `strict=False` plus only `assert len(missing_keys) < 20`, `unexpected_keys` never inspected
   (`composed_model.py:71-72`).
4. Module aliasing (`self.N`/`self._expN`/`self.refN`/`self._refN`). `named_parameters()`
   dedups but `state_dict()` does not, so each expert is written ~4× per checkpoint. With
   `save_top_k=-1, every_n_epochs=1` over 200 epochs that is ~880 GB of checkpoints per run.
5. `linker_emb` is never passed by any caller — a permanently-zero dead parameter that forces
   `find_unused_parameters=True` on every DDP strategy.
6. Hardcoded `device="cuda"`, `dev = "cuda" if ...` in 8+ files, hardcoded ALCF paths.
7. `proteoscribe_args()` duplicated 3×, the token vocab duplicated 6×, no config files at all.
8. Diffusion timestep sampled uniformly over all 1024 positions **including pad**. For a 133 aa
   PF13193 canvas only 13.2% of loss mass lands on real residues, and ~87% of its training
   timesteps never occur at generation.
9. Upstream `z_c` computed with **dynamic** caption padding; our verified contract is
   `padding="max_length", max_length=512`. Ingesting his `.pt` bundles would import that.
10. Corpus built by bare accession intersection — domain order comes from CLI arg order, never
    checked against real spans.
11. **Assembly double-counts the shared inter-domain linker** — found late, see below.

## Two facts verified directly (not from summaries)

- **`linear_attn` pools the partner canvas.** `k.softmax(dim=-2)` then
  `einsum('bhnd,bhne->bhde')` contracts the whole partner to a per-head `d_h × d_h` matrix, so
  the coupling is **permutation-invariant over partner positions**. It carries a global pooled
  partner summary, not pairwise residue contacts. Worth restating accurately anywhere the
  "epistatic coupling" framing appears.
- **The token vocabulary differs and his index arithmetic must not be ported.** Ours is
  `['*','<START>',A..Y,'<END>','-'] + [X,U,Z,B,O]` — mask `*`=0, START=1, END=22, PAD=23 — and
  `create_token_labels` applies **no** `+1` shift for proteins
  (`transformer_training_helper.py:401-403`). His vendored copy uses a 28-token vocab with
  `real_tokens = temp_real + 1` and `TOK_START/END/PAD = 0/21/22`.

## Decisions taken

| | |
|---|---|
| Architecture | Reproduce his K-canvas + cross-coupling design |
| Expert reuse | Hard requirement — modularity / combinatorial recombination is a goal |
| Experts | **Train our own per-family experts**, so expert-training data and the multidomain corpus share one domain definition by construction |
| Corpus | **Minimal smoke corpus only.** No `dbio` builder — data acquisition is moving out of `biom3` (`dbio` is already bloated) |
| Module boundary | **`biom3.Stage3.multidomain` is fully self-contained** — no edits to any existing Stage 3 file |
| His ep169 checkpoint | Report the defects with a reproduction; he re-scores. We proceed in parallel |

## Course correction: module boundary

Work initially began on a "Phase 0" of behaviour-preserving refactors to shared Stage 3 files,
so the composed model could call into the existing decoder rather than copy it:

- `cond_diff_transformer_layer.py` — decompose `forward` into
  `embed_inputs`/`layer_input`/`layer_forward`/`readout`
- `PL_wrapper.py` — split `cond_elbo_objective`; generalize `on_after_batch_transfer` to `(B, K)`
- `preprocess.py` — hoist the collate's element encoders
- `run_ProteoScribe_finetuning.py` — extract the training-session plumbing

**All of this was reverted at the user's direction**: multidomain finetuning must be an entirely
separate module, not part of the simple `run_ProteoScribe_finetuning` path. The working tree was
returned to its committed state and the plan rewritten. Follow-up questions confirmed the
strictest reading — copy the forward loop into multidomain, write a separate PL class, and own
the run-session plumbing — accepting duplication in exchange for the two paths being free to
diverge.

**The duplication is guarded, not merely accepted.** `test_composed_matches_shared_forward`
asserts that at the additive null each canvas is `torch.equal` to the *unmodified shared*
`LinearAttentionTransformerEmbedding.forward`. Any future change to the shared decoder that the
copy does not track becomes a red test instead of silent divergence — which is precisely how
his vendored `source/` drifted.

## Phase 1: what was built

New self-contained subpackage `src/biom3/Stage3/multidomain/`:

- **`coupling.py`** — `AllPairsCoupling`. General in K from the start; he shipped both a K=2
  `CrossCoupling` and a general `AllPairsCoupling` with identical K=2 behaviour, and two classes
  for one thing is exactly what enabled defect #1. Zero-init output projections,
  `reset_to_additive_null()`, `is_additive_null()`, `parameter_report()`. No `linker_emb`.
- **`model.py`** — `MultiDomainProteoScribe`. Batch contract is stacked `[B, K, …]` rather than
  positional `(x_N, x_C)` pairs, so the data path, device transfer, DDP bucketing and the
  sampler need no K-specific branching. Experts registered exactly once (defect #4). `PAD_ID`
  derived from `create_num_seqs(['-'])` so it cannot drift from the tokenizer.
- **`io.py`** — `MultiDomainSpec` (frozen dataclass) rides *in* the checkpoint, so architecture
  is read, never guessed; `load_composed_state_dict` raises on unexpected keys **and** missing
  ones; `state_dict_fingerprint`; `load_experts` delegates to the existing
  `prepare_model_ProteoScribe(strict=True, attempt_correction=True)` with optional sha256
  provenance pins kept separate from the load.
- **`audit.py`** — `assert_additive_null` (bit-exact, plus a converse probe that perturbs an
  output projection and requires the logits to move), `audit_trainable_parameters` /
  `enforce_audit` (raise `AuditFailure`, never `assert` — assertions vanish under `python -O`),
  `expert_delta_norms`.

Beyond his `audit_trainable`, ours also rejects: prior-regularized experts sitting in a
`weight_decay > 0` optimizer group (his `AdamW(weight_decay=1e-6)` pulls the same weights toward
zero while `λ‖W−W_ref‖²` pulls them toward the reference — two conflicting priors); aliased
parameter registration; and a trainable count that disagrees with the config.

### Tests

`tests/multidomain_tests/` — 55 tests, CPU only, no weights, all under `pytest tests/ --quick`.
Suite went from a 1065/153 baseline to **1120 passed / 153 skipped**, i.e. exactly +55 with the
existing suite untouched, confirming the module boundary held.

The two that matter:

- `test_composed_matches_shared_forward` — the drift guard described above.
- `test_checkpoint_round_trip` — the defect-#1 regression. Randomizes coupling *and* experts,
  saves, rebuilds **only from the persisted spec**, loads, and asserts identical tensors *and*
  identical forward logits. Asserting logits is what catches the failure; key counting does not.

One test was written wrong and corrected: the first PAD-masking test mutated a partner's PAD
tokens and expected the other canvas's logits to be unchanged. That premise is false — changing
PAD tokens also shifts that partner's *real*-position hidden states through its own
self-attention, which legitimately propagates. The mask guarantee is now tested on
`AllPairsCoupling.cross_terms` directly, isolated from expert self-attention.

## Dataset audit and defect #11

Late in the session the plan's dataset section was checked against what Rama actually does. **It
did not match**, in five material ways — most importantly that he does **not** train on bare Pfam
envelopes. `domain_extend.py` grows each envelope outward, clamping at the neighbouring domain on
the inter-domain side and capping on open termini, turning a 76 aa PF13193 core into a 133 aa
canvas. His sha-pinned experts were therefore trained on extended sequences.

Also missing from the plan: the fragment filter (documented to skew the median from 131 to 81 if
omitted), MMseqs representative selection at `--cluster-id 0.9`, the full-length sequence source
that extension requires, and the Ledoit-Wolf `dN`/`dC` envelope metadata.

Per the user's direction these are **all deferred** — corpus construction is leaving `biom3` — and
are recorded as a table in the plan so they are not lost.

### Defect #11, verified empirically

Extension deliberately gives the shared inter-domain linker to **both** adjacent canvases
(`domain_extend.py:17-18`, "the inter-domain linker is *shared* — good, keeps full context"), but
`blend_generate_both.py:126` assembles by plain concatenation (`nseq + cseq`), treating them as
disjoint.

`domain_extend.py` was run **unmodified** on three synthetic 2-domain proteins differing only in
linker length (scratchpad only, not committed):

| linker | N canvas | C canvas | overlap | gap | `nseq+cseq` | true | error |
|---|---|---|---|---|---|---|---|
| 15 aa | 1–465 | 451–560 | 15 | 0 | 575 | 560 | **+15** |
| 50 aa | 1–475 | 476–600 | 0 | 0 | 600 | 600 | 0 |
| 70 aa | 1–475 | 496–620 | 0 | 20 | 600 | 620 | **−20** |

Each canvas is a faithful slice of its parent, so `domain_extend.py` itself is correct. The
crossover is exactly at `cap_C + cap_N = 50`: below it residues are duplicated, above it residues
are lost. His own doc describes PF13193's 133 aa median as "Pfam 76 + **shared N-linker** +
C-tail", so the duplication regime is the operative one for real data.

The error is silent — the FASTA header records `|N=465|C=110` and the log prints the summed
length, never comparing it to the parent. Training and the additive-null gate are unaffected; the
corrupted artifact is the generated FASTA that feeds BLAST/ESMFold.

**Caveat on scope:** `domain_extend.py` was genuinely executed; `blend_generate_both.py` was
**not** (it needs the expert `.bin` files, a built corpus and a trained checkpoint). The overlap
is measured; the downstream consequence is derived from one line of his code.

For us this is currently inert: with extension deferred, canvases are bare Pfam envelopes, which
do not overlap, so concatenation is correct. Defect #11 is the marker for when extension lands.

## Not implemented / deferred

- **Phases 2–5** (data path, Component B, generation, evaluation) — planned, not started.
- **Corpus pipeline** — deliberately out of scope, moving out of `biom3`.
- **The single-canvas alternative.** A design review argued that one 1024 canvas holding the whole
  protein would learn boundaries/linkers/order for zero new parameters, with ~2× better
  real-residue loss mass, and that Component A duplicates the function of the existing `y_mlp`
  global-broadcast channel. Rejected because modular per-domain experts and combinatorial domain
  recombination are project goals. Recorded in the plan; a cheap measurement that would test the
  premise (`ProteoScribeLikelihoodEstimator` with `context_mask`, no training required) is
  described there but not run.
- **Re-scoring his ep169 checkpoint** — his side will do this once the defect report lands.
- **The defect report to Rama** — not yet written. The toy reproduction above is the strongest
  form of the #11 case since it needs no cluster data and no weights.
- **Trainer construction boundary** — the planned Phase 2 runner delegates only
  `base.train_model` for DeepSpeed/DDP/XCCL setup, on the grounds that duplicating the Aurora
  process-group configuration is a real hazard. Flagged for confirmation, not yet resolved.

## Files added

```
src/biom3/Stage3/multidomain/__init__.py
src/biom3/Stage3/multidomain/coupling.py
src/biom3/Stage3/multidomain/model.py
src/biom3/Stage3/multidomain/io.py
src/biom3/Stage3/multidomain/audit.py
tests/multidomain_tests/__init__.py
tests/multidomain_tests/conftest.py
tests/multidomain_tests/test_multidomain_model.py
tests/multidomain_tests/test_multidomain_io.py
tests/multidomain_tests/test_multidomain_audit.py
```

No existing files were modified. The implementation plan lives outside the repo at
`~/.claude/plans/the-goal-of-this-zany-cherny.md`.

## Verification

```bash
source environment.sh
conda run -n biom3-env python -m pytest tests/multidomain_tests/ -q   # 55 passed
conda run -n biom3-env python -m pytest tests/ --quick -q             # 1120 passed, 153 skipped
```

# Session: Multidomain Phase 2 — Training Objective, Runner, First Real Run

**Date:** 2026-08-12
**Branch:** dev
**Commits:** `b0fdbea` (objective/trainer/runner), `249a917` (fixes from the first real run)
**Prior session:** [2026-07-30_multidomain_finetuning_phase1.md](2026-07-30_multidomain_finetuning_phase1.md)
**Pre-session state:** `git checkout 226fd7a`

## Goal

Finish the multidomain path so a composed K-canvas model can actually train:
the Lightning objective (2b) and the runner (2c). Then run it end to end against
real per-family experts and a real corpus.

## Summary

Both landed, and the composed model now trains end to end. Every architectural
number cross-checks against the reference implementation's published figures, and
the checkpoint round-trip gate — the one guarding against the silent no-op weight
load that invalidated the reference's own results — now runs against a checkpoint
produced by real training rather than one built by hand in a test.

Three bugs surfaced only by running it for real; none were reachable by unit
tests.

## What was built

### 2b — `multidomain/PL_wrapper.py`

`PL_ProtARDM_Multidomain`, following the reference objective:

- **One diffusion timestep per example, shared across all K canvases**, with an
  **independent unmasking path per canvas**, and the per-canvas OA-ARDM
  reconstruction losses summed. Sharing the timestep is what makes the assembly a
  single coupled trajectory; drawing it per canvas would mask one domain at step
  40 while telling the model it sits at step 900 for its partner. Pinned by a test
  that spies on the helpers and asserts `idx` is drawn **once** and paths **K
  times**, differing.
- **Component B**: `lambda * sum_d ||W_d - W_ref_d||^2`, references held as
  **non-persistent buffers** so they move with `.to(device)` but never enter
  `state_dict()` — each expert is stored once per checkpoint.
- **Conditioning**: `y_d = alpha_d * z_p_d + (1 - alpha_d) * z_c_d`, alpha drawn
  per `(example, domain)` on the reference's `{1: .25, 0: .25, U(0,1): .5}`
  schedule. Validation alpha stays deterministic per domain sequence.
- **Optimizer groups**: prior-regularized experts go in a `weight_decay=0` group.
  Decay pulls toward zero while the prior pulls toward the reference, and the
  pre-flight audit rejects that combination — so the grouping here is what lets
  the audit pass. A test asserts the optimizer it builds satisfies `enforce_audit`.
- **Writes `MultiDomainSpec` + an architecture fingerprint into the checkpoint**,
  closing the read-only hole in `io.py`.

### 2c — `multidomain/trainer.py`, `run_multidomain_finetuning.py`, `__main__.py`

Trainer construction is local rather than borrowed from `run_PL_training`. That
function is ~340 lines and carries a two-corpus phase-transfer mode and Pfam
step-based switching that have no meaning here and are read off `args`
unconditionally — so using it would force our config to mimic HDF5-training
fields. What *is* copied deliberately, and pinned by test under a forced-XPU
backend, are the three strategy flags that each encode a distinct Aurora failure:

- `overlap_comm=False` on XPU — overlapping reduction gives nondeterministic
  bucket ordering across ranks on oneCCL, which deadlocks on a mismatched
  collective.
- `process_group_backend='xccl'` on XPU — frameworks/2025.3.1 removed
  `oneccl-bindings-for-pytorch` and replaced `ccl` with torch's native `xccl`.
- `static_graph=True` for DDP — precomputes the gradient-bucket ready order at
  iteration 0, removing the dynamic-hook race behind the same class of deadlock.

Losing one of these fails as a *hang on Aurora*, not a red test, which is why they
are asserted explicitly.

Entry point `biom3_finetune_multidomain`; configs under `configs/stage3_multidomain/`.

## The handoff assets

`biom3_multidomain_handoff.tar.gz` (654 MB) was dropped at the repo root by
another agent. All four SHA256 sums verify. Contents:

- `data/flucbothpfamid_multidomain_PF00501_PF13193.jsonl` — 30,563 two-domain
  records (PF00501 AMP-binding N + PF13193 AMP-binding_C), schema matching
  `tests/_data/multidomain_smoke.jsonl`, regions 1-based inclusive.
- `checkpoints/PF{00501,13193}_expert.state_dict.pth` — raw consolidated state
  dicts, both single-family finetunes of `run1_base_proteoscribe`. **Partially
  trained** (72 and 100 epochs, best val_loss 0.7021 and 0.1089) — correct in
  format and architecture, suitable for wiring and smoke-testing, not converged.

Symlinked into conventional gitignored paths rather than copied:

```
weights/ProteoScribe/expert_PF00501.pth  -> data/biom3_multidomain_handoff/checkpoints/...
weights/ProteoScribe/expert_PF13193.pth  -> ...
data/multidomain/luciferase_v1.jsonl     -> data/biom3_multidomain_handoff/data/...
```

**No split manifest ships with it**, and `MultiDomainDataModule` requires one.
Built with mmseqs 18.8cc5c (`module use /opt/biom3-shared/modulefiles && module
load mmseqs2`):

```bash
biom3_stratified_cluster_split \
    --data_path data/multidomain/luciferase_v1.jsonl \
    --sequence_key sequence --source_key source \
    --cluster_method easy_cluster --min_seq_id 0.5 \
    --train_frac 0.7 --val_frac 0.2 --test_frac 0.1 --seed 0 \
    -o data/multidomain/luciferase_v1_split.json
```

14 seconds; 21,393 train / 6,113 val / 3,057 test.

**Captions in this corpus are pre-composed strings under `caption`, not structured
`fields`.** The config reads them straight through with `{"from": "caption"}`.
Consequence worth remembering: per-epoch caption re-composition and dropout — a
main reason we chose JSONL over the reference's precomputed `.pt` bundles — are
**inert on this corpus**. The mechanism is wired and tested, but this data cannot
exercise it.

## Verification

Every architectural number matches the reference implementation:

| | ours | reference (published) |
|---|---|---|
| coupling params | 33,603,584 | 33,604,096 |
| expert params | 172,372,026 | 172,372,026 |

The 512 difference is exactly the reference's `linker_emb`, a `dim`-sized
parameter deliberately omitted here because no caller ever passes it. Arithmetic:
2 ordered pairs x 16 layers x 4 projections x 512^2 = 33,554,432, plus output
biases (2x16x512 = 16,384), plus per-domain LayerNorms (2x16x1024 = 32,768).

End-to-end results:

- **Additive-null gate bit-exact** over both real 512-dim experts.
- **Freeze audit**: coupling 33,603,584 trainable, experts 0, other 0.
- **Two epochs trained**: `val_loss` 0.752 -> 0.686, `train_loss_epoch` 0.622 ->
  0.567. 33.6 M trainable / 172 M frozen / 205 M total.
- **Checkpoint round-trip**: `build_multidomain_from_checkpoint` on the written
  `last.ckpt` loaded **670 tensors, all matched** (strict, no missing, no
  unexpected); spec recovered; coupling no longer at the additive null; and
  `max |coupled - uncoupled| = 0.0014 > 0`, so training genuinely moved it and
  the load genuinely restored it.

That last item is the point of the whole exercise: it is the gate that would have
caught the reference's silent no-op load, now exercised against a real training
artifact rather than a synthetic one.

Test suite: 123 multidomain tests, full suite **1254 passed / 161 skipped**.

## Bugs found by the first real run

None were reachable by unit tests.

1. **`train_alpha: "zc"` crashed the runner.** `alpha_spec_uses_zp` does
   `float(spec)`, so the named specs must be normalized first — the single-domain
   runner does this in its arg coercion and the multidomain one did not. Fixed
   with a regression test covering `zc`/`zp`/`blend` and the rejection of `blend`
   for `eval_alpha`.
2. **`--audit_only` loaded the entire corpus.** The pre-flight audit is a property
   of the model alone, so requiring a JSONL, a validated split manifest and a z_p
   precompute to run it was wrong. The additive-null milestone is now reachable
   with nothing but the experts.
3. **No way to bound train batches**, so a smoke run could not reach validation or
   checkpointing. Added `--limit_train_batches`.

Also cleaned up: the shipped config no longer inherits
`stage3_training/machines/_spark.json`. It carries only `device: "cuda"` and
`devices_per_node`, and `device` is actively misleading here since the accelerator
is resolved from `biom3.backend.device`. Verified the config now loads with no
ignored keys.

## Testing convention

Unit tests deliberately instantiate **miniature** models — 32-dim, depth 2,
`image_size=8`, plus a 12-record fixture — so no weights ship with the test tree.
The full suite runs in ~6 s for the multidomain package with no GPU and no
downloaded weights. Nothing from the handoff is needed to run tests; it is for
smoke runs only.

## Lingering issues

- **`biom3_multidomain_handoff.tar.gz` (654 MB) is untracked at the repo root and
  is NOT gitignored.** A stray `git add -A` would try to commit it. It should be
  deleted or `*.tar.gz` added to `.gitignore`.
- **The experts are partially trained**, so nothing here says anything about
  whether cross-domain coupling actually helps. That is the Phase 5 evaluation
  question, and it needs converged experts plus the ablation arms (zero-coupling,
  mismatched partner, and independent-experts-and-concatenate — the baseline the
  reference never ran).
- **Component B is implemented but never exercised on real data.** The
  `finetune_multidomain_ab_v1.json` config exists; no run has used it.
- **Generation (Phase 4) is not built.** No composed sampler, so nothing can
  produce sequences from a trained multidomain checkpoint yet.
- **The defect report to the collaborator is still unwritten.** The toy
  reproduction of the linker double-counting defect (from the 2026-07-30 session)
  was scratchpad-only and no longer exists.
- **A per-epoch caption-recomposition corpus does not exist.** Until one does, the
  JSONL-over-`.pt` choice is unvalidated in the one respect that motivated it.

## Verification commands

```bash
source environment.sh
conda run -n biom3-env python -m pytest tests/multidomain_tests/ -q   # 123 passed
conda run -n biom3-env python -m pytest tests/ --quick -q             # 1254 / 161

# Audit only — needs the experts, nothing else
conda run -n biom3-env python -m biom3.Stage3.multidomain \
    --config_path configs/stage3_multidomain/finetune_multidomain_luciferase_v1.json \
    --run_id audit_smoke --audit_only

# Short training run that reaches validation and checkpointing
conda run -n biom3-env python -m biom3.Stage3.multidomain \
    --config_path configs/stage3_multidomain/finetune_multidomain_luciferase_v1.json \
    --run_id smoke_train --epochs 2 --batch_size 2 \
    --limit_train_batches 0.0005 --limit_val_batches 0.001
```

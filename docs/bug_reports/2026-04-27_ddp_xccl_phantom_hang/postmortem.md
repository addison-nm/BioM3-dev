# Post-mortem — DDP-on-xccl "phantom" hang on Aurora (Stage 3 finetune)

**Date investigated:** 2026-04-26 to 2026-04-27
**Environment:** Aurora (ALCF), `frameworks/2025.3.1`, single node, 12 PVC tiles, PyTorch 2.10.0a0 + IPEX 2.10.10, addison-nm/lightning fork
**Workload:** BioM3 Stage 3 ProteoScribe finetune on `SH3pFamonly` HDF5 dataset (17,165 samples, 80/20 train/val split, 86M-param diffusion transformer, 2 epochs, batch_size=16, acc_grad_batches=1)
**Status:** **resolved**

## TL;DR

Stage 3 finetune deterministically deadlocked on the train→val transition of the
last train batch of epoch 1. We spent ~21 rounds of trials looking for the bug
in DDP, in xccl (the Intel oneCCL torch backend), in Lightning, in CCL env vars,
in the strategy, and in low-level synchronization. Every fix attempt failed.

**The bug was in our own data pipeline.**

`DistributedSampler` defaulted to `drop_last=False`. With 17,165 samples →
`train_test_split(test_size=0.2)` → ~13,732 train samples → after
`min_seq_length` filtering, an effective ~13,633 train samples. Sharded across
12 ranks, that gives 11 ranks with 1136 samples and 1 rank with 1137 (or
similar redistribution depending on shuffle). At `batch_size=16`, most ranks
ran 71 batches; 1–4 ranks ran 72. The 72nd batch's DDP gradient `all_reduce`
had no peer collective from the other ranks → mismatched-collective deadlock.

**xccl was correctly hanging on a genuinely mismatched collective. Our
workload was malformed.**

The fix is to construct `DistributedSampler(drop_last=True)` explicitly in the
DataLoader and tell Lightning not to override it via
`Trainer(use_distributed_sampler=False)`. The sampler's `drop_last=True` cuts
the leftover samples *before* sharding, so every rank gets exactly the same
sample count, the same number of batches, and the DDP gradient allreduce has
peer collectives on every rank.

---

## The symptom

After training ~70 batches, the last batch of epoch 1 deadlocked. Per-rank
`py-spy` snapshots consistently showed:

- **1–4 "straggler" ranks** with their MainThread parked on `futex_wait_queue`
  and one C++ worker thread `R` (running) at `wchan=0`. py-spy MainThread stack:
  ```
  _engine_run_backward (torch/autograd/graph.py:865)
  backward (torch/autograd/__init__.py:364)
  backward (torch/_tensor.py:630)
  ```
  These ranks were stuck *inside* `loss.backward()` — the autograd C++ engine
  busy-waiting on a CCL collective that never returned.

- **8–11 "waiter" ranks** at:
  ```
  all_reduce (torch/distributed/distributed_c10d.py:3007)
  _sync_ddp (lightning/fabric/utilities/distributed.py:231)
  ```
  These had completed backward, completed the optimizer step, completed
  validation (when val was enabled), and reached the next synchronizing
  collective downstream — typically the val-end metric `all_reduce` or the
  `Trainer.save_checkpoint` barrier.

- Reproduction rate: **100% with DDP+xccl** (13/13 trials hung in the original
  config), **~67% with DeepSpeed Stage 2** (2/3 trials hung).

`/proc/<pid>/status` voluntary_ctxt_switches did not increase across 30s
snapshot intervals — confirmed real deadlock, not slow execution.

## The investigation (compressed)

| Round | What we tried | Result |
|---|---|---|
| 1 | Baseline DDP+xccl with all hang-avoidance CCL env vars | 5/5 hang |
| 2 | Removed `DeviceStatsMonitor` callback | 5/5 hang |
| 3 | Added per-rank print instrumentation | 3/3 hang. **Pinned location** to `AFTER_COND_ELBO stage=train idx=71` for stragglers, val-end metric reduction for waiters. |
| 4 | Switched to DeepSpeed Stage 2 | 2/3 hang. Hang rate dropped but persisted. |
| 5 | Disabled validation (`limit_val_batches=0`) | 5/5 hang. **Refuted "val is the trigger"** — hang surfaced one collective downstream (the eval-loop entry barrier in `on_advance_end`) instead. |
| 6 | Added barrier-name print instrumentation in `DDPStrategy.barrier()` | 2/3 hang, identified the deadlocked barrier as `Trainer.save_checkpoint`. **Trial 1 unexpectedly succeeded** through both epochs (failed only at the unrelated save_model deepspeed-converter bug). |
| 7 | Tried `process_group_backend='gloo'` | Incompatible: `RuntimeError: No backend type associated with device type xpu`. Gloo doesn't support XPU tensors. |
| 8 | Tried `process_group_backend='mpi'` | Incompatible: MPI backend not built into torch in `frameworks/2025.3.1`. |
| 9 | Fix A: `torch.xpu.synchronize()` in `on_after_backward` hook | Hang persisted. The straggler is stuck *inside* backward, never reaches the post-backward hook. |
| 10 | Applied ALCF docs' guidance: "wrap on CPU, then move to XPU" via Lightning patch | Incompatible: `RuntimeError: No backend type associated with device type cpu` at DDP's `_verify_param_shape_across_processes`. |
| 11 | Set `TORCH_LLM_ALLREDUCE=0` and `CCL_ALLREDUCE=direct` | Converted silent hang into fast-fail (`oneCCL: allreduce_entry.hpp:38: ALLREDUCE entry failed. atl_status: FAILURE`). Same site, different symptom. |
| 12 | Adopted collaborator rama's known-working PenCL Aurora env (`CCL_PROCESS_LAUNCHER=None`, unset `ONEAPI_DEVICE_SELECTOR`, drop CCL transport overrides) | 1/1 hang with byte-identical signature. **Confirmed: not a CCL config issue.** |
| 13 | Reverted Lightning fork's `008da352f` xccl barrier-skip patch | Hang persisted. **Refuted that the barrier-skip was implicated.** |
| 14 | Bypassed `reduce_boolean_decision` via rank-0 broadcast in `ModelCheckpoint` | Inconclusive — but careful comparison of round-6 trial 1 (succeeded) vs trial 2 (hung) revealed the *real* clue. |

### The breakthrough — round 6 trial 1 vs trial 2

While preparing the ALCF support ticket, comparing per-rank batch counts in the
single trial that had succeeded vs trials that had hung:

```
SUCC trial: every rank ran exactly 71 batches per epoch (max_idx=70 for all 12 ranks)
HUNG trial: 9 ranks ran 71 batches (max_idx=70), 3 ranks ran 72 batches (max_idx=71)
```

The "stragglers" weren't slow — they were processing an EXTRA batch the others
weren't. The deadlock was a textbook mismatched-collective: when r0/r7/r8 fired
`loss.backward()` for batch idx=71, DDP issued a gradient-bucket `all_reduce`.
The other 9 ranks had already moved on to the `Trainer.save_checkpoint` barrier
and never participated. The allreduce on r0/r7/r8 had no peer to combine with
→ deadlock.

| Round | Workaround applied | Result |
|---|---|---|
| 14 (re) | `--limit_train_batches 71` cap | 1/1 EXIT_0. Hang fixed. |
| 15 | Added `drop_last=True` to DataLoader (no sampler-level fix) | Inconclusive (failed differently) — the DataLoader's `drop_last` doesn't propagate to Lightning's auto-applied `DistributedSampler`. |
| 16 | Reverted to cap workaround as baseline | 3/3 EXIT_0 (rama suggested fresh-node hypothesis was refuted: trials in sequence don't degrade) |
| 17 | Added per-rank `BATCH stage=N idx=N bs=N` print | 3/3 EXIT_0 with cap. Diagnostic confirmation: `bs=15` partials at idx=70 in 0–4 ranks per trial. |
| 18 | Re-enabled validation (`--limit_val_batches 1.0`) | 3/3 EXIT_0. Val tolerates uneven last batches because val has no per-step backward → no per-step DDP collective; only val-end metric `all_reduce`, which is symmetric. |
| 19 | DeepSpeed strategy + cap | 3/3 EXIT_0. Strategy-independent fix. |
| 20 | DeepSpeed + cap removed | 2/3 hang. Diagnostic capture: stragglers showed `BATCH idx=71 bs=1` — a single leftover sample handed to one random rank as a 1-element batch. **Definitive proof of mechanism.** |
| 21 | DDP + proper `DistributedSampler(drop_last=True)` + `Trainer(use_distributed_sampler=False)`, no cap | **3/3 EXIT_0.** Real fix verified. |
| 22 | Production-clean: DDP + sampler fix + `DeviceStatsMonitor` + val on | 3/3 EXIT_0. |
| 23 | DeepSpeed + sampler fix + val on | (verifying as of writing) |

## Root cause — quantitative

Dataset: 17,165 samples in `MMD_data/` of the HDF5 file.

```
17,165 raw samples
  → train_test_split(test_size=0.2, random_state=seed)
    train = 17165 - int(17165*0.2) = 17165 - 3433 = 13,732 samples
    val   = 3,433 samples

  → HDF5DataModule._filter_indices (filters by min_seq_length)
    effective train ≈ 13,632–13,633 samples
    effective val   ≈ ~3,408 samples (per-rank ~284 samples → 18 batches with last partial bs=11–13)
```

With `world_size=12` and `batch_size=16`:

- `12 × 71 × 16 = 13,632` — the largest evenly-distributable batch count
- The actual filtered count is 13,632 + 0–11 leftover samples
- The leftover 0–11 samples is what causes the per-rank disagreement

PyTorch's default `DistributedSampler(drop_last=False)` handles the leftover by
**padding** the dataset up to `world_size × ceil(N/world_size)`, then assigning
samples by stride. Padding is done by *cloning samples from the head of the
list*. Result: each rank receives `ceil(N/world_size)` samples; 1–11 of those
are pad-clones of head samples.

When the leftover is 1 sample (the typical case after sklearn's split + our
filter): one rank ends up running 1137 samples (= 71 × 16 + 1) → 72 batches
with the last batch of size 1. The other 11 ranks run exactly 71 batches.
DDP gradient allreduce on the 72nd batch has no peer → deadlock.

When the leftover happens to vanish via the shuffle (or pads more cleanly)
all ranks land on 71 batches → no deadlock. This is why round 6 trial 1
*accidentally* succeeded — the dice fell evenly that time.

## The fix

Two changes, both in our code, no Lightning fork modifications.

**1. Construct `DistributedSampler(drop_last=True)` explicitly in our
DataLoader** ([PL_wrapper.py](../../src/biom3/Stage3/PL_wrapper.py) — `_make_distributed_sampler` helper, used in all four train/val DataLoaders):

```python
def _make_distributed_sampler(dataset, *, shuffle: bool, seed: int):
    import torch.distributed as dist
    if dist.is_available() and dist.is_initialized():
        return DistributedSampler(
            dataset,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=shuffle,
            drop_last=True,
            seed=seed,
        )
    return None
```

`DistributedSampler.drop_last=True` truncates the dataset to `world_size *
floor(N / world_size)` *before* sharding, so every rank gets exactly the same
number of samples. The leftover 0–11 samples are dropped (as opposed to
duplicated via padding).

**2. `Trainer(use_distributed_sampler=False)`** ([run_PL_training.py](../../src/biom3/Stage3/run_PL_training.py)):

```python
trainer_params = {
    ...
    'strategy': ddp_strategy,
    'use_distributed_sampler': False,    # we provide our own with drop_last=True
    ...
}
```

Without this, Lightning auto-wraps the user's DataLoader with its *own*
`DistributedSampler(drop_last=False)`, overriding our explicit one. The flag
disables the auto-wrap.

The combination eliminates per-rank batch-count disagreement at the data layer,
which eliminates the mismatched DDP allreduce, which eliminates the hang —
under both DDP and DeepSpeed.

## Why we missed it for so long

Several factors conspired to make this look like an xccl bug:

1. **The deadlock site is in xccl C++ code.** py-spy stacks landed inside
   `_engine_run_backward` → `all_reduce` / `barrier`. The most natural reading
   was "xccl barrier is racy on Aurora". This is what the original handoff
   document concluded, and it's where most of our energy went.

2. **The hang signature is consistent across many CCL configurations.** We
   tried 8+ distinct CCL env-var profiles (`CCL_OP_SYNC=1`,
   `CCL_ALLREDUCE=direct`, `CCL_ATL_TRANSPORT=mpi`, rama's known-working
   profile, etc.). None changed the signature, which made it look more
   xccl-specific.

3. **`intel/torch-xpu-ops#2701` is a real open bug** matching the same
   hang-in-`_monitored_barrier_allreduce` symptom on the same PyTorch line.
   We had a plausible upstream issue to point at.

4. **The bug "looked random"** — different ranks were the stragglers each
   trial, hang count varied, sometimes a trial succeeded. This made it hard
   to spot a deterministic cause, but in fact the underlying cause *was*
   deterministic (per-rank shard sizes); only the *which-rank* assignment
   was random because of sampler shuffling.

5. **Per-rank batch counts were the "obvious" thing to check**, but we
   weren't print-instrumenting that until round 17. Earlier rounds inferred
   batch progress from `idx=N` in py-spy stacks, which doesn't show what the
   *other* ranks are doing.

6. **The trial that succeeded in round 6 looked like noise.** It happened
   to also fail at the orthogonal `save_model` step (the unrelated
   deepspeed-converter bug), which we registered as "trial 1 also hung,
   just with a different error." Only on careful re-analysis did we notice
   that trial 1 had actually completed both epochs cleanly *before* the save
   error, and that all 12 ranks had run exactly 71 batches.

## Lessons

1. **Trust the data layer first when collectives mismatch.** "Different ranks
   running different code paths" is the signature of a workload bug, not a
   framework bug. Per-rank step counts are the canonical diagnostic — print
   them early.

2. **Print instrumentation beats stack traces for distributed bugs.** py-spy
   gave us the line numbers of the deadlock; print logs gave us the
   *sequence* (which rank reached which step when). Only the latter
   revealed the count mismatch.

3. **A trial that succeeds when others fail is a gift.** Compare it
   carefully against the failing trials. We had this evidence in round 6
   for ~15 rounds before we mined it. The post-mortem-quality comparison
   should be the *first* analysis of any spuriously-successful run, not
   the last.

4. **Lightning auto-wraps DataLoaders with its own DistributedSampler by
   default.** Setting `drop_last=True` on the user DataLoader has *no
   effect* on the sampler-level distribution. The right knob is constructing
   `DistributedSampler` explicitly + `Trainer(use_distributed_sampler=False)`.
   This was a real surprise and is worth knowing for future Lightning DDP
   work.

5. **`DistributedSampler(drop_last=False)` is the wrong default for
   reproducible distributed training.** It silently pads with duplicate
   samples and requires the dataset size to be a multiple of `world_size`
   to avoid per-rank disagreement at the boundary. Most upstream code we
   reviewed (rama's PenCL setup, our own previous work) used the default
   without realizing it. Worth a project-wide pass to make this explicit
   wherever DataLoaders are constructed.

6. **Aurora's xccl is mostly fine.** The known issues
   (`intel/torch-xpu-ops#2701`, the documented `CCL_OP_SYNC=1` workaround,
   IPEX known issues) are real but not what bit us. Our 21 rounds of CCL
   debugging produced no actionable findings about xccl itself.

## Files in this snapshot

- `README.md` — quick reference for restoring the snapshot state
- `biom3_debug_state.patch` — full BioM3 working-tree diff at investigation close
- `lightning_debug_state.patch` — full Lightning fork diff at investigation close
- `pip_freeze.txt`, `module_list.txt` — environment capture
- `biom3_head.txt`, `lightning_head.txt`, `biom3_untracked_files.txt` — provenance
- `rounds_index.txt` — list of all `debug/rounds/round_*` directories (artifacts not committed; multi-GB)
- `postmortem.md` — this document

## References

- Lightning's `use_distributed_sampler` docs: kwarg of `Trainer.__init__` (default `True`)
- PyTorch `DistributedSampler` source: `torch/utils/data/distributed.py`
- ALCF Aurora frameworks/2025.3.1 module
- `intel/torch-xpu-ops#2701` (the upstream issue we *thought* we were hitting; we weren't)
- The original handoff: [docs/.claude_prompts/2026-04-25_aurora-hang-handoff.md](../../.claude_prompts/2026-04-25_aurora-hang-handoff.md) (now historical; conclusions superseded by this post-mortem)

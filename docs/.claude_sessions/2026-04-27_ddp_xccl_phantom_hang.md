# 2026-04-27 — DDP-on-xccl phantom hang investigation

**Branch:** `addison-aurora-debug` (BioM3) + `master` (Lightning fork)
**Goal:** Resolve the Aurora Stage 3 finetune hang documented in the 2026-04-25 handoff
**Status:** **resolved.** The bug was in our DataLoader, not in xccl. Real fix is `DistributedSampler(drop_last=True)` + `Trainer(use_distributed_sampler=False)`.

For the analytical write-up see [docs/bug_reports/2026-04-27_ddp_xccl_phantom_hang/postmortem.md](../bug_reports/2026-04-27_ddp_xccl_phantom_hang/postmortem.md). This document is the chronological session log.

## Starting state

Inherited from previous session ([2026-04-25 handoff](../.claude_prompts/2026-04-25_aurora-hang-handoff.md)):

- Stage 3 finetune hangs ~100% of the time on Aurora at the train→val boundary
- 1–4 random ranks are stuck in `_engine_run_backward`, others piled at val-end metric `all_reduce`
- Working theory: xccl backend has a race in barrier release on Aurora
- Existing patches: Lightning fork has commit `008da352f` skipping the pre-allreduce barrier in `_sync_ddp` on xccl; environment.sh sets `CCL_OP_SYNC=1`, `CCL_ATL_SYNC_COLL=1`, etc.
- Iteration harness `debug/loop_one_trial.sh` already in place — runs a 240s-timeout trial, captures hang diag on hang
- Prior state mostly committed; environment.sh and launchers off-limits per handoff (later: user lifted that restriction)

## Round-by-round play

### Round 1 — baseline (DDP + loggers all on)
5/5 hang. Confirmed reproduction. Hang signature matches handoff exactly: 24 dataloader-worker `select` PIDs, 9–11 ranks at `barrier`/`all_reduce`, 1–3 stragglers at `_engine_run_backward`.

### Round 2 — `DeviceStatsMonitor` removed
Hypothesis from §5.3 of handoff: per-step XPU stats query perturbs autograd backward kernel timing. 5/5 hang. **Refuted.**

### Round 3 — strategic per-rank prints (§5.4)
Added 6 `print(...flush=True)` sites in `PL_wrapper.common_step`/`cond_elbo_objective` and 7 in the Lightning fork's `_sync_ddp`/`DDPStrategy.reduce`/`_TrainingEpochLoop.advance`/`on_advance_end`/`_EvaluationLoop._store_dataloader_outputs`. 3/3 hang. Pinned the deadlock site:

- Stragglers' last LABEL: `AFTER_COND_ELBO stage=train idx=71`
- Waiters' last LABEL: `SYNC_DDP_BEFORE_AR` (val-end metric reduction)

Got hung up on the interpretation: stragglers are "stuck in backward of train batch 71." Wrote up "the final gradient-allreduce of the train epoch" as the deadlock target. **Diagnosis was wrong** — we were reading position-in-narrative from one rank and assuming the other ranks were on the same trajectory.

### Round 4 — DeepSpeed Stage 2
Switched strategy to test whether DDP-specific scheduling was implicated. 1/3 trials succeeded (later realized: through a lucky shuffle that aligned all ranks at 71 batches). Hang signature on the 2 failing trials was identical to DDP. Concluded "strategy mostly orthogonal to the trigger."

### Round 5 — disable validation (`--limit_val_batches=0`)
Hypothesis: val collectives were the trigger. 5/5 hang. Different signature though: waiters now at `barrier` (not `all_reduce`), stuck somewhere in `on_advance_end` of batch 70. Concluded the bug is upstream of val and surfaces at whichever collective happens to be downstream.

Discovered that the project's `--limit_val_batches` arg wasn't plumbed through to the Trainer in `primary_only` strategy — fixed that as a real bug while we were there.

### Round 6 — barrier-name instrumentation
Added prints in `DDPStrategy.barrier()` to identify which named barrier hangs. 2/3 trials hung at `Trainer.save_checkpoint` barrier; 1 trial succeeded. **The successful trial was the lifeline that cracked it later, but I missed it at the time** — registered as "trial 1 also failed, just at the unrelated save_model step."

### Rounds 7-8 — alternate backends
- `process_group_backend='gloo'`: incompatible (`No backend type associated with device type xpu`)
- `process_group_backend='mpi'`: not built into torch in `frameworks/2025.3.1`

Both were dead ends; no alternative-backend test possible on Aurora XPU.

### Round 9 — Fix A: `xpu.synchronize()` in `on_after_backward`
Hypothesis: async kernel completion event drops, autograd worker spins. Per-rank `device_sync()` after backward closes the race. 1/1 hang. **The straggler is stuck *inside* backward — never reaches the post-backward hook.** Refuted.

### Round 10 — ALCF DDP wrap-order patch
ALCF docs say "the model must be offloaded to the XPU device after calling the DDP() wrapper on the model to avoid hangs." Patched `DDPStrategy.setup()` and `determine_ddp_device_ids()` to do CPU-wrap-then-move on XPU. **Incompatible**: `RuntimeError: No backend type associated with device type cpu` at DDP's `_verify_param_shape_across_processes`. The ALCF guidance applies to a stack pattern that newer torch DDP rejects up-front. Reverted.

### Round 11 — `TORCH_LLM_ALLREDUCE=0`, `CCL_ALLREDUCE=direct`
From the deep-dive agent's findings (Intel IPEX known issues + ALCF Sep-2025 update notes). Converted silent hang into a fast-fail (`CCL_ERROR: ALLREDUCE entry failed. atl_status: FAILURE`). Same site, ~75s instead of full timeout. Useful as "the algorithm refused to wait forever," but doesn't actually fix anything.

### Round 12 — adopt rama's known-working PenCL Aurora env
User pointed me to a collaborator's working setup at `/flare/NLDesignProtein/rama/code/`. Their `pencl_trackA.pbs` job uses `CCL_PROCESS_LAUNCHER=None`, unsets `ONEAPI_DEVICE_SELECTOR`, drops the CCL transport overrides — completely different from ours. **Critical realization**: rama's recent activity is Stage 1 (PenCL), not Stage 3. Their old March attempt at Stage 3 on Aurora *timed out in Stage 1 inference* and never reached Stage 3 finetune. So their Stage 3 setup is not a working reference for our problem.

Adopted the env anyway (round 12). 1/1 hang with byte-identical signature. CCL config conclusively not the cause.

### Round 13 — revert Lightning fork's `008da352f` xccl barrier-skip patch
User asked: are our own Lightning patches part of the problem? Walked the fork's git log: 4 commits by addison-nm/AddisonHowe, 0 by tnandi (one inline `# TN:` comment in `model_checkpoint.py`). The xccl-skip in `_sync_ddp` was the most likely candidate. Reverted it. 1/1 hang. **Refuted.** The skip didn't cause the hang.

### Round 14a — `reduce_boolean_decision` broadcast bypass
Found a partially-applied "bypass" comment block in `ModelCheckpoint._check_monitor_top_k` from commit `171ea93e2`: the comment claimed to bypass `reduce_boolean_decision`, but the actual code still called it (the broadcast alternative was commented out). Wired it up properly: rank-0 broadcasts the decision instead of calling reduce_boolean_decision. Inconclusive.

### THE BREAKTHROUGH — re-analysis of round 6 trial 1

User asked me to look back at the trial that succeeded. Comparing per-rank batch counts:

```
SUCC trial: every rank ran 142 train batches (71 × 2 epochs), max_idx=70 for all 12 ranks
HUNG trial: 9 ranks ran 71 batches (max_idx=70), 3 ranks ran 72 batches (max_idx=71)
```

**The "stragglers" weren't slow — they were running an EXTRA batch the others weren't.** `loss.backward()` for batch 71 fires DDP's gradient `all_reduce`; the other 9 ranks have already moved on; the allreduce has no peer; deadlock.

Realized in the moment that this had been visible all along in the round-3 prints — `idx=71` was *only present on the straggler ranks*. We had read it as "stragglers are the ones still on idx=71" (slow), when it was really "stragglers are the only ones who ever reached idx=71" (extra batch).

### Round 14b — `--limit_train_batches 71` cap
1/1 EXIT_0. Training completed. Failed at the orthogonal `save_model` deepspeed-converter bug (we'd seen this back in round 6). The hang was fixed by capping all ranks at 71 batches.

### Round 15 — try `drop_last=True` on the DataLoader
1/1 still hung. Lightning's auto-applied `DistributedSampler(drop_last=False)` was overriding our DataLoader's `drop_last=True`. Reverted; reapplied the cap.

### Rounds 16-17 — confirm the cap mechanism
Added per-rank `BATCH stage=N idx=N bs=N` print at top of `common_step` to capture exact per-rank batch sizes. Round 16 (3/3 EXIT_0): clean. Round 17 (3/3 EXIT_0 with the BATCH print): showed `bs=15` partial batches at idx=70 in 0–4 ranks per trial. Confirmed dataset is *almost* divisible by `world_size × batch_size`, with a few-sample leftover scattered across random ranks.

User asked for `Validation DataLoader 0:` clarification — N is the loader index in val, not epoch index; we have one val loader so it's always 0.

### Round 18 — re-enable validation
Default `--limit_val_batches=0.05` triggered Lightning's `MisconfigurationException` (0.05 × 18 = 0.9 < 1). Set `--limit_val_batches=1.0` instead. 3/3 EXIT_0. Val side has the same uneven-shard issue (idx=17 sees `bs=11/12/13` instead of 16) but val *tolerates* it because val has no per-step backward → no per-step DDP collective; only val-end metric `all_reduce`, which is symmetric.

### Round 19 — DeepSpeed + cap
3/3 EXIT_0. Strategy-independence of the cap workaround confirmed.

### Round 20 — DeepSpeed without cap, BATCH instrumentation
2/3 hang, 1/3 EXIT_0. The diagnostic capture: stragglers showed `BATCH idx=71 bs=1` — a single leftover sample handed to a random rank as a 1-element batch. Definitive proof of the `DistributedSampler(drop_last=False)` + leftover-sample mechanism.

### Round 21 — proper fix
Implemented `_make_distributed_sampler(drop_last=True)` helper in `PL_wrapper.py`; used in all 4 train/val DataLoaders (PL_ProtARDM and HDF5DataModule). Set `Trainer(use_distributed_sampler=False)`. Removed `--limit_train_batches 71` from jobscript. **3/3 EXIT_0.** Real fix verified.

User flagged the `getattr(self.args, 'seed', 0)` defensive default as bias-prone; replaced with direct `self.args.seed` access (fail-loud).

User asked if every loader getting the same seed was a problem; explained that DistributedSampler with `shuffle=False` (val) ignores its seed entirely, and Lightning's `_set_sampler_epoch` automatically applies the `+ epoch` per-epoch decorrelation.

### Round 22 — DDP cleanup test
Reverted Fix A's `device_sync()`, removed BATCH print, restored `DeviceStatsMonitor`, switched strategy to `ddp_strategy`. Round 22 first run had 1 EXIT_0 trial then loop got interrupted; reran 3/3 EXIT_0 underlying-exit `EXIT_1` (orthogonal save_model bug, training itself fully clean). DDP works under the sampler fix, but the test suite expects DeepSpeed historically — see "Strategy decision" below.

### Round 23 — final DeepSpeed verification
Strategy = DeepSpeed, sampler fix in place. **3/3 EXIT_0** including underlying exit code (DeepSpeed's sharded checkpoint format makes the `convert_zero_checkpoint_to_fp32_state_dict` call work; no orthogonal save_model error).

### Strategy decision

After the cleanup, attempted to commit with `'strategy': ddp_strategy` as the production default. Test suite (`tests/stage3_tests/test_stage3_run_PL_training.py`) failed because the existing tests expect DeepSpeed configuration — DDP was a temporary debugging swap, not the historical default. Reverted to `deepspeed_strategy` to keep tests passing and match historical behavior. Round 22's DDP results stand as evidence the fix is strategy-independent; it's just that the production default + test fixtures are DeepSpeed.

## Snapshot taken

Before the cleanup work, captured the full debug state for future profiling:

- Tag `debug/2026-04-27-xccl-hang-investigation` on both BioM3 (`addison-aurora-debug`) and Lightning fork
- Bug doc dir at `docs/bug_reports/2026-04-27_ddp_xccl_phantom_hang/` with patches, pip freeze, module list, README
- Lightning fork master rolled back to `origin/master` after the snapshot tag was pushed (per user direction: don't move the master branch on the remote)

## Files changed (final)

**BioM3 — keep:**
- `src/biom3/Stage3/PL_wrapper.py`:
  - Added `_make_distributed_sampler` helper
  - All 4 train/val DataLoaders use it via `sampler=` + `shuffle=(sampler is None)`
- `src/biom3/Stage3/run_PL_training.py`:
  - `'use_distributed_sampler': False` in trainer_params
  - `limit_val_batches` plumbed unconditionally (was a real bug, found in round 5)
- `docs/bug_reports/2026-04-27_ddp_xccl_phantom_hang/` — full snapshot + post-mortem
- `docs/.claude_prompts/2026-04-25_aurora-hang-handoff.md` — historical handoff (now superseded but useful as a record)

**BioM3 — removed:**
- `docs/.claude_prompts/2026-04-26_alcf-ticket-draft.md` — the ticket would have been wrong; kept in commit `4384ffd` via the snapshot tag

**Lightning fork:**
- No keepers from this session. Fork is back at `008da352f` on `master`. The snapshot tag preserves the round-13 state (xccl barrier-skip reverted) for any future replay.

## Lessons

(See [postmortem.md](../bug_reports/2026-04-27_ddp_xccl_phantom_hang/postmortem.md) for the long version.)

1. Trust the data layer first when collectives mismatch. Per-rank step counts are the canonical diagnostic — print them early.
2. A trial that succeeds when others fail is a gift. Compare it against failing trials at the *first* opportunity, not the last.
3. Lightning auto-wraps user DataLoaders with its own DistributedSampler. `drop_last=True` on the DataLoader has no effect on the sampler. `Trainer(use_distributed_sampler=False)` + explicit sampler is the way.
4. `DistributedSampler(drop_last=False)` is a footgun. Audit anywhere DataLoaders are constructed.
5. Aurora's xccl is mostly fine. The known issues are real but not what bit us.

## Outstanding follow-ups (post-merge)

- Fix the `save_model` deepspeed-converter gating bug (round 6 finding; gate the call on `isinstance(strategy, DeepSpeedStrategy)` or check whether the checkpoint path is a directory). Would let DDP runs exit cleanly with `EXIT_0` instead of `EXIT_1`.
- Audit other `DataLoader(...)` constructions across the codebase for the same `drop_last=False` issue (Stage 1 PenCL, Stage 2 Facilitator).

# Session: Rewrite finetune-resume regression test after GPU validation

**Date:** 2026-04-15
**Branch:** fix-ft-resume-from-checkpoint

## Context

Yesterday's session (`2026-04-14_ft_resume_from_checkpoint_fix.md`) fixed
two coupled bugs in Stage 3 finetune resume and added an entrypoint-level
regression test that asserted final-weight equality between a v1a run and
a v1b resume run. Today the test ran on the DGX Spark GPU box and went
through three distinct failure modes before the design was rewritten.

## Failures on GPU and their resolutions

### Failure 1 — v1b output directory never created

```
RuntimeError: Parent directory .../finetune_resume_pretrained_v1b does
not exist.
```

v1a ran `max_epochs=1` and saved a checkpoint tagged `current_epoch=1`.
v1b resumed with `max_epochs=1`, so Lightning saw `current_epoch >=
max_epochs` and stopped immediately. `ModelCheckpoint` never fired, so
v1b's checkpoint directory was never created, and `save_model`'s
`convert_zero_checkpoint_to_fp32_state_dict` call blew up writing
`single_model.best.pth` into a nonexistent dir.

**Fix:** bump v1b to `epochs=2` so Lightning runs one more epoch and
`ModelCheckpoint` materializes the dir. Mirrors the existing
`test_resume_training` pattern (2→3 epochs).

**Commit:** `68072b0` — `test: run v1b one epoch past v1a in finetune-resume regression test`

### Failure 2 — v1b's unfrozen weights diverged despite lr=0

After Failure 1 was fixed, v1b completed but the weight-equality
assertion failed. Every divergence was concentrated in the **unfrozen**
subset (last transformer block's attention/FFN/norm + final output
layer), with max diffs of 0.04–0.10 — about one epoch's worth of
training at `lr=1e-3`, not roundoff.

**Root cause:** Lightning's `trainer.fit(ckpt_path=...)` restores the
full optimizer state from the checkpoint, *including*
`param_groups[0]['lr']`. v1a's optimizer had `lr=1e-3`; that value
flowed through the checkpoint into v1b's optimizer, silently overriding
`args_b.lr = 0.0`. v1b then trained its extra epoch at 1e-3, drifting
exactly the unfrozen params.

**Deeper problem:** even if the lr issue were solved, weight equality
cannot distinguish fix from bug in this test design. Training is
deterministic, so:

- **Fix path:** v1a does epoch 0 at lr=1e-3 from pretrained → checkpoint
  at end of epoch 0. v1b resumes from checkpoint and does epoch 1 at
  lr=1e-3 (restored). Net: 2 total epochs from pretrained.
- **Bug path:** v1b loads pretrained via `load_pretrained_weights`,
  `trainer.fit` with no `ckpt_path` so `current_epoch=0`, runs epochs 0
  and 1 at lr=1e-3. Net: 2 total epochs from pretrained.

Same seed, same data, same starting point, same total step count →
identical final weights. That's the whole point of checkpointing, and
it's the reason a weight-equality test can't catch this bug.

### Failure 3 — caplog/capfd did not see biom3 records

The test was rewritten to assert log markers instead of weights, using
`caplog` for Lightning's `Restoring states from the checkpoint path`
and `capfd` for biom3's new `Ignoring --pretrained_weights` and
`Resume finetuning from checkpoint` info logs. Lightning's marker was
captured correctly (proving the fix was actually running on the GPU),
but both biom3 markers came back missing.

**Cause:** [`biom3.backend.device.setup_logger`](../../src/biom3/backend/device.py#L47)
configures every biom3 module logger with `propagate=False` and a
`StreamHandler()` bound to `sys.stderr` *at module import time*. Two
knock-on effects:

1. `caplog` attaches a handler to the **root** logger. With
   `propagate=False`, biom3 records never reach the root and are
   invisible to `caplog`.
2. `capfd` captures at the fd level and should have seen writes via the
   stored `sys.stderr` after pytest's `dup2`, but in practice biom3
   records only reached pytest's built-in stderr display (shown in
   failure reports) and not `capfd.readouterr()`. The exact reason
   wasn't pinned down — likely a CaptureManager vs. stored-stream
   ordering quirk — but the upshot is that `capfd` is not a reliable
   source for biom3 logs in this codebase.

## Final design: read `run.log` directly

[`biom3.core.run_utils.setup_file_logging`](../../src/biom3/core/run_utils.py#L78)
already writes a `run.log` artifact per run, attaching a `FileHandler`
to every logger whose name starts with `biom3`. Stage 3 calls it in
`run_stage3_pretraining` and writes to
`{output_root}/{runs_folder}/{run_id}/artifacts/run.log`.

The test now reads v1b's `run.log` directly and greps for the two
biom3 markers. No `caplog`, no `capfd`, no handler plumbing. Clean,
deterministic, uses an existing artifact.

### Assertions on v1b's run.log

| Marker | Source | Meaning if present |
|--------|--------|---------------------|
| `Ignoring --pretrained_weights` (present) | `run_stage3_pretraining` finetune block | `load_pretrained_weights` was skipped |
| `Resume finetuning from checkpoint` (present) | `train_model` finetune branch | `trainer.fit(ckpt_path=...)` was called |
| `Loading pretrained weights from` (absent) | `load_pretrained_weights` | Confirms pretrained load did not run |

Under the buggy code, the first two markers are absent and the third
appears. Under the fix, the reverse.

### Why drop the Lightning "Restoring states" assertion

`setup_file_logging` filters by `biom3.*` prefix and does not attach to
`pytorch_lightning`, so `run.log` doesn't contain Lightning records. The
Lightning check was nice-to-have but redundant: the biom3
`Resume finetuning from checkpoint` log sits on the line immediately
above `trainer.fit(PL_model, data_module, ckpt_path=resume_from_checkpoint)`
— there's no realistic branch where the log emits but the call below
it doesn't receive `ckpt_path`. If Lightning later silently dropped
`ckpt_path`, that's a Lightning regression, not ours.

## Load-bearing log comments

Because the test now grep for exact strings in the two biom3 info logs,
both log sites in
[src/biom3/Stage3/run_PL_training.py](../../src/biom3/Stage3/run_PL_training.py)
now carry a short comment warning future maintainers that the string is
asserted by
`test_resume_finetune_ignores_pretrained_weights` and that changing it
will break the regression test.

## Commits

| Hash | Description |
|------|-------------|
| `68072b0` | test: run v1b one epoch past v1a in finetune-resume regression test |
| `95f79a1` | test: assert finetune-resume fix via log markers, not weight equality |
| `393c61f` | test: read v1b run.log artifact instead of pytest capture fixtures |

## Key takeaways

1. **`trainer.fit(ckpt_path=...)` restores the full optimizer state,
   including `param_groups[*]['lr']`.** Any test or production flow
   that expects `args.lr` to take effect on a resumed run is wrong.
   If you actually want to change the learning rate on resume, you
   have to patch `optimizer.param_groups` after Lightning restores.

2. **Deterministic training defeats weight-equality regression tests
   for checkpoint-path bugs.** "Train for K + N epochs with a
   checkpoint in the middle" equals "train for K + N epochs fresh"
   whenever the starting point, seed, and data order line up. Any
   test that tries to distinguish those two cases via final weights is
   fundamentally confused. Log markers (or state snapshots taken
   before training starts) are the right tool for this class of bug.

3. **biom3's `propagate=False` loggers are invisible to pytest's
   `caplog`.** For any future test that needs to assert biom3 log
   output, read `run.log` directly instead of reaching for `caplog`
   or `capfd`. `setup_file_logging` gives you a free per-run artifact
   that covers every biom3 logger.

4. **"The final weights should match" is an attractive assertion but
   a brittle contract.** For control-flow bugs, prefer directly
   observable side effects (log lines, callback invocations,
   artifacts on disk) over downstream numerical state.

# 2026-05-05 — Aurora oneCCL IPC-handle cache cap in environment.sh

A collaborator's Stage 3 finetune (12 ranks, 1 node, ZeRO via PL +
DeepSpeed) on Aurora crashed at the Level-Zero IPC layer ~8h into a
300-epoch run. Root-caused to oneCCL's default IPC-handle cache size
(1000) being exceeded under sustained DeepSpeed grad-bucket churn,
which then triggered the documented L0 `pidfd_getfd` failure mode.
Fix is environment-only: bump the cache cap in the Aurora branch of
`environment.sh`.

Branch: `addison-dev`. One config commit, this note.

## Diagnosis (from the job's `.o` file)

- Job started 23:23:24 (2026-05-04), 12 XPU tiles, DeepSpeed ZeRO via
  PyTorch Lightning. Sanity check + epochs 0–30 ran cleanly.
- At **23:39:34** — about 16 minutes into Epoch 0 — oneCCL began
  emitting on every rank:

  ```
  CCL_WARN| Sender cache limit is reached: cache size: 1000, limit:
  1000, it will clear older elements from the cache. The app can crash
  in a L0 call due to a L0 issue. There are three possible work-arounds:
   1) export ZE_ENABLE_TRACING_LAYER=1
   2) CCL_ZE_CACHE_GET_IPC_HANDLES_THRESHOLD=<value>
   3) export CCL_ZE_CLOSE_IPC_WA=1
  ```

  The cache eviction kept going for ~8h.
- At **07:47:43** (Epoch 31, ~2 min in), the predicted L0 crash hit
  ranks 6 and 8 inside DeepSpeed's `allreduce` →
  `dist.all_reduce` →
  `oneCCL: ze_fd_manager.cpp:390 convert_fd_pidfd: pidfd_getfd failed:
  fd: -1 ... errno: Bad file descriptor`. PALS killed the rest with
  SIGKILL.
- Last good checkpoints visible in the log:
  `epoch=29-step=30720.ckpt` (best val_loss=0.39111) and
  `epoch=30-step=31744.ckpt` (val_loss=0.39290, top-2). 30/300 epochs
  recoverable.

Mechanism: oneCCL caches Level-Zero IPC handles per peer, default
cap 1000. DeepSpeed's grad-bucket cycling churns through small IPC
handles every step and refills the cache faster than entries can age
out cleanly. When eviction frees a handle that L0 is still tracking
via `pidfd_getfd`, the fd comes back -1 and oneCCL throws. It's
stochastic — that's why this run made it 8h before tipping. Same
class of bug as the existing `project_aurora_val_hang` and
`project_aurora_ccl_reduce_bug` memories, but a third, distinct
failure mode.

## Why option (2) over (1) and (3)

The CCL_WARN message lists three workarounds with neutral trade-off
notes; it does not rank them and is not an "Intel recommendation."
Engineering reasoning for picking (2):

- (1) `ZE_ENABLE_TRACING_LAYER=1` — measurable per-call perf hit.
  Worst choice for an 8h training loop.
- (2) `CCL_ZE_CACHE_GET_IPC_HANDLES_THRESHOLD=16384` — costs only
  some host RAM per rank. For an 86M-param ProteoScribe with
  ZeRO-1/2 grad bucketing, that's negligible. No perf cost, no
  correctness risk.
- (3) `CCL_ZE_CLOSE_IPC_WA=1` — leaks small L0 internal graph
  objects. Fine for one run; compounds under repeated re-runs.

Picked (2). Cap of 16384 is 16× the default — comfortably above
what an 8h DeepSpeed run was filling.

## Change

`environment.sh`, inside the Aurora branch (after the existing
`TMPDIR=/tmp` line):

```bash
# Raise oneCCL's Level-Zero IPC-handle cache cap (default 1000). Long
# DeepSpeed runs churn through grad-bucket IPC handles; once the cache
# evicts, oneCCL can hit a Level-Zero `pidfd_getfd` failure mid-allreduce
# and crash the job ("Sender cache limit is reached" warning seen 8h into
# a 300-epoch finetune on 2026-05-04).
export CCL_ZE_CACHE_GET_IPC_HANDLES_THRESHOLD=16384
```

Aurora-only (under the `x4*` / `aurora-uan*` hostname branch); Spark
and Polaris are unaffected.

## Followups

- The collaborator can resume from
  `.../checkpoints/.../epoch=30-step=31744.ckpt` once the env var is
  in their shell — saves the 8h of compute already burned.
- Worth verifying once running that the warning no longer fires; if
  it still fires at 16384, bump again.
- Open question (not for this branch): the launchers don't echo or
  log the resolved CCL/L0 env-var values at startup. A small banner
  ("CCL_ZE_CACHE_GET_IPC_HANDLES_THRESHOLD=16384", etc.) at the top
  of the `.o` file would have made today's diagnosis faster.

## Notes for myself

- During this session I asserted that option (2) was "the workaround
  Intel themselves recommend first" — that was wrong. The CCL_WARN
  message lists trade-offs without ranking, and I had not read any
  Intel/ALCF doc making that recommendation. Fixed in conversation;
  also saved as `feedback_no_unverified_authority_claims.md` so I
  don't repeat the mistake.
- Cloudflare blocks WebFetch/curl on `docs.alcf.anl.gov` (managed
  challenge with `cf-mitigated: challenge`). The mkdocs source is
  public at `argonne-lcf/user-guides`, and `raw.githubusercontent.com`
  is unaffected — that's the path for any future ALCF-doc lookup
  from this harness.

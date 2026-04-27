# DDP-on-XCCL "phantom" hang — debug-state snapshot (2026-04-27)

This directory captures the in-tree state of BioM3-dev and the addison-nm/lightning
fork at the close of a 21-round investigation into what looked like an Aurora xccl
deadlock. The actual bug was an uneven train-shard size in our own data pipeline.
xccl was correctly hanging on a genuinely mismatched DDP gradient-allreduce. See
the post-mortem below.

The full post-mortem write-up will live alongside this snapshot at
`2026-04-27_ddp_xccl_phantom_hang_postmortem.md` (TODO).

## Reproducing the snapshot state

Both repos are tagged at `debug/2026-04-27-xccl-hang-investigation`. To check out:

```bash
git -C /lus/flare/projects/NLDesignProtein/ahowe/BioM3-dev-space/BioM3-dev \
    checkout debug/2026-04-27-xccl-hang-investigation
git -C /flare/NLDesignProtein/ahowe/BioM3-dev-space/lightning \
    checkout debug/2026-04-27-xccl-hang-investigation
```

Or apply the captured patches to a fresh `addison-dev` / fork-master checkout:

```bash
git apply biom3_debug_state.patch                # in BioM3-dev
git -C <lightning-fork> apply lightning_debug_state.patch
```

## Files in this directory

| File | What it contains |
|---|---|
| `biom3_debug_state.patch` | `git diff origin/addison-dev` — every BioM3 modification at snapshot time, including the `_make_distributed_sampler` fix, `use_distributed_sampler=False`, and instrumentation prints |
| `lightning_debug_state.patch` | `git diff origin/master` — the only Lightning fork modification at snapshot time (Finding A: pre-allreduce barrier on xccl restored, i.e. `008da352f` reverted in working tree) |
| `biom3_head.txt` | BioM3 HEAD SHA / author / commit message at snapshot |
| `lightning_head.txt` | Lightning fork HEAD SHA at snapshot |
| `biom3_untracked_files.txt` | Untracked files in BioM3 at snapshot (debug docs, stray files) |
| `pip_freeze.txt` | Full `pip freeze` of the `venvs/biom3-env` venv |
| `module_list.txt` | `module list` output (frameworks/2025.3.1 active) |
| `rounds_index.txt` | List of `debug/rounds/*` directories — every trial round's logs + hang_diags |

## Trial round artifacts

The `debug/rounds/round_*` directories on disk contain the per-trial logs and
`hang_diag_*` snapshots used for diagnosis. They are not committed (multi-GB) and
are listed in `rounds_index.txt` for reference. Key rounds:

- `round_06_barrier_names` — first trial that succeeded; revealed all 12 ranks ran
  exactly 71 batches per epoch in the success case
- `round_17_batch_size_trace` — first round with `BATCH bs=…` print, showed `bs=15`
  partials at idx=70 across runs
- `round_20_deepspeed_no_cap` — captured `bs=1` extra batch on a single rank, the
  exact mechanism of the per-rank divergence
- `round_21_drop_last_sampler` — first round with the proper `DistributedSampler(
  drop_last=True)` fix in place; clean

## TL;DR of the bug

The HDF5 dataset has 17,165 samples → after `train_test_split(test_size=0.2)` and
`min_seq_length` filtering, the effective train set is ~13,633 samples. With
`world_size=12` and `batch_size=16`, that's 71 full batches per rank plus 1
leftover sample. The default `DistributedSampler(drop_last=False)` distributes
that leftover to a single random rank as a `bs=1` 72nd batch, while the other 11
ranks stop at 71. The 72nd batch's DDP gradient allreduce has no peer collectives
to combine with → deadlock. xccl is correctly hanging.

The fix is `DistributedSampler(drop_last=True)` constructed explicitly in our
DataLoader plus `Trainer(use_distributed_sampler=False)`, both shipped in this
snapshot's `biom3_debug_state.patch`.

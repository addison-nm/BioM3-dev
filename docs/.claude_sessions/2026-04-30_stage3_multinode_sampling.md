# 2026-04-30 — Multi-node multi-GPU Stage 3 sampling + smoke suite

Continuation of the multi-tile / multi-node thread that started with
[2026-04-30_grpo_fix_and_multitile_rollout.md](2026-04-30_grpo_fix_and_multitile_rollout.md).
That session deferred MPI-backed Phase B.5 for the RL rollout pool. This
session takes the *generic* Stage 3 sampler down a different path: pure
`torch.distributed`, one process per device, world = N×M, designed
specifically for `biom3_ProteoScribe_sample` rather than retrofitted
through the threaded `RolloutPool`.

Branch: `addison-hack-grpo`. Two commits: `291cef7`
(feat: multi-node Stage 3 sampling + smoke suite) and `f5edb82`
(chore: explicit `--gpu-bind` per-tile in `aurora_singlenode.sh`).

## Headline changes

### 1. Distributed init helper — new `src/biom3/core/distributed.py`

`init_distributed_if_launched(device) -> (rank, local_rank, world_size, resolved_device)`.
No-op (returns `0,0,1,device`) when no launcher env vars are present, so
the single-process entrypoint test path is preserved exactly.
Backend selection off `BACKEND_NAME`: `xccl` on XPU, `nccl` on CUDA,
`gloo` on CPU. Order matters: resolve `local_rank` → `set_device(local_rank)`
→ `init_process_group(init_method="env://")`. Reversing the last two
puts every rank on device 0 — encoded in the docstring.

Aurora `ZE_AFFINITY_MASK` interaction: when set (or
`torch.xpu.device_count() == 1`), force `device_str = "xpu:0"` regardless
of `local_rank`. The launcher's per-process `--gpu-bind` already pins
one tile per process, so `xpu:0` is the only valid device id inside
each rank. Cross-references `scripts/run_gdpo_smoke_multixpu.sh:45-47`
which deliberately *unsets* it — that is the opposite (RolloutPool)
pattern, where one process needs all-tile visibility.

`MASTER_ADDR` is pulled from the first line of `$PBS_NODEFILE` and
`MASTER_PORT = 29500 + (blake2b($PBS_JOBID) mod 1000)` so two
concurrent jobs on the same login node don't collide on port 29500.
`atexit.register` cleanup avoids forcing every entry point to wrap a
`try/finally`.

Sibling helpers: `is_main_process`, `barrier`, `gather_object_to_main`,
`broadcast_int`, plus `get_global_rank` / `get_local_rank` /
`get_world_size` (the last scans `WORLD_SIZE` / `PMI_SIZE` /
`OMPI_COMM_WORLD_SIZE` and falls back to
`PALS_LOCAL_SIZE * len($PBS_NODEFILE)` because PALS does *not* set
`WORLD_SIZE`).

### 2. Stage 3 sampling rank-awareness

`run_ProteoScribe_sample.py:main` calls `init_distributed_if_launched`
near the top; if `world_size > 1`, overrides `args.device` with the
resolved per-rank device. Stashes `_rank`, `_world_size`, `_base_seed`
on `config_args` so `batch_stage3_generate_sequences` reads them
without parameter-list churn.

`batch_stage3_generate_sequences` now slices the flat
`(prompt × replica)` work-list by `i % world_size == rank` *after* the
global list is built, preserving the global index for RNG keying.
Returns a rank-local sparse dict instead of the public prompt-keyed
output. The public dict is reassembled in `main()` via a new
`_merge_shards` helper after `gather_object_to_main`.

I/O is rank-0 only: `setup_file_logging`, `torch.save`, FASTA, manifest.
Animations and per-step probability files stay rank-local — filenames
already encode global `(p_idx, r_idx)` so collisions are impossible.
Final `barrier()` keeps non-rank-0 alive until rank 0 finishes I/O.

### 3. World-size-invariant RNG

New `_derive_sample_seed(base_seed, prompt_idx, replica_idx)` uses
`numpy.random.SeedSequence(entropy=[...])` so the same triple yields
the same 32-bit seed across processes / Python interpreters / world
sizes. Avoids Python's salted `hash()`.

`_make_sample_seeds(batch, base_seed)` produces one int per row;
`_build_initial_mask_state` accepts `sample_seeds=` and uses per-row
CPU generators for `torch.randperm`.

`sampling_analysis.py`: both `batch_generate_denoised_sampled` and
`batch_generate_denoised_sampled_confidence` accept `sample_seeds`. New
`_fill_gumbel_buffer(buffer, sample_seeds, step)` builds a per-row
device-side `torch.Generator` (CPU fallback for older XPU builds)
keyed on `(seed, step)` and replaces the global-RNG
`gumbel_buffer.exponential_()` calls.

Verified by the existing `test_reproducibility` (same seed → same
output, different seed → different output) plus the new
`test_rng_world_size_invariance` smoke test (rank 0 builds an
in-process single-rank reference, distributed run gathers to rank 0,
exact byte-equality assertion).

When `seed <= 0` the time-based fallback (`np.random.randint(2**32)`)
runs on rank 0 only and is broadcast via `dist.broadcast_object_list`,
so all ranks share the same seed even in fallback mode.

### 4. Diffusion-loop perf tweaks for distributed

The per-step `_device_synchronize(args.device)` was added purely so
tqdm reflects actual GPU progress (not async kernel-launch rate).
Under multi-rank that serialises every kernel launch and was
(empirically) causing ranks to stall. Now gated on
`sync_per_step = world_size <= 1` — single-rank runs keep accurate
tqdm; distributed runs let kernels queue freely. The terminal `.cpu()`
in `_drain_sampling_state` still forces a single sync at end so
correctness is preserved.

`tqdm` `desc` is conditional too: `_diffusion_tqdm_desc(args)` returns
`"diffuse rank=R/W dev=DEVICE host=HOST"` only when `world_size > 1`,
else `None` (bare tqdm — unchanged for single-rank users).

### 5. Pytest smoke suite gated on `--multinode N --multidevice M`

[tests/conftest.py](../../tests/conftest.py): new flags + `multinode`
marker. `pytest_collection_modifyitems` skip-gate uses
`biom3.core.distributed.get_world_size()` (PALS-aware) rather than
`os.environ["WORLD_SIZE"]` — that was the bug that made the first
24-rank Aurora run skip every test silently.

[tests/stage3_tests/test_multinode_sample.py](../../tests/stage3_tests/test_multinode_sample.py):
three tests, all `@pytest.mark.multinode`:

1. `test_model_identity_across_ranks` — every rank loads the mini
   model, computes a fp64 `state_dict()` fingerprint, `all_gather_object`,
   asserts `|fingerprint_r - fingerprint_0| < 1e-6`.
2. `test_per_rank_generation` — every rank runs `main(args)` with the
   mini fixtures; rank 0 asserts the gathered output dict has the
   expected `prompt_*` keys and non-empty replica strings.
3. `test_rng_world_size_invariance` — rank 0 builds an in-process
   single-rank reference (calls `batch_stage3_generate_sequences` +
   `_merge_shards` directly with `world_size=1`); all ranks then run
   `main(args)` under the actual launcher world size; rank 0 asserts
   byte-equality of every replica string between merged and reference.

Test launcher [scripts/test_stage3_multinode.sh](../../scripts/test_stage3_multinode.sh)
dispatches via `${BIOM3_MACHINE}_multinode.sh python -m pytest ... -p no:randomly`.
The `-p no:randomly` is load-bearing: `pytest-randomly` was assigning
each rank a different `--randomly-seed`, which would have reordered
tests across ranks and broken the collective ops mid-test.

PBS template at [jobs/aurora/_template_test_stage3_multinode.pbs](../../jobs/aurora/_template_test_stage3_multinode.pbs).

### 6. Embedding pipeline `DEVICE` env var

Tiny but real: [scripts/embedding_pipeline.sh](../../scripts/embedding_pipeline.sh)
was hard-coded to `cuda` (PenCL had no `--device` flag forwarded; the
default was `cuda`), which fails immediately on Aurora. Now honors
`DEVICE` env var (default `cuda`) and forwards to both
`biom3_PenCL_inference` and `biom3_Facilitator_sample`. `DEVICE=xpu
./scripts/embedding_pipeline.sh ...` works.

### 7. `aurora_singlenode.sh` — explicit `--gpu-bind` per tile (separate commit `f5edb82`)

Mirrors the multinode launcher: every rank now gets its own tile via
`--gpu-bind=list:0.0:0.1:1.0:1.1:...:5.0:5.1` rather than relying on
mpiexec defaults (which can map all ranks to tile 0 under some launcher
configs). Same CPU bind list as before, just split into a named scheme
variable.

## Validation

Single-rank degenerate (CPU, `WORLD_SIZE=1 --multinode 1 --multidevice 1`):
3/3 multinode smoke tests pass.

Full Stage 3 suite under `--quick`: 208 passed, 123 skipped.
Full repo suite under `--quick`: 630 passed, 144 skipped, no regressions.

24-rank Aurora job (`select=2`, 12 tiles per node) ran end-to-end
through the diffusion loop. Two issues caught and fixed in the same
session:

- Conftest gate skipped every test because PALS doesn't set
  `WORLD_SIZE`. Fix: route through `get_world_size`.
- pytest-randomly reordered tests differently per rank. Fix: `-p no:randomly`.

After the fixes: end-to-end SH3 generation under `run_sh3_multinode.sh`
runs through, modulo the pre-existing per-step `_device_synchronize`
hang under XPU which the gating fix in section 4 resolves.

## Things deferred / to watch

- **DeepSpeed sharded checkpoints under multi-rank** ([Stage3/io.py:62-75](../../src/biom3/Stage3/io.py#L62-L75)):
  every rank simultaneously reads the source dir and writes its own
  tempfile via `convert_zero_checkpoint_to_fp32_state_dict`. Smoke tests
  use raw `.pth`, so this isn't on the test path. For production:
  rank-0 converts and broadcasts the path. Not done.
- **XPU device-side `torch.Generator`** has been the load-bearing
  perf optimization in this work. If older PTI / SYCL builds reject
  `torch.Generator(device="xpu")`, the CPU fallback in
  `_fill_gumbel_buffer` keeps things correct but pays a per-row
  cross-device copy. Unverified on the older Aurora frameworks
  modules.
- **Animation file collisions across ranks**: filenames embed global
  `(p_idx, r_idx)`, so this is fine *today*, but a future refactor
  that re-keys by within-rank counter would silently corrupt outputs.
  Comment added to `main()` documenting this invariant.
- **GIL / Python overhead at 24+ ranks**: pure `torch.distributed`
  doesn't have the GIL ceiling that the threaded `RolloutPool` does.
  No early measurements yet on actual Aurora throughput vs. theoretical
  24× ideal speedup.

## Files touched

```
new   src/biom3/core/distributed.py
new   tests/stage3_tests/test_multinode_sample.py
new   scripts/test_stage3_multinode.sh
new   jobs/aurora/_template_test_stage3_multinode.pbs
mod   src/biom3/Stage3/run_ProteoScribe_sample.py
mod   src/biom3/Stage3/sampling_analysis.py
mod   tests/conftest.py
mod   scripts/embedding_pipeline.sh
mod   scripts/launchers/aurora_singlenode.sh
```

Untracked / not committed (per-user run drivers and PBS account
specifics, intentionally out of the feature commit):
`run_sh3_multinode.sh`, `sh3_run_config.json`, `prompts.csv`,
`myoutputs/`, `jobs/aurora/job_test_stage3_multinode_n2.pbs` (has
personal `gpu_hack` PBS account).

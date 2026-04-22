# Session: benchmark plotting iteration + package integration

**Date:** 2026-04-22
**Branch:** `addison-dev` (small-fix work, no worktree)
**Commits:** `68a0c28`, `c9ff151`, `1427b4d`
**Predecessor:** [2026-04-21_generation_benchmark.md](2026-04-21_generation_benchmark.md)

## Goal

Polish the benchmark + plotter that landed yesterday and graduate them
from `scripts/` into the `biom3` package. Specifically:

1. Eliminate the slow first datapoint that contaminated cold-start runs.
2. Clarify plot titles and axis labels so the units (per how many
   sequences, per how many positions) are unambiguous.
3. Add an extrapolation plot — given measured per-step time at small D,
   project total wall-clock for full-length D=1024 generation.
4. Move the benchmark sweep harness and plotter into `src/biom3/` so they
   become real package modules with CLI entry points, importable from
   Python, and ready to host sibling benchmarks (Stage 3 training,
   Stage 1/Stage 2 inference) without further restructuring.

## Component A: extrapolation plot

File: `src/biom3/viz/benchmark.py` (`plot_extrapolated_time`).

The math:

```
t_per_step = T_total_s / (num_batches × D_measured)
T_extrap   = ceil(N_target / B) × D_target × t_per_step
```

Validity rests on one empirical fact from the v1 vs v2 sweeps: per-step
time is independent of D (each step is a full transformer forward on
`[B, 1024]` regardless of how many positions remain masked). v1 at D=16
and v2 at D=8 both show ~1 s/step at B=64; halving D doubled throughput
exactly as predicted. So a `t_per_step` measured at D=8 lets us project
D=1024 wall-clock without paying the 128× runtime cost.

CLI surface: `--extrap_N` (default 128), `--extrap_D` (default 1024).
Output filename embeds the targets so multiple runs don't collide.

The `ceil` matters: at B=256 generating 128 sequences, you do 1 batch
costing the full 256-wide step time but only get 128 useful sequences —
the extrapolation captures this honest waste, so the plot shows a step
up at B > N_target.

## Component B: warmup pass

File: `src/biom3/benchmarks/Stage3/generation.py`.

Before the sweep loop, run one untimed `batch_stage3_generate_sequences`
at the smallest sweep config (smallest `B`, smallest `P`, first
`token_strategy`). Eats the cold-state cost — kernel JIT, autocast init,
allocator first-touch, Gumbel buffer alloc — that previously inflated
the first datapoint.

Verified empirically: the v0 sweep run after this change shows the first
B=2 measurement at 0.26 s, identical to subsequent B=2 runs. Before the
warmup, equivalent runs would show a 2× outlier on the cold call.

The warmup logs as `Warmup: ... (untimed)` and `Warmup complete in
X.Xs` so the user can see it ran.

### Why one warmup is enough

In principle each unique kernel shape (each `B`) might trigger its own
selection cost. In practice, the dominant cold-state costs are
process-global (autocast context, allocator, RNG state) and one warmup
absorbs them. Per-shape kernel selection is a few ms — within the
measurement noise we already accept.

If we ever measure a cold/warm gap per shape, the fix is to warm each
unique `B` once before the timed run; the structure already supports
that since `_build_generation_args` is per-config.

## Component C: plot title / label cleanup

File: `src/biom3/viz/benchmark.py`.

Two passes:

1. First pass added bracketed clarifications under the Y axis (e.g.
   `Time per diffusion step (ms)\n[1 position unmasked across B
   sequences]`).
2. User feedback: that context belongs in the title, not in the Y
   label. Second pass moved each bracketed line into the title's
   second line. Y axes are now plain metric labels.

Final title structure (3 lines per plot):

```
<plot subject> vs <axis>
(<per-how-many-sequences / per-how-many-positions clarification>)
<arch_id> · <device_name> · <hostname>
```

The third line keeps inter-machine comparisons unambiguous when files
are shared across Spark / Polaris / Aurora.

## Component D: package integration

The original scripts lived in `scripts/`. Two motivations to move them:

- The benchmark already imports half of biom3 (`backend.device`,
  `Stage3.io`, `Stage3.run_ProteoScribe_sample`). Keeping it as a script
  meant `PYTHONPATH=src python scripts/benchmark_generation.py …` instead
  of a real CLI.
- The user expects to extend the benchmark surface — add per-epoch /
  per-step *training* benchmarks for Stage 3 (the
  `TrainingBenchmarkCallback` already exists in `Stage3/callbacks.py`
  and just needs a sweep harness wrapper), and eventually Stage 1 and
  Stage 2 inference benchmarks. A scripts-based approach forces every
  new benchmark to duplicate the CLI scaffold.

### New layout

```
src/biom3/
  benchmarks/
    __init__.py          # subpackage docstring describes layout
    __main__.py          # entry-point shims for all benchmarks
    Stage3/
      __init__.py
      generation.py      # was scripts/benchmark_generation.py
      # planned: training.py
    # planned: Stage1/, Stage2/
  viz/
    benchmark.py         # was scripts/plot_benchmark.py
```

The Stage3 hierarchy mirrors `src/biom3/Stage3/` so the cognitive map
carries over. Stage1 / Stage2 subpackages will be added when their
benchmarks are written.

### benchmark_type tagging

Each benchmark records `benchmark_type` in its `env.json` (currently
`"stage3_generation"`). The plotter validates this on load and refuses
to plot anything else. When a Stage3 training benchmark lands later
with `benchmark_type="stage3_training"`, it'll either get its own
plotter module or be dispatched within `viz/benchmark.py` — the field
is already there to drive that.

### Entry points

Registered in `pyproject.toml`:

- `biom3_benchmark_stage3_generation` →
  `biom3.benchmarks.__main__:run_benchmark_stage3_generation`
- `biom3_plot_benchmark` →
  `biom3.benchmarks.__main__:run_plot_benchmark`

Both are installed by `pip install -e .` (verified). The plot CLI
remains type-agnostic in name even though it currently only handles
generation — future plot CLIs will probably be benchmark-type-specific
(`biom3_plot_stage3_training`, etc.) once their schemas diverge.

### Conventions followed

- `parse_arguments(argv)` returns a Namespace; `main(args)` accepts it.
  Matches `Stage1.run_PenCL_inference`, `Stage3.run_ProteoScribe_sample`.
- `__main__.py` exposes shim functions with descriptive names that the
  pyproject entry points point at. Matches `Stage3/__main__.py`.

## Side discussion: GPU compute-boundedness

Long thread of analysis questions while looking at the v0 / v1 / v2
plots. Captured here as a reference for future readers:

1. **Per-step time scales linearly with B** across the full sweep
   range (B = 2 → 256). Slope ≈ 1 on log-log. No flat regime above
   B=2.
2. **Throughput is roughly flat** across B with a slight downward drift
   for larger B. This is the *already-saturated* signature; the GPU
   matmul kernels are running at their own per-shape ceiling (10–20 %
   of theoretical peak FLOPS) regardless of B, so adding more samples
   just adds proportional time.
3. **Free HBM does not buy compute speed.** GB10 has ~128 GB unified
   memory; benchmark uses up to ~12 GB. That's slack in a different
   resource. To use that slack you'd run a bigger model, longer
   sequence, or higher B — all of which add work without reducing
   per-sample latency.
4. **Why "sequence generation" feels heavier than expected**: ProteoScribe
   is a diffusion model, not autoregressive. Each step runs a *full*
   bidirectional transformer forward on the entire 1024-length
   sequence. Total compute per generated sequence ≈ L × O(L²·d·layers),
   roughly 16 TFLOPs at full D — comparable to a 7B-parameter LLM
   generating 1024 tokens autoregressively (which has decades of
   optimization that diffusion sampling lacks).
5. **Practical recommendation**: B=4 is the throughput sweet spot on
   GB10 for this model. Per-sample cost is within noise across B=4 →
   64; bigger B costs linear memory and latency for no throughput
   gain.

These are observations that will shape future architectural work
(FlashAttention port, kernel fusion, smaller-D defaults) more than
anything we can change in the benchmark itself.

## Files touched

| File | Change |
|------|--------|
| `src/biom3/benchmarks/__init__.py` | New |
| `src/biom3/benchmarks/__main__.py` | New — entry-point shims |
| `src/biom3/benchmarks/Stage3/__init__.py` | New |
| `src/biom3/benchmarks/Stage3/generation.py` | Was `scripts/benchmark_generation.py`; added warmup pass + `benchmark_type` field. |
| `src/biom3/viz/benchmark.py` | Was `scripts/plot_benchmark.py`; added extrapolation plot; cleaned titles/labels; added `benchmark_type` validation. |
| `pyproject.toml` | Registered `biom3_benchmark_stage3_generation` and `biom3_plot_benchmark`. |
| `scripts/benchmark_generation.py` | Deleted (replaced by package module). |
| `scripts/plot_benchmark.py` | Deleted (replaced by package module). |

Plus uncommitted artefacts:
- `configs/benchmark/stage3_bm_generation_ProteoScribe_1block_v1_spark_v{0,1,2}.json`
  — Spark-specific sweep configs at three sweep granularities (v0
  small-batch, v1 full sweep, v2 large-batch).

## Verification

- `pytest tests/ --quick` — 479 passed, 135 skipped. No regressions.
- `which biom3_benchmark_stage3_generation` →
  `/home/ahowe/.conda/envs/biom3-env/bin/biom3_benchmark_stage3_generation`.
  CLI `--help` works.
- `which biom3_plot_benchmark` likewise; smoke-plotted v0 sweep
  successfully.
- v0 sweep on Spark (10 runs, ~75 s) confirmed warmup eliminated the
  first-call outlier.

## Things this session deliberately did NOT do

- Did not implement `Stage3/training.py` benchmark module. The
  `TrainingBenchmarkCallback` is ready to be wrapped, but no immediate
  driver — defer until we want a CLI sweep over training configs.
- Did not split `viz/benchmark.py` into per-benchmark-type plotters.
  Single file is fine for one schema; we'll split when a second
  schema lands.
- Did not add unit tests for the benchmark harness or the plotter.
  The harness is hard to mock end-to-end (Stage 1+2 subprocess), and
  the plotter needs synthetic `results.npz` fixtures we haven't
  built yet. Worth doing when extending.
- Did not push to origin. Branch is 6 commits ahead.

## Follow-ups

- **`biom3_benchmark_stage3_training`**: write a sweep harness that
  varies `(effective_batch_size, num_devices, gradient_accumulation,
  …)`, launches `biom3_pretrain_stage3 --save_benchmark`, and
  post-processes the per-run `benchmark_history.json` into
  `results.npz` + `env.json` with `benchmark_type="stage3_training"`.
  Then add a matching plotter module.
- **Per-shape warmup** if we ever observe per-shape cold/warm gaps.
- **Multi-machine plot overlays**: extend `viz/benchmark.py` to accept
  multiple `--run_dir` args so Spark / Polaris / Aurora numbers
  render side-by-side on a single chart. Useful as soon as we have
  the second machine's numbers in hand.
- **Push branch + open PR.**

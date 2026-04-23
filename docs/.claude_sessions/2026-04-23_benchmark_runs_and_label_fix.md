# Session: Spark v1+v2 benchmark runs + plot label fix

**Date:** 2026-04-23
**Branch:** `addison-dev`
**Commit(s):** `cd94b8b`
**Predecessors:**
- [2026-04-21_generation_benchmark.md](2026-04-21_generation_benchmark.md)
- [2026-04-22_benchmark_plotting_and_package_integration.md](2026-04-22_benchmark_plotting_and_package_integration.md)

## Goal

Re-run the v1 (full sweep) and v2 (large-batch sweep) Spark
benchmarks with the warmup pass enabled — the previous numbers were
collected before that landed and showed cold-start outliers. Then
review the resulting plots and fix any remaining language issues
before merging the benchmark stack into `addison-dev` for the planned
biom3-app merge.

## Runs executed

| Run | Config | Records | Notable |
|-----|--------|---------|---------|
| v1  | `stage3_bm_generation_ProteoScribe_1block_v1_spark_v1.json` | 108 | B ∈ {2..128}, both token strategies, D=8 |
| v2  | `stage3_bm_generation_ProteoScribe_1block_v1_spark_v2.json` | 9   | B ∈ {64,128,256}, N up to 1024, D=8, sample only |
| v2 (re-run) | same | 9 | reproducibility check |

All three on Spark / GB10. Plot dirs:

- `outputs/Stage3_generation_bm/spark_v1/20260423T125034Z/images/`
- `outputs/Stage3_generation_bm/spark_v2/20260423T130322Z/images/`
- `outputs/Stage3_generation_bm/spark_v2/20260423T133814Z/images/`

## Confirmed observations

1. **Warmup eliminates the first-call outlier.** Across 108 v1 runs,
   per-step time at the smallest B is now consistent across repeated
   measurements — no ~2× spike on the first call. The outlier visible
   in pre-2026-04-22 v1 plots is gone.

2. **Per-step time scales linearly with B from B=2 → 128.** Slope ≈ 1
   on log-log. `sample` and `argmax` overlap to within noise. Bigger
   sweep range than yesterday confirms the same picture: no flat
   "free batching" regime within tested values.

3. **Throughput peaks at the smallest B.** With clean data:
   - B=2: ~9.2 seq/s (some configs hit 9.4)
   - B=4: ~9.15 seq/s
   - B=8: ~8.55 seq/s
   - B=16-64: ~8.0-8.2 seq/s
   - B=128: ~8.1 seq/s (slight recovery)

   Yesterday's note that "B=4 is the sweet spot" was an artifact of
   the warmup outlier. With clean numbers, B=2 is genuinely fastest
   per sample, though the practical difference between B=2 and B=4 is
   ~1 %.

4. **v2 is reproducible.** Two runs of identical config differ by
   <1 % on T_total and throughput. The B=256/N=128 quantization
   step-up in the extrapolation plot (~32 min → ~68 min) is real and
   not noise.

5. **Memory headroom remains ample.** Largest measured allocation
   was 13.6 GB (v2, B=256, N=1024). GB10's ~128 GB unified memory
   means we're not memory-bound at any tested config.

## The "full sequences" mislabel

While reviewing v1 plots before the merge, noticed:

- `throughput_vs_batch.png` Y axis read `Throughput (full sequences / s)`
- `total_time_vs_N.png` X axis read `Total sequences N (full sequences generated)` and title read `(N full sequences × D=8 unmasking steps each)`

"Full" was misleading because in D-budgeted runs (the typical case for
these benchmarks) each generated sequence is only D positions
diffused — the rest is pre-filled with PAD. Saying "full sequences"
suggests every sequence was diffused over its complete 1024 positions,
which is not true for D ∈ {4, 8, 16}.

Fix in `cd94b8b`: drop "full" from both labels. `extrap_*.png` is
unchanged because it explicitly projects to D=1024 (the only place
"full sequences" really applies).

After the fix, all 6 existing run dirs were re-plotted to pick up the
new labels.

## Files touched

| File | Change |
|------|--------|
| `src/biom3/viz/benchmark.py` | Two Y/X labels and one suptitle: drop "full" qualifier in `plot_throughput_vs_batch` and `plot_total_time_vs_N`. |

Re-rendered images (no code change, just regenerated PNGs):
- `outputs/Stage3_generation_bm/spark/20260421T190657Z/images/`
- `outputs/Stage3_generation_bm/spark_v0/20260422T143547Z/images/`
- `outputs/Stage3_generation_bm/spark_v1/20260423T125034Z/images/`
- `outputs/Stage3_generation_bm/spark_v2/20260421T192558Z/images/`
- `outputs/Stage3_generation_bm/spark_v2/20260423T130322Z/images/`
- `outputs/Stage3_generation_bm/spark_v2/20260423T133814Z/images/`

## Updated practical guidance

For ProteoScribe_1block_v1 generation on GB10:

- **Smallest B you can tolerate is the throughput sweet spot.** B=2
  gives the best seq/s; B=4 is within 1%; B=64 costs ~12% throughput.
- **Memory is not a constraint** at any tested B — it's compute-bound
  even at B=2 (per-sample cost flat across B).
- **Avoid B > N**. The extrapolation plot shows the ceil-quantization
  penalty: B=256 with N=128 pays the full B=256 step cost for half
  the useful slots, doubling projected wall-clock.
- **D scales linearly.** Per-step time is independent of D, so D=1024
  total wall-clock = (D=8 total wall-clock) × 128. Plan accordingly:
  N=128 D=1024 sequences ≈ 32 min on GB10 at B ≤ 128.

## Things deliberately not done

- Did not push to origin yet — user will push as part of the merge
  workflow following the biom3-app merge.
- Did not write `Stage3/training.py` benchmark module. Still planned
  per the 2026-04-22 session note.
- Did not add `--no_warmup` flag. Warmup is cheap (~1.5 s) and always
  desirable for benchmark integrity; no use case for skipping it
  surfaced.
- Did not investigate the slight super-linearity in per-sample cost
  at large B (12.5 ms/sample at B=4 vs ~15.6 at B=64). Hypothesized
  cause is bandwidth/cache effects on the Gumbel-noise tensor; would
  need `torch.profiler` to confirm.

## Follow-ups

- Push `addison-dev` after the biom3-app merge lands.
- Stage3 training benchmark module + plotter (planned in
  2026-04-22 note).
- Multi-run / multi-machine plot overlays once Polaris/Aurora numbers
  exist.

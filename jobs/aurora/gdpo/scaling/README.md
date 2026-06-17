# GDPO weak-scaling study (v01)

Multi-node weak-scaling sweep for `biom3.rl.gdpo_multinode.gdpo_train_multinode` on Aurora. Designed to answer: **how does per-step wallclock evolve as we scale N nodes at constant per-rank workload, and where does rank 0's serial work (ELBO + opt step on the gathered BG batch) become the bottleneck?**

## Design

Per-rank workload `K = 12` sequences (one per local Aurora tile), fixed across N.
Group size `G = 24` fixed (matches `production_v02_multixpu`; preserves advantage-estimator variance).
Prompts per step `B = K·N / G = N / 2`.
Total replicas per step `BG = K·N = 12·N`, growing linearly with N.

| N (nodes) | B (prompts) | G | BG |
|-----------|-------------|----|------|
| 8 | 4 | 24 | 96 |
| 16 | 8 | 24 | 192 |
| 32 | 16 | 24 | 384 |
| 64 | 32 | 24 | 768 |
| 128 | 64 | 24 | 1536 |
| 256 | 128 | 24 | 3072 |
| 512 | 256 | 24 | 6144 |

Each run is **50 steps** with `save_steps=25` (two intermediate checkpoints + final). The analysis script discards step 1 (covers ESMFold lazy-load) and averages `step_time_s` over steps 2–50 → 49 steady-state samples. All jobs target the `debug-scaling` queue with 1-hour walltime; at ~58s/step empirical that gives ~50 min of runtime + 10 min headroom.

## What we expect to see

This v01 uses the **distributed-gradient trainer** ([src/biom3/rl/gdpo_multinode.py](../../../../src/biom3/rl/gdpo_multinode.py)): every rank holds π_new + π_ref + AdamW state and computes the gradient on its own shard before an all-reduce. Rank-0 work no longer grows with N — this is what makes scaling past N≈8 possible. (The earlier rank-0-only-training design OOM'd at N=8 because rank 0 had to forward+backward the full BG=96 batch on one 64 GB tile.)

- **Rollout + reward** is parallel across ranks (each does K=12 sequences). Should stay near-flat across N.
- **ELBO + PG-clip + KL + backward** is parallel across ranks (each computes its shard's gradient). Should also stay near-flat across N.
- **All-reduce on gradients** (~350 MB of trainable params per step over CCL fabric) grows logarithmically with N. Expect this to add a small fixed overhead that grows slowly with N.
- **Throughput in sequences/sec** = `BG / step_time_mean`. Ideal weak scaling → linear throughput; deviation tells you where the gradient all-reduce or comm overhead starts to dominate.

### Known limitation

- **`diversity_weight > 0` is not honored** in the distributed trainer. The diversity reward needs within-group sequences that span ranks; bringing them back to rank 0 defeats the point of distributing the gradient work. None of the scaling jobs set `diversity_weight`, so this doesn't affect the study.
- **debug.out's per-corruption mask-visualization section is omitted** in the distributed dump (rank 0 doesn't have the other ranks' SDMC corruptions). The per-replica scalar table is still faithful — those are all-reduced per replica.

## Running

Submit the 7 jobs from the project root:

```bash
for n in 8 16 32 64 128 256 512; do
    qsub jobs/aurora/gdpo/scaling/job_gdpo_scaling_n${n}.pbs
done
```

All jobs target `debug-scaling` with 01:00:00 walltime.

Outputs land under `./outputs/gdpo/<run_id>/` with run-id prefix `example_gdpo_scaling_v01_n<N>_BG<BG>_s10_*` — that prefix is what the analysis script globs against.

## Analysis

After all 5 runs complete:

```bash
python scripts/analyze_gdpo_scaling.py outputs/gdpo/example_gdpo_scaling_v01_*
```

Produces:

- **stdout**: human-readable summary table sorted by N.
- `scaling_summary.csv`: same data machine-readable.
- `scaling_plot.png`: two panels — per-step wallclock vs N (log-log, with error bars and BG annotations) and throughput vs N with an ideal-linear reference.

Flags:
- `--skip-steps K` drops steps ≤ K (default 1; the ESMFold warm-up).
- `--no-plot` / `--no-csv` to skip output files.

## Tuning knobs if reality surprises us

- **OOM at any N**: the distributed-gradient design caps per-rank state at ~17 GB static + ~3 GB activations regardless of BG, but if a specific tile gets unlucky with fragmentation, drop `K` from 12 to 6 by halving `batch_size` in the affected per-N job.
- **Step times much longer than the walltime budget**: lengthen the per-job walltime and resubmit.
- **One run flakes (preempted, hung, crashed)**: the analysis script silently skips runs whose `train_log.json` is missing or unparseable, then reports on whatever's left. Re-run the failed N and re-analyze. The trainer's failure-fast handler ([gdpo_multinode.py:gdpo_train_multinode](../../../../src/biom3/rl/gdpo_multinode.py)) calls `os._exit(1)` on any exception, so failed runs terminate cleanly (no qdel needed).

## Provenance

- Trainer: `src/biom3/rl/gdpo_multinode.py` (rank-sharded rollout + reward, rank-0-only training).
- Validated on a 2-node smoke (`jobs/aurora/gdpo/job_gdpo_smoke_n2.pbs`) before kicking off this study — `ws=2`, `uniq=24`, full batch reassembled, learning signal present.
- Weights: SH3 set from `production_v02_multixpu` (PenCL V09152023, Facilitator MMD15, ProteoScribe_SH3_epoch52).
- Reward: ESMFold pLDDT.
- KL: tokenwise_k3 (cheap, single fully-masked forward).
- SDMC quadrature: N=2 points (`t = 0.25, 0.75`).
- Prompts: `configs/grpo/prompts/sh3_v01_256.txt` — 256 distinct SH3-themed prompts sampled (seed=0) from `data/datasets/SH3/FINAL_SH3_all_dataset_with_prompts.csv`. Chosen so the largest B in the study (B=64 at N=128) still draws from a varied prompt pool rather than degenerating to the same prompt repeated B times.

#!/usr/bin/env python3
"""Aggregate train_log.json files from a GDPO scaling-study run-set.

Reads every ``train_log.json`` under the given run directories, drops the
warm-up step(s), and reports per-N:

  - steady-state per-step wallclock (mean ± std)
  - throughput in sequences/second (BG / mean_wallclock)
  - mean reward and final cumulative weight drift

Designed for the weak-scaling study run by ``jobs/aurora/gdpo/scaling/``.
Each run's train_log.json carries enough metadata in its ``_meta`` row
to identify N (= world_size), B, G; the script doesn't need to parse
run-id strings.

USAGE
    python scripts/analyze_gdpo_scaling.py outputs/gdpo/example_gdpo_scaling_v01_*
    python scripts/analyze_gdpo_scaling.py --skip-steps 2 outputs/gdpo/run_a outputs/gdpo/run_b
    python scripts/analyze_gdpo_scaling.py --no-plot outputs/gdpo/example_gdpo_scaling_v01_*

Outputs:
  - stdout: human-readable summary table
  - --csv PATH (default ./scaling_summary.csv): machine-readable summary
  - --plot PATH (default ./scaling_plot.png): wallclock + throughput vs N
"""

import argparse
import json
import os
import statistics
import sys
from typing import Dict, List, Optional


def _load_log(run_dir: str) -> Optional[List[dict]]:
    path = os.path.join(run_dir, "train_log.json")
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)


def _summarize_run(
    rows: List[dict],
    skip_steps: int,
    max_step: Optional[int] = None,
) -> Dict[str, object]:
    """Pull meta + steady-state stats out of one train_log.json.

    ``skip_steps`` drops the warm-up at the front (default 1 — discards
    the ESMFold-cold-load outlier). ``max_step``, if given, caps the
    inclusive upper bound — useful when some runs in a study were
    truncated and you want a uniform comparison window across all of
    them (e.g. ``skip_steps=1, max_step=41`` → exactly 40 samples
    from steps 2..41 on every run, even runs that completed 50 steps).
    """
    if not rows or not rows[0].get("_meta"):
        raise ValueError("train_log.json missing _meta row at index 0")
    meta = rows[0]

    # Step rows = everything after the meta row, skipping the warm-up
    # and any steps above the explicit cap.
    step_rows = [r for r in rows[1:] if "step" in r]
    used_rows = [
        r for r in step_rows
        if r["step"] > skip_steps
        and (max_step is None or r["step"] <= max_step)
    ]
    if not used_rows:
        raise ValueError(
            f"no rows survive after dropping steps ≤ {skip_steps}"
            f"{f' and steps > {max_step}' if max_step is not None else ''}; "
            f"have {len(step_rows)} step rows total"
        )

    step_times = [r["step_time_s"] for r in used_rows if "step_time_s" in r]
    if not step_times:
        raise ValueError(
            "no step_time_s field in step rows — re-run training with a "
            "trainer version that records per-step wallclock"
        )

    rewards = [r.get("reward") for r in used_rows if r.get("reward") is not None]
    dw_total_final = used_rows[-1].get("dw_total")

    N = int(meta["world_size"])
    B = int(meta.get("batch_size") or len(used_rows[-1].get("rewards_per_replica", []))
            // max(int(meta.get("num_generations") or 1), 1))
    G = int(meta.get("num_generations") or 1)
    BG = B * G

    wall_mean = statistics.mean(step_times)
    wall_std = statistics.pstdev(step_times) if len(step_times) > 1 else 0.0
    throughput = BG / wall_mean if wall_mean > 0 else float("nan")

    return {
        "N": N,
        "B": B,
        "G": G,
        "BG": BG,
        "steps_used": len(used_rows),
        "wall_mean_s": wall_mean,
        "wall_std_s": wall_std,
        "throughput_seq_per_s": throughput,
        "reward_mean": statistics.mean(rewards) if rewards else float("nan"),
        "dw_total_final": float(dw_total_final) if dw_total_final is not None else float("nan"),
        "kl_estimator": meta.get("kl_estimator"),
        "n_quadrature": meta.get("n_quadrature"),
        "pre_unmask": meta.get("pre_unmask"),
        "diffusion_budget": meta.get("diffusion_budget"),
        "sequence_length": meta.get("sequence_length"),
    }


def _print_table(summaries: List[Dict[str, object]]) -> None:
    if not summaries:
        print("(no runs)")
        return
    cols = ["N", "B", "G", "BG", "steps_used",
            "wall_mean_s", "wall_std_s", "throughput_seq_per_s",
            "reward_mean", "dw_total_final"]
    widths = {c: max(len(c), 6) for c in cols}
    for s in summaries:
        for c in cols:
            v = s[c]
            if isinstance(v, float):
                rendered = f"{v:.3f}"
            else:
                rendered = str(v)
            widths[c] = max(widths[c], len(rendered))
    header = " | ".join(c.rjust(widths[c]) for c in cols)
    print(header)
    print("-" * len(header))
    for s in summaries:
        row = " | ".join(
            (f"{s[c]:.3f}" if isinstance(s[c], float) else str(s[c])).rjust(widths[c])
            for c in cols
        )
        print(row)


def _write_csv(summaries: List[Dict[str, object]], path: str) -> None:
    import csv
    cols = list(summaries[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for s in summaries:
            w.writerow(s)
    print(f"Wrote summary CSV: {path}")


def _plot(summaries: List[Dict[str, object]], path: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping plot")
        return

    Ns = [s["N"] for s in summaries]
    walls = [s["wall_mean_s"] for s in summaries]
    wall_stds = [s["wall_std_s"] for s in summaries]
    throughputs = [s["throughput_seq_per_s"] for s in summaries]
    BGs = [s["BG"] for s in summaries]

    fig, (ax_wall, ax_tput) = plt.subplots(1, 2, figsize=(12, 5))

    ax_wall.errorbar(Ns, walls, yerr=wall_stds, marker="o", capsize=4)
    ax_wall.set_xscale("log", base=2)
    ax_wall.set_yscale("log")
    ax_wall.set_xlabel("Number of nodes (N)")
    ax_wall.set_ylabel("Steady-state wallclock per step (s)")
    ax_wall.set_title("Per-step wallclock vs N (weak scaling, K=12/rank)")
    ax_wall.grid(True, which="both", linestyle=":", alpha=0.5)
    for n, w, bg in zip(Ns, walls, BGs):
        ax_wall.annotate(f"BG={bg}", (n, w), textcoords="offset points",
                         xytext=(6, 4), fontsize=8)

    ax_tput.plot(Ns, throughputs, marker="o", label="measured")
    ax_tput.set_xscale("log", base=2)
    ax_tput.set_yscale("log")
    ax_tput.set_xlabel("Number of nodes (N)")
    ax_tput.set_ylabel("Throughput (sequences / second)")
    ax_tput.set_title("Throughput vs N (log–log; ideal slope = 1)")
    ax_tput.grid(True, which="both", linestyle=":", alpha=0.5)
    # Ideal-scaling reference: linear-in-N from the smallest-N point.
    # On a log-log axis this appears as a straight diagonal with slope 1.
    if len(Ns) >= 2:
        t0 = throughputs[0]
        n0 = Ns[0]
        ideal = [t0 * (n / n0) for n in Ns]
        ax_tput.plot(Ns, ideal, linestyle="--", alpha=0.6, label="ideal linear")
        ax_tput.legend(loc="best")

    fig.suptitle("GDPO weak-scaling study", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Wrote plot: {path}")


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dirs", nargs="+",
                    help="Run output directories (each containing train_log.json). "
                         "Shell glob expansion is supported.")
    ap.add_argument("--skip-steps", type=int, default=1,
                    help="Discard steps ≤ this (default 1, drops ESMFold warm-up).")
    ap.add_argument("--max-step", type=int, default=None,
                    help="Cap the upper bound at this step (inclusive). "
                         "Use with --skip-steps to fix a uniform analysis window "
                         "across runs of differing length (e.g. "
                         "--skip-steps 1 --max-step 41 → 40 samples from steps 2..41).")
    ap.add_argument("--csv", default="./scaling_summary.csv",
                    help="Path for the summary CSV.")
    ap.add_argument("--plot", default="./scaling_plot.png",
                    help="Path for the matplotlib plot.")
    ap.add_argument("--no-plot", action="store_true",
                    help="Skip plot generation.")
    ap.add_argument("--no-csv", action="store_true",
                    help="Skip CSV writing.")
    args = ap.parse_args(argv)

    summaries: List[Dict[str, object]] = []
    for rd in args.run_dirs:
        rd_abs = os.path.abspath(rd)
        rows = _load_log(rd)
        if rows is None:
            print(f"warning: no train_log.json in {rd} — skipping",
                  file=sys.stderr)
            continue
        try:
            s = _summarize_run(rows, skip_steps=args.skip_steps, max_step=args.max_step)
        except Exception as e:
            print(f"warning: could not summarize {rd}: {e}", file=sys.stderr)
            continue
        s["run_dir"] = rd_abs
        summaries.append(s)

    if not summaries:
        print("No usable runs found.", file=sys.stderr)
        return 1

    summaries.sort(key=lambda s: s["N"])
    _print_table(summaries)
    if not args.no_csv:
        _write_csv(summaries, args.csv)
    if not args.no_plot:
        _plot(summaries, args.plot)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

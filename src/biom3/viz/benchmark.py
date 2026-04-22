"""Plot benchmark results written by ``biom3.benchmarks.generation``.

Reads ``results.npz`` + ``env.json`` from a benchmark run directory and
writes PNGs into ``<run_dir>/images/``:

- ``time_per_step_vs_batch.png``   — primary: time per diffusion step vs B
- ``peak_memory_vs_batch.png``     — peak device memory vs B
- ``throughput_vs_batch.png``      — sequences generated per second vs B
- ``total_time_vs_N.png``          — total run time vs N, one line per B
- ``extrap_N{N}_D{D}.png``         — projected total time at full-length D

CLI entry point: ``biom3_plot_benchmark --run_dir <path>``.
"""

import argparse
import json
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_arguments(args):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", required=True,
                        help="Benchmark run directory (contains results.npz, env.json)")
    parser.add_argument("--image_dir", default=None,
                        help="Override output dir (default: <run_dir>/images/)")
    parser.add_argument("--extrap_N", type=int, default=128,
                        help="Target N for extrapolation plot (default: 128)")
    parser.add_argument("--extrap_D", type=int, default=1024,
                        help="Target D (full-length diffusion) for extrapolation (default: 1024)")
    return parser.parse_args(args)


_SUPPORTED_BENCHMARK_TYPES = ("stage3_generation",)


def _load(run_dir):
    npz = np.load(os.path.join(run_dir, "results.npz"), allow_pickle=True)
    with open(os.path.join(run_dir, "env.json")) as fh:
        env = json.load(fh)
    bt = env.get("benchmark_type")
    if bt is not None and bt not in _SUPPORTED_BENCHMARK_TYPES:
        raise ValueError(
            f"viz.benchmark only supports {_SUPPORTED_BENCHMARK_TYPES}; "
            f"this run is benchmark_type={bt!r}. Use the matching plotter."
        )
    return npz, env


def _title_suffix(env):
    return (
        f"{env.get('arch_id', '?')} · "
        f"{env.get('device_name', '?')} · "
        f"{env.get('hostname', '?')}"
    )


def _token_strategies(npz):
    return sorted(set(str(t) for t in npz["token_strategy"]))


def _D_str(npz):
    """Compact description of the diffusion-step budget(s) in this run."""
    Ds = sorted(set(int(d) for d in npz["D"]))
    if len(Ds) == 1:
        return f"D={Ds[0]}"
    return f"D∈{{{','.join(str(d) for d in Ds)}}}"


_STYLE = {
    "sample":  {"color": "tab:blue",   "marker": "o", "label": "sample"},
    "argmax":  {"color": "tab:orange", "marker": "s", "label": "argmax"},
}


def _style(token_strategy):
    return _STYLE.get(token_strategy, {
        "color": "tab:gray", "marker": "^", "label": token_strategy,
    })


def _annotate_runs(ax, env, n_records):
    ax.text(
        0.01, 0.99,
        f"runs: {n_records}",
        transform=ax.transAxes, va="top", ha="left",
        fontsize=8, color="gray",
    )


def plot_time_per_step_vs_batch(npz, env, out_path):
    """Per-step time vs batch size.

    Per-step time = T_total_s / (num_batches * D). This is the average
    wall-clock cost of one diffusion step for a batch of size B. It should
    depend primarily on B (and token_strategy); N and P should not matter —
    clustering of same-B points is the visual confirmation.
    """
    B = npz["B"]
    T_total = npz["T_total_s"]
    nb = npz["num_batches"]
    D = npz["D"]
    ts = np.array([str(t) for t in npz["token_strategy"]])
    per_step = T_total / (nb * D)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for strategy in _token_strategies(npz):
        mask = ts == strategy
        s = _style(strategy)
        ax.scatter(B[mask], per_step[mask] * 1000.0,
                   alpha=0.6, s=40, **{k: s[k] for k in ("color", "marker")},
                   label=f"{s['label']} (per run)")
        unique_b = np.unique(B[mask])
        med = np.array([np.median(per_step[mask][B[mask] == b]) for b in unique_b])
        ax.plot(unique_b, med * 1000.0, color=s["color"], linewidth=1.5,
                linestyle="--", alpha=0.9, label=f"{s['label']} (median)")

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Batch size B (sequences processed in parallel)")
    ax.set_ylabel("Time per diffusion step (ms)")
    ax.set_title(
        "Time per unmasking step vs batch size\n"
        "(each step unmasks 1 position across B parallel sequences)\n"
        f"{_title_suffix(env)}",
        fontsize=10,
    )
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    _annotate_runs(ax, env, len(B))
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_peak_memory_vs_batch(npz, env, out_path):
    """Peak allocated memory vs batch size."""
    B = npz["B"]
    peak_alloc = npz["peak_alloc_bytes"]
    valid = ~np.isnan(peak_alloc)
    if not valid.any():
        return False
    ts = np.array([str(t) for t in npz["token_strategy"]])

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for strategy in _token_strategies(npz):
        mask = (ts == strategy) & valid
        if not mask.any():
            continue
        s = _style(strategy)
        gb = peak_alloc[mask] / (1024 ** 3)
        ax.scatter(B[mask], gb, alpha=0.6, s=40,
                   **{k: s[k] for k in ("color", "marker")},
                   label=f"{s['label']} (per run)")
        unique_b = np.unique(B[mask])
        med = np.array([
            np.median(peak_alloc[mask][B[mask] == b]) / (1024 ** 3)
            for b in unique_b
        ])
        ax.plot(unique_b, med, color=s["color"], linewidth=1.5,
                linestyle="--", alpha=0.9, label=f"{s['label']} (median)")

    ax.set_xscale("log", base=2)
    ax.set_xlabel("Batch size B (sequences processed in parallel)")
    ax.set_ylabel("Peak GPU memory allocated (GB)")
    ax.set_title(
        "Peak device memory vs batch size\n"
        "(during one forward pass; sequence length = 1024)\n"
        f"{_title_suffix(env)}",
        fontsize=10,
    )
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    _annotate_runs(ax, env, len(B))
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return True


def plot_throughput_vs_batch(npz, env, out_path):
    """Sequences/sec vs batch size."""
    B = npz["B"]
    N = npz["N"]
    T = npz["T_total_s"]
    ts = np.array([str(t) for t in npz["token_strategy"]])
    throughput = N / T

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for strategy in _token_strategies(npz):
        mask = ts == strategy
        s = _style(strategy)
        ax.scatter(B[mask], throughput[mask], alpha=0.6, s=40,
                   **{k: s[k] for k in ("color", "marker")},
                   label=f"{s['label']} (per run)")
        unique_b = np.unique(B[mask])
        med = np.array([
            np.median(throughput[mask][B[mask] == b]) for b in unique_b
        ])
        ax.plot(unique_b, med, color=s["color"], linewidth=1.5,
                linestyle="--", alpha=0.9, label=f"{s['label']} (median)")

    ax.set_xscale("log", base=2)
    ax.set_xlabel("Batch size B (sequences processed in parallel)")
    ax.set_ylabel("Throughput (full sequences / s)")
    ax.set_title(
        "Generation throughput vs batch size\n"
        f"(1 sequence = {_D_str(npz)} unmasking steps)\n"
        f"{_title_suffix(env)}",
        fontsize=10,
    )
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    _annotate_runs(ax, env, len(B))
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_extrapolated_time(npz, env, out_path, target_N, target_D):
    """Project total wall-clock to produce ``target_N`` sequences at full-length
    ``target_D`` diffusion, using per-step time measured in this sweep.

    Extrapolation: T = ceil(N / B) * D * t_per_step, where
    t_per_step = T_total_s / (num_batches * D_measured). This holds because
    per-step time is empirically independent of D (each step is a full
    transformer forward regardless of how many positions are masked).
    """
    B = npz["B"]
    T_total = npz["T_total_s"]
    nb = npz["num_batches"]
    D = npz["D"]
    ts = np.array([str(t) for t in npz["token_strategy"]])
    t_per_step = T_total / (nb * D)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for strategy in _token_strategies(npz):
        mask = ts == strategy
        s = _style(strategy)
        target_nb = np.ceil(target_N / B[mask]).astype(int)
        extrap = target_nb * target_D * t_per_step[mask]
        ax.scatter(B[mask], extrap / 60.0, alpha=0.6, s=40,
                   **{k: s[k] for k in ("color", "marker")},
                   label=f"{s['label']} (per run)")
        unique_b = np.unique(B[mask])
        med = []
        for b in unique_b:
            sub = (B[mask] == b)
            med.append(np.median(np.ceil(target_N / b) * target_D
                                 * t_per_step[mask][sub]))
        ax.plot(unique_b, np.array(med) / 60.0, color=s["color"],
                linewidth=1.5, linestyle="--", alpha=0.9,
                label=f"{s['label']} (median)")

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Batch size B (sequences processed in parallel)")
    ax.set_ylabel("Projected total wall-clock (min)")
    ax.set_title(
        f"Projected time to generate N={target_N} full sequences\n"
        f"({target_N} sequences × {target_D} unmasking steps each)\n"
        f"{_title_suffix(env)}",
        fontsize=10,
    )
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    ax.text(
        0.01, 0.99,
        f"runs: {len(B)}  (extrapolated from per-step time measured at {_D_str(npz)})",
        transform=ax.transAxes, va="top", ha="left",
        fontsize=8, color="gray",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_total_time_vs_N(npz, env, out_path):
    """Total time vs N, one line per batch size, faceted by token_strategy."""
    strategies = _token_strategies(npz)
    fig, axes = plt.subplots(1, len(strategies),
                             figsize=(6.0 * len(strategies), 4.5),
                             squeeze=False, sharey=True)
    ts = np.array([str(t) for t in npz["token_strategy"]])
    N = npz["N"]
    B = npz["B"]
    T = npz["T_total_s"]

    for ax, strategy in zip(axes[0], strategies):
        mask = ts == strategy
        for b in sorted(np.unique(B[mask])):
            sub = mask & (B == b)
            order = np.argsort(N[sub])
            ax.plot(N[sub][order], T[sub][order],
                    marker="o", linewidth=1.3, label=f"B={int(b)}")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("Total sequences N (full sequences generated)")
        ax.set_title(f"token_strategy = {strategy}", fontsize=10)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8, loc="best")
    axes[0, 0].set_ylabel("Total wall-clock time (s)")
    fig.suptitle(
        f"Total generation time vs N\n"
        f"(N full sequences × {_D_str(npz)} unmasking steps each)\n"
        f"{_title_suffix(env)}",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main(args):
    run_dir = os.path.abspath(args.run_dir)
    image_dir = args.image_dir or os.path.join(run_dir, "images")
    os.makedirs(image_dir, exist_ok=True)

    npz, env = _load(run_dir)

    plot_time_per_step_vs_batch(
        npz, env, os.path.join(image_dir, "time_per_step_vs_batch.png"),
    )
    plot_peak_memory_vs_batch(
        npz, env, os.path.join(image_dir, "peak_memory_vs_batch.png"),
    )
    plot_throughput_vs_batch(
        npz, env, os.path.join(image_dir, "throughput_vs_batch.png"),
    )
    plot_total_time_vs_N(
        npz, env, os.path.join(image_dir, "total_time_vs_N.png"),
    )
    plot_extrapolated_time(
        npz, env,
        os.path.join(image_dir,
                     f"extrap_N{args.extrap_N}_D{args.extrap_D}.png"),
        target_N=args.extrap_N, target_D=args.extrap_D,
    )
    print(f"Wrote plots to {image_dir}")


if __name__ == "__main__":
    main(parse_arguments(sys.argv[1:]))

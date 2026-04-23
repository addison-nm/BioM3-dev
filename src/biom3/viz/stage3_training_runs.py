"""Loading and plotting utilities for Stage 3 training-run artifacts.

A Stage 3 training run (pretraining or finetuning) writes an ``artifacts/``
directory with:

- ``metrics_history.pt`` — dict ``{"train": {...}, "val": {...}}`` of numpy
  arrays (per-step train metrics, per-epoch val metrics). Written by
  :class:`biom3.Stage3.callbacks.MetricsHistoryCallback`.
- ``metrics_history.rank{NNN}.pt`` — optional per-rank val_loss diagnostics
  (only present when training was run with ``all_ranks_val_loss=True``).
- ``args.json``, ``checkpoint_summary.json``, ``build_manifest.json``,
  ``benchmark_history.json`` — run metadata.

This module exposes loaders and plotters mirroring the
:mod:`biom3.viz.dynamics` pattern (typed arrays in, ``matplotlib`` Figure
out, with ``_from_dir`` convenience wrappers).
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Literal

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.figure
import matplotlib.pyplot as plt


_METRIC_FILENAME = "metrics_history.pt"
_RANK_PATTERN = re.compile(r"metrics_history\.rank(\d+)\.pt$")
_SUFFIXES = ("_epoch", "_step")

_SPLIT_STYLES = {
    "train": {"linestyle": "-", "marker": None, "linewidth": 1.8},
    "val":   {"linestyle": "--", "marker": "o", "markersize": 5, "linewidth": 1.8},
}
_SPLIT_COLORS = {"train": "#1f77b4", "val": "#d62728"}


def _finite_mask(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Drop (x, y) pairs where either is non-finite."""
    x = np.asarray(x)
    y = np.asarray(y)
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]


def _strip_split_prefix(key: str, split: str) -> str:
    """Remove ``{split}_`` prefix; pass-through if absent (e.g. ``loss_gap``)."""
    prefix = f"{split}_"
    return key[len(prefix):] if key.startswith(prefix) else key


def _parse_metric_key(key: str, split: str) -> tuple[str, str | None]:
    """Return ``(base_name, cadence)`` where cadence is ``"step"``, ``"epoch"``, or ``None``.

    ``None`` cadence means the key has no explicit suffix (e.g. raw
    ``train_loss``, ``loss_gap``) — callers decide how to treat it.
    """
    stripped = _strip_split_prefix(key, split)
    for suf in _SUFFIXES:
        if stripped.endswith(suf):
            return stripped[: -len(suf)], suf[1:]
    return stripped, None


def _build_split_metrics(split_raw: dict, split: str) -> dict[str, dict[str, tuple[np.ndarray, np.ndarray]]]:
    """Normalize a single split's raw metric dict into per-metric cadence views.

    Returns ``{base_name: {"by_step": (x, y), "by_epoch": (x, y)}, ...}``.
    Only cadences with a corresponding x-axis array populated get included.
    """
    if not split_raw:
        return {}
    step_x = split_raw.get("global_step")
    epoch_x = split_raw.get("epoch")

    grouped: dict[str, dict[str, np.ndarray]] = {}
    for key, arr in split_raw.items():
        if key in ("global_step", "epoch"):
            continue
        arr = np.asarray(arr)
        base, cadence = _parse_metric_key(key, split)
        slot = grouped.setdefault(base, {})
        if cadence is None:
            # Bare form (e.g. train_loss, loss_gap) — assume per-record cadence;
            # matches step_x length for train, epoch_x length for val.
            slot.setdefault("_bare", arr)
        else:
            slot[cadence] = arr

    metrics: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    for base, arrays in grouped.items():
        out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        step_arr = arrays.get("step", arrays.get("_bare"))
        epoch_arr = arrays.get("epoch")

        if step_arr is not None and step_x is not None and len(step_arr) == len(step_x):
            out["by_step"] = _finite_mask(step_x, step_arr)
        elif step_arr is not None and epoch_x is not None and step_x is not None and len(step_arr) == len(epoch_x):
            # Bare form with epoch cadence (val has no _step keys; bare == epoch)
            out["by_step"] = _finite_mask(step_x, step_arr)

        if epoch_arr is not None and epoch_x is not None and len(epoch_arr) == len(epoch_x):
            out["by_epoch"] = _finite_mask(epoch_x, epoch_arr)
        elif step_arr is not None and epoch_x is not None and len(step_arr) == len(epoch_x) and "by_epoch" not in out:
            # Bare form: epoch cadence from bare key
            out["by_epoch"] = _finite_mask(epoch_x, step_arr)

        if out:
            metrics[base] = out
    return metrics


def _build_per_rank_val_loss(rank_raw: dict) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Extract val_loss per-rank cadence views from a rank file's ``val`` dict."""
    val = rank_raw.get("val", {})
    step_x = val.get("global_step")
    epoch_x = val.get("epoch")

    y_step = val.get("val_loss") if val.get("val_loss") is not None else val.get("val_loss_step")
    y_epoch = val.get("val_loss_epoch") if val.get("val_loss_epoch") is not None else val.get("val_loss")

    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    if y_step is not None and step_x is not None and len(y_step) == len(step_x):
        out["by_step"] = _finite_mask(step_x, y_step)
    if y_epoch is not None and epoch_x is not None and len(y_epoch) == len(epoch_x):
        out["by_epoch"] = _finite_mask(epoch_x, y_epoch)
    return out


def _load_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def load_run_artifacts(artifacts_dir: str | os.PathLike) -> dict:
    """Load a Stage 3 run's ``artifacts/`` directory into a normalized dict.

    Parameters
    ----------
    artifacts_dir : path-like
        Directory containing ``metrics_history.pt`` (required) and optional
        sibling files.

    Returns
    -------
    dict
        Keys:

        ``artifacts_dir``
            Resolved :class:`~pathlib.Path`.
        ``metrics``
            ``{"train": {base: {"by_step": (x, y), "by_epoch": (x, y)}, ...}, "val": {...}}``.
        ``per_rank_val_loss``
            ``{rank_id: {"by_step": (x, y), "by_epoch": (x, y)}}`` — empty when
            no rank files are present.
        ``available_ranks``
            Sorted list of rank IDs found in per-rank files.
        ``args``, ``checkpoint_summary``, ``build_manifest``, ``benchmark_history``
            Parsed JSON dicts, or ``None`` when absent.
        ``raw``
            Original ``{"train", "val"}`` dict for power-user access.
    """
    root = Path(artifacts_dir).expanduser().resolve()
    metrics_path = root / _METRIC_FILENAME
    if not metrics_path.is_file():
        raise FileNotFoundError(
            f"{metrics_path} not found. Pass the artifacts/ directory of a "
            "Stage 3 training run."
        )

    raw = torch.load(metrics_path, weights_only=False, map_location="cpu")
    train_metrics = _build_split_metrics(raw.get("train", {}), "train")
    val_metrics = _build_split_metrics(raw.get("val", {}), "val")

    per_rank: dict[int, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    for path in sorted(root.glob("metrics_history.rank*.pt")):
        m = _RANK_PATTERN.search(path.name)
        if not m:
            continue
        rank_id = int(m.group(1))
        rank_raw = torch.load(path, weights_only=False, map_location="cpu")
        series = _build_per_rank_val_loss(rank_raw)
        if series:
            per_rank[rank_id] = series

    return {
        "artifacts_dir": root,
        "metrics": {"train": train_metrics, "val": val_metrics},
        "per_rank_val_loss": per_rank,
        "available_ranks": sorted(per_rank.keys()),
        "args": _load_json(root / "args.json"),
        "checkpoint_summary": _load_json(root / "checkpoint_summary.json"),
        "build_manifest": _load_json(root / "build_manifest.json"),
        "benchmark_history": _load_json(root / "benchmark_history.json"),
        "raw": raw,
    }


def discover_metric_bases(run_data: dict) -> dict[str, list[str]]:
    """Return sorted base metric names available in ``run_data``, per split."""
    return {
        "train": sorted(run_data["metrics"]["train"].keys()),
        "val": sorted(run_data["metrics"]["val"].keys()),
    }


def _pick_cadence_series(
    cadence_dict: dict[str, tuple[np.ndarray, np.ndarray]],
    x_axis: str,
) -> tuple[np.ndarray, np.ndarray] | None:
    key = "by_step" if x_axis == "step" else "by_epoch"
    return cadence_dict.get(key)


def plot_metric(
    run_data: dict,
    metric_base: str,
    *,
    x_axis: Literal["step", "epoch"] = "step",
    splits: tuple[str, ...] = ("train", "val"),
    ranks: list[int] | None = None,
    log_y: bool = False,
    figsize: tuple[float, float] = (10, 5),
    title: str | None = None,
) -> matplotlib.figure.Figure:
    """Plot one metric across selected splits, optionally with per-rank overlay.

    Parameters
    ----------
    run_data : dict
        Return value of :func:`load_run_artifacts`.
    metric_base : str
        Base metric name (e.g. ``"loss"``, ``"prev_hard_acc"``, ``"loss_gap"``).
    x_axis : {"step", "epoch"}
        X-axis selection. For split/metric combinations that only have epoch
        cadence (e.g. all val metrics), ``"step"`` renders points at the
        recorded ``global_step`` values (sparse markers).
    splits : tuple of str
        Subset of ``("train", "val")`` to render.
    ranks : list[int], optional
        Rank IDs to overlay (only applies when ``metric_base == "loss"`` and
        per-rank data exists).
    log_y : bool
        Set ``ax.set_yscale("log")``.
    figsize : tuple[float, float]
    title : str, optional
        Override the default title (``metric_base``).

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    any_plotted = False

    for split in splits:
        if split not in run_data["metrics"]:
            continue
        split_metrics = run_data["metrics"][split]
        if metric_base not in split_metrics:
            continue
        series = _pick_cadence_series(split_metrics[metric_base], x_axis)
        if series is None or len(series[0]) == 0:
            continue
        x, y = series
        style = _SPLIT_STYLES[split]
        ax.plot(
            x, y,
            color=_SPLIT_COLORS[split],
            label=f"{split}",
            **style,
        )
        any_plotted = True

    if metric_base == "loss" and ranks:
        rank_colors = plt.cm.viridis(np.linspace(0.15, 0.85, max(len(ranks), 1)))
        for color, rank_id in zip(rank_colors, ranks):
            series = run_data["per_rank_val_loss"].get(rank_id)
            if series is None:
                continue
            pick = _pick_cadence_series(series, x_axis)
            if pick is None or len(pick[0]) == 0:
                continue
            x, y = pick
            ax.plot(
                x, y,
                color=color,
                linestyle=":",
                linewidth=1.2,
                label=f"val_loss (rank {rank_id})",
            )
            any_plotted = True

    ax.set_xlabel("Global step" if x_axis == "step" else "Epoch")
    ax.set_ylabel(metric_base)
    ax.set_title(title if title is not None else metric_base)
    if log_y:
        ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    if any_plotted:
        ax.legend(loc="best", frameon=True)
    else:
        ax.text(
            0.5, 0.5,
            f"No data for metric '{metric_base}' with splits={list(splits)} "
            f"on {x_axis} axis",
            ha="center", va="center", transform=ax.transAxes,
            color="gray",
        )
    fig.tight_layout()
    return fig


def plot_benchmark_history(
    run_data: dict,
    *,
    kind: Literal["throughput", "memory"] = "throughput",
    per_rank: bool = False,
    figsize: tuple[float, float] = (10, 5),
) -> matplotlib.figure.Figure:
    """Plot per-epoch throughput or memory from ``benchmark_history.json``.

    Parameters
    ----------
    run_data : dict
        Return value of :func:`load_run_artifacts`. Requires
        ``run_data["benchmark_history"]`` to be populated.
    kind : {"throughput", "memory"}
        ``throughput`` → samples/sec + steps/sec per epoch.
        ``memory`` → peak allocated / reserved GB per epoch.
    per_rank : bool
        When ``kind == "memory"`` and per-rank lists are present, overlay
        each rank as a thin line.
    figsize : tuple[float, float]

    Returns
    -------
    matplotlib.figure.Figure
    """
    bench = run_data.get("benchmark_history")
    if not bench or not bench.get("epochs"):
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(
            0.5, 0.5, "No benchmark_history.json available",
            ha="center", va="center", transform=ax.transAxes, color="gray",
        )
        ax.set_xticks([])
        ax.set_yticks([])
        return fig

    epochs = bench["epochs"]
    x = np.array([e["epoch"] for e in epochs])

    fig, ax = plt.subplots(figsize=figsize)

    if kind == "throughput":
        samples = np.array([e.get("samples_per_sec", np.nan) for e in epochs], dtype=float)
        steps = np.array([e.get("steps_per_sec", np.nan) for e in epochs], dtype=float)
        x_s, y_s = _finite_mask(x, samples)
        x_st, y_st = _finite_mask(x, steps)
        ax.plot(x_s, y_s, color="#1f77b4", marker="o", label="samples/sec")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("samples / sec", color="#1f77b4")
        ax.tick_params(axis="y", labelcolor="#1f77b4")

        ax2 = ax.twinx()
        ax2.plot(x_st, y_st, color="#2ca02c", marker="s", label="steps/sec", linestyle="--")
        ax2.set_ylabel("steps / sec", color="#2ca02c")
        ax2.tick_params(axis="y", labelcolor="#2ca02c")
        ax.set_title("Throughput per epoch")
        ax.grid(True, alpha=0.3)

        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc="best")
    elif kind == "memory":
        alloc = np.array([e.get("peak_memory_allocated_gb", np.nan) for e in epochs], dtype=float)
        reserved = np.array([e.get("peak_memory_reserved_gb", np.nan) for e in epochs], dtype=float)
        x_a, y_a = _finite_mask(x, alloc)
        x_r, y_r = _finite_mask(x, reserved)
        if len(y_a):
            ax.plot(x_a, y_a, color="#d62728", marker="o", label="peak allocated")
        if len(y_r):
            ax.plot(x_r, y_r, color="#ff7f0e", marker="s", linestyle="--", label="peak reserved")

        if per_rank:
            per_rank_alloc = [e.get("peak_memory_allocated_gb_per_rank") for e in epochs]
            if any(isinstance(v, list) and v for v in per_rank_alloc):
                arr = np.array([
                    v if isinstance(v, list) and v else [np.nan]
                    for v in per_rank_alloc
                ], dtype=object)
                max_ranks = max(len(v) for v in per_rank_alloc if isinstance(v, list))
                rank_colors = plt.cm.plasma(np.linspace(0.15, 0.85, max(max_ranks, 1)))
                for r in range(max_ranks):
                    y_r_alloc = np.array([
                        (v[r] if isinstance(v, list) and r < len(v) else np.nan)
                        for v in per_rank_alloc
                    ], dtype=float)
                    xr, yr = _finite_mask(x, y_r_alloc)
                    if len(yr):
                        ax.plot(
                            xr, yr,
                            color=rank_colors[r], linestyle=":", linewidth=1.0,
                            label=f"alloc rank {r}",
                        )
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Peak memory (GB)")
        ax.set_title("Peak memory per epoch")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
    else:
        raise ValueError(f"kind must be 'throughput' or 'memory', got {kind!r}")

    fig.tight_layout()
    return fig


def plot_metric_from_dir(
    artifacts_dir: str | os.PathLike,
    metric_base: str,
    **kwargs,
) -> matplotlib.figure.Figure:
    """Convenience wrapper: load ``artifacts_dir`` then call :func:`plot_metric`."""
    run_data = load_run_artifacts(artifacts_dir)
    return plot_metric(run_data, metric_base, **kwargs)


def plot_benchmark_from_dir(
    artifacts_dir: str | os.PathLike,
    **kwargs,
) -> matplotlib.figure.Figure:
    """Convenience wrapper: load ``artifacts_dir`` then call :func:`plot_benchmark_history`."""
    run_data = load_run_artifacts(artifacts_dir)
    return plot_benchmark_history(run_data, **kwargs)

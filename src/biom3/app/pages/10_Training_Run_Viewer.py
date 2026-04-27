"""Stage 3 pretraining / finetuning run viewer.

Point at an ``artifacts/`` directory (or any ``metrics_history.pt`` inside one)
and the page renders:

- Run summary (best checkpoint, git/build info).
- Hyperparameters grouped by category (with full-args escape hatch).
- Learning curves — selectable metric × x-axis × splits, optional per-rank
  ``val_loss`` overlay when ``metrics_history.rank*.pt`` files exist.
- Benchmark history (throughput + memory per epoch).
- Raw metric explorer.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from biom3.app._data_browser import get_data_dirs
from biom3.viz.stage3_training_runs import (
    discover_metric_bases,
    load_run_artifacts,
    plot_benchmark_history,
    plot_metric,
)


@st.cache_data(show_spinner=False)
def _find_artifact_dirs(root: str) -> list[str]:
    """Walk ``root`` following symlinks, returning directories that contain
    ``metrics_history.pt``. Returned as strings so Streamlit can cache them."""
    root_path = Path(root)
    if not root_path.is_dir():
        return []
    found: list[str] = []
    for dirpath, _dirnames, filenames in os.walk(root_path, followlinks=True):
        if "metrics_history.pt" in filenames:
            found.append(dirpath)
    found.sort()
    return found


def _pick_run_artifacts() -> Path | None:
    """Two-step picker: data directory → run artifacts folder.

    Unlike the shared :func:`browse_file`, this walks with
    ``followlinks=True`` so symlinked weights trees (produced by
    ``scripts/link_weights.sh`` in worktrees) are traversed.
    """
    dirs = get_data_dirs()
    if not dirs:
        st.info(
            "No data directories configured. Provide an app settings JSON via "
            "`biom3_app --config <path>` or the `BIOM3_APP_CONFIG` env var."
        )
        return None
    labels = [d["label"] for d in dirs]
    default_idx = labels.index("Weights") if "Weights" in labels else 0
    chosen = st.selectbox(
        "Data directory", labels, index=default_idx, key="tv_data_dir",
    )
    root = Path(next(d["path"] for d in dirs if d["label"] == chosen))
    if not root.is_dir():
        st.warning(f"Directory `{root}` does not exist")
        return None

    runs = _find_artifact_dirs(str(root))
    if not runs:
        st.warning(
            f"No `metrics_history.pt` files found under `{root}`. "
            "Stage 3 runs write this file into their `artifacts/` directory."
        )
        return None

    filter_text = st.text_input(
        "Filter", key="tv_filter",
        placeholder="substring match (case-insensitive)",
    ).strip().lower()
    entries = [
        (str(Path(r).relative_to(root)) if str(r).startswith(str(root)) else r, r)
        for r in runs
    ]
    if filter_text:
        entries = [(name, path) for name, path in entries if filter_text in name.lower()]
    if not entries:
        st.warning(f"No run matches filter `{filter_text}`")
        return None
    names = [name for name, _ in entries]
    chosen_name = st.selectbox("Run (artifacts directory)", names, key="tv_run")
    chosen_path = entries[names.index(chosen_name)][1]
    return Path(chosen_path)


_ARGS_SECTIONS = {
    "Optimization": [
        "lr", "weight_decay", "warmup_steps", "total_steps", "max_steps",
        "epochs", "batch_size", "acc_grad_batches", "training_strategy",
        "precision",
    ],
    "Model": [
        "model_option", "transformer_dim", "transformer_heads",
        "transformer_depth", "dropout", "diffusion_steps",
    ],
    "Run": [
        "description", "tags", "notes", "gpu_devices", "num_nodes", "seed",
        "data_root", "pretrained_weights",
    ],
}


@st.cache_data(show_spinner=False)
def _cached_load(artifacts_dir_str: str, mtime: float):
    return load_run_artifacts(artifacts_dir_str)


def _load_with_cache(artifacts_dir: Path) -> dict:
    metrics_path = artifacts_dir / "metrics_history.pt"
    mtime = metrics_path.stat().st_mtime if metrics_path.is_file() else 0.0
    return _cached_load(str(artifacts_dir), mtime)


def _render_summary(run: dict) -> None:
    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader("Best checkpoint")
        summary = run.get("checkpoint_summary")
        if summary and summary.get("primary"):
            primary = summary["primary"]
            st.metric(
                label=f"Best {primary.get('metric', 'metric')}",
                value=f"{primary.get('best_score', float('nan')):.4f}",
            )
            best_path = primary.get("best_path", "")
            if best_path:
                st.caption(f"path: `{Path(best_path).name}`")
            additional = summary.get("additional") or []
            if additional:
                df = pd.DataFrame(additional).astype(str)
                st.dataframe(df, hide_index=True, use_container_width=True)
        else:
            st.caption("No checkpoint_summary.json")

    with col_right:
        st.subheader("Build manifest")
        manifest = run.get("build_manifest")
        if manifest:
            outputs = manifest.get("outputs") or {}
            info = {
                "Git branch": manifest.get("git_branch", "?"),
                "Git hash": (manifest.get("git_hash") or "?")[:10],
                "Seed": outputs.get("seed", "?"),
                "Total params": f"{outputs.get('total_params', 0):,}"
                    if outputs.get("total_params") is not None else "?",
                "Trainable params": f"{outputs.get('trainable_params', 0):,}"
                    if outputs.get("trainable_params") is not None else "?",
                "Timestamp": manifest.get("timestamp", "?"),
                "Host": (manifest.get("environment") or {}).get("hostname", "?"),
            }
            st.dataframe(
                pd.DataFrame(
                    [(k, str(v)) for k, v in info.items()],
                    columns=["key", "value"],
                ),
                hide_index=True,
                use_container_width=True,
            )
        else:
            st.caption("No build_manifest.json")


def _render_args(args: dict | None) -> None:
    if not args:
        st.caption("No args.json")
        return
    with st.expander("Hyperparameters", expanded=True):
        cols = st.columns(len(_ARGS_SECTIONS))
        for col, (title, keys) in zip(cols, _ARGS_SECTIONS.items()):
            with col:
                st.markdown(f"**{title}**")
                for k in keys:
                    if k not in args:
                        continue
                    v = args[k]
                    if isinstance(v, (list, dict)):
                        v = str(v) if v else "—"
                    st.markdown(f"- `{k}`: `{v}`")
        with st.expander(f"All args (raw, {len(args)} keys)"):
            st.json(args)


def _render_learning_curves(run: dict) -> None:
    st.subheader("Learning curves")
    bases = discover_metric_bases(run)
    all_bases = sorted(set(bases["train"]) | set(bases["val"]))
    if not all_bases:
        st.warning("No metrics discovered in metrics_history.pt")
        return

    default_idx = all_bases.index("loss") if "loss" in all_bases else 0
    metric_base = st.selectbox("Metric", all_bases, index=default_idx, key="tv_metric")

    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        x_axis_label = st.radio(
            "X-axis", ["Global step", "Epoch"], horizontal=True, key="tv_xaxis",
        )
        x_axis = "step" if x_axis_label == "Global step" else "epoch"
    with c2:
        show_train = st.checkbox("Train", value=True, key="tv_show_train")
        show_val = st.checkbox("Val", value=True, key="tv_show_val")
    with c3:
        log_y = st.checkbox("Log y-axis", value=False, key="tv_logy")
        show_by_rank = st.checkbox("Show by rank", value=False, key="tv_show_rank")

    splits = tuple([s for s, on in (("train", show_train), ("val", show_val)) if on])

    ranks_to_plot: list[int] | None = None
    if show_by_rank:
        if run["available_ranks"]:
            if metric_base != "loss":
                st.caption(
                    "Rank overlay is only supported for `loss` (rank files store "
                    "only `val_loss`). Select `loss` to see per-rank lines."
                )
            elif "val" not in splits:
                st.caption(
                    "Rank overlay lives on top of `val_loss`. Enable the Val "
                    "checkbox to see per-rank lines."
                )
            else:
                ranks_to_plot = st.multiselect(
                    "Ranks",
                    options=run["available_ranks"],
                    default=run["available_ranks"],
                    key="tv_ranks",
                )
        else:
            st.info(
                "No per-rank metrics stored for this run. "
                "Re-run training with `--all_ranks_val_loss True` to produce "
                "`metrics_history.rank*.pt` files."
            )

    if not splits:
        st.warning("Select at least one split (Train or Val) to render a plot.")
        return

    fig = plot_metric(
        run, metric_base,
        x_axis=x_axis,
        splits=splits,
        ranks=ranks_to_plot,
        log_y=log_y,
    )
    st.pyplot(fig)
    plt.close(fig)


def _render_benchmarks(run: dict) -> None:
    bench = run.get("benchmark_history")
    if not bench or not bench.get("epochs"):
        return

    st.subheader("Benchmarks")
    config = bench.get("config") or {}
    if config:
        df = pd.DataFrame(
            [(k, str(v)) for k, v in config.items()],
            columns=["key", "value"],
        )
        st.dataframe(df, hide_index=True, use_container_width=True)

    epochs = bench["epochs"]
    has_per_rank = any(
        isinstance(e.get("peak_memory_allocated_gb_per_rank"), list) and
        e["peak_memory_allocated_gb_per_rank"]
        for e in epochs
    )
    if has_per_rank:
        example = next(
            (e for e in epochs if isinstance(e.get("peak_memory_allocated_gb_per_rank"), list)),
            None,
        )
        n_ranks = len(example["peak_memory_allocated_gb_per_rank"]) if example else 0
    else:
        n_ranks = 0

    tab_throughput, tab_memory = st.tabs(["Throughput", "Memory"])
    with tab_throughput:
        fig = plot_benchmark_history(run, kind="throughput")
        st.pyplot(fig)
        plt.close(fig)
    with tab_memory:
        per_rank_default = has_per_rank and n_ranks <= 16
        per_rank = st.checkbox(
            f"Overlay per-rank memory ({n_ranks} ranks)" if has_per_rank
            else "Overlay per-rank memory (not available)",
            value=per_rank_default,
            disabled=not has_per_rank,
            key="tv_bench_per_rank",
        )
        if per_rank and n_ranks > 16:
            st.caption(
                f"⚠️ {n_ranks} ranks will produce a crowded plot; consider leaving this off."
            )
        fig = plot_benchmark_history(run, kind="memory", per_rank=per_rank)
        st.pyplot(fig)
        plt.close(fig)


def _render_raw_explorer(run: dict) -> None:
    raw = run.get("raw") or {}
    all_keys: list[tuple[str, str]] = []
    for split in ("train", "val"):
        for k in (raw.get(split) or {}).keys():
            if k in ("global_step", "epoch"):
                continue
            all_keys.append((split, k))
    if not all_keys:
        return
    with st.expander("Raw metrics explorer"):
        labels = [f"{split}.{k}" for split, k in all_keys]
        choice = st.selectbox("Raw metric", labels, key="tv_raw_metric")
        split, key = all_keys[labels.index(choice)]
        split_raw = raw.get(split) or {}
        y = np.asarray(split_raw[key])
        step = np.asarray(split_raw.get("global_step"))
        epoch = np.asarray(split_raw.get("epoch"))
        x_axis_label = st.radio(
            "X-axis",
            ["Global step", "Epoch"],
            horizontal=True,
            key="tv_raw_xaxis",
        )
        if x_axis_label == "Global step" and step.size == y.size:
            x, x_label = step, "Global step"
        elif epoch.size == y.size:
            x, x_label = epoch, "Epoch"
        else:
            st.warning(
                f"Index arrays don't align with metric shape {y.shape}; "
                "falling back to array index."
            )
            x, x_label = np.arange(y.size), "Index"
        mask = np.isfinite(x) & np.isfinite(y)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(x[mask], y[mask], color="#444", linewidth=1.5)
        ax.set_xlabel(x_label)
        ax.set_ylabel(key)
        ax.set_title(f"{split}.{key}")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)


def render() -> None:
    st.header("Training Run Viewer")
    st.write(
        "Browse Stage 3 training and finetuning runs. Pick any "
        "`metrics_history.pt` under an `artifacts/` directory."
    )

    artifacts_dir = _pick_run_artifacts()
    if artifacts_dir is None:
        return
    if not (artifacts_dir / "metrics_history.pt").is_file():
        st.error(
            f"`{artifacts_dir}` does not contain a `metrics_history.pt` file."
        )
        return

    try:
        run = _load_with_cache(artifacts_dir)
    except Exception as e:
        st.error(f"Failed to load run artifacts: {e}")
        return

    st.caption(f"Artifacts directory: `{artifacts_dir}`")
    _render_summary(run)
    _render_args(run.get("args"))
    _render_learning_curves(run)
    _render_benchmarks(run)
    _render_raw_explorer(run)


render()

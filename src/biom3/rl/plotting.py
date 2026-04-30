"""Auto-plot helpers for GRPO/GDPO training logs.

At the end of a training run the trainer calls ``plot_train_log`` to
materialize the canonical RL diagnostic figures into the run's output
directory. Failures here never abort training — plotting is wrapped in
a broad except in the trainer.

Conventions
-----------
- Reward panels show **per-replica scatter** (one dot per replica per
  step) plus a mean ± std band. This is what you want to *see* when
  judging an RL fine-tune: the spread within a group at each step
  determines the advantage signal.
- Loss panels split out ``pg`` and ``kl`` separately rather than
  superimposing them on the total — they live on different scales.
- GDPO adds an ELBO panel (``elbo_new`` vs ``elbo_old``) and a
  sequence-level log-ratio panel.
- Composite rewards get a second figure with one row per component.

Robust to logs produced with older trainers that lack
``rewards_per_replica`` — falls back to the per-step mean.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from biom3.backend.device import setup_logger

logger = setup_logger(__name__)


def write_train_log_atomic(log_path: str, log_rows: List[Dict[str, Any]]) -> None:
    """Write ``log_rows`` to ``log_path`` atomically via a tmp file +
    ``os.replace``.

    Used by both trainers after each step's row is appended, so the
    on-disk ``train_log.json`` is always a valid, fully-consistent
    snapshot of the run-so-far. If the job is killed mid-write the
    previous good copy is preserved (``os.replace`` is atomic on
    POSIX). At ~50 KB after 100 steps the per-step rewrite cost is
    negligible relative to a multi-minute-per-step training loop.
    """
    tmp_path = log_path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(log_rows, f, indent=2)
    os.replace(tmp_path, log_path)


def _load_rows(log_path: str) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Return ``(step_rows, meta_row)`` from ``train_log.json``.

    GDPO writes a leading row with ``_meta=True`` (quadrature config);
    GRPO does not. We strip it from the per-step rows and surface it
    separately so panels can annotate the figure title.
    """
    with open(log_path) as f:
        log = json.load(f)
    meta = None
    rows: List[Dict[str, Any]] = []
    for r in log:
        if r.get("_meta"):
            meta = r
        else:
            rows.append(r)
    return rows, meta


def _scatter_with_band(ax, rows, key_per_replica: str, key_mean: str, label: str, color: str):
    """Plot per-replica scatter + mean ± std band against step.

    Uses ``rewards_per_replica``-style fields when available; falls back
    to the per-step mean stored in ``key_mean`` if not.
    """
    steps = [r["step"] for r in rows]
    have_replicas = all(key_per_replica in r and r[key_per_replica] for r in rows)
    if have_replicas:
        xs, ys = [], []
        means, stds = [], []
        for r in rows:
            vals = r[key_per_replica]
            xs.extend([r["step"]] * len(vals))
            ys.extend(vals)
            means.append(float(np.mean(vals)))
            stds.append(float(np.std(vals)))
        means_a = np.asarray(means)
        stds_a = np.asarray(stds)
        ax.scatter(xs, ys, s=8, alpha=0.25, color=color, label=f"{label} (replicas)")
        ax.plot(steps, means, color=color, linewidth=1.6, label=f"{label} (mean)")
        ax.fill_between(
            steps, means_a - stds_a, means_a + stds_a, color=color, alpha=0.15
        )
    else:
        ax.plot(steps, [r[key_mean] for r in rows], color=color, linewidth=1.6,
                label=f"{label} (mean)")
    ax.set_xlabel("step")
    ax.legend(loc="best", fontsize=8)


def plot_train_log(log_path: str, out_dir: str, algo: str = "grpo") -> List[str]:
    """Render diagnostic figures for a finished training run.

    Args:
        log_path: ``train_log.json`` produced by ``grpo_train`` or ``gdpo_train``.
        out_dir: directory to write figures into. Created if missing.
        algo: ``"grpo"`` or ``"gdpo"`` — controls extra panels (GDPO adds
            ELBO + sequence-level log-ratio).

    Returns:
        List of file paths that were written.
    """
    # Local matplotlib import keeps the trainer importable on hosts where
    # the matplotlib backend is broken (e.g. some HPC compute nodes
    # without display libs); plotting itself is best-effort.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows, meta = _load_rows(log_path)
    if not rows:
        logger.warning("plot_train_log: no per-step rows in %s", log_path)
        return []

    os.makedirs(out_dir, exist_ok=True)
    written: List[str] = []
    is_gdpo = algo.lower() == "gdpo"

    # GRPO grew a token-level log-ratio panel after the PAD-vs-MASK fix
    # — older logs predating that change won't have the fields, in which
    # case we keep the original 6-panel layout.
    has_grpo_logratio = (
        not is_gdpo
        and any("log_ratio_tok" in r and "log_ratio_tok_max_abs" in r for r in rows)
    )
    # Diversity panel is added when the log carries the new fields. Older
    # logs without diversity_mean simply skip the panel.
    has_diversity = any("diversity_mean" in r for r in rows)

    # ─── Main diagnostics figure ─────────────────────────────────────────
    if is_gdpo:
        n_panels = 8
    elif has_grpo_logratio:
        n_panels = 7
    else:
        n_panels = 6
    if has_diversity:
        n_panels += 1
    n_cols = 2
    n_rows = (n_panels + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3.0 * n_rows), squeeze=False)
    flat_axes = [a for row in axes for a in row]

    steps = [r["step"] for r in rows]

    # 1. Reward (mean ± std band + per-replica scatter)
    ax = flat_axes[0]
    _scatter_with_band(ax, rows, "rewards_per_replica", "reward", "reward", "C0")
    # Running average from per-step mean
    if all("reward_avg" in r for r in rows):
        ax.plot(steps, [r["reward_avg"] for r in rows],
                color="C3", linestyle="--", linewidth=1.2, label="running avg")
        ax.legend(loc="best", fontsize=8)
    ax.set_ylabel("reward")
    ax.set_title("Reward")

    # 2. Total loss + components
    ax = flat_axes[1]
    ax.plot(steps, [r["loss"] for r in rows], label="loss", color="C0", linewidth=1.4)
    ax.plot(steps, [r["pg"] for r in rows], label="pg", color="C1", linewidth=1.0, alpha=0.85)
    ax.plot(steps, [r["kl"] for r in rows], label="kl", color="C2", linewidth=1.0, alpha=0.85)
    ax.axhline(0.0, color="k", linewidth=0.5, alpha=0.5)
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_title("Loss components")
    ax.legend(loc="best", fontsize=8)

    # 3. PPO clip fraction
    ax = flat_axes[2]
    ax.plot(steps, [r["clip_frac"] for r in rows], color="C4", linewidth=1.2)
    ax.set_xlabel("step")
    ax.set_ylabel("clip frac")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("PPO clip fraction")

    # 4. Weight drift (per-step on left, cumulative on right)
    ax = flat_axes[3]
    ax.plot(steps, [r["dw_step"] for r in rows], color="C5", linewidth=1.0, label="dw_step")
    ax.set_xlabel("step")
    ax.set_ylabel("dw_step", color="C5")
    ax.tick_params(axis="y", labelcolor="C5")
    ax2 = ax.twinx()
    ax2.plot(steps, [r["dw_total"] for r in rows], color="C6", linewidth=1.4, label="dw_total")
    ax2.set_ylabel("dw_total", color="C6")
    ax2.tick_params(axis="y", labelcolor="C6")
    ax.set_title("Weight drift (Frobenius)")

    # 5. Sequence length
    ax = flat_axes[4]
    ax.plot(steps, [r["avg_len"] for r in rows], color="C7", linewidth=1.4)
    ax.set_xlabel("step")
    ax.set_ylabel("avg seq length")
    ax.set_title("Average sequence length")

    # 6. Reward histogram (per-replica, all steps pooled)
    ax = flat_axes[5]
    have_replicas = all("rewards_per_replica" in r and r["rewards_per_replica"] for r in rows)
    if have_replicas:
        all_r = [v for r in rows for v in r["rewards_per_replica"]]
        ax.hist(all_r, bins=30, color="C0", alpha=0.7, edgecolor="k")
        ax.set_xlabel("reward")
        ax.set_ylabel("count")
        ax.set_title("Reward distribution (all replicas, all steps)")
    else:
        ax.text(0.5, 0.5, "no per-replica rewards in log",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Reward distribution")

    if is_gdpo:
        # 7. ELBOs
        ax = flat_axes[6]
        if all("elbo_new" in r for r in rows):
            ax.plot(steps, [r["elbo_new"] for r in rows], color="C0", label="elbo_new", linewidth=1.4)
        if all("elbo_old" in r for r in rows):
            ax.plot(steps, [r["elbo_old"] for r in rows], color="C3", label="elbo_old",
                    linestyle="--", linewidth=1.2)
        ax.set_xlabel("step")
        ax.set_ylabel("ELBO (per-sequence)")
        ax.set_title("Sequence-level ELBO")
        ax.legend(loc="best", fontsize=8)

        # 8. Sequence-level log-ratio
        ax = flat_axes[7]
        if all("log_ratio_seq" in r for r in rows):
            ax.plot(steps, [r["log_ratio_seq"] for r in rows], color="C0", linewidth=1.4,
                    label="log_ratio_seq (mean)")
        if all("log_ratio_seq_max_abs" in r for r in rows):
            ax.plot(steps, [r["log_ratio_seq_max_abs"] for r in rows], color="C3",
                    linewidth=1.0, alpha=0.85, label="|log_ratio_seq| (max)")
        ax.axhline(0.0, color="k", linewidth=0.5, alpha=0.5)
        ax.set_xlabel("step")
        ax.set_ylabel("log r_g")
        ax.set_title("Sequence-level log importance ratio")
        ax.legend(loc="best", fontsize=8)
    elif has_grpo_logratio:
        # 7. Token-level log-ratio (GRPO analogue of GDPO's panel 8).
        ax = flat_axes[6]
        ax.plot(steps, [r["log_ratio_tok"] for r in rows], color="C0", linewidth=1.4,
                label="log_ratio_tok (mean)")
        ax.plot(steps, [r["log_ratio_tok_max_abs"] for r in rows], color="C3",
                linewidth=1.0, alpha=0.85, label="|log_ratio_tok| (max)")
        ax.axhline(0.0, color="k", linewidth=0.5, alpha=0.5)
        ax.set_xlabel("step")
        ax.set_ylabel("log r_tok")
        ax.set_title("Token-level log importance ratio")
        ax.legend(loc="best", fontsize=8)

    if has_diversity:
        # Last panel: within-group diversity. Surfacing the v01
        # mode-collapse failure mode at a glance.
        div_idx = n_panels - 1
        ax = flat_axes[div_idx]
        _scatter_with_band(
            ax, rows, "per_replica_diversity", "diversity_mean", "diversity", "C8",
        )
        # Worst-pair identity on a twin axis — when this asymptotes near
        # 1.0 the group has collapsed even if mean diversity looks OK.
        if any("diversity_min_pair" in r for r in rows):
            ax2 = ax.twinx()
            ax2.plot(
                steps,
                [r.get("diversity_min_pair", float("nan")) for r in rows],
                color="C3", linestyle="--", linewidth=1.0, alpha=0.85,
                label="worst-pair identity",
            )
            ax2.set_ylabel("worst-pair identity", color="C3")
            ax2.tick_params(axis="y", labelcolor="C3")
            ax2.set_ylim(-0.02, 1.02)
        ax.set_ylabel("1 - mean identity (per replica)")
        ax.set_ylim(-0.02, 1.02)
        ax.set_title("Within-group sequence diversity")

    # Hide any unused axes (only if n_panels < n_rows*n_cols)
    for ax in flat_axes[n_panels:]:
        ax.axis("off")

    title_bits = [algo.upper(), os.path.basename(os.path.dirname(log_path))]
    if meta is not None:
        N = meta.get("n_quadrature")
        kl = meta.get("kl_estimator")
        if N is not None:
            title_bits.append(f"N={N}")
        if kl is not None:
            title_bits.append(f"kl={kl}")
    fig.suptitle(" | ".join(title_bits), fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    main_path = os.path.join(out_dir, "train_diagnostics.png")
    fig.savefig(main_path, dpi=150)
    plt.close(fig)
    written.append(main_path)
    logger.info("plot_train_log: wrote %s", main_path)

    # ─── Composite reward components (separate figure) ──────────────────
    has_components = any("components_per_replica" in r and r["components_per_replica"] for r in rows)
    if has_components:
        names: List[str] = []
        for r in rows:
            comps = r.get("components_per_replica") or {}
            for n in comps:
                if n not in names:
                    names.append(n)
        n_comp = len(names)
        n_cols_c = min(2, n_comp)
        n_rows_c = (n_comp + n_cols_c - 1) // n_cols_c
        fig_c, axes_c = plt.subplots(
            n_rows_c, n_cols_c, figsize=(6.0 * n_cols_c, 3.0 * n_rows_c), squeeze=False,
        )
        flat_c = [a for row in axes_c for a in row]
        for i, name in enumerate(names):
            ax = flat_c[i]
            xs, ys, means, stds, xs_steps = [], [], [], [], []
            for r in rows:
                comps = r.get("components_per_replica") or {}
                if name not in comps or not comps[name]:
                    continue
                vals = comps[name]
                xs_steps.append(r["step"])
                xs.extend([r["step"]] * len(vals))
                ys.extend(vals)
                means.append(float(np.mean(vals)))
                stds.append(float(np.std(vals)))
            ax.scatter(xs, ys, s=8, alpha=0.25, color=f"C{i % 10}")
            ax.plot(xs_steps, means, color=f"C{i % 10}", linewidth=1.6, label="mean")
            means_a, stds_a = np.asarray(means), np.asarray(stds)
            ax.fill_between(
                xs_steps, means_a - stds_a, means_a + stds_a,
                color=f"C{i % 10}", alpha=0.15,
            )
            ax.set_xlabel("step")
            ax.set_ylabel(name)
            ax.set_title(f"Reward component: {name}")
            ax.legend(loc="best", fontsize=8)
        for ax in flat_c[n_comp:]:
            ax.axis("off")
        fig_c.tight_layout()
        comp_path = os.path.join(out_dir, "train_reward_components.png")
        fig_c.savefig(comp_path, dpi=150)
        plt.close(fig_c)
        written.append(comp_path)
        logger.info("plot_train_log: wrote %s", comp_path)

    return written

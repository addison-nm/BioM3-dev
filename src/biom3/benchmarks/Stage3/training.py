"""Training-loop benchmark for Stage 3 (ProteoScribe).

Sweeps a config-driven grid of training hyperparameters and launches
``biom3_pretrain_stage3`` as a subprocess for each cell. The existing
``TrainingBenchmarkCallback`` emits ``benchmark_history.json`` (per-epoch)
and ``benchmark_steps.jsonl`` (per-step) into each cell's artifacts
directory; this driver reads those back and aggregates into a single
``results.json`` / ``results.npz``.

Outputs land in ``<output_root>/<UTC timestamp>/``:

- ``config.json``        — input config, verbatim
- ``env.json``           — arch_id + device/host info
- ``results.json``       — list of per-cell records
- ``results.npz``        — dense arrays keyed by sweep axes + metrics
- ``run.log``            — driver log (subprocess stdout/stderr is in per-cell dirs)
- ``cells/<cell_id>/``   — per-cell outputs (artifacts, logs, checkpoints)

CLI entry point: ``biom3_benchmark_stage3_training --config <path>``.
See ``configs/benchmark/stage3_training_example.json`` for the schema.
"""

import argparse
import copy
import itertools
import json
import logging
import os
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np

from biom3.backend.device import get_device_info
from biom3.core.helpers import load_json_config


SWEEP_KEYS = (
    "batch_size",
    "acc_grad_batches",
    "precision",
    "gpu_devices",
    "num_nodes",
    "num_workers",
)


def parse_arguments(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True,
                        help="Path to training-benchmark config JSON")
    return parser.parse_args(argv)


def _setup_logger(log_path):
    lg = logging.getLogger("biom3.benchmarks.stage3_training")
    lg.setLevel(logging.INFO)
    lg.propagate = False
    fh = logging.FileHandler(log_path)
    sh = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                            datefmt="%H:%M:%S")
    fh.setFormatter(fmt)
    sh.setFormatter(fmt)
    lg.handlers = [fh, sh]
    return lg


def _validate_config(cfg):
    for key in ("base_training_config_path", "sweep", "output_root"):
        if key not in cfg:
            raise ValueError(f"benchmark config missing required key: {key}")
    sweep = cfg["sweep"]
    for k, values in sweep.items():
        if k not in SWEEP_KEYS:
            raise ValueError(
                f"unknown sweep axis '{k}'; allowed: {', '.join(SWEEP_KEYS)}"
            )
        if not isinstance(values, list) or not values:
            raise ValueError(f"sweep.{k} must be a non-empty list")
    present = [k for k in SWEEP_KEYS if k in sweep]
    if not present:
        raise ValueError(
            "sweep must specify at least one of: " + ", ".join(SWEEP_KEYS)
        )


def _enumerate_cells(sweep):
    """Yield cell dicts over the cartesian product of provided sweep axes."""
    axes = [k for k in SWEEP_KEYS if k in sweep]
    values = [sweep[k] for k in axes]
    for combo in itertools.product(*values):
        yield dict(zip(axes, combo))


def _cell_id(cell):
    parts = []
    for k in SWEEP_KEYS:
        if k in cell:
            v = cell[k]
            parts.append(f"{k[:2]}{str(v).replace('-', '').replace('.', '')}")
    return "_".join(parts) or "cell0"


def _build_training_cmd(cfg, cell, cell_output_root, run_id):
    """Assemble the biom3_pretrain_stage3 command for one sweep cell."""
    cmd = [
        "biom3_pretrain_stage3",
        "--config_path", cfg["base_training_config_path"],
        "--output_root", cell_output_root,
        "--run_id", run_id,
        "--save_benchmark", "True",
        "--benchmark_per_step", str(cfg.get("benchmark_per_step", True)),
        "--benchmark_skip_first_epoch",
        str(cfg.get("benchmark_skip_first_epoch", True)),
    ]

    for k, v in cell.items():
        cmd.extend([f"--{k}", str(v)])

    for k, v in cfg.get("training_extra_args", {}).items():
        cmd.extend([f"--{k}", str(v)])

    return cmd


def _load_cell_metrics(cell_output_root, run_id, cfg):
    """Read benchmark_history.json + benchmark_steps.jsonl from a cell run."""
    runs_folder = cfg.get("training_extra_args", {}).get("runs_folder", "runs")
    artifacts_dir = os.path.join(
        cell_output_root, runs_folder, run_id, "artifacts"
    )
    history_path = os.path.join(artifacts_dir, "benchmark_history.json")
    steps_path = os.path.join(artifacts_dir, "benchmark_steps.jsonl")

    history = None
    if os.path.exists(history_path):
        with open(history_path, "r") as fh:
            history = json.load(fh)

    step_times = []
    if os.path.exists(steps_path):
        with open(steps_path, "r") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "step_wall_time_sec" in rec:
                    step_times.append(rec["step_wall_time_sec"])

    return history, step_times, artifacts_dir


def _summarize(history, step_times, skip_first_epoch):
    """Compute scalar summary metrics from one cell's output."""
    summary = {
        "epochs_observed": 0,
        "mean_epoch_wall_time_sec": None,
        "mean_samples_per_sec": None,
        "mean_steps_per_sec": None,
        "peak_memory_allocated_gb": None,
        "peak_memory_reserved_gb": None,
        "steps_observed": len(step_times),
        "mean_step_wall_time_sec": None,
        "median_step_wall_time_sec": None,
        "p95_step_wall_time_sec": None,
    }
    if history is not None:
        epochs = history.get("epochs", [])
        if skip_first_epoch and len(epochs) > 1:
            epochs = epochs[1:]
        summary["epochs_observed"] = len(epochs)
        if epochs:
            def _mean(key):
                values = [r[key] for r in epochs if r.get(key) is not None]
                return float(np.mean(values)) if values else None

            def _max(key):
                values = [r[key] for r in epochs if r.get(key) is not None]
                return float(np.max(values)) if values else None

            summary["mean_epoch_wall_time_sec"] = _mean("epoch_wall_time_sec")
            summary["mean_samples_per_sec"] = _mean("samples_per_sec")
            summary["mean_steps_per_sec"] = _mean("steps_per_sec")
            summary["peak_memory_allocated_gb"] = _max("peak_memory_allocated_gb")
            summary["peak_memory_reserved_gb"] = _max("peak_memory_reserved_gb")

    if step_times:
        # Drop the first step to exclude one-off warmup (kernel JIT, alloc
        # first-touch) when there are enough samples to afford it.
        trimmed = step_times[1:] if len(step_times) > 1 else step_times
        arr = np.asarray(trimmed, dtype=float)
        summary["mean_step_wall_time_sec"] = float(arr.mean())
        summary["median_step_wall_time_sec"] = float(np.median(arr))
        summary["p95_step_wall_time_sec"] = float(np.quantile(arr, 0.95))

    return summary


def _records_to_npz(records, axes_order, metric_keys, out_path):
    arrays = {}
    for ax in axes_order:
        arrays[ax] = np.array([r.get(ax) for r in records])
    for m in metric_keys:
        arrays[m] = np.array(
            [(r[m] if r.get(m) is not None else np.nan) for r in records],
            dtype=float,
        )
    arrays["_axes_order"] = np.array(axes_order, dtype=object)
    np.savez(out_path, **arrays)


def main(args):
    cfg = load_json_config(args.config)
    _validate_config(cfg)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = os.path.join(cfg["output_root"], timestamp)
    os.makedirs(run_dir, exist_ok=True)
    log_path = os.path.join(run_dir, "run.log")
    logger = _setup_logger(log_path)

    logger.info("Training-benchmark run dir: %s", run_dir)
    logger.info("Base training config:      %s", cfg["base_training_config_path"])

    with open(os.path.join(run_dir, "config.json"), "w") as fh:
        json.dump(cfg, fh, indent=2)

    env = {
        "benchmark_type": "stage3_training",
        "arch_id": cfg.get("arch_id"),
        "timestamp_utc": timestamp,
        **get_device_info(),
    }
    with open(os.path.join(run_dir, "env.json"), "w") as fh:
        json.dump(env, fh, indent=2)
    logger.info("Device: %s (%s)", env.get("device_name"), env.get("backend"))

    cells_root = os.path.join(run_dir, "cells")
    os.makedirs(cells_root, exist_ok=True)

    skip_first_epoch = cfg.get("benchmark_skip_first_epoch", True)
    records = []
    axes_order = [k for k in SWEEP_KEYS if k in cfg["sweep"]]
    metric_keys = [
        "epochs_observed",
        "mean_epoch_wall_time_sec",
        "mean_samples_per_sec",
        "mean_steps_per_sec",
        "peak_memory_allocated_gb",
        "peak_memory_reserved_gb",
        "steps_observed",
        "mean_step_wall_time_sec",
        "median_step_wall_time_sec",
        "p95_step_wall_time_sec",
    ]

    for cell in _enumerate_cells(cfg["sweep"]):
        cell_id = _cell_id(cell)
        cell_output_root = os.path.join(cells_root, cell_id)
        os.makedirs(cell_output_root, exist_ok=True)
        run_id = f"bench_{cell_id}"

        cmd = _build_training_cmd(cfg, cell, cell_output_root, run_id)
        cell_log = os.path.join(cell_output_root, "subprocess.log")
        logger.info("Running cell %s: %s", cell_id, " ".join(cmd))

        env_vars = {**os.environ, "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "true"}
        status = "ok"
        returncode = 0
        try:
            with open(cell_log, "w") as lfh:
                proc = subprocess.run(
                    cmd, stdout=lfh, stderr=subprocess.STDOUT,
                    env=env_vars, check=False,
                )
            returncode = proc.returncode
            if returncode != 0:
                status = f"subprocess_failed (rc={returncode})"
                logger.warning(
                    "Cell %s failed (rc=%d); partial metrics may be available",
                    cell_id, returncode,
                )
        except Exception as exc:  # noqa: BLE001
            status = f"exception: {exc}"
            logger.exception("Cell %s raised an exception", cell_id)

        history, step_times, artifacts_dir = _load_cell_metrics(
            cell_output_root, run_id, cfg,
        )
        summary = _summarize(history, step_times, skip_first_epoch)

        record = {
            **cell,
            "cell_id": cell_id,
            "status": status,
            "returncode": returncode,
            "artifacts_dir": artifacts_dir,
            **summary,
        }
        records.append(record)

        logger.info(
            "Cell %s done: status=%s epochs=%d steps=%d "
            "mean_epoch_time=%.3fs mean_step_time=%.4fs peak_alloc=%s GB",
            cell_id, status,
            summary["epochs_observed"], summary["steps_observed"],
            summary["mean_epoch_wall_time_sec"] or float("nan"),
            summary["mean_step_wall_time_sec"] or float("nan"),
            summary["peak_memory_allocated_gb"],
        )

    with open(os.path.join(run_dir, "results.json"), "w") as fh:
        json.dump(records, fh, indent=2)
    _records_to_npz(
        records, axes_order, metric_keys,
        os.path.join(run_dir, "results.npz"),
    )
    logger.info("Wrote %d records to %s", len(records), run_dir)


if __name__ == "__main__":
    main(parse_arguments(sys.argv[1:]))

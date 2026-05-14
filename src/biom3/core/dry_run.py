"""Dry-run support for biom3_train_stage{1,2,3}.

Renders a side-effect-free pre-flight report covering effective args
(with per-source provenance), output paths the run would create,
distributed/data math, and an a-priori memory estimate.

Used by each stage's ``main()`` via :func:`run_dry_run`. The data and
model probes are stage-specific callables passed in by the caller.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

import torch

from biom3.backend.device import setup_logger
from biom3.core.helpers import load_json_config
from biom3.core.run_utils import backup_if_exists, collect_training_env

logger = setup_logger(__name__)


_LOGS_SUBDIR = "logs"
_ARTIFACTS_SUBDIR = "artifacts"
_BYTES_PER_GB = 1024 ** 3


@dataclass
class DryRunResult:
    args_table: list[tuple[str, Any, str]] = field(default_factory=list)
    paths: dict[str, str] = field(default_factory=dict)
    distributed: dict[str, Any] = field(default_factory=dict)
    memory: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {
            "stage": getattr(self, "_stage", None),
            "timestamp": datetime.now().isoformat(),
            "effective_config": [
                {"key": k, "value": _jsonable(v), "source": s}
                for (k, v, s) in self.args_table
            ],
            "output_paths": self.paths,
            "distributed": self.distributed,
            "memory_estimate": self.memory,
            "notes": self.notes,
        }


def _jsonable(v):
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    return str(v)


# ----- Provenance detection -----

def detect_cli_keys(dest_names, argv: list[str]) -> set[str]:
    """Return the set of dest names that appear in ``argv`` as ``--<name>``
    or ``--<name>=...``.

    Operates on plain strings — no parser introspection, no namespace
    stashing. Assumes the conventional argparse mapping where dest
    ``foo_bar`` corresponds to ``--foo_bar``.
    """
    cli_keys: set[str] = set()
    for dest in dest_names:
        flag = f"--{dest}"
        if flag in argv or any(t.startswith(flag + "=") for t in argv):
            cli_keys.add(dest)
    return cli_keys


def build_args_table(
    args: argparse.Namespace,
    argv: list[str],
    config_path: str | None,
) -> list[tuple[str, Any, str]]:
    """Build a sorted list of (key, value, source) tuples.

    ``source`` is one of:
      - ``"CLI"``
      - ``"JSON: <abs path>"``
      - ``"default"``
    """
    dest_names = [k for k in vars(args) if not k.startswith("_")]
    cli_keys = detect_cli_keys(dest_names, argv)
    if config_path:
        try:
            _, json_provenance = load_json_config(
                config_path, track_provenance=True,
            )
        except Exception as exc:
            json_provenance = {}
            logger.warning("Could not load JSON for provenance: %s", exc)
    else:
        json_provenance = {}

    rows = []
    for k, v in sorted(vars(args).items()):
        if k.startswith("_"):
            continue
        if k in cli_keys:
            src = "CLI"
        elif k in json_provenance:
            src = f"JSON: {json_provenance[k]}"
        else:
            src = "default"
        rows.append((k, v, src))
    return rows


# ----- Output paths -----

def render_paths(args, stage: str, dry_run_report_path: str | None) -> dict[str, str]:
    output_root = getattr(args, "output_root", None)
    runs_folder = getattr(args, "runs_folder", "runs")
    checkpoints_folder = getattr(args, "checkpoints_folder", "checkpoints")
    run_id = getattr(args, "run_id", None)
    if not output_root or not run_id:
        return {}
    run_dir = os.path.join(output_root, runs_folder, run_id)
    logs_dir = os.path.join(run_dir, _LOGS_SUBDIR)
    artifacts_dir = os.path.join(run_dir, _ARTIFACTS_SUBDIR)
    checkpoint_dir = os.path.join(output_root, checkpoints_folder, run_id)
    paths = {
        "run_dir": os.path.abspath(run_dir),
        "logs_dir": os.path.abspath(logs_dir),
        "artifacts_dir": os.path.abspath(artifacts_dir),
        "checkpoint_dir": os.path.abspath(checkpoint_dir),
        "args_json": os.path.abspath(os.path.join(artifacts_dir, "args.json")),
        "build_manifest": os.path.abspath(
            os.path.join(artifacts_dir, "build_manifest.json")
        ),
        "run_log": os.path.abspath(os.path.join(artifacts_dir, "run.log")),
    }
    if dry_run_report_path is not None:
        paths["dry_run_report"] = os.path.abspath(dry_run_report_path)
    return paths


# ----- Distributed summary -----

def render_distributed_summary(
    args,
    train_dataset_len: int | None,
    val_dataset_len: int | None,
) -> dict[str, Any]:
    devices_per_node = int(getattr(args, "devices_per_node", 1) or 1)
    num_nodes = int(getattr(args, "num_nodes", 1) or 1)
    world_size = devices_per_node * num_nodes
    micro_bs = int(getattr(args, "batch_size", 1) or 1)
    accum = int(getattr(args, "acc_grad_batches", 1) or 1)
    effective_bs = micro_bs * world_size * accum

    summary: dict[str, Any] = {
        "num_nodes": num_nodes,
        "devices_per_node": devices_per_node,
        "world_size": world_size,
        "micro_batch_size": micro_bs,
        "acc_grad_batches": accum,
        "effective_batch_size": effective_bs,
        "train_dataset_len": train_dataset_len,
        "val_dataset_len": val_dataset_len,
    }

    if train_dataset_len is not None and world_size > 0:
        # DistributedSampler(drop_last=True): floor(N / world_size)
        per_rank_batches = (train_dataset_len // world_size) // max(micro_bs, 1)
        steps_per_epoch = per_rank_batches // max(accum, 1)
        summary["batches_per_epoch_per_rank"] = per_rank_batches
        summary["steps_per_epoch"] = steps_per_epoch

    for key in (
        "training_strategy", "epochs", "max_steps", "val_check_interval",
        "limit_val_batches", "limit_train_batches",
    ):
        if hasattr(args, key):
            summary[key] = getattr(args, key)
    return summary


# ----- Strategy inference -----

def infer_strategy(args, stage: str) -> str:
    """Mirror each stage's strategy-selection logic.

    Returns one of: ``'ddp'``, ``'single_device'``, ``'deepspeed_zero2'``.
    """
    if stage == "stage3":
        return getattr(args, "distributed_strategy", None) or "deepspeed_zero2"
    device = getattr(args, "device", "cpu")
    devices_per_node = int(getattr(args, "devices_per_node", 1) or 1)
    num_nodes = int(getattr(args, "num_nodes", 1) or 1)
    if device == "cuda" and (devices_per_node > 1 or num_nodes > 1):
        return "ddp"
    if device == "xpu" and devices_per_node == 1 and num_nodes == 1:
        return "single_device"
    return "single_device"


# ----- Memory estimate -----

def batch_footprint_bytes(batch) -> int:
    """Sum tensor.element_size() * tensor.numel() recursively over a batch."""
    if isinstance(batch, torch.Tensor):
        return batch.element_size() * batch.numel()
    if isinstance(batch, dict):
        return sum(batch_footprint_bytes(v) for v in batch.values())
    if isinstance(batch, (list, tuple)):
        return sum(batch_footprint_bytes(v) for v in batch)
    return 0


def render_memory_estimate(
    model: torch.nn.Module | None,
    sample_batch,
    strategy: str,
    devices_per_node: int,
    num_nodes: int,
    *,
    param_bytes: int = 4,
    grad_bytes: int = 4,
    optimizer_bytes_per_param: int = 8,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "strategy": strategy,
        "param_dtype_bytes": param_bytes,
        "optimizer_bytes_per_param": optimizer_bytes_per_param,
        "caveat": "Activation memory is NOT included; requires a forward pass.",
    }
    if model is None:
        out["error"] = "model probe failed; could not estimate"
        return out

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    out["total_params"] = int(total)
    out["trainable_params"] = int(trainable)

    world_size = max(devices_per_node * num_nodes, 1)
    if strategy == "deepspeed_zero2":
        try:
            from deepspeed.runtime.zero.stage_1_and_2 import (
                estimate_zero2_model_states_mem_needs_all_cold,
            )
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                estimate_zero2_model_states_mem_needs_all_cold(
                    total_params=int(total),
                    num_gpus_per_node=int(devices_per_node),
                    num_nodes=int(num_nodes),
                )
            out["deepspeed_breakdown"] = buf.getvalue().rstrip()
        except Exception as exc:
            out["deepspeed_estimator_error"] = str(exc)
        # Manual breakdown: ZeRO-2 shards optimizer states and gradients
        # across ranks; params stay replicated.
        out["per_rank_param_gb"] = (total * param_bytes) / _BYTES_PER_GB
        out["per_rank_grad_gb"] = (
            (total * grad_bytes) / world_size / _BYTES_PER_GB
        )
        out["per_rank_optimizer_gb"] = (
            (trainable * optimizer_bytes_per_param) / world_size / _BYTES_PER_GB
        )
    else:
        # Full replication: every rank holds params + grads + optimizer state.
        out["per_rank_param_gb"] = (total * param_bytes) / _BYTES_PER_GB
        out["per_rank_grad_gb"] = (total * grad_bytes) / _BYTES_PER_GB
        out["per_rank_optimizer_gb"] = (
            (trainable * optimizer_bytes_per_param) / _BYTES_PER_GB
        )
    out["per_rank_static_total_gb"] = (
        out["per_rank_param_gb"]
        + out["per_rank_grad_gb"]
        + out["per_rank_optimizer_gb"]
    )

    if sample_batch is not None:
        bs_bytes = batch_footprint_bytes(sample_batch)
        out["per_minibatch_input_gb"] = bs_bytes / _BYTES_PER_GB
        out["per_minibatch_input_bytes"] = int(bs_bytes)
    return out


# ----- Output target resolution -----

def resolve_dry_run_output(args, paths: dict[str, str]) -> str | None:
    """Map ``args.dry_run_output`` (False / True / str path) to a target.

    - ``False`` (or unset): returns None (stdout-only).
    - ``True``: returns ``<artifacts_dir>/dry_run_report.json``.
    - any other string: treated as a filepath (absolute or relative).
    """
    val = getattr(args, "dry_run_output", False)
    if val is False or val is None:
        return None
    if val is True:
        artifacts_dir = paths.get("artifacts_dir")
        if not artifacts_dir:
            return None
        return os.path.join(artifacts_dir, "dry_run_report.json")
    return os.path.abspath(str(val))


def write_report(report: dict[str, Any], output_path: str | None) -> None:
    if output_path is None:
        return
    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    backup_if_exists(output_path)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, default=str)


# ----- Stdout rendering -----

def _fmt_value(v: Any) -> str:
    if v is None:
        return "None"
    if isinstance(v, float):
        return f"{v:.6g}"
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_fmt_value(x) for x in v) + "]"
    return str(v)


def print_report(result: DryRunResult, stage: str, output_path: str | None) -> None:
    bar = "=" * 72
    pretty_stage = stage.replace("stage", "Stage ").strip().title() \
        if stage.lower().startswith("stage") and stage[5:].isdigit() else stage
    print(bar)
    print(f"BioM3 {pretty_stage} — DRY RUN (no training will be executed)")
    print(bar)
    print()

    rows = result.args_table
    print(f"[Effective configuration]  {len(rows)} keys total")
    if rows:
        key_w = min(max(len(k) for k, _, _ in rows), 32)
        val_w = 18
        print(f"  {'KEY':<{key_w}}  {'VALUE':<{val_w}}  SOURCE")
        print(f"  {'-' * key_w}  {'-' * val_w}  {'-' * 32}")
        for k, v, s in rows:
            vs = _fmt_value(v)
            if len(vs) > val_w:
                vs = vs[: val_w - 1] + "…"
            ks = k if len(k) <= key_w else k[: key_w - 1] + "…"
            print(f"  {ks:<{key_w}}  {vs:<{val_w}}  {s}")
    print()

    if result.paths:
        print("[Output paths the run would create]")
        kw = max(len(k) for k in result.paths)
        for k, v in result.paths.items():
            print(f"  {k:<{kw}}  {v}")
        print()

    if result.distributed:
        print("[Distributed / batch math]")
        kw = max(len(k) for k in result.distributed)
        for k, v in result.distributed.items():
            print(f"  {k:<{kw}}  {_fmt_value(v)}")
        print()

    if result.memory:
        strat = result.memory.get("strategy", "?")
        print(f"[Memory estimate]  trainer_strategy={strat}")
        for k, v in result.memory.items():
            if k == "deepspeed_breakdown":
                print(f"  {k}:")
                for line in str(v).splitlines():
                    print(f"    {line}")
                continue
            if isinstance(v, float) and k.endswith("_gb"):
                print(f"  {k:<32}  {v:.3f} GB")
            elif isinstance(v, int) and k.endswith("_bytes"):
                print(f"  {k:<32}  {v:,}")
            else:
                print(f"  {k:<32}  {_fmt_value(v)}")
        print()

    if result.notes:
        print("[Notes]")
        for n in result.notes:
            print(f"  - {n}")
        print()

    if output_path is None:
        print("(JSON report not written; pass --dry_run_output=True to write to "
              "artifacts/, or --dry_run_output=./path.json for a custom path.)")
    else:
        print(f"JSON report written to {output_path}")


# ----- Top-level entrypoint -----

def run_dry_run(
    args: argparse.Namespace,
    *,
    stage: str,
    argv: list[str] | None = None,
    config_path: str | None = None,
    dataset_probe: Callable[[], tuple[int | None, int | None, Any]] | None = None,
    model_probe: Callable[[], torch.nn.Module] | None = None,
) -> int:
    """Execute the dry-run pipeline. Returns an exit code (0).

    ``argv`` defaults to ``sys.argv[1:]``. ``dataset_probe`` should
    return ``(train_len, val_len, sample_batch)``. ``model_probe``
    should return an ``nn.Module`` constructed on CPU. Either probe
    may be None (the corresponding sections degrade gracefully).
    """
    if argv is None:
        # Prefer argv stashed on args by retrieve_all_args; fall back to
        # sys.argv when invoked via the binary entrypoint without stashing.
        argv = getattr(args, "_argv", None)
        if argv is None:
            argv = sys.argv[1:]

    result = DryRunResult()
    result._stage = stage  # type: ignore[attr-defined]

    if config_path is None:
        config_path = getattr(args, "config_path", None)

    try:
        result.args_table = build_args_table(args, argv, config_path)
    except Exception as exc:
        result.notes.append(f"args provenance failed: {exc}")

    output_target_pre = resolve_dry_run_output(args, render_paths(args, stage, None))
    result.paths = render_paths(args, stage, output_target_pre)

    train_len = val_len = None
    sample_batch = None
    if dataset_probe is not None:
        try:
            train_len, val_len, sample_batch = dataset_probe()
        except Exception as exc:
            result.notes.append(f"dataset probe failed: {exc}")

    result.distributed = render_distributed_summary(args, train_len, val_len)

    model = None
    if model_probe is not None:
        try:
            model = model_probe()
        except Exception as exc:
            result.notes.append(f"model probe failed: {exc}")

    strategy = infer_strategy(args, stage)
    result.memory = render_memory_estimate(
        model,
        sample_batch,
        strategy,
        devices_per_node=int(getattr(args, "devices_per_node", 1) or 1),
        num_nodes=int(getattr(args, "num_nodes", 1) or 1),
    )

    output_path = resolve_dry_run_output(args, result.paths)

    report = result.to_json()
    report["environment"] = collect_training_env()
    report["argv"] = list(argv)

    print_report(result, stage=stage, output_path=output_path)

    if output_path is not None:
        try:
            write_report(report, output_path)
        except Exception as exc:
            print(f"[warn] could not write JSON report to {output_path}: {exc}",
                  file=sys.stderr)
    return 0


def coerce_dry_run_output(value):
    """Normalize the --dry_run_output flag value.

    Accepts: bool, ``'True'`` / ``'False'`` / ``'None'`` (case-insensitive),
    or any other string (treated as a filepath).
    """
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, str):
        low = value.strip().lower()
        if low in ("true", "1", "yes"):
            return True
        if low in ("false", "0", "no", "", "none"):
            return False
        return value
    return False

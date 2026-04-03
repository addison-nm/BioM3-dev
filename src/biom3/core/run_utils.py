"""Shared utilities for entrypoint logging and reproducibility manifests."""

import json
import logging
import os
import subprocess
import sys
from datetime import datetime, timedelta


def get_biom3_version():
    try:
        from importlib.metadata import version
        return version("biom3")
    except Exception:
        return "unknown"


def get_git_hash():
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def get_git_dirty():
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, timeout=5,
        )
        return bool(result.stdout.strip()) if result.returncode == 0 else None
    except Exception:
        return None


def get_git_branch():
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def get_git_remote():
    try:
        result = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            capture_output=True, text=True, timeout=5,
        )
        url = result.stdout.strip() if result.returncode == 0 else "unknown"
        # Strip embedded credentials (https://user:token@host/... -> https://host/...)
        import re
        url = re.sub(r"(https?://)([^@]+@)", r"\1", url)
        return url
    except Exception:
        return "unknown"


def get_biom3_build():
    """Return a PEP 440-style local version string, e.g. '0.0.1+g80b4453.dirty'."""
    version = get_biom3_version()
    git_hash = get_git_hash()
    dirty = get_git_dirty()
    parts = [version]
    if git_hash != "unknown":
        parts.append(f"+g{git_hash}")
    if dirty:
        parts[0] = parts[0] + parts.pop(1) if len(parts) > 1 else parts[0]
        parts.append(".dirty" if "+" in parts[0] else "+dirty")
    return "".join(parts)


def setup_file_logging(outdir, logger_prefix="biom3", log_filename="run.log"):
    """Add a FileHandler to all loggers matching *logger_prefix* so output
    goes to both the console and a file in *outdir*.

    Only attaches the handler on rank 0 (distributed-safe).

    Returns (log_path, file_handler).  Pass the handler to
    ``teardown_file_logging`` when done.
    """
    from biom3.backend.device import get_rank

    log_path = os.path.join(outdir, log_filename)

    if get_rank() != 0:
        return log_path, None

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    file_handler.setLevel(logging.INFO)

    for name, lg in logging.Logger.manager.loggerDict.items():
        if isinstance(lg, logging.Logger) and name.startswith(logger_prefix):
            lg.addHandler(file_handler)

    return log_path, file_handler


def teardown_file_logging(logger_prefix, file_handler):
    """Remove *file_handler* from all loggers matching *logger_prefix* and
    close it."""
    if file_handler is None:
        return
    for name, lg in logging.Logger.manager.loggerDict.items():
        if isinstance(lg, logging.Logger) and name.startswith(logger_prefix):
            lg.removeHandler(file_handler)
    file_handler.close()


def get_file_metadata(filepath):
    """Return a dict with size, mtime, and real path for a file.

    Resolves symlinks so the true location is recorded.  Returns None
    if the file does not exist.
    """
    if not os.path.exists(filepath):
        return None
    real = os.path.realpath(filepath)
    stat = os.stat(real)
    return {
        "path": os.path.abspath(filepath),
        "realpath": real,
        "size_bytes": stat.st_size,
        "mtime": datetime.fromtimestamp(stat.st_mtime).isoformat(),
    }


def write_manifest(args, outdir, start_time, elapsed,
                   outputs=None, resolved_paths=None,
                   database_versions=None, environment=None,
                   config_contents=None,
                   manifest_filename="build_manifest.json"):
    """Write a JSON manifest with version info, args, timing, and outputs.

    Returns the path to the written manifest file.
    """
    manifest = {
        "biom3_version": get_biom3_version(),
        "biom3_build": get_biom3_build(),
        "git_hash": get_git_hash(),
        "git_dirty": get_git_dirty(),
        "git_branch": get_git_branch(),
        "git_remote": get_git_remote(),
        "timestamp": start_time.isoformat(),
        "elapsed_seconds": elapsed.total_seconds(),
        "command": " ".join(sys.argv),
        "args": {k: v for k, v in vars(args).items()},
        "python_version": sys.version,
    }
    if resolved_paths is not None:
        manifest["resolved_paths"] = resolved_paths
    if outputs is not None:
        manifest["outputs"] = outputs
    if database_versions is not None:
        manifest["database_versions"] = database_versions
    if environment is not None:
        manifest["environment"] = environment
    if config_contents is not None:
        manifest["config_contents"] = config_contents

    manifest_path = os.path.join(outdir, manifest_filename)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    return manifest_path

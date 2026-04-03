"""Tests for biom3.core.run_utils — shared logging and manifest utilities."""

import json
import logging
import os
import tempfile
from argparse import Namespace
from datetime import datetime, timedelta

import pytest

from biom3.core.run_utils import (
    get_biom3_version,
    get_git_hash,
    setup_file_logging,
    teardown_file_logging,
    write_manifest,
)


def test_get_biom3_version():
    version = get_biom3_version()
    assert isinstance(version, str)
    assert len(version) > 0


def test_get_git_hash():
    git_hash = get_git_hash()
    assert isinstance(git_hash, str)
    assert len(git_hash) > 0
    # In a git repo, should be a short hex string
    if git_hash != "unknown":
        assert all(c in "0123456789abcdef" for c in git_hash)


def test_setup_and_teardown_file_logging():
    """File logging captures logger messages, and teardown stops capture."""
    with tempfile.TemporaryDirectory() as tmpdir:
        prefix = "biom3.core.test_run_utils._test_logger"
        test_logger = logging.getLogger(prefix)
        test_logger.setLevel(logging.INFO)
        test_logger.propagate = False
        if not test_logger.handlers:
            test_logger.addHandler(logging.StreamHandler())

        log_path, handler = setup_file_logging(
            tmpdir, logger_prefix=prefix, log_filename="test.log",
        )

        assert os.path.exists(log_path)

        test_logger.info("hello from test")
        handler.flush()

        with open(log_path) as f:
            contents = f.read()
        assert "hello from test" in contents

        # After teardown, new messages should not appear
        teardown_file_logging(prefix, handler)
        test_logger.info("after teardown")

        with open(log_path) as f:
            contents = f.read()
        assert "after teardown" not in contents


def test_setup_file_logging_custom_filename():
    with tempfile.TemporaryDirectory() as tmpdir:
        prefix = "biom3.core.test_run_utils._custom"
        test_logger = logging.getLogger(prefix)
        test_logger.setLevel(logging.INFO)
        test_logger.propagate = False
        if not test_logger.handlers:
            test_logger.addHandler(logging.StreamHandler())

        log_path, handler = setup_file_logging(
            tmpdir, logger_prefix=prefix, log_filename="custom.log",
        )
        assert log_path.endswith("custom.log")
        teardown_file_logging(prefix, handler)


def test_write_manifest_structure():
    with tempfile.TemporaryDirectory() as tmpdir:
        args = Namespace(foo="bar", count=42)
        start = datetime.now()
        elapsed = timedelta(seconds=1.5)

        path = write_manifest(args, tmpdir, start, elapsed)

        assert os.path.exists(path)
        with open(path) as f:
            manifest = json.load(f)

        for key in ("biom3_version", "biom3_build", "git_hash", "git_dirty",
                     "git_branch", "git_remote", "timestamp",
                     "elapsed_seconds", "command", "args", "python_version"):
            assert key in manifest, f"Missing key: {key}"

        assert manifest["args"]["foo"] == "bar"
        assert manifest["args"]["count"] == 42
        assert manifest["elapsed_seconds"] == 1.5


def test_write_manifest_with_outputs_and_resolved_paths():
    with tempfile.TemporaryDirectory() as tmpdir:
        args = Namespace(x=1)
        start = datetime.now()
        elapsed = timedelta(seconds=0.1)

        path = write_manifest(
            args, tmpdir, start, elapsed,
            outputs={"num_samples": 100, "file": "/tmp/out.pt"},
            resolved_paths={"input": "/data/in.csv"},
        )

        with open(path) as f:
            manifest = json.load(f)

        assert manifest["outputs"]["num_samples"] == 100
        assert manifest["resolved_paths"]["input"] == "/data/in.csv"


def test_write_manifest_default_excludes_optional_keys():
    """When outputs/resolved_paths are not provided, they are omitted."""
    with tempfile.TemporaryDirectory() as tmpdir:
        args = Namespace()
        start = datetime.now()
        elapsed = timedelta(seconds=0)

        path = write_manifest(args, tmpdir, start, elapsed)

        with open(path) as f:
            manifest = json.load(f)

        assert "outputs" not in manifest
        assert "resolved_paths" not in manifest
        assert "config_contents" not in manifest


def test_write_manifest_with_config_contents():
    """Config contents dict is embedded in the manifest."""
    with tempfile.TemporaryDirectory() as tmpdir:
        args = Namespace(x=1)
        start = datetime.now()
        elapsed = timedelta(seconds=0.1)
        config = {
            "model_type": "pfam",
            "batch_size": 64,
            "nested": {"lr": 0.001, "weight_decay": 1e-5},
        }

        path = write_manifest(
            args, tmpdir, start, elapsed,
            config_contents=config,
        )

        with open(path) as f:
            manifest = json.load(f)

        assert "config_contents" in manifest
        assert manifest["config_contents"]["model_type"] == "pfam"
        assert manifest["config_contents"]["batch_size"] == 64
        assert manifest["config_contents"]["nested"]["lr"] == 0.001


def test_write_manifest_with_multi_config_contents():
    """Dict-of-dicts config_contents works for the pipeline case."""
    with tempfile.TemporaryDirectory() as tmpdir:
        args = Namespace(x=1)
        start = datetime.now()
        elapsed = timedelta(seconds=0.1)
        config = {
            "pencl": {"dim": 512, "layers": 33},
            "facilitator": {"dim": 1024, "dropout": 0.1},
        }

        path = write_manifest(
            args, tmpdir, start, elapsed,
            config_contents=config,
        )

        with open(path) as f:
            manifest = json.load(f)

        assert manifest["config_contents"]["pencl"]["dim"] == 512
        assert manifest["config_contents"]["facilitator"]["dropout"] == 0.1

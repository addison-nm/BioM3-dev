"""Tests for biom3.core.run_utils — shared logging and manifest utilities."""

import json
import logging
import os
import tempfile
from argparse import Namespace
from datetime import datetime, timedelta

import pytest

from biom3.core.run_utils import (
    backup_if_exists,
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


# ---- backup_if_exists tests ----

def test_backup_if_exists_no_file():
    """Returns None when the file does not exist."""
    with tempfile.TemporaryDirectory() as tmpdir:
        result = backup_if_exists(os.path.join(tmpdir, "nonexistent.json"))
        assert result is None


def test_backup_if_exists_creates_backup():
    """Renames existing file to a .bak.<timestamp> path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fpath = os.path.join(tmpdir, "data.json")
        with open(fpath, "w") as f:
            f.write("original")

        backup_path = backup_if_exists(fpath)

        assert backup_path is not None
        assert not os.path.exists(fpath)
        assert os.path.exists(backup_path)
        assert ".bak." in os.path.basename(backup_path)
        with open(backup_path) as f:
            assert f.read() == "original"


def test_backup_if_exists_avoids_clobbering():
    """When a backup with the same timestamp already exists, appends a counter."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fpath = os.path.join(tmpdir, "data.json")

        # Create first file and back it up
        with open(fpath, "w") as f:
            f.write("v1")
        first_backup = backup_if_exists(fpath)

        # Create second file with the same mtime and back it up
        with open(fpath, "w") as f:
            f.write("v2")
        # Force same mtime as the original
        mtime = os.path.getmtime(first_backup)
        os.utime(fpath, (mtime, mtime))
        second_backup = backup_if_exists(fpath)

        assert first_backup != second_backup
        assert os.path.exists(first_backup)
        assert os.path.exists(second_backup)
        with open(first_backup) as f:
            assert f.read() == "v1"
        with open(second_backup) as f:
            assert f.read() == "v2"


def test_backup_if_exists_handles_symlinks():
    """Backs up symlinks (renames the link, not the target)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        target = os.path.join(tmpdir, "target.pth")
        link = os.path.join(tmpdir, "state_dict.pth")
        with open(target, "w") as f:
            f.write("weights")
        os.symlink(target, link)

        backup_path = backup_if_exists(link)

        assert backup_path is not None
        assert not os.path.exists(link)
        # Target should still exist (rename moved the symlink, not the target)
        assert os.path.exists(target)
        # Backup is the renamed symlink
        assert os.path.islink(backup_path)


# ---- write_manifest overwrite protection ----

def test_write_manifest_backs_up_existing():
    """Calling write_manifest twice backs up the first manifest."""
    with tempfile.TemporaryDirectory() as tmpdir:
        args_v1 = Namespace(run="first")
        args_v2 = Namespace(run="second")
        start = datetime.now()
        elapsed = timedelta(seconds=1)

        path1 = write_manifest(args_v1, tmpdir, start, elapsed)
        assert os.path.exists(path1)

        path2 = write_manifest(args_v2, tmpdir, start, elapsed)
        assert path1 == path2  # same canonical path

        # The current manifest should have the second run's data
        with open(path2) as f:
            current = json.load(f)
        assert current["args"]["run"] == "second"

        # There should be exactly one .bak file with the first run's data
        bak_files = [f for f in os.listdir(tmpdir) if ".bak." in f]
        assert len(bak_files) == 1
        with open(os.path.join(tmpdir, bak_files[0])) as f:
            backed_up = json.load(f)
        assert backed_up["args"]["run"] == "first"

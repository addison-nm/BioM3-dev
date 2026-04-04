"""Pytest Configuration File

"""

import pytest
import os
import shutil

DATDIR = "tests/_data"  # data directory for all tests.
TMPDIR = "tests/_tmp"  # output directory for all tests.

def remove_dir(dir:str):
    """Helper function to remove a directory recursively."""
    if not dir.startswith(TMPDIR):
        msg = f"Can only use function `remove_dir` from tests.conftest to \
        remove directories in the directory {TMPDIR}. Got: {dir}"
        raise RuntimeError(msg)
    shutil.rmtree(dir)

def get_args(fpath):
    """Read command line arguments from a text file."""
    with open(fpath, 'r') as f:
        # Strip whitespace and ignore empty lines
        lines = [line.strip() for line in f if line.strip()]
    # Join into a single string and split into arguments
    argstring = " ".join(lines)
    arglist = argstring.split()
    return arglist

def check_downloads(paths_to_check):
    """Returns list of missing files and a warning message."""
    issues = []
    for fpath in paths_to_check:
        if not os.path.exists(fpath):
            msg = f"Weight files not found: {fpath}"
            issues.append(msg)
    msg = ""
    if issues:
        msg = "Entrypoint test relies on downloaded weights!"
        msg += "\nThis test will be skipped until the following issues are resolved:"
        for issue in issues:
            msg += f"\n  {issue}"
    return issues, msg

#####################
##  Configuration  ##
#####################

def pytest_addoption(parser):
    parser.addoption(
        "--benchmark", action="store_true", default=False,
        help="run benchmarking tests"
    )
    parser.addoption(
        "--use_gpu", action="store_true", default=False,
        help="run GPU specific tests"
    )
    parser.addoption(
        "--database_files", action="store_true", default=False,
        help="run tests that require full database files"
    )
    parser.addoption(
        "--network", action="store_true", default=False,
        help="run tests that require network access"
    )

def pytest_configure(config):
    config.addinivalue_line("markers", "benchmark: mark test as benchmarking")
    config.addinivalue_line("markers", "use_gpu: mark test as GPU specific")
    config.addinivalue_line("markers", "database_files: mark test as requiring full database files")
    config.addinivalue_line("markers", "network: mark test as requiring network access")

def pytest_collection_modifyitems(config, items):
    benchmark_flag_given = False
    use_gpu_flag_given = False
    if config.getoption("--benchmark"):
        # --benchmark given in cli: do not skip benchmarking tests
        benchmark_flag_given = True
    if config.getoption("--use_gpu"):
        # --use_gpu given in cli: do not skip GPU tests
        use_gpu_flag_given = True
    database_files_flag_given = False
    if config.getoption("--database_files"):
        database_files_flag_given = True
    network_flag_given = False
    if config.getoption("--network"):
        network_flag_given = True
    skip_benchmark = pytest.mark.skip(reason="need --benchmark option to run")
    skip_use_gpu = pytest.mark.skip(reason="need --use_gpu option to run")
    skip_database_files = pytest.mark.skip(reason="need --database_files option to run")
    skip_network = pytest.mark.skip(reason="need --network option to run")
    for item in items:
        if "benchmark" in item.keywords and not benchmark_flag_given:
            item.add_marker(skip_benchmark)
        if "use_gpu" in item.keywords and not use_gpu_flag_given:
            item.add_marker(skip_use_gpu)
        if "database_files" in item.keywords and not database_files_flag_given:
            item.add_marker(skip_database_files)
        if "network" in item.keywords and not network_flag_given:
            item.add_marker(skip_network)
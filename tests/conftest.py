"""Pytest Configuration File

"""

import pytest
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

def pytest_configure(config):
    config.addinivalue_line("markers", "benchmark: mark test as benchmarking")
    config.addinivalue_line("markers", "use_gpu: mark test as GPU specific")

def pytest_collection_modifyitems(config, items):
    benchmark_flag_given = False
    use_gpu_flag_given = False
    if config.getoption("--benchmark"):
        # --benchmark given in cli: do not skip benchmarking tests
        benchmark_flag_given = True
    if config.getoption("--use_gpu"):
        # --use_gpu given in cli: do not skip GPU tests
        use_gpu_flag_given = True
    skip_benchmark = pytest.mark.skip(reason="need --benchmark option to run")
    skip_use_gpu = pytest.mark.skip(reason="need --use_gpu option to run")
    for item in items:
        if "benchmark" in item.keywords and not benchmark_flag_given:
            item.add_marker(skip_benchmark)
        if "use_gpu" in item.keywords and not use_gpu_flag_given:
            item.add_marker(skip_use_gpu)
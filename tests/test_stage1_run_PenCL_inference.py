"""Tests for entrypoint biom3_run_PenCL_inference

Tests script: src/Stage1_source/run_PenCL_inference.py

"""

import pytest
import os
from contextlib import nullcontext as does_not_raise

from tests.conftest import DATDIR, TMPDIR, remove_dir

from Stage1_source.run_PenCL_inference import parse_arguments, main

#####################
##  Configuration  ##
#####################

# Directory containing text files with command line arguments
ARGS_DIR = os.path.join(DATDIR, "entrypoint_args")
OUTPUTS_DIR = os.path.join(TMPDIR, "outputs")

# Required weights that need to be downloaded to run entrypoint test
REQUIRED_DOWNLOADS = [
    "weights/LLMs/esm2_t33_650M_UR50D.pt",
]

def get_args(fpath):
    with open(fpath, 'r') as f:
        argstring = f.readline()
        arglist = argstring.split(" ")
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


###############################################################################
###############################   BEGIN TESTS   ###############################
###############################################################################

@pytest.mark.parametrize("argstring_fpath, expect_error_context", [
    [f"{ARGS_DIR}/stage1_args_v1.txt", does_not_raise()],
    [f"{ARGS_DIR}/stage1_args_v2.txt", does_not_raise()],
])
def test_entrypoint(
        argstring_fpath, expect_error_context
    ):
    # This test relies on the following downloaded weights. Check existence.
    issues, skip_reason = check_downloads(REQUIRED_DOWNLOADS)
    if issues:
        pytest.skip(reason=skip_reason)
    # Parse the command line string
    argstring = get_args(argstring_fpath)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    # Run entrypoint
    with expect_error_context:
        args = parse_arguments(argstring)
        main(args)
    remove_dir(OUTPUTS_DIR)

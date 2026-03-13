"""Tests for entrypoint biom3_ProteoScribe_sample

Tests script: src/biom3/Stage2/run_ProteoScribe_sample.py

"""

import pytest
import os
from contextlib import nullcontext as does_not_raise

from tests.conftest import DATDIR, TMPDIR, remove_dir

import torch
import numpy as np
from biom3.Stage3.run_ProteoScribe_sample import parse_arguments, main

#####################
##  Configuration  ##
#####################

# Directory containing text files with command line arguments
ARGS_DIR = os.path.join(DATDIR, "entrypoint_args")
OUTPUTS_DIR = os.path.join(TMPDIR, "outputs")

# Required weights that need to be downloaded to run entrypoint test
REQUIRED_DOWNLOADS = [
    "weights/ProteoScribe/BioM3_ProteoScribe_pfam_epoch20_v1.bin",
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

@pytest.mark.parametrize(
    "argstring_fpath, expect_error_context", [
    [f"{ARGS_DIR}/stage3_args_v2.txt", does_not_raise()],
])
@pytest.mark.parametrize("device", ["cpu", "cuda", "xpu"])
def test_entrypoint(
        argstring_fpath, expect_error_context, device
    ):
    # This test relies on the following downloaded weights. Check existence.
    issues, skip_reason = check_downloads(REQUIRED_DOWNLOADS)
    if issues:
        pytest.skip(reason=skip_reason)
    # Skip device if not available on machine
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip(reason="device=cuda and cuda not available")
    elif device == "xpu" and not torch.xpu.is_available():
        pytest.skip(reason="device=xpu and xpu not available")
    # Parse the command line string
    argstring = get_args(argstring_fpath)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    # Run entrypoint
    with expect_error_context:
        args = parse_arguments(argstring)
        args.device = device
        main(args)
    remove_dir(OUTPUTS_DIR)


@pytest.mark.parametrize(
    "argstring_fpath", [
    f"{ARGS_DIR}/stage3_args_v2.txt",
])
@pytest.mark.parametrize(
    "seed1, seed2", [
        [1, 2],
        [1, 1]
    ]
)
@pytest.mark.parametrize("device", ["cpu", "cuda", "xpu"])
def test_reproducibility(
        argstring_fpath, seed1, seed2, device
    ):
    # This test relies on the following downloaded weights. Check existence.
    issues, skip_reason = check_downloads(REQUIRED_DOWNLOADS)
    if issues:
        pytest.skip(reason=skip_reason)
    # Skip device if not available on machine
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip(reason="device=cuda and cuda not available")
    elif device == "xpu" and not torch.xpu.is_available():
        pytest.skip(reason="device=xpu and xpu not available")
    # Parse the command line string
    argstring = get_args(argstring_fpath)
    # Run 1
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    args = parse_arguments(argstring)
    args.device = device
    args.seed = seed1
    main(args)
    res_dict1 = torch.load(
        os.path.join(OUTPUTS_DIR, "test_ProteoScribe_samples.pt")
    )
    remove_dir(OUTPUTS_DIR)
    # Run 2
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    args = parse_arguments(argstring)
    args.device = device
    args.seed = seed2
    main(args)
    res_dict2 = torch.load(
        os.path.join(OUTPUTS_DIR, "test_ProteoScribe_samples.pt")
    )
    remove_dir(OUTPUTS_DIR)
    # Compare results
    expect_same = seed1 == seed2
    errors = []
    assert len(res_dict1) == len(res_dict2)
    for i in range(len(res_dict1)):
        replicates1 = res_dict1[f"replica_{i}"]
        replicates2 = res_dict2[f"replica_{i}"]
        observed_same = np.all(
            [s1 == s2 for s1, s2 in zip(replicates1, replicates2)]
        )
        if expect_same and not observed_same:
            msg = "Expected same results but results differed."
            msg += f"\n  Replicates 1: {replicates1}"
            msg += f"\n  Replicates 2: {replicates2}"
            errors.append(msg)
        elif not expect_same and observed_same:
            msg = "Expected different results but results matched."
            msg += f"\n  Replicates 1 == Replicates 2: {replicates1}"
            errors.append(msg)
    assert not errors, "Errors occurred:\n{}".format("\n".join(errors))

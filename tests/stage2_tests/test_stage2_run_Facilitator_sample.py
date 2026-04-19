"""Tests for entrypoint biom3_Facilitator_sample

Tests script: src/biom3/Stage2/run_Facilitator_sample.py

"""

import pytest
import os
from contextlib import nullcontext as does_not_raise

from tests.conftest import DATDIR, TMPDIR, remove_dir, get_args, check_downloads

import torch
from biom3.Stage2.run_Facilitator_sample import parse_arguments, main

pytestmark = [pytest.mark.slow]

#####################
##  Configuration  ##
#####################

# Directory containing text files with command line arguments
ARGS_DIR = os.path.join(DATDIR, "entrypoint_args")
OUTPUTS_DIR = os.path.join(TMPDIR, "outputs")

# Required weights that need to be downloaded to run entrypoint test
REQUIRED_DOWNLOADS = [
    "weights/Facilitator/BioM3_Facilitator_epoch20.bin",
]


###############################################################################
###############################   BEGIN TESTS   ###############################
###############################################################################

@pytest.mark.parametrize(
    "argstring_fpath, expect_error_context", [
    [f"{ARGS_DIR}/stage2_args_v1.txt", does_not_raise()],
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
        # Verify results can be loaded
        res = torch.load(
            os.path.join(OUTPUTS_DIR, "test_Facilitator_embeddings.pt")
        )
        errors = []
        expected_keys = [
            "z_t", "z_p", "z_c"
        ]
        for k in expected_keys:
            if k not in res:
                msg = f"key {k} not found in results"
                errors.append(msg)
        remove_dir(OUTPUTS_DIR)
        assert not errors, "Errors occurred:\n{}".format("\n".join(errors))

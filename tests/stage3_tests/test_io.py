"""
Tests IO module for Stage 3 ProteoScribe

"""

import pytest
import os
from contextlib import nullcontext as does_not_raise
from tests.conftest import DATDIR, TMPDIR, remove_dir

import json
import argparse

from biom3.core.helpers import load_json_config, convert_to_namespace
from biom3.core.helpers import get_num_named_weights
from biom3.Stage3.io import prepare_model_ProteoScribe

#####################
##  Configuration  ##
#####################

# Directory containing model configuration files (e.g. .json)
CONFIGS_DIR = os.path.join(DATDIR, "models/stage3/configs")
# Directory containing model weight files (e.g. .bin)
WEIGHTS_DIR = os.path.join(DATDIR, "models/stage3/weights")


###############################################################################
###############################   BEGIN TESTS   ###############################
###############################################################################

@pytest.mark.parametrize("config_fpath, tot_weights_exp", [
    [f"{CONFIGS_DIR}/minimodel1.json", None],
    [f"{CONFIGS_DIR}/orig_model.json", None],
])
@pytest.mark.parametrize("device", ["cpu"])
@pytest.mark.parametrize("eval_flag", [True, False])
def test_model_load_scratch(
        config_fpath, tot_weights_exp, device, eval_flag
    ):
    config_dict = load_json_config(config_fpath)
    config_args = convert_to_namespace(config_dict)
    model = prepare_model_ProteoScribe(
        config_args,
        weights_fpath=None,
        device=device,
        eval=eval_flag,
        verbosity=0,
    )

    if tot_weights_exp is not None:
        num_weights = get_num_named_weights(model)
        msg = f"Unexpected number of (named) weights!" \
            f" Expected {tot_weights_exp}. Got {num_weights}."
        assert num_weights == tot_weights_exp, msg


@pytest.mark.parametrize(
        "config_fpath, weights_fpath, tot_weights_exp, " \
        "names_mismatched, expect_error_context", [
    [
        f"{CONFIGS_DIR}/orig_model.json", 
        "weights/ProteoScribe/BioM3_ProteoScribe_pfam_epoch20_v1.bin", 
        None, 
        True, 
        # pytest.raises(RuntimeError, match=r"Unexpected key\(s\) in state_dict"),
        does_not_raise(),
    ],
])
@pytest.mark.parametrize("attempt_correction", [True, False])
@pytest.mark.parametrize("strict", [True, False])
@pytest.mark.parametrize("device", ["cpu"])
@pytest.mark.parametrize("eval_flag", [True, False])
def test_model_load_from_bin(
        config_fpath, weights_fpath, tot_weights_exp, 
        names_mismatched, expect_error_context,
        attempt_correction, strict, device, eval_flag
    ):
    """
    Test the ability to load model weights contained in e.g. a .bin file.
    In some cases we expect a mismatch between the names of parameters specified
    in the model definition and the saved weights. This is the case for the model
    parameters saved and made available in the original BioM3 code on HuggingFace.
    In this case, if we require STRICT assignment and attempt to infer the correct 
    assignment using the argument attempt_correction=True, we may or may not be 
    able to fix the issue. If we expect the attempt to be successful, we can 
    stipulate as such by including the suitable expect_error_context.
    
    """
    config_dict = load_json_config(config_fpath)
    config_args = convert_to_namespace(config_dict)

    if names_mismatched and strict and not attempt_correction:
        context = expect_error_context
    else:
        context = does_not_raise()    
    
    with context:
        model = prepare_model_ProteoScribe(
            config_args,
            weights_fpath=weights_fpath,
            strict=strict,
            device=device,
            eval=eval_flag,
            attempt_correction=attempt_correction,
            verbosity=0,
        )

    if tot_weights_exp is not None:
        num_weights = get_num_named_weights(model)
        msg = f"Unexpected number of (named) weights!" \
            f" Expected {tot_weights_exp}. Got {num_weights}."
        assert num_weights == tot_weights_exp, msg


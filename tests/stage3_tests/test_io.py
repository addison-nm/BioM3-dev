"""
Tests IO module for Stage 3 ProteoScribe

"""

import pytest
import os
from contextlib import nullcontext as does_not_raise
from tests.conftest import DATDIR, TMPDIR, remove_dir

import numpy as np
import json
import argparse

from biom3.core.helpers import load_json_config, convert_to_namespace
from biom3.core.helpers import get_num_named_weights, compare_model_params
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
    [f"{CONFIGS_DIR}/minimodel2.json", None],
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


@pytest.mark.parametrize(
        "config_fpath1, config_fpath2, weights_fpath1, weights_fpath2, " \
        "exp_same_params, exp_same_vals", [
    [
        f"{CONFIGS_DIR}/minimodel1.json", 
        f"{CONFIGS_DIR}/minimodel1.json",
        f"{WEIGHTS_DIR}/minimodel1_weights1.pth",
        f"{WEIGHTS_DIR}/minimodel1_weights1.pth",
        True, True
    ],
    [
        f"{CONFIGS_DIR}/minimodel1.json", 
        f"{CONFIGS_DIR}/minimodel1.json",
        f"{WEIGHTS_DIR}/minimodel1_weights1.pth",
        f"{WEIGHTS_DIR}/minimodel1_weights2.pth",
        True, False
    ],
    [
        f"{CONFIGS_DIR}/minimodel1.json", 
        f"{CONFIGS_DIR}/minimodel2.json",
        f"{WEIGHTS_DIR}/minimodel1_weights1.pth",
        f"{WEIGHTS_DIR}/minimodel2_weights1.pth",
        True, False
    ],
])
@pytest.mark.parametrize("device", ["cpu"])
@pytest.mark.parametrize("eval_flag", [False])  # add True
def test_compare_models(
        config_fpath1, config_fpath2, 
        weights_fpath1, weights_fpath2, 
        device, eval_flag,
        exp_same_params, exp_same_vals,
    ):
    config_dict1 = load_json_config(config_fpath1)
    config_dict2 = load_json_config(config_fpath2)
    config_args1 = convert_to_namespace(config_dict1)
    config_args2 = convert_to_namespace(config_dict2)
    model1 = prepare_model_ProteoScribe(
            config_args1,
            weights_fpath=weights_fpath1,
            strict=True,
            device=device,
            eval=eval_flag,
            attempt_correction=False,
            verbosity=0,
    )
    model2 = prepare_model_ProteoScribe(
            config_args2,
            weights_fpath=weights_fpath2,
            strict=True,
            device=device,
            eval=eval_flag,
            attempt_correction=False,
            verbosity=0,
    )

    results = compare_model_params(model1, model2)
    num_weights1 = results["num_weights1"]
    num_weights2 = results["num_weights2"]
    model1_only = results["model1_only"]
    model2_only = results["model2_only"]
    common_names = results["common_names"]
    differences = results["differences"]
    errors = []
    same_params = len(model1_only) == 0 and len(model2_only) == 0
    if same_params and not exp_same_params:
        msg = f"Expected models to have different param names but all match."
        errors.append(msg)
    elif exp_same_params and not same_params:
        msg = f"Expected models to have the same param names but names differ."
        msg += f"  model1 only: {model1_only}\n  model2 only: {model2_only}"
        errors.append(msg)
    
    tol = 1e-6
    max_diff_in_common = np.max([differences[k] for k in common_names])
    if exp_same_params and exp_same_vals:
        # Expect same parameter names and all values to match
        if max_diff_in_common > tol:
            msg = f"Expected values to match. Max diff: {max_diff_in_common}"
            errors.append(msg)
    elif exp_same_params and not exp_same_vals:
        # Expect same parameter names and some values to differ
        if max_diff_in_common == 0:
            msg = f"Expected some values to differ, but all same."
            errors.append(msg)
    elif not exp_same_params and exp_same_vals:
        # Expect different parameter names and all names in common to match
        if max_diff_in_common > tol:
            msg = f"Expected values to match. Max diff: {max_diff_in_common}"
            errors.append(msg)
    elif not exp_same_params and not exp_same_vals:
        # Expect different parameter names and some names in common to differ
        if max_diff_in_common == 0:
            msg = f"Expected some values to differ, but all same."
            errors.append(msg)

    assert not errors, "Errors occurred:\n{}".format("\n".join(errors))

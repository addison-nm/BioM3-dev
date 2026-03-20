"""Tests for entrypoint biom3_stage3_training

Tests script: src/biom3/Stage3/run_PL_training.py

"""

import pytest
import os
from contextlib import nullcontext as does_not_raise

import numpy as np
import torch

from tests.conftest import DATDIR, TMPDIR, remove_dir

from biom3.Stage3.run_PL_training import parse_arguments, main
from biom3.core.io import load_state_dict

#####################
##  Configuration  ##
#####################

# Directory containing text files with command line arguments
ARGS_DIR = os.path.join(DATDIR, "entrypoint_args", "training")
OUTPUTS_DIR = os.path.join(TMPDIR, "outputs")

# Required weights that need to be downloaded to run entrypoint test
REQUIRED_DOWNLOADS = [
    "weights/ProteoScribe/BioM3_ProteoScribe_pfam_epoch20_v1.bin",
]

# Transformer output layer parameter names
TRANSFORMER_OUTPUT_LAYER_PARAMS = [
    "transformer.norm.weight", "transformer.norm.bias",
    "transformer.out.weight", "transformer.out.bias",
]

def get_args(fpath):
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

def prefix_paths(args):
    """Prepend data files with path to test directories"""
    if args.swissprot_data_root != "None" and args.swissprot_data_root is not None:
        args.swissprot_data_root = os.path.join(
            DATDIR, args.swissprot_data_root
        )
    if args.pfam_data_root != "None" and args.pfam_data_root is not None:
        args.pfam_data_root = os.path.join(
            DATDIR, args.pfam_data_root
        )
    if args.pretrained_weights != "None" and args.pretrained_weights is not None:
        args.pretrained_weights = os.path.join(
            DATDIR, args.pretrained_weights
        )
    if args.wandb_logging_dir != "None" and args.wandb_logging_dir is not None:
        args.wandb_logging_dir = os.path.join(
            TMPDIR, args.wandb_logging_dir
        )
    if args.output_folder != "None" and args.output_folder is not None:
        args.output_folder = os.path.join(
            TMPDIR, args.output_folder
        )
    if args.output_hist_folder != "None" and args.output_hist_folder is not None:
        args.output_hist_folder = os.path.join(
            TMPDIR, args.output_hist_folder
        )
    if args.tb_logger_path != "None" and args.tb_logger_path is not None:
        args.tb_logger_path = os.path.join(
            TMPDIR, args.tb_logger_path
        )
    if args.resume_from_checkpoint != "None" and args.resume_from_checkpoint is not None:
        args.resume_from_checkpoint = os.path.join(
            TMPDIR, args.resume_from_checkpoint
        )


###############################################################################
###############################   BEGIN TESTS   ###############################
###############################################################################

# @pytest.mark.skip()
@pytest.mark.parametrize(
    "argstring_fpath, expect_error_context", [
    [f"{ARGS_DIR}/training_args_scratch_v1.txt", does_not_raise()],
])
@pytest.mark.parametrize("device", ["cuda", "xpu"])
def test_train_from_scratch(
        argstring_fpath, expect_error_context, device
    ):
    """
    Tests that basic training succeeds.

    Uses argfile(s):
        training_args_scratch_v1.txt
    """
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
        # Modifications to args:
        prefix_paths(args)
        args.device = device
        main(args)
    remove_dir(OUTPUTS_DIR)


# @pytest.mark.skip()
@pytest.mark.parametrize(
    "argstring_fpath, expect_error, exp_end_weights_fpath, exp_ckpt_fpath", [
    # Test that pretrained ProteoScribe weights can be loaded correctly
    [
        f"{ARGS_DIR}/training_args_pretrained_weights_v1.txt", 
        False, 
        f"{DATDIR}/models/stage3/weights/minimodel1_weights1.pth",
        f"{TMPDIR}/outputs/logs/history/checkpoints/training_args_pretrained_weights_v1/state_dict.pth"
    ],
    # Test that an error is raised when incompatible weights are specified.
    [
        f"{ARGS_DIR}/training_args_pretrained_weights_v2.txt", 
        True, 
        None,
        None
    ],
])
@pytest.mark.parametrize(
    "learning_rate, exp_same_weights", [
    [1e-4, False],
    [0.0, True],
])
@pytest.mark.parametrize("device", ["cuda", "xpu"])
def test_train_from_pretrained_weights(
        argstring_fpath, expect_error, exp_end_weights_fpath, 
        exp_ckpt_fpath, learning_rate, exp_same_weights, device
    ):
    """
    Tests that a model can be trained from an initial state of pretrained 
    weights. Verifies that when trained with AdamW and learning rate of zero,
    the resulting weights match those loaded at the start (v1). Also checks 
    that an error is raised if incompatible weights are loaded (v2).

    Uses argfile(s):
        training_args_pretrained_weights_v1.txt
        training_args_pretrained_weights_v2.txt
    """
    # Skip device if not available on machine
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip(reason="device=cuda and cuda not available")
    elif device == "xpu" and not torch.xpu.is_available():
        pytest.skip(reason="device=xpu and xpu not available")

    num_epochs = 3
    # Parse the command line string
    argstring = get_args(argstring_fpath)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    # Parse and modify args
    args = parse_arguments(argstring)
    prefix_paths(args)
    args.device = device
    args.epochs = num_epochs
    args.lr = learning_rate
    
    # ---- ENTRYPOINT ERROR CASE ----
    if expect_error:
        with pytest.raises(RuntimeError, match=r"Error\(s\) in loading state_dict"):
            main(args)
        remove_dir(OUTPUTS_DIR)
        return
    
    # ---- ENTRYPOINT SUCCESS CASE ----
    main(args)

    # ---- ERROR CHECKS ----
    # Load the weights we expect the model to have loaded
    expected_end_weights_dict = torch.load(exp_end_weights_fpath)
    # Load the weights of the model at the last epoch
    end_weights_dict = load_state_dict(exp_ckpt_fpath)

    errors = []
    if len(expected_end_weights_dict) == 0:
        errors.append("No parameters found in expected end weights dictionary!")
    for name in expected_end_weights_dict:
        expected_end_weights = expected_end_weights_dict[name]
        if name not in end_weights_dict:
            msg = f"Missing parameter: {name}"
            errors.append(msg)
        else:
            end_weights = end_weights_dict[name]
            weights_changed = not torch.allclose(expected_end_weights, end_weights)
            if exp_same_weights and weights_changed:
                msg = f"Expected same weights but change occurred in: {name}"
                errors.append(msg)
            elif not exp_same_weights and not weights_changed:
                msg = f"Expected updates to weights but no change in: {name}"
                errors.append(msg)
    remove_dir(OUTPUTS_DIR)
    if len(expected_end_weights_dict) != len(end_weights_dict):
        msg = "Mismatch in number of expected parameters."
        msg += f"Expected {expected_end_weights_dict.keys()}. Got {end_weights_dict.keys()}"
        errors.append(msg)
    assert not errors, "Errors occurred:\n{}".format("\n".join(errors))


# @pytest.mark.skip()
@pytest.mark.parametrize(
    "argstring_fpath1, argstring_fpath2, expect_error, exp_state_dict_path1, exp_state_dict_path2", [
    # Test that resuming from a checkpoint succeeds
    [
        f"{ARGS_DIR}/training_args_resume_from_checkpoint_v1a.txt", 
        f"{ARGS_DIR}/training_args_resume_from_checkpoint_v1b.txt", 
        False, 
        f"{TMPDIR}/outputs/logs/history/checkpoints/training_args_resume_from_checkpoint_v1a/state_dict.pth",
        f"{TMPDIR}/outputs/logs/history/checkpoints/training_args_resume_from_checkpoint_v1b/state_dict.pth"
    ],
])
@pytest.mark.parametrize(
    "learning_rate1, learning_rate2, exp_same_weights", [
    [1e-4, 1e-4, False],
    [0., 0., True],
])
@pytest.mark.parametrize("device", ["cuda", "xpu"])
def test_resume_training(
        argstring_fpath1, argstring_fpath2, expect_error, 
        exp_state_dict_path1, exp_state_dict_path2, learning_rate1, learning_rate2, 
        exp_same_weights, device
    ):
    """
    Tests that model training can resume from a saved checkpoint. Runs initial
    training using first argfile, then continues for one or more additional 
    epochs. Verifies that when learning rate is 0 for both runs, the 
    the resulting weights of both models match. Both runs must use lr=0 otherwise
    momentum results in different weights.
    
    Uses argfile(s):
        training_args_resume_from_checkpoint_v1a.txt
        training_args_resume_from_checkpoint_v1b.txt
    """
    # Skip device if not available on machine
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip(reason="device=cuda and cuda not available")
    elif device == "xpu" and not torch.xpu.is_available():
        pytest.skip(reason="device=xpu and xpu not available")
    
    num_epochs1 = 2
    num_epochs2 = 3 # train for additional epoch(s) (n2 - n1)

    # Parse the command line string
    argstring1 = get_args(argstring_fpath1)
    argstring2 = get_args(argstring_fpath2)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    # Parse and modify args for initial run
    args1 = parse_arguments(argstring1)
    prefix_paths(args1)
    args1.device = device
    args1.epochs = num_epochs1
    args1.lr = learning_rate1

    # Parse and modify args for secondary run
    args2 = parse_arguments(argstring2)
    prefix_paths(args2)
    args2.device = device
    args2.epochs = num_epochs2
    args2.lr = learning_rate2

    # ---- Initial training ----
    main(args1)
    # Store model weights
    end_weights1 = torch.load(exp_state_dict_path1)

    # ---- Resume training ----
    main(args2)
    # Store model weights
    end_weights2 = torch.load(exp_state_dict_path2)

    # ---- ERROR CHECKS ----
    errors = []
    if len(end_weights1) != len(end_weights2):
        errors.append("Mismatch in number of parameters!")
    for name in end_weights1:
        w1 = end_weights1[name]
        if name not in end_weights2:
            msg = f"Missing parameter in subsequent training phase: {name}"
            errors.append(msg)
        else:
            w2 = end_weights2[name]
            weights_changed = not torch.allclose(w1, w2)
            if exp_same_weights and weights_changed:
                msg = f"Expected same weights but change occurred in: {name}"
                msg += f"\n\t max diff: {torch.max(torch.abs(w1-w2))}"
                errors.append(msg)
            elif not exp_same_weights and not weights_changed:
                msg = f"Expected updates to weights but no change in: {name}"
                errors.append(msg)
    remove_dir(OUTPUTS_DIR)
    assert not errors, "Errors occurred:\n{}".format("\n".join(errors))


# @pytest.mark.skip()
@pytest.mark.parametrize(
    "argstring_fpath, expect_error, weights_orig_fpath, exp_ckpt_fpath", [
    # Test that finetuning succeeds and only certain weights change
    [
        f"{ARGS_DIR}/finetuning_args_v1.txt", 
        False, 
        f"{DATDIR}/models/stage3/weights/minimodel1_weights1.pth",
        f"{TMPDIR}/outputs/logs/history/checkpoints/finetuning_args_v1/state_dict.pth",
    ],
])
@pytest.mark.parametrize(
    "finetune_output_layers, finetune_last_n_blocks, finetune_last_n_layers", [
    [True, -1, -1],  # train all blocks, all layers
    [True, 0, 0],  # train no blocks, no layers
    [True, 1, 1],  # last block, last layer
    [True, 1, 2],  # last block, last 2 layers
    [True, 1, 20],  # last block, last 20 layers (more than exist)
    [True, 2, 2],  # last 2 blocks, last 2 layers
    [True, 20, 2],  # last 20 blocks (more than exist), last 2 layers
    [True, -2, 1],  # unspecified blocks, last layer -> last block, last layer
    [True, -2, -2],  # unspecified blocks, unspecified layers -> last block, last layer
    [True, 1, -2],  # last block, unspecified layers -> last block, last layer
])
@pytest.mark.parametrize("device", ["cuda", "xpu"])
def test_finetuning(
        argstring_fpath, expect_error, 
        weights_orig_fpath, exp_ckpt_fpath, 
        finetune_output_layers, finetune_last_n_blocks, finetune_last_n_layers,
        device
    ):
    """
    Tests that a model can be finetuned from pretrained weights, specifying 
    different configurations of blocks and layers to finetune. Verifies that
    only the specified blocks/layers are updated.

    Uses argfile(s):
        finetuning_args_v1
    """
    # Skip device if not available on machine
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip(reason="device=cuda and cuda not available")
    elif device == "xpu" and not torch.xpu.is_available():
        pytest.skip(reason="device=xpu and xpu not available")

    num_epochs = 1

    # Parse the command line string
    argstring = get_args(argstring_fpath)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    # Parse and modify args
    args = parse_arguments(argstring)
    prefix_paths(args)
    args.device = device
    args.epochs = num_epochs
    args.finetune_output_layers = finetune_output_layers
    args.finetune_last_n_blocks = finetune_last_n_blocks
    args.finetune_last_n_layers = finetune_last_n_layers

    # Identify transformer architecture  # TODO: determine from model instead of args?
    transformer_blocks = args.transformer_blocks
    transformer_layers = args.transformer_depth
    
    expected_changes = []
    if finetune_output_layers:
        expected_changes += TRANSFORMER_OUTPUT_LAYER_PARAMS

    n_trained_blocks = args.finetune_last_n_blocks
    n_trained_layers = args.finetune_last_n_layers
    if n_trained_blocks == -2:
        n_trained_blocks = 1
    elif n_trained_blocks == -1:
        n_trained_blocks = transformer_blocks
    if n_trained_layers == -2:
        n_trained_layers = 1
    elif n_trained_layers == -1:
        n_trained_layers = transformer_layers
    
    trained_blocks = [
        i for i in range(transformer_blocks) 
        if i >= transformer_blocks - n_trained_blocks
    ]
    trained_layers = [
        i for i in range(transformer_layers) 
        if i >= transformer_layers - n_trained_layers
    ]
    for i in trained_blocks:
        for j in trained_layers:
            expected_changes.append(f"transformer.transformer_blocks.{i}.{j}.")

    # ---- Finetuning ----
    main(args)
    # Store model weights
    weights_final = torch.load(exp_ckpt_fpath)
    # Load original weights
    weights_orig = torch.load(weights_orig_fpath)

    # ---- ERROR CHECKS ----

    def _expect_change(name):
        return np.any([s in name for s in expected_changes])

    errors = []
    if len(weights_final) != len(weights_orig):
        errors.append("Mismatch in number of parameters!")
    for name in weights_orig:
        w_o = weights_orig[name]
        if name not in weights_final:
            msg = f"Missing parameter: {name}"
            errors.append(msg)
        else:
            w_f = weights_final[name]
            weights_changed = not torch.allclose(w_o, w_f)
            exp_change = _expect_change(name)
            if not exp_change and weights_changed:
                msg = f"Expected same weights but change occurred in: {name}"
                msg += f"\n\t max diff: {torch.max(torch.abs(w_f - w_o))}"
                errors.append(msg)
            elif exp_change and not weights_changed:
                msg = f"Expected updates to weights but no change in: {name}"
                errors.append(msg)
    remove_dir(OUTPUTS_DIR)
    assert not errors, "Errors occurred:\n{}".format("\n".join(errors))


# @pytest.mark.skip()
@pytest.mark.parametrize(
    "argstring_fpath1, argstring_fpath2, expect_error, exp_state_dict_path1, exp_state_dict_path2", [
    # Test that resumption succeeds
    [
        f"{ARGS_DIR}/training_args_phase2_training_v1a.txt", 
        f"{ARGS_DIR}/training_args_phase2_training_v1b.txt", 
        False, 
        f"{TMPDIR}/outputs/logs/history/checkpoints/training_args_phase2_training_v1a/state_dict.best.pth",
        f"{TMPDIR}/outputs/logs/history/checkpoints/training_args_phase2_training_v1b/state_dict.last.pth"
    ],
])
@pytest.mark.parametrize(
    "learning_rate1, learning_rate2, exp_same_weights", [
    [1e-4, 1e-4, False],
    [0., 0., True],
])
@pytest.mark.parametrize("device", ["cuda", "xpu"])
def test_start_phase2_training(
        argstring_fpath1, argstring_fpath2, expect_error, 
        exp_state_dict_path1, exp_state_dict_path2, learning_rate1, learning_rate2, 
        exp_same_weights, device
    ):
    """
    Tests phase 2 model training, in which a pretrained model is loaded and 
    further pretrained on a different data set (e.g. Pfam + SwissProt). Initial
    phase runs only on Pfam data. The best model state is then loaded in phase 2
    and trained for additional epochs. We verify that with learning rate of 0,
    The final model state in phase 2 has the same weights.

    Uses argfile(s):
        training_args_phase2_training_v1a
        training_args_phase2_training_v1b
    """
    # Skip device if not available on machine
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip(reason="device=cuda and cuda not available")
    elif device == "xpu" and not torch.xpu.is_available():
        pytest.skip(reason="device=xpu and xpu not available")

    # Parse the command line string
    argstring1 = get_args(argstring_fpath1)
    argstring2 = get_args(argstring_fpath2)
    if os.path.exists(OUTPUTS_DIR):
        remove_dir(OUTPUTS_DIR)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    # Parse and modify args for initial run
    args1 = parse_arguments(argstring1)
    prefix_paths(args1)
    args1.device = device
    args1.lr = learning_rate1

    # Parse and modify args for secondary run
    args2 = parse_arguments(argstring2)
    orig_pretrained_weights = args2.pretrained_weights
    prefix_paths(args2)
    args2.device = device
    args2.pretrained_weights = os.path.join(TMPDIR, orig_pretrained_weights)
    args2.lr = learning_rate2

    # ---- Initial training ----
    main(args1)
    # Store model weights
    end_weights1 = torch.load(exp_state_dict_path1)

    # ---- Second phase training ----
    main(args2)
    # Store model weights
    end_weights2 = torch.load(exp_state_dict_path2)

    # ---- ERROR CHECKS ----
    errors = []
    if len(end_weights1) != len(end_weights2):
        errors.append("Mismatch in number of parameters!")
    for name in end_weights1:
        w1 = end_weights1[name]
        if name not in end_weights2:
            msg = f"Missing parameter in subsequent training phase: {name}"
            errors.append(msg)
        else:
            w2 = end_weights2[name]
            weights_changed = not torch.allclose(w1, w2)
            if exp_same_weights and weights_changed:
                msg = f"Expected same weights but change occurred in: {name}"
                msg += f"\n\t max diff: {torch.max(torch.abs(w1-w2))}"
                errors.append(msg)
            elif not exp_same_weights and not weights_changed:
                msg = f"Expected updates to weights but no change in: {name}"
                errors.append(msg)
    remove_dir(OUTPUTS_DIR)
    assert not errors, "Errors occurred:\n{}".format("\n".join(errors))

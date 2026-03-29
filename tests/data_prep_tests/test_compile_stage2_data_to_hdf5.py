"""Tests for entrypoint biom3_compile_hdf5

Tests script: src/biom3/data_prep/compile_stage2_data_to_hdf5.py

"""

import pytest
import os
from contextlib import nullcontext as does_not_raise

from tests.conftest import DATDIR, TMPDIR, remove_dir

import h5py
from biom3.data_prep.compile_stage2_data_to_hdf5 import parse_arguments, main

#####################
##  Configuration  ##
#####################

INPUT_FPATH = os.path.join(DATDIR, "embeddings", "test_Facilitator_embeddings_with_acc_id.pt")
OUTPUTS_DIR = os.path.join(TMPDIR, "outputs")


###############################################################################
###############################   BEGIN TESTS   ###############################
###############################################################################

@pytest.mark.parametrize(
    "dataset_key, expect_error_context", [
    ["MMD_data", does_not_raise()],
    ["MSE_data", does_not_raise()],
])
def test_compile_hdf5(dataset_key, expect_error_context):
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    outfpath = os.path.join(OUTPUTS_DIR, "test_compiled_emb.hdf5")
    with expect_error_context:
        args = parse_arguments([
            "-i", INPUT_FPATH,
            "-o", outfpath,
            "--dataset_key", dataset_key,
        ])
        main(args)
        # Verify output file exists and contains expected structure
        assert os.path.isfile(outfpath), f"Output file not created: {outfpath}"
        with h5py.File(outfpath, "r") as f:
            assert dataset_key in f, f"Group '{dataset_key}' not in HDF5 file"
            errors = []
            expected_datasets = [
                "acc_id",
                "sequence",
                "sequence_length",
                "text_to_protein_embedding",
            ]
            for ds in expected_datasets:
                full_key = f"{dataset_key}/{ds}"
                if full_key not in f:
                    errors.append(f"dataset {full_key} not found in HDF5 file")
            assert not errors, "Errors occurred:\n{}".format("\n".join(errors))
        remove_dir(OUTPUTS_DIR)

"""Tests for entrypoint biom3_embedding_pipeline

Tests script: src/biom3/pipeline/embedding_pipeline.py

"""

import pytest
import os
from contextlib import nullcontext as does_not_raise

# Stage 2 torch.load calls need this for .pt files containing non-tensor data.
# Normally set by environment.sh; make it explicit here for CI/test contexts.
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

from tests.conftest import DATDIR, TMPDIR, remove_dir, check_downloads

import h5py
from biom3.pipeline.embedding_pipeline import parse_arguments, main

#####################
##  Configuration  ##
#####################

OUTPUTS_DIR = os.path.join(TMPDIR, "pipeline_outputs")

# Required weights that need to be downloaded to run entrypoint test
REQUIRED_DOWNLOADS = [
    "weights/LLMs/esm2_t33_650M_UR50D.pt",
    "weights/PenCL/BioM3_PenCL_epoch20.bin",
    "weights/Facilitator/BioM3_Facilitator_epoch20.bin",
]


###############################################################################
###############################   BEGIN TESTS   ###############################
###############################################################################

@pytest.mark.parametrize(
    "expect_error_context", [
    does_not_raise(),
])
@pytest.mark.parametrize("device", ["cpu", "cuda", "xpu"])
def test_embedding_pipeline(expect_error_context, device):
    import torch
    # This test relies on the following downloaded weights. Check existence.
    issues, skip_reason = check_downloads(REQUIRED_DOWNLOADS)
    if issues:
        pytest.skip(reason=skip_reason)
    # Skip device if not available on machine
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip(reason="device=cuda and cuda not available")
    elif device == "xpu" and not torch.xpu.is_available():
        pytest.skip(reason="device=xpu and xpu not available")

    prefix = "test_pipeline"
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    with expect_error_context:
        args = parse_arguments([
            "-i", "None",
            "-o", OUTPUTS_DIR,
            "--pencl_weights", "weights/PenCL/BioM3_PenCL_epoch20.bin",
            "--facilitator_weights", "weights/Facilitator/BioM3_Facilitator_epoch20.bin",
            "--pencl_config", "tests/_data/configs/test_stage1_config_v1.json",
            "--facilitator_config", "tests/_data/configs/test_stage2_config_v1.json",
            "--prefix", prefix,
            "--device", device,
            "--batch_size", "4",
            "--mmd_sample_limit", "5",
        ])
        main(args)

        # Verify all intermediate and final output files exist
        errors = []
        expected_files = [
            f"{prefix}.PenCL_emb.pt",
            f"{prefix}.Facilitator_emb.pt",
            f"{prefix}.compiled_emb.hdf5",
        ]
        for fname in expected_files:
            fpath = os.path.join(OUTPUTS_DIR, fname)
            if not os.path.isfile(fpath):
                errors.append(f"Expected output file not found: {fpath}")

        # Verify HDF5 structure
        hdf5_path = os.path.join(OUTPUTS_DIR, f"{prefix}.compiled_emb.hdf5")
        if os.path.isfile(hdf5_path):
            with h5py.File(hdf5_path, "r") as f:
                if "MMD_data" not in f:
                    errors.append("Group 'MMD_data' not in HDF5 file")
                else:
                    for ds in ["acc_id", "sequence", "sequence_length",
                               "text_to_protein_embedding"]:
                        if f"MMD_data/{ds}" not in f:
                            errors.append(f"dataset MMD_data/{ds} not in HDF5")

        remove_dir(OUTPUTS_DIR)
        assert not errors, "Errors occurred:\n{}".format("\n".join(errors))

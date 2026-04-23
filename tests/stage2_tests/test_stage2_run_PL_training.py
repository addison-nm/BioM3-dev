"""Smoke test for `biom3_pretrain_stage2` (Stage 2 Facilitator training).

Builds a tiny embedding-dict fixture on the fly, runs one epoch, and asserts
that the transformed SwissProt embedding dict gets written with the expected
`text_to_protein_embedding` key.
"""

import json
import os
from contextlib import nullcontext as does_not_raise

import pytest
import torch

from tests.conftest import DATDIR, TMPDIR, remove_dir, get_args

from biom3.Stage2.run_PL_training import parse_arguments, main


ARGS_DIR = os.path.join(DATDIR, "entrypoint_args", "training")
OUTPUTS_DIR = os.path.join(TMPDIR, "outputs", "stage2_smoke")


def _make_fixture(tmp_path, n=8, dim=512):
    fpath = os.path.join(tmp_path, "sample_embeddings.pt")
    torch.save(
        {
            "text_embedding": torch.randn(n, dim),
            "protein_embedding": torch.randn(n, dim),
        },
        fpath,
    )
    return fpath


@pytest.mark.parametrize(
    "argstring_fpath, expect_error_context", [
        [f"{ARGS_DIR}/stage2_training_args_scratch_v1.txt", does_not_raise()],
    ],
)
@pytest.mark.parametrize("device", ["cuda", "xpu"])
def test_stage2_train_from_scratch(argstring_fpath, expect_error_context, device, tmp_path):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip(reason="device=cuda and cuda not available")
    elif device == "xpu" and not (hasattr(torch, "xpu") and torch.xpu.is_available()):
        pytest.skip(reason="device=xpu and xpu not available")

    argstring = get_args(argstring_fpath)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    fixture = _make_fixture(str(tmp_path))
    output_swiss = os.path.join(str(tmp_path), "stage2_out.pt")

    # Override input/output paths at the CLI level
    argstring = argstring + [
        "--swissprot_data_path", fixture,
        "--pfam_data_path", "None",
        "--output_swissprot_dict_path", output_swiss,
    ]

    with expect_error_context:
        args = parse_arguments(argstring)
        args.output_root = os.path.join(TMPDIR, args.output_root)
        args.device = device
        main(args)

    run_dir = os.path.join(args.output_root, args.runs_folder, args.run_id)
    artifacts_dir = os.path.join(run_dir, "artifacts")
    checkpoint_dir = os.path.join(args.output_root, args.checkpoints_folder, args.run_id)

    assert os.path.exists(os.path.join(artifacts_dir, "args.json")), \
        f"args.json missing under {artifacts_dir}"
    assert os.path.exists(os.path.join(artifacts_dir, "build_manifest.json")), \
        f"build_manifest.json missing under {artifacts_dir}"
    assert os.path.exists(os.path.join(checkpoint_dir, "state_dict.best.pth")), \
        f"state_dict.best.pth missing under {checkpoint_dir}"

    assert os.path.exists(output_swiss), f"transformed embedding dict missing at {output_swiss}"
    out = torch.load(output_swiss, map_location="cpu", weights_only=False)
    assert "text_to_protein_embedding" in out, \
        f"text_to_protein_embedding missing from output dict keys: {list(out.keys())}"
    assert len(out["text_to_protein_embedding"]) == len(out["text_embedding"])

    remove_dir(OUTPUTS_DIR)

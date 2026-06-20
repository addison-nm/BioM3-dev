"""End-to-end smoke test for biom3_cluster_split via the --clusters_tsv path.

Exercises parse_arguments + main without requiring the mmseqs binary.
"""

import os

import pytest
import h5py

from biom3.split.run_split import parse_arguments, main, _member_id
from biom3.split import manifest as mf

DATDIR = "tests/_data"
TEST_HDF5 = f"{DATDIR}/data/Stage2_MMD_swissprot_embedding_subset_1000.hdf5"

pytestmark = pytest.mark.skipif(
    not os.path.exists(TEST_HDF5), reason="test HDF5 fixture not present"
)


def test_cli_writes_manifest_and_stats(tmp_path):
    with h5py.File(TEST_HDF5, "r") as f:
        n = len(f["MMD_data"]["sequence"])

    tsv = os.path.join(tmp_path, "clu.tsv")
    with open(tsv, "w") as fh:
        for i in range(n):
            rep = _member_id(0, i)
            fh.write(f"{rep}\t{rep}\n")

    out = os.path.join(tmp_path, "split_manifest.json")
    args = parse_arguments([
        "--primary_data_path", TEST_HDF5,
        "--facilitator", "MMD",
        "--train_frac", "0.8", "--val_frac", "0.1", "--test_frac", "0.1",
        "--no_length_filter",
        "--clusters_tsv", tsv,
        "-o", out,
    ])
    main(args)

    assert os.path.exists(out)
    assert os.path.exists(os.path.splitext(out)[0] + ".stats.md")

    manifest = mf.read_manifest(out)
    entry = manifest["files"][0]
    covered = sorted(entry["train"] + entry["val"] + entry["test"])
    assert covered == list(range(n))
    assert manifest["tool"]["name"] == "precomputed"
    # Singletons → ratios land essentially on target.
    assert manifest["ratios_achieved"]["train"] == pytest.approx(0.8, abs=0.02)

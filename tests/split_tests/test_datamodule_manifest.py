"""Stage 3 HDF5DataModule consuming a curated split manifest.

Verifies that, given a manifest, the DataModule uses its train/val indices
(not the random split), holds the test rows out of both loaders, and fails loud
on a fingerprint mismatch.
"""

import os

import pytest
import h5py

from biom3.Stage3.PL_wrapper import HDF5DataModule
from biom3.split import manifest as mf
from biom3.split.run_split import build_split_manifest, _member_id

DATDIR = "tests/_data"
TEST_HDF5 = f"{DATDIR}/data/Stage2_MMD_swissprot_embedding_subset_1000.hdf5"

pytestmark = pytest.mark.skipif(
    not os.path.exists(TEST_HDF5), reason="test HDF5 fixture not present"
)


def _n_rows():
    with h5py.File(TEST_HDF5, "r") as f:
        return len(f["MMD_data"]["sequence"])


def _write_manifest(tmp_path, n):
    # Pair consecutive rows into clusters of two, no length filter.
    tsv = os.path.join(tmp_path, "clu.tsv")
    with open(tsv, "w") as fh:
        for i in range(0, n, 2):
            rep = _member_id(0, i)
            fh.write(f"{rep}\t{rep}\n")
            if i + 1 < n:
                fh.write(f"{rep}\t{_member_id(0, i + 1)}\n")
    manifest, _ = build_split_manifest(
        file_paths=[TEST_HDF5], group_name="MMD_data",
        ratios={"train": 0.8, "val": 0.1, "test": 0.1},
        seed=0, min_seq_length=None, clusters_tsv=tsv,
    )
    path = os.path.join(tmp_path, "split_manifest.json")
    mf.write_manifest(path, manifest)
    return path, manifest


def _make_dm(manifest_path):
    dm = HDF5DataModule(
        batch_size=32, num_workers=0, valid_size=0.2, seed=42,
        diffusion_steps=1024, image_size=32,
        primary_path=TEST_HDF5, group_name="MMD_data",
        split_manifest_path=manifest_path,
    )
    dm.setup()
    return dm


def test_datamodule_uses_manifest_indices(tmp_path):
    n = _n_rows()
    path, manifest = _write_manifest(tmp_path, n)
    dm = _make_dm(path)
    info = dm.split_info[0]
    entry = manifest["files"][0]
    # Manifest indices (length filter is a no-op for this fixture) are used.
    assert list(info["train_indices"]) == entry["train"]
    assert list(info["val_indices"]) == entry["val"]
    assert list(info["test_indices"]) == entry["test"]


def test_test_rows_held_out_of_loaders(tmp_path):
    n = _n_rows()
    path, manifest = _write_manifest(tmp_path, n)
    dm = _make_dm(path)
    info = dm.split_info[0]
    loaded = set(info["train_indices"]) | set(info["val_indices"])
    test_rows = set(info["test_indices"])
    assert test_rows
    assert loaded.isdisjoint(test_rows)
    assert len(dm.train_dataset) == len(info["train_indices"])
    assert len(dm.val_dataset) == len(info["val_indices"])


def test_fingerprint_mismatch_raises(tmp_path):
    n = _n_rows()
    path, manifest = _write_manifest(tmp_path, n)
    manifest["files"][0]["fingerprint"] = "tampered"
    mf.write_manifest(path, manifest)
    dm = HDF5DataModule(
        batch_size=32, num_workers=0, valid_size=0.2, seed=42,
        diffusion_steps=1024, image_size=32,
        primary_path=TEST_HDF5, group_name="MMD_data",
        split_manifest_path=path,
    )
    with pytest.raises(ValueError, match="fingerprint"):
        dm.setup()


def test_file_count_mismatch_raises(tmp_path):
    n = _n_rows()
    path, manifest = _write_manifest(tmp_path, n)
    # Manifest has 1 file; provide 2 paths.
    dm = HDF5DataModule(
        batch_size=32, num_workers=0, valid_size=0.2, seed=42,
        diffusion_steps=1024, image_size=32,
        primary_path=TEST_HDF5, secondary_paths=[TEST_HDF5],
        group_name="MMD_data", split_manifest_path=path,
    )
    with pytest.raises(ValueError, match="manifest"):
        dm.setup()

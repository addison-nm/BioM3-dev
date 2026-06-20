"""Tests for split-manifest construction, round-trip, and fingerprinting."""

import os

import pytest

from biom3.split.pack import pack_clusters
from biom3.split import manifest as mf


def _build(tmp_path, n_files=1):
    # Two files, members keyed by (file_index, row).
    clusters = [
        [(0, 0), (0, 1), (0, 2)],
        [(0, 3), (1, 0)],
        [(1, 1), (1, 2)],
        [(1, 3)],
    ]
    result = pack_clusters(clusters, {"train": 0.75, "val": 0.25}, seed=0)
    files = [
        {"path": f"/data/file{fi}.hdf5", "group": "MMD_data",
         "n_rows": 4, "fingerprint": f"fp{fi}"}
        for fi in range(2)
    ]
    return mf.build_manifest(
        files=files, pack_result=result, ratios_target={"train": 0.75, "val": 0.25},
        seed=0, n_clusters=len(clusters), tool={"name": "mmseqs", "min_seq_id": 0.3},
    )


def test_build_manifest_regroups_members_per_file():
    manifest = _build(None)
    assert len(manifest["files"]) == 2
    # Every row across both files appears in exactly one split.
    for fi, entry in enumerate(manifest["files"]):
        rows = entry["train"] + entry["val"] + entry["test"]
        assert sorted(rows) == [0, 1, 2, 3]
        assert entry["train"] == sorted(entry["train"])  # sorted on write


def test_manifest_round_trip(tmp_path):
    manifest = _build(tmp_path)
    path = os.path.join(tmp_path, "split_manifest.json")
    mf.write_manifest(path, manifest)
    loaded = mf.read_manifest(path)
    assert loaded == manifest


def test_read_rejects_unknown_schema_version(tmp_path):
    manifest = _build(tmp_path)
    manifest["schema_version"] = 999
    path = os.path.join(tmp_path, "bad.json")
    mf.write_manifest(path, manifest)
    with pytest.raises(ValueError, match="schema_version"):
        mf.read_manifest(path)


def test_fingerprint_changes_on_content_and_order():
    a = [b"ACDE", b"FGHI", b"KLMN"]
    assert mf.compute_fingerprint(a) == mf.compute_fingerprint(list(a))
    assert mf.compute_fingerprint(a) != mf.compute_fingerprint([b"ACDE", b"FGHI"])
    assert mf.compute_fingerprint(a) != mf.compute_fingerprint(a[::-1])
    # str and bytes of the same content fingerprint identically.
    assert mf.compute_fingerprint([b"ACDE"]) == mf.compute_fingerprint(["ACDE"])


def test_validate_file_entry():
    entry = {"path": "/data/f.hdf5", "n_rows": 10, "fingerprint": "fp"}
    mf.validate_file_entry(entry, n_rows=10, fingerprint="fp")
    with pytest.raises(ValueError, match="row count"):
        mf.validate_file_entry(entry, n_rows=11, fingerprint="fp")
    with pytest.raises(ValueError, match="fingerprint"):
        mf.validate_file_entry(entry, n_rows=10, fingerprint="other")

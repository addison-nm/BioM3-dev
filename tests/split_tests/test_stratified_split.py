"""Tests for the JSONL source-stratified split builder (no mmseqs required).

Uses a precomputed clusters TSV so clustering is deterministic and offline.
"""

import json

import pytest

from biom3.split.manifest import compute_fingerprint
from biom3.split.run_stratified_split import build_stratified_split_manifest


def _write_jsonl(path, records):
    with open(path, "w") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")


def _write_singleton_tsv(path, n_rows):
    with open(path, "w") as fh:
        for row in range(n_rows):
            mid = f"0-{row}"
            fh.write(f"{mid}\t{mid}\n")


def _records():
    recs = []
    for i in range(20):
        recs.append({"sequence": "AC" * (i % 5 + 1), "source": "pfam",
                     "sequence_length": 2 * (i % 5 + 1)})
    for i in range(10):
        recs.append({"sequence": "DE" * (i % 4 + 1), "source": "swissprot",
                     "sequence_length": 2 * (i % 4 + 1)})
    return recs


def test_manifest_covers_all_rows_disjointly(tmp_path):
    recs = _records()
    data = tmp_path / "d.jsonl"
    _write_jsonl(data, recs)
    tsv = tmp_path / "clusters.tsv"
    _write_singleton_tsv(tsv, len(recs))

    manifest, result = build_stratified_split_manifest(
        data_path=str(data),
        ratios={"train": 0.7, "val": 0.2, "test": 0.1},
        clusters_tsv=str(tsv),
        min_seq_length=None,
    )
    assert len(manifest["files"]) == 1
    entry = manifest["files"][0]
    all_rows = sorted(entry["train"] + entry["val"] + entry["test"])
    assert all_rows == list(range(len(recs)))
    assert entry["n_rows"] == len(recs)
    assert entry["group"] is None


def test_fingerprint_matches_read_order(tmp_path):
    recs = _records()
    data = tmp_path / "d.jsonl"
    _write_jsonl(data, recs)
    tsv = tmp_path / "clusters.tsv"
    _write_singleton_tsv(tsv, len(recs))

    manifest, _ = build_stratified_split_manifest(
        data_path=str(data), ratios={"train": 0.8, "val": 0.2},
        clusters_tsv=str(tsv), min_seq_length=None,
    )
    expected = compute_fingerprint([r["sequence"] for r in recs])
    assert manifest["files"][0]["fingerprint"] == expected


def test_both_sources_present_in_train_and_val(tmp_path):
    recs = _records()
    data = tmp_path / "d.jsonl"
    _write_jsonl(data, recs)
    tsv = tmp_path / "clusters.tsv"
    _write_singleton_tsv(tsv, len(recs))

    manifest, result = build_stratified_split_manifest(
        data_path=str(data), ratios={"train": 0.7, "val": 0.3},
        clusters_tsv=str(tsv), min_seq_length=None,
    )
    for source in ("pfam", "swissprot"):
        assert result.per_source_counts[source]["train"] > 0
        assert result.per_source_counts[source]["val"] > 0
    assert manifest["per_source_achieved"]["swissprot"]["val"] == pytest.approx(0.3, abs=0.1)


def test_length_filter_excludes_long_rows(tmp_path):
    recs = [
        {"sequence": "AC", "source": "pfam", "sequence_length": 2},
        {"sequence": "A" * 50, "source": "pfam", "sequence_length": 50},
        {"sequence": "DE", "source": "swissprot", "sequence_length": 2},
    ]
    data = tmp_path / "d.jsonl"
    _write_jsonl(data, recs)
    tsv = tmp_path / "clusters.tsv"
    _write_singleton_tsv(tsv, len(recs))

    manifest, _ = build_stratified_split_manifest(
        data_path=str(data), ratios={"train": 1.0},
        clusters_tsv=str(tsv), min_seq_length=8,  # excludes the length-50 row
    )
    entry = manifest["files"][0]
    assigned = entry["train"] + entry["val"] + entry["test"]
    assert 1 not in assigned  # row 1 (length 50) filtered out
    assert sorted(assigned) == [0, 2]


def test_missing_length_field_computes_from_sequence(tmp_path):
    recs = [
        {"sequence": "AC", "source": "pfam"},
        {"sequence": "A" * 50, "source": "pfam"},
    ]
    data = tmp_path / "d.jsonl"
    _write_jsonl(data, recs)
    tsv = tmp_path / "clusters.tsv"
    _write_singleton_tsv(tsv, len(recs))

    manifest, _ = build_stratified_split_manifest(
        data_path=str(data), ratios={"train": 1.0},
        clusters_tsv=str(tsv), min_seq_length=8,
    )
    assigned = manifest["files"][0]["train"]
    assert assigned == [0]  # row 1 computed length 50 > 8, filtered


class TestConnectedComponentsPath:
    def test_edges_merge_into_same_split(self, tmp_path):
        # 30 pfam + 15 swissprot; connect rows 0,1,2 into one component.
        recs = ([{"sequence": "AC" * (i % 6 + 1), "source": "pfam"} for i in range(30)]
                + [{"sequence": "DE" * (i % 4 + 1), "source": "swissprot"} for i in range(15)])
        data = tmp_path / "d.jsonl"
        _write_jsonl(data, recs)
        edges = tmp_path / "edges.tsv"
        with open(edges, "w") as fh:
            fh.write("0-0\t0-1\n")
            fh.write("0-1\t0-2\n")  # transitive: {0,1,2} one cluster

        manifest, _ = build_stratified_split_manifest(
            data_path=str(data), ratios={"train": 0.7, "val": 0.2, "test": 0.1},
            cluster_method="connected_components", edges_tsv=str(edges),
            min_seq_length=None, seed=0,
        )
        e = manifest["files"][0]
        assert manifest["tool"]["method"] == "connected_components"
        # all rows covered, disjoint
        allrows = sorted(e["train"] + e["val"] + e["test"])
        assert allrows == list(range(len(recs)))
        # the connected rows 0,1,2 land in one split (whole cluster)
        where = {r: s for s in ("train", "val", "test") for r in e[s]}
        assert where[0] == where[1] == where[2]

    def test_unknown_method_raises(self, tmp_path):
        data = tmp_path / "d.jsonl"
        _write_jsonl(data, [{"sequence": "AC", "source": "pfam"}])
        with pytest.raises(ValueError, match="unknown cluster_method"):
            build_stratified_split_manifest(
                data_path=str(data), ratios={"train": 1.0},
                cluster_method="bogus", min_seq_length=None,
            )

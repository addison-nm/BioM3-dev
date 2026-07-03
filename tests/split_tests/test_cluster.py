"""Tests for the mmseqs seam: TSV parsing, FASTA writing, missing-binary guard."""

import os

import pytest

from biom3.split import cluster as clu


def test_parse_cluster_tsv_groups_by_representative(tmp_path):
    tsv = os.path.join(tmp_path, "clu.tsv")
    with open(tsv, "w") as fh:
        fh.write("0-0\t0-0\n")
        fh.write("0-0\t0-1\n")
        fh.write("0-2\t0-2\n")
        fh.write("0-3\t0-3\n")
        fh.write("0-3\t1-0\n")
    clusters = clu.parse_cluster_tsv(tsv)
    assert clusters == [["0-0", "0-1"], ["0-2"], ["0-3", "1-0"]]


def test_parse_cluster_tsv_rejects_malformed(tmp_path):
    tsv = os.path.join(tmp_path, "bad.tsv")
    with open(tsv, "w") as fh:
        fh.write("just-one-column\n")
    with pytest.raises(ValueError, match="malformed"):
        clu.parse_cluster_tsv(tsv)


def test_write_pooled_fasta(tmp_path):
    fasta = os.path.join(tmp_path, "p.fasta")
    clu.write_pooled_fasta(fasta, [("0-0", "ACDE"), ("1-2", "FGHI")])
    with open(fasta) as fh:
        content = fh.read()
    assert content == ">0-0\nACDE\n>1-2\nFGHI\n"


def test_missing_mmseqs_raises(monkeypatch, tmp_path):
    monkeypatch.setattr(clu.shutil, "which", lambda _: None)
    with pytest.raises(RuntimeError, match="mmseqs not found"):
        clu.run_mmseqs_easy_cluster(str(tmp_path / "x.fasta"), str(tmp_path))


def test_parse_edges_tsv(tmp_path):
    tsv = os.path.join(tmp_path, "edges.tsv")
    with open(tsv, "w") as fh:
        fh.write("0-0\t0-1\n")
        fh.write("0-1\t0-2\n")
        fh.write("bad-line-no-tab\n")
    assert list(clu.parse_edges_tsv(tsv)) == [("0-0", "0-1"), ("0-1", "0-2")]


def test_connected_components_transitive_merge():
    # 0~1, 1~2 -> one component {0,1,2}; 3 isolated -> singleton
    ids = ["0-0", "0-1", "0-2", "0-3"]
    edges = [("0-0", "0-1"), ("0-1", "0-2")]
    comps = clu.connected_components(ids, edges)
    as_sets = sorted((sorted(c) for c in comps), key=len, reverse=True)
    assert as_sets[0] == ["0-0", "0-1", "0-2"]
    assert ["0-3"] in comps


def test_connected_components_covers_all_ids_once():
    ids = [f"0-{i}" for i in range(6)]
    edges = [("0-0", "0-5"), ("0-2", "0-3")]
    comps = clu.connected_components(ids, edges)
    flat = [m for c in comps for m in c]
    assert sorted(flat) == sorted(ids)  # every id exactly once
    assert len(flat) == len(set(flat))


def test_connected_components_ignores_unknown_ids():
    comps = clu.connected_components(["0-0", "0-1"], [("0-0", "9-9")])
    assert sorted(sorted(c) for c in comps) == [["0-0"], ["0-1"]]


def test_run_mmseqs_search_requires_binary(monkeypatch, tmp_path):
    monkeypatch.setattr(clu.shutil, "which", lambda _: None)
    with pytest.raises(RuntimeError, match="mmseqs not found"):
        clu.run_mmseqs_search(str(tmp_path / "x.fasta"), str(tmp_path))

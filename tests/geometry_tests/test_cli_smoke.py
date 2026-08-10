"""End-to-end smoke test for biom3_fit_manifold and biom3_score_manifold."""

import csv
import os

import numpy as np

from biom3.geometry.io import load_manifold
from biom3.geometry.run_fit_manifold import main as fit_main
from biom3.geometry.run_fit_manifold import parse_arguments as fit_args
from biom3.geometry.run_score_manifold import main as score_main
from biom3.geometry.run_score_manifold import parse_arguments as score_args


def _write_cloud(path, n, d, seed):
    rng = np.random.default_rng(seed)
    array = rng.normal(size=(n, d)) @ (rng.normal(size=(d, d)) / np.sqrt(d))
    np.save(path, array)
    return array


def test_fit_then_score_writes_csv(tmp_path):
    reference_path = os.path.join(tmp_path, "reference.npy")
    queries_path = os.path.join(tmp_path, "queries.npy")
    manifold_path = os.path.join(tmp_path, "manifold.npz")
    ids_path = os.path.join(tmp_path, "ids.txt")
    out_path = os.path.join(tmp_path, "scores.csv")

    _write_cloud(reference_path, 40, 12, seed=0)
    _write_cloud(queries_path, 6, 12, seed=1)
    with open(ids_path, "w") as fh:
        fh.write("\n".join(f"design_{i}" for i in range(6)) + "\n")

    fit_main(fit_args([
        "--reference", reference_path,
        "--method", "gaussian_shrinkage",
        "--label", "synthetic reference",
        "-o", manifold_path,
    ]))
    assert os.path.exists(manifold_path)
    manifold = load_manifold(manifold_path)
    assert manifold.METHOD == "gaussian_shrinkage"
    assert manifold.n_reference == 40
    assert manifold.label == "synthetic reference"

    score_main(score_args([
        "--manifold", manifold_path,
        "--queries", queries_path,
        "--ids", ids_path,
        "--no_norm_check",
        "-o", out_path,
    ]))

    with open(out_path) as fh:
        rows = list(csv.DictReader(fh))
    assert [r["id"] for r in rows] == [f"design_{i}" for i in range(6)]
    assert all(float(r["score"]) > 0 for r in rows)
    assert all(r["band"] in {"below", "within", "above"} for r in rows)


def test_score_writes_npy_and_fit_appends_extension(tmp_path):
    reference_path = os.path.join(tmp_path, "reference.npy")
    queries_path = os.path.join(tmp_path, "queries.npy")
    manifold_stem = os.path.join(tmp_path, "manifold")
    out_path = os.path.join(tmp_path, "scores.npy")

    _write_cloud(reference_path, 30, 8, seed=2)
    queries = _write_cloud(queries_path, 5, 8, seed=3)

    fit_main(fit_args(["--reference", reference_path, "-o", manifold_stem]))
    assert os.path.exists(manifold_stem + ".npz")

    scores = score_main(score_args([
        "--manifold", manifold_stem + ".npz",
        "--queries", queries_path,
        "--no_norm_check",
        "-o", out_path,
    ]))

    saved = np.load(out_path)
    assert saved.shape == (len(queries),)
    np.testing.assert_array_equal(saved, scores)

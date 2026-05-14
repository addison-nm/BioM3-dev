"""Tests for the optional ``track_provenance`` mode of ``load_json_config``."""

import json

import pytest

from biom3.core.helpers import load_json_config


def _write(path, data):
    with open(path, "w") as f:
        json.dump(data, f)


def test_load_json_config_default_unchanged(tmp_path):
    p = tmp_path / "c.json"
    _write(p, {"lr": 0.1, "batch_size": 8})
    result = load_json_config(str(p))
    assert isinstance(result, dict)
    assert result == {"lr": 0.1, "batch_size": 8}


def test_load_json_config_track_provenance_three_tier(tmp_path):
    base = tmp_path / "base.json"
    main = tmp_path / "main.json"
    over = tmp_path / "over.json"

    _write(base, {"lr": 0.001, "batch_size": 32, "from_base_only": "yes"})
    _write(main, {
        "_base_configs": ["base.json"],
        "_overwrite_configs": ["over.json"],
        "lr": 0.01,
        "from_main_only": "yes",
    })
    _write(over, {"batch_size": 64, "from_over_only": "yes"})

    merged, prov = load_json_config(str(main), track_provenance=True)

    assert merged == {
        "lr": 0.01,
        "batch_size": 64,
        "from_base_only": "yes",
        "from_main_only": "yes",
        "from_over_only": "yes",
    }

    assert prov["from_base_only"].endswith("base.json")
    assert prov["from_main_only"].endswith("main.json")
    assert prov["from_over_only"].endswith("over.json")
    # main file overrides base
    assert prov["lr"].endswith("main.json")
    # overwrite file overrides main
    assert prov["batch_size"].endswith("over.json")


def test_load_json_config_track_provenance_no_composition(tmp_path):
    p = tmp_path / "flat.json"
    _write(p, {"foo": 1, "bar": 2})
    merged, prov = load_json_config(str(p), track_provenance=True)
    assert merged == {"foo": 1, "bar": 2}
    assert prov["foo"].endswith("flat.json")
    assert prov["bar"].endswith("flat.json")


def test_load_json_config_circular_still_raises(tmp_path):
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    _write(a, {"_base_configs": ["b.json"], "x": 1})
    _write(b, {"_base_configs": ["a.json"], "y": 2})
    with pytest.raises(ValueError, match="Circular"):
        load_json_config(str(a))

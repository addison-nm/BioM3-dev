"""Named weight-set bundles.

A weight set clumps the Stage 1/2/3 trained checkpoints that belong together
into one JSON file (e.g. ``configs/weights/run1_base.json``) so a whole model
stack can be selected with a single ``--weight_set`` flag instead of three
separate path arguments.

Bundle paths follow the same convention as the rest of the repo's configs:
relative paths resolve against the working directory (run from the repo root),
absolute paths are used as-is. The bundle itself supports config composition
via ``load_json_config``.
"""

from __future__ import annotations

from biom3.core.helpers import load_json_config

WEIGHT_KEYS = ("pencl_weights", "facilitator_weights", "proteoscribe_weights")


def load_weight_set(path):
    """Load a weight-set bundle and return the recognized weight keys.

    Unknown keys in the bundle are ignored; missing keys come back absent.
    """
    cfg = load_json_config(path)
    return {k: cfg[k] for k in WEIGHT_KEYS if k in cfg and cfg[k] is not None}


def merge_weight_set(args, weight_set_path, keys):
    """Fill ``args.<key>`` from a bundle for each key not already set on the CLI.

    Explicit CLI values (anything other than ``None`` / ``"None"``) win over the
    bundle. ``keys`` restricts which weight keys this consumer cares about
    (e.g. the embedding pipeline only needs pencl + facilitator).
    """
    if not weight_set_path or str(weight_set_path) == "None":
        return
    bundle = load_weight_set(weight_set_path)
    for key in keys:
        current = getattr(args, key, None)
        if current in (None, "None") and bundle.get(key):
            setattr(args, key, bundle[key])

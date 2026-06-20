"""Split-manifest schema, (de)serialization, and fingerprint validation.

A manifest records, for each input HDF5 file, which row indices belong to the
train / val / test splits, plus a fingerprint so a rebuilt or reordered file is
detected at load time instead of silently mis-split.

Member keys produced by :mod:`biom3.split.pack` are ``(file_index, row)``
tuples; :func:`build_manifest` regroups them per file and per split.
"""

from __future__ import annotations

import hashlib
import json

from biom3.split.pack import SPLITS

SCHEMA_VERSION = 1


def compute_fingerprint(sequences):
    """Stable fingerprint of an indexable sequence collection.

    Combines the row count with a sample of sequence contents so that a
    different row count, reordering, or content change yields a different
    fingerprint. ``sequences`` may hold ``bytes`` or ``str`` entries (e.g. an
    HDF5 dataset or a Python list).
    """
    n = len(sequences)
    h = hashlib.sha1()
    h.update(str(n).encode())
    if n:
        sample_idx = sorted({0, n - 1, *(int(n * k / 8) for k in range(8))})
        for idx in sample_idx:
            s = sequences[idx]
            if not isinstance(s, bytes):
                s = str(s).encode()
            h.update(s)
    return f"n{n}-{h.hexdigest()[:16]}"


def build_manifest(*, files, pack_result, ratios_target, seed, n_clusters, tool=None):
    """Assemble a manifest dict from packing output.

    Args:
        files: list of per-file metadata dicts, each with ``path``, ``group``,
            ``n_rows``, ``fingerprint``. Order must match the ``file_index``
            used in the member keys.
        pack_result: a :class:`biom3.split.pack.PackResult`.
        ratios_target: mapping split -> target fraction.
        seed: packing seed.
        n_clusters: number of clusters packed.
        tool: optional free-form dict describing the clustering tool/params.
    """
    per_file = []
    for fi, meta in enumerate(files):
        entry = {
            "path": meta["path"],
            "group": meta["group"],
            "n_rows": int(meta["n_rows"]),
            "fingerprint": meta["fingerprint"],
        }
        for split in SPLITS:
            entry[split] = []
        per_file.append(entry)

    for split, member_keys in pack_result.members.items():
        for file_index, row in member_keys:
            per_file[file_index][split].append(int(row))

    for entry in per_file:
        for split in SPLITS:
            entry[split].sort()

    return {
        "schema_version": SCHEMA_VERSION,
        "tool": tool or {},
        "ratios_target": {k: float(v) for k, v in ratios_target.items()},
        "ratios_achieved": {k: float(v) for k, v in pack_result.achieved.items()},
        "counts": {k: int(v) for k, v in pack_result.counts.items()},
        "seed": int(seed),
        "n_clusters": int(n_clusters),
        "files": per_file,
    }


def write_manifest(path, manifest):
    with open(path, "w") as fh:
        json.dump(manifest, fh, indent=2)


def read_manifest(path):
    with open(path) as fh:
        manifest = json.load(fh)
    version = manifest.get("schema_version")
    if version != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported manifest schema_version {version!r}; "
            f"this build understands {SCHEMA_VERSION}"
        )
    return manifest


def validate_file_entry(entry, *, n_rows, fingerprint):
    """Raise if a manifest file entry does not match the opened HDF5 file."""
    if entry["n_rows"] != n_rows:
        raise ValueError(
            f"manifest row count {entry['n_rows']} != file row count {n_rows} "
            f"for {entry['path']}; the dataset has changed since the manifest "
            f"was built — regenerate the split manifest"
        )
    if entry["fingerprint"] != fingerprint:
        raise ValueError(
            f"manifest fingerprint mismatch for {entry['path']}; the dataset "
            f"contents have changed since the manifest was built — regenerate "
            f"the split manifest"
        )

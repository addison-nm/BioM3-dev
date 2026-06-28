"""Schema-driven Dataset over cleaned structured records.

:class:`GeneralizedRecordDataset` maps each cleaned record (a JSON object such
as ``{"sequence": ..., "fields": {key: description, ...}}``) to a dict of named
outputs defined by a *schema*. Each output is produced either by passing a
record key straight through or by running a composition function (see
:mod:`biom3.core.dataloaders.compose_functions`).

Efficiency model
----------------
Composition (dropout / shuffle / label / concatenate) is pure-Python string
work on a handful of fields — microseconds per record, negligible next to the
model. So ``__getitem__`` does *only* that cheap composition and returns plain
Python values (strings, the raw sequence). The expensive, batchable work —
tokenization, moving to device — belongs in a ``collate_fn`` and is run once per
batch, and the whole per-sample step runs inside DataLoader workers that overlap
with GPU compute (use ``num_workers``, ``prefetch_factor``, ``persistent_workers``,
``pin_memory``). If the downstream per-step cost of embedding re-rolled prompts
(e.g. a BioBERT forward for z_c) ever dominates, the same compose functions can
be run offline to expand each record into a finite variant set whose embeddings
are precomputed once — the Dataset and the offline pass share one code path.

Schema
------
A schema maps an output name to an *output spec*, one of:

    "sequence"                       # bare string  -> obj["sequence"]
    {"from": "sequence"}             # passthrough  -> obj["sequence"]
    {"compose": "fields_to_caption", "args": {...}}      # registered by name
    {"compose": my_callable, "args": {...}}              # custom callable
    ("fields_to_caption", {...})     # (name_or_callable, args) pair

A compose target is resolved to a registered function when given a string, or
used directly when given a ``fn(obj, args, rng) -> value`` callable.
"""

import json
import os
import random

import torch
from torch.utils.data import Dataset, get_worker_info

from biom3.core.dataloaders.compose_functions import get_compose_function


def read_jsonl_records(path):
    """Read a JSONL file eagerly into a list of record dicts (blanks skipped).

    Loads the whole file into memory; fine for small/medium datasets. For large
    corpora use :class:`JsonlRecordStore`, which reads lazily from disk and holds
    only a byte-offset index. Both satisfy the ``len`` + integer-``__getitem__``
    interface :class:`GeneralizedRecordDataset` needs, so they are interchangeable.
    """
    records = []
    with open(path, "r") as fh:
        for line_num, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_num}: invalid JSON: {exc}") from exc
    return records


class JsonlRecordStore:
    """Lazy, offset-indexed read-only view over a JSONL file.

    Mirrors the out-of-core approach of Stage 3's ``HDF5Dataset``: at
    construction it scans the file once to record the byte offset of each
    non-blank line, then reads and parses a *single* record per ``__getitem__``
    via ``seek``. Memory cost is the offset array only (~8 bytes/record), not the
    parsed dataset, so it scales to corpora that don't fit in RAM.

    It implements the ``__len__`` + integer-``__getitem__`` sequence protocol, so
    it is a drop-in replacement for the list returned by :func:`read_jsonl_records`
    when constructing a :class:`GeneralizedRecordDataset`.

    Fork/spawn safety: the file handle is opened lazily and re-opened whenever the
    owning pid changes, so each forked DataLoader worker uses its own handle
    rather than sharing (and corrupting) one file position. The handle is dropped
    on pickling so the store can also be sent to spawned workers.

    Optionally, ``scalar_fields`` names top-level keys whose scalar value is
    captured during the single index scan and exposed via :meth:`get_scalar`. This
    lets callers read, say, a precomputed ``sequence_length`` for every record
    (e.g. for length filtering) without a second pass. Capturing fields requires
    parsing each line during indexing, so it trades the offsets-only fast path for
    one full parse at construction; the per-``__getitem__`` read stays single-record.
    """

    def __init__(self, path, scalar_fields=None):
        self.path = str(path)
        self.scalar_fields = list(scalar_fields or [])
        self._offsets, self._scalars = self._build_index(
            self.path, self.scalar_fields
        )
        self._fh = None
        self._fh_pid = None

    @staticmethod
    def _build_index(path, scalar_fields):
        offsets = []
        scalars = {field: [] for field in scalar_fields}
        with open(path, "rb") as fh:
            offset = fh.tell()
            line = fh.readline()
            while line:
                if line.strip():
                    offsets.append(offset)
                    if scalar_fields:
                        obj = json.loads(line)
                        for field in scalar_fields:
                            scalars[field].append(obj.get(field))
                offset = fh.tell()
                line = fh.readline()
        return offsets, scalars

    def get_scalar(self, field):
        """Return the captured values for ``field`` (parallel to record order).

        ``field`` must have been listed in ``scalar_fields`` at construction.
        Records missing the key contribute ``None``.
        """
        try:
            return self._scalars[field]
        except KeyError:
            raise KeyError(
                f"{field!r} was not captured; pass scalar_fields={[field]} "
                f"to JsonlRecordStore (captured: {sorted(self._scalars)})"
            ) from None

    def _handle(self):
        pid = os.getpid()
        if self._fh is None or self._fh_pid != pid:
            if self._fh is not None:
                try:
                    self._fh.close()
                except OSError:
                    pass
            self._fh = open(self.path, "rb")
            self._fh_pid = pid
        return self._fh

    def __len__(self):
        return len(self._offsets)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return [self[i] for i in range(*idx.indices(len(self._offsets)))]
        if idx < 0:
            idx += len(self._offsets)
        if not 0 <= idx < len(self._offsets):
            raise IndexError(idx)
        fh = self._handle()
        fh.seek(self._offsets[idx])
        line = fh.readline()
        try:
            return json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"{self.path}: invalid JSON at record {idx}: {exc}"
            ) from exc

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_fh"] = None
        state["_fh_pid"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def close(self):
        if self._fh is not None:
            self._fh.close()
            self._fh = None
            self._fh_pid = None


def _resolve_compose_target(target):
    if callable(target):
        return target
    if isinstance(target, str):
        return get_compose_function(target)
    raise TypeError(
        "compose target must be a registered name (str) or a callable, "
        f"got {type(target).__name__}"
    )


def _build_resolver(spec):
    """Compile an output spec into a ``resolver(obj, rng) -> value`` callable."""
    if isinstance(spec, str):
        key = spec
        return lambda obj, rng: obj[key]

    if isinstance(spec, (tuple, list)):
        if len(spec) != 2:
            raise ValueError(
                "tuple/list output spec must be (name_or_callable, args), "
                f"got length {len(spec)}"
            )
        fn = _resolve_compose_target(spec[0])
        args = spec[1] or {}
        return lambda obj, rng: fn(obj, args, rng)

    if isinstance(spec, dict):
        has_from = "from" in spec
        has_compose = "compose" in spec
        if has_from == has_compose:
            raise ValueError(
                "dict output spec must have exactly one of 'from' or 'compose', "
                f"got keys {sorted(spec)}"
            )
        if has_from:
            key = spec["from"]
            return lambda obj, rng: obj[key]
        fn = _resolve_compose_target(spec["compose"])
        args = spec.get("args") or {}
        return lambda obj, rng: fn(obj, args, rng)

    raise TypeError(f"unsupported output spec type: {type(spec).__name__}")


class GeneralizedRecordDataset(Dataset):
    """Dataset producing schema-defined outputs from cleaned records.

    Args:
        records: list of record dicts (e.g. from :func:`read_jsonl_records`).
        schema: mapping of output name -> output spec (see module docstring).
            Resolvers are compiled once at construction; unknown compose names
            or malformed specs raise immediately.
        rng: optional ``random.Random`` forcing deterministic composition (for
            tests / single-process reproducibility). When ``None``, each
            DataLoader worker lazily derives its own ``random.Random`` from
            ``torch``'s per-worker seed, and the main process falls back to the
            global ``random`` module.

    ``__getitem__`` returns ``{output_name: value}``, re-rolling stochastic
    composition on every access.

    RNG caveat: with the default (``rng=None``) and ``persistent_workers=True``,
    a worker keeps the same derived RNG across epochs, so augmentation will not
    vary epoch-to-epoch. A future DataModule should reseed workers per epoch
    (``worker_init_fn`` / ``set_epoch``); pass an explicit ``rng`` only for
    determinism.
    """

    def __init__(self, records, schema, rng=None):
        if not schema:
            raise ValueError("schema must define at least one output")
        self.records = records
        self.schema = dict(schema)
        self._resolvers = {
            name: _build_resolver(spec) for name, spec in self.schema.items()
        }
        self._explicit_rng = rng
        self._worker_rng = None

    def __len__(self):
        return len(self.records)

    def collect(self, key, default=None):
        """Return ``rec.get(key, default)`` for every record (e.g. for filtering)."""
        return [rec.get(key, default) for rec in self.records]

    def _get_rng(self):
        if self._explicit_rng is not None:
            return self._explicit_rng
        worker_info = get_worker_info()
        if worker_info is None:
            return random
        if self._worker_rng is None:
            self._worker_rng = random.Random(worker_info.seed)
        return self._worker_rng

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        obj = self.records[idx]
        rng = self._get_rng()
        return {name: resolve(obj, rng) for name, resolve in self._resolvers.items()}


def collate_to_lists(batch):
    """Group a batch of record-dicts into a dict of per-output lists.

    A minimal, tokenizer-agnostic default collate: ``[{a, b}, {a, b}]`` becomes
    ``{a: [...], b: [...]}``. Real training collates tokenize text and stack
    tensors (cf. ``make_seq_caption_collate_fn`` in Stage 3); kept here so the
    Dataset is usable end-to-end without committing to a tokenizer.
    """
    if not batch:
        return {}
    keys = batch[0].keys()
    return {key: [sample[key] for sample in batch] for key in keys}

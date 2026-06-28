"""Generalized, schema-driven data loading for BioM3 training.

This subpackage turns *cleaned* structured records (JSON objects pairing a
sequence with a dict of ``field -> description`` fragments) into model-ready
training samples via declarative, per-output composition. The composition is
re-rolled on every access so dropout/shuffle augmentation varies across epochs.

Modules:
    compose_functions   Registry of pure, standalone composition functions
                        (dropout / shuffle / add_label / concatenate, plus the
                        ready-made ``fields_to_caption``). Importable and usable
                        outside any Dataset (e.g. an offline z_c precompute pass).
    generalized_dataloader
                        ``GeneralizedRecordDataset`` and the JSONL reader that
                        interpret an output schema over those compose functions.
"""

from biom3.core.dataloaders.compose_functions import (
    get_compose_function,
    list_compose_functions,
    register_compose,
)
from biom3.core.dataloaders.generalized_dataloader import (
    GeneralizedRecordDataset,
    JsonlRecordStore,
    collate_to_lists,
    read_jsonl_records,
)

__all__ = [
    "GeneralizedRecordDataset",
    "JsonlRecordStore",
    "collate_to_lists",
    "read_jsonl_records",
    "get_compose_function",
    "list_compose_functions",
    "register_compose",
]

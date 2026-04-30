"""TSV-backed per-sequence reward lookup."""

import csv
from typing import Dict, List

from biom3.backend.device import setup_logger
from biom3.rl.rewards.base import _clean_aa

logger = setup_logger(__name__)


class TsvLookupReward:
    """Look up a per-sequence scalar reward from a TSV/CSV file.

    The file must have a header. ``key_column`` names the column holding
    sequences (cleaned to canonical AA before lookup); ``value_column``
    names the column holding the scalar.

    On a miss (sequence not in the table), behavior is controlled by
    ``miss_strategy``:

    - ``"penalty"`` — return ``miss_value`` (default 0.0). Simple, but
      zero advantage when *every* group member misses.
    - ``"skip"`` — return ``float('nan')``. The trainer treats NaN
      rewards as missing; if the entire group is NaN, the step is
      skipped. (Not yet wired through grpo_train; for now it behaves
      like ``penalty`` with miss_value=NaN.)

    Closed-world tasks (ranking known variants) are the natural fit for
    pure lookup. For novel-sequence generation where most rollouts will
    miss, train a regressor on the TSV and use ``SurrogateReward``
    instead (not in this revision — see docs/grpo_finetuning.md).
    """

    def __init__(
        self,
        path: str,
        key_column: str = "sequence",
        value_column: str = "value",
        delimiter: str = "\t",
        miss_value: float = 0.0,
        miss_strategy: str = "penalty",
        clean: bool = True,
    ):
        if miss_strategy not in ("penalty", "skip"):
            raise ValueError(f"miss_strategy must be 'penalty' or 'skip', got {miss_strategy!r}")
        self.path = path
        self.key_column = key_column
        self.value_column = value_column
        self.delimiter = delimiter
        self.miss_value = float("nan") if miss_strategy == "skip" else float(miss_value)
        self.miss_strategy = miss_strategy
        self.clean = clean
        self._table: Dict[str, float] = {}
        self._loaded = False

    def _load(self):
        if self._loaded:
            return
        with open(self.path, newline="") as f:
            reader = csv.DictReader(f, delimiter=self.delimiter)
            if self.key_column not in (reader.fieldnames or []):
                raise KeyError(f"key_column {self.key_column!r} not in {reader.fieldnames}")
            if self.value_column not in (reader.fieldnames or []):
                raise KeyError(f"value_column {self.value_column!r} not in {reader.fieldnames}")
            for row in reader:
                seq = row[self.key_column].strip()
                if self.clean:
                    seq = _clean_aa(seq)
                if not seq:
                    continue
                try:
                    self._table[seq] = float(row[self.value_column])
                except ValueError:
                    continue
        logger.info("TsvLookupReward: loaded %d entries from %s", len(self._table), self.path)
        self._loaded = True

    def __call__(self, completions: List[str], **kwargs) -> List[float]:
        self._load()
        out: List[float] = []
        for seq in completions:
            key = _clean_aa(seq) if self.clean else seq
            out.append(self._table.get(key, self.miss_value))
        return out

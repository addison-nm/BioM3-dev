"""DataModule for multidomain finetuning over cleaned JSONL records.

Consumes the same artifacts the single-domain generalized path does — a JSONL
built upstream (``bioparsers``) plus a curated split manifest from
``biom3_stratified_cluster_split`` — with three multidomain-specific
differences:

* **Fingerprint on the full-length ``sequence``.** The splitter clusters one
  string per record, so manifest validation must hash that same string. The
  per-domain canvases come out of ``map_domains``, a compose, so they are not a
  passthrough and cannot serve as the fingerprint key.
* **Length filtering on the longest domain**, since each domain occupies its own
  canvas. The splitter's own filter looks at the full-length protein, so records
  can pass splitting and still be dropped here; the count is logged rather than
  silently absorbed.
* **The split manifest is required.** Multidomain assemblies drawn from one
  family pair are heavily homologous, and a random shuffle makes every held-out
  number optimistic.
"""

import torch
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer

from biom3.backend.device import BACKEND_NAME, _XPU
from biom3.core.dataloaders import (
    GeneralizedRecordDataset,
    JsonlRecordStore,
    read_jsonl_records,
)
from biom3.Stage3.multidomain.preprocess import make_multidomain_collate_fn

if BACKEND_NAME == _XPU:
    import lightning as pl
else:
    import pytorch_lightning as pl

from biom3.backend.device import setup_logger

logger = setup_logger(__name__)


class _FullSequenceView:
    """Indexable view over each record's full-length sequence.

    Lets :func:`biom3.split.manifest.compute_fingerprint` sample by row index
    without materializing every record, so it stays cheap over a lazy store.
    """

    def __init__(self, source, key):
        self._source = source
        self._key = key

    def __len__(self):
        return len(self._source)

    def __getitem__(self, idx):
        return self._source[idx][self._key]


def _make_distributed_sampler(dataset, *, shuffle, seed):
    """DistributedSampler with drop_last=True when torch.distributed is up.

    drop_last at the sampler level keeps every rank's shard the same size.
    Without it the leftover samples spread unevenly and the next gradient
    allreduce deadlocks on a genuinely mismatched collective. Pair with
    ``Trainer(use_distributed_sampler=False)``.
    """
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        return torch.utils.data.DistributedSampler(
            dataset,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=shuffle,
            drop_last=True,
            seed=seed,
        )
    return None


class MultiDomainDataModule(pl.LightningDataModule):
    """Schema-driven DataModule producing K-canvas batches.

    ``record_schema`` must map both the sequence and caption outputs through
    ``map_domains``, so each yields a list of K strings per record.
    """

    def __init__(
        self, *,
        jsonl_path,
        record_schema,
        text_model_path,
        text_max_length,
        batch_size,
        num_workers,
        seed,
        diffusion_steps,
        image_size,
        num_domains,
        split_manifest_path,
        sequence_key="sequences",
        caption_key="captions",
        full_sequence_key="sequence",
        domain_length_field="sequence_length",
        domains_key="domains",
        lazy=False,
        needs_unique_sequences=False,
    ):
        super().__init__()
        if split_manifest_path is None:
            raise ValueError(
                "multidomain finetuning requires split_manifest_path; build one "
                "with biom3_stratified_cluster_split so homologous assemblies "
                "cannot straddle the train/val boundary"
            )
        self.jsonl_path = jsonl_path
        self.record_schema = record_schema
        self.text_model_path = text_model_path
        self.text_max_length = text_max_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.diffusion_steps = diffusion_steps
        self.image_size = image_size
        self.num_domains = int(num_domains)
        self.split_manifest_path = split_manifest_path
        self.sequence_key = sequence_key
        self.caption_key = caption_key
        self.full_sequence_key = full_sequence_key
        self.domain_length_field = domain_length_field
        self.domains_key = domains_key
        self.lazy = lazy
        self.needs_unique_sequences = needs_unique_sequences
        self.max_domain_length = diffusion_steps - 2

    def setup(self, stage=None):
        if self.lazy:
            source = JsonlRecordStore(self.jsonl_path)
        else:
            source = read_jsonl_records(self.jsonl_path)

        dataset = GeneralizedRecordDataset(source, schema=self.record_schema, rng=None)
        self.text_tokenizer = AutoTokenizer.from_pretrained(self.text_model_path)
        self._collate_fn = make_multidomain_collate_fn(
            text_tokenizer=self.text_tokenizer,
            text_max_length=self.text_max_length,
            image_size=self.image_size,
            num_domains=self.num_domains,
            sequence_key=self.sequence_key,
            caption_key=self.caption_key,
            include_sequences=self.needs_unique_sequences,
        )

        lengths = self._domain_lengths(source)
        train_idx, val_idx, test_idx = self._manifest_split(source)
        train_idx = self._filter_by_length(train_idx, lengths, "train")
        val_idx = self._filter_by_length(val_idx, lengths, "val")

        self.train_dataset = Subset(dataset, train_idx)
        self.val_dataset = Subset(dataset, val_idx)
        self.split_info = [{
            "path": self.jsonl_path,
            "train_indices": train_idx,
            "val_indices": val_idx,
            "test_indices": test_idx,
        }]

        if self.needs_unique_sequences:
            # Keys must match what the collate emits: the per-domain canvas
            # sequences, flattened in the same row-major order.
            seqs = (
                domain["sequence"]
                for i in list(train_idx) + list(val_idx)
                for domain in source[i][self.domains_key]
            )
            self._unique_sequences = list(dict.fromkeys(seqs))
        else:
            self._unique_sequences = None

        logger.info(
            "Loaded multidomain dataset from %s (K=%d; %d train, %d val, "
            "%d held-out test)",
            self.jsonl_path, self.num_domains,
            len(train_idx), len(val_idx), len(test_idx),
        )

    def unique_sequences(self):
        """Deduplicated train+val *domain* sequences, for the z_p precompute."""
        if self._unique_sequences is None:
            raise RuntimeError(
                "unique_sequences() requires needs_unique_sequences=True, which "
                "the runner sets whenever train_alpha puts weight on z_p"
            )
        return self._unique_sequences

    def _domain_lengths(self, source):
        """Longest domain length per record — each domain gets its own canvas."""
        lengths = []
        for record in source:
            domains = record[self.domains_key]
            if len(domains) != self.num_domains:
                raise ValueError(
                    f"{self.jsonl_path}: record has {len(domains)} domains but "
                    f"the run is configured for {self.num_domains}"
                )
            per_domain = []
            for domain in domains:
                length = domain.get(self.domain_length_field)
                if length is None:
                    length = len(domain["sequence"].replace("-", ""))
                per_domain.append(int(length))
            lengths.append(max(per_domain))
        return lengths

    def _manifest_split(self, source):
        """Row indices from the curated split manifest, fingerprint-validated.

        Hashes the full-length ``sequence`` — the same string
        ``biom3_stratified_cluster_split`` clustered on — so the manifest is
        bound to this exact JSONL.
        """
        from biom3.split import manifest as split_manifest

        manifest = split_manifest.read_manifest(self.split_manifest_path)
        if len(manifest["files"]) != 1:
            raise ValueError(
                f"split manifest describes {len(manifest['files'])} file(s) but "
                "the multidomain dataloader reads a single JSONL; regenerate it"
            )
        entry = manifest["files"][0]
        split_manifest.validate_file_entry(
            entry,
            n_rows=len(source),
            fingerprint=split_manifest.compute_fingerprint(
                _FullSequenceView(source, self.full_sequence_key)
            ),
        )
        logger.info("Using curated split manifest %s", self.split_manifest_path)
        return entry["train"], entry["val"], entry["test"]

    def _filter_by_length(self, indices, lengths, label):
        kept = [idx for idx in indices if lengths[idx] <= self.max_domain_length]
        dropped = len(indices) - len(kept)
        if dropped:
            logger.info(
                "Dropped %d/%d %s records whose longest domain exceeds %d residues; "
                "the split manifest counted them because it filters on the "
                "full-length protein",
                dropped, len(indices), label, self.max_domain_length,
            )
        return kept

    def train_dataloader(self):
        sampler = _make_distributed_sampler(
            self.train_dataset, shuffle=True, seed=self.seed)
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=(sampler is None),
            sampler=sampler,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
            drop_last=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        sampler = _make_distributed_sampler(
            self.val_dataset, shuffle=False, seed=self.seed)
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            sampler=sampler,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
            drop_last=True,
            pin_memory=True,
        )

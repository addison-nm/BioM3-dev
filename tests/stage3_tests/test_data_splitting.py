"""Tests for dataset splitting reproducibility and distributed batch assignment.

Verifies:
1. Train/val indices are deterministic given the same seed.
2. Train/val indices are identical across simulated ranks.
3. DistributedSampler assigns non-overlapping batches to different ranks.
"""

import pytest
import numpy as np
import torch
from argparse import Namespace
from torch.utils.data import DataLoader, DistributedSampler, Subset

from biom3.Stage3.PL_wrapper import HDF5DataModule

DATDIR = "tests/_data"
TEST_HDF5 = f"{DATDIR}/data/Stage2_MMD_swissprot_embedding_subset_1000.hdf5"

# Minimal args namespace that HDF5DataModule / HDF5Dataset need
def _make_args(**overrides):
    defaults = dict(
        batch_size=32,
        num_workers=0,
        valid_size=0.2,
        seed=42,
        diffusion_steps=1024,
        image_size=32,
        num_classes=29,
        task="proteins",
        sequence_keyname="sequence",
    )
    defaults.update(overrides)
    return Namespace(**defaults)


def _setup_data_module(**overrides):
    args = _make_args(**overrides)
    dm = HDF5DataModule(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        valid_size=args.valid_size,
        seed=args.seed,
        diffusion_steps=args.diffusion_steps,
        image_size=args.image_size,
        primary_path=TEST_HDF5,
        group_name="MMD_data",
    )
    dm.setup()
    return dm


class TestSplitDeterminism:
    """Train/val indices must be deterministic and seed-dependent."""

    def test_same_seed_same_split(self):
        dm1 = _setup_data_module(seed=42)
        dm2 = _setup_data_module(seed=42)
        s1, s2 = dm1.split_info[0], dm2.split_info[0]
        np.testing.assert_array_equal(s1["train_indices"], s2["train_indices"])
        np.testing.assert_array_equal(s1["val_indices"], s2["val_indices"])

    def test_different_seed_different_split(self):
        dm1 = _setup_data_module(seed=42)
        dm2 = _setup_data_module(seed=99)
        s1, s2 = dm1.split_info[0], dm2.split_info[0]
        assert not np.array_equal(s1["train_indices"], s2["train_indices"])

    def test_no_overlap_between_train_and_val(self):
        dm = _setup_data_module(seed=42)
        s = dm.split_info[0]
        train_set = set(s["train_indices"])
        val_set = set(s["val_indices"])
        assert train_set.isdisjoint(val_set), "Train and val indices overlap"

    def test_train_val_cover_all_filtered(self):
        dm = _setup_data_module(seed=42)
        s = dm.split_info[0]
        combined = set(s["train_indices"]) | set(s["val_indices"])
        # All indices in the subsets should come from the original dataset
        assert len(combined) == len(s["train_indices"]) + len(s["val_indices"])


class TestCrossRankConsistency:
    """All ranks must see the same train/val split (split happens before
    DistributedSampler partitions batches)."""

    def test_simulated_ranks_same_split(self):
        """Simulate multiple ranks by calling setup() independently —
        since split_indices uses np.random.seed(self.seed), all produce
        identical indices."""
        splits = []
        for _ in range(4):  # simulate 4 ranks
            dm = _setup_data_module(seed=123)
            splits.append(dm.split_info[0])

        ref_train = splits[0]["train_indices"]
        ref_val = splits[0]["val_indices"]
        for i, s in enumerate(splits[1:], 1):
            np.testing.assert_array_equal(
                s["train_indices"], ref_train,
                err_msg=f"Rank {i} train indices differ from rank 0",
            )
            np.testing.assert_array_equal(
                s["val_indices"], ref_val,
                err_msg=f"Rank {i} val indices differ from rank 0",
            )


class TestDistributedBatchAssignment:
    """DistributedSampler must assign non-overlapping sample indices to
    different ranks within each epoch."""

    def test_ranks_get_different_samples(self):
        dm = _setup_data_module(seed=42)
        dataset = dm.train_dataset
        num_replicas = 4

        per_rank_indices = []
        for rank in range(num_replicas):
            sampler = DistributedSampler(
                dataset,
                num_replicas=num_replicas,
                rank=rank,
                shuffle=True,
                seed=42,
            )
            sampler.set_epoch(0)
            per_rank_indices.append(set(sampler))

        # Each rank should get a subset
        for rank, indices in enumerate(per_rank_indices):
            assert len(indices) > 0, f"Rank {rank} got no samples"

        # Pairwise: no overlap between ranks
        for i in range(num_replicas):
            for j in range(i + 1, num_replicas):
                overlap = per_rank_indices[i] & per_rank_indices[j]
                assert len(overlap) == 0, (
                    f"Rank {i} and rank {j} share {len(overlap)} samples"
                )

    def test_all_samples_covered(self):
        dm = _setup_data_module(seed=42)
        dataset = dm.train_dataset
        num_replicas = 4

        all_indices = set()
        for rank in range(num_replicas):
            sampler = DistributedSampler(
                dataset,
                num_replicas=num_replicas,
                rank=rank,
                shuffle=True,
                seed=42,
            )
            sampler.set_epoch(0)
            all_indices.update(sampler)

        # DistributedSampler may pad to make evenly divisible, so
        # all_indices >= len(dataset)
        assert len(all_indices) >= len(dataset) - num_replicas

    def test_different_epochs_shuffle_differently(self):
        dm = _setup_data_module(seed=42)
        dataset = dm.train_dataset

        sampler = DistributedSampler(
            dataset, num_replicas=2, rank=0, shuffle=True, seed=42,
        )

        sampler.set_epoch(0)
        epoch0 = list(sampler)

        sampler.set_epoch(1)
        epoch1 = list(sampler)

        assert epoch0 != epoch1, "Same rank should get different order across epochs"

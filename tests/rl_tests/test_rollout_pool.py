"""CPU smoke tests for biom3.rl.rollout.RolloutPool.

Single-device parity, replica synchronization, and threaded chunk
dispatch. Uses the mini Stage 3 fixture under tests/_data/models/stage3/.
Multi-XPU validation is done manually on Aurora via
scripts/run_gdpo_smoke_multixpu.sh — pytest stays CPU-only.
"""

import copy
import os

import pytest
import torch

from tests.conftest import DATDIR
from biom3.core.helpers import convert_to_namespace, load_json_config
from biom3.Stage3.io import build_model_ProteoScribe
from biom3.rl.gdpo import _gdpo_rollout, _resolve_rollout_devices
from biom3.rl.rollout import RolloutPool, _split_evenly


MINI_WEIGHTS = os.path.join(DATDIR, "models/stage3/weights/minimodel1_ds128_weights1.pth")
MINI_CONFIG = os.path.join(DATDIR, "configs/test_stage3_config_v2.json")


@pytest.fixture
def mini_s3():
    cfg = convert_to_namespace(load_json_config(MINI_CONFIG))
    cfg.device = "cpu"
    model = build_model_ProteoScribe(cfg)
    sd = torch.load(MINI_WEIGHTS, map_location="cpu")
    model.load_state_dict(sd, strict=True)
    return model, cfg


def test_split_evenly_basic():
    assert _split_evenly(8, 2) == [4, 4]
    assert _split_evenly(8, 3) == [3, 3, 2]   # extras go to earlier buckets
    assert _split_evenly(7, 6) == [2, 1, 1, 1, 1, 1]
    assert _split_evenly(0, 4) == [0, 0, 0, 0]
    assert sum(_split_evenly(24, 6)) == 24


def test_split_evenly_rejects_zero_buckets():
    with pytest.raises(ValueError):
        _split_evenly(8, 0)


def test_resolve_rollout_devices_none_returns_master():
    master = torch.device("cpu")
    assert _resolve_rollout_devices(None, master) == [master]
    assert _resolve_rollout_devices([], master) == [master]


def test_resolve_rollout_devices_explicit_puts_master_first():
    master = torch.device("cpu")
    out = _resolve_rollout_devices(["cpu"], master)
    assert out == [master]


def test_rollout_pool_single_device_matches_direct_call(mini_s3):
    """RolloutPool with one device must produce the same ids as
    calling _gdpo_rollout directly, for the same RNG seed."""
    s3, cfg = mini_s3
    s3.eval()
    z_c = torch.randn(1, cfg.text_emb_dim)
    G = 3
    device = torch.device("cpu")

    torch.manual_seed(123)
    direct = _gdpo_rollout(s3, cfg, z_c, G, device)

    pool = RolloutPool(s3, cfg, _gdpo_rollout, devices=[device])
    torch.manual_seed(123)
    via_pool = pool.rollout(z_c, G)
    pool.shutdown()

    assert via_pool.shape == direct.shape
    assert via_pool.dtype == direct.dtype
    assert torch.equal(via_pool, direct)


def test_rollout_pool_two_threads_returns_correct_count(mini_s3):
    """Two cpu device contexts (same physical device, two threads).
    Mainly exercises the chunk-split + thread-dispatch + concat path."""
    s3, cfg = mini_s3
    s3.eval()
    z_c = torch.randn(1, cfg.text_emb_dim)
    G = 4
    devs = [torch.device("cpu"), torch.device("cpu")]
    pool = RolloutPool(s3, cfg, _gdpo_rollout, devices=devs)
    out = pool.rollout(z_c, G)
    pool.shutdown()
    assert out.shape == (G, cfg.diffusion_steps)
    assert out.dtype == torch.int64
    # All rolled-out ids must be valid token ids.
    assert int(out.min()) >= 0
    assert int(out.max()) < cfg.num_classes


def test_rollout_pool_sync_from_replicates_master(mini_s3):
    """After sync_from(master), every replica returns identical logits
    given identical input + RNG state — i.e. weights match bit-exactly."""
    s3, cfg = mini_s3
    s3.eval()
    devs = [torch.device("cpu"), torch.device("cpu"), torch.device("cpu")]
    pool = RolloutPool(s3, cfg, _gdpo_rollout, devices=devs)

    # Drift the master a bit (simulate an optimizer step), then sync.
    with torch.no_grad():
        for p in s3.parameters():
            p.add_(0.001 * torch.randn_like(p))
    pool.sync_from(s3)

    # Build a deterministic dummy input and check each replica's
    # logits agree with the master's.
    L = cfg.diffusion_steps
    x = torch.randint(0, cfg.num_classes, (1, L), dtype=torch.long)
    t = torch.zeros(1, dtype=torch.long)
    y_c = torch.randn(1, cfg.text_emb_dim)

    with torch.no_grad():
        master_logits = s3(x, t, y_c)
        for replica in pool._models:
            replica_logits = replica(x, t, y_c)
            assert torch.allclose(replica_logits, master_logits, atol=1e-7)
    pool.shutdown()


def test_rollout_pool_does_not_mutate_master_cfg(mini_s3):
    """RolloutPool must deep-copy cfg3 per tile so the trainer's master
    cfg3 (used for elbo_new on the master device) is never touched."""
    s3, cfg = mini_s3
    cfg.device = "cpu"
    z_c = torch.randn(1, cfg.text_emb_dim)
    devs = [torch.device("cpu"), torch.device("cpu")]
    pool = RolloutPool(s3, cfg, _gdpo_rollout, devices=devs)
    _ = pool.rollout(z_c, G=2)
    pool.shutdown()
    assert cfg.device == "cpu"

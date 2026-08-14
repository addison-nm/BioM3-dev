"""Derived single-file weights, and the artifact generation actually loads.

A training run leaves either a sharded DeepSpeed directory or a single .ckpt,
and neither is what downstream tooling reaches for. The runner derives two
files; only one of them carries the architecture spec, and these tests pin which
is which so the wrong one cannot quietly become the documented artifact.
"""

import argparse
import json
import os

import pytest
import torch

from biom3.Stage3.multidomain import trainer as md_trainer
from biom3.Stage3.multidomain.io import (
    MultiDomainSpec,
    build_multidomain_from_checkpoint,
    consolidate_checkpoint,
    state_dict_fingerprint,
)
from biom3.Stage3.multidomain.run_multidomain_finetuning import save_derived_weights

from .conftest import EMB_DIM, NUM_DOMAINS, SEQ_LEN, make_composed


class _FakeCallback:
    def __init__(self, best_model_path, monitor="val_loss", best_model_score=0.5):
        self.best_model_path = best_model_path
        self.monitor = monitor
        self.best_model_score = torch.tensor(best_model_score)


def _write_lightning_checkpoint(path, model, spec):
    torch.save({
        "state_dict": {f"model.{k}": v for k, v in model.state_dict().items()},
        "hyper_parameters": {
            "multidomain_spec": spec.to_dict(),
            "multidomain_fingerprint": state_dict_fingerprint(model),
        },
    }, path)


@pytest.fixture
def spec(tiny_args):
    return MultiDomainSpec.from_args(
        tiny_args, NUM_DOMAINS, domain_ids=("PF00501", "PF13193"))


# ── consolidation ─────────────────────────────────────────────────────────


def test_consolidate_copies_a_single_file(tmp_path, tiny_args, spec):
    model = make_composed(tiny_args)
    src = tmp_path / "best.ckpt"
    _write_lightning_checkpoint(src, model, spec)

    dst = tmp_path / "single_model.best.pth"
    consolidate_checkpoint(str(src), str(dst))

    assert dst.exists()
    restored = torch.load(dst, map_location="cpu", weights_only=False)
    assert restored["hyper_parameters"]["multidomain_spec"] == spec.to_dict()


def test_consolidated_file_is_loadable_by_generation(tmp_path, tiny_args, spec):
    """The whole point: the derived artifact must round-trip through the loader."""
    model = make_composed(tiny_args)
    with torch.no_grad():
        for param in model.coupling.parameters():
            param.add_(0.05)
    src = tmp_path / "best.ckpt"
    _write_lightning_checkpoint(src, model, spec)

    dst = tmp_path / "single_model.best.pth"
    consolidate_checkpoint(str(src), str(dst))

    restored, restored_spec = build_multidomain_from_checkpoint(
        str(dst), template_args=tiny_args)
    assert restored_spec == spec

    model.eval()
    restored.eval()
    torch.manual_seed(0)
    x = torch.randint(0, 29, (1, NUM_DOMAINS, SEQ_LEN))
    t = torch.randint(0, SEQ_LEN, (1,)).float()
    y = torch.randn(1, NUM_DOMAINS, EMB_DIM)
    with torch.no_grad():
        assert torch.equal(model(x, t, y), restored(x, t, y))


# ── derived artifacts ─────────────────────────────────────────────────────


def _run_save(tmp_path, tiny_args, spec, **overrides):
    model = make_composed(tiny_args)
    ckpt_dir = tmp_path / "checkpoints"
    artifacts_dir = tmp_path / "artifacts"
    ckpt_dir.mkdir()
    artifacts_dir.mkdir()

    best = ckpt_dir / "best-001.ckpt"
    _write_lightning_checkpoint(best, model, spec)

    args = argparse.Namespace(**overrides)
    path = save_derived_weights(
        args, str(ckpt_dir), str(artifacts_dir), _FakeCallback(str(best)))
    return model, ckpt_dir, artifacts_dir, path


def test_both_artifacts_are_emitted(tmp_path, tiny_args, spec):
    _, ckpt_dir, artifacts_dir, path = _run_save(tmp_path, tiny_args, spec)

    assert path == str(ckpt_dir / "single_model.best.pth")
    assert (ckpt_dir / "single_model.best.pth").exists()
    assert (ckpt_dir / "state_dict.best.pth").exists()
    assert (artifacts_dir / "state_dict.best.pth").exists()
    assert (artifacts_dir / "checkpoint_summary.json").exists()


def test_the_generation_artifact_carries_the_spec(tmp_path, tiny_args, spec):
    """single_model.best.pth loads; state_dict.best.pth deliberately does not.

    The bare state dict is kept for parity with Stage 3's convention, but it has
    no hyper_parameters and therefore no architecture. Asserting the asymmetry
    keeps the docs honest about which file generation takes.
    """
    _, ckpt_dir, _, path = _run_save(tmp_path, tiny_args, spec)

    _, restored_spec = build_multidomain_from_checkpoint(
        path, template_args=tiny_args)
    assert restored_spec == spec

    with pytest.raises(ValueError, match="carries no multidomain_spec"):
        build_multidomain_from_checkpoint(
            str(ckpt_dir / "state_dict.best.pth"), template_args=tiny_args)


def test_state_dict_artifact_holds_bare_weights(tmp_path, tiny_args, spec):
    model, ckpt_dir, _, _ = _run_save(tmp_path, tiny_args, spec)
    saved = torch.load(ckpt_dir / "state_dict.best.pth", map_location="cpu",
                       weights_only=False)
    assert set(saved) == set(model.state_dict())
    assert not any(k.startswith("model.") for k in saved)
    for key, tensor in model.state_dict().items():
        assert torch.equal(tensor, saved[key])


def test_summary_records_both_paths(tmp_path, tiny_args, spec):
    _, ckpt_dir, artifacts_dir, path = _run_save(tmp_path, tiny_args, spec)
    summary = json.loads((artifacts_dir / "checkpoint_summary.json").read_text())
    assert summary["generation_artifact"] == path
    assert summary["state_dict"] == str(ckpt_dir / "state_dict.best.pth")
    assert summary["monitor"] == "val_loss"


def test_no_best_checkpoint_is_not_fatal(tmp_path, tiny_args):
    """A run killed before any validation has no best path; that must not raise."""
    args = argparse.Namespace()
    assert save_derived_weights(
        args, str(tmp_path), str(tmp_path), _FakeCallback("")) is None
    assert save_derived_weights(
        args, str(tmp_path), str(tmp_path),
        _FakeCallback(str(tmp_path / "absent.ckpt"))) is None


# ── mid-run sync wiring ───────────────────────────────────────────────────


def test_sync_callback_is_attached_when_a_sync_fn_is_given(tmp_path):
    from biom3.Stage3.callbacks import BestArtifactSyncCallback

    args = argparse.Namespace(save_top_k=1, checkpoint_every_n_epochs=None)
    without = md_trainer.build_callbacks(args, str(tmp_path))
    assert not any(isinstance(c, BestArtifactSyncCallback) for c in without)

    with_sync = md_trainer.build_callbacks(
        args, str(tmp_path), sync_fn=lambda **kw: None)
    assert any(isinstance(c, BestArtifactSyncCallback) for c in with_sync)


def test_primary_callback_ignores_the_periodic_snapshot(tmp_path):
    """The periodic callback is also a ModelCheckpoint but carries no monitor."""
    args = argparse.Namespace(save_top_k=1, checkpoint_every_n_epochs=5)
    callbacks = md_trainer.build_callbacks(args, str(tmp_path))

    trainer = argparse.Namespace(callbacks=callbacks)
    primary = md_trainer.primary_checkpoint_callback(trainer)
    assert primary is not None
    assert primary.monitor == "val_loss"


def test_primary_callback_is_none_without_monitored_callbacks():
    trainer = argparse.Namespace(callbacks=[])
    assert md_trainer.primary_checkpoint_callback(trainer) is None

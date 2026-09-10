"""--checkpoint_every_n_epochs / _steps must reach ModelCheckpoint as ints,
whether they come from the CLI (strings) or a JSON config (ints)."""
import contextlib
import io
import json

import pytest

from biom3.Stage1.run_PL_training import retrieve_all_args
from biom3.Stage3.callbacks import build_checkpoint_callbacks


def _resolve(tmp_path, cfg_extra, cli=()):
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps({"dataset_type": "pfam", "output_root": str(tmp_path), **cfg_extra}))
    with contextlib.redirect_stdout(io.StringIO()):
        return retrieve_all_args(["--config_path", str(cfg), "--run_id", "t", *cli])


@pytest.mark.parametrize("cfg_extra,cli,want", [
    ({}, ("--checkpoint_every_n_epochs", "1"), 1),        # CLI string
    ({"checkpoint_every_n_epochs": 1}, (), 1),            # JSON int
    ({"checkpoint_every_n_epochs": "None"}, (), None),
    ({}, (), None),
])
def test_epochs_cadence_is_int_or_none(tmp_path, cfg_extra, cli, want):
    a = _resolve(tmp_path, cfg_extra, cli)
    assert a.checkpoint_every_n_epochs == want and type(a.checkpoint_every_n_epochs) is type(want)


def test_steps_cadence_from_json_int(tmp_path):
    assert _resolve(tmp_path, {"checkpoint_every_n_steps": 500}).checkpoint_every_n_steps == 500


def test_periodic_callback_gets_every_n_epochs_1(tmp_path):
    a = _resolve(tmp_path, {"checkpoint_every_n_epochs": 1})
    _, periodic = build_checkpoint_callbacks(
        checkpoint_dir=str(tmp_path / "ckpt"),
        periodic_every_n_epochs=a.checkpoint_every_n_epochs,
        periodic_max_keep=a.checkpoint_periodic_max_keep)
    assert periodic is not None
    assert periodic._every_n_epochs == 1 and periodic.save_top_k == -1

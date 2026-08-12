"""Runner, trainer-construction and config tests.

``test_strategy_flags_*`` pin the distributed-strategy kwargs. Each encodes a
distinct Aurora failure, and losing one would surface as a hang under oneCCL
rather than a test failure — so they are asserted explicitly rather than trusted
to survive edits.
"""

import json

import pytest

from biom3.Stage3.multidomain import trainer as md_trainer
from biom3.Stage3.multidomain.run_multidomain_finetuning import (
    _coerce_args, parse_arguments,
)


# ── distributed strategy ──────────────────────────────────────────────────


def test_deepspeed_strategy_flags():
    strategy = md_trainer.build_deepspeed_strategy()
    config = strategy.config
    assert config["zero_optimization"]["stage"] == 2
    assert config["zero_optimization"]["contiguous_gradients"] is True


def test_ddp_strategy_flags():
    strategy = md_trainer.build_ddp_strategy()
    assert strategy._ddp_kwargs["static_graph"] is True
    assert strategy._ddp_kwargs["gradient_as_bucket_view"] is True


def test_xpu_selects_xccl_and_disables_overlap_comm(monkeypatch):
    """On XPU: native xccl backend, and no overlapping gradient reduction.

    overlap_comm produces nondeterministic bucket ordering across ranks on
    oneCCL, which deadlocks on a mismatched collective.
    """
    monkeypatch.setattr(md_trainer, "BACKEND_NAME", md_trainer._XPU)
    deepspeed = md_trainer.build_deepspeed_strategy()
    assert deepspeed.config["zero_optimization"]["overlap_comm"] is False
    assert deepspeed._process_group_backend == "xccl"
    assert md_trainer.build_ddp_strategy()._process_group_backend == "xccl"


def test_non_xpu_keeps_overlap_comm_and_default_backend(monkeypatch):
    monkeypatch.setattr(md_trainer, "BACKEND_NAME", "cuda")
    deepspeed = md_trainer.build_deepspeed_strategy()
    assert deepspeed.config["zero_optimization"]["overlap_comm"] is True
    assert deepspeed._process_group_backend is None


def test_single_process_run_uses_auto_strategy():
    assert md_trainer.build_strategy("deepspeed", devices=1, num_nodes=1) == "auto"


def test_multi_device_selects_the_named_strategy():
    strategy = md_trainer.build_strategy("ddp", devices=4, num_nodes=1)
    assert isinstance(strategy, type(md_trainer.build_ddp_strategy()))


def test_unknown_strategy_is_rejected():
    with pytest.raises(ValueError, match="unknown distributed_strategy"):
        md_trainer.build_strategy("horovod", devices=4, num_nodes=1)


def test_resume_checkpoint_is_last_ckpt(tmp_path):
    assert md_trainer.find_resume_checkpoint(str(tmp_path)) is None
    (tmp_path / "last.ckpt").write_text("x")
    assert md_trainer.find_resume_checkpoint(str(tmp_path)).endswith("last.ckpt")


def test_callbacks_include_a_periodic_checkpoint(tmp_path):
    import argparse
    args = argparse.Namespace(save_top_k=3, checkpoint_every_n_epochs=10)
    callbacks = md_trainer.build_callbacks(args, str(tmp_path))
    periodic = [c for c in callbacks
                if getattr(c, "_every_n_epochs", None) == 10]
    assert periodic, "a long run needs periodic checkpoints, not only the best one"


# ── argument coercion ─────────────────────────────────────────────────────


def _base_argv(**overrides):
    argv = [
        "--expert_weights", "a.bin", "b.bin",
        "--domain_ids", "PF00501", "PF13193",
        "--finetune_data_path", "data.jsonl",
        "--split_manifest_path", "split.json",
    ]
    for key, value in overrides.items():
        argv += [f"--{key}", str(value)]
    return argv


def test_num_domains_defaults_to_the_expert_count():
    args = parse_arguments(_base_argv())
    assert args.num_domains == 2


def test_mismatched_num_domains_is_rejected():
    with pytest.raises(ValueError, match="expert weight paths"):
        parse_arguments(_base_argv(num_domains=3))


def test_mismatched_domain_ids_is_rejected():
    argv = ["--expert_weights", "a.bin", "b.bin",
            "--domain_ids", "only_one",
            "--finetune_data_path", "d.jsonl",
            "--split_manifest_path", "s.json"]
    with pytest.raises(ValueError, match="--domain_ids has 1 entries"):
        parse_arguments(argv)


def test_missing_expert_weights_is_rejected():
    with pytest.raises(ValueError, match="--expert_weights is required"):
        parse_arguments(["--finetune_data_path", "d.jsonl"])


def test_string_booleans_from_config_are_coerced():
    args = parse_arguments(_base_argv(train_experts="true",
                                      audit_additive_null="False"))
    assert args.train_experts is True
    assert args.audit_additive_null is False


def test_record_schema_json_string_is_parsed():
    schema = {"sequences": {"from": "sequence"}}
    args = parse_arguments(_base_argv(record_schema=json.dumps(schema)))
    assert args.record_schema == schema


def test_domain_ids_default_when_absent():
    argv = ["--expert_weights", "a.bin", "b.bin",
            "--finetune_data_path", "d.jsonl",
            "--split_manifest_path", "s.json"]
    args = parse_arguments(argv)
    assert args.domain_ids == ["domain0", "domain1"]


def test_unconstrained_expert_training_warns():
    """Component B with no prior lets the experts drift off their fold.

    A handler is attached to the runner's own logger: biom3 loggers set
    propagate=False and bind their stream at import time, so neither caplog nor
    capfd observes these records.
    """
    import logging

    from biom3.Stage3.multidomain import run_multidomain_finetuning as runner

    records = []

    class _Collect(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    handler = _Collect()
    runner.logger.addHandler(handler)
    try:
        parse_arguments(_base_argv(train_experts="true", expert_prior_lambda=0.0))
    finally:
        runner.logger.removeHandler(handler)

    assert any("unconstrained" in message for message in records)


# ── shipped configs ───────────────────────────────────────────────────────


CONFIG_DIR = "configs/stage3_multidomain"


def test_component_a_config_loads_and_is_consistent():
    from biom3.core.helpers import load_json_config

    config = load_json_config(f"{CONFIG_DIR}/finetune_multidomain_v1.json")
    assert config["num_domains"] == len(config["expert_weights"])
    assert config["num_domains"] == len(config["domain_ids"])
    assert config["train_experts"] is False
    for key in ("sequences", "captions"):
        assert config["record_schema"][key]["compose"] == "map_domains"
        assert config["record_schema"][key]["args"]["expect_k"] == config["num_domains"]
    assert config["split_manifest_path"]


def test_component_ab_config_enables_the_prior():
    from biom3.core.helpers import load_json_config

    config = load_json_config(f"{CONFIG_DIR}/finetune_multidomain_ab_v1.json")
    assert config["train_experts"] is True
    assert config["expert_prior_lambda"] > 0
    # Inherited from the Component-A base.
    assert config["num_domains"] == 2
    assert config["record_schema"]["captions"]["compose"] == "map_domains"


def test_model_base_matches_the_expert_architecture():
    from biom3.core.helpers import load_json_config

    config = load_json_config(f"{CONFIG_DIR}/models/_base_ProteoScribe_multidomain.json")
    assert config["transformer_dim"] == 512
    assert config["transformer_depth"] == 16
    assert config["diffusion_steps"] == config["image_size"] ** 2
    assert config["transformer_dim"] % config["transformer_heads"] == 0

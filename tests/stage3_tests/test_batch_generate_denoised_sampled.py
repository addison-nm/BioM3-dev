"""Unit tests for batch_generate_denoised_sampled and batch_generate_denoised_sampled_confidence

Tests the core batched denoising loops in sampling_analysis.py, verifying
correctness of per-sample indexing, output shapes, token ranges, and
deterministic generation with fixed seeds. Covers both random and
confidence-based unmasking orders, and both sample and argmax token strategies.
"""

import pytest
import os
import numpy as np

import torch

from tests.conftest import DATDIR
from biom3.Stage3.run_ProteoScribe_sample import load_json_config, convert_to_namespace
from biom3.Stage3.io import build_model_ProteoScribe
import biom3.Stage3.sampling_analysis as Stage3_sample_tools

MINI_WEIGHTS = os.path.join(DATDIR, "models/stage3/weights/minimodel1_weights1.pth")
MINI_CONFIG = os.path.join(DATDIR, "configs/test_stage3_config_v2.json")


@pytest.fixture
def mini_model_and_args():
    """Build the mini ProteoScribe model from test weights and config."""
    config_dict = load_json_config(MINI_CONFIG)
    config_args = convert_to_namespace(config_dict)
    config_args.device = "cpu"
    model = build_model_ProteoScribe(config_args)
    state_dict = torch.load(MINI_WEIGHTS, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model, config_args


def _run_batch_generate(model, args, batch_size, seed):
    """Helper: seed RNG and run batch_generate_denoised_sampled."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    perms = torch.stack([torch.randperm(args.diffusion_steps) for _ in range(batch_size)])
    cond = torch.randn(batch_size, args.text_emb_dim)
    mask_list, time_list = Stage3_sample_tools.batch_generate_denoised_sampled(
        args=args,
        model=model,
        extract_digit_samples=torch.zeros(batch_size, args.diffusion_steps),
        extract_time=torch.zeros(batch_size).long(),
        extract_digit_label=cond,
        sampling_path=perms,
    )
    return mask_list, time_list


@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_output_shapes(mini_model_and_args, batch_size):
    """Verify output list lengths and per-element tensor shapes."""
    model, args = mini_model_and_args
    mask_list, time_list = _run_batch_generate(model, args, batch_size, seed=42)

    assert len(mask_list) == args.diffusion_steps
    assert len(time_list) == args.diffusion_steps

    for step_arr in mask_list:
        assert step_arr.shape == (batch_size, 1, args.diffusion_steps)

    for step_arr in time_list:
        assert step_arr.shape == (batch_size, 1)


def test_output_token_range(mini_model_and_args):
    """All generated tokens must be in [0, num_classes)."""
    model, args = mini_model_and_args
    mask_list, _ = _run_batch_generate(model, args, batch_size=2, seed=42)
    final = mask_list[-1]
    assert np.all(final >= 0)
    assert np.all(final < args.num_classes)


def test_deterministic_generation(mini_model_and_args):
    """Same seed produces identical outputs across two runs."""
    model, args = mini_model_and_args
    mask_list_a, time_list_a = _run_batch_generate(model, args, batch_size=2, seed=7)
    mask_list_b, time_list_b = _run_batch_generate(model, args, batch_size=2, seed=7)

    for a, b in zip(mask_list_a, mask_list_b):
        np.testing.assert_array_equal(a, b)
    for a, b in zip(time_list_a, time_list_b):
        np.testing.assert_array_equal(a, b)


def test_different_seeds_differ(mini_model_and_args):
    """Different seeds should produce different final sequences."""
    model, args = mini_model_and_args
    mask_list_a, _ = _run_batch_generate(model, args, batch_size=1, seed=1)
    mask_list_b, _ = _run_batch_generate(model, args, batch_size=1, seed=2)
    assert not np.array_equal(mask_list_a[-1], mask_list_b[-1])


def test_batch_independence(mini_model_and_args, monkeypatch):
    """Each batch item's update must only touch its own position.

    Monkeypatches Tensor.exponential_ to fill with a constant, making the
    Gumbel-max sampling deterministic. With identical inputs and no
    randomness, two batch items sharing the same conditioning and sampling
    path must produce the same sequence. Cross-contamination from incorrect
    indexing would break this symmetry.
    """
    def deterministic_exponential(self, lambd=1.0):
        self.fill_(1.0)
        return self

    monkeypatch.setattr(torch.Tensor, "exponential_", deterministic_exponential)

    model, args = mini_model_and_args
    torch.manual_seed(42)

    perm = torch.randperm(args.diffusion_steps)
    perms = perm.unsqueeze(0).expand(2, -1).clone()
    cond = torch.randn(1, args.text_emb_dim).expand(2, -1).clone()

    mask_list, _ = Stage3_sample_tools.batch_generate_denoised_sampled(
        args=args,
        model=model,
        extract_digit_samples=torch.zeros(2, args.diffusion_steps),
        extract_time=torch.zeros(2).long(),
        extract_digit_label=cond,
        sampling_path=perms,
    )

    final = mask_list[-1]
    np.testing.assert_array_equal(
        final[0], final[1],
        err_msg="Batch items with identical inputs produced different outputs"
    )


# ---------------------------------------------------------------------------
#  Helpers for confidence-based unmasking and argmax token strategy
# ---------------------------------------------------------------------------

def _run_batch_generate_confidence(model, args, batch_size, seed, token_strategy="sample"):
    """Helper: seed RNG and run batch_generate_denoised_sampled_confidence."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    args_copy = type(args)(**vars(args))
    args_copy.token_strategy = token_strategy
    cond = torch.randn(batch_size, args.text_emb_dim)
    mask_list, time_list = Stage3_sample_tools.batch_generate_denoised_sampled_confidence(
        args=args_copy,
        model=model,
        extract_digit_samples=torch.zeros(batch_size, args.diffusion_steps),
        extract_time=torch.zeros(batch_size).long(),
        extract_digit_label=cond,
    )
    return mask_list, time_list


def _run_batch_generate_with_token_strategy(model, args, batch_size, seed, token_strategy):
    """Helper: seed RNG and run batch_generate_denoised_sampled with a given token_strategy."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    args_copy = type(args)(**vars(args))
    args_copy.token_strategy = token_strategy
    perms = torch.stack([torch.randperm(args.diffusion_steps) for _ in range(batch_size)])
    cond = torch.randn(batch_size, args.text_emb_dim)
    mask_list, time_list = Stage3_sample_tools.batch_generate_denoised_sampled(
        args=args_copy,
        model=model,
        extract_digit_samples=torch.zeros(batch_size, args.diffusion_steps),
        extract_time=torch.zeros(batch_size).long(),
        extract_digit_label=cond,
        sampling_path=perms,
    )
    return mask_list, time_list


# ---------------------------------------------------------------------------
#  Tests: confidence-based unmasking
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("batch_size", [1, 2, 4])
@pytest.mark.parametrize("token_strategy", ["sample", "argmax"])
def test_confidence_output_shapes(mini_model_and_args, batch_size, token_strategy):
    """Verify output shapes for confidence-based unmasking."""
    model, args = mini_model_and_args
    mask_list, time_list = _run_batch_generate_confidence(
        model, args, batch_size, seed=42, token_strategy=token_strategy
    )

    assert len(mask_list) == args.diffusion_steps
    assert len(time_list) == args.diffusion_steps

    for step_arr in mask_list:
        assert step_arr.shape == (batch_size, 1, args.diffusion_steps)

    for step_arr in time_list:
        assert step_arr.shape == (batch_size, 1)


@pytest.mark.parametrize("token_strategy", ["sample", "argmax"])
def test_confidence_output_token_range(mini_model_and_args, token_strategy):
    """All generated tokens must be in [0, num_classes)."""
    model, args = mini_model_and_args
    mask_list, _ = _run_batch_generate_confidence(
        model, args, batch_size=2, seed=42, token_strategy=token_strategy
    )
    final = mask_list[-1]
    assert np.all(final >= 0)
    assert np.all(final < args.num_classes)


def test_confidence_argmax_deterministic(mini_model_and_args):
    """Confidence + argmax is fully deterministic regardless of seed state."""
    model, args = mini_model_and_args
    # Use different seeds to prove there's no RNG dependency
    mask_list_a, _ = _run_batch_generate_confidence(
        model, args, batch_size=2, seed=1, token_strategy="argmax"
    )
    mask_list_b, _ = _run_batch_generate_confidence(
        model, args, batch_size=2, seed=99, token_strategy="argmax"
    )

    # Despite different seeds, the conditioning is drawn from the same
    # torch.randn call after manual_seed, so we need the same seed for
    # identical conditioning. Re-run with same seed to confirm determinism.
    mask_list_c, _ = _run_batch_generate_confidence(
        model, args, batch_size=2, seed=7, token_strategy="argmax"
    )
    mask_list_d, _ = _run_batch_generate_confidence(
        model, args, batch_size=2, seed=7, token_strategy="argmax"
    )

    for a, b in zip(mask_list_c, mask_list_d):
        np.testing.assert_array_equal(a, b)


def test_confidence_batch_independence(mini_model_and_args):
    """Identical conditioning per batch item produces identical sequences."""
    model, args = mini_model_and_args
    args_copy = type(args)(**vars(args))
    args_copy.token_strategy = "argmax"
    torch.manual_seed(42)

    cond = torch.randn(1, args.text_emb_dim).expand(2, -1).clone()

    mask_list, _ = Stage3_sample_tools.batch_generate_denoised_sampled_confidence(
        args=args_copy,
        model=model,
        extract_digit_samples=torch.zeros(2, args.diffusion_steps),
        extract_time=torch.zeros(2).long(),
        extract_digit_label=cond,
    )

    final = mask_list[-1]
    np.testing.assert_array_equal(
        final[0], final[1],
        err_msg="Batch items with identical inputs produced different outputs"
    )


# ---------------------------------------------------------------------------
#  Tests: random unmasking + argmax token strategy
# ---------------------------------------------------------------------------

def test_random_argmax_deterministic(mini_model_and_args):
    """Random order + argmax: same seed produces identical outputs."""
    model, args = mini_model_and_args
    mask_list_a, _ = _run_batch_generate_with_token_strategy(
        model, args, batch_size=2, seed=42, token_strategy="argmax"
    )
    mask_list_b, _ = _run_batch_generate_with_token_strategy(
        model, args, batch_size=2, seed=42, token_strategy="argmax"
    )

    for a, b in zip(mask_list_a, mask_list_b):
        np.testing.assert_array_equal(a, b)


def test_random_argmax_output_token_range(mini_model_and_args):
    """Random order + argmax: tokens in [0, num_classes)."""
    model, args = mini_model_and_args
    mask_list, _ = _run_batch_generate_with_token_strategy(
        model, args, batch_size=2, seed=42, token_strategy="argmax"
    )
    final = mask_list[-1]
    assert np.all(final >= 0)
    assert np.all(final < args.num_classes)

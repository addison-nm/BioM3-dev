"""Multi-node multi-GPU smoke tests for Stage 3 (ProteoScribe) sampling.

Run via the launcher wrapper, e.g.::

    ./scripts/test_stage3_multinode.sh 2 12

which expands to::

    aurora_multinode.sh \\
        python -m pytest tests/stage3_tests/test_multinode_sample.py \\
            --multinode 2 --multidevice 12 -m multinode

The conftest hook gates these tests on ``--multinode N --multidevice M``
matching ``WORLD_SIZE``, so they are silently skipped when not launched
under mpiexec / torchrun.

Single-rank degenerate mode (``--multinode 1 --multidevice 1``) is
supported as a local fast-loop smoke check.
"""

import os

import pytest
import torch
import torch.distributed as dist

from biom3.core.distributed import (
    init_distributed_if_launched,
    is_main_process,
    barrier,
)
from biom3.core.helpers import load_json_config, convert_to_namespace
from biom3.Stage3.io import prepare_model_ProteoScribe
from biom3.Stage3.run_ProteoScribe_sample import main as run_sample_main, parse_arguments

from tests.conftest import DATDIR, TMPDIR, remove_dir


pytestmark = [pytest.mark.multinode]


MINI_WEIGHTS = os.path.join(DATDIR, "models/stage3/weights/minimodel1_ds128_weights1.pth")
MINI_CONFIG = os.path.join(DATDIR, "configs/test_stage3_config_v2.json")
TEST_EMBEDDINGS = os.path.join(DATDIR, "embeddings/test_Facilitator_embeddings.pt")

OUTPUTS_DIR = os.path.join(TMPDIR, "multinode_outputs")


@pytest.fixture(scope="session")
def dist_init():
    """Initialise torch.distributed once for the whole test session."""
    rank, local_rank, world_size, device_str = init_distributed_if_launched("cpu")
    yield rank, local_rank, world_size, device_str
    barrier()


def _state_dict_fingerprint(model: torch.nn.Module) -> float:
    """Stable scalar fingerprint of a model's weights (sum-of-sums, fp64)."""
    total = 0.0
    for tensor in model.state_dict().values():
        if tensor.is_floating_point():
            total += float(tensor.detach().cpu().double().sum().item())
        else:
            total += float(tensor.detach().cpu().to(torch.float64).sum().item())
    return total


def test_model_identity_across_ranks(dist_init):
    """Every rank must load the same weights into the model."""
    rank, _, world_size, device_str = dist_init

    config_dict = load_json_config(MINI_CONFIG)
    config_args = convert_to_namespace(config_dict)
    config_args.device = device_str
    model = prepare_model_ProteoScribe(
        config_args=config_args,
        model_fpath=MINI_WEIGHTS,
        device=device_str,
        eval=True,
    )

    fingerprint = _state_dict_fingerprint(model)

    if dist.is_available() and dist.is_initialized():
        gathered = [None] * world_size
        dist.all_gather_object(gathered, fingerprint)
    else:
        gathered = [fingerprint]

    assert len(gathered) == world_size
    ref = gathered[0]
    for r, fp in enumerate(gathered):
        assert abs(fp - ref) < 1e-6, (
            f"Rank {r} fingerprint {fp} differs from rank 0 {ref}"
        )


def _build_args(output_filename: str, device: str, *, seed: int = 42):
    output_path = os.path.join(OUTPUTS_DIR, output_filename)
    argstring = [
        "-i", TEST_EMBEDDINGS,
        "-c", MINI_CONFIG,
        "-m", MINI_WEIGHTS,
        "-o", output_path,
        "--seed", str(seed),
    ]
    args = parse_arguments(argstring)
    args.device = device
    return args, output_path


def test_per_rank_generation(dist_init):
    """Every rank must run the entrypoint without errors and rank 0 must
    produce the gathered output dict."""
    rank, _, _, device_str = dist_init

    if rank == 0:
        os.makedirs(OUTPUTS_DIR, exist_ok=True)
    barrier()

    args, output_path = _build_args(
        "test_per_rank_generation.pt", device_str, seed=123
    )
    run_sample_main(args)

    if rank == 0:
        assert os.path.exists(output_path), "Rank 0 did not write the gathered output"
        result = torch.load(output_path)
        assert isinstance(result, dict)
        prompt_keys = [k for k in result if not k.startswith("_")]
        assert prompt_keys, "Output should contain at least one prompt key"
        for k in prompt_keys:
            replicas = result[k]
            assert isinstance(replicas, list)
            assert all(isinstance(s, str) and s for s in replicas), (
                f"Empty / non-string replica found under {k}: {replicas}"
            )

    barrier()
    if rank == 0 and os.path.exists(OUTPUTS_DIR):
        remove_dir(OUTPUTS_DIR)


def test_rng_world_size_invariance(dist_init):
    """Output sequences must depend only on (base_seed, prompt_idx, replica_idx)
    — never on world_size or how the work is partitioned across ranks.

    Strategy: the multi-rank merged output must equal the single-rank
    baseline that rank 0 builds locally at startup. The baseline is
    generated outside the distributed context (no init_process_group
    call) so it doesn't need the launcher.
    """
    rank, _, world_size, device_str = dist_init

    seed = 4242
    if rank == 0:
        os.makedirs(OUTPUTS_DIR, exist_ok=True)
    barrier()

    # Rank 0 builds a single-rank reference by directly invoking the
    # rank-aware sampling path with world_size=1. Easier than spawning
    # an out-of-band process and avoids the second init_process_group.
    reference = None
    if rank == 0:
        from biom3.Stage3.run_ProteoScribe_sample import (
            batch_stage3_generate_sequences,
            _merge_shards,
            prepare_model,
        )

        config_dict = load_json_config(MINI_CONFIG)
        config_args = convert_to_namespace(config_dict)
        config_args.device = device_str
        config_args._rank = 0
        config_args._world_size = 1
        config_args._base_seed = int(seed)
        config_args.unmasking_order = getattr(config_args, "unmasking_order", "random")
        config_args.token_strategy = getattr(config_args, "token_strategy", "sample")
        config_args.sequence_length = config_args.diffusion_steps
        config_args.pre_unmask = False

        ref_args = type("Args", (), {})()
        ref_args.model_path = MINI_WEIGHTS
        model = prepare_model(args=ref_args, config_args=config_args)

        embedding_dataset = torch.load(TEST_EMBEDDINGS)
        z_c = embedding_dataset["z_c"]

        ref_results = batch_stage3_generate_sequences(
            args=config_args,
            model=model,
            z_t=z_c,
        )
        reference = _merge_shards(
            [ref_results["rank_local_sequences"]],
            ref_results["num_prompts"],
            config_args.num_replicas,
        )

    # All ranks run the entrypoint under the launcher's world_size.
    args, output_path = _build_args(
        "test_rng_invariance.pt", device_str, seed=seed
    )
    run_sample_main(args)

    if rank == 0:
        merged = torch.load(output_path)
        ref_keys = sorted(k for k in reference if not k.startswith("_"))
        merged_keys = sorted(k for k in merged if not k.startswith("_"))
        assert ref_keys == merged_keys
        for k in ref_keys:
            assert reference[k] == merged[k], (
                f"World-size invariance violated for {k}:\n"
                f"  reference:   {reference[k]}\n"
                f"  world={world_size}: {merged[k]}"
            )

    barrier()
    if rank == 0 and os.path.exists(OUTPUTS_DIR):
        remove_dir(OUTPUTS_DIR)

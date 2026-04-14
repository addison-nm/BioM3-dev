"""Benchmark Stage 3 sequence generation.

Gated behind ``@pytest.mark.benchmark`` so it's skipped in normal runs.
To execute:

    pytest tests/stage3_tests/test_bench_stage3_sample.py \
        --benchmark -s

``-s`` disables pytest stdout capture so the timing table is visible.

Sweeps (``num_prompts`` × ``num_replicas`` × ``batch_size_sample``) using
the real inference config with a random-init model — kernel dispatch
depends on shapes and dtypes, not weight values, so random init is fine
for timing. Reports best-of-N wall clock, per-sequence latency, and
throughput per config.

Tune the three lists at the top of each test to the workload you care
about. Small defaults keep casual benchmark runs fast.
"""
import itertools
import time

import pytest
import torch

from biom3.core.helpers import load_json_config, convert_to_namespace
from biom3.Stage3.io import build_model_ProteoScribe
from biom3.Stage3.run_ProteoScribe_sample import batch_stage3_generate_sequences
import biom3.Stage3.sampling_analysis as sampling_analysis


CONFIG_PATH = "configs/inference/stage3_ProteoScribe_sample.json"


def _sync(device):
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elif device.startswith("xpu"):
        torch.xpu.synchronize()


def _pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    return "cpu"


@pytest.fixture(scope="module")
def bench_model():
    """Random-init real-config model. Built once per pytest module."""
    device = _pick_device()
    cfg = convert_to_namespace(load_json_config(CONFIG_PATH))
    cfg.device = device
    model = build_model_ProteoScribe(cfg)
    model.to(device).eval()
    return model, cfg, device


def _time_one(
    model,
    cfg,
    device,
    num_prompts,
    num_replicas,
    batch_size_sample,
    unmasking_order="random",
    token_strategy="sample",
):
    """Run one full generation and return wall-clock seconds."""
    torch.manual_seed(0)
    z_c = torch.randn(num_prompts, cfg.text_emb_dim, device=device)

    cfg.num_replicas = num_replicas
    cfg.batch_size_sample = batch_size_sample
    cfg.unmasking_order = unmasking_order
    cfg.token_strategy = token_strategy

    _sync(device)
    t0 = time.perf_counter()
    batch_stage3_generate_sequences(
        args=cfg,
        model=model,
        z_t=z_c,
        animate_prompts=None,
        animate_replicas=None,
        store_probabilities=False,
    )
    _sync(device)
    return time.perf_counter() - t0


def _run_sweep(
    model,
    cfg,
    device,
    num_prompts_list,
    num_replicas_list,
    batch_size_list,
    warmup=1,
    repeats=1,
    unmasking_order="random",
    token_strategy="sample",
):
    """Sweep the cartesian product and print a timing table."""
    n_params = sum(p.numel() for p in model.parameters())
    print()
    print(
        f"model: {n_params/1e6:.1f}M params | dim={cfg.transformer_dim} "
        f"depth={cfg.transformer_depth} blocks={cfg.transformer_blocks} "
        f"steps={cfg.diffusion_steps}"
    )
    print(
        f"device={device} | unmasking={unmasking_order} "
        f"tokens={token_strategy} | warmup={warmup} repeats={repeats}"
    )
    header = (
        f"{'prompts':>8} {'reps':>5} {'bss':>5} "
        f"{'total':>6} {'batches':>8} "
        f"{'best_s':>10} {'s/seq':>10} {'seq/s':>10}"
    )
    print(header)
    print("-" * len(header))

    results = []
    for num_prompts, num_replicas, batch_size_sample in itertools.product(
        num_prompts_list, num_replicas_list, batch_size_list,
    ):
        total = num_prompts * num_replicas
        n_batches = (total + batch_size_sample - 1) // batch_size_sample

        try:
            for _ in range(warmup):
                _time_one(
                    model, cfg, device,
                    num_prompts, num_replicas, batch_size_sample,
                    unmasking_order=unmasking_order,
                    token_strategy=token_strategy,
                )
            times = [
                _time_one(
                    model, cfg, device,
                    num_prompts, num_replicas, batch_size_sample,
                    unmasking_order=unmasking_order,
                    token_strategy=token_strategy,
                )
                for _ in range(repeats)
            ]
        except torch.cuda.OutOfMemoryError:
            print(
                f"{num_prompts:>8} {num_replicas:>5} {batch_size_sample:>5} "
                f"{total:>6} {n_batches:>8}   OOM"
            )
            torch.cuda.empty_cache()
            results.append((num_prompts, num_replicas, batch_size_sample, None))
            continue

        best = min(times)
        per_seq = best / total
        seq_per_s = total / best
        print(
            f"{num_prompts:>8} {num_replicas:>5} {batch_size_sample:>5} "
            f"{total:>6} {n_batches:>8} "
            f"{best:>10.2f} {per_seq:>10.3f} {seq_per_s:>10.2f}"
        )
        results.append((num_prompts, num_replicas, batch_size_sample, best))
    return results


def _force_autocast(mode):
    """Monkey-patch sampling_analysis._inference_autocast to force on/off."""
    enabled = mode == "on"
    original = sampling_analysis._inference_autocast

    def _forced(device):
        device_type = torch.device(device).type
        return torch.autocast(
            device_type=device_type,
            dtype=torch.bfloat16,
            enabled=enabled,
        )
    sampling_analysis._inference_autocast = _forced
    return original


@pytest.mark.benchmark
def test_bench_stage3_sample_sweep(bench_model):
    """Sweep (num_prompts, num_replicas, batch_size_sample) and print table."""
    model, cfg, device = bench_model

    num_prompts_list = [1, 4]
    num_replicas_list = [1, 4]
    batch_size_list = [1, 4, 8]

    results = _run_sweep(
        model, cfg, device,
        num_prompts_list, num_replicas_list, batch_size_list,
        warmup=1, repeats=1,
    )
    assert any(r[3] is not None for r in results), "all sweep configs OOM'd"


@pytest.mark.benchmark
def test_bench_stage3_sample_autocast_on_off(bench_model):
    """Compare bf16 autocast on vs off at a single representative config.

    Expected on Blackwell (sm_121a): autocast-on is several times faster
    because fp32 paths fall back to non-tensor-core SIMT kernels.
    """
    model, cfg, device = bench_model

    num_prompts, num_replicas, batch_size_sample = 1, 8, 8

    original = sampling_analysis._inference_autocast
    try:
        _force_autocast("off")
        t_off = _time_one(
            model, cfg, device,
            num_prompts, num_replicas, batch_size_sample,
        )

        _force_autocast("on")
        t_on = _time_one(
            model, cfg, device,
            num_prompts, num_replicas, batch_size_sample,
        )
    finally:
        sampling_analysis._inference_autocast = original

    print()
    print(
        f"config: prompts={num_prompts} reps={num_replicas} "
        f"bss={batch_size_sample} (total={num_prompts * num_replicas})"
    )
    print(f"autocast off: {t_off:.2f}s")
    print(f"autocast on:  {t_on:.2f}s")
    print(f"speedup:      {t_off / t_on:.2f}x")

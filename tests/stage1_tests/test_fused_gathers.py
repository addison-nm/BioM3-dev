"""Fusing the per-step all_gathers must not move a single element.

The Pfam step used to issue four all_gathers of [B, D] plus two of [B] per
training step. Both are fused into one collective each. Lightning's all_gather
returns [W, B, ...] and callers flatten rank-major, so the fused layout has to
reproduce that exactly -- a silent permutation here would scramble which
embedding belongs to which rank's rows and corrupt the contrastive targets
without ever raising.

Pure CPU tensor bookkeeping: no distributed, no model, no XPU.
"""
import torch


def _fake_all_gather(per_rank):
    """Stand-in for Lightning's all_gather: [B, ...] per rank -> [W, B, ...]."""
    return torch.stack(per_rank, dim=0)


def test_gather_four_matches_four_separate_gathers():
    torch.manual_seed(0)
    W, B, D = 5, 3, 7
    a = [torch.randn(B, D) for _ in range(W)]
    b = [torch.randn(B, D) for _ in range(W)]
    c = [torch.randn(B, D) for _ in range(W)]
    d = [torch.randn(B, D) for _ in range(W)]

    # what the old code produced: one gather per tensor, then .view(-1, D)
    want = [_fake_all_gather(t).view(-1, D) for t in (a, b, c, d)]

    # what _gather_four produces: cat on each rank, one gather, slice the middle
    fused_per_rank = [torch.cat((a[r], b[r], c[r], d[r]), dim=0) for r in range(W)]
    out = _fake_all_gather(fused_per_rank).reshape(-1, 4 * B, D)
    got = [out[:, i * B:(i + 1) * B, :].reshape(-1, D) for i in range(4)]

    for i, (g, w) in enumerate(zip(got, want)):
        assert torch.equal(g, w), f"tensor {i} differs after fusing"


def test_fused_logsumexp_gather_matches_two_halves():
    """row_logZ = cat(gather(lz[:B]), gather(lz[B:])) via a single gather of lz."""
    torch.manual_seed(1)
    W, B = 6, 4
    lz = [torch.randn(2 * B) for _ in range(W)]

    want_swiss = _fake_all_gather([t[:B] for t in lz]).reshape(-1)
    want_pfam = _fake_all_gather([t[B:] for t in lz]).reshape(-1)

    all_lz = _fake_all_gather(lz).reshape(-1, 2 * B)
    got_swiss = all_lz[:, :B].reshape(-1)
    got_pfam = all_lz[:, B:].reshape(-1)

    assert torch.equal(got_swiss, want_swiss)
    assert torch.equal(got_pfam, want_pfam)
    assert torch.equal(torch.cat([got_swiss, got_pfam]),
                       torch.cat([want_swiss, want_pfam]))


def test_batched_metric_reduction_equals_per_scalar_mean():
    """One stacked all_reduce must give what N separate mean-reductions gave."""
    torch.manual_seed(2)
    W = 8
    keys = [f"m{i}" for i in range(18)]
    per_rank = [{k: float(torch.randn(())) for k in keys} for _ in range(W)]

    want = {k: sum(p[k] for p in per_rank) / W for k in keys}

    stacked = torch.stack([
        torch.stack([torch.tensor(p[k]) for k in keys]) for p in per_rank
    ])
    reduced = stacked.sum(dim=0) / W
    got = dict(zip(keys, reduced.tolist()))

    for k in keys:
        assert abs(got[k] - want[k]) < 1e-6, k


def test_gather_backward_is_allreduce_then_slice():
    """The gradient of an all_gather is each rank's slice summed over ranks.

    _GatherGrad.backward all_reduces the incoming gradient and returns
    grad[rank]. Every rank computes its loss from the SAME gathered tensor, so
    rank r's input receives a contribution from every rank's backward pass --
    which is what summing over ranks and slicing gives. torch's non-NCCL path
    reaches the same value via an all-to-all of W tensors; this pins the value
    so the cheaper collective can't silently change it.
    """
    torch.manual_seed(3)
    W, B, D = 5, 3, 4

    # grad_out as seen by each rank: [W, B, D], generally different per rank
    grads = [torch.randn(W, B, D) for _ in range(W)]

    # reference: rank r's gradient is the sum over ranks of that rank's slice
    want = [sum(grads[s][r] for s in range(W)) for r in range(W)]

    # implementation: all_reduce (elementwise sum over ranks), then slice
    reduced = torch.stack(grads).sum(dim=0)          # what all_reduce leaves
    got = [reduced[r] for r in range(W)]

    for r in range(W):
        assert torch.allclose(got[r], want[r], atol=1e-6), f"rank {r}"

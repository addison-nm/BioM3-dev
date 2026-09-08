"""Row-sharded contrastive losses must equal the dense ones exactly.

The dense implementations build the full M x M similarity matrix on every rank
(O(W^2)); the sharded ones build only [2B, M] per rank (O(W)). They must agree,
otherwise switching to sharded silently changes the objective.

Runs on CPU with tiny tensors -- no distributed, no XPU.
"""
import torch
import torch.nn as nn

from biom3.Stage1.model import pfam_PEN_CL


class _Stub(pfam_PEN_CL):
    """Bare instance exposing only the loss methods (skips encoder construction)."""
    def __init__(self, temperature):
        nn.Module.__init__(self)
        self.temperature = temperature


def _dense_inter(m, z_p, z_t, N):
    """Verbatim re-implementation of compute_inter_loss, per-row (no .mean())."""
    M = 2 * N
    mask = torch.zeros((M, M))
    mask[N:, :N] = torch.eye(N)
    mask[:N, N:] = torch.eye(N)
    mask = mask.bool()
    logits = (z_t @ z_p.T) / m.temperature
    mp = m.set_inf(z_p @ z_p.T, mask)
    mt = m.set_inf(z_t @ z_t.T, mask)
    ml = m.set_inf(logits, mask)
    targets = torch.softmax((mp + mt) / (2 * m.temperature), dim=-1)
    text = (-targets * torch.log_softmax(ml, dim=-1)).sum(1)
    prot = (-targets.T * torch.log_softmax(ml.T, dim=-1)).sum(1)
    return ((prot + text) / 2)


def _dense_intra(m, z_p):
    M = z_p.shape[0]
    cs = (z_p @ z_p.T) / m.temperature
    eye = torch.eye(M, dtype=torch.bool)
    cs = m.set_inf(cs, eye)
    pos_mask = eye.roll(shifts=M // 2, dims=0)
    return -cs[pos_mask] + torch.logsumexp(cs, dim=-1)


# float32, not float64: set_inf() whitelists float32/float16 only
# (RUN1_KNOWN_ISSUES item E). Tolerance set for fp32 accumulation.
TOL = 2e-5


def _run(W, B, D=16, tau=0.8, seed=0):
    torch.manual_seed(seed)
    N, M = W * B, 2 * W * B
    z_p = torch.randn(M, D, dtype=torch.float32)
    z_t = torch.randn(M, D, dtype=torch.float32)
    m = _Stub(tau)

    dense_inter = _dense_inter(m, z_p, z_t, N)
    dense_intra = _dense_intra(m, z_p)

    # rank r owns swiss rows [r*B,(r+1)*B) and pfam rows [N+r*B, N+(r+1)*B)
    rows = [torch.cat([torch.arange(r * B, (r + 1) * B),
                       torch.arange(N + r * B, N + (r + 1) * B)]) for r in range(W)]

    # the all_gather the real code performs: per-row logsumexp for every row
    logZ = torch.empty(M, dtype=torch.float32)
    for ri in rows:
        logZ[ri] = m.inter_row_logsumexp(z_p, z_t, ri)

    max_inter = max_intra = 0.0
    for ri in rows:
        s_inter, _ = m.compute_inter_loss_sharded(z_p, z_t, N, ri, logZ)
        s_intra, _ = m.compute_intra_loss_sharded(z_p, N, ri)
        max_inter = max(max_inter, abs(s_inter.item() - dense_inter[ri].mean().item()))
        max_intra = max(max_intra, abs(s_intra.item() - dense_intra[ri].mean().item()))

    # the mean over ranks must also equal the global mean (DDP averages these)
    glob_inter = sum(m.compute_inter_loss_sharded(z_p, z_t, N, ri, logZ)[0] for ri in rows) / W
    glob_intra = sum(m.compute_intra_loss_sharded(z_p, N, ri)[0] for ri in rows) / W
    return (max_inter, max_intra,
            abs(glob_inter.item() - dense_inter.mean().item()),
            abs(glob_intra.item() - dense_intra.mean().item()))


def test_sharded_matches_dense():
    for W, B in ((1, 4), (2, 3), (4, 2), (8, 2), (3, 5)):
        pr_i, pr_a, gl_i, gl_a = _run(W, B)
        assert pr_i < TOL, f"W={W} B={B}: per-rank inter mismatch {pr_i}"
        assert pr_a < TOL, f"W={W} B={B}: per-rank intra mismatch {pr_a}"
        assert gl_i < TOL, f"W={W} B={B}: global inter mismatch {gl_i}"
        assert gl_a < TOL, f"W={W} B={B}: global intra mismatch {gl_a}"


if __name__ == "__main__":
    for W, B in ((1, 4), (2, 3), (4, 2), (8, 2), (3, 5)):
        pr_i, pr_a, gl_i, gl_a = _run(W, B)
        ok = max(pr_i, pr_a, gl_i, gl_a) < TOL
        print(f"  W={W:2} B={B}  M={2*W*B:3}  per-rank inter {pr_i:.2e} intra {pr_a:.2e} "
              f" global inter {gl_i:.2e} intra {gl_a:.2e}   {'PASS' if ok else 'FAIL'}")

"""Tests for the custom CUDA histogram kernel and GPU-accelerated binning.

Both are GPU performance paths with exact-CPU-equivalent results; these assert
the equivalence. Skipped when CUDA (or the compiled kernel) is unavailable.
"""
import numpy as np
import pytest
import torch

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _torch_scatter_hist(binned, slot, g, h, w, K, F, B, dev):
    """Reference: the torch scatter the kernel replaces."""
    n = slot.shape[0]
    fidx = torch.arange(F, device=dev)
    flat = (slot[:, None] * (F * B) + fidx[None, :] * B + binned.long()).reshape(-1)
    out = torch.zeros(3, K * F * B, device=dev)
    out[0].scatter_add_(0, flat, g[:, None].expand(n, F).reshape(-1))
    out[1].scatter_add_(0, flat, h[:, None].expand(n, F).reshape(-1))
    out[2].scatter_add_(0, flat, w[:, None].expand(n, F).reshape(-1))
    return out.view(3, K, F, B)


@cuda
@pytest.mark.parametrize("K", [1, 4, 31])
def test_cuda_hist_matches_torch_scatter(K):
    from yabt.cuda_hist import is_available, build_hist
    if not is_available():
        pytest.skip("CUDA histogram kernel did not build")
    dev = "cuda"
    torch.manual_seed(0)
    n, F, B = 20000, 24, 256
    binned = torch.randint(0, B, (n, F), dtype=torch.uint8, device=dev)
    g = torch.randn(n, device=dev)
    h = torch.rand(n, device=dev) + 0.1
    w = torch.ones(n, device=dev)
    slot = (torch.zeros(n, dtype=torch.long, device=dev) if K == 1
            else torch.randint(0, K, (n,), device=dev))
    ref = _torch_scatter_hist(binned, slot, g, h, w, K, F, B, dev)
    got = build_hist(binned, slot, g, h, w, K, F, B)
    # Float atomic-add ordering differs from torch's reduction, so allow a tiny
    # relative tolerance (torch's own CUDA scatter_add_ is non-deterministic too).
    rel = (ref - got).abs().max().item() / max(ref.abs().max().item(), 1e-9)
    assert rel < 1e-5


@cuda
def test_cuda_hist_count_weight():
    """The count channel uses the 0/1 weight (smaller-child scatter path)."""
    from yabt.cuda_hist import is_available, build_hist
    if not is_available():
        pytest.skip("CUDA histogram kernel did not build")
    dev = "cuda"
    torch.manual_seed(1)
    n, F, B, K = 8000, 16, 256, 6
    binned = torch.randint(0, B, (n, F), dtype=torch.uint8, device=dev)
    g = torch.randn(n, device=dev)
    h = torch.rand(n, device=dev) + 0.1
    w = (torch.rand(n, device=dev) > 0.5).float()  # mask
    slot = torch.randint(0, K, (n,), device=dev)
    ref = _torch_scatter_hist(binned, slot, g * w, h * w, w, K, F, B, dev)
    got = build_hist(binned, slot, g * w, h * w, w, K, F, B)
    rel = (ref - got).abs().max().item() / max(ref.abs().max().item(), 1e-9)
    assert rel < 1e-5
    # count channel sums to the number of weighted rows
    assert abs(got[2].sum().item() - float(w.sum()) * F) < 1.0


@cuda
def test_gpu_binning_matches_cpu():
    """GPU quantile fit + GPU transform produce the same binned codes as the
    numpy/CPU path (the whole point: a faster path, identical bins)."""
    from yabt.binning import Binner
    rng = np.random.RandomState(0)
    X = rng.randn(120000, 40).astype(np.float32)
    # A couple of low-cardinality columns to exercise unique-edge dedup.
    X[:, 3] = (rng.rand(X.shape[0]) > 0.7).astype(np.float32)
    b = Binner(max_bins=256).fit(X)              # uses GPU path (n*F large, no NaN)
    gpu_codes = b.transform(X, device="cuda").cpu()
    cpu_codes = b.transform(X, device="cpu")
    assert torch.equal(gpu_codes, cpu_codes)

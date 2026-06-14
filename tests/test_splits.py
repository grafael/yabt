import numpy as np
import pytest
import torch

from yabt.binning import MAX_BINS, Binner
from yabt.histogram import build_histogram, find_best_split


def brute_force_best_split(binned, grad, hess, lam):
    """Exhaustive split search over (feature, bin) on binned data."""
    n, F = binned.shape
    G, H = grad.sum(), hess.sum()
    parent = G * G / (H + lam)
    best = (-np.inf, -1, -1)
    for f in range(F):
        for b in range(MAX_BINS - 1):
            mask = binned[:, f] <= b
            if mask.sum() < 1 or (~mask).sum() < 1:
                continue
            GL, HL = grad[mask].sum(), hess[mask].sum()
            GR, HR = G - GL, H - HL
            gain = 0.5 * (GL * GL / (HL + lam) + GR * GR / (HR + lam) - parent)
            if gain > best[0]:
                best = (gain, f, b)
    return best


def test_split_matches_brute_force():
    rng = np.random.default_rng(42)
    n, F = 500, 6
    binned_np = rng.integers(0, 30, size=(n, F)).astype(np.uint8)
    grad_np = rng.normal(size=n).astype(np.float32)
    hess_np = rng.uniform(0.1, 1.0, size=n).astype(np.float32)

    binned = torch.from_numpy(binned_np)
    grad = torch.from_numpy(grad_np)
    hess = torch.from_numpy(hess_np)

    hist = build_histogram(binned, grad, hess)
    gain, f, b = find_best_split(hist, lam=1.0, gamma=0.0, min_child_weight=0.0, min_samples_leaf=1)

    bf_gain, bf_f, bf_b = brute_force_best_split(binned_np, grad_np.astype(np.float64), hess_np.astype(np.float64), lam=1.0)
    assert (f, b) == (bf_f, bf_b)
    assert gain == pytest.approx(bf_gain, rel=1e-3)


def test_histogram_sums():
    rng = np.random.default_rng(0)
    binned = torch.from_numpy(rng.integers(0, 256, size=(200, 4)).astype(np.uint8))
    grad = torch.from_numpy(rng.normal(size=200).astype(np.float32))
    hess = torch.ones(200)
    hist = build_histogram(binned, grad, hess)
    for f in range(4):
        assert float(hist[0, f].sum()) == pytest.approx(float(grad.sum()), abs=1e-3)
        assert float(hist[2, f].sum()) == 200


def test_binner_split_equivalence():
    """bin(x) <= b must be exactly equivalent to x <= edges[b]."""
    rng = np.random.default_rng(1)
    X = rng.normal(size=(1000, 3)).astype(np.float32)
    binner = Binner(max_bins=16).fit(X)
    binned = binner.transform(X)
    for f in range(3):
        for b in range(len(binner.edges_[f])):
            lhs = (binned[:, f].numpy() <= b)
            rhs = (X[:, f] <= binner.edge_value(f, b))
            assert (lhs == rhs).all()

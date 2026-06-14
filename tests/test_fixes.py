"""Regression tests for fixed bugs: fp32 histograms, real GOSS, wired-up
interaction detection, working global leaf refit, refinement margin fix and
device auto-detection."""

import numpy as np
import pytest
import torch
from sklearn.datasets import make_regression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from yabt import Booster, BoostParams, MSELoss, YABTRegressor
from yabt.adaptive_features import GradientBasedOneSideSampling
from yabt.binning import Binner
from yabt.histogram import build_histogram
from yabt.refine_fast import global_leaf_refit_fast, refine_tree_fast
from yabt.tree import TreeParams, grow_tree

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def test_histogram_exact_at_large_counts():
    """Counts and sums must stay exact for bins holding > 2048 rows (fp16
    accumulation could not represent these and could overflow at 65504)."""
    n = 100_000
    binned = torch.zeros((n, 2), dtype=torch.uint8, device=DEVICE)  # all rows in bin 0
    grad = torch.ones(n, device=DEVICE)
    hess = torch.full((n,), 0.5, device=DEVICE)
    hist = build_histogram(binned, grad, hess)
    assert float(hist[2, 0, 0]) == n
    assert float(hist[0, 0, 0]) == pytest.approx(n, rel=1e-6)
    assert float(hist[1, 0, 0]) == pytest.approx(0.5 * n, rel=1e-6)


def test_goss_weights_unbiased():
    """GOSS must keep ~ratio*n rows and amplify sampled rows so expected
    grad sums match the full data."""
    torch.manual_seed(0)
    n = 10_000
    grad = torch.randn(n, device=DEVICE)
    hess = torch.rand(n, device=DEVICE)
    goss = GradientBasedOneSideSampling(device=DEVICE, seed=0)
    rows, weights = goss.sample(grad, ratio=0.8)

    assert rows.numel() == pytest.approx(0.8 * n, abs=2)
    assert weights is not None and weights.shape == rows.shape
    top_n = int(n * 0.2)  # ratio=0.8 -> top fraction = 1 - ratio = 0.2
    assert torch.all(weights[:top_n] == 1.0)
    # weighted hess sum is an unbiased estimate of the full hess sum
    est = float((hess[rows] * weights).sum())
    assert est == pytest.approx(float(hess.sum()), rel=0.1)

    # deterministic given the seed
    rows2, _ = GradientBasedOneSideSampling(device=DEVICE, seed=0).sample(grad, ratio=0.8)
    assert torch.equal(rows, rows2)


def test_interaction_detection_reports_pairs():
    """detect_interactions=True must actually record split co-occurrences."""
    rng = np.random.default_rng(0)
    n = 3000
    X = rng.normal(size=(n, 5)).astype(np.float32)
    y = (X[:, 0] * X[:, 1]).astype(np.float32)  # pure interaction of features 0 and 1

    reg = YABTRegressor(n_estimators=30, max_leaves=15, device=DEVICE, refine_steps=0,
                        detect_interactions=True)
    reg.fit(X, y)

    top = reg.booster_.top_interactions(k=3)
    assert top, "interaction detector recorded nothing"
    assert tuple(sorted(top[0][:2])) == (0, 1)


def test_global_leaf_refit_reduces_loss():
    """global_leaf_refit_fast must actually update leaf values and reduce loss."""
    X, y = make_regression(n_samples=2000, n_features=8, noise=5.0, random_state=0)
    y = ((y - y.mean()) / y.std()).astype(np.float32)

    params = BoostParams(n_estimators=20, device=DEVICE, refine_steps=0,
                         refit_steps=10, refit_lr=0.3)
    booster = Booster(params, MSELoss()).fit(X, y)

    Xt = torch.from_numpy(booster.binner.impute(X)).to(booster.device_)
    yt = torch.as_tensor(y, device=booster.device_)
    loss_before = float(MSELoss.loss(booster._margin(Xt), yt))
    values_before = [t.value.clone() for t in booster.trees]

    margin = global_leaf_refit_fast(booster.trees, Xt, yt, booster.base_score,
                                    MSELoss(), params)
    loss_after = float(MSELoss.loss(margin, yt))

    assert any(not torch.equal(b, t.value) for b, t in zip(values_before, booster.trees))
    assert loss_after < loss_before
    # returned margin must be consistent with the mutated trees
    assert torch.allclose(margin, booster._margin(Xt), atol=1e-4)


def test_refinement_reduces_loss():
    """refine_tree_fast must improve on the unrefined tree (gradients taken at
    the margin including the tree's own contribution)."""
    rng = np.random.default_rng(0)
    n = 5000
    X = rng.normal(size=(n, 4)).astype(np.float32)
    y = (X[:, 0] + 0.5 * X[:, 1] ** 2 + 0.1 * rng.normal(size=n)).astype(np.float32)

    dev = DEVICE
    binner = Binner().fit(X)
    binned = binner.transform(X, device=dev)
    Xraw = torch.from_numpy(binner.impute(X)).to(dev)
    yt = torch.as_tensor(y, device=dev)
    margin = torch.zeros(n, device=dev)

    grad, hess = MSELoss.grad_hess(margin, yt)
    tree = grow_tree(binned, grad, hess, binner, TreeParams())

    params = BoostParams(refine_steps=10, refine_lr=0.2, device=dev)
    refined = refine_tree_fast(tree, Xraw, yt, margin, MSELoss(), params)

    loss_orig = float(MSELoss.loss(margin + tree.predict(Xraw), yt))
    loss_refined = float(MSELoss.loss(margin + refined.predict(Xraw), yt))
    assert loss_refined < loss_orig


def test_device_auto_default():
    """Default device must work whether or not CUDA is available."""
    X, y = make_regression(n_samples=500, n_features=5, noise=1.0, random_state=0)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0)
    reg = YABTRegressor(n_estimators=30)  # no device specified
    reg.fit(Xtr, ytr)
    assert r2_score(yte, reg.predict(Xte)) > 0.5

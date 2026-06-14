"""Tests for kernel-based (RBF landmark) splits."""

import numpy as np
import pytest
import torch

from yabt import YABTClassifier, YABTRegressor
from yabt.binning import Binner
from yabt.histogram import build_histogram, per_feature_gain
from yabt.kernel_splits import WEIGHT_FLOOR, find_best_kernel_split, importance_weights
from yabt.tree import KERNEL_SPLIT, LEAF, TreeParams, grow_tree


def _circles(n, seed=0, noise=0.05):
    """Two concentric rings; label 1 = inner ring."""
    rng = np.random.default_rng(seed)
    half = n // 2
    theta = rng.uniform(0, 2 * np.pi, n)
    r = np.concatenate([np.full(half, 0.4), np.full(n - half, 1.0)])
    X = np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1)
    X += rng.normal(scale=noise, size=X.shape)
    y = np.concatenate([np.ones(half), np.zeros(n - half)]).astype(np.float32)
    return X.astype(np.float32), y


def test_find_best_kernel_split_consistent_and_positive():
    X, y = _circles(600, seed=1)
    Xn = torch.from_numpy(X)
    # Newton grads for logloss at p=0.5: a radial split has large gain.
    grad = torch.from_numpy(0.5 - y)
    hess = torch.full((600,), 0.25)
    gen = torch.Generator().manual_seed(0)

    res = find_best_kernel_split(
        Xn, grad, hess, n_candidates=16, gamma=0.0, lam=1.0,
        split_penalty=0.0, min_child_weight=1e-3, min_samples_leaf=20, gen=gen,
    )
    assert res is not None
    gain, center, gam, thr, go_left, z_std = res
    assert gain > 0
    assert center.shape == (2,)
    assert gam > 0
    assert z_std > 0

    # Routing must be reproducible from the returned (center, gamma, threshold).
    z = torch.exp(-gam * (Xn - center).square().sum(dim=1))
    assert torch.equal(z <= thr, go_left)
    assert 20 <= int(go_left.sum()) <= 580


def test_find_best_kernel_split_rejects_degenerate():
    # Constant rows: every kernel feature is constant, no valid split exists.
    Xn = torch.ones(200, 3)
    grad = torch.randn(200)
    hess = torch.ones(200)
    gen = torch.Generator().manual_seed(0)
    res = find_best_kernel_split(
        Xn, grad, hess, n_candidates=4, gamma=0.0, lam=1.0,
        split_penalty=0.0, min_child_weight=1e-3, min_samples_leaf=10, gen=gen,
    )
    assert res is None


def test_grow_tree_uses_kernel_splits_on_radial_data():
    X, y = _circles(1000, seed=2)
    binner = Binner(max_bins=64).fit(X)
    binned = binner.transform(X)
    Xraw = torch.from_numpy(binner.impute(X))
    Xn = Xraw / binner.scales_.clamp_min(1e-12)
    grad = torch.from_numpy(0.5 - y)
    hess = torch.full((1000,), 0.25)

    params = TreeParams(max_leaves=8, kernel_splits=True, kernel_candidates=16,
                        kernel_min_samples=64)
    gen = torch.Generator().manual_seed(0)
    tree = grow_tree(binned, grad, hess, binner, params, Xnorm=Xn, gen=gen)

    assert bool((tree.feature == KERNEL_SPLIT).any()), "expected at least one kernel split"
    k = int((tree.feature == KERNEL_SPLIT).sum())
    assert tree.kernel_centers.shape == (k, 2)
    assert tree.kernel_prec.shape == (k, 2)
    assert bool((tree.kernel_prec > 0).all())

    # Every row must land on a leaf, and apply must be deterministic.
    leaf = tree.apply(Xraw)
    assert bool((tree.feature[leaf] == LEAF).all())
    assert torch.equal(leaf, tree.apply(Xraw))


def test_grow_tree_without_kernel_has_no_kernel_state():
    X, y = _circles(500, seed=3)
    binner = Binner(max_bins=64).fit(X)
    binned = binner.transform(X)
    grad = torch.from_numpy(0.5 - y)
    hess = torch.full((500,), 0.25)
    tree = grow_tree(binned, grad, hess, binner, TreeParams(max_leaves=8))
    assert tree.kernel_id is None
    assert bool((tree.feature >= LEAF).all())


def test_classifier_kernel_splits_circles():
    X, y = _circles(2000, seed=4)
    Xtr, ytr = X[:1500], y[:1500]
    Xte, yte = X[1500:], y[1500:]

    clf = YABTClassifier(
        n_estimators=30, max_leaves=8, learning_rate=0.3,
        kernel_splits=True, kernel_candidates=16, kernel_min_samples=64,
        refine_steps=0, seed=0,
    )
    clf.fit(Xtr, ytr)
    acc = float((clf.predict(Xte) == yte).mean())
    assert acc >= 0.9, f"accuracy {acc:.3f}"
    assert any(t.kernel_id is not None for t in clf.booster_.trees)

    proba = clf.predict_proba(Xte)
    assert proba.shape == (500, 2)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)


def test_regressor_kernel_splits_radial_target():
    rng = np.random.default_rng(5)
    X = rng.uniform(-1, 1, size=(2000, 3)).astype(np.float32)
    y = np.exp(-4.0 * (X**2).sum(axis=1)) + rng.normal(scale=0.01, size=2000)
    reg = YABTRegressor(
        n_estimators=50, max_leaves=8, kernel_splits=True,
        kernel_candidates=8, refine_steps=0, seed=0,
    )
    reg.fit(X[:1500], y[:1500])
    pred = reg.predict(X[1500:])
    resid = y[1500:] - pred
    r2 = 1 - resid.var() / y[1500:].var()
    assert r2 > 0.8, f"R^2 {r2:.3f}"


def test_importance_weights_helper():
    assert importance_weights(torch.zeros(5)) is None
    w = importance_weights(torch.tensor([4.0, 1.0, 0.0]))
    assert float(w.max()) == pytest.approx(1.0)   # dominant feature gets weight 1
    assert float(w[1]) == pytest.approx(0.5)      # sqrt tempering: sqrt(1)/sqrt(4)
    assert float(w[2]) == pytest.approx(WEIGHT_FLOOR)  # zero-gain feature floored, not dropped


def test_per_feature_gain_ranks_informative_feature():
    rng = np.random.default_rng(7)
    n = 1000
    X = rng.normal(size=(n, 3)).astype(np.float32)
    grad = np.where(X[:, 1] > 0, 1.0, -1.0).astype(np.float32)  # feature 1 carries the signal
    binner = Binner(max_bins=32).fit(X)
    hist = build_histogram(binner.transform(X), torch.from_numpy(grad), torch.ones(n))
    g = per_feature_gain(hist, lam=1.0, min_child_weight=1e-3, min_samples_leaf=20)
    assert int(g.argmax()) == 1
    assert bool((g >= 0).all())


def test_weighted_kernel_split_routing_consistent():
    # Feature 2 is pure noise; with weighting it must get a low precision but
    # routing must still be exactly reproducible from the stored parameters.
    X, y = _circles(600, seed=8)
    rng = np.random.default_rng(8)
    Xn = torch.from_numpy(np.hstack([X, rng.normal(size=(600, 1)).astype(np.float32)]))
    grad = torch.from_numpy(0.5 - y)
    hess = torch.full((600,), 0.25)
    gen = torch.Generator().manual_seed(0)
    fw = torch.tensor([1.0, 1.0, WEIGHT_FLOOR])

    res = find_best_kernel_split(
        Xn, grad, hess, n_candidates=16, gamma=0.0, lam=1.0,
        split_penalty=0.0, min_child_weight=1e-3, min_samples_leaf=20, gen=gen,
        feature_weights=fw,
    )
    assert res is not None
    gain, center, gam, thr, go_left, _ = res
    z = torch.exp(-gam * ((Xn - center).square() * fw).sum(dim=1))
    assert torch.equal(z <= thr, go_left)


@pytest.mark.parametrize("mode", ["node", "ema"])
def test_classifier_importance_weighting_modes(mode):
    X, y = _circles(2000, seed=9)
    clf = YABTClassifier(
        n_estimators=20, max_leaves=8, learning_rate=0.3,
        kernel_splits=True, kernel_importance_weighting=mode,
        refine_steps=0, seed=0,
    )
    clf.fit(X, y)
    assert float((clf.predict(X) == y).mean()) >= 0.9
    assert any(t.kernel_id is not None for t in clf.booster_.trees)


def test_kernel_splits_with_goss_and_adaptive_features():
    X, y = _circles(2000, seed=6)
    clf = YABTClassifier(
        n_estimators=20, max_leaves=8, kernel_splits=True,
        goss_enabled=True, adaptive_features=True, refine_steps=2, seed=0,
    )
    clf.fit(X, y)
    acc = float((clf.predict(X) == y).mean())
    assert acc >= 0.9

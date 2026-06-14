"""Tests for interaction-aware splits (learned interactions guide growth)."""

import numpy as np
import pytest
import torch

from yabt import YABTClassifier, YABTRegressor
from yabt.adaptive_features import FeatureInteractionDetector
from yabt.histogram import build_histogram, find_best_split
from yabt.tree import Tree


def _xor_in_noise(n, n_features, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, size=(n, n_features)).astype(np.float32)
    y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(np.float32)
    return X, y


def test_feature_boost_flips_near_ties_but_returns_true_gain():
    rng = np.random.default_rng(0)
    n = 2000
    binned = torch.from_numpy(rng.integers(0, 32, size=(n, 2)).astype(np.uint8))
    # Feature 0 carries slightly more signal than feature 1.
    grad = torch.where(binned[:, 0] < 16, -1.0, 1.0) * 1.0
    grad = grad + torch.where(binned[:, 1] < 16, -1.0, 1.0) * 0.9
    hess = torch.ones(n)
    hist = build_histogram(binned, grad, hess)

    g0, f0, _ = find_best_split(hist, 1.0, 0.0, 1e-3, 1)
    assert f0 == 0

    boost = torch.tensor([1.0, 1.5])
    g1, f1, b1 = find_best_split(hist, 1.0, 0.0, 1e-3, 1, feature_boost=boost)
    assert f1 == 1, "boost should flip the near-tie to feature 1"
    # Returned gain must be the true (unboosted) gain of the selected split.
    gain_check, _, _ = find_best_split(
        hist, 1.0, 0.0, 1e-3, 1, feature_mask=torch.tensor([False, True])
    )
    assert g1 == pytest.approx(gain_check)
    assert g1 < g0


def test_path_feature_pairs():
    # root(f0) -> [leaf, node(f2) -> [leaf, leaf]]
    dev = "cpu"
    tree = Tree(
        feature=torch.tensor([0, -1, 2, -1, -1], device=dev),
        threshold=torch.zeros(5),
        left=torch.tensor([1, -1, 3, -1, -1]),
        right=torch.tensor([2, -1, 4, -1, -1]),
        value=torch.zeros(5),
        depth=3,
    )
    assert tree.path_feature_pairs() == [(0, 2)]


def test_detector_ranks_interacting_pair():
    det = FeatureInteractionDetector(5, device="cpu")
    for _ in range(10):
        det.update_from_path_pairs([(0, 1), (0, 1), (2, 3)])
    top = det.get_top_interactions(1)[0]
    assert (top[0], top[1]) == (0, 1)
    m = det.normalized_matrix()
    assert float(m.max()) == pytest.approx(1.0)
    # Background-relative normalization: the dominant pair saturates, the
    # background-level pair gets no boost.
    assert float(m[0, 1]) == pytest.approx(1.0)
    assert float(m[2, 3]) == 0.0
    assert torch.equal(m, m.T)


def test_uniform_noise_matrix_normalizes_to_zero():
    det = FeatureInteractionDetector(6, device="cpu")
    det.update_from_path_pairs([(0, 1), (2, 3), (4, 5)])  # all pairs equal
    assert float(det.normalized_matrix().max()) == 0.0


def test_detector_learns_xor_pair_end_to_end():
    X, y = _xor_in_noise(4000, 10)
    clf = YABTClassifier(n_estimators=30, max_leaves=8, interaction_aware=True,
                         refine_steps=0, seed=0).fit(X, y)
    top = clf.booster_.top_interactions(1)[0]
    assert {top[0], top[1]} == {0, 1}, f"expected (0,1) on top, got {top}"


def test_interaction_aware_accuracy_on_xor_in_noise():
    X, y = _xor_in_noise(6000, 20, seed=1)
    Xtr, ytr, Xte, yte = X[:4500], y[:4500], X[4500:], y[4500:]
    kw = dict(n_estimators=30, max_leaves=4, learning_rate=0.3, refine_steps=0, seed=0)

    base = YABTClassifier(interaction_aware=False, **kw).fit(Xtr, ytr)
    aware = YABTClassifier(interaction_aware=True, **kw).fit(Xtr, ytr)
    acc_base = float((base.predict(Xte) == yte).mean())
    acc_aware = float((aware.predict(Xte) == yte).mean())
    assert acc_aware >= acc_base - 0.01, f"base {acc_base:.3f} vs aware {acc_aware:.3f}"
    assert acc_aware >= 0.9, f"aware accuracy {acc_aware:.3f}"


def test_regressor_smoke_and_explicit_opt_out():
    rng = np.random.default_rng(2)
    X = rng.uniform(-1, 1, size=(2000, 5)).astype(np.float32)
    y = (X[:, 0] * X[:, 1] + 0.1 * rng.normal(size=2000)).astype(np.float32)
    reg = YABTRegressor(n_estimators=20, interaction_aware=True, refine_steps=0,
                        seed=0).fit(X, y)
    assert np.all(np.isfinite(reg.predict(X)))
    on = YABTRegressor(n_estimators=5, refine_steps=0, seed=0).fit(X, y)
    assert on.booster_.interaction_detector is not None  # enabled by default
    off = YABTRegressor(n_estimators=5, interaction_aware=False, refine_steps=0,
                        seed=0).fit(X, y)
    assert off.booster_.interaction_detector is None

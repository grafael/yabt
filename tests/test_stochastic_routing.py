"""Tests for stochastic routing (soft routing at inference)."""

import numpy as np
import torch

from yabt import YABTClassifier, YABTRegressor
from yabt.binning import Binner
from yabt.tree import TreeParams, grow_tree


def _grown_tree(n=2000, seed=0, **params):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4)).astype(np.float32)
    y = (X[:, 0] > 0.3).astype(np.float32) * 2 - 1 + 0.5 * X[:, 1]
    binner = Binner(max_bins=64).fit(X)
    grad = torch.from_numpy(-(y - 0.0).astype(np.float32))  # MSE grads at margin 0
    hess = torch.ones(n)
    tree = grow_tree(binner.transform(X), grad, hess, binner,
                     TreeParams(max_leaves=8, **params))
    return tree, torch.from_numpy(X)


def test_soft_converges_to_hard_as_tau_shrinks():
    tree, X = _grown_tree()
    # Thresholds sit on data quantiles, so nudge rows off the exact boundaries:
    # at x == threshold the gate is 0.5 for every tau (convergence is a.e.).
    X = X + 1e-3
    hard = tree.predict(X)
    soft = tree.predict_soft(X, tau=1e-5)
    np.testing.assert_allclose(soft.numpy(), hard.numpy(), rtol=1e-4, atol=1e-5)


def test_soft_prediction_is_convex_combination_of_leaves():
    tree, X = _grown_tree(seed=1)
    soft = tree.predict_soft(X, tau=0.3)
    leaf_vals = tree.value[tree.is_leaf]
    assert float(soft.min()) >= float(leaf_vals.min()) - 1e-5
    assert float(soft.max()) <= float(leaf_vals.max()) + 1e-5


def test_soft_prediction_is_smooth_across_boundary():
    tree, _ = _grown_tree(seed=2, learning_rate=1.0)
    # Dense sweep along feature 0 (the dominant split feature).
    xs = torch.zeros(1000, 4)
    xs[:, 0] = torch.linspace(-3, 3, 1000)
    hard = tree.predict(xs)
    soft = tree.predict_soft(xs, tau=0.2)
    hard_jump = (hard[1:] - hard[:-1]).abs().max()
    soft_jump = (soft[1:] - soft[:-1]).abs().max()
    assert hard_jump > 0.5, "test setup: hard prediction should jump at the split"
    assert soft_jump < 0.5 * hard_jump, f"soft {soft_jump:.4f} vs hard {hard_jump:.4f}"


def test_estimator_stochastic_routing_accuracy():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(3000, 5)).astype(np.float32)
    y = ((X[:, 0] + X[:, 1] ** 2 + rng.normal(scale=0.3, size=3000)) > 1).astype(np.float32)
    Xtr, ytr, Xte, yte = X[:2000], y[:2000], X[2000:], y[2000:]

    hard = YABTClassifier(n_estimators=30, refine_steps=0, seed=0).fit(Xtr, ytr)
    soft = YABTClassifier(n_estimators=30, refine_steps=0, seed=0,
                          stochastic_routing=True).fit(Xtr, ytr)
    # Identical training (routing only affects inference): same trees.
    t_h, t_s = hard.booster_.trees[0], soft.booster_.trees[0]
    assert torch.equal(t_h.threshold, t_s.threshold)

    acc = float((soft.predict(Xte) == yte).mean())
    assert acc >= 0.85, f"accuracy {acc:.3f}"
    proba = soft.predict_proba(Xte)
    assert np.all(np.isfinite(proba)) and proba.shape == (1000, 2)


def test_soft_routing_composes_with_kernel_and_neural_leaves():
    rng = np.random.default_rng(4)
    X = rng.uniform(-1, 1, size=(3000, 3)).astype(np.float32)
    y = (np.where((X[:, :2] ** 2).sum(1) < 0.5, 1.0, -1.0) * X[:, 2]).astype(np.float32)
    reg = YABTRegressor(
        n_estimators=20, max_leaves=8, kernel_splits=True, neural_leaves=True,
        leaf_net_hidden=0, stochastic_routing=True, routing_tau=0.05,
        refine_steps=0, seed=0,
    ).fit(X, y)
    pred = reg.predict(X)
    assert np.all(np.isfinite(pred))
    # tau=0.05 is close to hard routing; predictions should broadly agree.
    reg.booster_.p.stochastic_routing = False
    hard_pred = reg.predict(X)
    assert np.corrcoef(pred, hard_pred)[0, 1] > 0.98


def test_regressor_soft_vs_hard_smoke():
    rng = np.random.default_rng(5)
    X = rng.uniform(-2, 2, size=(3000, 3)).astype(np.float32)
    y = (np.sin(2 * X[:, 0]) + X[:, 1]).astype(np.float32)
    reg = YABTRegressor(n_estimators=50, stochastic_routing=True, refine_steps=0,
                        seed=0).fit(X[:2000], y[:2000])
    r = y[2000:] - reg.predict(X[2000:])
    assert 1 - r.var() / y[2000:].var() > 0.8

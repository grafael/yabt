"""Tests for multi-task learning (shared tree structure, per-task leaves)."""

import numpy as np
import pytest
import torch

from yabt import YABTMultiTaskRegressor, YABTRegressor
from yabt.histogram import (
    build_histogram, build_histogram_multi, find_best_split, find_best_split_multi,
)
from yabt.tree import LEAF


def _multitask_data(n, n_tasks=3, n_features=6, noise=0.1, share=1.0, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, size=(n, n_features)).astype(np.float32)
    latent = np.sin(2 * X[:, 0]) + X[:, 1] * X[:, 2]
    Y = np.stack(
        [share * (1.0 + 0.3 * t) * latent + (1 - share) * X[:, 3 + (t % 3)]
         for t in range(n_tasks)], axis=1,
    ).astype(np.float32)
    Y += rng.normal(scale=noise, size=Y.shape).astype(np.float32)
    return X, Y


def _r2(yt, yp):
    r = yt - yp
    return 1 - r.var() / yt.var()


def test_multi_histogram_matches_single_task_replicated():
    # With T identical tasks, the summed split gain is T x a single task's, but
    # the chosen (feature, bin) must be exactly the single-task choice.
    rng = np.random.default_rng(0)
    n, F, T = 1500, 5, 3
    binned = torch.from_numpy(rng.integers(0, 40, size=(n, F)).astype(np.uint8))
    g1 = torch.from_numpy(rng.normal(size=n).astype(np.float32))
    h1 = torch.from_numpy(rng.uniform(0.2, 1.0, size=n).astype(np.float32))

    hist1 = build_histogram(binned, g1, h1)
    gain1, f1, b1 = find_best_split(hist1, 1.0, 0.0, 1e-3, 20)

    grad = g1.unsqueeze(1).repeat(1, T)
    hess = h1.unsqueeze(1).repeat(1, T)
    histT = build_histogram_multi(binned, grad, hess)
    gainT, fT, bT = find_best_split_multi(histT, T, 1.0, 0.0, 1e-3, 20)

    assert (fT, bT) == (f1, b1)
    assert gainT == pytest.approx(T * gain1, rel=1e-4)


def test_multi_histogram_counts_shared():
    rng = np.random.default_rng(1)
    n, F, T = 800, 4, 2
    binned = torch.from_numpy(rng.integers(0, 30, size=(n, F)).astype(np.uint8))
    grad = torch.from_numpy(rng.normal(size=(n, T)).astype(np.float32))
    hess = torch.from_numpy(rng.uniform(0.1, 1, size=(n, T)).astype(np.float32))
    hist = build_histogram_multi(binned, grad, hess)
    # count channel is the last; per feature it must sum to n
    counts = hist[2 * T]
    assert torch.allclose(counts.sum(dim=1), torch.full((F,), float(n)))


def test_predict_shape_and_1d_passthrough():
    X, Y = _multitask_data(1500, n_tasks=3)
    reg = YABTMultiTaskRegressor(n_estimators=20, seed=0).fit(X, Y)
    assert reg.predict(X).shape == (1500, 3)

    reg1 = YABTMultiTaskRegressor(n_estimators=20, seed=0).fit(X, Y[:, 0])
    assert reg1.predict(X).shape == (1500,)


def test_trees_share_structure_with_vector_leaves():
    X, Y = _multitask_data(2000, n_tasks=4)
    reg = YABTMultiTaskRegressor(n_estimators=10, max_leaves=8, seed=0).fit(X, Y)
    for tree in reg.booster_.trees:
        assert tree.value.shape[1] == 4                  # one column per task
        assert tree.value.shape[0] == tree.feature.shape[0]
        # internal nodes carry zero contribution, leaves carry the per-task value
        internal = tree.feature != LEAF
        assert torch.allclose(tree.value[internal], torch.zeros_like(tree.value[internal]))


def test_recovers_multioutput_signal():
    X, Y = _multitask_data(4000, n_tasks=3, noise=0.05)
    Xtr, Ytr, Xte, Yte = X[:3000], Y[:3000], X[3000:], Y[3000:]
    reg = YABTMultiTaskRegressor(n_estimators=100, max_leaves=16, seed=0).fit(Xtr, Ytr)
    P = reg.predict(Xte)
    for t in range(3):
        assert _r2(Yte[:, t], P[:, t]) > 0.9


def test_multitask_helps_when_data_scarce():
    # Correlated tasks + little data: sharing structure should not lose, and
    # generally helps (regularization through shared splits). Big shared test
    # set from the same generator (seed) gives a low-variance comparison.
    Xall, Yall = _multitask_data(1750, n_tasks=6, noise=0.3, share=1.0, seed=3)
    Xtr, Ytr, Xte, Yte = Xall[:250], Yall[:250], Xall[250:], Yall[250:]
    kw = dict(n_estimators=60, max_leaves=8, learning_rate=0.1, seed=0)
    mt = YABTMultiTaskRegressor(**kw).fit(Xtr, Ytr)
    Pmt = mt.predict(Xte)
    Pind = np.stack([
        YABTRegressor(neural_leaves=False, interaction_aware=False, **kw)
        .fit(Xtr, Ytr[:, t]).predict(Xte) for t in range(6)], axis=1)
    mt_r2 = np.mean([_r2(Yte[:, t], Pmt[:, t]) for t in range(6)])
    ind_r2 = np.mean([_r2(Yte[:, t], Pind[:, t]) for t in range(6)])
    assert mt_r2 >= ind_r2 - 0.005, f"multitask {mt_r2:.4f} vs independent {ind_r2:.4f}"


def test_early_stopping_with_eval_set():
    X, Y = _multitask_data(3000, n_tasks=3, seed=4)
    Xtr, Ytr, Xv, Yv = X[:2000], Y[:2000], X[2000:2500], Y[2000:2500]
    reg = YABTMultiTaskRegressor(n_estimators=200, early_stopping_rounds=10, seed=0)
    reg.fit(Xtr, Ytr, eval_set=(Xv, Yv))
    assert reg.booster_.best_iter is not None
    assert len(reg.booster_.trees) <= 200


def test_deterministic():
    X, Y = _multitask_data(1500, n_tasks=3, seed=5)
    a = YABTMultiTaskRegressor(n_estimators=30, seed=0).fit(X, Y).predict(X)
    b = YABTMultiTaskRegressor(n_estimators=30, seed=0).fit(X, Y).predict(X)
    np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-5)


def test_unrelated_tasks_do_not_break():
    X, Y = _multitask_data(2000, n_tasks=4, share=0.0, noise=0.1, seed=6)
    Xtr, Ytr, Xte, Yte = X[:1500], Y[:1500], X[1500:], Y[1500:]
    reg = YABTMultiTaskRegressor(n_estimators=60, max_leaves=16, seed=0).fit(Xtr, Ytr)
    P = reg.predict(Xte)
    assert np.all(np.isfinite(P))
    # each task still learns its own private feature reasonably
    assert np.mean([_r2(Yte[:, t], P[:, t]) for t in range(4)]) > 0.5


def test_sklearn_clone_and_params():
    from sklearn.base import clone
    reg = YABTMultiTaskRegressor(n_estimators=42, max_leaves=7)
    p = clone(reg).get_params()
    assert p["n_estimators"] == 42 and p["max_leaves"] == 7


def test_single_task_regressor_redirects_multioutput_target():
    X, Y = _multitask_data(800, n_tasks=3)
    with pytest.raises(ValueError, match="YABTMultiTaskRegressor"):
        YABTRegressor(n_estimators=10).fit(X, Y)
    # a column vector is still accepted as a single target (sklearn convention)
    single = YABTRegressor(n_estimators=10, seed=0).fit(X, Y[:, [0]])
    assert single.predict(X).shape == (800,)

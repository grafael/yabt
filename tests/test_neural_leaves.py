"""Tests for neural leaf networks (per-leaf linear/MLP models)."""

import numpy as np

from yabt import YABTClassifier, YABTRegressor


def _piecewise_linear(n, seed=0):
    """Different linear functions on each side of a step: constant leaves need
    many splits, one linear leaf per region nails it."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(-2, 2, size=(n, 3)).astype(np.float32)
    y = np.where(X[:, 0] > 0, 2.0 * X[:, 1] - X[:, 2], -1.5 * X[:, 1] + 0.5 * X[:, 2])
    y += rng.normal(scale=0.05, size=n)
    return X, y.astype(np.float32)


def _r2(model, X, y):
    r = y - model.predict(X)
    return 1 - r.var() / y.var()


def test_linear_leaves_beat_constants_on_piecewise_linear():
    X, y = _piecewise_linear(4000)
    Xtr, ytr, Xte, yte = X[:3000], y[:3000], X[3000:], y[3000:]
    kw = dict(n_estimators=10, max_leaves=4, learning_rate=0.5, refine_steps=0, seed=0)

    const = YABTRegressor(neural_leaves=False, **kw).fit(Xtr, ytr)
    lin = YABTRegressor(neural_leaves=True, leaf_net_hidden=0, **kw).fit(Xtr, ytr)

    r2_const, r2_lin = _r2(const, Xte, yte), _r2(lin, Xte, yte)
    assert r2_lin > 0.92, f"linear leaves R^2 {r2_lin:.3f}"
    assert r2_lin > r2_const + 0.02, f"const {r2_const:.3f} vs linear {r2_lin:.3f}"
    assert any(t.leaf_net_linear is not None for t in lin.booster_.trees)


def test_mlp_leaves_fit_smooth_nonlinearity():
    rng = np.random.default_rng(1)
    X = rng.uniform(-2, 2, size=(4000, 2)).astype(np.float32)
    y = (np.sin(2.0 * X[:, 0]) + 0.3 * X[:, 1]).astype(np.float32)
    Xtr, ytr, Xte, yte = X[:3000], y[:3000], X[3000:], y[3000:]

    mlp = YABTRegressor(
        n_estimators=20, max_leaves=4, learning_rate=0.3, refine_steps=0,
        neural_leaves=True, leaf_net_hidden=4, seed=0,
    ).fit(Xtr, ytr)
    r2 = _r2(mlp, Xte, yte)
    assert r2 > 0.9, f"MLP leaves R^2 {r2:.3f}"
    assert any(t.leaf_net_W1 is not None for t in mlp.booster_.trees)
    # Reproducible given the same seed (atol absorbs CUDA atomic-add jitter).
    again = YABTRegressor(
        n_estimators=20, max_leaves=4, learning_rate=0.3, refine_steps=0,
        neural_leaves=True, leaf_net_hidden=4, seed=0,
    ).fit(Xtr, ytr)
    np.testing.assert_allclose(mlp.predict(Xte), again.predict(Xte), rtol=1e-4, atol=1e-5)


def test_classifier_with_neural_leaves():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(3000, 5)).astype(np.float32)
    logit = 1.5 * X[:, 0] - 2.0 * X[:, 1] + X[:, 2]
    y = (logit + rng.logistic(size=3000) > 0).astype(np.float32)

    clf = YABTClassifier(
        n_estimators=30, max_leaves=8, neural_leaves=True, leaf_net_hidden=0,
        refine_steps=0, seed=0,
    ).fit(X[:2000], y[:2000])
    acc = float((clf.predict(X[2000:]) == y[2000:]).mean())
    assert acc >= 0.8, f"accuracy {acc:.3f}"
    proba = clf.predict_proba(X[2000:])
    assert np.all(np.isfinite(proba)) and proba.shape == (1000, 2)


def test_neural_leaves_compose_with_global_refit_and_refinement():
    X, y = _piecewise_linear(3000, seed=3)
    reg = YABTRegressor(
        n_estimators=20, max_leaves=8, neural_leaves=True, leaf_net_hidden=0,
        refine_steps=3, refit_every=5, seed=0,
    ).fit(X, y)
    pred = reg.predict(X)
    assert np.all(np.isfinite(pred))
    assert _r2(reg, X, y) > 0.9


def test_small_leaves_keep_constant_values():
    X, y = _piecewise_linear(2000, seed=4)
    reg = YABTRegressor(
        n_estimators=5, max_leaves=8, neural_leaves=True, leaf_net_hidden=0,
        leaf_net_min_samples=10**9, refine_steps=0, seed=0,  # nothing is eligible
    ).fit(X, y)
    assert all(t.leaf_net_feats is None for t in reg.booster_.trees)


def test_enabled_by_default_and_explicit_opt_out():
    X, y = _piecewise_linear(1500, seed=5)
    on = YABTRegressor(n_estimators=5, refine_steps=0, seed=0).fit(X, y)
    assert any(t.leaf_net_feats is not None for t in on.booster_.trees)
    off = YABTRegressor(n_estimators=5, neural_leaves=False, refine_steps=0, seed=0).fit(X, y)
    assert all(t.leaf_net_feats is None for t in off.booster_.trees)


def test_leaf_nets_do_not_extrapolate_on_outliers():
    # Leaf regions are unbounded; a linear leaf model evaluated far outside
    # its training range must be clamped, not extrapolated.
    X, y = _piecewise_linear(3000, seed=7)
    reg = YABTRegressor(n_estimators=20, neural_leaves=True, leaf_net_hidden=0,
                        refine_steps=0, seed=0).fit(X, y)
    X_out = X[:50].copy()
    X_out[:, 1] = 1e6  # absurd outlier on a net input feature
    pred = reg.predict(X_out)
    assert np.all(np.isfinite(pred))
    assert np.abs(pred).max() < 10 * np.abs(y).max(), f"max |pred| {np.abs(pred).max():.1f}"


def test_neural_leaves_with_kernel_splits():
    # Both novel features active at once: kernel routing + per-leaf models.
    rng = np.random.default_rng(6)
    X = rng.uniform(-1, 1, size=(3000, 3)).astype(np.float32)
    y = (np.where((X[:, :2] ** 2).sum(1) < 0.5, 1.0, -1.0) * X[:, 2]).astype(np.float32)
    reg = YABTRegressor(
        n_estimators=30, max_leaves=8, kernel_splits=True, neural_leaves=True,
        leaf_net_hidden=0, refine_steps=0, seed=0,
    ).fit(X, y)
    assert np.all(np.isfinite(reg.predict(X)))

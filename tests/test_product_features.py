"""Tests for gradient-guided multiplicative feature construction."""

import numpy as np
import pytest

from yabt import YABTClassifier, YABTRegressor
from yabt.product_features import detect_product_specs, expand_products


def _mult3(n=4000, d=8, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32)
    y = (X[:, 0] * X[:, 1] * X[:, 2] + 0.1 * rng.standard_normal(n)).astype(np.float32)
    return X, y


def test_detects_multiplicative_group():
    """The {0,1,2} triple (zero marginal correlation) is recovered via the
    magnitude signal and surfaces as a kept product."""
    X, y = _mult3()
    specs = detect_product_specs(X, y)
    assert specs, "no products detected on a 3-way multiplicative target"
    # The strongest kept product should be exactly the driving triple.
    assert (0, 1, 2) in specs


def test_no_products_on_additive_data():
    """Purely additive / linear targets have no multiplicative structure, so the
    correlation guard keeps nothing (exact neutrality, no extra columns)."""
    rng = np.random.default_rng(1)
    X = rng.standard_normal((4000, 8)).astype(np.float32)
    y = (X @ rng.standard_normal(8) + 0.1 * rng.standard_normal(4000)).astype(np.float32)
    assert detect_product_specs(X, y) == []


def test_expand_products_shapes_and_values():
    X = np.arange(12, dtype=np.float32).reshape(4, 3)
    out = expand_products(X, [(0, 2), (0, 1, 2)])
    assert out.shape == (4, 5)
    np.testing.assert_allclose(out[:, 3], X[:, 0] * X[:, 2], rtol=1e-6)
    np.testing.assert_allclose(out[:, 4], X[:, 0] * X[:, 1] * X[:, 2], rtol=1e-6)
    # empty specs is a no-op (same columns)
    np.testing.assert_array_equal(expand_products(X, []), X)


def test_degenerate_inputs_return_no_specs():
    rng = np.random.default_rng(2)
    X = rng.standard_normal((50, 4)).astype(np.float32)
    # too few rows
    assert detect_product_specs(X, rng.standard_normal(50)) == []
    # constant target
    big = rng.standard_normal((500, 4)).astype(np.float32)
    assert detect_product_specs(big, np.ones(500, dtype=np.float32)) == []


def test_regressor_improves_on_multiplicative_target():
    """End-to-end: enabling product_features substantially raises R^2 on a
    multiplicative target the greedy splitter cannot otherwise fit."""
    Xtr, ytr = _mult3(seed=0)
    Xte, yte = _mult3(seed=99)

    base = YABTRegressor(n_estimators=120, learning_rate=0.1, seed=0).fit(Xtr, ytr)
    prod = YABTRegressor(n_estimators=120, learning_rate=0.1, seed=0,
                         product_features=True).fit(Xtr, ytr)

    def r2(m):
        p = m.predict(Xte)
        return 1 - ((yte - p) ** 2).sum() / ((yte - yte.mean()) ** 2).sum()

    r2_base, r2_prod = r2(base), r2(prod)
    assert prod.booster_.product_spec_, "product features were expected to fire"
    # A decisive lift well beyond the 10% target on this regime.
    assert r2_prod > r2_base + 0.10


def test_predict_roundtrips_with_products():
    """Predictions are well-formed and deterministic with products enabled."""
    Xtr, ytr = _mult3(seed=3)
    m = YABTRegressor(n_estimators=40, seed=0, product_features=True).fit(Xtr, ytr)
    p1 = m.predict(Xtr[:100])
    p2 = m.predict(Xtr[:100])
    assert p1.shape == (100,)
    np.testing.assert_array_equal(p1, p2)


def test_classifier_accepts_product_features():
    rng = np.random.default_rng(4)
    X = rng.standard_normal((3000, 6)).astype(np.float32)
    y = (X[:, 0] * X[:, 1] * X[:, 2] > 0).astype(int)
    m = YABTClassifier(n_estimators=60, seed=0, product_features=True).fit(X, y)
    acc = (m.predict(X) == y).mean()
    assert acc > 0.8

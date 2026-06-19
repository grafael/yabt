"""The OpenMP C grower must produce bit-identical trees to the Numba grower
(same split math, per-feature accumulation order preserved), and the ``c_grower``
gate must route to it and fall back cleanly. Skipped entirely if no C compiler
is available on the box."""

import numpy as np
import torch
import pytest

from sklearn.datasets import make_regression, make_classification

from yabt.binning import Binner
from yabt.tree import TreeParams
from yabt.grow_numba import grow_tree_numba, build_sparse_layout
from yabt.grow_c import is_available, grow_tree_c
from yabt.boosting import Booster, BoostParams, MSELoss, LogLoss

pytestmark = pytest.mark.skipif(
    not is_available(), reason="no C compiler available for the C grower")


def _binned(X):
    binner = Binner().fit(X)
    return binner, binner.transform(X)


def _assert_identical(ta, tb, tol=1e-5):
    assert ta.feature.shape == tb.feature.shape
    assert bool((ta.feature == tb.feature).all())
    assert bool((ta.left == tb.left).all())
    assert bool((ta.right == tb.right).all())
    assert float((ta.threshold - tb.threshold).abs().max()) <= tol
    assert float((ta.value - tb.value).abs().max()) <= tol


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("threads", [1, 4])
def test_matches_numba_bare(seed, threads):
    X, y = make_regression(n_samples=3000, n_features=12, n_informative=8,
                           noise=1.0, random_state=seed)
    X = X.astype(np.float32); y = y.astype(np.float32)
    binner, binned = _binned(X)
    grad = torch.from_numpy((y - y.mean()).astype(np.float32))
    hess = torch.ones(len(y))
    tp = TreeParams(max_leaves=31)
    ta = grow_tree_numba(binned, grad, hess, binner, tp)
    tb = grow_tree_c(binned, grad, hess, binner, tp, n_threads=threads)
    _assert_identical(ta, tb)


def test_matches_numba_with_interaction():
    X, y = make_regression(n_samples=3000, n_features=12, n_informative=8,
                           noise=1.0, random_state=0)
    X = X.astype(np.float32); y = y.astype(np.float32)
    binner, binned = _binned(X)
    grad = torch.from_numpy((y - y.mean()).astype(np.float32))
    hess = torch.ones(len(y))
    tp = TreeParams(max_leaves=31)
    F = X.shape[1]
    rng = np.random.default_rng(0)
    imat = torch.from_numpy(rng.random((F, F)).astype(np.float32))
    ta = grow_tree_numba(binned, grad, hess, binner, tp, interaction_matrix=imat,
                         interaction_boost=0.5)
    tb = grow_tree_c(binned, grad, hess, binner, tp, interaction_matrix=imat,
                     interaction_boost=0.5, n_threads=4)
    _assert_identical(ta, tb)


def test_matches_numba_sparse():
    rng = np.random.default_rng(1)
    n, F = 3000, 40
    X = np.zeros((n, F), dtype=np.float32)
    m = rng.random((n, F)) < 0.1
    X[m] = rng.standard_normal(int(m.sum())).astype(np.float32)
    y = (X[:, 0] + 2 * X[:, 5]).astype(np.float32)
    binner, binned = _binned(X)
    grad = torch.from_numpy((y - y.mean()).astype(np.float32))
    hess = torch.ones(n)
    tp = TreeParams(max_leaves=31)
    sl = build_sparse_layout(binned)
    ta = grow_tree_numba(binned, grad, hess, binner, tp, sparse_layout=sl)
    tb = grow_tree_c(binned, grad, hess, binner, tp, sparse_layout=sl, n_threads=4)
    _assert_identical(ta, tb)


def test_feature_mask_respected():
    X, y = make_regression(n_samples=2000, n_features=10, n_informative=8,
                           noise=1.0, random_state=0)
    X = X.astype(np.float32); y = y.astype(np.float32)
    binner, binned = _binned(X)
    grad = torch.from_numpy((y - y.mean()).astype(np.float32))
    hess = torch.ones(len(y))
    mask = torch.zeros(10, dtype=torch.bool)
    mask[[1, 3, 5]] = True
    tree = grow_tree_c(binned, grad, hess, binner, TreeParams(max_leaves=16),
                       feature_mask=mask, n_threads=4)
    used = set(int(f) for f in tree.feature.tolist() if f >= 0)
    assert used <= {1, 3, 5}


def test_gate_produces_equivalent_model():
    # End-to-end: c_grower on vs off (Numba) -> near-identical predictions.
    X, y = make_regression(n_samples=4000, n_features=20, n_informative=14,
                           noise=1.0, random_state=0)
    X = X.astype(np.float32); y = y.astype(np.float32)
    Xtr, Xte, ytr, yte = X[:3200], X[3200:], y[:3200], y[3200:]

    def fit(c_flag):
        b = Booster(BoostParams(n_estimators=40, device="cpu", seed=0,
                                c_grower=c_flag, c_grower_threads=4), MSELoss())
        return b.fit(Xtr, ytr).predict_margin(Xte)

    p_on = fit(True)
    p_off = fit(False)
    assert np.corrcoef(p_on, p_off)[0, 1] > 0.9999


def test_classification_gate():
    X, y = make_classification(n_samples=4000, n_features=20, n_informative=12,
                               random_state=0)
    X = X.astype(np.float32); y = y.astype(np.float32)
    b = Booster(BoostParams(n_estimators=40, device="cpu", seed=0,
                            c_grower=True, c_grower_threads=4), LogLoss())
    b.fit(X[:3200], y[:3200])
    acc = ((b.predict_margin(X[3200:]) > 0).astype(int) == y[3200:]).mean()
    assert acc > 0.8

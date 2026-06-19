"""The Numba grower must match the torch heap grower's split math, and the
``numba_grower`` gate must route to it correctly."""

import numpy as np
import torch
import pytest

from sklearn.datasets import make_regression, make_classification

from yabt.binning import Binner
from yabt.tree import grow_tree, TreeParams
from yabt.grow_numba import grow_tree_numba
from yabt.boosting import Booster, BoostParams, MSELoss, LogLoss


def _binned(X, y):
    binner = Binner().fit(X)
    return binner, binner.transform(X)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_matches_torch_grower_bare(seed):
    X, y = make_regression(n_samples=3000, n_features=12, n_informative=8,
                           noise=1.0, random_state=seed)
    X = X.astype(np.float32); y = y.astype(np.float32)
    binner, binned = _binned(X, y)
    grad = torch.from_numpy((y - y.mean()).astype(np.float32))
    hess = torch.ones(len(y))
    tp = TreeParams(max_leaves=31)

    ta = grow_tree(binned, grad, hess, binner, tp)
    tb = grow_tree_numba(binned, grad, hess, binner, tp)

    assert ta.feature.shape == tb.feature.shape
    assert bool((ta.feature == tb.feature).all())
    Xraw = torch.from_numpy(binner.impute(X))
    assert float((ta.predict(Xraw) - tb.predict(Xraw)).abs().max()) < 1e-3


def test_matches_torch_grower_with_interaction():
    X, y = make_regression(n_samples=3000, n_features=12, n_informative=8,
                           noise=1.0, random_state=0)
    X = X.astype(np.float32); y = y.astype(np.float32)
    binner, binned = _binned(X, y)
    grad = torch.from_numpy((y - y.mean()).astype(np.float32))
    hess = torch.ones(len(y))
    tp = TreeParams(max_leaves=31)
    F = X.shape[1]
    rng = np.random.default_rng(0)
    imat = torch.from_numpy(rng.random((F, F)).astype(np.float32))

    ta = grow_tree(binned, grad, hess, binner, tp, interaction_matrix=imat,
                   interaction_boost=0.5)
    tb = grow_tree_numba(binned, grad, hess, binner, tp, interaction_matrix=imat,
                         interaction_boost=0.5)
    assert bool((ta.feature == tb.feature).all())


def test_feature_mask_respected():
    X, y = make_regression(n_samples=2000, n_features=10, n_informative=8,
                           noise=1.0, random_state=0)
    X = X.astype(np.float32); y = y.astype(np.float32)
    binner, binned = _binned(X, y)
    grad = torch.from_numpy((y - y.mean()).astype(np.float32))
    hess = torch.ones(len(y))
    mask = torch.zeros(10, dtype=torch.bool)
    mask[[1, 3, 5]] = True
    tree = grow_tree_numba(binned, grad, hess, binner, TreeParams(max_leaves=16),
                           feature_mask=mask)
    used = set(int(f) for f in tree.feature.tolist() if f >= 0)
    assert used <= {1, 3, 5}


def test_gate_produces_equivalent_model():
    # End-to-end: numba grower on vs off should give near-identical predictions.
    X, y = make_regression(n_samples=4000, n_features=20, n_informative=14,
                           noise=1.0, random_state=0)
    X = X.astype(np.float32); y = y.astype(np.float32)
    Xtr, Xte, ytr, yte = X[:3200], X[3200:], y[:3200], y[3200:]

    def fit(flag):
        b = Booster(BoostParams(n_estimators=40, device="cpu", seed=0,
                                numba_grower=flag), MSELoss())
        return b.fit(Xtr, ytr).predict_margin(Xte)

    p_on = fit(True)
    p_off = fit(False)
    # Same growth order and split math -> models agree to a tight tolerance.
    assert np.corrcoef(p_on, p_off)[0, 1] > 0.999


def test_classification_gate():
    X, y = make_classification(n_samples=4000, n_features=20, n_informative=12,
                               random_state=0)
    X = X.astype(np.float32); y = y.astype(np.float32)
    b = Booster(BoostParams(n_estimators=40, device="cpu", seed=0,
                            numba_grower=True), LogLoss())
    b.fit(X[:3200], y[:3200])
    acc = ((b.predict_margin(X[3200:]) > 0).astype(int) == y[3200:]).mean()
    assert acc > 0.8

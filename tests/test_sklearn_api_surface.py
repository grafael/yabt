"""The API surface a user actually hits: sample weights, sparse input, and
round-tripping a fitted model through pickle."""

import pickle

import numpy as np
import pytest
import scipy.sparse as sp
import torch
from sklearn.datasets import make_classification, make_regression
from sklearn.metrics import roc_auc_score

from yabt import YABTClassifier, YABTRegressor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
KW = dict(n_estimators=40, device=DEVICE, early_stopping_rounds=0,
          validation_fraction=0.0)


def test_zero_weighted_rows_stop_contributing():
    """Zero-weighting half the rows must pull the fit towards the model trained
    on the other half.

    Not to an exact match: weights scale the gradients and Hessians, but the
    binner's quantile edges, ``min_samples_leaf`` and ``leaf_net_min_samples``
    all still count zero-weight rows, so a zero weight silences a row's
    influence on the objective without removing it from the data.
    """
    X, y = make_classification(n_samples=2000, n_features=10, n_informative=6,
                               random_state=0)
    X = X.astype(np.float32)
    keep = np.zeros(len(y), dtype=bool)
    keep[::2] = True
    weighted = YABTClassifier(**KW).fit(X, y, sample_weight=keep.astype(np.float32))
    subset = YABTClassifier(**KW).fit(X[keep], y[keep])
    plain = YABTClassifier(**KW).fit(X, y)
    ref = subset.predict_proba(X[:400])[:, 1]
    d_weighted = np.abs(weighted.predict_proba(X[:400])[:, 1] - ref).mean()
    d_plain = np.abs(plain.predict_proba(X[:400])[:, 1] - ref).mean()
    assert d_weighted < d_plain


def test_sample_weight_upweights_a_minority():
    """Weighting the rare class up must move its predicted scores up."""
    rng = np.random.default_rng(0)
    n = 4000
    X = rng.normal(size=(n, 6)).astype(np.float32)
    y = (X[:, 0] + 0.5 * rng.normal(size=n) > 1.6).astype(np.float32)
    w = np.where(y == 1, 20.0, 1.0).astype(np.float32)
    plain = YABTClassifier(**KW).fit(X, y)
    up = YABTClassifier(**KW).fit(X, y, sample_weight=w)
    assert up.predict_proba(X)[:, 1].mean() > plain.predict_proba(X)[:, 1].mean()
    # Both should still rank well; weighting is not supposed to break the model.
    assert roc_auc_score(y, up.predict_proba(X)[:, 1]) > 0.9


def test_sample_weight_regressor_and_validation():
    X, y = make_regression(n_samples=1500, n_features=8, noise=1.0, random_state=0)
    X, y = X.astype(np.float32), y.astype(np.float32)
    w = np.linspace(0.5, 2.0, len(y)).astype(np.float32)
    r = YABTRegressor(**KW).fit(X, y, sample_weight=w)
    assert np.isfinite(r.predict(X)).all()
    with pytest.raises(ValueError):
        YABTRegressor(**KW).fit(X, y, sample_weight=w[:-1])
    with pytest.raises(ValueError):
        YABTRegressor(**KW).fit(X, y, sample_weight=-w)


def test_sample_weight_multiclass():
    X, y = make_classification(n_samples=2000, n_features=10, n_informative=6,
                               n_classes=3, n_clusters_per_class=1, random_state=0)
    X = X.astype(np.float32)
    w = np.where(y == 2, 5.0, 1.0).astype(np.float32)
    m = YABTClassifier(**KW).fit(X, y, sample_weight=w)
    P = m.predict_proba(X)
    assert P.shape == (2000, 3) and np.allclose(P.sum(axis=1), 1.0, atol=1e-5)


def test_sample_weight_survives_the_auto_validation_split():
    X, y = make_classification(n_samples=3000, n_features=8, random_state=0)
    X = X.astype(np.float32)
    w = np.ones(len(y), dtype=np.float32)
    m = YABTClassifier(n_estimators=200, early_stopping_rounds=20,
                       device=DEVICE).fit(X, y, sample_weight=w)
    assert np.isfinite(m.predict_proba(X)).all()


def test_sparse_input_accepted():
    X, y = make_classification(n_samples=1500, n_features=20, n_informative=8,
                               random_state=0)
    X = X.astype(np.float32)
    X[np.abs(X) < 1.0] = 0.0
    dense = YABTClassifier(**KW).fit(X, y)
    sparse = YABTClassifier(**KW).fit(sp.csr_matrix(X), y)
    assert np.allclose(dense.predict_proba(X), sparse.predict_proba(sp.csr_matrix(X)),
                       atol=1e-5)


def test_pickle_round_trip():
    X, y = make_classification(n_samples=2000, n_features=10, n_informative=6,
                               random_state=0)
    X = X.astype(np.float32)
    m = YABTClassifier(n_ensemble=2, subsample=0.8, **KW).fit(X, y)
    before = m.predict_proba(X[:300])
    m2 = pickle.loads(pickle.dumps(m))
    assert np.array_equal(before, m2.predict_proba(X[:300]))
    assert list(m2.classes_) == list(m.classes_)


def test_pickle_round_trip_regressor():
    X, y = make_regression(n_samples=1200, n_features=8, noise=1.0, random_state=0)
    X, y = X.astype(np.float32), y.astype(np.float32)
    r = YABTRegressor(**KW).fit(X, y)
    r2 = pickle.loads(pickle.dumps(r))
    assert np.array_equal(r.predict(X[:300]), r2.predict(X[:300]))

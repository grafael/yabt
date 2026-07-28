"""Missing values get their own bin above every real one, and the raw-space
threshold a split is stored as must route them the same way the binned matrix
does -- otherwise training and inference disagree on every NaN row."""

import numpy as np
import pytest
import torch
from sklearn.metrics import roc_auc_score

from yabt import YABTClassifier, YABTRegressor
from yabt.binning import Binner

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _nan_data(seed=0, n=4000, F=6, frac=0.15):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, F)).astype(np.float32)
    X[rng.random(X.shape) < frac] = np.nan
    X[:, F - 1] = 1.0  # constant, never missing: must keep its full bin budget
    return X


def test_nan_gets_its_own_top_bin():
    X = _nan_data()
    b = Binner(max_bins=32).fit(X)
    binned = b.transform(X).numpy()
    assert binned.max() <= 255
    for f in range(X.shape[1] - 1):
        m = np.isnan(X[:, f])
        assert b.nan_bin_[f] >= 0
        assert (binned[m, f] == b.nan_bin_[f]).all()
        assert binned[~m, f].max() < b.nan_bin_[f]
    # A column with no training NaNs spends no bin on one.
    assert b.nan_bin_[X.shape[1] - 1] == -1


def test_binned_and_raw_thresholds_agree():
    """For every *valid* split position, "bin <= b" and "x <= edge_value(f, b)"
    must select exactly the same rows -- including the missing ones. The last
    used bin is never a split position, so it is excluded."""
    X = _nan_data()
    b = Binner(max_bins=32).fit(X)
    binned = b.transform(X).numpy()
    Xi = b.impute(X)
    assert not np.isnan(Xi).any()
    for f in range(X.shape[1]):
        for bi in range(int(b.used_bins()[f]) - 1):
            thr = b.edge_value(f, bi)
            assert ((binned[:, f] <= bi) == (Xi[:, f] <= thr)).all(), (f, bi)


def test_missing_at_predict_time_in_a_clean_column():
    """A column with no training NaNs has no reserved bin; a NaN arriving at
    predict time must still bin and route consistently (as "above the max")."""
    rng = np.random.default_rng(1)
    X = rng.normal(size=(2000, 4)).astype(np.float32)
    b = Binner(max_bins=16).fit(X)
    assert (b.nan_bin_ == -1).all()
    Xq = X[:50].copy()
    Xq[:, 0] = np.nan
    binned = b.transform(Xq).numpy()
    Xi = b.impute(Xq)
    assert not np.isnan(Xi).any()
    for bi in range(int(b.used_bins()[0]) - 1):
        thr = b.edge_value(0, bi)
        assert ((binned[:, 0] <= bi) == (Xi[:, 0] <= thr)).all()


@pytest.mark.parametrize("device", [DEVICE])
def test_informative_missingness_is_learnable(device):
    """The whole point: when *which* values are missing carries the signal,
    median imputation hides it and a dedicated bin exposes it."""
    rng = np.random.default_rng(0)
    n = 6000
    X = rng.normal(size=(n, 8)).astype(np.float32)
    y = (rng.random(n) < 0.5).astype(np.float32)
    # Missingness in column 0 is the label; the observed values carry nothing.
    miss = y.astype(bool) ^ (rng.random(n) < 0.05)
    X[miss, 0] = np.nan
    tr, te = slice(0, 4000), slice(4000, n)
    m = YABTClassifier(n_estimators=60, max_leaves=8, device=device,
                       early_stopping_rounds=0, validation_fraction=0.0)
    m.fit(X[tr], y[tr])
    auc = roc_auc_score(y[te], m.predict_proba(X[te])[:, 1])
    assert auc > 0.9, auc


def test_all_nan_column_is_harmless():
    X = _nan_data(n=1500, F=4)
    X[:, 1] = np.nan
    y = (X[:, 0] > 0).astype(np.float32)
    y[np.isnan(y)] = 0.0
    r = YABTRegressor(n_estimators=20, device=DEVICE, early_stopping_rounds=0,
                      validation_fraction=0.0).fit(X, y)
    assert np.isfinite(r.predict(X)).all()

"""Small-data capacity caps and seed ensembling."""

import numpy as np
import torch
from sklearn.datasets import make_classification, make_regression

from yabt import YABTClassifier, YABTRegressor
from yabt.auto_tune import _candidates
from yabt.boosting import (
    Booster,
    BoostParams,
    MSELoss,
    SeedEnsemble,
    _SMALL_N_MAX_DEPTH,
    _SMALL_N_MAX_LEAVES,
    _SMALL_N_ROWS,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _tp(n, **kw):
    kw.setdefault("small_data_caps", True)
    return Booster(BoostParams(**kw), MSELoss())._tree_params(n)


def test_caps_apply_only_below_the_row_threshold():
    small = _tp(_SMALL_N_ROWS - 1)
    assert small.max_leaves == _SMALL_N_MAX_LEAVES
    assert small.max_depth == _SMALL_N_MAX_DEPTH
    big = _tp(_SMALL_N_ROWS)
    assert big.max_leaves == BoostParams().max_leaves
    assert big.max_depth == BoostParams().max_depth
    # No row count (multi-task / direct callers) -> no caps.
    assert _tp(None).max_leaves == BoostParams().max_leaves


def test_caps_never_inflate_a_smaller_budget():
    tp = _tp(500, max_leaves=4, max_depth=2)
    assert tp.max_leaves == 4 and tp.max_depth == 2


def test_caps_are_off_by_default():
    """Opt-in: across eight sub-2000-row datasets the caps are a median +0.36%
    with a regression tail, so they ship as an auto_tune candidate, not a
    default. Only the explicit flag turns them on."""
    assert BoostParams().small_data_caps is False
    assert (Booster(BoostParams(), MSELoss())._tree_params(500).max_leaves
            == BoostParams().max_leaves)
    assert _tp(500, small_data_caps=False).max_leaves == BoostParams().max_leaves


def test_capped_model_still_trains_and_predicts():
    X, y = make_regression(n_samples=800, n_features=10, noise=1.0, random_state=0)
    r = YABTRegressor(n_estimators=40, device=DEVICE, small_data_caps=True,
                      early_stopping_rounds=0,
                      validation_fraction=0.0).fit(X.astype(np.float32),
                                                   y.astype(np.float32))
    assert max(t.num_leaves() for t in r.booster_.trees) <= _SMALL_N_MAX_LEAVES
    assert r.score(X.astype(np.float32), y.astype(np.float32)) > 0.5


def test_seed_ensemble_averages_distinct_members():
    X, y = make_classification(n_samples=3000, n_features=12, n_informative=8,
                               random_state=0)
    X = X.astype(np.float32)
    kw = dict(n_estimators=30, subsample=0.7, colsample=0.7, device=DEVICE,
              early_stopping_rounds=0, validation_fraction=0.0)
    ens = YABTClassifier(n_ensemble=3, **kw).fit(X, y)
    assert isinstance(ens.booster_, SeedEnsemble)
    assert len(ens.booster_.members_) == 3
    # Members must actually differ (different seeds -> different subsamples),
    # otherwise the ensemble is 3x the cost for nothing.
    margins = [m.predict_margin(X[:200]) for m in ens.booster_.members_]
    assert not np.allclose(margins[0], margins[1])
    # ...and the ensemble must be their mean, not just the first member.
    assert np.allclose(ens.booster_.predict_margin(X[:200]),
                       np.mean(margins, axis=0), atol=1e-5)
    # Attribute lookups fall through to a representative member.
    assert ens.booster_.binner is ens.booster_.members_[0].binner


def test_n_ensemble_one_is_the_plain_booster():
    X, y = make_classification(n_samples=1000, n_features=8, random_state=0)
    m = YABTClassifier(n_estimators=10, device=DEVICE, early_stopping_rounds=0,
                       validation_fraction=0.0).fit(X.astype(np.float32), y)
    assert isinstance(m.booster_, Booster)


def test_seed_ensemble_multiclass():
    X, y = make_classification(n_samples=2000, n_features=12, n_informative=8,
                               n_classes=3, n_clusters_per_class=1, random_state=0)
    X = X.astype(np.float32)
    m = YABTClassifier(n_ensemble=2, n_estimators=20, subsample=0.8, device=DEVICE,
                       early_stopping_rounds=0, validation_fraction=0.0).fit(X, y)
    P = m.predict_proba(X[:100])
    assert P.shape == (100, 3)
    assert np.allclose(P.sum(axis=1), 1.0, atol=1e-5)


def test_caps_offered_as_an_auto_tune_candidate_only_where_they_bind():
    names = [n for n, _ in _candidates(_SMALL_N_ROWS - 1)]
    assert "small-data-caps" in names
    # Above the threshold the caps do nothing, so the candidate would be a
    # bit-identical duplicate of user-config -- one wasted fit per tune.
    assert "small-data-caps" not in [n for n, _ in _candidates(_SMALL_N_ROWS)]

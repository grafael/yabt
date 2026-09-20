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
    _SMALL_N_LEAF_NET_MIN,
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
    big = _tp(_SMALL_N_ROWS)
    assert big.max_leaves == BoostParams().max_leaves
    # No row count (multi-task / direct callers) -> no caps.
    assert _tp(None).max_leaves == BoostParams().max_leaves


def test_caps_never_inflate_a_smaller_budget():
    tp = _tp(500, max_leaves=4)
    assert tp.max_leaves == 4


def test_caps_are_on_by_default_and_the_flag_turns_them_off():
    """Default: 14 of 17 sub-2500-row TabArena tasks improve, median -3.97%
    metric error, at 0.63x train time. The flag is the escape hatch."""
    assert BoostParams().small_data_caps is True
    assert (Booster(BoostParams(), MSELoss())._tree_params(500).max_leaves
            == _SMALL_N_MAX_LEAVES)
    assert _tp(500, small_data_caps=False).max_leaves == BoostParams().max_leaves


def test_caps_bound_the_leaf_model_floor_too():
    """The second half of the cap: with only 4 regions every leaf is large, so
    the 50-row floor that keeps small leaves constant is miscalibrated."""
    assert _tp(500).max_leaves == _SMALL_N_MAX_LEAVES
    b = Booster(BoostParams(small_data_caps=True), MSELoss())
    assert b.small_data_params(500).leaf_net_min_samples == _SMALL_N_LEAF_NET_MIN
    assert (b.small_data_params(_SMALL_N_ROWS).leaf_net_min_samples
            == BoostParams().leaf_net_min_samples)
    # Lowering the floor *adds* capacity, so unlike max_leaves it is retuned
    # only while it sits at its default -- an explicit setting is never
    # overridden, in either direction.
    for explicit in (5, 10 ** 9):
        b = Booster(BoostParams(small_data_caps=True,
                                leaf_net_min_samples=explicit), MSELoss())
        assert b.small_data_params(500).leaf_net_min_samples == explicit


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
    assert "uncapped-small-data" in names
    # Above the threshold the caps do nothing, so the candidate would be a
    # bit-identical duplicate of user-config -- one wasted fit per tune.
    assert "uncapped-small-data" not in [n for n, _ in _candidates(_SMALL_N_ROWS)]


def test_half_budget_candidate_is_the_mirror_image():
    """15 leaves is a weak-but-real win above the threshold (14/22 tasks) and a
    duplicate of user-config below it, where the cap already binds to 4."""
    assert "half-budget" in [n for n, _ in _candidates(_SMALL_N_ROWS)]
    assert "half-budget" not in [n for n, _ in _candidates(_SMALL_N_ROWS - 1)]

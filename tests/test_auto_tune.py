"""Tests for validation-based auto-tuning."""

import numpy as np

from yabt import YABTClassifier, YABTRegressor


def _classification(n, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 8)).astype(np.float32)
    logit = 1.5 * X[:, 0] - X[:, 1] + X[:, 2] * X[:, 3]
    y = (logit + rng.logistic(size=n) > 0).astype(np.float32)
    return X, y


def test_auto_tune_selects_and_reports():
    X, y = _classification(3000)
    clf = YABTClassifier(n_estimators=80, auto_tune=True, refine_steps=0, seed=0)
    clf.fit(X[:2200], y[:2200])

    rep = clf.booster_.tuning_report_
    assert rep is not None
    names = [r["name"] for r in rep["results"]]
    assert "user-config" in names
    assert rep["selected"] in names
    # results sorted by validation loss, winner first
    losses = [r["val_loss"] for r in rep["results"]]
    assert losses == sorted(losses)
    assert rep["n_estimators"] == len(clf.booster_.trees) or clf.booster_.best_iter is not None

    acc = float((clf.predict(X[2200:]) == y[2200:]).mean())
    assert acc >= 0.72, f"accuracy {acc:.3f}"  # logistic label noise: Bayes ~0.79


def test_auto_tune_not_much_worse_than_default():
    X, y = _classification(4000, seed=1)
    Xtr, ytr, Xte, yte = X[:3000], y[:3000], X[3000:], y[3000:]
    base = YABTClassifier(n_estimators=80, refine_steps=0, seed=0).fit(Xtr, ytr)
    tuned = YABTClassifier(n_estimators=80, auto_tune=True, refine_steps=0, seed=0).fit(Xtr, ytr)
    acc_base = float((base.predict(Xte) == yte).mean())
    acc_tuned = float((tuned.predict(Xte) == yte).mean())
    assert acc_tuned >= acc_base - 0.02, f"base {acc_base:.3f} vs tuned {acc_tuned:.3f}"


def test_auto_tune_skipped_on_tiny_data():
    X, y = _classification(120, seed=2)  # below MIN_ROWS_CV: too few even for CV
    clf = YABTClassifier(n_estimators=30, auto_tune=True, refine_steps=0, seed=0).fit(X, y)
    assert clf.booster_.tuning_report_ is None
    assert clf.booster_.p.auto_tune is False  # no recursive tuning state left


def test_auto_tune_small_data_uses_cv():
    # between MIN_ROWS_CV and MIN_ROWS_TO_TUNE: tuned via k-fold CV, not skipped
    X, y = _classification(400, seed=6)
    clf = YABTClassifier(n_estimators=40, auto_tune=True, refine_steps=0, seed=0).fit(X, y)
    rep = clf.booster_.tuning_report_
    assert rep is not None
    assert rep["scoring"] == "cv"
    assert rep["selected"] in [r["name"] for r in rep["results"]]


def test_auto_tune_uses_external_eval_set():
    X, y = _classification(3000, seed=3)
    Xtr, ytr, Xv, yv = X[:2000], y[:2000], X[2000:2600], y[2000:2600]
    clf = YABTClassifier(n_estimators=60, auto_tune=True, refine_steps=0,
                         early_stopping_rounds=20, seed=0)
    clf.fit(Xtr, ytr, eval_set=(Xv, yv))
    rep = clf.booster_.tuning_report_
    assert rep is not None
    # external eval set: final n_estimators left to caller's early stopping
    assert rep["n_estimators"] == 60


def test_auto_tune_deterministic():
    X, y = _classification(2000, seed=4)
    a = YABTClassifier(n_estimators=40, auto_tune=True, refine_steps=0, seed=0).fit(X, y)
    b = YABTClassifier(n_estimators=40, auto_tune=True, refine_steps=0, seed=0).fit(X, y)
    assert a.booster_.tuning_report_["selected"] == b.booster_.tuning_report_["selected"]


def test_auto_tune_regressor():
    rng = np.random.default_rng(5)
    X = rng.uniform(-2, 2, size=(3000, 5)).astype(np.float32)
    y = (X[:, 0] ** 2 + X[:, 1] * X[:, 2] + 0.1 * rng.normal(size=3000)).astype(np.float32)
    reg = YABTRegressor(n_estimators=80, auto_tune=True, refine_steps=0, seed=0)
    reg.fit(X[:2200], y[:2200])
    r = y[2200:] - reg.predict(X[2200:])
    assert 1 - r.var() / y[2200:].var() > 0.7
    assert reg.booster_.tuning_report_ is not None


def test_min_split_gain_rel_curbs_noise_overfit():
    # On a pure-noise target (y independent of X) the scale-invariant min-gain
    # floor refuses noise splits, so test R2 is no worse than the overfitting
    # default (which goes negative).
    rng = np.random.default_rng(0)
    X = rng.normal(size=(2000, 6)).astype(np.float32)
    y = rng.normal(size=2000).astype(np.float32)
    Xtr, ytr, Xte, yte = X[:1400], y[:1400], X[1400:], y[1400:]
    base = YABTRegressor(n_estimators=100, max_leaves=31, refine_steps=0, seed=0).fit(Xtr, ytr)
    reg = YABTRegressor(n_estimators=100, max_leaves=31, min_split_gain_rel=2.0,
                        refine_steps=0, seed=0).fit(Xtr, ytr)
    def r2(m):
        p = m.predict(Xte)
        return 1 - ((yte - p) ** 2).mean() / yte.var()
    assert r2(reg) > r2(base)


def test_min_split_gain_rel_scale_equivariant():
    # gain and the rho*var(grad) floor both scale with the target variance, so
    # split structure is invariant to a constant rescaling of y and predictions
    # scale linearly with it (this is what a fixed absolute gamma cannot do).
    rng = np.random.default_rng(1)
    X = rng.normal(size=(800, 5)).astype(np.float32)
    y = (X[:, 0] > 0).astype(np.float32) * 3 + 0.1 * rng.normal(size=800).astype(np.float32)
    kw = dict(n_estimators=30, max_leaves=15, reg_lambda=1e-6, neural_leaves=False,
              min_split_gain_rel=1.0, refine_steps=0, seed=0)
    p1 = YABTRegressor(**kw).fit(X, y).predict(X)
    p2 = YABTRegressor(**kw).fit(X, 100.0 * y).predict(X)
    np.testing.assert_allclose(p2, 100.0 * p1, rtol=1e-3, atol=1e-3)


def test_auto_tune_offers_relative_gamma_candidate():
    from yabt.auto_tune import _candidates
    names = {name for name, _ in _candidates(10000)}
    assert {"regularized-splits", "regularized-splits-strong"} <= names
    # the regularized candidates use the scale-invariant floor, not a fixed gamma
    over = dict(_candidates(10000))
    assert "min_split_gain_rel" in over["regularized-splits"]
    assert "gamma" not in over["regularized-splits"]

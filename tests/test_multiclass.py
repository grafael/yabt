"""Multiclass: native softmax booster vs the One-vs-Rest fallback."""

import numpy as np
import torch
from sklearn.datasets import load_digits, make_classification
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split

from yabt import YABTClassifier
from yabt.multiclass import MulticlassBooster, SoftmaxBooster

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _split(n_classes=5, n=4000, seed=0):
    X, y = make_classification(n_samples=n, n_features=20, n_informative=10,
                               n_classes=n_classes, random_state=seed)
    return train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)


def test_softmax_is_default_for_multiclass():
    Xtr, Xte, ytr, yte = _split()
    clf = YABTClassifier(n_estimators=50, device=DEVICE)
    clf.fit(Xtr, ytr)
    assert isinstance(clf.booster_, SoftmaxBooster)


def test_ovr_fallback_param_and_unsupported_features():
    Xtr, Xte, ytr, yte = _split(n=1200)
    clf = YABTClassifier(multiclass="ovr", n_estimators=20, device=DEVICE)
    clf.fit(Xtr, ytr)
    assert isinstance(clf.booster_, MulticlassBooster)
    # softmax requested but an unsupported opt-in forces the OvR fallback
    clf = YABTClassifier(stochastic_routing=True, n_estimators=20, device=DEVICE)
    clf.fit(Xtr, ytr)
    assert isinstance(clf.booster_, MulticlassBooster)


def test_softmax_proba_shape_and_normalization():
    Xtr, Xte, ytr, yte = _split()
    clf = YABTClassifier(n_estimators=60, device=DEVICE)
    clf.fit(Xtr, ytr)
    p = clf.predict_proba(Xte)
    assert p.shape == (len(yte), 5)
    assert np.allclose(p.sum(axis=1), 1.0, atol=1e-5)
    assert set(np.unique(clf.predict(Xte))) <= set(np.unique(ytr))


def test_softmax_accuracy_digits():
    X, y = load_digits(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
    clf = YABTClassifier(n_estimators=150, device=DEVICE)
    clf.fit(Xtr, ytr)
    acc = accuracy_score(yte, clf.predict(Xte))
    assert acc > 0.93, f"digits accuracy too low: {acc}"


def test_softmax_beats_ovr_log_loss():
    # The point of the softmax path: better-calibrated joint probabilities.
    Xtr, Xte, ytr, yte = _split(n_classes=6, n=6000)
    losses = {}
    for mode in ["softmax", "ovr"]:
        clf = YABTClassifier(multiclass=mode, n_estimators=150, device=DEVICE, seed=0)
        clf.fit(Xtr, ytr)
        losses[mode] = log_loss(yte, clf.predict_proba(Xte))
    assert losses["softmax"] < losses["ovr"] * 1.02, losses


def test_softmax_early_stopping_joint():
    Xtr, Xte, ytr, yte = _split()
    Xtr, Xv, ytr, yv = train_test_split(Xtr, ytr, test_size=0.25, random_state=0, stratify=ytr)
    clf = YABTClassifier(n_estimators=2000, early_stopping_rounds=15, device=DEVICE)
    clf.fit(Xtr, ytr, eval_set=(Xv, yv))
    b = clf.booster_
    assert isinstance(b, SoftmaxBooster)
    assert b.best_iter is not None
    # one early-stopping clock: every class stopped at the same round count
    assert len({len(trees) for trees in b.trees_}) == 1
    assert len(b.trees_[0]) < 2000


def test_softmax_subsample_colsample():
    Xtr, Xte, ytr, yte = _split(n=2500)
    clf = YABTClassifier(n_estimators=60, subsample=0.7, colsample=0.7, device=DEVICE)
    clf.fit(Xtr, ytr)
    assert isinstance(clf.booster_, SoftmaxBooster)
    assert accuracy_score(yte, clf.predict(Xte)) > 0.6

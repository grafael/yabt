import numpy as np
import pytest
import torch
from sklearn.datasets import load_breast_cancer, make_regression
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import train_test_split

from yabt import YABTClassifier, YABTRegressor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def test_classifier_breast_cancer():
    X, y = load_breast_cancer(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
    clf = YABTClassifier(n_estimators=200, learning_rate=0.1, max_leaves=15, device=DEVICE)
    clf.fit(Xtr, ytr)
    auc = roc_auc_score(yte, clf.predict_proba(Xte)[:, 1])
    assert auc > 0.98, f"AUC too low: {auc}"


def test_regressor_synthetic():
    X, y = make_regression(n_samples=3000, n_features=10, noise=5.0, random_state=0)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
    reg = YABTRegressor(n_estimators=300, learning_rate=0.1, max_leaves=31, device=DEVICE)
    reg.fit(Xtr, ytr)
    r2 = r2_score(yte, reg.predict(Xte))
    assert r2 > 0.85, f"R2 too low: {r2}"


def test_early_stopping():
    X, y = load_breast_cancer(return_X_y=True)
    Xtr, Xv, ytr, yv = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
    clf = YABTClassifier(n_estimators=2000, early_stopping_rounds=20, device=DEVICE)
    clf.fit(Xtr, ytr, eval_set=(Xv, yv))
    assert clf.booster_.best_iter is not None
    assert len(clf.booster_.trees) < 2000


def test_categorical_encoding():
    rng = np.random.default_rng(0)
    n = 2000
    cat = rng.integers(0, 8, size=n)
    noise = rng.normal(size=n)
    y = (cat % 2 + 0.3 * noise > 0.5).astype(int)
    X = np.stack([cat.astype(float), rng.normal(size=n)], axis=1)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
    clf = YABTClassifier(n_estimators=100, device=DEVICE)
    clf.fit(Xtr, ytr, categorical_features=[0])
    auc = roc_auc_score(yte, clf.predict_proba(Xte)[:, 1])
    assert auc > 0.9


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")
def test_cpu_gpu_agreement():
    X, y = load_breast_cancer(return_X_y=True)
    preds, aucs = [], []
    for dev in ["cpu", "cuda"]:
        clf = YABTClassifier(n_estimators=30, device=dev, seed=0)
        clf.fit(X, y)
        p = clf.predict_proba(X)[:, 1]
        preds.append(p)
        aucs.append(roc_auc_score(y, p))
    # CPU and GPU must agree at the decision level: the histogram gain argmax is
    # not backend-stable on near-tied splits (float reduction order differs), and
    # that choice cascades over trees, so exact per-sample probabilities can drift
    # substantially. The models are still equivalent (AUC agrees to ~1e-3), so
    # we assert decision agreement, not bitwise probability agreement.
    assert abs(aucs[0] - aucs[1]) < 5e-3
    # Sanity: most rows still match closely; only a small fraction sit on the
    # divergent near-tie paths.
    assert np.mean(np.abs(preds[0] - preds[1]) < 2e-3) > 0.7

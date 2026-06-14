"""Test novel boosting features."""

import torch
from sklearn.datasets import make_regression, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, roc_auc_score

from yabt import YABTClassifier, YABTRegressor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def test_adaptive_features_regression():
    """Test adaptive feature selection in regression."""
    X, y = make_regression(n_samples=1000, n_features=20, noise=10.0, random_state=0)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0)

    # Baseline
    reg_baseline = YABTRegressor(
        n_estimators=50,
        learning_rate=0.1,
        max_leaves=15,
        device=DEVICE,
        adaptive_features=False,
    )
    reg_baseline.fit(Xtr, ytr)
    baseline_r2 = r2_score(yte, reg_baseline.predict(Xte))

    # With adaptive features
    reg_adaptive = YABTRegressor(
        n_estimators=50,
        learning_rate=0.1,
        max_leaves=15,
        device=DEVICE,
        adaptive_features=True,
        feature_importance_alpha=0.1,
    )
    reg_adaptive.fit(Xtr, ytr)
    adaptive_r2 = r2_score(yte, reg_adaptive.predict(Xte))

    # Both should work
    assert baseline_r2 > 0.5, f"Baseline R2 too low: {baseline_r2}"
    assert adaptive_r2 > 0.5, f"Adaptive R2 too low: {adaptive_r2}"
    assert reg_adaptive.booster_.adaptive_selector is not None


def test_goss_sampling():
    """Test gradient-based one-side sampling."""
    X, y = make_regression(n_samples=1000, n_features=10, noise=5.0, random_state=0)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0)

    # With GOSS
    reg = YABTRegressor(
        n_estimators=50,
        learning_rate=0.1,
        max_leaves=15,
        device=DEVICE,
        goss_enabled=True,
        goss_ratio=0.9,
    )
    reg.fit(Xtr, ytr)
    goss_r2 = r2_score(yte, reg.predict(Xte))

    assert goss_r2 > 0.5, f"GOSS R2 too low: {goss_r2}"
    assert reg.booster_.goss is not None


def test_differentiable_refinement():
    """Test differentiable tree refinement."""
    X, y = load_breast_cancer(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

    # Without refinement
    clf_no_refine = YABTClassifier(
        n_estimators=50,
        learning_rate=0.1,
        max_leaves=15,
        device=DEVICE,
        refine_steps=0,
    )
    clf_no_refine.fit(Xtr, ytr)
    no_refine_auc = roc_auc_score(yte, clf_no_refine.predict_proba(Xte)[:, 1])

    # With refinement
    clf_refine = YABTClassifier(
        n_estimators=50,
        learning_rate=0.1,
        max_leaves=15,
        device=DEVICE,
        refine_steps=5,
        refine_lr=0.02,
    )
    clf_refine.fit(Xtr, ytr)
    refine_auc = roc_auc_score(yte, clf_refine.predict_proba(Xte)[:, 1])

    assert no_refine_auc > 0.95, f"No-refine AUC too low: {no_refine_auc}"
    assert refine_auc > 0.95, f"Refine AUC too low: {refine_auc}"


def test_global_leaf_refit():
    """Test global leaf value refitting."""
    X, y = make_regression(n_samples=1000, n_features=10, noise=5.0, random_state=0)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0)

    # With refit
    reg = YABTRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_leaves=15,
        device=DEVICE,
        refit_every=25,
        refit_steps=10,
        refit_lr=0.05,
    )
    reg.fit(Xtr, ytr)
    refit_r2 = r2_score(yte, reg.predict(Xte))

    assert refit_r2 > 0.5, f"Refit R2 too low: {refit_r2}"


def test_all_novel_features_together():
    """Test combining all novel features."""
    X, y = make_regression(n_samples=1000, n_features=15, noise=5.0, random_state=0)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0)

    # All features enabled
    reg = YABTRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_leaves=15,
        device=DEVICE,
        # Refinement
        refine_steps=5,
        refine_lr=0.02,
        # Adaptive features
        adaptive_features=True,
        feature_importance_alpha=0.1,
        # GOSS
        goss_enabled=True,
        goss_ratio=0.9,
        # Refit
        refit_every=25,
        refit_steps=10,
        refit_lr=0.05,
        # Interactions
        detect_interactions=True,
    )
    reg.fit(Xtr, ytr)
    combined_r2 = r2_score(yte, reg.predict(Xte))

    assert combined_r2 > 0.5, f"Combined R2 too low: {combined_r2}"
    assert reg.booster_.adaptive_selector is not None
    assert reg.booster_.goss is not None


def test_novel_features_classification():
    """Test novel features in classification."""
    X, y = load_breast_cancer(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

    # All features enabled
    clf = YABTClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_leaves=15,
        device=DEVICE,
        refine_steps=3,
        adaptive_features=True,
        goss_enabled=True,
        refit_every=25,
    )
    clf.fit(Xtr, ytr)
    auc = roc_auc_score(yte, clf.predict_proba(Xte)[:, 1])

    assert auc > 0.95, f"AUC too low: {auc}"

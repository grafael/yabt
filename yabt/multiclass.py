"""Multiclass classification support via One-vs-Rest (OvR)."""

from __future__ import annotations

import numpy as np
from .boosting import Booster, BoostParams, LogLoss


class MulticlassBooster:
    """One-vs-Rest multiclass booster.

    Trains one binary classifier per class, then combines predictions
    for multiclass classification.
    """

    def __init__(self, params: BoostParams):
        self.params = params
        self.boosters_: list[Booster] = []
        self.classes_: np.ndarray | None = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        eval_set: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> MulticlassBooster:
        """Train OvR classifiers.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training features
        y : np.ndarray of shape (n_samples,)
            Class labels (multiclass)
        eval_set : tuple of (X_eval, y_eval), optional
            Evaluation set for early stopping

        Returns
        -------
        self
        """
        self.classes_ = np.unique(y)

        # Train one binary classifier per class
        for class_label in self.classes_:
            # Binary target: this class vs rest
            y_binary = (y == class_label).astype(np.float32)

            # Create binary eval set if provided
            eval_set_binary = None
            if eval_set is not None:
                y_eval = (eval_set[1] == class_label).astype(np.float32)
                eval_set_binary = (eval_set[0], y_eval)

            # Train binary classifier
            booster = Booster(self.params, LogLoss())
            booster.fit(X, y_binary, eval_set=eval_set_binary)
            self.boosters_.append(booster)

        return self

    def predict_margin(self, X: np.ndarray) -> np.ndarray:
        """Get decision margins for all classes.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Features

        Returns
        -------
        margins : np.ndarray of shape (n_samples, n_classes)
            Raw scores (margins) for each class
        """
        margins = []
        for booster in self.boosters_:
            m = booster.predict_margin(X)
            margins.append(m)
        return np.stack(margins, axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get class probabilities using softmax.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Features

        Returns
        -------
        probabilities : np.ndarray of shape (n_samples, n_classes)
            Probability for each class
        """
        margins = self.predict_margin(X)

        # Apply softmax for probability normalization
        margins_shifted = margins - margins.max(axis=1, keepdims=True)
        exp_margins = np.exp(margins_shifted)
        proba = exp_margins / exp_margins.sum(axis=1, keepdims=True)

        return proba

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Features

        Returns
        -------
        predictions : np.ndarray of shape (n_samples,)
            Predicted class labels
        """
        proba = self.predict_proba(X)
        class_indices = np.argmax(proba, axis=1)
        return self.classes_[class_indices]

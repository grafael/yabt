"""Multiclass classification support via One-vs-Rest (OvR)."""

from __future__ import annotations

import numpy as np
from .boosting import Booster, BoostParams, LogLoss


class MulticlassBooster:
    """One-vs-Rest multiclass booster: one binary booster per class, combined
    via softmax over the per-class margins."""

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
        self.classes_ = np.unique(y)
        for class_label in self.classes_:
            y_binary = (y == class_label).astype(np.float32)
            eval_set_binary = None
            if eval_set is not None:
                y_eval = (eval_set[1] == class_label).astype(np.float32)
                eval_set_binary = (eval_set[0], y_eval)
            booster = Booster(self.params, LogLoss())
            booster.fit(X, y_binary, eval_set=eval_set_binary)
            self.boosters_.append(booster)
        return self

    def predict_margin(self, X: np.ndarray) -> np.ndarray:
        """Per-class raw scores, (n_samples, n_classes)."""
        return np.stack([b.predict_margin(X) for b in self.boosters_], axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Softmax of the per-class margins, (n_samples, n_classes)."""
        margins = self.predict_margin(X)
        exp_margins = np.exp(margins - margins.max(axis=1, keepdims=True))
        return exp_margins / exp_margins.sum(axis=1, keepdims=True)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicted class label per row."""
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

"""Validation-based auto-tuning: learn per-dataset hyperparameters.

A small curated candidate set (informed by the library's A/B results) is
evaluated on a held-out split with early stopping; the winner is refit on the
full data with the tree count learned from early stopping. This is a bounded
search (a handful of early-stopped fits), not an open-ended sweep.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import torch

SEARCH_MAX_ESTIMATORS = 300
VAL_FRACTION = 0.2
MIN_ROWS_TO_TUNE = 600  # below this a validation split is too noisy to trust


def _candidates(n: int) -> list[tuple[str, dict]]:
    cands = [
        ("user-config", {}),
        ("slow-deep", {"learning_rate": 0.05, "max_leaves": 63}),
        ("fast-shallow", {"learning_rate": 0.2, "max_leaves": 15}),
        ("constant-leaves", {"neural_leaves": False}),       # sharp-boundary data
        ("strong-interactions", {"interaction_boost": 1.0}),  # interaction-heavy data
        ("fine-grain", {"min_samples_leaf": 5, "max_leaves": 63}),
    ]
    if n < 5000:
        cands = [c for c in cands if c[0] != "fine-grain"]  # overfits small data
    return cands


def tune_params(
    params,
    loss,
    X: np.ndarray,
    y: np.ndarray,
    eval_set: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[object, dict | None]:
    """Returns (tuned_params, report). Report is None when tuning was skipped.

    Candidates are ranked by validation loss at (close to) the deployment
    tree count, with no early stopping in the search: ranking at each
    candidate's early-stop point and deploying at full length measurably
    mis-selected in A/Bs (a fast config can win at its plateau while a slower
    one keeps improving). The final fit keeps the caller's n_estimators.
    With an external eval_set the candidates are fit on all of X and scored
    on it.
    """
    from .boosting import Booster  # local import: boosting imports this module

    n = X.shape[0]
    if n < MIN_ROWS_TO_TUNE or params.n_estimators < 20:
        return replace(params, auto_tune=False), None

    if eval_set is not None:
        Xtr, ytr = X, y
        Xv, yv = np.asarray(eval_set[0]), np.asarray(eval_set[1], dtype=np.float32)
    else:
        rng = np.random.default_rng(params.seed)
        m = min(max(150, int(n * VAL_FRACTION)), n // 2)
        perm = rng.permutation(n)
        Xtr, ytr = X[perm[m:]], y[perm[m:]]
        Xv, yv = X[perm[:m]], y[perm[:m]]

    yv_t = torch.as_tensor(yv, dtype=torch.float32)
    search_estimators = min(params.n_estimators, SEARCH_MAX_ESTIMATORS)
    results = []
    for name, over in _candidates(n):
        p = replace(params, auto_tune=False, n_estimators=search_estimators,
                    early_stopping_rounds=0, verbose=False, **over)
        b = Booster(p, loss).fit(Xtr, ytr)
        margin = torch.as_tensor(b.predict_margin(Xv), dtype=torch.float32)
        vl = float(loss.loss(margin, yv_t))
        results.append({"name": name, "val_loss": vl, "overrides": over})

    results.sort(key=lambda r: r["val_loss"])
    best = results[0]
    tuned = replace(params, auto_tune=False, **best["overrides"])
    return tuned, {"selected": best["name"], "val_loss": best["val_loss"],
                   "n_estimators": tuned.n_estimators, "results": results}

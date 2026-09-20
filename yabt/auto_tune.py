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
MIN_ROWS_TO_TUNE = 600  # at/above this a single holdout split is trustworthy
MIN_ROWS_CV = 150       # below MIN_ROWS_TO_TUNE but at/above this: use k-fold CV
CV_FOLDS = 3


def _candidates(n: int) -> list[tuple[str, dict]]:
    from .boosting import _SMALL_N_ROWS  # local import: boosting imports this module

    cands = [
        ("user-config", {}),
        ("slow-deep", {"learning_rate": 0.05, "max_leaves": 63}),
        ("fast-shallow", {"learning_rate": 0.2, "max_leaves": 15}),
        ("constant-leaves", {"neural_leaves": False}),       # sharp-boundary data
        ("strong-interactions", {"interaction_boost": 1.0}),  # interaction-heavy data
        # low-signal/noisy targets: a min-gain floor refuses to split on noise.
        # A *fixed* gamma is net-negative suite-wide -- its right value is
        # target-scale dependent, so it wrecks high-signal smooth targets. The
        # scale-invariant min_split_gain_rel floor (gamma = rho * var(grad) per
        # tree) avoids that; validation-gating then deploys it only where it wins
        # (full-suite A/B: large gains on quake, forest_fires, solar_flare,
        # abalone) and rejects it on smooth targets it would hurt (airfoil,
        # kin8nm, white_wine). The strong variant pairs it with a shallower tree
        # for the noisiest targets.
        ("regularized-splits", {"min_split_gain_rel": 0.5}),
        ("regularized-splits-strong", {"min_split_gain_rel": 2.0, "max_leaves": 15}),
        ("fine-grain", {"min_samples_leaf": 5, "max_leaves": 63}),
        # Half the default leaf budget. The capacity lever that dominates below
        # _SMALL_N_ROWS decays with n but does not vanish: 15 leaves against the
        # default 31 is 8/11 wins (median -0.35%) over the TabArena tasks with
        # 2500 <= n < 9000 and 6/11 (median -0.17%) over 9000 <= n < 30000 --
        # 14 of 22, a real but weak effect. It also gets *more expensive* as n
        # grows, because a smaller budget means more trees before early stopping
        # fires: 1.2x train time on the mid band and 1.84x on the large one. A
        # weak win that costs 84% more compute is exactly what a per-dataset
        # validation gate is for; shipping it blind is the mistake the old
        # 16-leaf small-data cap made.
        ("half-budget", {"max_leaves": 15}),
        # Small data: the caps are now the default (see _SMALL_N_ROWS), so the
        # candidate is the *escape hatch*. They win 14 of the 17 sub-2500-row
        # TabArena tasks but lose on smooth regression targets that genuinely
        # want the capacity (airfoil_self_noise +8.7%), which is exactly what a
        # validation gate is for.
        ("uncapped-small-data", {"small_data_caps": False}),
    ]
    if n < 5000:
        cands = [c for c in cands if c[0] != "fine-grain"]  # overfits small data
    if n >= _SMALL_N_ROWS:
        # The caps only bind below _SMALL_N_ROWS; above it this candidate is a
        # bit-identical duplicate of user-config, so it would just cost a fit.
        cands = [c for c in cands if c[0] != "uncapped-small-data"]
    else:
        # Below _SMALL_N_ROWS the small-data cap binds max_leaves to 4, so a
        # 15-leaf candidate is a duplicate of user-config -- one wasted fit.
        cands = [c for c in cands if c[0] != "half-budget"]
    return cands


def _val_loss(params, loss, over, search_estimators, Xtr, ytr, Xv, yv_t) -> float:
    """Fit one candidate to a fixed tree count and score it on a holdout."""
    from .boosting import Booster  # local import: boosting imports this module

    p = replace(params, auto_tune=False, n_estimators=search_estimators,
                early_stopping_rounds=0, verbose=False, **over)
    b = Booster(p, loss).fit(Xtr, ytr)
    margin = torch.as_tensor(b.predict_margin(Xv), dtype=torch.float32)
    return float(loss.loss(margin, yv_t))


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

    Scoring split: an external eval_set is used directly; data at/above
    MIN_ROWS_TO_TUNE uses a single holdout; smaller data (down to MIN_ROWS_CV,
    too noisy for one split) uses CV_FOLDS-fold cross-validation.
    """
    n = X.shape[0]
    if params.n_estimators < 20 or (eval_set is None and n < MIN_ROWS_CV):
        return replace(params, auto_tune=False), None

    # Build the (train, val) folds candidates are scored on. Large data uses one
    # holdout; small data (too noisy for a single split) uses k-fold CV; an
    # external eval_set is always used directly. Each fold is
    # (Xtr, ytr, Xval, yval_tensor).
    if eval_set is not None:
        yv = np.asarray(eval_set[1], dtype=np.float32)
        folds = [(X, y, np.asarray(eval_set[0]), torch.as_tensor(yv, dtype=torch.float32))]
        scoring = "eval_set"
    elif n >= MIN_ROWS_TO_TUNE:
        rng = np.random.default_rng(params.seed)
        m = min(max(150, int(n * VAL_FRACTION)), n // 2)
        perm = rng.permutation(n)
        folds = [(X[perm[m:]], y[perm[m:]], X[perm[:m]],
                  torch.as_tensor(y[perm[:m]], dtype=torch.float32))]
        scoring = "holdout"
    else:
        rng = np.random.default_rng(params.seed)
        chunks = np.array_split(rng.permutation(n), CV_FOLDS)
        folds = []
        for i, te in enumerate(chunks):
            tr = np.concatenate([chunks[j] for j in range(CV_FOLDS) if j != i])
            folds.append((X[tr], y[tr], X[te],
                          torch.as_tensor(y[te], dtype=torch.float32)))
        scoring = "cv"

    search_estimators = min(params.n_estimators, SEARCH_MAX_ESTIMATORS)
    results = []
    for name, over in _candidates(n):
        vl = float(np.mean([
            _val_loss(params, loss, over, search_estimators, ftr_X, ftr_y, fXv, fyv)
            for ftr_X, ftr_y, fXv, fyv in folds]))
        results.append({"name": name, "val_loss": vl, "overrides": over})

    results.sort(key=lambda r: r["val_loss"])
    best = results[0]
    tuned = replace(params, auto_tune=False, **best["overrides"])
    return tuned, {"selected": best["name"], "val_loss": best["val_loss"],
                   "n_estimators": tuned.n_estimators, "results": results,
                   "scoring": scoring}

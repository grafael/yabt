"""Gradient-guided multiplicative feature construction.

Greedy histogram trees with additive leaves cannot represent products of
features whose individual *and* low-order correlations with the target vanish --
e.g. ``y = x_i * x_j * x_k`` of zero-mean features. There no single feature, and
no pair, shows marginal split gain, so the splitter only stumbles into the
interaction by sampling noise and plateaus well below the achievable fit (A/B:
3-way multiplicative regression tops out near R^2 0.81 regardless of depth,
trees, or MLP leaves, against a ~0.99 ceiling).

Such multiplicative groups are still detectable without trying every subset: the
*magnitude* of a product tracks the product of magnitudes, so a feature that
drives the residual multiplicatively has its squared value correlated with the
squared residual, even when its signed value is uncorrelated with the residual
(the signal vanishing low-order interaction detectors rely on). We rank features
by that magnitude signal, form candidate products among the top few, and keep
only products that correlate with the residual better than their component
features do. Kept products are appended as ordinary columns, so binning, split
search, leaves, and inference are all unchanged -- the feature is a thin,
opt-in front end that hands the tree the interaction it cannot otherwise find.

On data without multiplicative structure the correlation guard keeps nothing, so
training is left untouched (exact neutrality, no extra columns, no extra time).
"""

from __future__ import annotations

import itertools

import numpy as np


def _abs_corr(a: np.ndarray, r: np.ndarray) -> float:
    """|Pearson correlation| between column ``a`` and centered residual ``r``."""
    a = a - a.mean()
    denom = a.std() * r.std()
    if denom < 1e-12:
        return 0.0
    return float(abs((a * r).mean()) / denom)


def detect_product_specs(
    X: np.ndarray,
    y: np.ndarray,
    *,
    max_features: int = 5,
    max_order: int = 3,
    min_corr: float = 0.03,
    corr_gain: float = 1.3,
    max_products: int = 16,
) -> list[tuple[int, ...]]:
    """Feature-index tuples whose products should be appended to ``X``.

    ``X`` is the (n, F) float feature matrix and ``y`` the target; the residual
    proxy is ``r = y - mean(y)`` (the gradient direction at the base score, up to
    sign, for both MSE and log loss). Features are ranked by ``corr(x^2, r^2)``
    -- the magnitude signal that survives even when ``corr(x, r) == 0`` -- and
    products among the top ``max_features`` are kept only when they correlate
    with ``r`` more strongly (by ``corr_gain``) than any of their components and
    clear an absolute floor ``min_corr``. Returns at most ``max_products`` specs,
    strongest first; an empty list means "add nothing".
    """
    X = np.asarray(X, dtype=np.float64)
    n, F = X.shape
    if n < 200 or F < 2 or max_order < 2:
        return []
    r = np.asarray(y, dtype=np.float64).ravel()
    r = r - r.mean()
    if r.std() < 1e-12:
        return []

    sd = X.std(0)
    live = sd > 1e-12
    if int(live.sum()) < 2:
        return []
    Xs = np.zeros_like(X)
    Xs[:, live] = (X[:, live] - X[:, live].mean(0)) / sd[live]

    # Magnitude-interaction score: corr(x^2, r^2). High for features whose
    # spread drives the residual spread (multiplicative drivers), ~0 for
    # additive or irrelevant features.
    x2 = Xs ** 2
    x2 -= x2.mean(0)
    r2 = r ** 2
    r2 -= r2.mean()
    denom = x2.std(0) * (r2.std() + 1e-12)
    score = np.where(denom > 1e-12, np.abs((x2 * r2[:, None]).mean(0)) / denom, 0.0)
    score[~live] = -1.0

    m = min(max_features, int(live.sum()))
    idx = sorted(int(j) for j in np.argsort(-score)[:m])
    base_corr = {j: _abs_corr(X[:, j], r) for j in idx}

    scored: list[tuple[float, tuple[int, ...]]] = []
    for order in range(2, max_order + 1):
        for combo in itertools.combinations(idx, order):
            p = np.ones(n)
            for k in combo:
                p = p * X[:, k]
            pc = _abs_corr(p, r)
            comp = max(base_corr[k] for k in combo)
            if pc > min_corr and pc > corr_gain * comp:
                scored.append((pc, combo))
    scored.sort(key=lambda t: t[0], reverse=True)
    return [combo for _, combo in scored[:max_products]]


def expand_products(X: np.ndarray, specs: list[tuple[int, ...]]) -> np.ndarray:
    """Return ``X`` (float32) with one appended column per spec (the product of
    the spec's feature columns). A no-op when ``specs`` is empty."""
    X = np.asarray(X, dtype=np.float32)
    if not specs:
        return X
    cols = [X]
    for combo in specs:
        p = np.ones(X.shape[0], dtype=np.float32)
        for k in combo:
            p = p * X[:, k]
        cols.append(p[:, None])
    return np.concatenate(cols, axis=1).astype(np.float32)

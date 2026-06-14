"""Quantile binning and leakage-free categorical target encoding."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

MAX_BINS = 256  # bins are stored as uint8


class Binner:
    """Quantile-bins float features into uint8 codes.

    Bin semantics: for feature f with interior edges e_0 < ... < e_{k-1},
    bin(x) = #{j : e_j < x}, so ``bin(x) <= b  <=>  x <= e_b``. This makes a
    split "bin <= b" on binned data exactly equivalent to "x <= edges[b]" on
    raw data, which the refinement stage relies on.
    """

    def __init__(self, max_bins: int = MAX_BINS):
        if not 2 <= max_bins <= 256:
            raise ValueError("max_bins must be in [2, 256]")
        self.max_bins = max_bins
        self.edges_: list[torch.Tensor] | None = None  # per-feature interior edges
        self.medians_: np.ndarray | None = None
        self.scales_: torch.Tensor | None = None  # per-feature robust scale (for soft routing)

    def fit(self, X: np.ndarray) -> "Binner":
        X = np.asarray(X, dtype=np.float32)
        n, F = X.shape
        self.medians_ = np.nanmedian(X, axis=0)
        self.medians_ = np.where(np.isnan(self.medians_), 0.0, self.medians_)
        edges = []
        scales = np.empty(F, dtype=np.float32)
        # Subsample rows for quantile estimation on very large data.
        if n > 200_000:
            rng = np.random.default_rng(0)
            Xq = X[rng.choice(n, 200_000, replace=False)]
        else:
            Xq = X
        for f in range(F):
            col = Xq[:, f]
            col = col[~np.isnan(col)]
            if col.size == 0:
                col = np.zeros(1, dtype=np.float32)
            qs = np.quantile(col, np.linspace(0, 1, self.max_bins + 1)[1:-1])
            e = np.unique(qs.astype(np.float32))
            edges.append(torch.from_numpy(e))
            q25, q75 = np.quantile(col, [0.25, 0.75])
            s = float(q75 - q25)
            if s == 0.0:
                s = float(col.std()) or 1.0
            scales[f] = s
        self.edges_ = edges
        self.scales_ = torch.from_numpy(scales)
        return self

    def transform(self, X: np.ndarray, device: str = "cpu") -> torch.Tensor:
        assert self.edges_ is not None, "Binner not fitted"
        X = np.asarray(X, dtype=np.float32)
        X = np.where(np.isnan(X), self.medians_, X)
        Xt = torch.from_numpy(np.ascontiguousarray(X))
        n, F = Xt.shape
        out = torch.empty((n, F), dtype=torch.uint8)
        for f in range(F):
            out[:, f] = torch.searchsorted(self.edges_[f], Xt[:, f].contiguous()).to(torch.uint8)
        return out.to(device)

    def impute(self, X: np.ndarray) -> np.ndarray:
        """Median-impute NaNs; raw float matrix used for inference/refinement."""
        X = np.asarray(X, dtype=np.float32)
        return np.where(np.isnan(X), self.medians_, X)

    def edge_value(self, feature: int, bin_idx: int) -> float:
        """Raw-space threshold equivalent to the split ``bin <= bin_idx``."""
        e = self.edges_[feature]
        return float(e[min(bin_idx, len(e) - 1)])


class PermutationTargetEncoder:
    """CatBoost-style smoothed target encoding with permutation-ordered statistics.

    Training rows are encoded using only the targets of rows that precede them
    in a random permutation (averaged over ``n_permutations``), which prevents
    target leakage. Test rows use the full-train smoothed category means.
    """

    def __init__(self, smoothing: float = 10.0, n_permutations: int = 3, seed: int = 0):
        self.smoothing = smoothing
        self.n_permutations = n_permutations
        self.seed = seed
        self.full_means_: list[dict] = []
        self.prior_: float = 0.0

    def fit_transform(self, X_cat: np.ndarray, y: np.ndarray) -> np.ndarray:
        n, C = X_cat.shape
        y = np.asarray(y, dtype=np.float64)
        self.prior_ = float(y.mean())
        rng = np.random.default_rng(self.seed)
        out = np.zeros((n, C), dtype=np.float32)
        self.full_means_ = []
        for c in range(C):
            codes, inv = np.unique(X_cat[:, c], return_inverse=True)
            k = len(codes)
            acc = np.zeros(n, dtype=np.float64)
            for _ in range(self.n_permutations):
                perm = rng.permutation(n)
                g_p = pd.Series(inv[perm])
                y_p = pd.Series(y[perm])
                grp = y_p.groupby(g_p)
                csum = grp.cumsum() - y_p  # target sum of preceding same-category rows
                ccnt = g_p.groupby(g_p).cumcount()
                enc_p = (csum + self.smoothing * self.prior_) / (ccnt + self.smoothing)
                acc[perm] += enc_p.to_numpy()
            out[:, c] = (acc / self.n_permutations).astype(np.float32)
            sums = np.bincount(inv, weights=y, minlength=k)
            cnts = np.bincount(inv, minlength=k).astype(np.float64)
            means = (sums + self.smoothing * self.prior_) / (cnts + self.smoothing)
            self.full_means_.append(dict(zip(codes.tolist(), means.tolist())))
        return out

    def transform(self, X_cat: np.ndarray) -> np.ndarray:
        n, C = X_cat.shape
        out = np.full((n, C), self.prior_, dtype=np.float32)
        for c in range(C):
            m = self.full_means_[c]
            out[:, c] = np.array([m.get(v, self.prior_) for v in X_cat[:, c]], dtype=np.float32)
        return out

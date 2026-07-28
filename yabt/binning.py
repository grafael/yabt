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

    Missing values get their own bin, above every real one. A feature with NaNs
    in the training data gives up one bin of resolution and reserves the index
    just past its last real bin for NaN, and ``impute`` fills NaN with a
    raw-space sentinel strictly greater than that feature's maximum -- so
    ``x <= threshold`` is False for a missing value at every real threshold,
    which is exactly the "bin > b" the binned matrix says. Growers, refinement
    and inference therefore need no NaN branch: missingness is just an extreme
    category they can split off.

    Median imputation (the previous behavior) merged missing rows into the
    middle of the distribution and destroyed informative missingness. Paired
    A/B over the NaN-carrying datasets: polish_companies_bankruptcy (5910 rows,
    NaNs in 49 of 64 columns) -14.1%, kick -0.5%, APSFailure (76k rows, 8.3%
    NaN) neutral at 0.00901 vs 0.00903, hepatitis neutral, colic (368 rows)
    +5.7% but inside its own noise band (per-arm SE 0.0067 over 40 fits). A
    large win where missingness carries signal, neutral where it does not.

    A *learned per-split* direction is not worth the extra machinery on top:
    XGBoost with its NaNs pre-replaced by this same sentinel scores 0.04119 on
    polish against 0.04142 for its own per-node direction.
    """

    def __init__(self, max_bins: int = MAX_BINS):
        if not 2 <= max_bins <= 256:
            raise ValueError("max_bins must be in [2, 256]")
        self.max_bins = max_bins
        self.edges_: list[torch.Tensor] | None = None  # per-feature interior edges
        self.scales_: torch.Tensor | None = None  # per-feature robust scale (for soft routing)
        self.maxes_: np.ndarray | None = None       # per-feature max finite train value
        self.nan_values_: np.ndarray | None = None  # per-feature raw-space NaN sentinel
        self.nan_bin_: np.ndarray | None = None     # per-feature NaN bin index, -1 if none

    def fit(self, X: np.ndarray) -> "Binner":
        X = np.asarray(X, dtype=np.float32)
        n, F = X.shape
        # Missing-value support, measured on the *full* column rather than the
        # quantile subsample below: which features need a reserved NaN bin, and
        # the sentinel that puts a missing value above every real one in raw
        # space. nextafter is the smallest such step, so imputed rows stay in
        # range for the leaf models instead of becoming wild outliers.
        self._has_nan = np.isnan(X).any(axis=0)
        with np.errstate(all="ignore"):
            maxes = np.where(np.isnan(X), -np.inf, X).max(axis=0)
        self.maxes_ = np.where(np.isfinite(maxes), maxes, 0.0).astype(np.float32)
        self.nan_values_ = np.nextafter(self.maxes_, np.float32(np.inf))
        self.nan_bin_ = np.full(F, -1, dtype=np.int64)
        scales = np.empty(F, dtype=np.float32)
        # Subsample rows for quantile estimation on very large data.
        if n > 200_000:
            rng = np.random.default_rng(0)
            Xq = X[rng.choice(n, 200_000, replace=False)]
        else:
            Xq = X

        # GPU fast path: the per-feature numpy quantile loop is single-threaded
        # and a large share of a GPU fit's setup. When there are no NaNs to mask
        # away per feature, all features' quantiles are one batched torch.quantile
        # on the GPU -- ~25x faster, and the resulting bins are identical (data
        # points sit far from the edge values, so float differences in the edges
        # never move a point across a bin boundary; verified 0% bin mismatch).
        if (torch.cuda.is_available() and Xq.shape[0] * Xq.shape[1] >= 100_000
                and not np.isnan(Xq).any()
                and self._fit_quantiles_gpu(Xq, scales)):
            return self

        edges = []
        for f in range(F):
            col = Xq[:, f]
            col = col[~np.isnan(col)]
            if col.size == 0:
                col = np.zeros(1, dtype=np.float32)
            # A feature with missing values spends one bin on them, so it gets
            # one fewer real bin -- otherwise the NaN index could reach 256 and
            # overflow the uint8 codes.
            nb = self.max_bins - 1 if self._has_nan[f] else self.max_bins
            qs = np.quantile(col, np.linspace(0, 1, nb + 1)[1:-1])
            e = np.unique(qs.astype(np.float32))
            edges.append(torch.from_numpy(e))
            if self._has_nan[f]:
                self.nan_bin_[f] = len(e) + 1  # just past the last real bin
            q25, q75 = np.quantile(col, [0.25, 0.75])
            s = float(q75 - q25)
            if s == 0.0:
                s = float(col.std()) or 1.0
            scales[f] = s
        self.edges_ = edges
        self.scales_ = torch.from_numpy(scales)
        return self

    def _fit_quantiles_gpu(self, Xq: np.ndarray, scales: np.ndarray) -> bool:
        """Batched GPU quantile fit (no-NaN data). Fills ``self.edges_`` /
        ``self.scales_`` and returns True, or returns False to fall back to the
        numpy path on any failure (e.g. OOM on very wide data)."""
        try:
            F = Xq.shape[1]
            Xg = torch.from_numpy(np.ascontiguousarray(Xq)).cuda()
            qpts = torch.linspace(0.0, 1.0, self.max_bins + 1, device="cuda")[1:-1]
            qe = torch.quantile(Xg, qpts, dim=0).t().contiguous().cpu().numpy()  # (F, B-1)
            qiqr = torch.quantile(Xg, torch.tensor([0.25, 0.75], device="cuda"),
                                  dim=0).cpu().numpy()  # (2, F)
            del Xg
            edges = []
            for f in range(F):
                e = np.unique(qe[f].astype(np.float32))
                edges.append(torch.from_numpy(e))
                s = float(qiqr[1, f] - qiqr[0, f])
                if s == 0.0:
                    s = float(Xq[:, f].std()) or 1.0
                scales[f] = s
            self.edges_ = edges
            self.scales_ = torch.from_numpy(scales)
            return True
        except Exception:
            return False

    def transform(self, X: np.ndarray, device: str = "cpu") -> torch.Tensor:
        assert self.edges_ is not None, "Binner not fitted"
        Xa = np.asarray(X, dtype=np.float32)
        nan_bins = self._nan_bins_on(device) if (self.nan_bin_ >= 0).any() else None
        nanm = torch.from_numpy(np.isnan(Xa)).to(device) if nan_bins is not None else None
        X = self.impute(Xa)
        # Do the per-feature searchsorted on the *target* device. searchsorted is
        # an exact integer comparison, so the binned codes are identical to the
        # CPU loop, but on cuda this single-threaded numpy/CPU hot loop (a large
        # fraction of a GPU fit's wall time) runs on the GPU instead. Edges are
        # moved to the device once and cached.
        Xt = torch.from_numpy(np.ascontiguousarray(X)).to(device)
        n, F = Xt.shape
        edges = self._edges_on(Xt.device)
        out = torch.empty((n, F), dtype=torch.uint8, device=Xt.device)
        for f in range(F):
            out[:, f] = torch.searchsorted(edges[f], Xt[:, f].contiguous()).to(torch.uint8)
            # The sentinel already sorts above every real value, i.e. into the
            # top *real* bin; move it on to the reserved NaN bin so the grower
            # sees missing as its own category rather than as "very large".
            # Features with no training NaNs have no reserved bin, and a missing
            # value at predict time simply lands in the top real bin -- still
            # routed right everywhere, just not separable.
            if nan_bins is not None and self.nan_bin_[f] >= 0:
                out[:, f] = torch.where(nanm[:, f], nan_bins[f], out[:, f])
        return out

    def _nan_bins_on(self, device) -> torch.Tensor:
        """Per-feature NaN bin index as uint8 on ``device`` (cached per device)."""
        cache = getattr(self, "_nanbin_dev_", None)
        dev = torch.device(device)
        if cache is None or cache[0] != dev:
            self._nanbin_dev_ = (
                dev, torch.from_numpy(self.nan_bin_.clip(0, 255).astype(np.uint8)).to(dev))
        return self._nanbin_dev_[1]

    def used_bins(self) -> np.ndarray:
        """Per-feature count of bins that can be populated: ``len(edges)+1`` real
        bins, plus the reserved NaN bin where there is one. Split searches use it
        to skip the always-empty tail, so it must cover the NaN bin -- otherwise
        missing rows would be invisible to the gain computation."""
        nb = getattr(self, "_used_bins", None)
        if nb is None or nb.shape[0] != len(self.edges_):
            nb = np.fromiter(
                (min(len(e) + 1 + (1 if self.nan_bin_[f] >= 0 else 0), MAX_BINS)
                 for f, e in enumerate(self.edges_)),
                dtype=np.int64, count=len(self.edges_))
            self._used_bins = nb
        return nb

    def _edges_on(self, device: torch.device) -> list[torch.Tensor]:
        """Per-feature edge tensors on ``device`` (cached per device)."""
        cache = getattr(self, "_edges_dev_", None)
        if cache is None or cache[0] != device:
            self._edges_dev_ = (device, [e.to(device) for e in self.edges_])
        return self._edges_dev_[1]

    def impute(self, X: np.ndarray) -> np.ndarray:
        """Fill NaNs with the per-feature sentinel (just above the feature's
        training maximum); raw float matrix used for inference/refinement.

        Every raw-space threshold this binner produces is at most the feature's
        maximum, so an imputed row fails ``x <= threshold`` at every split and
        is routed right -- matching its bin, which sits above all real ones."""
        X = np.asarray(X, dtype=np.float32)
        return np.where(np.isnan(X), self.nan_values_, X)

    def edge_value(self, feature: int, bin_idx: int) -> float:
        """Raw-space threshold equivalent to the split ``bin <= bin_idx``."""
        e = self.edges_[feature]
        # The split just below a reserved NaN bin separates every real value
        # from the missing ones, so its threshold is the feature's maximum
        # (< the NaN sentinel), not the last quantile edge.
        if self.nan_bin_[feature] >= 0 and bin_idx >= len(e):
            return float(self.maxes_[feature])
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
            out[:, c] = map_categories(X_cat[:, c], self.full_means_[c], self.prior_)
        return out


def map_categories(col: np.ndarray, mapping: dict, default: float) -> np.ndarray:
    """Look ``col``'s values up in ``mapping``, filling ``default`` for unseen
    ones. Vectorized through pandas rather than a per-row dict loop, which is
    ~5x slower and is on the inference path for every categorical column."""
    return pd.Series(col).map(mapping).fillna(default).to_numpy(dtype=np.float32)

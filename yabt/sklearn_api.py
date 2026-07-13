"""Scikit-learn style estimators."""

from __future__ import annotations

import numpy as np
from scipy.special import expit
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

from .binning import PermutationTargetEncoder
from .boosting import Booster, BoostParams, LogLoss, MSELoss
from .multiclass import MulticlassBooster, SoftmaxBooster
from .multitask import MultiTaskBooster

_PARAM_NAMES = [f.name for f in BoostParams.__dataclass_fields__.values()]

# Parameter reference, rendered into each estimator's docstring so help()/IDEs
# surface the hyperparameters (estimators take them through **kwargs, forwarded
# to boosting.BoostParams). Single source of truth, grouped; keep in sync with
# BoostParams. The multi-task estimator only honors the core split/sampling
# params, so it gets a trimmed rendering (see _MULTITASK_PARAMS).
_PARAM_GROUPS: list[list[tuple[str, str, str]]] = [
    [
        ("n_estimators", "int, default=500",
         "Number of boosting iterations (trees)."),
        ("learning_rate", "float, default=0.1",
         "Shrinkage applied to each tree's contribution."),
        ("max_leaves", "int, default=31",
         "Maximum number of leaves per tree."),
        ("max_depth", "int, default=64",
         "Maximum tree depth."),
        ("reg_lambda", "float, default=1.0",
         "L2 regularization on leaf weights."),
        ("gamma", "float, default=0.0",
         "Minimum loss reduction required to make a split."),
        ("min_split_gain_rel", "float, default=0.0",
         "Scale-invariant min-split-gain floor: per tree, an effective gamma of "
         "min_split_gain_rel * var(gradients) is added, refusing noise splits "
         "without a target-scale-dependent absolute threshold."),
        ("min_child_weight", "float, default=1e-3",
         "Minimum sum of Hessian (instance weight) allowed in a child."),
        ("min_samples_leaf", "int, default=20",
         "Minimum number of samples per leaf."),
        ("subsample", "float, default=1.0",
         "Row subsampling ratio drawn per tree."),
        ("colsample", "float, default=1.0",
         "Column (feature) subsampling ratio per tree."),
        ("max_bins", "int, default=256",
         "Number of histogram bins used to discretize features."),
    ],
    [
        ("refine_steps", "int, default=0",
         "Gradient-descent refinement steps applied to splits and leaves after\n"
         "each tree (0 disables; effective steps adapt to dataset size). Off by\n"
         "default: costs ~10% of fit time for negligible gain on real tabular\n"
         "data. Opt in with ``refine_steps > 0``."),
        ("refine_lr", "float, default=0.02",
         "Learning rate for differentiable refinement."),
        ("refine_min_gain", "float, default=1e-4",
         "Skip refinement when the loss is already below this threshold."),
        ("refit_every", "int, default=0",
         "Refit all leaf values across the ensemble every N trees (0 disables)."),
        ("refit_steps", "int, default=30",
         "Gradient steps per ensemble refit."),
        ("refit_lr", "float, default=0.05",
         "Learning rate for ensemble refit."),
    ],
    [
        ("adaptive_features", "bool, default=False",
         "Learn feature importances during training and bias sampling toward\n"
         "them."),
        ("feature_importance_alpha", "float, default=0.1",
         "EMA smoothing factor for the learned feature importances."),
        ("goss_enabled", "bool, default=False",
         "Gradient-based One-Side Sampling: keep large-gradient rows and\n"
         "subsample the rest."),
        ("goss_ratio", "float, default=0.9",
         "Fraction of large-gradient rows retained when GOSS is enabled."),
    ],
    [
        ("detect_interactions", "bool, default=False",
         "Track which feature pairs interact during training."),
        ("interaction_aware", "bool, default=True",
         "Steer split selection toward features that interact with those already\n"
         "on the node's path. Only flips near-ties and never inflates the gain\n"
         "used to accept a split. On by default (A/B-verified on tabular data)."),
        ("interaction_boost", "float, default=0.5",
         "Maximum multiplicative boost (capped at ``1 + interaction_boost``)\n"
         "applied to near-tie gains by interaction steering."),
    ],
    [
        ("product_features", "bool, default=False",
         "Detect feature groups that drive the residual multiplicatively (via the\n"
         "magnitude signal corr(x^2, r^2)) and append their products as columns\n"
         "before training, so the greedy splitter can use interactions like\n"
         "x_i*x_j*x_k that have no marginal gain. A correlation guard keeps a\n"
         "product only when it beats its components, so data without\n"
         "multiplicative structure is left untouched. Off by default (A/B: large\n"
         "win on multiplicative targets, neutral elsewhere)."),
        ("product_max_features", "int, default=5",
         "Number of top magnitude-signal features scanned for products."),
        ("product_max_order", "int, default=3",
         "Highest product order considered (3 = up to triple products)."),
        ("product_min_corr", "float, default=0.03",
         "Absolute residual-correlation floor for a product to be kept."),
        ("product_corr_gain", "float, default=1.3",
         "A product is kept only if its residual correlation exceeds this factor\n"
         "times the best correlation of its component features."),
    ],
    [
        ("kernel_splits", "bool, default=False",
         "Enable RBF landmark (\"blob\") splits for non-linear boundaries."),
        ("kernel_candidates", "int, default=8",
         "Number of candidate landmarks evaluated per node."),
        ("kernel_gamma", "float, default=0.0",
         "RBF bandwidth; 0 uses a per-landmark median-distance heuristic."),
        ("kernel_min_samples", "int, default=64",
         "Minimum node size for a kernel split to be considered."),
        ("kernel_importance_weighting", "bool or str, default=False",
         "EXPERIMENTAL. Weight kernel distances by per-feature split gain. False\n"
         "= uniform (best overall in A/B tests); \"node\" / True = gains of the\n"
         "node being split; \"ema\" = EMA of root-level gains from previous\n"
         "iterations."),
    ],
    [
        ("neural_leaves", "bool, default=True",
         "Fit a small per-leaf model instead of a constant value. On by default\n"
         "(linear leaves A/B-verified to win or tie at ~equal cost)."),
        ("leaf_net_hidden", "int, default=0",
         "Hidden width; 0 = ridge-linear leaves, >0 = tanh MLP of this width."),
        ("leaf_net_features", "int, default=8",
         "Number of top tree-split features used as leaf-model inputs."),
        ("leaf_net_l2", "float, default=1.0",
         "L2 regularization for the leaf models."),
        ("leaf_net_steps", "int, default=40",
         "Adam steps per tree (MLP leaves only)."),
        ("leaf_net_lr", "float, default=0.05",
         "Adam learning rate (MLP leaves only)."),
        ("leaf_net_min_samples", "int, default=50",
         "Leaves smaller than this keep their constant value."),
    ],
    [
        ("auto_tune", "bool, default=False",
         "Search curated hyperparameter candidates on a validation split before\n"
         "the final fit (skipped for datasets with < 600 rows)."),
    ],
    [
        ("stochastic_routing", "bool, default=False",
         "Use soft (expected-path) routing at inference; trees are still grown\n"
         "and trained hard, but predictions become smooth in X."),
        ("routing_tau", "float, default=0.05",
         "Gate width as a fraction of the split feature's scale."),
    ],
    [
        ("levelwise", "bool or str, default=\"auto\"",
         "Breadth-first (level-wise) growth with sibling subtraction. \"auto\"\n"
         "enables it on CUDA when ``max_leaves >= 16`` (~1.8x faster), otherwise\n"
         "uses the best-first heap grower; it also falls back to the heap below\n"
         "16 leaves or when ``kernel_splits`` is on. True/False force it on/off\n"
         "(True still skips kernel splits)."),
        ("numba_grower", "bool or str, default=\"auto\"",
         "Use the Numba-JIT compiled best-first grower on CPU (1.5-4x faster than\n"
         "the torch grower at identical accuracy). \"auto\" enables it on CPU for\n"
         "the axis-split path; it falls back to the torch grower on CUDA, on the\n"
         "level-wise path, or when ``kernel_splits`` is on. True/False force it\n"
         "on/off (True still falls back where unsupported)."),
        ("c_grower", "bool or str, default=\"auto\"",
         "Use the OpenMP-parallel C grower: the same leaf-wise kernel as the\n"
         "Numba grower but with the histogram build and split search threaded\n"
         "across cores (the Numba grower is single-threaded), bit-identical\n"
         "trees. \"auto\" uses it wherever the Numba grower runs when a C compiler\n"
         "is available and the problem is large enough to amortize threads,\n"
         "else falls back to Numba. True forces it on (still falls back if no\n"
         "compiler); False disables it."),
        ("c_grower_threads", "int, default=0",
         "OpenMP thread cap for the C grower; 0 picks a default (min(cores, 8),\n"
         "or ``OMP_NUM_THREADS`` if set above 1)."),
        ("sparse_hist", "bool or str, default=\"auto\"",
         "Sparse histogram build for the Numba grower: store each feature's\n"
         "non-modal bins and fill the modal bin by subtraction, making a\n"
         "histogram cost O(node_nnz + F) instead of O(node_rows * F). The win is\n"
         "on wide, sparse data (e.g. ~1.3x on Santander, 4991 features 97% zero),\n"
         "accuracy-neutral. \"auto\" uses it only when the data is dense enough\n"
         "below ``sparse_hist_max_density`` and rows are not subsampled; True/False\n"
         "force it (still requires the Numba grower)."),
        ("sparse_hist_max_density", "float, default=0.5",
         "Max fraction of explicitly-stored cells for \"auto\" ``sparse_hist`` to\n"
         "engage; above this the dense builder is used (no sparsity to exploit)."),
    ],
    [
        ("multiclass", "str, default=\"softmax\"",
         "Multiclass strategy: \"softmax\" grows one tree per class per round on\n"
         "the joint softmax cross-entropy gradients (shared binning, joint early\n"
         "stopping on multiclass log loss); \"ovr\" trains one independent binary\n"
         "booster per class. The classifier falls back to OvR when an opt-in\n"
         "feature the softmax loop does not support is enabled (kernel splits,\n"
         "GOSS, adaptive/product features, refinement/refit, auto-tune,\n"
         "stochastic routing)."),
        ("early_stopping_rounds", "int, default=50",
         "Stop if the eval metric does not improve for this many rounds (0\n"
         "disables). The metric is computed on the ``eval_set`` passed to\n"
         "``fit``, or on a ``validation_fraction`` holdout carved\n"
         "automatically when no ``eval_set`` is given."),
        ("validation_fraction", "float, default=0.15",
         "Fraction of the training data carved off (stratified for the\n"
         "classifier, seeded by ``seed``) to drive early stopping when ``fit``\n"
         "is called without an ``eval_set``, so the tree count adapts to the\n"
         "dataset instead of overfitting small data at the full\n"
         "``n_estimators`` budget. Not carved -- the full data is trained on,\n"
         "as before -- when early stopping is disabled or could never trigger\n"
         "(``n_estimators`` <= ``early_stopping_rounds``), when set to 0, when\n"
         "the holdout would have fewer than 50 rows, or under ``auto_tune``\n"
         "(which manages its own validation)."),
        ("seed", "int, default=0",
         "Random seed."),
        ("device", "str, default=\"auto\"",
         "\"auto\" picks \"cuda\" when available, else \"cpu\"; may also be set to\n"
         "\"cuda\" or \"cpu\" explicitly."),
        ("verbose", "bool, default=False",
         "Print per-iteration training progress."),
        ("cat_smoothing", "float, default=10.0",
         "Smoothing strength for the leakage-free target encoding of categorical\n"
         "columns (selected via the ``categorical_features`` argument to ``fit``)."),
        ("cat_per_class", "bool, default=False",
         "Multiclass only: target-encode each categorical column once per class\n"
         "(one-vs-rest indicators), replacing the column with class 0's encoding\n"
         "and appending the rest, instead of encoding the raw class index (whose\n"
         "mean is ordinal noise)."),
        ("cat_count_features", "bool, default=False",
         "Append a log1p train-frequency column per categorical, so trees can\n"
         "split on how common a category is independently of its target rate."),
        ("cat_combinations", "int, default=0",
         "Target-encode up to this many categorical *pairs* (CatBoost-style\n"
         "feature combinations, strongest parent columns first) and append them\n"
         "as extra columns, so trees can split on conjunctions that neither\n"
         "parent captures alone. 0 disables."),
        ("cat_combinations_min_card", "int, default=8",
         "Only categorical columns with at least this many distinct training\n"
         "values are eligible as pair parents: small-cat conjunctions are\n"
         "reachable with two ordinary splits, so their pair encodings are noise\n"
         "columns, while high-cardinality conjunctions carry unique signal."),
        ("calibrate_multiclass", "bool, default=False",
         "Vector-scale multiclass probabilities (per-class scale and bias on\n"
         "log p, re-softmaxed) fit on the ``eval_set`` after training. Bounded\n"
         "near identity and L2-pulled toward it, so a small validation set\n"
         "cannot push predictions far from the uncalibrated ones. Requires an\n"
         "``eval_set``; classifier-only, no effect on binary problems."),
        ("svd_features", "int, default=0",
         "Append this many PCA projections of the (encoded, standardized)\n"
         "feature matrix as extra columns. Axis-aligned splits cannot express\n"
         "linear combinations of features; the top principal directions hand\n"
         "the strongest ones to the grower as ordinary columns. 0 disables."),
        ("svd_min_features", "int, default=32",
         "Minimum encoded width for ``svd_features`` to engage: with few\n"
         "columns the top principal directions are near-copies of raw features\n"
         "and the extra columns just dilute sampling, while many correlated\n"
         "columns carry real linear structure."),
    ],
]

# Hyperparameters honored by the multi-task path (grow_multitask_tree /
# MultiTaskBooster); the rest are single-task-only and are omitted from that
# estimator's docstring.
_MULTITASK_PARAMS = frozenset({
    "n_estimators", "learning_rate", "max_leaves", "max_depth", "reg_lambda",
    "gamma", "min_child_weight", "min_samples_leaf", "subsample", "colsample",
    "max_bins", "early_stopping_rounds", "seed", "device",
})


def _render_params(names: frozenset[str] | None = None) -> str:
    """Render the (optionally filtered) parameter reference as a NumPy-style
    ``Parameters`` docstring section."""
    lines = ["", "    Parameters", "    ----------"]
    for group in _PARAM_GROUPS:
        rows = [r for r in group if names is None or r[0] in names]
        if not rows:
            continue
        for name, sig, desc in rows:
            lines.append(f"    {name} : {sig}")
            lines.extend(f"        {ln}" for ln in desc.split("\n"))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


_PARAMETERS_DOC = _render_params()
_MULTITASK_PARAMETERS_DOC = _render_params(_MULTITASK_PARAMS)


def _require_single_target(y: np.ndarray) -> np.ndarray:
    """Single-target estimators accept (n,) or a column vector (n, 1); a true
    multi-output (n, T>1) target is redirected to YABTMultiTaskRegressor rather
    than silently flattened."""
    if y.ndim == 2 and y.shape[1] == 1:
        return y.ravel()
    if y.ndim > 1:
        raise ValueError(
            f"y has shape {y.shape}; this estimator predicts a single target. "
            "For multiple targets use YABTMultiTaskRegressor, which grows one "
            "shared tree structure across all targets."
        )
    return y


class _YABTBase(BaseEstimator):
    def __init__(self, **kwargs):
        defaults = BoostParams()
        for name in _PARAM_NAMES:
            setattr(self, name, kwargs.pop(name, getattr(defaults, name)))
        self.cat_smoothing = kwargs.pop("cat_smoothing", 10.0)
        self.cat_per_class = kwargs.pop("cat_per_class", False)
        self.cat_count_features = kwargs.pop("cat_count_features", False)
        self.cat_combinations = kwargs.pop("cat_combinations", 0)
        self.cat_combinations_min_card = kwargs.pop("cat_combinations_min_card", 8)
        self.calibrate_multiclass = kwargs.pop("calibrate_multiclass", False)
        self.svd_features = kwargs.pop("svd_features", 0)
        self.svd_min_features = kwargs.pop("svd_min_features", 32)
        self.validation_fraction = kwargs.pop("validation_fraction", 0.15)
        if kwargs:
            raise TypeError(f"Unknown parameters: {sorted(kwargs)}")

    @classmethod
    def _get_param_names(cls):
        return sorted(_PARAM_NAMES + ["cat_smoothing", "cat_per_class",
                                      "cat_count_features", "cat_combinations",
                                      "cat_combinations_min_card",
                                      "calibrate_multiclass", "svd_features",
                                      "svd_min_features", "validation_fraction"])

    def _boost_params(self) -> BoostParams:
        return BoostParams(**{n: getattr(self, n) for n in _PARAM_NAMES})

    def _encode(self, X: np.ndarray, Y: np.ndarray | None, fit: bool) -> np.ndarray:
        """Replace categorical columns with leakage-free target encodings.

        ``Y`` may be a matrix (n, T) of T encoding targets (the multiclass
        per-class one-vs-rest indicators): target 0's encoding replaces the
        categorical column in place and targets 1..T-1 are appended as extra
        columns, one block per target. With ``cat_count_features`` a log1p
        train-frequency column per categorical is appended after those. With
        ``cat_combinations`` = P > 0, up to P categorical *pairs* (CatBoost-style
        feature combinations, picked by the target relevance of their parent
        columns) are target-encoded on target 0 and appended as extra columns,
        so trees can split on conjunctions like user x resource that neither
        parent encodes alone.
        """
        if not getattr(self, "_cat_idx", None):
            return self._append_svd(np.asarray(X, dtype=np.float32), fit)
        X = np.asarray(X)
        Xc = X[:, self._cat_idx]
        if fit:
            Y = np.asarray(Y, dtype=np.float32)
            Y2 = Y[:, None] if Y.ndim == 1 else Y
            self._cat_encs = []
            encs = []
            for t in range(Y2.shape[1]):
                e = PermutationTargetEncoder(smoothing=self.cat_smoothing, seed=self.seed)
                encs.append(e.fit_transform(Xc, Y2[:, t]))
                self._cat_encs.append(e)
            if self.cat_count_features:
                # Train-set frequency per category value; unseen values at
                # predict time fall back to count 0 (log1p -> 0).
                self._cat_counts = []
                for c in range(Xc.shape[1]):
                    vals, cnts = np.unique(Xc[:, c], return_counts=True)
                    self._cat_counts.append(dict(zip(vals.tolist(), cnts.tolist())))
            self._pair_specs = []
            if self.cat_combinations > 0 and Xc.shape[1] >= 2:
                # Pair only high-cardinality parents: a pair of small cats is
                # reachable with two ordinary splits, so its encoding is noise
                # columns (A/B: anneal, ~30 tiny cats, regressed +14%), while
                # high-cardinality conjunctions (e.g. user x resource) carry
                # signal trees cannot reconstruct (A/B: Amazon -12%). Among
                # those, rank parents by the target relevance of their single
                # encoding and keep the strongest-first top P pairs.
                y0 = Y2[:, 0]
                cards = [len(np.unique(Xc[:, c])) for c in range(Xc.shape[1])]
                elig = [c for c in range(Xc.shape[1])
                        if cards[c] >= self.cat_combinations_min_card]
                strength = {c: abs(float(np.corrcoef(encs[0][:, c], y0)[0, 1]))
                            if np.std(encs[0][:, c]) > 0 else 0.0
                            for c in elig}
                order = sorted(elig, key=lambda c: -strength[c])
                cands = [(order[i], order[j])
                         for i in range(len(order)) for j in range(i + 1, len(order))]
                cands = cands[: 3 * int(self.cat_combinations)]
                # Keep a pair only if its (leakage-free) encoding beats both
                # parents' target correlation by a margin — the same guard that
                # keeps product_features exact-neutral on data without the
                # structure (A/B: drops the date x demographic noise pairs that
                # regressed Marketing_Campaign while keeping user x resource
                # style conjunctions).
                if cands:
                    self._pair_specs = cands
                    Xp = self._pair_keys(Xc, fit=True)
                    e = PermutationTargetEncoder(smoothing=self.cat_smoothing, seed=self.seed)
                    encp = e.fit_transform(Xp, y0)
                    keep = []
                    for p, (i, j) in enumerate(cands):
                        if np.std(encp[:, p]) == 0:
                            continue
                        pc = abs(float(np.corrcoef(encp[:, p], y0)[0, 1]))
                        if pc > 1.1 * max(strength[i], strength[j]):
                            keep.append((i, j))
                    self._pair_specs = keep[: int(self.cat_combinations)]
        else:
            encs = [e.transform(Xc) for e in self._cat_encs]
        out = X.astype(np.float32, copy=True) if X.dtype != object else None
        if out is None:
            num_idx = [i for i in range(X.shape[1]) if i not in set(self._cat_idx)]
            out = np.empty(X.shape, dtype=np.float32)
            out[:, num_idx] = X[:, num_idx].astype(np.float32)
        out[:, self._cat_idx] = encs[0]
        extra = [np.asarray(e, dtype=np.float32) for e in encs[1:]]
        if self.cat_count_features:
            cnt = np.empty(Xc.shape, dtype=np.float32)
            for c in range(Xc.shape[1]):
                m = self._cat_counts[c]
                cnt[:, c] = [m.get(v, 0.0) for v in Xc[:, c]]
            extra.append(np.log1p(cnt))
        if getattr(self, "_pair_specs", None):
            Xp = self._pair_keys(Xc, fit=fit)
            if fit:
                self._pair_enc = PermutationTargetEncoder(
                    smoothing=self.cat_smoothing, seed=self.seed)
                extra.append(self._pair_enc.fit_transform(Xp, Y2[:, 0]))
            else:
                extra.append(self._pair_enc.transform(Xp))
        if extra:
            out = np.hstack([out] + extra)
        return self._append_svd(out, fit)

    def _append_svd(self, Xe: np.ndarray, fit: bool) -> np.ndarray:
        """Append ``svd_features`` PCA projections of the encoded matrix as
        extra columns. Axis-aligned splits cannot express linear combinations
        of features; the top principal directions hand the strongest ones to
        the grower as ordinary columns. NaNs are median-imputed for the
        projection only (the tree path keeps its own NaN handling)."""
        if not self.svd_features:
            return Xe
        if fit:
            # Gate on width: with few columns the top principal directions are
            # near-copies of raw features, and the added columns just dilute
            # colsample (A/B: airfoil F=5 +10%, concrete F=8 +6%); with many
            # correlated columns they carry real linear structure (A/B:
            # qsar-biodeg F=41 -9%, Bioresponse F=1776 -5%).
            self._svd_active = Xe.shape[1] >= self.svd_min_features
        if not self._svd_active:
            return Xe
        k = int(min(self.svd_features, Xe.shape[1] - 1, Xe.shape[0] - 1))
        if k < 1:
            return Xe
        if fit:
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler
            self._svd_medians = np.nanmedian(Xe, axis=0)
            self._svd_medians = np.where(np.isnan(self._svd_medians), 0.0, self._svd_medians)
            Z = np.where(np.isnan(Xe), self._svd_medians, Xe)
            self._svd_scaler = StandardScaler().fit(Z)
            self._svd = PCA(n_components=k, random_state=self.seed).fit(
                self._svd_scaler.transform(Z))
        Z = np.where(np.isnan(Xe), self._svd_medians, Xe)
        proj = self._svd.transform(self._svd_scaler.transform(Z)).astype(np.float32)
        return np.hstack([Xe, proj])

    def _pair_keys(self, Xc: np.ndarray, fit: bool) -> np.ndarray:
        """Combine each selected categorical pair into one key column. Codes are
        shifted so NaN codes (-1) stay valid; a fit-time multiplier keeps keys
        unique, and unseen fit-time values simply produce unseen keys, which the
        pair encoder maps to the prior."""
        if fit:
            self._pair_mult = [float(Xc[:, j].astype(np.float64).max()) + 2.0
                               for (_, j) in self._pair_specs]
        Xp = np.empty((Xc.shape[0], len(self._pair_specs)), dtype=np.float64)
        for p, ((i, j), mult) in enumerate(zip(self._pair_specs, self._pair_mult)):
            Xp[:, p] = (Xc[:, i].astype(np.float64) + 1.0) * mult + Xc[:, j].astype(np.float64)
        return Xp

    def _auto_eval_split(self, X, y, stratify: bool):
        """Carve a seeded ``validation_fraction`` holdout to drive early
        stopping when ``fit`` is called without an ``eval_set``. Skipped (and
        the full data trained on, as before) when early stopping or the
        fraction is disabled, or when the holdout would have fewer than 50
        rows, where the stopping signal is mostly noise. Also skipped when
        ``n_estimators <= early_stopping_rounds`` (stopping can then never
        trigger, so the holdout would be pure training-data loss) and under
        ``auto_tune``, which manages its own validation (CV on small data,
        which a small carved holdout would preempt and degrade)."""
        frac = float(self.validation_fraction or 0.0)
        n = len(y)
        if (self.early_stopping_rounds <= 0 or frac <= 0.0
                or int(n * frac) < 50 or self.auto_tune
                or self.n_estimators <= self.early_stopping_rounds):
            return X, y, None
        rng = np.random.default_rng(self.seed)
        if stratify:
            parts = []
            for c in np.unique(y):
                idx = np.flatnonzero(y == c)
                k = min(int(round(len(idx) * frac)), len(idx) - 1)
                if k > 0:
                    parts.append(rng.permutation(idx)[:k])
            val_idx = np.concatenate(parts) if parts else np.empty(0, dtype=np.intp)
        else:
            val_idx = rng.permutation(n)[: int(round(n * frac))]
        if len(val_idx) == 0:
            return X, y, None
        mask = np.zeros(n, dtype=bool)
        mask[val_idx] = True
        Xa = np.asarray(X)
        return Xa[~mask], y[~mask], (Xa[mask], y[mask])

    def fit(self, X, y, eval_set=None, categorical_features: list[int] | None = None):
        self._cat_idx = list(categorical_features) if categorical_features else []
        y = np.asarray(y, dtype=np.float32)
        y = _require_single_target(y)
        if eval_set is None:
            X, y, eval_set = self._auto_eval_split(X, y, stratify=False)
        yt = self._transform_y(y, fit=True)
        Xe = self._encode(X, yt, fit=True)
        ev = None
        if eval_set is not None:
            ev = (self._encode(eval_set[0], None, fit=False),
                  self._transform_y(np.asarray(eval_set[1], dtype=np.float32), fit=False))
        self.booster_ = Booster(self._boost_params(), self._loss())
        self.booster_.fit(Xe, yt, eval_set=ev)
        return self

    def _margin(self, X) -> np.ndarray:
        Xe = self._encode(X, None, fit=False)
        return self.booster_.predict_margin(Xe)


class YABTClassifier(_YABTBase, ClassifierMixin):
    """Binary and multiclass classifier. Multiclass trains native softmax
    boosting by default (see the ``multiclass`` parameter; One-vs-Rest is the
    fallback for opt-in features the softmax loop does not support)."""

    def fit(self, X, y, eval_set=None, categorical_features: list[int] | None = None):
        self._cat_idx = list(categorical_features) if categorical_features else []
        y = np.asarray(y, dtype=np.float32)
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        if eval_set is None:
            X, y, eval_set = self._auto_eval_split(X, y, stratify=True)

        self._is_binary = self.n_classes_ == 2
        if not self._is_binary and self.cat_per_class:
            # Per-class one-vs-rest encoding targets: the mean of the raw class
            # index is ordinal noise, but P(class k | category) is signal.
            Y_enc = (y[:, None] == self.classes_[None, :]).astype(np.float32)
        else:
            Y_enc = y
        Xe = self._encode(X, Y_enc, fit=True)

        if self._is_binary:
            yt = (y == self.classes_[1]).astype(np.float32)
            ev = None
            if eval_set is not None:
                y_eval = (np.asarray(eval_set[1], dtype=np.float32) == self.classes_[1]).astype(np.float32)
                ev = (self._encode(eval_set[0], None, fit=False), y_eval)
            self.booster_ = Booster(self._boost_params(), LogLoss())
            self.booster_.fit(Xe, yt, eval_set=ev)
        else:  # multiclass: native softmax by default, OvR as fallback
            ev = None
            if eval_set is not None:
                ev = (self._encode(eval_set[0], None, fit=False), eval_set[1])
            params = self._boost_params()
            # The softmax loop covers the default path; opt-ins that carry
            # per-booster state still need the independent OvR boosters.
            softmax_ok = not (
                params.kernel_splits or params.goss_enabled
                or params.adaptive_features or params.product_features
                or params.refine_steps > 0 or params.refit_every > 0
                or params.auto_tune or params.stochastic_routing
            )
            if params.multiclass == "softmax" and softmax_ok:
                self.booster_ = SoftmaxBooster(params)
            else:
                self.booster_ = MulticlassBooster(params)
            self.booster_.fit(Xe, y, eval_set=ev)
            self._calibration = None
            if self.calibrate_multiclass and eval_set is not None:
                self._fit_vector_calibration(
                    self.booster_.predict_proba(ev[0]),
                    np.searchsorted(self.classes_, np.asarray(eval_set[1], dtype=np.float32)),
                )

        return self

    def _fit_vector_calibration(self, P: np.ndarray, y_idx: np.ndarray) -> None:
        """Vector scaling of multiclass log-probabilities, fit on the eval split
        (per-class scale w and bias b on log p, re-softmaxed). Bounded close to
        identity and pulled toward it by an L2 penalty, so a small validation
        set cannot drag the mapping far from the uncalibrated probabilities."""
        from scipy import optimize
        K = P.shape[1]
        eps = 1e-15
        z = np.log(P + eps)
        n = len(y_idx)

        def loss(params):
            w, b = params[:K], params[K:]
            zp = w * z + b
            zp = zp - zp.max(axis=1, keepdims=True)
            p = np.exp(zp)
            p /= p.sum(axis=1, keepdims=True)
            nll = -np.mean(np.log(p[np.arange(n), y_idx] + eps))
            return nll + 1e-3 * (np.sum((w - 1.0) ** 2) + np.sum(b ** 2))

        res = optimize.minimize(
            loss, np.concatenate([np.ones(K), np.zeros(K)]),
            bounds=[(0.8, 1.2)] * K + [(-1.0, 1.0)] * K, method="L-BFGS-B",
        )
        self._calibration = (res.x[:K], res.x[K:])

    def _apply_calibration(self, P: np.ndarray) -> np.ndarray:
        if getattr(self, "_calibration", None) is None:
            return P
        w, b = self._calibration
        zp = w * np.log(P + 1e-15) + b
        zp -= zp.max(axis=1, keepdims=True)
        p = np.exp(zp)
        return p / p.sum(axis=1, keepdims=True)

    def predict_proba(self, X) -> np.ndarray:
        """Class probabilities, (n_samples, n_classes)."""
        Xe = self._encode(X, None, fit=False)
        if self._is_binary:
            p = expit(self.booster_.predict_margin(Xe))
            return np.stack([1 - p, p], axis=1)
        return self._apply_calibration(self.booster_.predict_proba(Xe))

    def predict(self, X) -> np.ndarray:
        """Predicted class label per row."""
        Xe = self._encode(X, None, fit=False)
        if self._is_binary:
            margin = self.booster_.predict_margin(Xe)
            return self.classes_[(margin > 0).astype(int)]
        proba = self._apply_calibration(self.booster_.predict_proba(Xe))
        return self.classes_[np.argmax(proba, axis=1)]


class YABTRegressor(_YABTBase, RegressorMixin):
    """Regressor; standardizes the target internally."""

    def _loss(self):
        return MSELoss()

    def _transform_y(self, y, fit):
        if fit:
            self._y_mean = float(y.mean())
            self._y_std = float(y.std()) or 1.0
        return (y - self._y_mean) / self._y_std

    def predict(self, X) -> np.ndarray:
        return self._margin(X) * self._y_std + self._y_mean


class YABTMultiTaskRegressor(_YABTBase, RegressorMixin):
    """Multi-output regression with one shared tree structure across all
    targets and per-target leaf values.

    Splits are chosen on the summed per-target gain, so correlated targets
    transfer through common splits and regularize one another; uncorrelated
    targets fall back to roughly independent fits. Each target is standardized
    internally. Numeric features only (no categorical target encoding in the
    multi-task path). Advanced single-task options (kernel splits, neural
    leaves, soft routing, interaction-aware growth) do not apply here.
    """

    def fit(self, X, Y, eval_set=None):
        Y = np.asarray(Y, dtype=np.float32)
        self._y_1d = Y.ndim == 1
        if self._y_1d:
            Y = Y[:, None]
        self.n_tasks_ = Y.shape[1]
        self._y_mean = Y.mean(axis=0)
        self._y_std = np.where(Y.std(axis=0) == 0, 1.0, Y.std(axis=0)).astype(np.float32)
        Yt = (Y - self._y_mean) / self._y_std

        ev = None
        if eval_set is not None:
            Yv = np.asarray(eval_set[1], dtype=np.float32)
            if Yv.ndim == 1:
                Yv = Yv[:, None]
            ev = (np.asarray(eval_set[0], dtype=np.float32), (Yv - self._y_mean) / self._y_std)

        self.booster_ = MultiTaskBooster(self._boost_params(), MSELoss())
        self.booster_.fit(np.asarray(X, dtype=np.float32), Yt, eval_set=ev)
        return self

    def predict(self, X) -> np.ndarray:
        m = self.booster_.predict_margin(np.asarray(X, dtype=np.float32))
        out = m * self._y_std + self._y_mean
        return out[:, 0] if self._y_1d else out


# Append the parameter reference to each estimator's docstring so the
# hyperparameters are discoverable via help()/IDEs. The multi-task estimator
# only honors the core split/sampling params, so it gets the trimmed version.
for _cls, _doc in (
    (YABTClassifier, _PARAMETERS_DOC),
    (YABTRegressor, _PARAMETERS_DOC),
    (YABTMultiTaskRegressor, _MULTITASK_PARAMETERS_DOC),
):
    _cls.__doc__ = (_cls.__doc__ or "").rstrip() + "\n" + _doc

"""Scikit-learn style estimators."""

from __future__ import annotations

import numpy as np
from scipy.special import expit
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

from .binning import PermutationTargetEncoder
from .boosting import Booster, BoostParams, LogLoss, MSELoss
from .multiclass import MulticlassBooster
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
        ("early_stopping_rounds", "int, default=0",
         "Stop if the eval metric does not improve for this many rounds (0\n"
         "disables; requires ``eval_set`` to be passed to ``fit``)."),
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
        if kwargs:
            raise TypeError(f"Unknown parameters: {sorted(kwargs)}")

    @classmethod
    def _get_param_names(cls):
        return sorted(_PARAM_NAMES + ["cat_smoothing"])

    def _boost_params(self) -> BoostParams:
        return BoostParams(**{n: getattr(self, n) for n in _PARAM_NAMES})

    def _encode(self, X: np.ndarray, y: np.ndarray | None, fit: bool) -> np.ndarray:
        """Replace categorical columns with leakage-free target encodings."""
        if not getattr(self, "_cat_idx", None):
            return np.asarray(X, dtype=np.float32)
        X = np.asarray(X)
        Xc = X[:, self._cat_idx]
        if fit:
            self._cat_enc = PermutationTargetEncoder(smoothing=self.cat_smoothing, seed=self.seed)
            enc = self._cat_enc.fit_transform(Xc, y)
        else:
            enc = self._cat_enc.transform(Xc)
        out = X.astype(np.float32, copy=True) if X.dtype != object else None
        if out is None:
            num_idx = [i for i in range(X.shape[1]) if i not in set(self._cat_idx)]
            out = np.empty(X.shape, dtype=np.float32)
            out[:, num_idx] = X[:, num_idx].astype(np.float32)
        out[:, self._cat_idx] = enc
        return out

    def fit(self, X, y, eval_set=None, categorical_features: list[int] | None = None):
        self._cat_idx = list(categorical_features) if categorical_features else []
        y = np.asarray(y, dtype=np.float32)
        y = _require_single_target(y)
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
    """Binary and multiclass classifier using One-vs-Rest for multiclass."""

    def fit(self, X, y, eval_set=None, categorical_features: list[int] | None = None):
        self._cat_idx = list(categorical_features) if categorical_features else []
        y = np.asarray(y, dtype=np.float32)
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)

        Xe = self._encode(X, y, fit=True)

        if self.n_classes_ == 2:
            # Binary classification: use standard binary booster
            yt = (y == self.classes_[1]).astype(np.float32)
            ev = None
            if eval_set is not None:
                y_eval = (np.asarray(eval_set[1], dtype=np.float32) == self.classes_[1]).astype(np.float32)
                ev = (self._encode(eval_set[0], None, fit=False), y_eval)
            self.booster_ = Booster(self._boost_params(), LogLoss())
            self.booster_.fit(Xe, yt, eval_set=ev)
            self._is_binary = True
        else:
            # Multiclass: use One-vs-Rest
            ev = None
            if eval_set is not None:
                ev = (self._encode(eval_set[0], None, fit=False), eval_set[1])
            self.booster_ = MulticlassBooster(self._boost_params())
            self.booster_.fit(Xe, y, eval_set=ev)
            self._is_binary = False

        return self

    def predict_proba(self, X) -> np.ndarray:
        """Predict class probabilities.

        Returns
        -------
        proba : np.ndarray of shape (n_samples, n_classes)
            Probability for each class
        """
        Xe = self._encode(X, None, fit=False)

        if self._is_binary:
            # Binary: use logistic function
            p = expit(self.booster_.predict_margin(Xe))
            return np.stack([1 - p, p], axis=1)
        else:
            # Multiclass: use softmax (via MulticlassBooster)
            return self.booster_.predict_proba(Xe)

    def predict(self, X) -> np.ndarray:
        """Predict class labels.

        Returns
        -------
        predictions : np.ndarray of shape (n_samples,)
            Predicted class label for each sample
        """
        Xe = self._encode(X, None, fit=False)

        if self._is_binary:
            # Binary: use margin threshold
            margin = self.booster_.predict_margin(Xe)
            return self.classes_[(margin > 0).astype(int)]
        else:
            # Multiclass: use argmax of probabilities
            return self.booster_.predict(Xe)


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

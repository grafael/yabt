"""Newton boosting training loop."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import torch

from .binning import Binner
from .histogram import build_histogram, per_feature_gain
from .kernel_splits import importance_weights
from .neural_leaves import fit_leaf_networks
from .tree import Tree, TreeParams, grow_tree, grow_tree_levelwise
from .adaptive_features import (
    AdaptiveFeatureSelector,
    GradientBasedOneSideSampling,
    FeatureInteractionDetector,
)

# Interaction steering helps on interaction-heavy mid/large tabular data but
# regresses on small datasets (A/B: -2.5 R2 pts on diabetes, n=442), so the
# "on by default" flag is gated off below this row count.
_INTERACTION_MIN_ROWS = 2000


class LogLoss:
    is_classification = True

    @staticmethod
    def base_score(y: torch.Tensor) -> float:
        p = float(y.mean().clamp(1e-6, 1 - 1e-6))
        return float(np.log(p / (1 - p)))

    @staticmethod
    def grad_hess(margin: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        p = torch.sigmoid(margin)
        return p - y, (p * (1 - p)).clamp_min(1e-6)

    @staticmethod
    def loss(margin: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.binary_cross_entropy_with_logits(margin, y)


class MSELoss:
    is_classification = False

    @staticmethod
    def base_score(y: torch.Tensor) -> float:
        return float(y.mean())

    @staticmethod
    def grad_hess(margin: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return margin - y, torch.ones_like(y)

    @staticmethod
    def loss(margin: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return 0.5 * (margin - y).square().mean()


@dataclass
class BoostParams:
    n_estimators: int = 500
    learning_rate: float = 0.1
    max_leaves: int = 31
    max_depth: int = 64
    reg_lambda: float = 1.0
    gamma: float = 0.0
    min_child_weight: float = 1e-3
    min_samples_leaf: int = 20
    subsample: float = 1.0
    colsample: float = 1.0
    max_bins: int = 256
    # Differentiable refinement (0 disables; effective steps adapt to dataset
    # size in fit). refine_min_gain skips refinement when the loss is already low.
    # Off by default: A/B shows it costs ~10% of fit time for a negligible gain
    # on real tabular data and a small *regression* on some targets. Opt in with
    # refine_steps > 0 when you want it.
    refine_steps: int = 0
    refine_lr: float = 0.02
    refine_min_gain: float = 1e-4
    refit_every: int = 0
    refit_steps: int = 30
    refit_lr: float = 0.05
    # Adaptive feature selection (novel)
    adaptive_features: bool = False
    feature_importance_alpha: float = 0.1
    goss_enabled: bool = False
    goss_ratio: float = 0.9
    # Feature interaction detection
    detect_interactions: bool = False
    # Interaction-aware splits: steer split selection toward features that
    # historically interact with features already on the node's path (novel).
    # The boost only flips near-ties (capped at 1 + interaction_boost) and
    # never inflates the gain used for split acceptance. On by default:
    # A/B-verified wins on interaction-heavy tabular data at ~10-20% cost.
    interaction_aware: bool = True
    interaction_boost: float = 0.5
    # Gradient-guided multiplicative feature construction (novel). Detects
    # feature groups that drive the residual multiplicatively -- via the
    # magnitude signal corr(x^2, r^2), which survives even when corr(x, r)==0 --
    # and appends their products as ordinary columns before training, handing the
    # greedy splitter interactions (e.g. x_i*x_j*x_k) it cannot otherwise find. A
    # correlation guard keeps a product only when it beats its components, so data
    # without multiplicative structure is left untouched (exact neutrality). Off
    # by default; A/B shows a large win on multiplicative targets (+~20% R^2 on
    # 3-way products) at neutral cost elsewhere.
    product_features: bool = False
    product_max_features: int = 5   # group size scanned for products
    product_max_order: int = 3      # highest product order considered
    product_min_corr: float = 0.03  # absolute residual-correlation floor to keep a product
    product_corr_gain: float = 1.3  # product must beat its best component corr by this factor
    # Kernel-based splits: RBF landmark splits for non-linear boundaries (novel)
    kernel_splits: bool = False
    kernel_candidates: int = 8
    kernel_gamma: float = 0.0   # 0 = median heuristic per landmark
    kernel_min_samples: int = 64
    # EXPERIMENTAL: weight kernel distances by per-feature split gain.
    # False = uniform (default; A/B-tested best overall), True/"node" = gains of
    # the node being split, "ema" = EMA of root-level gains from previous
    # iterations. Neither variant consistently beat uniform in benchmarks:
    # gain-adaptive distances inflate in-sample gain and select kernel splits
    # that generalize worse than the axis splits they displace.
    kernel_importance_weighting: bool | str = False
    # Neural leaf networks: per-leaf models instead of constants (novel).
    # On by default: linear leaves A/B-verified to win or tie at ~equal cost.
    neural_leaves: bool = True
    leaf_net_hidden: int = 0       # 0 = ridge-linear leaves; >0 = tanh MLP width
    leaf_net_features: int = 8     # top tree split features used as net inputs
    leaf_net_l2: float = 1.0
    leaf_net_steps: int = 40       # MLP Adam steps per tree
    leaf_net_lr: float = 0.05      # MLP Adam learning rate
    leaf_net_min_samples: int = 50  # smaller leaves keep their constant value
    # Auto-tuning: pick hyperparameters per dataset via a bounded validation
    # search over curated candidates before the final fit (novel; costs a
    # handful of early-stopped fits, skipped for datasets < 600 rows).
    auto_tune: bool = False
    # Stochastic routing: soft (expected-path) routing at inference (novel).
    # Trees are grown and trained hard; prediction becomes smooth in X.
    stochastic_routing: bool = False
    routing_tau: float = 0.05  # gate width as a fraction of the split feature's scale
    # Level-wise (breadth-first) growth: batched per-depth histograms + split
    # search with sibling subtraction, instead of the per-node best-first heap.
    # "auto" (default) device-gates it: ON for cuda when max_leaves >= 16, OFF
    # for cpu, because A/B shows ~1.8x faster on GPU (batching hides
    # kernel-launch/host-sync latency) but slower on CPU (no latency to
    # amortize), at neutral accuracy. It honors interaction steering but regresses
    # at tiny leaf budgets and can't do kernel splits, so "auto" falls back to the
    # heap grower below 16 leaves or when kernel_splits is on. True/False force
    # it on/off (True still skips kernel splits).
    levelwise: bool | str = "auto"
    # Training control
    early_stopping_rounds: int = 0
    seed: int = 0
    device: str = "auto"  # "auto" picks cuda when available, else cpu
    verbose: bool = False

    def resolve_device(self) -> str:
        if self.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.device


class Booster:
    def __init__(self, params: BoostParams, loss):
        self.p = params
        self.loss = loss
        self.trees: list[Tree] = []
        self.binner: Binner | None = None
        self.base_score: float = 0.0
        self.best_iter: int | None = None
        self.adaptive_selector: AdaptiveFeatureSelector | None = None
        self.goss: GradientBasedOneSideSampling | None = None
        self.interaction_detector: FeatureInteractionDetector | None = None
        self.tuning_report_: dict | None = None
        self.product_spec_: list[tuple[int, ...]] = []

    def _tree_params(self) -> TreeParams:
        p = self.p
        return TreeParams(
            max_leaves=p.max_leaves, max_depth=p.max_depth, reg_lambda=p.reg_lambda,
            gamma=p.gamma, min_child_weight=p.min_child_weight,
            min_samples_leaf=p.min_samples_leaf, learning_rate=p.learning_rate,
            kernel_splits=p.kernel_splits, kernel_candidates=p.kernel_candidates,
            kernel_gamma=p.kernel_gamma, kernel_min_samples=p.kernel_min_samples,
            kernel_importance_weighting=p.kernel_importance_weighting in (True, "node"),
        )

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        eval_set: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> "Booster":
        if self.p.auto_tune:
            from .auto_tune import tune_params
            self.p, self.tuning_report_ = tune_params(
                self.p, self.loss,
                np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32),
                eval_set,
            )
        p = self.p
        dev = self.device_ = p.resolve_device()
        gen = torch.Generator(device="cpu").manual_seed(p.seed)

        # Multiplicative feature construction: detect product groups from the
        # base-score residual and append them as columns before binning, so the
        # rest of the pipeline treats them as ordinary features. Detection runs on
        # the (encoded) X/y handed to the booster; for the OvR multiclass path
        # each per-class booster picks its own products.
        if p.product_features:
            from .product_features import detect_product_specs, expand_products
            self.product_spec_ = detect_product_specs(
                np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32),
                max_features=p.product_max_features, max_order=p.product_max_order,
                min_corr=p.product_min_corr, corr_gain=p.product_corr_gain,
            )
            if self.product_spec_:
                X = expand_products(X, self.product_spec_)
                if eval_set is not None:
                    eval_set = (expand_products(eval_set[0], self.product_spec_), eval_set[1])

        self.binner = Binner(max_bins=p.max_bins).fit(X)
        binned = self.binner.transform(X, device=dev)
        Xraw = torch.from_numpy(self.binner.impute(X)).to(dev)
        yt = torch.as_tensor(np.asarray(y, dtype=np.float32), device=dev)
        n, F = Xraw.shape

        # OPTIMIZATION: Adaptive refinement based on dataset size
        # Smart balance between speed and accuracy
        # Cap (never inflate) the user's refine_steps for smaller datasets, where
        # extra refinement tends to overfit. The caps are upper bounds only, so
        # refine_steps=0 always disables refinement (the old max(...) floors made
        # it impossible to turn off for 1K-50K-row datasets).
        effective_refine_steps = p.refine_steps
        if n < 1000:
            effective_refine_steps = 0  # Skip refinement (too small, would overfit)
            msg = "Dataset < 1K: refinement disabled"
        elif n < 5000:
            effective_refine_steps = min(p.refine_steps, 1)  # Light refinement only
            msg = f"Dataset < 5K: light refinement ({effective_refine_steps} step)"
        elif n < 50000:
            effective_refine_steps = min(p.refine_steps, 3)  # Balanced
            msg = f"Dataset 5K-50K: balanced refinement ({effective_refine_steps} steps)"
        # else: keep user's refine_steps (large dataset)

        if effective_refine_steps != p.refine_steps and p.verbose:
            print(f"[Auto-optimize] {msg}")

        # Initialize adaptive components
        if p.adaptive_features:
            self.adaptive_selector = AdaptiveFeatureSelector(
                F, device=dev, alpha=p.feature_importance_alpha, seed=p.seed
            )
        if p.goss_enabled:
            self.goss = GradientBasedOneSideSampling(device=dev, seed=p.seed)
        use_interaction_aware = p.interaction_aware and n >= _INTERACTION_MIN_ROWS
        if p.interaction_aware and not use_interaction_aware and p.verbose:
            print(f"[Auto-optimize] Dataset < {_INTERACTION_MIN_ROWS} rows: "
                  "interaction steering disabled")
        if p.detect_interactions or use_interaction_aware:
            self.interaction_detector = FeatureInteractionDetector(F, device=dev)

        self.base_score = self.loss.base_score(yt)
        margin = torch.full((n,), self.base_score, device=dev)

        if eval_set is not None:
            Xv = torch.from_numpy(self.binner.impute(eval_set[0])).to(dev)
            yv = torch.as_tensor(np.asarray(eval_set[1], dtype=np.float32), device=dev)
            margin_v = torch.full((Xv.shape[0],), self.base_score, device=dev)
            best_val, rounds_since_best = float("inf"), 0

        scales = self.binner.scales_.to(dev)
        tp = self._tree_params()
        Xn = Xraw / scales.clamp_min(1e-12) if p.kernel_splits else None
        kernel_ema_mode = p.kernel_splits and p.kernel_importance_weighting == "ema"
        kernel_ema = None

        for t in range(p.n_estimators):
            grad, hess = self.loss.grad_hess(margin, yt)

            # Gradient-based One-Side Sampling (GOSS): a novel sampling strategy
            row_weights = None
            if p.goss_enabled and self.goss is not None:
                rows, row_weights = self.goss.sample(grad, ratio=p.goss_ratio)
            elif p.subsample < 1.0:
                m = int(n * p.subsample)
                rows = torch.randperm(n, generator=gen)[:m].to(dev)
            else:
                rows = None

            # Adaptive feature selection: learn which features matter
            if p.adaptive_features and self.adaptive_selector is not None:
                k = max(1, int(F * p.colsample))
                fmask = self.adaptive_selector.get_feature_mask(k)
            elif p.colsample < 1.0:
                k = max(1, int(F * p.colsample))
                fmask = torch.zeros(F, dtype=torch.bool, device=dev)
                fmask[torch.randperm(F, generator=gen)[:k].to(dev)] = True
            else:
                fmask = None

            if rows is not None:
                g, h = grad[rows], hess[rows]
                if row_weights is not None:
                    g, h = g * row_weights, h * row_weights
            else:
                g, h = grad, hess

            kw_override = None
            if kernel_ema_mode:
                hist0 = build_histogram(binned if rows is None else binned[rows], g, h)
                gains = per_feature_gain(hist0, p.reg_lambda, p.min_child_weight, p.min_samples_leaf)
                gn = gains / gains.max().clamp_min(1e-12)
                # Weights come from *previous* iterations only, so the distance
                # is not adapted to the gradients the current tree fits.
                if kernel_ema is not None:
                    kw_override = importance_weights(kernel_ema)
                kernel_ema = gn if kernel_ema is None else 0.9 * kernel_ema + 0.1 * gn

            imat = None
            if use_interaction_aware and self.interaction_detector is not None:
                imat = self.interaction_detector.normalized_matrix()

            # "auto" turns level-wise on for cuda (A/B: ~1.8x faster, accuracy
            # neutral, and it honors interaction steering), EXCEPT at small leaf
            # budgets: with few leaves the heap's best-first order spends them
            # more optimally on sharp-boundary/XOR data (A/B: level-wise -7.5pt at
            # max_leaves=4, parity by 16), so defer to the heap there. Kernel
            # splits are unsupported, so fall back to the heap when those are on.
            auto_lw = p.levelwise == "auto" and dev == "cuda" and p.max_leaves >= 16
            use_levelwise = (p.levelwise is True or auto_lw) and not p.kernel_splits
            gb, gg2, gh2 = (binned[rows], g, h) if rows is not None else (binned, grad, hess)
            if use_levelwise:
                tree = grow_tree_levelwise(gb, gg2, gh2, self.binner, tp, fmask,
                                           interaction_matrix=imat,
                                           interaction_boost=p.interaction_boost)
            elif rows is not None:
                Xn_t = Xn[rows] if Xn is not None else None
                tree = grow_tree(gb, gg2, gh2, self.binner, tp, fmask, Xnorm=Xn_t,
                                 gen=gen, kernel_weights_override=kw_override,
                                 interaction_matrix=imat, interaction_boost=p.interaction_boost)
            else:
                tree = grow_tree(gb, gg2, gh2, self.binner, tp, fmask, Xnorm=Xn,
                                 gen=gen, kernel_weights_override=kw_override,
                                 interaction_matrix=imat, interaction_boost=p.interaction_boost)

            # Differentiable tree refinement: a novel optimization step
            # Use effective_refine_steps (may be reduced for small datasets)
            if effective_refine_steps > 0 and tree.num_leaves() > 1:
                from .refine_fast import refine_tree_fast
                refine_params = replace(p, refine_steps=effective_refine_steps)
                tree = refine_tree_fast(tree, Xraw, yt, margin, self.loss, refine_params)

            # Route the full training set once and reuse the leaf assignment for
            # both the per-leaf model fit and the margin update (adding leaf nets
            # does not change routing).
            leaf_idx = tree.apply(Xraw)

            # Neural leaf networks: fit per-leaf models to within-leaf residuals
            if p.neural_leaves:
                tree = fit_leaf_networks(tree, Xraw, yt, margin, self.loss, p, gen,
                                         leaf_idx=leaf_idx)

            # Update feature importance if adaptive selection is enabled
            if p.adaptive_features and self.adaptive_selector is not None:
                if rows is not None:
                    hist = build_histogram(binned[rows], grad[rows], hess[rows])
                else:
                    hist = build_histogram(binned, grad, hess)
                self.adaptive_selector.update_from_tree(hist)

            # Record ancestor-descendant split pairs for interaction detection
            if self.interaction_detector is not None:
                self.interaction_detector.update_from_path_pairs(tree.path_feature_pairs())

            contrib = tree.value[leaf_idx]
            if tree.leaf_net_feats is not None:
                contrib = contrib + tree.net_contribution(Xraw, leaf_idx)
            margin = margin + contrib
            self.trees.append(tree)

            if p.refit_every > 0 and (t + 1) % p.refit_every == 0:
                from .refine_fast import global_leaf_refit_fast
                margin = global_leaf_refit_fast(self.trees, Xraw, yt, self.base_score, self.loss, p)

            if eval_set is not None:
                margin_v = margin_v + tree.predict(Xv)
                if p.refit_every > 0 and (t + 1) % p.refit_every == 0:
                    margin_v = self._margin(Xv, soft=False)
                vl = float(self.loss.loss(margin_v, yv))
                if vl < best_val - 1e-7:
                    best_val, self.best_iter, rounds_since_best = vl, t, 0
                else:
                    rounds_since_best += 1
                if p.verbose and t % 50 == 0:
                    print(f"[{t}] train={float(self.loss.loss(margin, yt)):.5f} val={vl:.5f}")
                if p.early_stopping_rounds and rounds_since_best >= p.early_stopping_rounds:
                    break
        return self

    def top_interactions(self, k: int = 5) -> list[tuple[int, int, float]]:
        """Top-k detected feature interaction pairs (requires detect_interactions=True)."""
        if self.interaction_detector is None:
            return []
        return self.interaction_detector.get_top_interactions(k)

    def _margin(self, Xt: torch.Tensor, n_trees: int | None = None, soft: bool | None = None) -> torch.Tensor:
        out = torch.full((Xt.shape[0],), self.base_score, device=Xt.device)
        soft = self.p.stochastic_routing if soft is None else soft
        for tree in self.trees[:n_trees]:
            if soft and tree.gate_scale is not None:
                out = out + tree.predict_soft(Xt, self.p.routing_tau)
            else:
                out = out + tree.predict(Xt)
        return out

    def predict_margin(self, X: np.ndarray, use_best_iter: bool = True) -> np.ndarray:
        dev = getattr(self, "device_", None) or self.p.resolve_device()
        if self.product_spec_:
            from .product_features import expand_products
            X = expand_products(X, self.product_spec_)
        Xt = torch.from_numpy(self.binner.impute(X)).to(dev)
        n_trees = None
        if use_best_iter and self.best_iter is not None and self.p.refit_every == 0:
            n_trees = self.best_iter + 1
        return self._margin(Xt, n_trees).cpu().numpy()

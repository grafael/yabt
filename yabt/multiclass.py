"""Multiclass classification: native softmax boosting and One-vs-Rest (OvR).

``SoftmaxBooster`` is the default (see ``BoostParams.multiclass``): per round it
grows one tree per class from the gradients of the *joint* softmax cross-entropy
(XGBoost-style, class probabilities frozen at the start of the round), with a
single shared binning and joint early stopping on the multiclass log loss.
A/B on the TabArena multiclass suite (benchmarks/ab_softmax_multiclass.py) it
beats OvR on log loss on 7/8 datasets (mean delta -0.078, up to -0.18; the one
regression is +0.013 on the smallest, ~790-row anneal) at a median 1.27x faster
(shared binning + one early-stopping clock instead of K).

``MulticlassBooster`` (OvR) remains for the opt-in features the softmax loop
does not support (see ``sklearn_api.YABTClassifier``): it trains K independent
binary boosters, so every single-output feature works unchanged.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import torch

from .binning import Binner
from .boosting import _INTERACTION_MIN_ROWS, Booster, BoostParams, LogLoss
from .neural_leaves import fit_leaf_networks


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


class SoftmaxBooster:
    """Native softmax multiclass booster: K trees per round on joint gradients.

    Each round freezes the class probabilities ``P = softmax(margins)`` and, for
    every class k, grows one tree on the softmax cross-entropy gradients
    ``g = P_k - 1[y=k]``, ``h = P_k (1 - P_k)`` — so unlike OvR the classes
    compete inside one probability model, which is what the log-loss metric
    scores. The binning, grower caches, and interaction detector are shared
    across classes (OvR re-bins K times), and early stopping runs one joint
    clock on the eval multiclass log loss instead of K independent ones.

    Neural leaves stay exact under the joint loss via a margin shift: with the
    other classes' margins frozen, class k's softmax loss equals binary log
    loss at margin ``m_k - log(sum_{j!=k} exp(m_j))``, so the per-leaf models
    are fit with the ordinary ``LogLoss`` at that shifted margin.

    Supports the default-path features (growers, subsample/colsample,
    interaction steering, ``min_split_gain_rel``, neural leaves, early
    stopping). Opt-ins tied to per-booster state (kernel splits, GOSS,
    adaptive features, product features, refinement/refit, auto-tune,
    stochastic routing) are not wired up here — ``YABTClassifier`` falls back
    to OvR when any of them is enabled.
    """

    def __init__(self, params: BoostParams):
        self.params = params
        self.trees_: list[list] = []          # trees_[k] = class k's trees, one per round
        self.classes_: np.ndarray | None = None
        self.base_scores_: np.ndarray | None = None
        self.binner: Binner | None = None
        self.best_iter: int | None = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        eval_set: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> "SoftmaxBooster":
        p = self.params
        dev = self.device_ = p.resolve_device()
        gen = torch.Generator(device="cpu").manual_seed(p.seed)

        self.classes_ = np.unique(y)
        K = len(self.classes_)
        yi = torch.as_tensor(np.searchsorted(self.classes_, y), dtype=torch.long, device=dev)

        # One engine for grower dispatch: its per-fit caches (sparse layout,
        # feature-major binned copy, C-grower probe) are shared by all K classes.
        engine = self._engine = Booster(p, LogLoss())
        self.binner = engine.binner = Binner(max_bins=p.max_bins).fit(X)
        binned = self.binner.transform(X, device=dev)
        Xraw = torch.from_numpy(self.binner.impute(X)).to(dev)
        n, F = Xraw.shape
        onehot = torch.nn.functional.one_hot(yi, K).to(torch.float32)

        # Log-prior init, the softmax analogue of LogLoss.base_score.
        prior = (onehot.mean(dim=0)).clamp(1e-6, 1 - 1e-6)
        self.base_scores_ = torch.log(prior).cpu().numpy()
        M = torch.log(prior).expand(n, K).contiguous()

        if eval_set is not None:
            Xv = torch.from_numpy(self.binner.impute(eval_set[0])).to(dev)
            yvi = torch.as_tensor(np.searchsorted(self.classes_, eval_set[1]),
                                  dtype=torch.long, device=dev)
            Mv = torch.log(prior).expand(Xv.shape[0], K).contiguous()
            best_val, rounds_since_best = float("inf"), 0

        use_interaction_aware = p.interaction_aware and n >= _INTERACTION_MIN_ROWS
        if p.detect_interactions or use_interaction_aware:
            from .adaptive_features import FeatureInteractionDetector
            engine.interaction_detector = FeatureInteractionDetector(F, device=dev)

        tp = engine._tree_params()
        self.trees_ = [[] for _ in range(K)]

        for t in range(p.n_estimators):
            # Round-frozen probabilities (XGBoost-style): every class's tree this
            # round is grown against the same P, so class order does not matter.
            P = torch.softmax(M, dim=1)
            lse = torch.logsumexp(M, dim=1)

            if p.subsample < 1.0:
                m = int(n * p.subsample)
                rows = torch.randperm(n, generator=gen)[:m].to(dev)
            else:
                rows = None

            imat = None
            if use_interaction_aware and engine.interaction_detector is not None:
                imat = engine.interaction_detector.normalized_matrix()

            for k in range(K):
                pk = P[:, k]
                grad = pk - onehot[:, k]
                hess = (pk * (1 - pk)).clamp_min(1e-6)

                tp_t = tp
                if p.min_split_gain_rel > 0.0:
                    tp_t = replace(tp, gamma=p.gamma + p.min_split_gain_rel * float(grad.var()))

                if p.colsample < 1.0:
                    nf = max(1, int(F * p.colsample))
                    fmask = torch.zeros(F, dtype=torch.bool, device=dev)
                    fmask[torch.randperm(F, generator=gen)[:nf].to(dev)] = True
                else:
                    fmask = None

                g, h = (grad[rows], hess[rows]) if rows is not None else (grad, hess)
                tree = engine._grow_one_tree(binned, rows, g, h, tp_t, fmask, imat, gen)
                leaf_idx = tree.apply(Xraw)

                if p.neural_leaves:
                    # Class k's joint loss at frozen other-class margins is binary
                    # log loss at the shifted margin m_k - log(sum_{j!=k} e^{m_j}).
                    shifted = M[:, k] - lse - torch.log1p(-pk.clamp(max=1 - 1e-6))
                    tree = fit_leaf_networks(tree, Xraw, onehot[:, k], shifted,
                                             LogLoss(), p, gen, leaf_idx=leaf_idx)

                if engine.interaction_detector is not None:
                    engine.interaction_detector.update_from_path_pairs(tree.path_feature_pairs())

                contrib = tree.value[leaf_idx]
                if tree.leaf_net_feats is not None:
                    contrib = contrib + tree.net_contribution(Xraw, leaf_idx)
                M[:, k] = M[:, k] + contrib
                self.trees_[k].append(tree)
                if eval_set is not None:
                    Mv[:, k] = Mv[:, k] + tree.predict(Xv)

            if eval_set is not None:
                vl = float(torch.nn.functional.cross_entropy(Mv, yvi))
                if vl < best_val - 1e-7:
                    best_val, self.best_iter, rounds_since_best = vl, t, 0
                else:
                    rounds_since_best += 1
                if p.verbose and t % 50 == 0:
                    tl = float(torch.nn.functional.cross_entropy(M, yi))
                    print(f"[{t}] train={tl:.5f} val={vl:.5f}")
                if p.early_stopping_rounds and rounds_since_best >= p.early_stopping_rounds:
                    break
            elif p.verbose and t % 50 == 0:
                print(f"[{t}] train={float(torch.nn.functional.cross_entropy(M, yi)):.5f}")
        return self

    def predict_margin(self, X: np.ndarray) -> np.ndarray:
        """Per-class raw scores, (n_samples, n_classes)."""
        dev = getattr(self, "device_", None) or self.params.resolve_device()
        Xt = torch.from_numpy(self.binner.impute(X)).to(dev)
        n_trees = self.best_iter + 1 if self.best_iter is not None else None
        cols = []
        for k, trees in enumerate(self.trees_):
            out = torch.full((Xt.shape[0],), float(self.base_scores_[k]), device=dev)
            for tree in trees[:n_trees]:
                out = out + tree.predict(Xt)
            cols.append(out)
        return torch.stack(cols, dim=1).cpu().numpy()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Softmax of the per-class margins, (n_samples, n_classes)."""
        margins = self.predict_margin(X)
        exp_margins = np.exp(margins - margins.max(axis=1, keepdims=True))
        return exp_margins / exp_margins.sum(axis=1, keepdims=True)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicted class label per row."""
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

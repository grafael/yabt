"""Multi-task learning: shared tree structure with per-task leaf values.

One set of split structures is grown for all targets at once. At each node the
split is chosen on the *summed* per-task Newton gain, so correlated targets
share splits (transfer and mutual regularization); each leaf then stores a
value per task. This is the gradient-boosting analogue of a multi-output head
on a shared trunk, and it is what makes multi-task worthwhile over fitting one
independent model per target.

Scope: axis-aligned splits with constant vector leaves. The scalar-tree extras
(kernel splits, neural leaves, soft routing, interaction-aware growth) are
orthogonal opt-ins and are not wired into this path.
"""

from __future__ import annotations

import heapq
import itertools
from dataclasses import dataclass

import numpy as np
import torch

from .binning import Binner
from .boosting import BoostParams
from .histogram import build_histogram_multi, find_best_split_multi
from .tree import LEAF


@dataclass
class MultiTaskTree:
    """Shared split structure (axis splits) with vector leaf values."""

    feature: torch.Tensor    # (num_nodes,) long; -1 for leaves
    threshold: torch.Tensor  # (num_nodes,) float32
    left: torch.Tensor       # (num_nodes,) long
    right: torch.Tensor      # (num_nodes,) long
    value: torch.Tensor      # (num_nodes, T) float32; per-task leaf contribution
    depth: int

    def apply(self, X: torch.Tensor) -> torch.Tensor:
        n = X.shape[0]
        node = torch.zeros(n, dtype=torch.long, device=X.device)
        for _ in range(self.depth):
            f = self.feature[node]
            lf = f == LEAF
            if bool(lf.all()):
                break
            fc = f.clamp(min=0)
            go_left = X.gather(1, fc.unsqueeze(1)).squeeze(1) <= self.threshold[node]
            nxt = torch.where(go_left, self.left[node], self.right[node])
            node = torch.where(lf, node, nxt)
        return node

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """(n, T) per-task contributions."""
        return self.value[self.apply(X)]


def grow_multitask_tree(
    binned: torch.Tensor,  # (n, F) uint8
    grad: torch.Tensor,    # (n, T)
    hess: torch.Tensor,    # (n, T)
    binner: Binner,
    params: BoostParams,
    feature_mask: torch.Tensor | None = None,
) -> MultiTaskTree:
    dev = binned.device
    n, F = binned.shape
    T = grad.shape[1]
    lam, lr = params.reg_lambda, params.learning_rate

    feature: list[int] = []
    threshold: list[float] = []
    left: list[int] = []
    right: list[int] = []
    node_depth: list[int] = []
    values: list[torch.Tensor] = []

    def new_node(g_sum: torch.Tensor, h_sum: torch.Tensor, depth: int) -> int:
        nid = len(feature)
        feature.append(LEAF)
        threshold.append(0.0)
        left.append(-1)
        right.append(-1)
        values.append(-lr * g_sum / (h_sum + lam))
        node_depth.append(depth)
        return nid

    def node_totals(hist: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Per-task G/H from any feature's bins (feature 0); shape (T,).
        g = hist[:T, 0, :].sum(dim=1)
        h = hist[T : 2 * T, 0, :].sum(dim=1)
        return g, h

    all_rows = torch.arange(n, device=dev)
    root_hist = build_histogram_multi(binned, grad, hess)
    g0, h0 = node_totals(root_hist)
    root = new_node(g0, h0, 0)

    tiebreak = itertools.count()
    heap: list = []

    def push_candidate(nid: int, rows: torch.Tensor, hist: torch.Tensor):
        if node_depth[nid] >= params.max_depth:
            return
        gain, f, b = find_best_split_multi(
            hist, T, lam, params.gamma, params.min_child_weight,
            params.min_samples_leaf, feature_mask,
        )
        if gain > 0:
            heapq.heappush(heap, (-gain, next(tiebreak), nid, rows, hist, f, b))

    push_candidate(root, all_rows, root_hist)
    n_leaves = 1

    while heap and n_leaves < params.max_leaves:
        _, _, nid, rows, hist, f, b = heapq.heappop(heap)
        go_left = binned[rows, f] <= b
        rows_l, rows_r = rows[go_left], rows[~go_left]

        if rows_l.numel() <= rows_r.numel():
            hist_l = build_histogram_multi(binned[rows_l], grad[rows_l], hess[rows_l])
            hist_r = hist - hist_l
        else:
            hist_r = build_histogram_multi(binned[rows_r], grad[rows_r], hess[rows_r])
            hist_l = hist - hist_r

        d = node_depth[nid] + 1
        gl, hl = node_totals(hist_l)
        gr, hr = node_totals(hist_r)
        nl = new_node(gl, hl, d)
        nr = new_node(gr, hr, d)
        feature[nid] = f
        threshold[nid] = binner.edge_value(f, b)
        left[nid], right[nid] = nl, nr
        values[nid] = torch.zeros(T, device=dev)
        n_leaves += 1

        push_candidate(nl, rows_l, hist_l)
        push_candidate(nr, rows_r, hist_r)

    return MultiTaskTree(
        feature=torch.tensor(feature, dtype=torch.long, device=dev),
        threshold=torch.tensor(threshold, dtype=torch.float32, device=dev),
        left=torch.tensor(left, dtype=torch.long, device=dev),
        right=torch.tensor(right, dtype=torch.long, device=dev),
        value=torch.stack(values),
        depth=max(node_depth) + 1,
    )


class MultiTaskBooster:
    """Newton boosting with shared-structure trees over T targets at once."""

    def __init__(self, params: BoostParams, loss):
        self.p = params
        self.loss = loss
        self.trees: list[MultiTaskTree] = []
        self.binner: Binner | None = None
        self.base_score: torch.Tensor | None = None  # (T,)
        self.best_iter: int | None = None

    def _base_scores(self, Y: torch.Tensor) -> torch.Tensor:
        return torch.tensor(
            [self.loss.base_score(Y[:, t]) for t in range(Y.shape[1])],
            device=Y.device,
        )

    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        eval_set: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> "MultiTaskBooster":
        p = self.p
        dev = self.device_ = p.resolve_device()
        gen = torch.Generator(device="cpu").manual_seed(p.seed)

        self.binner = Binner(max_bins=p.max_bins).fit(X)
        binned = self.binner.transform(X, device=dev)
        Xraw = torch.from_numpy(self.binner.impute(X)).to(dev)
        Y = np.asarray(Y, dtype=np.float32)
        if Y.ndim == 1:
            Y = Y[:, None]
        Yt = torch.as_tensor(Y, device=dev)
        n, F = Xraw.shape
        self.n_tasks_ = Yt.shape[1]

        self.base_score = self._base_scores(Yt)
        margin = self.base_score.expand(n, self.n_tasks_).clone()

        if eval_set is not None:
            Xv = torch.from_numpy(self.binner.impute(eval_set[0])).to(dev)
            Yv = np.asarray(eval_set[1], dtype=np.float32)
            if Yv.ndim == 1:
                Yv = Yv[:, None]
            Yvt = torch.as_tensor(Yv, device=dev)
            margin_v = self.base_score.expand(Xv.shape[0], self.n_tasks_).clone()
            best_val, rounds_since_best = float("inf"), 0

        for t in range(p.n_estimators):
            grad, hess = self.loss.grad_hess(margin, Yt)

            if p.subsample < 1.0:
                m = int(n * p.subsample)
                rows = torch.randperm(n, generator=gen)[:m].to(dev)
            else:
                rows = None

            if p.colsample < 1.0:
                k = max(1, int(F * p.colsample))
                fmask = torch.zeros(F, dtype=torch.bool, device=dev)
                fmask[torch.randperm(F, generator=gen)[:k].to(dev)] = True
            else:
                fmask = None

            if rows is not None:
                tree = grow_multitask_tree(binned[rows], grad[rows], hess[rows],
                                           self.binner, p, fmask)
            else:
                tree = grow_multitask_tree(binned, grad, hess, self.binner, p, fmask)

            margin = margin + tree.predict(Xraw)
            self.trees.append(tree)

            if eval_set is not None:
                margin_v = margin_v + tree.predict(Xv)
                vl = float(self.loss.loss(margin_v, Yvt))
                if vl < best_val - 1e-7:
                    best_val, self.best_iter, rounds_since_best = vl, t, 0
                else:
                    rounds_since_best += 1
                if p.early_stopping_rounds and rounds_since_best >= p.early_stopping_rounds:
                    break

        return self

    def _margin(self, Xt: torch.Tensor, n_trees: int | None = None) -> torch.Tensor:
        out = self.base_score.expand(Xt.shape[0], self.n_tasks_).clone()
        for tree in self.trees[:n_trees]:
            out = out + tree.predict(Xt)
        return out

    def predict_margin(self, X: np.ndarray, use_best_iter: bool = True) -> np.ndarray:
        dev = getattr(self, "device_", None) or self.p.resolve_device()
        Xt = torch.from_numpy(self.binner.impute(X)).to(dev)
        n_trees = None
        if use_best_iter and self.best_iter is not None:
            n_trees = self.best_iter + 1
        return self._margin(Xt, n_trees).cpu().numpy()

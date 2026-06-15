"""Leaf-wise greedy tree growth and vectorized GPU inference."""

from __future__ import annotations

import heapq
import itertools
from dataclasses import dataclass

import torch

from .binning import MAX_BINS, Binner
from .histogram import build_histogram, find_best_split, per_feature_gain
from .kernel_splits import find_best_kernel_split, importance_weights

LEAF = -1
KERNEL_SPLIT = -2  # sentinel in Tree.feature for kernel-based (RBF) splits


@dataclass
class TreeParams:
    max_leaves: int = 31
    max_depth: int = 64
    reg_lambda: float = 1.0
    gamma: float = 0.0
    min_child_weight: float = 1e-3
    min_samples_leaf: int = 20
    learning_rate: float = 0.1
    # Kernel-based splits (non-linear decision boundaries; off by default)
    kernel_splits: bool = False
    kernel_candidates: int = 8
    kernel_gamma: float = 0.0   # 0 = median heuristic per landmark
    kernel_min_samples: int = 64
    kernel_importance_weighting: bool = False  # weight distances by node split gains (experimental)


@dataclass
class Tree:
    """Flat tensor representation. Internal nodes split as ``x[f] <= threshold``
    (left) using raw feature values; ``threshold`` is initialized to the bin
    edge chosen by the grower and may later be moved by refinement.

    ``feature`` is -1 (LEAF) for leaves and -2 (KERNEL_SPLIT) for kernel-based
    splits, which route by RBF similarity to a landmark instead of one feature:
    rows with ``exp(-sum_f prec[f] * (x[f] - center[f])^2) <= threshold`` go
    left. Kernel nodes index into ``kernel_centers``/``kernel_prec`` via
    ``kernel_id``; trees without kernel splits leave those fields None."""

    feature: torch.Tensor       # (num_nodes,) long; -1 leaf, -2 kernel split
    threshold: torch.Tensor     # (num_nodes,) float32 raw-space thresholds
    left: torch.Tensor          # (num_nodes,) long; -1 for leaves
    right: torch.Tensor         # (num_nodes,) long
    value: torch.Tensor         # (num_nodes,) float32; lr-scaled contribution at leaves
    depth: int
    kernel_id: torch.Tensor | None = None       # (num_nodes,) long; -1 unless kernel split
    kernel_centers: torch.Tensor | None = None  # (K, F) raw-space landmark centers
    kernel_prec: torch.Tensor | None = None     # (K, F) per-feature precision gamma/scale^2
    # Neural leaves: per-leaf models over a tree-level feature subset, added to
    # ``value`` at the assigned leaf. Either linear (leaf_net_linear) or a tiny
    # tanh MLP (leaf_net_W1/b1/W2); weights are raw-space, learning-rate folded.
    leaf_net_feats: torch.Tensor | None = None   # (K,) long feature indices
    leaf_net_linear: torch.Tensor | None = None  # (num_nodes, K)
    leaf_net_W1: torch.Tensor | None = None      # (num_nodes, H, K)
    leaf_net_b1: torch.Tensor | None = None      # (num_nodes, H)
    leaf_net_W2: torch.Tensor | None = None      # (num_nodes, H)
    # Per-leaf input envelope: net inputs are clamped to the range the leaf's
    # model was fit on. Leaf regions are unbounded in some directions, and a
    # linear model extrapolated far outside its training range explodes.
    leaf_net_lo: torch.Tensor | None = None      # (num_nodes, K)
    leaf_net_hi: torch.Tensor | None = None      # (num_nodes, K)
    # Per-node gate scale for stochastic (soft) routing: the split feature's
    # robust scale (or the kernel feature's std), so gate widths in
    # ``predict_soft`` are scale-invariant. 1.0 at leaves.
    gate_scale: torch.Tensor | None = None       # (num_nodes,)

    @property
    def is_leaf(self) -> torch.Tensor:
        return self.feature == LEAF

    def apply(self, X: torch.Tensor) -> torch.Tensor:
        """Leaf (node) index per row of raw feature matrix X (n, F)."""
        n = X.shape[0]
        node = torch.zeros(n, dtype=torch.long, device=X.device)
        for _ in range(self.depth):
            f = self.feature[node]
            lf = f == LEAF
            if bool(lf.all()):
                break
            fc = f.clamp(min=0)
            go_left = X.gather(1, fc.unsqueeze(1)).squeeze(1) <= self.threshold[node]
            if self.kernel_id is not None:
                km = f == KERNEL_SPLIT
                if bool(km.any()):
                    kid = self.kernel_id[node[km]]
                    d2 = ((X[km] - self.kernel_centers[kid]).square() * self.kernel_prec[kid]).sum(1)
                    go_left[km] = torch.exp(-d2) <= self.threshold[node[km]]
            nxt = torch.where(go_left, self.left[node], self.right[node])
            node = torch.where(lf, node, nxt)
        return node

    def net_contribution(self, X: torch.Tensor, node: torch.Tensor) -> torch.Tensor:
        """Per-row leaf-network output for the given node assignment; 0 if none."""
        if self.leaf_net_feats is None:
            return torch.zeros(X.shape[0], device=X.device)
        Xs = X[:, self.leaf_net_feats]
        if self.leaf_net_lo is not None:
            Xs = Xs.clamp(self.leaf_net_lo[node], self.leaf_net_hi[node])
        if self.leaf_net_linear is not None:
            return (self.leaf_net_linear[node] * Xs).sum(dim=1)
        hid = torch.tanh(torch.einsum("nhk,nk->nh", self.leaf_net_W1[node], Xs) + self.leaf_net_b1[node])
        return (self.leaf_net_W2[node] * hid).sum(dim=1)

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        node = self.apply(X)
        out = self.value[node]
        if self.leaf_net_feats is not None:
            out = out + self.net_contribution(X, node)
        return out

    def predict_soft(self, X: torch.Tensor, tau: float = 0.1) -> torch.Tensor:
        """Expected prediction under stochastic routing: each internal node
        routes left with probability sigmoid((threshold - x) / (tau * scale)),
        and the output is the leaf-probability-weighted sum of leaf models.
        Smooth in X; converges to ``predict`` as tau -> 0."""
        assert self.gate_scale is not None, "tree was grown without gate scales"
        n = X.shape[0]
        n_nodes = self.value.shape[0]
        prob = torch.zeros(n, n_nodes, device=X.device)
        prob[:, 0] = 1.0
        # Children are always created after their parent, so ascending node id
        # is a topological order.
        for i in torch.nonzero(self.feature != LEAF).squeeze(1).tolist():
            f = int(self.feature[i])
            if f == KERNEL_SPLIT:
                kid = int(self.kernel_id[i])
                xval = torch.exp(
                    -((X - self.kernel_centers[kid]).square() * self.kernel_prec[kid]).sum(dim=1)
                )
            else:
                xval = X[:, f]
            width = tau * float(self.gate_scale[i]) + 1e-12
            g = torch.sigmoid((self.threshold[i] - xval) / width)
            prob[:, int(self.left[i])] += prob[:, i] * g
            prob[:, int(self.right[i])] += prob[:, i] * (1 - g)

        leaf = self.is_leaf
        p_leaf = prob[:, leaf]
        out = p_leaf @ self.value[leaf]
        if self.leaf_net_feats is not None:
            Xs = X[:, self.leaf_net_feats]
            if self.leaf_net_lo is not None:
                # (n, L, K): each leaf's net sees inputs clamped to its own envelope
                Xs = Xs.unsqueeze(1).clamp(self.leaf_net_lo[leaf], self.leaf_net_hi[leaf])
            else:
                Xs = Xs.unsqueeze(1)
            if self.leaf_net_linear is not None:
                nets = (self.leaf_net_linear[leaf] * Xs).sum(dim=-1)  # (n, L)
            else:
                hid = torch.tanh(
                    torch.einsum("lhk,nlk->nlh", self.leaf_net_W1[leaf], Xs) + self.leaf_net_b1[leaf]
                )
                nets = (self.leaf_net_W2[leaf] * hid).sum(dim=-1)     # (n, L)
            out = out + (p_leaf * nets).sum(dim=1)
        return out

    def num_leaves(self) -> int:
        return int((self.feature == LEAF).sum())

    def path_feature_pairs(self) -> list[tuple[int, int]]:
        """(ancestor_feature, descendant_feature) pairs along root-to-leaf
        paths, axis splits only, the raw signal for interaction detection."""
        feature = self.feature.tolist()
        left, right = self.left.tolist(), self.right.tolist()
        pairs: list[tuple[int, int]] = []
        stack: list[tuple[int, tuple[int, ...]]] = [(0, ())]
        while stack:
            nid, anc = stack.pop()
            f = feature[nid]
            if f == LEAF:
                continue
            if f >= 0:
                pairs.extend((a, f) for a in anc)
                anc = anc + (f,)
            stack.append((left[nid], anc))
            stack.append((right[nid], anc))
        return pairs


def grow_tree(
    binned: torch.Tensor,   # (n, F) uint8, training rows for this tree
    grad: torch.Tensor,     # (n,)
    hess: torch.Tensor,     # (n,)
    binner: Binner,
    params: TreeParams,
    feature_mask: torch.Tensor | None = None,
    Xnorm: torch.Tensor | None = None,  # (n, F) scale-normalized raw rows; enables kernel splits
    gen: torch.Generator | None = None,
    kernel_weights_override: torch.Tensor | None = None,  # (F,) tree-level distance weights
    interaction_matrix: torch.Tensor | None = None,  # (F, F) in [0,1]; learned interactions
    interaction_boost: float = 0.5,  # selection boost cap for interacting features
) -> Tree:
    dev = binned.device
    n, F = binned.shape
    scales = binner.scales_.to(dev).clamp_min(1e-12)
    use_kernel = params.kernel_splits and Xnorm is not None
    if use_kernel and gen is None:
        gen = torch.Generator(device="cpu").manual_seed(0)

    feature: list[int] = []
    threshold: list[float] = []
    left: list[int] = []
    right: list[int] = []
    value: list[float] = []
    node_depth: list[int] = []
    kernel_id: list[int] = []
    kcenters: list[torch.Tensor] = []   # normalized-space centers, converted at the end
    kgammas: list[float] = []
    kweights: list[torch.Tensor] = []   # per-feature distance weights per kernel node
    node_gate_scale: list[float] = []   # soft-routing gate width per node

    def new_node(g_sum: float, h_sum: float, depth: int) -> int:
        nid = len(feature)
        feature.append(LEAF)
        threshold.append(0.0)
        left.append(-1)
        right.append(-1)
        value.append(-params.learning_rate * g_sum / (h_sum + params.reg_lambda))
        node_depth.append(depth)
        kernel_id.append(-1)
        node_gate_scale.append(1.0)
        return nid

    all_rows = torch.arange(n, device=dev)
    root_hist = build_histogram(binned, grad, hess)
    root = new_node(float(grad.sum()), float(hess.sum()), 0)

    # heap entries: (-gain, tiebreak, node_id, rows, hist, spec, path) where
    # spec is ("axis", f, b) or ("kernel", ...) and path is the tuple of axis
    # features split on between the root and this node.
    tiebreak = itertools.count()
    heap: list = []

    def push_candidate(nid: int, rows: torch.Tensor, hist: torch.Tensor,
                       path: tuple[int, ...] = ()):
        if node_depth[nid] >= params.max_depth:
            return
        boost = None
        if interaction_matrix is not None and path:
            pf = torch.tensor(sorted(set(path)), dtype=torch.long, device=dev)
            boost = 1.0 + interaction_boost * interaction_matrix[pf].amax(dim=0)
        gain, f, b = find_best_split(
            hist, params.reg_lambda, params.gamma,
            params.min_child_weight, params.min_samples_leaf, feature_mask,
            feature_boost=boost,
        )
        spec = ("axis", f, b) if gain > 0 else None
        if use_kernel and rows.numel() >= params.kernel_min_samples:
            fw = kernel_weights_override
            if fw is None and params.kernel_importance_weighting:
                fw = importance_weights(per_feature_gain(
                    hist, params.reg_lambda, params.min_child_weight, params.min_samples_leaf,
                ))
            ks = find_best_kernel_split(
                Xnorm[rows], grad[rows], hess[rows],
                params.kernel_candidates, params.kernel_gamma, params.reg_lambda,
                params.gamma, params.min_child_weight, params.min_samples_leaf, gen,
                feature_weights=fw,
            )
            if ks is not None and ks[0] > gain:
                gain, center, kgam, thr, go_left, z_std = ks
                spec = ("kernel", center, kgam, thr, go_left, fw, z_std)
        if spec is not None:
            heapq.heappush(heap, (-gain, next(tiebreak), nid, rows, hist, spec, path))

    push_candidate(root, all_rows, root_hist)
    n_leaves = 1

    while heap and n_leaves < params.max_leaves:
        _, _, nid, rows, hist, spec, path = heapq.heappop(heap)
        if spec[0] == "axis":
            _, f, b = spec
            go_left = binned[rows, f] <= b
        else:
            go_left = spec[4]
        rows_l, rows_r = rows[go_left], rows[~go_left]

        # Histogram for the smaller child by scatter; sibling by subtraction.
        if rows_l.numel() <= rows_r.numel():
            hist_l = build_histogram(binned[rows_l], grad[rows_l], hess[rows_l])
            hist_r = hist - hist_l
        else:
            hist_r = build_histogram(binned[rows_r], grad[rows_r], hess[rows_r])
            hist_l = hist - hist_r

        # Any single feature's bins sum to the node totals; use feature 0.
        d = node_depth[nid] + 1
        nl = new_node(float(hist_l[0, 0].sum()), float(hist_l[1, 0].sum()), d)
        nr = new_node(float(hist_r[0, 0].sum()), float(hist_r[1, 0].sum()), d)
        if spec[0] == "axis":
            feature[nid] = f
            threshold[nid] = binner.edge_value(f, b)
            node_gate_scale[nid] = float(scales[f])
        else:
            _, center, kgam, thr, _, fw, z_std = spec
            feature[nid] = KERNEL_SPLIT
            threshold[nid] = thr
            kernel_id[nid] = len(kcenters)
            kcenters.append(center)
            kgammas.append(kgam)
            kweights.append(fw if fw is not None else torch.ones(F, device=dev))
            node_gate_scale[nid] = max(z_std, 1e-6)
        left[nid], right[nid] = nl, nr
        value[nid] = 0.0
        n_leaves += 1

        child_path = path + (f,) if spec[0] == "axis" else path
        push_candidate(nl, rows_l, hist_l, child_path)
        push_candidate(nr, rows_r, hist_r, child_path)

    if kcenters:
        # Fold normalization and distance weights into raw space:
        # gamma * sum_f w_f * ((x_f - c_f) / s_f)^2 ==
        # sum_f (gamma * w_f / s_f^2) * (x_f - c_f)^2, so inference needs only X raw.
        gam = torch.tensor(kgammas, dtype=torch.float32, device=dev).unsqueeze(1)
        kid_t = torch.tensor(kernel_id, dtype=torch.long, device=dev)
        kernel_centers = torch.stack(kcenters) * scales
        kernel_prec = gam * torch.stack(kweights) / scales.square()
    else:
        kid_t = kernel_centers = kernel_prec = None

    return Tree(
        feature=torch.tensor(feature, dtype=torch.long, device=dev),
        threshold=torch.tensor(threshold, dtype=torch.float32, device=dev),
        left=torch.tensor(left, dtype=torch.long, device=dev),
        right=torch.tensor(right, dtype=torch.long, device=dev),
        value=torch.tensor(value, dtype=torch.float32, device=dev),
        depth=max(node_depth) + 1,
        kernel_id=kid_t,
        kernel_centers=kernel_centers,
        kernel_prec=kernel_prec,
        gate_scale=torch.tensor(node_gate_scale, dtype=torch.float32, device=dev),
    )


def grow_tree_levelwise(
    binned: torch.Tensor,   # (n, F) uint8, training rows for this tree
    grad: torch.Tensor,     # (n,)
    hess: torch.Tensor,     # (n,)
    binner: Binner,
    params: TreeParams,
    feature_mask: torch.Tensor | None = None,
    interaction_matrix: torch.Tensor | None = None,  # (F, F) in [0,1]; learned interactions
    interaction_boost: float = 0.5,  # selection boost cap for interacting features
) -> Tree:
    """Breadth-first (level-wise) axis-split grower.

    Equivalent split math to :func:`grow_tree`, but every node at a given depth
    is processed in one batch: histograms for all active nodes are held as
    ``(3, M, F, B)``, one vectorized argmax picks every split, and one masked
    write repartitions all rows. Python and host<->device syncs are O(depth)
    instead of O(leaves), so the GPU sees a handful of large kernels rather than
    thousands of tiny latency-bound ones.

    Histograms are carried between levels and the next level is built with the
    sibling-subtraction trick: for each split, only the *smaller* child's rows
    are scattered, and the larger sibling is ``parent - smaller``. This roughly
    halves the scatter work (the leaf-wise grower's key optimization), so the
    grower is competitive on CPU too, not just GPU.

    ``interaction_matrix`` enables interaction-aware steering, identically to
    :func:`grow_tree`: the per-node split is selected by ``argmax(gain * boost)``
    where ``boost`` lifts features that historically interact with a feature
    already on the node's root path, but acceptance and the leaf budget use the
    true unboosted gain. A per-node path-feature mask is carried alongside the
    histograms.

    The ``max_leaves`` budget is honored by keeping the top-gain splits per level
    (``topk``), so growth order differs from the best-first heap in ``grow_tree``;
    chosen splits can differ on near-ties. Axis splits only; kernel splits are
    not supported here (caller falls back to grow_tree).
    """
    dev = binned.device
    n, F = binned.shape
    B = MAX_BINS
    lam, gamma = params.reg_lambda, params.gamma
    mcw, msl = params.min_child_weight, params.min_samples_leaf
    lr = params.learning_rate
    scales_cpu = binner.scales_.to(dev).clamp_min(1e-12).tolist()
    fidx = torch.arange(F, device=dev)
    imat = interaction_matrix  # (F, F) in [0,1] or None; row p = how feature p interacts

    def scatter_hist(slot: torch.Tensor, brows: torch.Tensor, grows: torch.Tensor,
                     hrows: torch.Tensor, K: int) -> torch.Tensor:
        """Histogram (3, K, F, B): rows accumulate into node ``slot[i]``."""
        nr = slot.shape[0]
        flat = (slot[:, None] * (F * B) + fidx[None, :] * B + brows.long()).reshape(-1)
        out = torch.zeros(3, K * F * B, dtype=torch.float32, device=dev)
        out[0].scatter_add_(0, flat, grows[:, None].expand(nr, F).reshape(-1))
        out[1].scatter_add_(0, flat, hrows[:, None].expand(nr, F).reshape(-1))
        out[2].scatter_add_(0, flat, torch.ones(nr * F, dtype=torch.float32, device=dev))
        return out.view(3, K, F, B)

    feature: list[int] = []
    threshold: list[float] = []
    left: list[int] = []
    right: list[int] = []
    value: list[float] = []
    node_depth: list[int] = []
    gate: list[float] = []

    def add_leaf(v: float, d: int) -> int:
        nid = len(feature)
        feature.append(LEAF)
        threshold.append(0.0)
        left.append(-1)
        right.append(-1)
        value.append(v)
        node_depth.append(d)
        gate.append(1.0)
        return nid

    root = add_leaf(-lr * float(grad.sum()) / (float(hess.sum()) + lam), 0)
    node_of_row = torch.zeros(n, dtype=torch.long, device=dev)
    active = [root]
    # Root histogram is built once up front; deeper levels come from subtraction.
    hist = scatter_hist(torch.zeros(n, dtype=torch.long, device=dev), binned, grad, hess, 1)
    # Per-active-node mask of features split on its root path (for interaction
    # steering); carried alongside hist. None when steering is off.
    path_mask = torch.zeros(1, F, dtype=torch.bool, device=dev) if imat is not None else None
    splits_remaining = params.max_leaves - 1
    depth = 0

    while active and depth < params.max_depth and splits_remaining > 0:
        M = len(active)
        if depth == 0:
            # Root level: every row maps to the single root node, so the
            # active-row gather is the identity over the full (and largest)
            # matrix. Skip it -- use binned/grad/hess directly with pr = 0.
            arows = torch.arange(n, device=dev)
            pr = torch.zeros(n, dtype=torch.long, device=dev)
            bb, gg, hh = binned, grad, hess
        else:
            active_t = torch.tensor(active, dtype=torch.long, device=dev)
            node_to_pos = torch.full((len(feature),), -1, dtype=torch.long, device=dev)
            node_to_pos[active_t] = torch.arange(M, device=dev)
            pos_of_row = node_to_pos[node_of_row]
            # Active-row indices computed once and reused: boolean-mask indexing
            # (binned[arows], grad[arows], ...) each runs its own nonzero kernel,
            # so gathering with a shared integer index instead drops 3 nonzero
            # passes per level (nonzero was ~22% of GPU tree-grow time).
            arows = (pos_of_row >= 0).nonzero(as_tuple=True)[0]
            pr = pos_of_row[arows]
            bb = binned[arows]
            gg, hh = grad[arows], hess[arows]

        cum = hist.cumsum(-1)  # fused over the (grad, hess, count) channels
        GL, HL, CL = cum[0], cum[1], cum[2]
        G, H, C = GL[..., -1:], HL[..., -1:], CL[..., -1:]
        GR, HR, CR = G - GL, H - HL, C - CL
        gain = 0.5 * (GL.square() / (HL + lam) + GR.square() / (HR + lam) - G.square() / (H + lam)) - gamma
        valid = (CL >= msl) & (CR >= msl) & (HL >= mcw) & (HR >= mcw)
        valid[..., -1] = False
        if feature_mask is not None:
            valid &= feature_mask.view(1, F, 1)
        gain = torch.where(valid, gain, torch.full_like(gain, float("-inf")))

        flat_gain = gain.view(M, -1)
        if imat is None:
            best_gain, best_idx = flat_gain.max(dim=1)
        else:
            # Steer selection toward path-interacting features; boost[m,j] =
            # 1 + ib * max over path features p of imat[p, j] (0 when path empty,
            # since imat >= 0). Acceptance still uses the true unboosted gain.
            boost = 1.0 + interaction_boost * (path_mask.unsqueeze(2) * imat).amax(dim=1)
            best_idx = (gain * boost.unsqueeze(-1)).view(M, -1).argmax(dim=1)
            best_gain = flat_gain.gather(1, best_idx[:, None]).squeeze(1)
        f, b = best_idx // B, best_idx % B
        do_split = best_gain > 0
        if int(do_split.sum()) > splits_remaining:  # honor the leaf budget: keep top gains
            cand = best_gain.masked_fill(~do_split, float("-inf"))
            keep = torch.topk(cand, splits_remaining).indices
            do_split = torch.zeros_like(do_split).index_fill_(0, keep, True)

        idxM = torch.arange(M, device=dev)
        Gtot, Htot = hist[0, :, 0, :].sum(-1), hist[1, :, 0, :].sum(-1)
        Ctot = hist[2, :, 0, :].sum(-1)
        gL, hL, cL = GL[idxM, f, b], HL[idxM, f, b], CL[idxM, f, b]
        vL = -lr * gL / (hL + lam)
        vR = -lr * (Gtot - gL) / ((Htot - hL) + lam)
        small_is_left = cL <= (Ctot - cL)  # smaller child by row count

        ds, fl, bl = do_split.tolist(), f.tolist(), b.tolist()
        vLl, vRl = vL.tolist(), vR.tolist()
        child_left = [-1] * M
        child_right = [-1] * M
        next_active: list[int] = []
        d1 = depth + 1
        for m in range(M):
            if not ds[m]:
                continue
            gid, fm, bm = active[m], fl[m], bl[m]
            feature[gid] = fm
            threshold[gid] = binner.edge_value(fm, bm)
            gate[gid] = scales_cpu[fm]
            value[gid] = 0.0
            nl, nr = add_leaf(vLl[m], d1), add_leaf(vRl[m], d1)
            left[gid], right[gid] = nl, nr
            child_left[m], child_right[m] = nl, nr
            next_active += (nl, nr)
        splits_remaining -= len(next_active) // 2

        # Repartition every active row in one masked write.
        cl_t = torch.tensor(child_left, dtype=torch.long, device=dev)
        cr_t = torch.tensor(child_right, dtype=torch.long, device=dev)
        xb = bb.gather(1, f[pr, None]).squeeze(1).long()
        go_left = xb <= b[pr]
        newn = torch.where(go_left, cl_t[pr], cr_t[pr])
        node_of_row[arows] = torch.where(do_split[pr], newn, node_of_row[arows])

        # Build the next level's histograms: scatter only the smaller child of
        # each split; the larger sibling is parent - smaller. next_active is laid
        # out as [left0, right0, left1, right1, ...], one pair per split in
        # ascending position order, matching split_pos / the CPU loop above.
        if next_active:
            split_pos = do_split.nonzero(as_tuple=False).squeeze(1)  # (P,) ascending
            P = split_pos.numel()
            pos_to_pair = torch.full((M,), -1, dtype=torch.long, device=dev)
            pos_to_pair[split_pos] = torch.arange(P, device=dev)
            sil_pair = small_is_left[split_pos]                      # (P,)
            hist_parent = hist[:, split_pos]                         # (3, P, F, B)

            pair_row = pos_to_pair[pr]
            participates = pair_row >= 0
            sil_row = sil_pair[pair_row.clamp(min=0)]
            is_small = (participates & (go_left == sil_row)).nonzero(as_tuple=True)[0]
            hist_small = scatter_hist(pair_row[is_small], bb[is_small],
                                      gg[is_small], hh[is_small], P)
            hist_other = hist_parent - hist_small
            sil = sil_pair.view(1, P, 1, 1)
            left_hist = torch.where(sil, hist_small, hist_other)
            right_hist = torch.where(sil, hist_other, hist_small)
            hist = torch.empty(3, 2 * P, F, B, dtype=torch.float32, device=dev)
            hist[:, 0::2] = left_hist
            hist[:, 1::2] = right_hist

            if imat is not None:
                # Both children inherit the parent's path plus its split feature.
                child_pm = path_mask[split_pos].clone()
                child_pm[torch.arange(P, device=dev), f[split_pos]] = True
                path_mask = torch.empty(2 * P, F, dtype=torch.bool, device=dev)
                path_mask[0::2] = child_pm
                path_mask[1::2] = child_pm

        active = next_active
        depth = d1

    return Tree(
        feature=torch.tensor(feature, dtype=torch.long, device=dev),
        threshold=torch.tensor(threshold, dtype=torch.float32, device=dev),
        left=torch.tensor(left, dtype=torch.long, device=dev),
        right=torch.tensor(right, dtype=torch.long, device=dev),
        value=torch.tensor(value, dtype=torch.float32, device=dev),
        depth=max(node_depth) + 1,
        gate_scale=torch.tensor(gate, dtype=torch.float32, device=dev),
    )

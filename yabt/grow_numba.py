"""Numba-JIT leaf-wise grower: a compiled CPU replacement for the torch heap path.

Replicates the axis-split growth of :func:`yabt.tree.grow_tree` with identical
split math -- Newton gain, sibling-subtraction histograms, global best-first leaf
selection, and interaction-aware selection steering -- but as one compiled kernel
of tight scalar loops, so it pays no per-op torch dispatch/allocation overhead.
The CPU grower was dispatch-bound, not FLOP-bound: this is 1.5-4x faster than the
torch heap grower at identical accuracy (benchmarks/ab_grow_numba.py).

Kernel (RBF) splits and soft-routing gate scales beyond the split feature's own
scale are not modelled here; Booster keeps the torch grower for kernel splits.
"""

from __future__ import annotations

import numpy as np
import torch
from numba import njit

from .binning import MAX_BINS
from .tree import Tree, TreeParams, LEAF

_NEG_INF = -np.inf


@njit(cache=True, fastmath=True)
def _build_hist(binned, grad, hess, rows, start, end, out):
    """Accumulate (3, F, B) histogram for rows[start:end] into ``out`` (zeroed)."""
    F = binned.shape[1]
    out[:] = 0.0
    for i in range(start, end):
        r = rows[i]
        g = grad[r]
        h = hess[r]
        for f in range(F):
            b = binned[r, f]
            out[0, f, b] += g
            out[1, f, b] += h
            out[2, f, b] += 1.0


@njit(cache=True, fastmath=True)
def _compute_boost(path_row, imat, ib, out):
    """Per-feature selection boost: out[j] = 1 + ib * max over path features p of
    imat[p, j] (1 when the path is empty). Mirrors grow_tree's interaction boost."""
    F = out.shape[0]
    for j in range(F):
        mx = 0.0
        for p in range(F):
            if path_row[p]:
                v = imat[p, j]
                if v > mx:
                    mx = v
        out[j] = 1.0 + ib * mx


@njit(cache=True, fastmath=True)
def _best_split(hist, lam, gamma, mcw, msl, fmask, boost):
    """Best (true_gain, f, b) for one node; true_gain <= 0 means don't split.

    Selection runs over ``gain * boost[f]`` (interaction steering), but the
    returned gain is the true unboosted gain at the selected position, so
    acceptance and best-first ordering are never inflated -- exactly as
    histogram.find_best_split does. "bin <= b" sends bins [0,b] left, the last
    bin is never valid, ties resolve to the first (feature-major) max.
    """
    F = hist.shape[1]
    B = hist.shape[2]
    best_sel = _NEG_INF
    best_true = _NEG_INF
    bf = -1
    bb = -1
    for f in range(F):
        if not fmask[f]:
            continue
        bo = boost[f]
        Gt = 0.0
        Ht = 0.0
        Ct = 0.0
        for b in range(B):
            Gt += hist[0, f, b]
            Ht += hist[1, f, b]
            Ct += hist[2, f, b]
        parent = Gt * Gt / (Ht + lam)
        GL = 0.0
        HL = 0.0
        CL = 0.0
        for b in range(B - 1):  # last bin: empty right child, never valid
            GL += hist[0, f, b]
            HL += hist[1, f, b]
            CL += hist[2, f, b]
            GR = Gt - GL
            HR = Ht - HL
            CR = Ct - CL
            if CL >= msl and CR >= msl and HL >= mcw and HR >= mcw:
                gain = 0.5 * (GL * GL / (HL + lam) + GR * GR / (HR + lam) - parent) - gamma
                sel = gain * bo
                if sel > best_sel:
                    best_sel = sel
                    best_true = gain
                    bf = f
                    bb = b
    return best_true, bf, bb


@njit(cache=True, fastmath=True)
def _partition(binned, rows, start, end, f, b):
    """In-place partition rows[start:end]: ``binned[:,f] <= b`` to the front.
    Returns mid; [start,mid) is left, [mid,end) is right."""
    i = start
    j = end - 1
    while i <= j:
        r = rows[i]
        if binned[r, f] <= b:
            i += 1
        else:
            rows[i] = rows[j]
            rows[j] = r
            j -= 1
    return i


@njit(cache=True, fastmath=True)
def _grow(binned, grad, hess, fmask, imat, ib, use_imat,
          lam, gamma, mcw, msl, lr, max_leaves, max_depth):
    n, F = binned.shape
    B = MAX_BINS
    max_nodes = 2 * max_leaves + 1

    feature = np.full(max_nodes, LEAF, dtype=np.int64)
    thr_bin = np.zeros(max_nodes, dtype=np.int64)
    left = np.full(max_nodes, -1, dtype=np.int64)
    right = np.full(max_nodes, -1, dtype=np.int64)
    value = np.zeros(max_nodes, dtype=np.float32)
    node_depth = np.zeros(max_nodes, dtype=np.int64)
    node_start = np.zeros(max_nodes, dtype=np.int64)
    node_end = np.zeros(max_nodes, dtype=np.int64)
    hist_store = np.zeros((max_nodes, 3, F, B), dtype=np.float32)
    path_mask = np.zeros((max_nodes, F), dtype=np.bool_)

    rows = np.arange(n).astype(np.int64)
    boost = np.ones(F, dtype=np.float32)

    # Active-leaf candidate table (parallel arrays, compacted by index k).
    leaf_node = np.zeros(max_nodes, dtype=np.int64)
    leaf_gain = np.zeros(max_nodes, dtype=np.float32)
    leaf_f = np.zeros(max_nodes, dtype=np.int64)
    leaf_b = np.zeros(max_nodes, dtype=np.int64)

    # Root.
    node_start[0] = 0
    node_end[0] = n
    node_depth[0] = 0
    _build_hist(binned, grad, hess, rows, 0, n, hist_store[0])
    Gsum = 0.0
    Hsum = 0.0
    for b in range(B):
        Gsum += hist_store[0, 0, 0, b]
        Hsum += hist_store[0, 1, 0, b]
    value[0] = -lr * Gsum / (Hsum + lam)
    n_nodes = 1

    g0, f0, b0 = _best_split(hist_store[0], lam, gamma, mcw, msl, fmask, boost)
    if node_depth[0] >= max_depth:
        g0 = _NEG_INF
    leaf_node[0] = 0
    leaf_gain[0] = g0
    leaf_f[0] = f0
    leaf_b[0] = b0
    n_active = 1
    n_leaves = 1

    while n_leaves < max_leaves:
        # Best-first: pick the active leaf with the largest positive true gain.
        best_k = -1
        best_g = 0.0
        for k in range(n_active):
            if leaf_gain[k] > best_g:
                best_g = leaf_gain[k]
                best_k = k
        if best_k < 0:
            break

        nid = leaf_node[best_k]
        f = leaf_f[best_k]
        b = leaf_b[best_k]
        s = node_start[nid]
        e = node_end[nid]
        mid = _partition(binned, rows, s, e, f, b)

        nl = n_nodes
        nr = n_nodes + 1
        n_nodes += 2
        d = node_depth[nid] + 1
        node_start[nl] = s
        node_end[nl] = mid
        node_start[nr] = mid
        node_end[nr] = e
        node_depth[nl] = d
        node_depth[nr] = d
        # Both children inherit the parent path plus the split feature.
        for p in range(F):
            path_mask[nl, p] = path_mask[nid, p]
            path_mask[nr, p] = path_mask[nid, p]
        path_mask[nl, f] = True
        path_mask[nr, f] = True

        # Smaller child by scatter; sibling by subtraction from the parent hist.
        if (mid - s) <= (e - mid):
            _build_hist(binned, grad, hess, rows, s, mid, hist_store[nl])
            hist_store[nr] = hist_store[nid] - hist_store[nl]
        else:
            _build_hist(binned, grad, hess, rows, mid, e, hist_store[nr])
            hist_store[nl] = hist_store[nid] - hist_store[nr]

        gl = 0.0
        hl = 0.0
        gr = 0.0
        hr = 0.0
        for bb in range(B):
            gl += hist_store[nl, 0, 0, bb]
            hl += hist_store[nl, 1, 0, bb]
            gr += hist_store[nr, 0, 0, bb]
            hr += hist_store[nr, 1, 0, bb]
        value[nl] = -lr * gl / (hl + lam)
        value[nr] = -lr * gr / (hr + lam)

        feature[nid] = f
        thr_bin[nid] = b
        left[nid] = nl
        right[nid] = nr
        value[nid] = 0.0
        n_leaves += 1

        # Replace the split leaf with its left child; append the right child.
        if use_imat:
            _compute_boost(path_mask[nl], imat, ib, boost)
        glg, glf, glb = _best_split(hist_store[nl], lam, gamma, mcw, msl, fmask, boost)
        if d >= max_depth:
            glg = _NEG_INF
        leaf_node[best_k] = nl
        leaf_gain[best_k] = glg
        leaf_f[best_k] = glf
        leaf_b[best_k] = glb

        if use_imat:
            _compute_boost(path_mask[nr], imat, ib, boost)
        grg, grf, grb = _best_split(hist_store[nr], lam, gamma, mcw, msl, fmask, boost)
        if d >= max_depth:
            grg = _NEG_INF
        leaf_node[n_active] = nr
        leaf_gain[n_active] = grg
        leaf_f[n_active] = grf
        leaf_b[n_active] = grb
        n_active += 1

    return feature[:n_nodes], thr_bin[:n_nodes], left[:n_nodes], right[:n_nodes], \
        value[:n_nodes], node_depth[:n_nodes]


def grow_tree_numba(
    binned: torch.Tensor,
    grad: torch.Tensor,
    hess: torch.Tensor,
    binner,
    params: TreeParams,
    feature_mask: torch.Tensor | None = None,
    interaction_matrix: torch.Tensor | None = None,
    interaction_boost: float = 0.5,
) -> Tree:
    """Drop-in for the axis path of :func:`yabt.tree.grow_tree` (no kernel splits).

    Honors ``feature_mask`` and interaction steering; returns the same Tree
    output including per-split gate scales."""
    dev = binned.device
    n, F = binned.shape
    bn = np.ascontiguousarray(binned.detach().cpu().numpy())
    gn = grad.detach().cpu().numpy().astype(np.float32)
    hn = hess.detach().cpu().numpy().astype(np.float32)
    if feature_mask is None:
        fmask = np.ones(F, dtype=np.bool_)
    else:
        fmask = feature_mask.detach().cpu().numpy().astype(np.bool_)
    use_imat = interaction_matrix is not None
    if use_imat:
        imat = np.ascontiguousarray(interaction_matrix.detach().cpu().numpy().astype(np.float32))
    else:
        imat = np.zeros((F, F), dtype=np.float32)

    feat, thr_bin, left, right, value, depth = _grow(
        bn, gn, hn, fmask, imat, float(interaction_boost), use_imat,
        float(params.reg_lambda), float(params.gamma),
        float(params.min_child_weight), int(params.min_samples_leaf),
        float(params.learning_rate), int(params.max_leaves), int(params.max_depth),
    )

    scales = binner.scales_.clamp_min(1e-12)
    threshold = np.zeros(len(feat), dtype=np.float32)
    gate = np.ones(len(feat), dtype=np.float32)
    for nid in range(len(feat)):
        f = int(feat[nid])
        if f != LEAF:
            threshold[nid] = binner.edge_value(f, int(thr_bin[nid]))
            gate[nid] = float(scales[f])

    return Tree(
        feature=torch.from_numpy(feat).to(dev),
        threshold=torch.from_numpy(threshold).to(dev),
        left=torch.from_numpy(left).to(dev),
        right=torch.from_numpy(right).to(dev),
        value=torch.from_numpy(value).to(dev),
        depth=int(depth.max()) + 1,
        gate_scale=torch.from_numpy(gate).to(dev),
    )

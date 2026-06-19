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
def _build_hist_sparse(indptr, indices, data, default_bin, grad, hess,
                       rows, start, end, out, expl_g, expl_h, expl_c):
    """Sparse (CSC-of-nonzeros) histogram for rows[start:end] into ``out``.

    Each feature has a *default bin* (its most common value, typically the
    binned zero of a sparse column). The CSR arrays ``indptr/indices/data`` store
    only the entries whose bin differs from that default. We scatter those
    explicit entries and accumulate per-feature explicit (g,h,count) sums, then
    fill every feature's default bin with the node total minus its explicit sum.
    Cost is O(node_nnz + F) instead of the dense O(node_rows * F), which is the
    whole win on wide, sparse data. Produces the same (3,F,B) histogram the dense
    builder does, so sibling subtraction and split search are unchanged.
    """
    F = out.shape[1]
    out[:] = 0.0
    expl_g[:] = 0.0
    expl_h[:] = 0.0
    expl_c[:] = 0.0
    G = 0.0
    H = 0.0
    C = 0.0
    for i in range(start, end):
        r = rows[i]
        g = grad[r]
        h = hess[r]
        G += g
        H += h
        C += 1.0
        for idx in range(indptr[r], indptr[r + 1]):
            f = indices[idx]
            b = data[idx]
            out[0, f, b] += g
            out[1, f, b] += h
            out[2, f, b] += 1.0
            expl_g[f] += g
            expl_h[f] += h
            expl_c[f] += 1.0
    # Default bin gets the node total minus what the explicit entries carried;
    # no explicit entry lands here (those are excluded when the layout is built),
    # so this is a write, not a read-modify of real data.
    for f in range(F):
        df = default_bin[f]
        out[0, f, df] += G - expl_g[f]
        out[1, f, df] += H - expl_h[f]
        out[2, f, df] += C - expl_c[f]


@njit(cache=True, fastmath=True)
def _compute_boost(path_row, imat, ib, out):
    """Per-feature selection boost: out[j] = 1 + ib * max over path features p of
    imat[p, j] (1 when the path is empty). Mirrors grow_tree's interaction boost.

    Iterates over the few features actually on the node's path rather than the
    full F x F matrix: the path has at most depth (~log) features, so this is
    O(F * path_len) instead of O(F^2). On wide data (F in the thousands) the old
    dense double loop dominated the whole fit; this is bit-identical (same running
    max, same 0.0 floor) but ~1000x cheaper. The inner sweep over j walks imat
    row-contiguously, so it is also cache-friendly."""
    F = out.shape[0]
    for j in range(F):
        out[j] = 0.0  # running max over path features (clamped at 0, as before)
    for p in range(F):
        if path_row[p]:
            for j in range(F):
                v = imat[p, j]
                if v > out[j]:
                    out[j] = v
    for j in range(F):
        out[j] = 1.0 + ib * out[j]


@njit(cache=True, fastmath=True)
def _best_split(hist, nbins, lam, gamma, mcw, msl, fmask, boost):
    """Best (true_gain, f, b) for one node; true_gain <= 0 means don't split.

    Selection runs over ``gain * boost[f]`` (interaction steering), but the
    returned gain is the true unboosted gain at the selected position, so
    acceptance and best-first ordering are never inflated -- exactly as
    histogram.find_best_split does. "bin <= b" sends bins [0,b] left, the last
    bin is never valid, ties resolve to the first (feature-major) max.

    ``nbins[f]`` is feature f's used bin count (#edges + 1). Bins at or above it
    are always empty (searchsorted can't produce them), so scanning only [0,
    nbins[f]) is exact and skips the all-zero tail -- a large saving on
    low-cardinality features (Santander's median feature uses ~35 of 256 bins).
    """
    F = hist.shape[1]
    best_sel = _NEG_INF
    best_true = _NEG_INF
    bf = -1
    bb = -1
    for f in range(F):
        if not fmask[f]:
            continue
        nb = nbins[f]
        bo = boost[f]
        Gt = 0.0
        Ht = 0.0
        Ct = 0.0
        for b in range(nb):
            Gt += hist[0, f, b]
            Ht += hist[1, f, b]
            Ct += hist[2, f, b]
        parent = Gt * Gt / (Ht + lam)
        GL = 0.0
        HL = 0.0
        CL = 0.0
        for b in range(nb - 1):  # last used bin: empty right child, never valid
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
          indptr, indices, data, default_bin, use_sparse, nbins,
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
    # Per-feature explicit-sum scratch for the sparse histogram builder.
    expl_g = np.zeros(F, dtype=np.float32)
    expl_h = np.zeros(F, dtype=np.float32)
    expl_c = np.zeros(F, dtype=np.float32)

    # Active-leaf candidate table (parallel arrays, compacted by index k).
    leaf_node = np.zeros(max_nodes, dtype=np.int64)
    leaf_gain = np.zeros(max_nodes, dtype=np.float32)
    leaf_f = np.zeros(max_nodes, dtype=np.int64)
    leaf_b = np.zeros(max_nodes, dtype=np.int64)

    # Root.
    node_start[0] = 0
    node_end[0] = n
    node_depth[0] = 0
    if use_sparse:
        _build_hist_sparse(indptr, indices, data, default_bin, grad, hess,
                           rows, 0, n, hist_store[0], expl_g, expl_h, expl_c)
    else:
        _build_hist(binned, grad, hess, rows, 0, n, hist_store[0])
    Gsum = 0.0
    Hsum = 0.0
    for b in range(B):
        Gsum += hist_store[0, 0, 0, b]
        Hsum += hist_store[0, 1, 0, b]
    value[0] = -lr * Gsum / (Hsum + lam)
    n_nodes = 1

    g0, f0, b0 = _best_split(hist_store[0], nbins, lam, gamma, mcw, msl, fmask, boost)
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
        # Slice assignment compiles to a memcpy of the (F,) bool row, avoiding the
        # per-feature scalar loop (a real cost on wide F).
        path_mask[nl] = path_mask[nid]
        path_mask[nr] = path_mask[nid]
        path_mask[nl, f] = True
        path_mask[nr, f] = True

        # Smaller child by scatter; sibling by subtraction from the parent hist.
        if (mid - s) <= (e - mid):
            if use_sparse:
                _build_hist_sparse(indptr, indices, data, default_bin, grad, hess,
                                   rows, s, mid, hist_store[nl], expl_g, expl_h, expl_c)
            else:
                _build_hist(binned, grad, hess, rows, s, mid, hist_store[nl])
            hist_store[nr] = hist_store[nid] - hist_store[nl]
        else:
            if use_sparse:
                _build_hist_sparse(indptr, indices, data, default_bin, grad, hess,
                                   rows, mid, e, hist_store[nr], expl_g, expl_h, expl_c)
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
        glg, glf, glb = _best_split(hist_store[nl], nbins, lam, gamma, mcw, msl, fmask, boost)
        if d >= max_depth:
            glg = _NEG_INF
        leaf_node[best_k] = nl
        leaf_gain[best_k] = glg
        leaf_f[best_k] = glf
        leaf_b[best_k] = glb

        if use_imat:
            _compute_boost(path_mask[nr], imat, ib, boost)
        grg, grf, grb = _best_split(hist_store[nr], nbins, lam, gamma, mcw, msl, fmask, boost)
        if d >= max_depth:
            grg = _NEG_INF
        leaf_node[n_active] = nr
        leaf_gain[n_active] = grg
        leaf_f[n_active] = grf
        leaf_b[n_active] = grb
        n_active += 1

    return feature[:n_nodes], thr_bin[:n_nodes], left[:n_nodes], right[:n_nodes], \
        value[:n_nodes], node_depth[:n_nodes]


def build_sparse_layout(binned: torch.Tensor):
    """Build the CSR-of-nonzeros layout used by the sparse histogram builder.

    Returns ``(indptr, indices, data, default_bin)``:
      * ``default_bin[f]`` is feature f's most common bin (the implicit value),
      * ``indptr/indices/data`` are a row-major CSR over the entries whose bin
        differs from that default -- so iterating a row visits only its non-default
        features.

    Computed once per fit from the (fixed) full binned matrix and reused across
    boosting rounds. On dense data the dominant bin covers few rows, so nnz ~= n*F
    and there is no win; callers gate on the achieved density (see ``layout_density``).
    """
    bn = np.ascontiguousarray(binned.detach().cpu().numpy())
    n, F = bn.shape
    # Per-feature most-common bin via a single (F, MAX_BINS) count.
    counts = np.zeros((F, MAX_BINS), dtype=np.int64)
    np.add.at(counts, (np.arange(F)[None, :].repeat(n, 0).ravel(), bn.ravel()), 1)
    default_bin = counts.argmax(axis=1).astype(np.int64)

    nondef = bn != default_bin[None, :].astype(bn.dtype)
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(nondef.sum(axis=1), out=indptr[1:])
    rows_idx, cols_idx = np.nonzero(nondef)  # C-order => grouped by row (CSR)
    indices = cols_idx.astype(np.int64)
    data = bn[rows_idx, cols_idx].astype(np.int64)
    return indptr, indices, data, default_bin


def layout_density(layout, n: int, F: int) -> float:
    """Fraction of cells stored explicitly (nnz / (n*F)); lower = more savings."""
    return float(len(layout[1]) / max(1, n * F))


def grow_tree_numba(
    binned: torch.Tensor,
    grad: torch.Tensor,
    hess: torch.Tensor,
    binner,
    params: TreeParams,
    feature_mask: torch.Tensor | None = None,
    interaction_matrix: torch.Tensor | None = None,
    interaction_boost: float = 0.5,
    sparse_layout=None,
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

    if sparse_layout is not None:
        indptr, indices, data, default_bin = sparse_layout
        use_sparse = True
    else:
        indptr = np.zeros(1, dtype=np.int64)
        indices = np.zeros(1, dtype=np.int64)
        data = np.zeros(1, dtype=np.int64)
        default_bin = np.zeros(1, dtype=np.int64)
        use_sparse = False

    # Used bin count per feature: bin(x) = #{edges < x} in [0, len(edges)], so a
    # feature uses at most len(edges)+1 bins; the split search skips the empty tail.
    nbins = np.fromiter(
        (min(len(e) + 1, MAX_BINS) for e in binner.edges_), dtype=np.int64, count=F
    )

    feat, thr_bin, left, right, value, depth = _grow(
        bn, gn, hn, fmask, imat, float(interaction_boost), use_imat,
        indptr, indices, data, default_bin, use_sparse, nbins,
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

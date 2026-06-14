"""Neural leaf networks: replace constant leaf values with small per-leaf models.

After a tree is grown and its constants refined, each sufficiently populated
leaf gets a small model over the tree's own split features, fitted to the
Newton objective at the current ensemble margin (so it captures within-leaf
residual structure the constant cannot). ``leaf_net_hidden=0`` fits a
closed-form ridge-linear model per leaf; ``>0`` trains a tiny tanh MLP per
leaf with a few Adam steps. Outputs are shrunk by the boosting learning rate
and stored in raw feature space, so inference needs no normalization state.
"""

from __future__ import annotations

from dataclasses import replace

import torch

from .tree import LEAF, Tree


def _select_features(tree: Tree, Xraw: torch.Tensor, grad: torch.Tensor, k: int) -> torch.Tensor:
    """Up to k net input features: the tree's own split features first (most
    used first), padded with the strongest remaining features by standardized
    gradient correlation |corr(x_f, g)|, since the tree may split on a step
    feature while the within-leaf structure lives on others."""
    k = min(k, Xraw.shape[1])
    f = tree.feature[tree.feature >= 0]
    if f.numel() > 0:
        vals, counts = f.unique(return_counts=True)
        used = vals[counts.argsort(descending=True)][:k]
    else:
        used = torch.empty(0, dtype=torch.long, device=Xraw.device)
    if used.numel() >= k:
        return used
    score = (Xraw.T @ grad - Xraw.mean(dim=0) * grad.sum()).abs()
    score = score / Xraw.std(dim=0).clamp_min(1e-6)
    score[used] = float("-inf")
    extra = score.argsort(descending=True)[: k - used.numel()]
    return torch.cat([used, extra])


def _fit_linear(
    Xn: torch.Tensor,        # (n, K) normalized leaf-net inputs
    grad: torch.Tensor,
    hess: torch.Tensor,
    leaf_idx: torch.Tensor,
    eligible: torch.Tensor,  # (num_nodes,) bool
    n_nodes: int,
    lam: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-leaf ridge solve of the Newton objective; returns weights and bias.

    The per-leaf normal equations all share the same (K+1, K+1) shape, so they
    are assembled with scatter (index_add) and solved in a single batched
    linalg.solve instead of a Python loop of tiny per-leaf solves -- the latter
    is dominated by op-launch overhead, especially on GPU. The accumulation
    order differs from a per-leaf matmul, so results match the old path to
    double-precision tolerance rather than bitwise."""
    dev = Xn.device
    K = Xn.shape[1]
    W = torch.zeros(n_nodes, K, device=dev)
    db = torch.zeros(n_nodes, device=dev)
    elig = torch.nonzero(eligible).squeeze(1)
    if elig.numel() == 0:
        return W, db

    n = Xn.shape[0]
    ones = torch.ones(n, 1, dtype=torch.float64, device=dev)
    X1 = torch.cat([Xn.double(), ones], dim=1)        # (n, K+1)
    g, h = grad.double(), hess.double()
    Xh = X1 * h.unsqueeze(1)
    rhs_rows = -(X1 * g.unsqueeze(1))

    # M[l] = sum_{i in leaf l} h_i x1_i x1_i^T ; b[l] = -sum g_i x1_i.
    # Chunk the per-row outer products so peak memory stays bounded for large n.
    M = torch.zeros(n_nodes, K + 1, K + 1, dtype=torch.float64, device=dev)
    bvec = torch.zeros(n_nodes, K + 1, dtype=torch.float64, device=dev)
    CH = 65536
    for s in range(0, n, CH):
        sl = slice(s, s + CH)
        idx = leaf_idx[sl]
        M.index_add_(0, idx, X1[sl].unsqueeze(2) * Xh[sl].unsqueeze(1))
        bvec.index_add_(0, idx, rhs_rows[sl])

    reg = torch.eye(K + 1, dtype=torch.float64, device=dev) * lam
    reg[K, K] = 1e-8  # bias is not penalized
    theta = torch.linalg.solve(M[elig] + reg, bvec[elig]).float()  # (E, K+1)
    W[elig] = theta[:, :K]
    db[elig] = theta[:, K]
    return W, db


def _fit_mlp(
    Xn: torch.Tensor,
    grad: torch.Tensor,
    hess: torch.Tensor,
    leaf_idx: torch.Tensor,
    eligible: torch.Tensor,
    n_nodes: int,
    params,
    gen: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """Jointly train one tiny tanh MLP per eligible leaf on the Newton objective."""
    dev = Xn.device
    K, H = Xn.shape[1], params.leaf_net_hidden
    row_ok = eligible[leaf_idx]
    Xt, gt, ht, lt = Xn[row_ok], grad[row_ok], hess[row_ok], leaf_idx[row_ok]
    if Xt.shape[0] == 0:
        return None

    W1 = (0.5 * torch.randn(n_nodes, H, K, generator=gen)).to(dev).requires_grad_(True)
    b1 = torch.zeros(n_nodes, H, device=dev, requires_grad=True)
    # Near-zero output init: nets start as a no-op and grow into the residuals.
    w2 = (0.01 * torch.randn(n_nodes, H, generator=gen)).to(dev).requires_grad_(True)
    opt = torch.optim.Adam([W1, b1, w2], lr=params.leaf_net_lr)
    for _ in range(params.leaf_net_steps):
        opt.zero_grad()
        hid = torch.tanh(torch.einsum("nhk,nk->nh", W1[lt], Xt) + b1[lt])
        f = (w2[lt] * hid).sum(dim=1)
        loss = (gt * f + 0.5 * ht * f.square()).mean() + params.leaf_net_l2 * (
            W1.square().mean() + w2.square().mean()
        )
        loss.backward()
        opt.step()

    W1, b1, w2 = W1.detach(), b1.detach(), w2.detach()
    w2[~eligible] = 0.0  # ineligible leaves and internal nodes stay constant
    return W1, b1, w2


def fit_leaf_networks(
    tree: Tree,
    Xraw: torch.Tensor,    # (n, F)
    y: torch.Tensor,
    margin: torch.Tensor,  # (n,) ensemble margin EXCLUDING this tree
    loss_fn,
    params,
    gen: torch.Generator,
    leaf_idx: torch.Tensor | None = None,
) -> Tree:
    dev = Xraw.device
    # Adding leaf nets does not change routing, so callers that already routed
    # Xraw (e.g. the boosting margin update) can pass leaf_idx to avoid a second
    # full traversal.
    if leaf_idx is None:
        leaf_idx = tree.apply(Xraw)
    grad, hess = loss_fn.grad_hess(margin + tree.value[leaf_idx], y)
    feats = _select_features(tree, Xraw, grad, params.leaf_net_features)
    if feats.numel() == 0:
        return tree

    Xs = Xraw[:, feats]
    mu = Xs.mean(dim=0)
    sd = Xs.std(dim=0).clamp_min(1e-6)
    Xn = (Xs - mu) / sd

    n_nodes = tree.value.shape[0]
    counts = torch.zeros(n_nodes, device=dev).scatter_add_(0, leaf_idx, torch.ones_like(grad))
    eligible = (tree.feature == LEAF) & (counts >= params.leaf_net_min_samples)
    if not bool(eligible.any()):
        return tree
    lr = params.learning_rate

    # Per-leaf input envelope (raw space): at inference the leaf's net inputs
    # are clamped to the range it was fit on, so linear/MLP extrapolation on
    # out-of-range rows (leaf regions are unbounded) cannot explode.
    K = feats.numel()
    # Per-leaf min/max over each net input, vectorized via scatter_reduce (one
    # pass instead of a full-n boolean mask per leaf). Leaves with no rows keep
    # the +-inf init; only eligible leaves are consulted at inference, and
    # ineligible leaves carry zero net weight, so their envelope is irrelevant.
    idx = leaf_idx.unsqueeze(1).expand(-1, K)
    lo = torch.full((n_nodes, K), float("inf"), device=dev)
    hi = torch.full((n_nodes, K), float("-inf"), device=dev)
    lo.scatter_reduce_(0, idx, Xs, reduce="amin", include_self=True)
    hi.scatter_reduce_(0, idx, Xs, reduce="amax", include_self=True)

    if params.leaf_net_hidden == 0:
        W, db = _fit_linear(Xn, grad, hess, leaf_idx, eligible, n_nodes, params.leaf_net_l2)
        # Fold normalization and shrinkage into raw space:
        # lr * (w . (x - mu) / sd + b) == (lr * w / sd) . x + lr * (b - w . (mu / sd))
        value = tree.value + lr * (db - (W * (mu / sd)).sum(dim=1))
        return replace(tree, value=value, leaf_net_feats=feats, leaf_net_linear=lr * W / sd,
                       leaf_net_lo=lo, leaf_net_hi=hi)

    mlp = _fit_mlp(Xn, grad, hess, leaf_idx, eligible, n_nodes, params, gen)
    if mlp is None:
        return tree
    W1, b1, w2 = mlp
    return replace(
        tree,
        leaf_net_feats=feats,
        leaf_net_W1=W1 / sd,
        leaf_net_b1=b1 - torch.einsum("nhk,k->nh", W1, mu / sd),
        leaf_net_W2=lr * w2,
        leaf_net_lo=lo,
        leaf_net_hi=hi,
    )

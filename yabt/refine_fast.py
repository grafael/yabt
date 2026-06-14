"""Fast differentiable tree refinement with manual gradients and Numba JIT."""

from __future__ import annotations

from dataclasses import replace

import torch
import numpy as np
from numba import jit

from .tree import Tree


@jit(nopython=True)
def compute_leaf_stats(
    leaf_idx: np.ndarray,   # (n,) leaf index per sample
    grad: np.ndarray,       # (n,) sample gradients
    hess: np.ndarray,       # (n,) sample hessians
    n_leaves: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute aggregated gradients and hessians per leaf (Numba-JIT)."""
    leaf_grad = np.zeros(n_leaves, dtype=np.float32)
    leaf_hess = np.zeros(n_leaves, dtype=np.float32)

    for i in range(len(leaf_idx)):
        leaf = leaf_idx[i]
        if 0 <= leaf < n_leaves:
            leaf_grad[leaf] += grad[i]
            leaf_hess[leaf] += hess[i]

    return leaf_grad, leaf_hess


@jit(nopython=True)
def apply_leaf_updates(
    leaf_idx: np.ndarray,      # (n,) sample leaf assignments
    leaf_updates: np.ndarray,  # (n_leaves,) delta values per leaf
    margin: np.ndarray,        # (n,) inplace update
) -> np.ndarray:
    """Apply leaf value updates to margin (Numba-JIT)."""
    for i in range(len(leaf_idx)):
        leaf = leaf_idx[i]
        if 0 <= leaf < len(leaf_updates):
            margin[i] += leaf_updates[leaf]
    return margin


def refine_tree_fast(
    tree: Tree,
    Xraw: torch.Tensor,      # (n, F) raw features
    y: torch.Tensor,          # (n,)
    margin: torch.Tensor,     # (n,) current predictions
    loss_fn,
    params,
) -> Tree:
    """Refine leaf values with manual Newton steps under hard routing, with the
    per-leaf gradient aggregation and margin updates JIT-compiled by Numba."""
    device = Xraw.device

    if params.refine_steps == 0:
        return tree

    # Skip when the ensemble already fits well; refinement can only add noise.
    if params.refine_min_gain > 0 and float(loss_fn.loss(margin, y)) < 0.001:
        return tree

    lam = params.reg_lambda
    lr = params.refine_lr

    # Compute leaf assignments using hard routing (very fast)
    leaf_idx = tree.apply(Xraw).cpu().numpy().astype(np.int32)
    n_nodes = len(tree.value)

    # Convert to numpy for vectorized operations
    y_np = y.detach().cpu().numpy().astype(np.float32)
    margin_np = margin.detach().cpu().numpy().astype(np.float32).copy()
    value_np = tree.value.detach().cpu().numpy().astype(np.float32).copy()

    # Refinement gradients must be taken at the ensemble margin INCLUDING this
    # tree's own contribution; `margin` is the pre-tree margin.
    margin_np = margin_np + value_np[leaf_idx]

    # Optimization loop
    for step in range(params.refine_steps):
        # Compute gradients of loss w.r.t. margin
        if loss_fn.is_classification:
            margin_clipped = np.clip(margin_np, -500, 500)
            p = 1.0 / (1.0 + np.exp(-margin_clipped))
            grad_np = (p - y_np).astype(np.float32)
            hess_np = (p * (1.0 - p)).clip(1e-6).astype(np.float32)
        else:
            grad_np = (margin_np - y_np).astype(np.float32)
            hess_np = np.ones_like(y_np, dtype=np.float32)

        leaf_grad, leaf_hess = compute_leaf_stats(leaf_idx, grad_np, hess_np, n_nodes)

        # Newton step: delta_v = -lr * grad / (hess + lambda)
        leaf_updates = np.zeros(n_nodes, dtype=np.float32)
        for node in range(n_nodes):
            if leaf_hess[node] > 1e-6:
                leaf_updates[node] = -lr * leaf_grad[node] / (leaf_hess[node] + lam)

        # Update leaf values
        value_np = value_np + leaf_updates

        # Apply updates to margin
        margin_np = apply_leaf_updates(leaf_idx, leaf_updates, margin_np)

    # Convert back to torch
    value = torch.from_numpy(value_np).to(device)

    return replace(tree, value=value)


def global_leaf_refit_fast(
    trees: list[Tree],
    Xraw: torch.Tensor,      # (n, F) raw features
    y: torch.Tensor,          # (n,)
    base_score: float,
    loss_fn,
    params,
) -> torch.Tensor:
    """Jointly refit leaf values of all trees against the global loss.

    Newton coordinate descent: sweep the trees ``refit_steps`` times; for each
    tree, aggregate grad/hess per leaf at the current ensemble margin and apply
    a damped Newton update to its leaf values (in place). Leaf assignments are
    cached once, so each sweep is just scatter-adds, fast on GPU and CPU.
    """
    device = Xraw.device
    n = Xraw.shape[0]
    lam = params.reg_lambda

    leaf_idx = [t.apply(Xraw) for t in trees]
    margin = torch.full((n,), base_score, device=device)
    for tree, idx in zip(trees, leaf_idx):
        margin = margin + tree.value[idx] + tree.net_contribution(Xraw, idx)

    for _ in range(params.refit_steps):
        for tree, idx in zip(trees, leaf_idx):
            grad, hess = loss_fn.grad_hess(margin, y)
            m = tree.value.shape[0]
            g = torch.zeros(m, device=device).scatter_add_(0, idx, grad)
            h = torch.zeros(m, device=device).scatter_add_(0, idx, hess)
            # Internal nodes receive no samples (h == 0) and stay untouched.
            delta = torch.where(h > 0, -params.refit_lr * g / (h + lam), torch.zeros(m, device=device))
            tree.value += delta
            margin = margin + delta[idx]

    return margin

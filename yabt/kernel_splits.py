"""Kernel-based splits: RBF landmark candidates for non-linear decision boundaries.

A kernel split routes a row by the similarity z = exp(-gamma * ||x - c||^2)
to a landmark point c sampled from the node's own rows, instead of by a single
feature threshold. Rows with z <= threshold go left (far from the landmark),
which carves out spherical regions in scale-normalized feature space.
"""

from __future__ import annotations

import torch

KERNEL_BINS = 64
WEIGHT_FLOOR = 0.05  # no feature's distance weight drops below this fraction of the max


def importance_weights(gains: torch.Tensor) -> torch.Tensor | None:
    """Per-feature distance weights from per-feature split gains, or None for
    uniform. Square-root tempering keeps one dominant feature from collapsing
    the distance to a single axis; the floor keeps weak features from being
    dropped entirely (the gain signal is marginal and misses pure interactions).
    """
    if not float(gains.max()) > 0:
        return None
    w = gains.clamp_min(0).sqrt()
    return (w / w.max()).clamp_min(WEIGHT_FLOOR)


def find_best_kernel_split(
    Xn: torch.Tensor,        # (m, F) scale-normalized rows of one leaf
    grad: torch.Tensor,      # (m,)
    hess: torch.Tensor,      # (m,)
    n_candidates: int,
    gamma: float,            # bandwidth; 0 = per-landmark median heuristic
    lam: float,
    split_penalty: float,
    min_child_weight: float,
    min_samples_leaf: int,
    gen: torch.Generator,
    feature_weights: torch.Tensor | None = None,  # (F,) distance weights; None = uniform
) -> tuple[float, torch.Tensor, float, float, torch.Tensor, float] | None:
    """Best RBF-landmark split for one leaf, or None if no valid positive gain.

    Each sampled landmark induces a 1-d kernel feature that is linearly binned
    and scanned with the same Newton gain formula as axis-aligned splits. With
    ``feature_weights`` the distance is sum_f w_f * (x_f - c_f)^2, so important
    features dominate and noise dimensions stop washing out the signal.

    Returns (gain, center, gamma, threshold, go_left, z_std); center is in the
    same (unweighted) normalized space as Xn, go_left is the boolean row
    routing, and z_std is the spread of the kernel feature (the natural gate
    scale for soft routing).
    """
    m, F = Xn.shape
    dev = Xn.device
    L = min(n_candidates, m)
    idx = torch.randperm(m, generator=gen)[:L].to(dev)
    centers = Xn[idx]                                # (L, F)
    Xw = Xn * feature_weights.sqrt() if feature_weights is not None else Xn
    d2 = torch.cdist(Xw, Xw[idx]).square()           # (m, L)
    if gamma > 0:
        gam = torch.full((L,), gamma, device=dev)
    else:
        gam = 1.0 / (2.0 * d2.median(dim=0).values + 1e-8)
    z = torch.exp(-d2 * gam)                         # (m, L)

    B = KERNEL_BINS
    zmin = z.min(dim=0).values
    width = (z.max(dim=0).values - zmin).clamp_min(1e-12)
    bins = ((z - zmin) / width * (B - 1)).long().clamp_(0, B - 1)  # (m, L)
    flat = (bins + torch.arange(L, device=dev) * B).reshape(-1)

    hist = torch.zeros(3, L * B, dtype=torch.float32, device=dev)
    hist[0].scatter_add_(0, flat, grad.unsqueeze(1).expand(m, L).reshape(-1))
    hist[1].scatter_add_(0, flat, hess.unsqueeze(1).expand(m, L).reshape(-1))
    hist[2].scatter_add_(0, flat, torch.ones(m * L, dtype=torch.float32, device=dev))
    hist = hist.view(3, L, B)

    GL, HL, CL = hist[0].cumsum(1), hist[1].cumsum(1), hist[2].cumsum(1)
    G, H, C = GL[:, -1:], HL[:, -1:], CL[:, -1:]
    GR, HR, CR = G - GL, H - HL, C - CL
    gain = (
        0.5 * (GL.square() / (HL + lam) + GR.square() / (HR + lam) - G.square() / (H + lam))
        - split_penalty
    )
    valid = (
        (CL >= max(1, min_samples_leaf))
        & (CR >= max(1, min_samples_leaf))
        & (HL >= min_child_weight)
        & (HR >= min_child_weight)
    )
    valid[:, -1] = False
    gain = torch.where(valid, gain, torch.tensor(float("-inf"), device=dev))

    best = int(gain.reshape(-1).argmax())
    l, b = divmod(best, B)
    best_gain = float(gain[l, b])
    if not best_gain > 0:  # also rejects -inf / nan
        return None

    go_left = bins[:, l] <= b
    # Place the threshold mid-gap so train/inference routing agrees even with
    # small float differences between the two z computations.
    threshold = float((z[go_left, l].max() + z[~go_left, l].min()) / 2)
    return best_gain, centers[l].clone(), float(gam[l]), threshold, go_left, float(z[:, l].std())

"""GPU histogram construction and best-split search."""

from __future__ import annotations

import torch

from .binning import MAX_BINS


def build_histogram(
    binned: torch.Tensor,  # (n, F) uint8, rows of one leaf
    grad: torch.Tensor,    # (n,) float32
    hess: torch.Tensor,    # (n,) float32
) -> torch.Tensor:
    """Returns hist (3, F, B): per feature-bin sums of grad, hess and counts.

    Accumulation is always FP32: FP16 scatter_add is both slower on modern GPUs
    and inexact (counts above 2048 are unrepresentable, sums overflow at 65504),
    which broke CPU/GPU prediction agreement.
    """
    n, F = binned.shape
    B = MAX_BINS
    dev = binned.device
    flat = (torch.arange(F, device=dev, dtype=torch.long).unsqueeze(0) * B + binned.long()).reshape(-1)

    hist = torch.zeros(3, F * B, dtype=torch.float32, device=dev)
    hist[0].scatter_add_(0, flat, grad.unsqueeze(1).expand(n, F).reshape(-1))
    hist[1].scatter_add_(0, flat, hess.unsqueeze(1).expand(n, F).reshape(-1))
    hist[2].scatter_add_(0, flat, torch.ones(n * F, dtype=torch.float32, device=dev))

    return hist.view(3, F, B)


def _split_gain_matrix(
    hist: torch.Tensor,  # (3, F, B)
    lam: float,
    min_child_weight: float,
    min_samples_leaf: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Newton gain (F, B) for every "bin <= b" split, plus its validity mask."""
    # One fused cumsum over the (grad, hess, count) channels instead of three.
    cum = hist.cumsum(dim=2)
    GL, HL, CL = cum[0], cum[1], cum[2]
    G = GL[:, -1:]
    H = HL[:, -1:]
    C = CL[:, -1:]
    GR, HR, CR = G - GL, H - HL, C - CL

    gain = 0.5 * (GL.square() / (HL + lam) + GR.square() / (HR + lam) - G.square() / (H + lam))
    valid = (
        (CL >= min_samples_leaf)
        & (CR >= min_samples_leaf)
        & (HL >= min_child_weight)
        & (HR >= min_child_weight)
    )
    valid[:, -1] = False
    return gain, valid


def find_best_split(
    hist: torch.Tensor,  # (3, F, B)
    lam: float,
    gamma: float,
    min_child_weight: float,
    min_samples_leaf: int,
    feature_mask: torch.Tensor | None = None,   # (F,) bool, True = usable
    feature_boost: torch.Tensor | None = None,  # (F,) >= 1 selection multiplier
) -> tuple[float, int, int]:
    """Best (gain, feature, bin) for splitting one leaf; gain <= 0 means don't split.

    Split "bin <= b" sends bins [0, b] left. The last bin is never a valid
    split point (empty right child).

    ``feature_boost`` steers the *selection* among positive-gain candidates
    (interaction-aware growth): the argmax runs over gain * boost, but the
    returned gain is always the true unboosted gain, so split acceptance and
    leaf-priority ordering are never inflated by the steering signal.
    """
    gain, valid = _split_gain_matrix(hist, lam, min_child_weight, min_samples_leaf)
    gain = gain - gamma
    if feature_mask is not None:
        valid &= feature_mask.unsqueeze(1)
    gain = torch.where(valid, gain, torch.tensor(float("-inf"), device=gain.device))

    sel = gain if feature_boost is None else gain * feature_boost.unsqueeze(1)
    flat_idx = int(sel.reshape(-1).argmax())
    F, B = gain.shape
    f, b = divmod(flat_idx, B)
    return float(gain[f, b]), f, b


def per_feature_gain(
    hist: torch.Tensor,  # (3, F, B)
    lam: float,
    min_child_weight: float,
    min_samples_leaf: int,
) -> torch.Tensor:
    """Best achievable split gain per feature (F,), >= 0; the importance signal
    used to weight kernel-split distances."""
    gain, valid = _split_gain_matrix(hist, lam, min_child_weight, min_samples_leaf)
    return torch.where(valid, gain, torch.zeros_like(gain)).max(dim=1).values.clamp_min(0)


def build_histogram_multi(
    binned: torch.Tensor,  # (n, F) uint8, rows of one leaf
    grad: torch.Tensor,    # (n, T) float32, per-task gradients
    hess: torch.Tensor,    # (n, T) float32, per-task hessians
) -> torch.Tensor:
    """Multi-task histogram (2T+1, F, B): T grad channels, T hess channels, and
    one shared count channel (all tasks share rows). Stacked so the
    parent-minus-child subtraction trick works on the whole tensor at once."""
    n, F = binned.shape
    T = grad.shape[1]
    B = MAX_BINS
    dev = binned.device
    flat = (torch.arange(F, device=dev, dtype=torch.long).unsqueeze(0) * B + binned.long()).reshape(-1)

    hist = torch.zeros(2 * T + 1, F * B, dtype=torch.float32, device=dev)
    for t in range(T):
        hist[t].scatter_add_(0, flat, grad[:, t].unsqueeze(1).expand(n, F).reshape(-1))
        hist[T + t].scatter_add_(0, flat, hess[:, t].unsqueeze(1).expand(n, F).reshape(-1))
    hist[2 * T].scatter_add_(0, flat, torch.ones(n * F, dtype=torch.float32, device=dev))
    return hist.view(2 * T + 1, F, B)


def find_best_split_multi(
    hist: torch.Tensor,  # (2T+1, F, B)
    n_tasks: int,
    lam: float,
    gamma: float,
    min_child_weight: float,
    min_samples_leaf: int,
    feature_mask: torch.Tensor | None = None,
) -> tuple[float, int, int]:
    """Best (gain, feature, bin) for a split shared across all tasks; the total
    gain is the sum of per-task Newton gains, so a split is chosen when it helps
    the tasks jointly. Validity uses shared counts and aggregate child hessian.
    Returns gain <= 0 to mean don't split."""
    T = n_tasks
    counts = hist[2 * T]
    CL = counts.cumsum(dim=1)
    C = CL[:, -1:]
    CR = C - CL

    total_gain = torch.zeros_like(CL)
    sum_HL = torch.zeros_like(CL)
    for t in range(T):
        GL = hist[t].cumsum(dim=1)
        HL = hist[T + t].cumsum(dim=1)
        G, H = GL[:, -1:], HL[:, -1:]
        GR, HR = G - GL, H - HL
        total_gain += 0.5 * (GL.square() / (HL + lam) + GR.square() / (HR + lam) - G.square() / (H + lam))
        sum_HL += HL
    sum_HR = sum_HL[:, -1:] - sum_HL
    total_gain = total_gain - gamma

    valid = (
        (CL >= min_samples_leaf)
        & (CR >= min_samples_leaf)
        & (sum_HL >= min_child_weight)
        & (sum_HR >= min_child_weight)
    )
    valid[:, -1] = False
    if feature_mask is not None:
        valid &= feature_mask.unsqueeze(1)
    total_gain = torch.where(valid, total_gain, torch.tensor(float("-inf"), device=hist.device))

    flat_idx = int(total_gain.reshape(-1).argmax())
    F, B = total_gain.shape
    f, b = divmod(flat_idx, B)
    return float(total_gain[f, b]), f, b

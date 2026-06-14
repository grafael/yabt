"""Adaptive feature selection based on learned importance."""

from __future__ import annotations

import torch


class AdaptiveFeatureSelector:
    """Learn which features are most valuable for splitting.

    This is inspired by attention mechanisms in transformers but applied to
    gradient boosting. After each tree, we update feature importance scores
    based on how much gradient/hessian information each feature captures.

    This guides future tree growth to prioritize high-importance features.
    """

    def __init__(self, n_features: int, device: str = "cuda", alpha: float = 0.1,
                 seed: int = 0):
        self.n_features = n_features
        self.device = device
        self.alpha = alpha  # learning rate for importance updates
        self.gen = torch.Generator(device=device).manual_seed(seed)

        # Feature importance scores (initialize uniformly)
        self.importance = torch.ones(n_features, device=device) / n_features

    def update_from_tree(self, hist: torch.Tensor):
        """Update feature importance based on histogram splits."""
        if hist.shape[1] != self.n_features:
            return

        # hist shape: (3, F, B), holding grad, hess, count per feature-bin.
        # Dispersion of gradient mass across a feature's bins is the
        # importance proxy: informative features concentrate signed gradient
        # mass unevenly, so the sum of per-bin |grad sums| is large.
        feature_gain = hist[0].abs().sum(dim=1)

        # Normalize and update importance with exponential moving average
        feature_gain = feature_gain / (feature_gain.sum() + 1e-8)
        self.importance = (
            (1 - self.alpha) * self.importance + self.alpha * feature_gain
        )

    def get_feature_mask(
        self, n_features_sample: int
    ) -> torch.Tensor:
        """Sample k features without replacement, weighted by importance."""
        if n_features_sample >= self.n_features:
            return torch.ones(self.n_features, dtype=torch.bool, device=self.device)

        mask = torch.zeros(self.n_features, dtype=torch.bool, device=self.device)
        weights = self.importance / (self.importance.sum() + 1e-8)
        sampled_idx = torch.multinomial(
            weights, n_features_sample, replacement=False, generator=self.gen
        )
        mask[sampled_idx] = True
        return mask


class GradientBasedOneSideSampling:
    """Gradient-based One-Side Sampling (LightGBM-style GOSS).

    Rows with the largest |gradient| are always kept; the rest of the kept
    budget is filled with a uniform random sample whose grad/hess are
    amplified so split statistics remain unbiased estimates of the full data.
    """

    def __init__(self, device: str = "cuda", seed: int = 0):
        self.device = device
        self.gen = torch.Generator(device=device).manual_seed(seed)

    def sample(
        self, grad: torch.Tensor, ratio: float = 0.9
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Sample rows based on gradient magnitude.

        Args:
            grad: gradient vector (n,)
            ratio: total fraction of rows to keep. The top ``1 - ratio``
                fraction by |grad| is kept deterministically and the rest of
                the budget is sampled uniformly (0.9 = top 10% + random 80%).

        Returns:
            (selected_rows, sample_weights) where ``sample_weights`` is None
            when no reweighting is needed, else a per-row multiplier for
            grad/hess (1 for top rows, amplified for randomly sampled rows).
        """
        n = grad.shape[0]
        ratio = min(max(ratio, 0.0), 1.0)
        keep_n = max(1, round(n * ratio))
        top_n = max(1, min(keep_n, round(n * (1.0 - ratio))))
        rand_n = keep_n - top_n

        order = torch.argsort(grad.abs(), descending=True)
        top_idx = order[:top_n]
        rest_idx = order[top_n:]

        if rand_n <= 0 or rest_idx.numel() == 0:
            return top_idx, None

        perm = torch.randperm(rest_idx.numel(), generator=self.gen, device=self.device)
        rand_idx = rest_idx[perm[:rand_n]]
        rows = torch.cat([top_idx, rand_idx])

        # Amplify sampled rows by (n - top_n) / rand_n so the expected
        # grad/hess sums match the full dataset (LightGBM's (1-a)/b factor).
        weights = torch.ones(rows.numel(), device=grad.device)
        weights[top_n:] = (n - top_n) / rand_idx.numel()
        return rows, weights


class FeatureInteractionDetector:
    """Detect feature pairs with strong interactions.

    Interactions occur when splits on feature A are most effective when
    feature B has a specific value. We detect this by computing correlation
    of split effectiveness across feature values.
    """

    def __init__(self, n_features: int, device: str = "cuda"):
        self.n_features = n_features
        self.device = device
        self.interaction_scores = torch.zeros(
            (n_features, n_features), device=device
        )
        self.update_count = 0

    def update_from_path_pairs(self, pairs: list[tuple[int, int]]):
        """Update scores from (ancestor_feature, descendant_feature) split pairs
        along root-to-leaf paths. A split on B underneath a split on A means
        B's effect is conditioned on A, a much stronger interaction signal
        than same-tree co-occurrence."""
        if not pairs:
            return
        # Vectorized symmetric accumulation: one index_put_ per direction instead
        # of two scalar tensor writes per pair (which dominate at deep trees).
        idx = torch.tensor(pairs, dtype=torch.long, device=self.device)
        a, b = idx[:, 0], idx[:, 1]
        keep = a != b
        a, b = a[keep], b[keep]
        ones = torch.ones(a.shape[0], device=self.device)
        self.interaction_scores.index_put_((a, b), ones, accumulate=True)
        self.interaction_scores.index_put_((b, a), ones, accumulate=True)
        self.update_count += 1

    def normalized_matrix(self) -> torch.Tensor:
        """Interaction strength above background, scaled to [0, 1].

        Counts are measured relative to the mean nonzero pair count: a matrix
        of uniform noise (early training, no real interactions) normalizes to
        all zeros, so growth steering only kicks in once some pair clearly
        stands out. Plain max-normalization instead treats the luckiest noise
        pair as maximal confidence and creates a self-reinforcing bias."""
        m = self.interaction_scores
        mx = m.max()
        if mx <= 0:
            return m
        bg = m[m > 0].mean()
        return ((m - bg) / (mx - bg + 1e-8)).clamp_min(0)

    def get_top_interactions(self, k: int = 5) -> list[tuple[int, int, float]]:
        """Get top-k feature interaction pairs."""
        if self.update_count == 0:
            return []

        # Normalize scores
        scores = self.interaction_scores / (self.update_count + 1e-8)

        # Get upper triangle (avoid duplicates)
        interactions = []
        for i in range(self.n_features):
            for j in range(i + 1, self.n_features):
                interactions.append((i, j, float(scores[i, j])))

        interactions.sort(key=lambda x: x[2], reverse=True)
        return interactions[:k]

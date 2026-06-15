"""Quick per-phase profiler for a default YABT fit. Usage: python profile_fit.py [cpu|cuda]"""
import sys, time
import numpy as np
import torch
from sklearn.datasets import make_regression

dev = sys.argv[1] if len(sys.argv) > 1 else ("cuda" if torch.cuda.is_available() else "cpu")
n, F = 20000, 30
X, y = make_regression(n_samples=n, n_features=F, n_informative=20, noise=1.0, random_state=0)
X = X.astype(np.float32); y = y.astype(np.float32)

from yabt.boosting import Booster, BoostParams, MSELoss

def run(n_est=100, **kw):
    p = BoostParams(n_estimators=n_est, device=dev, seed=0, **kw)
    b = Booster(p, MSELoss())
    if dev == "cuda": torch.cuda.synchronize()
    t = time.perf_counter()
    b.fit(X, y)
    if dev == "cuda": torch.cuda.synchronize()
    return time.perf_counter() - t

# warmup
run(n_est=10)
print(f"device={dev} n={n} F={F}")
for label, kw in [
    ("default", {}),
    ("no neural_leaves", dict(neural_leaves=False)),
    ("no interaction", dict(interaction_aware=False)),
    ("bare (no nl, no inter)", dict(neural_leaves=False, interaction_aware=False)),
]:
    ts = [run(100, **kw) for _ in range(3)]
    print(f"  {label:28s} {min(ts):.3f}s  (median {sorted(ts)[1]:.3f}s)")

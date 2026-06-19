"""A/B: torch heap grower vs the Numba JIT grower prototype (CPU).

Monkeypatches the bare-path grower in the boosting loop and times a full fit +
held-out accuracy for both. Run: python benchmarks/ab_grow_numba.py
"""
import time, sys
import numpy as np
import torch
from sklearn.datasets import make_regression, make_classification
from sklearn.metrics import r2_score, accuracy_score

import yabt.boosting as B
from yabt.boosting import Booster, BoostParams, MSELoss, LogLoss
from yabt.grow_numba import grow_tree_numba

_torch_grow = B.grow_tree


def patch(use_numba):
    if use_numba:
        def g(binned, grad, hess, binner, params, feature_mask=None, **kw):
            # Prototype handles only the bare axis path; fall back otherwise.
            if kw.get("Xnorm") is not None or kw.get("interaction_matrix") is not None:
                return _torch_grow(binned, grad, hess, binner, params, feature_mask, **kw)
            return grow_tree_numba(binned, grad, hess, binner, params, feature_mask)
        B.grow_tree = g
    else:
        B.grow_tree = _torch_grow


def bench(name, Xtr, ytr, Xte, yte, loss, scorer, **params):
    torch.set_num_threads(int(sys.argv[1]) if len(sys.argv) > 1 else 1)
    res = {}
    for tag, use_numba in [("torch", False), ("numba", True)]:
        patch(use_numba)
        def run():
            b = Booster(BoostParams(n_estimators=100, device="cpu", seed=0,
                                    interaction_aware=False, **params), loss)
            b.fit(Xtr, ytr)
            return b
        run()  # warmup (JIT compile for numba)
        ts, score = [], None
        for _ in range(3):
            t = time.perf_counter(); b = run(); ts.append(time.perf_counter() - t)
            score = scorer(yte, b.predict_margin(Xte))
        res[tag] = (min(ts), sorted(ts)[1], score)
    patch(False)
    tt, tm = res["torch"], res["numba"]
    print(f"{name:32s} torch {tt[0]:.3f}s (med {tt[1]:.3f}) score={tt[2]:.4f}  |  "
          f"numba {tm[0]:.3f}s (med {tm[1]:.3f}) score={tm[2]:.4f}  |  "
          f"speedup {tt[0]/tm[0]:.2f}x")


def main():
    threads = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    print(f"threads={threads}\n")
    # Regression, varying n
    for n, F in [(5000, 30), (20000, 30), (50000, 50)]:
        X, y = make_regression(n_samples=n, n_features=F, n_informative=int(F*0.7),
                               noise=1.0, random_state=0)
        X = X.astype(np.float32); y = y.astype(np.float32)
        s = int(n*0.8)
        bench(f"reg n={n} F={F} (default)", X[:s], y[:s], X[s:], y[s:],
              MSELoss(), r2_score)
        bench(f"reg n={n} F={F} (no neural)", X[:s], y[:s], X[s:], y[s:],
              MSELoss(), r2_score, neural_leaves=False)
    # Classification
    X, y = make_classification(n_samples=20000, n_features=30, n_informative=15,
                               random_state=0)
    X = X.astype(np.float32); y = y.astype(np.float32)
    s = 16000
    bench("clf n=20000 F=30 (default)", X[:s], y[:s], X[s:], y[s:],
          LogLoss(), lambda yt, m: accuracy_score(yt, (m > 0).astype(int)))


if __name__ == "__main__":
    main()

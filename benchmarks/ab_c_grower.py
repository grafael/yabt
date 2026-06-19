"""A/B: single-threaded Numba grower vs the OpenMP C grower (+ C apply), CPU.

Times a full fit and held-out R2 with ``c_grower`` off (Numba, 1 thread) vs on
(C, multi-thread), and lines them up against XGBoost / LightGBM at the same
thread count so the "on par with other boosting tools" claim is measurable.

The C grower reimplements the Numba leaf-wise kernel with the histogram build
and per-feature split search parallelized across cores (feature-major data
layout, lock-free), and Tree.apply gets an OpenMP per-row routing fast path.
Trees are bit-identical to the Numba grower; only wall-clock changes.

Run: python benchmarks/ab_c_grower.py [n_threads]
"""
import sys
import time

import numpy as np
from sklearn.datasets import make_regression

from yabt.boosting import Booster, BoostParams, MSELoss

TH = int(sys.argv[1]) if len(sys.argv) > 1 else 8
NEST, LR, LEAVES = 200, 0.1, 31


def _r2(y, p):
    return 1.0 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def _best(fn, reps=3):
    fn()  # warm (JIT compile / lib build)
    times = []
    for _ in range(reps):
        el, score = fn()
        times.append(el)
    return min(times), score


def yabt(Xtr, ytr, Xte, yte, c_grower):
    def run():
        p = BoostParams(n_estimators=NEST, learning_rate=LR, max_leaves=LEAVES,
                        device="cpu", seed=0, c_grower=c_grower,
                        c_grower_threads=TH, neural_leaves=False,
                        interaction_aware=False, refine_steps=0)
        b = Booster(p, MSELoss())
        t = time.perf_counter()
        b.fit(Xtr, ytr)
        el = time.perf_counter() - t
        return el, _r2(yte, b.predict_margin(Xte))
    return _best(run)


def xgboost(Xtr, ytr, Xte, yte):
    import xgboost as xgb

    def run():
        m = xgb.XGBRegressor(n_estimators=NEST, learning_rate=LR,
                             max_leaves=LEAVES, tree_method="hist",
                             grow_policy="lossguide", n_jobs=TH, max_depth=0,
                             reg_lambda=1.0)
        t = time.perf_counter()
        m.fit(Xtr, ytr)
        el = time.perf_counter() - t
        return el, _r2(yte, m.predict(Xte))
    return _best(run)


def lightgbm(Xtr, ytr, Xte, yte):
    import lightgbm as lgb

    def run():
        m = lgb.LGBMRegressor(n_estimators=NEST, learning_rate=LR,
                              num_leaves=LEAVES, n_jobs=TH, verbosity=-1,
                              reg_lambda=1.0)
        t = time.perf_counter()
        m.fit(Xtr, ytr)
        el = time.perf_counter() - t
        return el, _r2(yte, m.predict(Xte))
    return _best(run)


def main():
    n, F = 50000, 30
    X, y = make_regression(n_samples=n, n_features=F, n_informative=20,
                           noise=1.0, random_state=0)
    X = X.astype(np.float32)
    y = y.astype(np.float32)
    Xtr, Xte, ytr, yte = X[:40000], X[40000:], y[:40000], y[40000:]

    print(f"n={n} F={F} trees={NEST} leaves={LEAVES} threads={TH}")
    rows = [
        ("YABT (Numba, 1t)", lambda: yabt(Xtr, ytr, Xte, yte, False)),
        (f"YABT (C, {TH}t)", lambda: yabt(Xtr, ytr, Xte, yte, True)),
        (f"XGBoost ({TH}t)", lambda: xgboost(Xtr, ytr, Xte, yte)),
        (f"LightGBM ({TH}t)", lambda: lightgbm(Xtr, ytr, Xte, yte)),
    ]
    base = None
    for name, fn in rows:
        try:
            t, score = fn()
        except ImportError as exc:
            print(f"  {name:18s} skipped ({exc})")
            continue
        if base is None:
            base = t
        print(f"  {name:18s} {t:.3f}s  R2={score:.4f}  ({base / t:.2f}x vs Numba)")


if __name__ == "__main__":
    main()

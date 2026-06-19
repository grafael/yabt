"""A/B: sparse vs dense Numba histogram build on wide/sparse data.

Mirrors the openml_benchmark protocol (70/30 split, R^2, default YABT config at
100 trees / lr 0.1 / max_leaves 31) and toggles only ``sparse_hist``. Reports
per-seed and mean R^2 + fit time for dense (sparse_hist=False) vs sparse
(sparse_hist=True), plus a bitwise-ish correctness check on a synthetic sparse
set. The win is expected on wide+sparse data and neutrality elsewhere.

    python benchmarks/ab_sparse_hist.py                 # Santander (42572)
    python benchmarks/ab_sparse_hist.py --did 42730     # another reg dataset
"""

from __future__ import annotations

import argparse
import time

import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from datasets import load_dataset
from yabt import YABTRegressor

CFG = dict(n_estimators=100, learning_rate=0.1, max_leaves=31, device="cpu")


def correctness_check():
    """Dense and sparse builders must give the same model on sparse data."""
    rng = np.random.default_rng(0)
    n, F = 1500, 400
    X = np.zeros((n, F), dtype=np.float32)
    mask = rng.random((n, F)) < 0.05            # 5% nonzero
    X[mask] = rng.standard_normal(mask.sum()).astype(np.float32)
    w = rng.standard_normal(F).astype(np.float32)
    y = (X @ w + 0.1 * rng.standard_normal(n)).astype(np.float32)

    dense = YABTRegressor(**CFG, sparse_hist=False).fit(X, y).predict(X)
    sparse = YABTRegressor(**CFG, sparse_hist=True).fit(X, y).predict(X)
    diff = float(np.abs(dense - sparse).max())
    rel = diff / (float(np.abs(dense).mean()) + 1e-12)
    print(f"correctness: max|dense-sparse|={diff:.3e}  rel={rel:.3e}  "
          f"R2 dense={r2_score(y, dense):.5f} sparse={r2_score(y, sparse):.5f}")


def warmup():
    """Pay the one-time Numba JIT compile for both build paths before timing."""
    rng = np.random.default_rng(0)
    X = np.zeros((200, 30), dtype=np.float32)
    X[rng.random(X.shape) < 0.1] = 1.0
    y = rng.standard_normal(200).astype(np.float32)
    for sh in (False, True):
        YABTRegressor(n_estimators=3, max_leaves=8, device="cpu", sparse_hist=sh).fit(X, y)


def ab(did, name, task, seeds):
    X, y, _, _ = load_dataset(did, task)
    Xn, yn = X.to_numpy(np.float32), np.asarray(y, np.float32)
    n, F = Xn.shape
    zero_frac = float((Xn == 0).mean())
    print(f"\n{name} (id={did}): n={n} F={F} zero_frac={zero_frac:.3f}\n")

    res = {"dense": {"r2": [], "t": []}, "sparse": {"r2": [], "t": []}}
    for seed in range(seeds):
        Xtr, Xte, ytr, yte = train_test_split(Xn, yn, test_size=0.3, random_state=seed)
        for label, sh in (("dense", False), ("sparse", True)):
            t0 = time.time()
            m = YABTRegressor(**CFG, sparse_hist=sh).fit(Xtr, ytr)
            dt = time.time() - t0
            r2 = r2_score(yte, m.predict(Xte))
            res[label]["r2"].append(r2)
            res[label]["t"].append(dt)
            print(f"  seed {seed}  {label:6s}  R2={r2:.5f}  time={dt:7.2f}s")

    print()
    for label in ("dense", "sparse"):
        r2 = np.array(res[label]["r2"]); t = np.array(res[label]["t"])
        print(f"  {label:6s}  R2={r2.mean():.5f}+/-{r2.std():.5f}  "
              f"time={t.mean():.2f}s")
    sp = np.array(res["sparse"]["t"]).mean()
    dn = np.array(res["dense"]["t"]).mean()
    print(f"\n  speedup: {dn / sp:.2f}x  "
          f"(dense {dn:.1f}s -> sparse {sp:.1f}s)")
    print(f"  R2 delta (sparse-dense): "
          f"{np.array(res['sparse']['r2']).mean() - np.array(res['dense']['r2']).mean():+.5f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--did", type=int, default=42572)
    ap.add_argument("--name", default="Santander_transaction_value")
    ap.add_argument("--task", default="reg")
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()

    print("warming up JIT...")
    warmup()
    correctness_check()
    ab(args.did, args.name, args.task, args.seeds)

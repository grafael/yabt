"""A/B: product_features OFF vs ON.

Isolates gradient-guided multiplicative feature construction. Reports accuracy
delta and wall-clock so the "large win where multiplicative structure exists,
neutral (and time-neutral) elsewhere" claim is checked on:
  * synthetic multiplicative targets (where the greedy splitter plateaus), and
  * cached real OpenML tabular datasets (where it should stay neutral).
"""
import sys, time, warnings
sys.path.insert(0, "benchmarks")
warnings.filterwarnings("ignore")
import numpy as np, torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
import datasets
from yabt import YABTRegressor, YABTClassifier

BASE = dict(n_estimators=200, learning_rate=0.1)
SEEDS = [0, 1, 2]
REAL = [("num_reg", 45033, "abalone"), ("num_reg", 44132, "cpu_act"),
        ("num_reg", 44138, "houses"), ("num_reg", 44148, "superconduct"),
        ("num_clf", 44125, "MagicTelescope"), ("num_clf", 45026, "heloc"),
        ("num_clf", 44126, "bank-marketing")]


def synth(kind, seed):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((12000, 10)).astype(np.float32)
    if kind == "mult2":
        y = X[:, 0] * X[:, 1]
    elif kind == "mult3":
        y = X[:, 0] * X[:, 1] * X[:, 2]
    elif kind == "mult2_plus_linear":   # mixed additive + multiplicative
        y = X[:, 0] * X[:, 1] + 0.7 * X[:, 3] + 0.5 * X[:, 4]
    y = (y + 0.1 * rng.standard_normal(12000)).astype(np.float32)
    return X, y


def evaluate(Est, score, make_split, dev):
    out = {}
    for arm, extra in [("off", {}), ("on", dict(product_features=True))]:
        sc, ti = [], []
        for s in SEEDS:
            Xtr, Xte, ytr, yte = make_split(s)
            t0 = time.perf_counter()
            m = Est(device=dev, seed=0, **BASE, **extra).fit(Xtr, ytr)
            pred = m.predict(Xte)
            if dev == "cuda":
                torch.cuda.synchronize()
            ti.append(time.perf_counter() - t0)
            sc.append(score(yte, pred))
        out[arm] = (float(np.mean(sc)), float(np.median(ti)))
    return out


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {dev}\n")
    if dev == "cuda":  # warm up GPU clocks so the first measured fit isn't cold
        Xw = np.random.randn(8000, 12).astype(np.float32)
        yw = np.random.randn(8000).astype(np.float32)
        YABTRegressor(device="cuda", seed=0, **BASE).fit(Xw, yw)
        torch.cuda.synchronize()

    print("=== synthetic (multiplicative structure) ===")
    for kind in ["mult2", "mult3", "mult2_plus_linear"]:
        def mk(s, kind=kind):
            X, y = synth(kind, s)
            return train_test_split(X, y, test_size=0.3, random_state=s)
        r = evaluate(YABTRegressor, r2_score, mk, dev)
        (sa, ta), (sb, tb) = r["off"], r["on"]
        print(f"{kind:18s} r2 off={sa:.4f} on={sb:.4f} (d={sb-sa:+.4f}, "
              f"{(sb-sa)/abs(sa)*100:+.1f}%) | t {ta:.2f}->{tb:.2f}s")

    print("\n=== real OpenML (neutrality check) ===")
    deltas = []
    for suite, did, name in REAL:
        task = "reg" if suite == "num_reg" else "clf"
        Xdf, y, _, _ = datasets.load_dataset(did, task, max_rows=20000)
        X = Xdf.to_numpy(np.float32); y = y.astype(np.float32)
        Est = YABTRegressor if task == "reg" else YABTClassifier
        score = r2_score if task == "reg" else accuracy_score
        def mk(s, X=X, y=y):
            return train_test_split(X, y, test_size=0.3, random_state=s)
        r = evaluate(Est, score, mk, dev)
        (sa, ta), (sb, tb) = r["off"], r["on"]
        deltas.append(sb - sa)
        print(f"{name:16s} {task} off={sa:.4f} on={sb:.4f} (d={sb-sa:+.4f}) | "
              f"t {ta:.2f}->{tb:.2f}s")
    print(f"\nreal-data mean |delta| = {np.mean(np.abs(deltas)):.4f} "
          f"(mean delta {np.mean(deltas):+.4f})")


if __name__ == "__main__":
    main()

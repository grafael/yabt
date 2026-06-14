"""A/B: leaf-wise (default) vs level-wise grower. Isolates the grower by turning
off interaction steering / neural leaves / refinement for both arms."""
import sys, time, warnings
sys.path.insert(0, "benchmarks")
warnings.filterwarnings("ignore")
import numpy as np, torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
import datasets
from yabt import YABTRegressor, YABTClassifier

BASE = dict(n_estimators=100, learning_rate=0.1, max_leaves=31, max_depth=6,
            refine_steps=0, neural_leaves=False, interaction_aware=False)
SEEDS = [0, 1, 2]
# (suite, id, name): cached numerical datasets, ordered small to large to show scaling
PICKS = [("num_reg", 45033, "abalone"),        # 4.2k
         ("num_reg", 44132, "cpu_act"),        # 8.2k
         ("num_clf", 44125, "MagicTelescope"), # 13k
         ("num_reg", 44138, "houses"),         # 20k
         ("num_reg", 44148, "superconduct"),   # 21k x79 (wide)
         ("num_clf", 45021, "jannis"),         # 57k x54
         ("num_clf", 44128, "MiniBooNE")]      # 73k x50


def timed_fit_predict(Est, Xtr, ytr, Xte, dev, reps=3, **extra):
    pred = None
    times = []
    for _ in range(reps):  # repeat; report median to dampen GPU-clock variance
        m = Est(device=dev, seed=0, **BASE, **extra)
        if dev == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        m.fit(Xtr, ytr)
        pred = m.predict(Xte)
        if dev == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return pred, float(np.median(times))


def main():
    devs = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    print(f"devices: {devs}\n")
    if "cuda" in devs:  # warm up GPU clocks/caches so first measured fit isn't cold
        Xw = np.random.randn(8000, 20).astype(np.float32)
        yw = np.random.randn(8000).astype(np.float32)
        for ex in ({}, dict(levelwise=True)):
            YABTRegressor(device="cuda", seed=0, **BASE, **ex).fit(Xw, yw)
        torch.cuda.synchronize()
    rows = []
    for suite, did, name in PICKS:
        task = "reg" if suite == "num_reg" else "clf"
        try:
            Xdf, y, cat, _ = datasets.load_dataset(did, task, max_rows=20000)
        except Exception as e:
            print(f"skip {name}: {e}"); continue
        X = Xdf.to_numpy(np.float32)
        y = y.astype(np.float32)
        Est = YABTRegressor if task == "reg" else YABTClassifier
        score = r2_score if task == "reg" else accuracy_score
        for dev in devs:
            agg = {}  # arm -> (scores, times)
            for arm, extra in [("leaf", {}), ("level", dict(levelwise=True))]:
                sc, ti = [], []
                for s in SEEDS:
                    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=s)
                    pred, dt = timed_fit_predict(Est, Xtr, ytr, Xte, dev, **extra)
                    sc.append(score(yte, pred)); ti.append(dt)
                agg[arm] = (np.mean(sc), np.mean(ti))
            (sa, ta), (sb, tb) = agg["leaf"], agg["level"]
            speed = ta / tb
            rows.append((name, dev, sa, sb, ta, tb, speed))
            print(f"{name:16s} [{dev:4s}] score leaf={sa:.4f} level={sb:.4f} (d={sb-sa:+.4f}) | "
                  f"time leaf={ta:.2f}s level={tb:.2f}s  speedup={speed:.2f}x")
    print("\n=== summary ===")
    for dev in devs:
        r = [x for x in rows if x[1] == dev]
        if not r: continue
        dscore = np.mean([x[3] - x[2] for x in r])
        sp = np.mean([x[6] for x in r])
        print(f"[{dev:4s}] mean score delta (level-leaf): {dscore:+.4f}   mean speedup: {sp:.2f}x")


if __name__ == "__main__":
    main()

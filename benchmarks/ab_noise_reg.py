"""A/B: regularization on the three amlb_reg 'noise target' datasets.

Hypothesis: on near-zero-signal targets YABT's default capacity (max_leaves=31,
neural/linear leaves on, interaction-aware on, gamma=0) fits noise, landing a
more-negative R2 than the histogram baselines. Sweep single reg levers + a
combined config, same CV as openml_benchmark (3 seeds, 30% test, 50k cap).
"""
import sys, time, warnings
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

sys.path.insert(0, ".")
warnings.filterwarnings("ignore")
from datasets import load_dataset          # noqa: E402
from yabt import YABTRegressor             # noqa: E402

NOISE = [(550, "quake"), (42724, "OnlineNewsPopularity"), (42728, "Airlines_DepDelay_10M")]
SEEDS = [0, 1, 2]
BASE = dict(n_estimators=100, learning_rate=0.1, max_leaves=31, device="cpu")

VARIANTS = {
    "baseline":          {},
    "reg_lambda=10":     dict(reg_lambda=10.0),
    "reg_lambda=50":     dict(reg_lambda=50.0),
    "gamma=1.0":         dict(gamma=1.0),
    "min_samples_leaf=100": dict(min_samples_leaf=100),
    "max_leaves=8":      dict(max_leaves=8),
    "no_neural_leaves":  dict(neural_leaves=False),
    "no_interaction":    dict(interaction_aware=False),
    "subsample=.7,col=.7": dict(subsample=0.7, colsample=0.7),
    "combined":          dict(reg_lambda=10.0, gamma=1.0, min_samples_leaf=50,
                              max_leaves=15, neural_leaves=False),
}

def warmup():
    X = np.random.rand(64, 4).astype(np.float32); y = np.random.rand(64)
    YABTRegressor(n_estimators=2, max_leaves=4, device="cpu").fit(X, y)

def run():
    warmup()
    # cache splits per dataset
    results = {v: {} for v in VARIANTS}
    for did, name in NOISE:
        X, y, cat_idx, _ = load_dataset(did, "reg", max_rows=50000, seed=0)
        Xnp_full = X.to_numpy(np.float32)
        splits = []
        for s in SEEDS:
            idx = np.arange(len(y))
            tr, te = train_test_split(idx, test_size=0.3, random_state=s)
            splits.append((tr, te))
        for vname, over in VARIANTS.items():
            scs, tms = [], []
            for tr, te in splits:
                p = {**BASE, **over}
                m = YABTRegressor(**p)
                t0 = time.time()
                m.fit(Xnp_full[tr], y[tr],
                      categorical_features=cat_idx or None)
                dt = time.time() - t0
                scs.append(r2_score(y[te], m.predict(Xnp_full[te])))
                tms.append(dt)
            results[vname][name] = (np.mean(scs), np.mean(tms))
        print(f"done {name}")
    return results

def main():
    r = run()
    names = [n for _, n in NOISE]
    w = 13
    hdr = f"{'variant':22s}" + "".join(f"{n[:w]:>{w+2}s}" for n in names) + f"{'mean R2':>10s}"
    print("\n" + hdr); print("-" * len(hdr))
    base = {n: r["baseline"][n][0] for n in names}
    for v in VARIANTS:
        cells = ""
        deltas = []
        for n in names:
            sc = r[v][n][0]
            d = sc - base[n]
            deltas.append(sc)
            mark = "" if v == "baseline" else f" ({d:+.3f})"
            cells += f"{sc:>{w-5}.3f}{mark:>7s}"
        print(f"{v:22s}{cells}{np.mean(deltas):>10.4f}")
    print("\nfit time (s, mean over seeds):")
    for v in VARIANTS:
        ts = "  ".join(f"{n[:6]}={r[v][n][1]:.2f}" for n in names)
        print(f"  {v:22s} {ts}")

if __name__ == "__main__":
    main()

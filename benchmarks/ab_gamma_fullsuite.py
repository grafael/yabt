"""Full-suite A/B: does a non-zero `gamma` (min split gain) regress YABT?

gamma=1.0 nearly erased the noise-target overfit (ab_noise_reg.py). gamma is
target-scale-dependent, so before it could be a default we must confirm it's
net-neutral-or-positive across all 90 datasets. YABT-only, same CV as
openml_benchmark (3 seeds, 30% test, 50k cap), dedup shared datasets.
"""
import sys, time, warnings, json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score

sys.path.insert(0, ".")
warnings.filterwarnings("ignore")
from datasets import list_datasets, load_dataset, canonical_name  # noqa: E402
from yabt import YABTClassifier, YABTRegressor                    # noqa: E402

SEEDS = [0, 1, 2]
BASE = dict(n_estimators=100, learning_rate=0.1, max_leaves=31, device="cpu")
GAMMAS = [0.0, 0.1, 1.0]   # 0.0 == baseline

def warmup():
    X = np.random.rand(64, 4).astype(np.float32); y = np.random.rand(64)
    YABTRegressor(n_estimators=2, max_leaves=4, device="cpu").fit(X, y)

def score(task, yte, yp):
    return accuracy_score(yte, np.asarray(yp).ravel()) if task == "clf" else r2_score(yte, yp)

def run():
    warmup()
    seen = set()
    rows = []  # (tag, name, task, {gamma: mean_score})
    for tag, did, name, task in list_datasets():
        cn = canonical_name(name)
        if cn in seen:
            continue
        seen.add(cn)
        try:
            X, y, cat_idx, _ = load_dataset(did, task, max_rows=50000, seed=0)
        except Exception as e:
            print(f"skip {name}: {str(e)[:60]}"); continue
        Xnp = X.to_numpy(np.float32)
        splits = [train_test_split(np.arange(len(y)), test_size=0.3, random_state=s,
                                   stratify=(y if task == "clf" else None)) for s in SEEDS]
        Est = YABTClassifier if task == "clf" else YABTRegressor
        per_g = {}
        for g in GAMMAS:
            scs = []
            for tr, te in splits:
                m = Est(**BASE, gamma=g)
                m.fit(Xnp[tr], y[tr], categorical_features=cat_idx or None)
                scs.append(score(task, y[te], m.predict(Xnp[te])))
            per_g[g] = float(np.mean(scs))
        rows.append((tag, name, task, per_g))
        b = per_g[0.0]
        flags = "  ".join(f"g{g}={per_g[g]:+.4f}({per_g[g]-b:+.4f})" for g in GAMMAS[1:])
        print(f"{name[:34]:34s} base={b:+.4f}  {flags}")
    return rows

def main():
    rows = run()
    json.dump(rows, open("ab_gamma_fullsuite.json", "w"), indent=1)
    print("\n" + "=" * 70)
    for g in GAMMAS[1:]:
        deltas = [r[3][g] - r[3][0.0] for r in rows]
        improved = [d for d in deltas if d > 0.002]
        regressed = [d for d in deltas if d < -0.002]
        flat = [d for d in deltas if -0.002 <= d <= 0.002]
        print(f"\ngamma={g} vs baseline  (n={len(rows)})")
        print(f"  mean Δ      = {np.mean(deltas):+.4f}   median Δ = {np.median(deltas):+.4f}")
        print(f"  improved    = {len(improved)}   flat(±.002) = {len(flat)}   regressed = {len(regressed)}")
        print(f"  sum improved= {sum(improved):+.3f}   sum regressed = {sum(regressed):+.3f}")
        worst = sorted(rows, key=lambda r: r[3][g] - r[3][0.0])[:6]
        best = sorted(rows, key=lambda r: r[3][0.0] - r[3][g])[:6]
        print("  biggest regressions:")
        for r in worst:
            print(f"    {r[0]}:{r[1][:30]:30s} {r[3][0.0]:+.4f} -> {r[3][g]:+.4f} ({r[3][g]-r[3][0.0]:+.4f})")
        print("  biggest improvements:")
        for r in best:
            print(f"    {r[0]}:{r[1][:30]:30s} {r[3][0.0]:+.4f} -> {r[3][g]:+.4f} ({r[3][g]-r[3][0.0]:+.4f})")

if __name__ == "__main__":
    main()

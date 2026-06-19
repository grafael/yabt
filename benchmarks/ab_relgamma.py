"""A/B: scale-invariant relative min-split-gain (min_split_gain_rel) as a default.

Sweeps rho across the full deduped OpenML suite (same protocol as
openml_benchmark) and reports per-suite mean score plus regression counts, to
decide whether a relative-gamma floor is a net-positive default."""
import sys, json, warnings
sys.path.insert(0, "benchmarks")
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from datasets import list_datasets, load_dataset, canonical_name, SUITES
from yabt import YABTClassifier, YABTRegressor
warnings.filterwarnings("ignore")

RHOS = [0.0, 0.5, 1.0, 2.0]
SEEDS = 3
MAX_ROWS = 50000

def fit_score(task, X, y, cat_idx, rho, seed):
    strat = y if task == "clf" else None
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=strat)
    Est = YABTClassifier if task == "clf" else YABTRegressor
    m = Est(n_estimators=100, learning_rate=0.1, max_leaves=31, device="cpu",
            min_split_gain_rel=rho)
    m.fit(Xtr, ytr, categorical_features=cat_idx or None)
    p = m.predict(Xte)
    return accuracy_score(yte, p) if task == "clf" else r2_score(yte, p)

results = {}
seen = set()
for tag in SUITES:
    for _, did, name, task in list_datasets(tag):
        key = (canonical_name(name), task)
        if key in seen:
            continue
        seen.add(key)
        try:
            X, y, cat_idx, _ = load_dataset(did, task, max_rows=MAX_ROWS, seed=0)
        except Exception as e:
            print(f"  load FAIL {name}: {str(e)[:60]}", flush=True)
            continue
        Xn = X.to_numpy(np.float32)
        row = {}
        for rho in RHOS:
            scs = [fit_score(task, Xn, y, cat_idx, rho, s) for s in range(SEEDS)]
            row[rho] = float(np.mean(scs))
        results[f"{tag}:{name}"] = {"task": task, "scores": row}
        base = row[0.0]
        deltas = "  ".join(f"{r}:{row[r]-base:+.4f}" for r in RHOS[1:])
        print(f"{tag:10s} {name[:30]:30s} base={base:+.4f}  {deltas}", flush=True)

json.dump(results, open("benchmarks/ab_relgamma.json", "w"), indent=2)

# Summary
print("\n" + "=" * 70)
for tag in SUITES:
    rows = [v["scores"] for k, v in results.items() if k.startswith(tag + ":")]
    if not rows:
        continue
    line = f"{tag:10s} n={len(rows):2d}  "
    for rho in RHOS:
        line += f"rho{rho}={np.mean([r[rho] for r in rows]):+.4f}  "
    print(line)
print("-" * 70)
allrows = [v["scores"] for v in results.values()]
for rho in RHOS:
    mean = np.mean([r[rho] for r in allrows])
    regr = sum(1 for r in allrows if r[rho] - r[0.0] < -0.005)
    impr = sum(1 for r in allrows if r[rho] - r[0.0] > 0.005)
    print(f"OVERALL rho={rho}: mean={mean:+.4f}  improved>{impr}  regressed={regr}")

"""End-to-end goal check: does auto_tune (with the scale-invariant relative-gamma
candidate) lift the cat_reg / amlb_reg suite means vs the plain default, while
staying neutral on a sample of other suites?"""
import sys, warnings
sys.path.insert(0, "benchmarks")
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from datasets import list_datasets, load_dataset, canonical_name
from yabt import YABTClassifier, YABTRegressor
warnings.filterwarnings("ignore")

SEEDS = 2
MAX_ROWS = 50000
SKIP_BIG = {"Airlines_DepDelay_10M", "Buzzinsocialmedia_Twitter", "Yolanda",
            "nyc-taxi-green-dec-2016", "black_friday"}  # huge: auto_tune too slow here
# target suites in full + a neutral spot-check sample from other suites
TARGET = ["cat_reg", "amlb_reg"]
SPOT = {"num_reg": {"wine_quality", "elevators", "diamonds", "houses"},
        "ctr23_reg": {"airfoil_self_noise", "kin8nm", "concrete_compressive_strength"},
        "num_clf": {"electricity", "covertype", "credit"}}

def evals(task, X, y, cat_idx, auto):
    out = []
    for s in range(SEEDS):
        strat = y if task == "clf" else None
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=s, stratify=strat)
        Est = YABTClassifier if task == "clf" else YABTRegressor
        m = Est(n_estimators=100, learning_rate=0.1, max_leaves=31, device="cpu", auto_tune=auto)
        m.fit(Xtr, ytr, categorical_features=cat_idx or None)
        p = m.predict(Xte)
        out.append(accuracy_score(yte, p) if task == "clf" else r2_score(yte, p))
    return float(np.mean(out))

def run(tag, name, did, task):
    X, y, cat_idx, _ = load_dataset(did, task, max_rows=MAX_ROWS, seed=0)
    Xn = X.to_numpy(np.float32)
    b = evals(task, Xn, y, cat_idx, False)
    a = evals(task, Xn, y, cat_idx, True)
    print(f"{tag:10s} {name[:28]:28s} base={b:+.4f} auto={a:+.4f} d={a-b:+.4f}", flush=True)
    return b, a

seen = set()
suite_d = {}
for tag in TARGET + list(SPOT):
    rows = []
    for _, did, name, task in list_datasets(tag):
        key = (canonical_name(name), task)
        if key in seen:
            continue
        if tag in SPOT and name not in SPOT[tag]:
            continue
        if name in SKIP_BIG:
            continue
        seen.add(key)
        try:
            rows.append(run(tag, name, did, task))
        except Exception as e:
            print(f"  {name} FAIL {str(e)[:50]}", flush=True)
    if rows:
        db = np.mean([r[0] for r in rows]); da = np.mean([r[1] for r in rows])
        suite_d[tag] = (db, da, len(rows))

print("\n" + "=" * 60)
for tag, (db, da, n) in suite_d.items():
    print(f"{tag:10s} n={n:2d}  base={db:+.4f}  auto={da:+.4f}  delta={da-db:+.4f}")

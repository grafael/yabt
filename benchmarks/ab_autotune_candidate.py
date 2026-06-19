"""Before/after for the auto_tune candidate swap: does the scale-invariant
relative-gamma candidate beat the old scale-dependent absolute gamma=1.0 on the
noisy targets (different y-scales), while staying neutral on smooth targets?"""
import sys, warnings
sys.path.insert(0, "benchmarks")
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from datasets import list_datasets, load_dataset
import yabt.auto_tune as at
from yabt import YABTRegressor
warnings.filterwarnings("ignore")

BASE = [
    ("user-config", {}),
    ("slow-deep", {"learning_rate": 0.05, "max_leaves": 63}),
    ("fast-shallow", {"learning_rate": 0.2, "max_leaves": 15}),
    ("constant-leaves", {"neural_leaves": False}),
    ("strong-interactions", {"interaction_boost": 1.0}),
]
TAIL = [("fine-grain", {"min_samples_leaf": 5, "max_leaves": 63})]
OLD = BASE + [("regularized-splits", {"gamma": 1.0, "max_leaves": 15})] + TAIL
NEW = BASE + [("regularized-splits", {"min_split_gain_rel": 0.5}),
              ("regularized-splits-strong", {"min_split_gain_rel": 2.0, "max_leaves": 15})] + TAIL

def make(cands):
    def _c(n):
        return [c for c in cands if not (c[0] == "fine-grain" and n < 5000)]
    return _c

NAMES = ["quake", "forest_fires", "solar_flare", "abalone", "yprop_4_1",
         "topo_2_1", "sensory", "airfoil_self_noise", "kin8nm", "white_wine"]
ids = {}
for tag in ["num_reg", "cat_reg", "amlb_reg", "ctr23_reg"]:
    for _, did, name, task in list_datasets(tag):
        if name in NAMES and name not in ids:
            ids[name] = did

print(f"{'dataset':22s}{'OLD(abs)':>10}{'NEW(rel)':>10}{'delta':>9}  old_sel -> new_sel")
for name in NAMES:
    if name not in ids:
        print(f"{name}: not found"); continue
    X, y, cat_idx, _ = load_dataset(ids[name], "reg", max_rows=50000, seed=0)
    Xn = X.to_numpy(np.float32)
    res = {}
    for label, cands in [("old", OLD), ("new", NEW)]:
        at._candidates = make(cands)
        scs, sels = [], []
        for s in range(3):
            Xtr, Xte, ytr, yte = train_test_split(Xn, y, test_size=0.3, random_state=s)
            m = YABTRegressor(n_estimators=100, learning_rate=0.1, max_leaves=31,
                              device="cpu", auto_tune=True)
            m.fit(Xtr, ytr, categorical_features=cat_idx or None)
            scs.append(r2_score(yte, m.predict(Xte)))
            sels.append(m.booster_.tuning_report_["selected"])
        from collections import Counter
        res[label] = (np.mean(scs), Counter(sels).most_common(1)[0][0])
    (mo, so), (mn, sn) = res["old"], res["new"]
    print(f"{name:22s}{mo:>10.4f}{mn:>10.4f}{mn-mo:>+9.4f}  {so} -> {sn}")

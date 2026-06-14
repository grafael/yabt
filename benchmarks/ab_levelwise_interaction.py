"""A/B: does level-wise preserve the interaction_aware accuracy win that
leaf-wise gets? Compares grower x steering (2x2) on interaction-heavy data."""
import sys, warnings
sys.path.insert(0, "benchmarks")
warnings.filterwarnings("ignore")
import numpy as np, torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from yabt import YABTRegressor, YABTClassifier

BASE = dict(n_estimators=150, learning_rate=0.1, max_leaves=31, max_depth=6,
            refine_steps=0, neural_leaves=False)
SEEDS = [0, 1, 2, 3]
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def make(kind, seed):
    """Interaction-heavy: target driven by feature *products* (pairwise)."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(7000, 12)).astype("float32")
    signal = X[:, 0] * X[:, 1] + X[:, 2] * X[:, 3] - X[:, 4] * X[:, 5]
    if kind == "reg":
        y = (signal + 0.3 * rng.normal(size=7000)).astype("float32")
    else:
        y = (signal + 0.3 * rng.normal(size=7000) > 0).astype("float32")
    return train_test_split(X, y, test_size=0.3, random_state=seed)


def cell(Est, score, lw, ia):
    out = []
    for s in SEEDS:
        Xtr, Xte, ytr, yte = make("reg" if Est is YABTRegressor else "clf", s)
        m = Est(device=DEV, seed=0, levelwise=lw, interaction_aware=ia, **BASE).fit(Xtr, ytr)
        out.append(score(yte, m.predict(Xte)))
    return float(np.mean(out))


def main():
    print(f"device={DEV}, {len(SEEDS)} seeds, interaction-heavy products\n")
    for name, Est, score in [("regression R2", YABTRegressor, r2_score),
                             ("classification acc", YABTClassifier, accuracy_score)]:
        leaf_off = cell(Est, score, False, False)
        leaf_on = cell(Est, score, False, True)
        lvl_off = cell(Est, score, True, False)
        lvl_on = cell(Est, score, True, True)
        print(f"== {name} ==")
        print(f"  leaf : off={leaf_off:.4f}  on={leaf_on:.4f}  steering gain={leaf_on-leaf_off:+.4f}")
        print(f"  level: off={lvl_off:.4f}  on={lvl_on:.4f}  steering gain={lvl_on-lvl_off:+.4f}")
        print(f"  parity (level_on - leaf_on): {lvl_on-leaf_on:+.4f}\n")


if __name__ == "__main__":
    main()

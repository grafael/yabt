"""Does auto_tune recover the noise-target win that a global gamma gave?

auto_tune's grid has NO gamma candidate (auto_tune.py), but it has lower-capacity
candidates (fast-shallow lr=0.2/leaves=15, constant-leaves). Question: does the
validation search adaptively (a) improve the gamma-winner datasets and (b) leave
the gamma-loser high-signal datasets alone? Compares baseline (user-config) vs
auto_tune=True, prints the selected candidate, and shows the gamma=1.0 delta from
the full-suite run for reference. Same CV (3 seeds, 30% test, 50k cap).
"""
import sys, json, warnings, collections
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

sys.path.insert(0, ".")
warnings.filterwarnings("ignore")
from datasets import list_datasets, load_dataset  # noqa: E402
from yabt import YABTRegressor                    # noqa: E402

SEEDS = [0, 1, 2]
BASE = dict(n_estimators=100, learning_rate=0.1, max_leaves=31, device="cpu")
# gamma=1.0 winners and losers from ab_gamma_fullsuite
WINNERS = ["quake", "forest_fires", "solar_flare", "student_performance_por", "abalone"]
LOSERS  = ["airfoil_self_noise", "kin8nm", "concrete_compressive_strength",
           "physiochemical_protein", "white_wine", "sulfur"]
TARGETS = WINNERS + LOSERS

def warmup():
    X = np.random.rand(64, 4).astype(np.float32); y = np.random.rand(64)
    YABTRegressor(n_estimators=2, max_leaves=4, device="cpu").fit(X, y)

def gamma_ref():
    try:
        rows = json.load(open("ab_gamma_fullsuite.json"))
        return {r[1]: r[3]["1.0"] - r[3]["0.0"] for r in rows}
    except Exception:
        return {}

def main():
    warmup()
    gref = gamma_ref()
    name2did = {nm: did for _, did, nm, tk in list_datasets() if tk == "reg"}
    print(f"{'dataset':30s} {'base':>8s} {'auto':>8s} {'Δauto':>8s} {'selected':>20s} {'Δgamma1':>9s}")
    print("-" * 90)
    summ = []
    for nm in TARGETS:
        did = name2did.get(nm)
        if did is None:
            print(f"{nm:30s}  (not found)"); continue
        X, y, cat_idx, _ = load_dataset(did, "reg", max_rows=50000, seed=0)
        Xnp = X.to_numpy(np.float32)
        bsc, asc, sel = [], [], []
        for s in SEEDS:
            tr, te = train_test_split(np.arange(len(y)), test_size=0.3, random_state=s)
            mb = YABTRegressor(**BASE)
            mb.fit(Xnp[tr], y[tr], categorical_features=cat_idx or None)
            bsc.append(r2_score(y[te], mb.predict(Xnp[te])))
            ma = YABTRegressor(**BASE, auto_tune=True)
            ma.fit(Xnp[tr], y[tr], categorical_features=cat_idx or None)
            asc.append(r2_score(y[te], ma.predict(Xnp[te])))
            rep = ma.booster_.tuning_report_
            sel.append(rep["selected"] if rep else "skipped")
        b, a = np.mean(bsc), np.mean(asc)
        pick = collections.Counter(sel).most_common(1)[0][0]
        grp = "WIN " if nm in WINNERS else "LOSE"
        print(f"[{grp}] {nm[:24]:24s} {b:+8.4f} {a:+8.4f} {a-b:+8.4f} {pick:>20s} {gref.get(nm, float('nan')):+9.4f}")
        summ.append((nm, nm in WINNERS, b, a, gref.get(nm, float('nan'))))

    print("\n--- did auto_tune get the best of both? ---")
    for label, want in [("gamma-WINNERS (want auto Δ>0)", True), ("gamma-LOSERS (want auto Δ≈0)", False)]:
        rows = [s for s in summ if s[1] == want]
        da = np.mean([s[3] - s[2] for s in rows])
        dg = np.mean([s[4] for s in rows])
        print(f"  {label}: mean Δauto={da:+.4f}   (gamma=1.0 ref mean Δ={dg:+.4f})")

if __name__ == "__main__":
    main()

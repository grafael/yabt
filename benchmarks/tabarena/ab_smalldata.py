"""Small-data A/B harness: the band where YABT's TabArena rank collapses.

Over the TabArena-Lite leaderboard YABT's median rank is 30/78 on the 33
datasets with n >= 2500 and 51/78 on the 17 below it -- the whole Elo gap to
CatBoost's default lives in that band. Single fold-0 fits (what
``ab_tabarena_proxy.py`` does) cannot resolve a candidate there: on 748 rows a
fold is worth several percent of metric error on its own. So this harness fits
every candidate on N_SPLITS outer CV splits per dataset (the task's own
repeat/fold grid) and compares candidates *paired* on the same splits.

Usage:
    python benchmarks/tabarena/ab_smalldata.py baseline cand1 cand2 ...
    python benchmarks/tabarena/ab_smalldata.py --report      # compare stored runs
Configs live in CONFIGS; results append to ab_smalldata_results.json.
"""

from __future__ import annotations

import json
import os
import statistics as st
import sys
import time
from pathlib import Path

import numpy as np
import openml
import pandas as pd
from sklearn.metrics import log_loss, mean_squared_error, roc_auc_score
from sklearn.model_selection import train_test_split

from yabt import YABTClassifier, YABTRegressor

# The n < 2500 band of TabArena-Lite, by OpenML task id.
TASK_IDS = [363612, 363613, 363614, 363615, 363616, 363618, 363619, 363620,
            363621, 363623, 363624, 363625, 363626, 363627, 363628, 363629,
            363630, 363631, 363632, 363671, 363672, 363673, 363674, 363675,
            363676, 363677, 363678, 363679, 363681, 363682, 363683, 363684,
            363685, 363686, 363689, 363691, 363693, 363694, 363696, 363697,
            363698, 363699, 363700, 363702, 363704, 363705, 363706, 363707,
            363708, 363711, 363712]
# The row band and device are env-overridable so the same harness can hold the
# mid-size band (YABT_AB_MIN_ROWS=2500 YABT_AB_MAX_ROWS=20000 YABT_AB_DEVICE=cuda
# YABT_AB_TAG=mid) on the GPU while the small band runs on the CPU.
MIN_ROWS = int(os.environ.get("YABT_AB_MIN_ROWS", 0))
MAX_ROWS = int(os.environ.get("YABT_AB_MAX_ROWS", 2500))
DEVICE = os.environ.get("YABT_AB_DEVICE", "cpu")
TAG = os.environ.get("YABT_AB_TAG", "smalldata")
N_SPLITS = int(os.environ.get("YABT_AB_SPLITS", 6))  # (repeat, fold) pairs per dataset

RESULTS = Path(__file__).parent / f"ab_{TAG}_results.json"
CACHE = Path(__file__).parent / f".{TAG}_cache.npz"

# ``small_data_caps`` is pinned off here: it is on by default now, and below
# _SMALL_N_ROWS it overrides ``max_leaves``, so leaving it unset would make
# every leaf-budget candidate on this band silently identical. Candidates that
# want the shipped rule ask for it explicitly ("caps_v2").
BASE = {"n_estimators": 10_000, "early_stopping_rounds": 50,
        "learning_rate": 0.05, "subsample": 0.9, "colsample": 0.9,
        "cat_count_features": True, "cat_combinations": 16,
        "calibrate_multiclass": True, "small_data_caps": False,
        "device": DEVICE}

CONFIGS: dict[str, dict] = {
    "baseline": {},
    # round 1: where does the small-data loss come from? Capacity, leaf models,
    # early stopping, or subsampling?
    "caps": {"small_data_caps": True},
    "leaves8": {"max_leaves": 8},
    "leaves15": {"max_leaves": 15},
    "depth4": {"max_depth": 4},
    "msl40": {"min_samples_leaf": 40},
    "reg10": {"reg_lambda": 10.0},
    "no_nl": {"neural_leaves": False},
    "es200": {"early_stopping_rounds": 200},
    "nosub": {"subsample": 1.0, "colsample": 1.0},
    "lr02": {"learning_rate": 0.02},
    "ens4": {"n_ensemble": 4},
    # round 2: the categorical stack was tuned on the full 51-dataset proxy,
    # where the big datasets dominate. On 900 rows, 16 pair encodings + a count
    # column per categorical is a lot of target-derived columns.
    "nocomb": {"cat_combinations": 0},
    "nocnt": {"cat_count_features": False},
    "nocat_extras": {"cat_combinations": 0, "cat_count_features": False},
    "smooth50": {"cat_smoothing": 50.0},
    # round 3: capacity is the lever (leaves8: 13/17, mean -2.35%, 0.75x time)
    # and neural leaves *help* on small data (no_nl regressed), i.e. the right
    # small-data model is few regions x a linear model inside each. Push it.
    "leaves4": {"max_leaves": 4},
    "leaves6": {"max_leaves": 6},
    "leaves8_nlmin20": {"max_leaves": 8, "leaf_net_min_samples": 20},
    "leaves8_reg3": {"max_leaves": 8, "reg_lambda": 3.0},
    "leaves8_lr03": {"max_leaves": 8, "learning_rate": 0.03},
    # round 4: out-of-fold (ordered) leaf values, alone and stacked
    "ord4": {"ordered_leaves": 4},
    "ord2": {"ordered_leaves": 2},
    "leaves8_ord4": {"max_leaves": 8, "ordered_leaves": 4},
    # round 5: with neural leaves on, a tree's parameter count is
    # leaves x (leaf_net_features + 1), so the leaf budget is not the only
    # capacity knob -- narrowing the per-leaf model is the other half.
    "nlfeat3": {"leaf_net_features": 3},
    "nlfeat3_leaves15": {"leaf_net_features": 3, "max_leaves": 15},
    "nlfeat3_leaves8": {"leaf_net_features": 3, "max_leaves": 8},
    "nll2_10": {"leaf_net_l2": 10.0},
    "nll2_10_leaves8": {"leaf_net_l2": 10.0, "max_leaves": 8},
    # round 6: leaves4 beat leaves8 (-3.81% vs -1.77% median). How far down?
    "leaves2": {"max_leaves": 2},
    "leaves3": {"max_leaves": 3},
    "leaves5": {"max_leaves": 5},
    "leaves4_nlmin20": {"max_leaves": 4, "leaf_net_min_samples": 20},
    "leaves4_lr03": {"max_leaves": 4, "learning_rate": 0.03},
    "leaves4_ord4": {"max_leaves": 4, "ordered_leaves": 4},
    "leaves4_msl10": {"max_leaves": 4, "min_samples_leaf": 10},
    # round 7: reg_lambda stacks on the capacity cut (leaves8 -1.77% ->
    # leaves8_reg3 -2.90%), so re-test it on the better leaf budget.
    "leaves4_reg3": {"max_leaves": 4, "reg_lambda": 3.0},
    "leaves4_reg10": {"max_leaves": 4, "reg_lambda": 10.0},
    "leaves6_reg3": {"max_leaves": 6, "reg_lambda": 3.0},
    # The shipped cap plus reg_lambda=3 (this is what the mislabelled
    # leaves5/leaves6 rows actually measured before BASE pinned the flag).
    "caps_v2_reg3": {"small_data_caps": True, "reg_lambda": 3.0},
    # mid band (2500-9000 rows) only: the cap that helps there is much looser
    # than the small band's -- leaves4 +0.07%, leaves8 +0.45%, leaves15 -1.18%.
    "leaves20": {"max_leaves": 20},
    "leaves15_reg3": {"max_leaves": 15, "reg_lambda": 3.0},
    # verification: the shipped code path (small_data_caps retuned to 4 leaves
    # + a 20-row leaf-model floor) should reproduce "leaves4_nlmin20". The
    # "caps" row above is the old 16-leaf/depth-4 rule, kept for the record.
    "caps_v2": {"small_data_caps": True},
}


def _splits(task):
    out = []
    for r in range(4):
        for f in range(3):
            if len(out) >= N_SPLITS:
                return out
            try:
                out.append(task.get_train_test_split_indices(repeat=r, fold=f))
            except Exception:
                return out
    return out


def load_small() -> list[dict]:
    """Every TabArena-Lite task under MAX_ROWS rows, as dense float32 arrays."""
    if CACHE.exists():
        z = np.load(CACHE, allow_pickle=True)
        return list(z["data"])
    out = []
    for tid in TASK_IDS:
        task = openml.tasks.get_task(tid, download_splits=True)
        ds = task.get_dataset()
        X, y, cat_mask, _ = ds.get_data(target=task.target_name)
        if not (MIN_ROWS <= X.shape[0] < MAX_ROWS):
            continue
        labels = getattr(task, "class_labels", None)
        if labels is not None:
            problem = "multiclass" if len(labels) > 2 else "binary"
            yn = pd.Categorical(y).codes.astype(np.int64)
        else:
            problem = "regression"
            yn = y.to_numpy(dtype=np.float32)
        Xn = np.empty(X.shape, dtype=np.float32)
        cat_idx = []
        for j, c in enumerate(X.columns):
            col = X[c]
            if cat_mask[j] or col.dtype.name in ("category", "object"):
                Xn[:, j] = pd.Categorical(col).codes.astype(np.float32)
                cat_idx.append(j)
            else:
                Xn[:, j] = pd.to_numeric(col, errors="coerce").to_numpy(dtype=np.float32)
        out.append({"name": ds.name, "problem": problem, "X": Xn, "y": yn,
                    "cat": cat_idx, "splits": _splits(task)})
    np.savez(CACHE, data=np.array(out, dtype=object))
    return out


def metric_error(problem: str, y_true, proba=None, pred=None) -> float:
    if problem == "binary":
        return 1.0 - roc_auc_score(y_true, proba[:, 1])
    if problem == "multiclass":
        return log_loss(y_true, proba, labels=np.arange(proba.shape[1]))
    return float(np.sqrt(mean_squared_error(y_true, pred)))


def run_config(name: str, overrides: dict, data: list[dict]) -> None:
    params = {**BASE, **overrides}
    rows, t_all = {}, time.time()
    for d in data:
        errs, secs = [], []
        for si, (tr, te) in enumerate(d["splits"]):
            Xtr, ytr, Xte, yte = d["X"][tr], d["y"][tr], d["X"][te], d["y"][te]
            strat = ytr if d["problem"] != "regression" else None
            Xf, Xv, yf, yv = train_test_split(Xtr, ytr, test_size=0.125,
                                              random_state=si, stratify=strat)
            t0 = time.time()
            p = {**params, "seed": si}
            if d["problem"] == "regression":
                m = YABTRegressor(**p)
                m.fit(Xf, yf, eval_set=(Xv, yv))
                e = metric_error(d["problem"], yte, pred=m.predict(Xte))
            else:
                m = YABTClassifier(**p)
                m.fit(Xf, yf, eval_set=(Xv, yv),
                      categorical_features=d["cat"] or None)
                e = metric_error(d["problem"], yte, proba=m.predict_proba(Xte))
            errs.append(e)
            secs.append(time.time() - t0)
        rows[d["name"]] = {"problem": d["problem"], "errs": errs,
                           "err": st.mean(errs), "s": st.mean(secs)}
        print(f"[{name}] {d['name']:42s} {d['problem']:10s} "
              f"err={st.mean(errs):.5f} +-{st.stdev(errs) if len(errs)>1 else 0:.5f} "
              f"({st.mean(secs):.1f}s)", flush=True)
    all_res = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    all_res[name] = {"params": params, "datasets": rows, "total_s": time.time() - t_all}
    RESULTS.write_text(json.dumps(all_res, indent=1))
    print(f"[{name}] done in {time.time()-t_all:.0f}s", flush=True)


def report(base: str = "baseline") -> None:
    """Paired per-split comparison of every stored config against ``base``."""
    res = json.loads(RESULTS.read_text())
    b = res[base]["datasets"]
    for name, r in res.items():
        if name == base:
            continue
        gaps, wins, n = [], 0, 0
        for ds, row in r["datasets"].items():
            if ds not in b:
                continue
            g = (row["err"] - b[ds]["err"]) / max(abs(b[ds]["err"]), 1e-12)
            gaps.append(g)
            wins += g < 0
            n += 1
        t = r["total_s"] / max(res[base]["total_s"], 1e-9)
        print(f"{name:24s} vs {base}: wins {wins}/{n}  "
              f"median {st.median(gaps):+.2%}  mean {st.mean(gaps):+.2%}  time x{t:.2f}")
        worst = sorted(((( row['err'] - b[ds]['err'])/max(abs(b[ds]['err']),1e-12), ds)
                        for ds, row in r["datasets"].items() if ds in b), reverse=True)
        print("   worst:", ", ".join(f"{d} {g:+.1%}" for g, d in worst[:3]))
        print("   best :", ", ".join(f"{d} {g:+.1%}" for g, d in worst[-3:]))


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "--report":
        report(*args[1:])
    else:
        data = load_small()
        print(f"{len(data)} datasets in [{MIN_ROWS}, {MAX_ROWS}) rows, "
              f"{len(data[0]['splits'])} splits each", flush=True)
        for cfg in args:
            run_config(cfg, CONFIGS[cfg], data)

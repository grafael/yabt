"""Fast TabArena proxy: single (unbagged) fold-0 fits over all 51 cached
TabArena-Lite tasks, for A/B-ing candidate YABT default configs against the
current defaults before paying for a full bagged TabArena re-run.

Metric follows TabArena's metric_error: binary = 1 - ROC AUC, multiclass =
log loss, regression = RMSE. Each fit gets a stratified 12.5% early-stopping
holdout carved from the fold-0 train split (mirroring one bag fold).

Usage:
    python benchmarks/tabarena/ab_tabarena_proxy.py baseline lr05 lr03 ...
Named configs live in CONFIGS below; results append to
ab_tabarena_proxy_results.json (next to this script) keyed by config name.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import openml
import pandas as pd
from sklearn.metrics import log_loss, mean_squared_error, roc_auc_score
from sklearn.model_selection import train_test_split

from yabt import YABTClassifier, YABTRegressor

TASK_IDS = [363612, 363613, 363614, 363615, 363616, 363618, 363619, 363620,
            363621, 363623, 363624, 363625, 363626, 363627, 363628, 363629,
            363630, 363631, 363632, 363671, 363672, 363673, 363674, 363675,
            363676, 363677, 363678, 363679, 363681, 363682, 363683, 363684,
            363685, 363686, 363689, 363691, 363693, 363694, 363696, 363697,
            363698, 363699, 363700, 363702, 363704, 363705, 363706, 363707,
            363708, 363711, 363712]

RESULTS = Path(__file__).parent / "ab_tabarena_proxy_results.json"

# Shared protocol bits: high tree cap + early stopping on the holdout, like the
# TabArena wrapper. Candidates override/extend these.
BASE = {"n_estimators": 10_000, "early_stopping_rounds": 50, "seed": 0}

CONFIGS: dict[str, dict] = {
    "baseline": {},
    "lr05": {"learning_rate": 0.05},
    "lr03": {"learning_rate": 0.03},
    "lr05_reg3": {"learning_rate": 0.05, "reg_lambda": 3.0},
    "lr05_leaves63": {"learning_rate": 0.05, "max_leaves": 63},
    "lr05_msl5": {"learning_rate": 0.05, "min_samples_leaf": 5},
    "lr05_col09": {"learning_rate": 0.05, "colsample": 0.9, "subsample": 0.9},
    "autotune": {"auto_tune": True},
    "lr05_es100": {"learning_rate": 0.05, "early_stopping_rounds": 100},
    # categorical-handling candidates (only move cat / multiclass datasets)
    "catpc": {"cat_per_class": True},
    "catcnt": {"cat_count_features": True},
    "catpc_cnt": {"cat_per_class": True, "cat_count_features": True},
    # round 2: stack the round-1 winners
    "combo_core": {"learning_rate": 0.05, "colsample": 0.9, "subsample": 0.9,
                   "reg_lambda": 3.0},
    "combo_cat": {"learning_rate": 0.05, "colsample": 0.9, "subsample": 0.9,
                  "cat_count_features": True},
    "combo_all": {"learning_rate": 0.05, "colsample": 0.9, "subsample": 0.9,
                  "reg_lambda": 3.0, "cat_count_features": True,
                  "cat_per_class": True},
    # round 3: CatBoost-style pair combinations on top of the shipped default
    # (combo_cat). combo_cat re-run under the fixed problem-type detection so
    # round-3 rows compare within the same protocol.
    "r3_base": {"learning_rate": 0.05, "colsample": 0.9, "subsample": 0.9,
                "cat_count_features": True},
    "r3_comb16": {"learning_rate": 0.05, "colsample": 0.9, "subsample": 0.9,
                  "cat_count_features": True, "cat_combinations": 16},
    "r3_comb32": {"learning_rate": 0.05, "colsample": 0.9, "subsample": 0.9,
                  "cat_count_features": True, "cat_combinations": 32},
    "r3_comb16_pc": {"learning_rate": 0.05, "colsample": 0.9, "subsample": 0.9,
                     "cat_count_features": True, "cat_combinations": 16,
                     "cat_per_class": True},
    "r3_reg3": {"learning_rate": 0.05, "colsample": 0.9, "subsample": 0.9,
                "cat_count_features": True, "reg_lambda": 3.0},
    # round 4: pair combinations gated to high-cardinality parents (>= 8 levels)
    "r4_comb16hc": {"learning_rate": 0.05, "colsample": 0.9, "subsample": 0.9,
                    "cat_count_features": True, "cat_combinations": 16},
    # round 5: TabFM-inspired additions on top of the shipped stack.
    # r5_ship re-measures the shipped stack (the pair guard changed since r4).
    "r5_ship": {"learning_rate": 0.05, "colsample": 0.9, "subsample": 0.9,
                "cat_count_features": True, "cat_combinations": 16},
    "r5_tabfm": {"learning_rate": 0.05, "colsample": 0.9, "subsample": 0.9,
                 "cat_count_features": True, "cat_combinations": 16,
                 "calibrate_multiclass": True, "svd_features": 8},
    # round 6: identical params to r5_ship, re-measured after the missing-value
    # rework (NaN gets its own bin instead of being median-imputed) and the
    # mask-aware histogram build. Same name would overwrite the r5 row, so this
    # is the paired before/after: r5_ship = old code, r6_nanbin = new code.
    "r6_nanbin": {"learning_rate": 0.05, "colsample": 0.9, "subsample": 0.9,
                  "cat_count_features": True, "cat_combinations": 16},
}


def load_task(tid: int):
    task = openml.tasks.get_task(tid, download_splits=True)
    dataset = task.get_dataset()
    X, y, cat_mask, _ = dataset.get_data(target=task.target_name)
    tr_idx, te_idx = task.get_train_test_split_indices(repeat=0, fold=0)
    # Problem type from the task, not the dtype (binary targets stored as
    # numeric 0/1 must not be treated as regression).
    if getattr(task, "class_labels", None) is not None:
        problem = "multiclass" if len(task.class_labels) > 2 else "binary"
        yn = pd.Categorical(y).codes.astype(np.int64)
    else:
        problem = "regression"
        yn = y.to_numpy(dtype=np.float32)
    out = np.empty(X.shape, dtype=np.float32)
    cat_idx = []
    for j, c in enumerate(X.columns):
        col = X[c]
        if cat_mask[j] or col.dtype.name in ("category", "object"):
            out[:, j] = pd.Categorical(col).codes.astype(np.float32)
            cat_idx.append(j)
        else:
            out[:, j] = pd.to_numeric(col, errors="coerce").to_numpy(dtype=np.float32)
    return dataset.name, problem, out, yn, cat_idx, tr_idx, te_idx


def metric_error(problem: str, y_true, model) -> float:
    if problem == "binary":
        return 1.0 - roc_auc_score(y_true, model.predict_proba_cache[:, 1])
    if problem == "multiclass":
        k = model.predict_proba_cache.shape[1]
        return log_loss(y_true, model.predict_proba_cache, labels=np.arange(k))
    return float(np.sqrt(mean_squared_error(y_true, model.predict_cache)))


def run_config(name: str, overrides: dict) -> None:
    params = {**BASE, **overrides}
    rows = {}
    t_all = time.time()
    for tid in TASK_IDS:
        name_ds, problem, Xn, yn, cat_idx, tr_idx, te_idx = load_task(tid)
        Xtr, ytr = Xn[tr_idx], yn[tr_idx]
        Xte, yte = Xn[te_idx], yn[te_idx]
        strat = ytr if problem != "regression" else None
        Xfit, Xval, yfit, yval = train_test_split(
            Xtr, ytr, test_size=0.125, random_state=0, stratify=strat)
        t0 = time.time()
        if problem == "regression":
            model = YABTRegressor(**params)
            model.fit(Xfit, yfit, eval_set=(Xval, yval))
            model.predict_cache = model.predict(Xte)
        else:
            model = YABTClassifier(**params)
            model.fit(Xfit, yfit, eval_set=(Xval, yval),
                      categorical_features=cat_idx or None)
            model.predict_proba_cache = model.predict_proba(Xte)
        err = metric_error(problem, yte, model)
        rows[name_ds] = {"problem": problem, "err": err, "s": time.time() - t0}
        print(f"[{name}] {name_ds:45s} {problem:10s} err={err:.5f} ({rows[name_ds]['s']:.0f}s)",
              flush=True)

    all_res = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    all_res[name] = {"params": params, "datasets": rows,
                     "total_s": time.time() - t_all}
    RESULTS.write_text(json.dumps(all_res, indent=1))
    print(f"[{name}] done in {time.time()-t_all:.0f}s -> {RESULTS}", flush=True)


if __name__ == "__main__":
    for cfg in sys.argv[1:]:
        run_config(cfg, CONFIGS[cfg])

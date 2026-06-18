#!/usr/bin/env python
"""Standard tabular GBM benchmark on the Grinsztajn et al. (2022) OpenML suites.

Compares YABT against XGBoost, LightGBM and CatBoost with default hyper-parameters
(100 trees, lr 0.1, depth 6) over several random splits per dataset. Each model
gets native categorical handling, all models run on the same device, and scores
are reported as mean +/- std across seeds.

Examples
--------
    python openml_benchmark.py --suite num_reg --seeds 3
    python openml_benchmark.py --suite num_clf --max-rows 20000 --max-datasets 6
    python openml_benchmark.py --suite all                       # everything (slow)
    python openml_benchmark.py --list                            # show datasets
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split

from datasets import SUITES, list_datasets, load_dataset
from yabt import YABTClassifier, YABTRegressor

warnings.filterwarnings("ignore")

try:
    import torch
    HAS_CUDA = torch.cuda.is_available()
except Exception:
    HAS_CUDA = False

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
try:
    import catboost as cat
    HAS_CAT = True
except ImportError:
    HAS_CAT = False
try:
    from sklearn.ensemble import (
        HistGradientBoostingClassifier, HistGradientBoostingRegressor)
    HAS_HGB = True
except ImportError:
    HAS_HGB = False

NTREES, LR, DEPTH, LEAVES = 100, 0.1, 6, 31


def _cat_frame(X: pd.DataFrame, cat_idx, as_int=False) -> pd.DataFrame:
    """Return a copy with categorical columns typed for native handling."""
    if not cat_idx:
        return X
    Xc = X.copy()
    for i in cat_idx:
        col = Xc.columns[i]
        # codes are integer-valued floats; go via int so category labels are
        # integers (XGBoost rejects float-typed categories).
        Xc[col] = Xc[col].astype("int64")
        if not as_int:
            Xc[col] = Xc[col].astype("category")
    return Xc


# Each builder takes device in {"cpu", "gpu"} and returns (fit_fn, predict_fn).
# fit_fn(Xtr, ytr) -> fitted model; predict_fn(model, Xte) -> labels (clf) or values (reg).
def _yabt(task, device, cat_idx):
    Est = YABTClassifier if task == "clf" else YABTRegressor
    dev = "cuda" if device == "gpu" else "cpu"
    def fit(Xtr, ytr):
        m = Est(n_estimators=NTREES, learning_rate=LR, max_leaves=LEAVES, device=dev)
        m.fit(Xtr.to_numpy(np.float32), ytr, categorical_features=cat_idx or None)
        return m
    def pred(m, Xte):
        return m.predict(Xte.to_numpy(np.float32))
    return fit, pred


def _xgb(task, device, cat_idx):
    Est = xgb.XGBClassifier if task == "clf" else xgb.XGBRegressor
    dev = "cuda" if device == "gpu" else "cpu"
    def fit(Xtr, ytr):
        m = Est(n_estimators=NTREES, learning_rate=LR, max_depth=DEPTH, device=dev,
                enable_categorical=bool(cat_idx), tree_method="hist",
                random_state=0, verbosity=0)
        m.fit(_cat_frame(Xtr, cat_idx), ytr)
        return m
    def pred(m, Xte):
        return m.predict(_cat_frame(Xte, cat_idx))
    return fit, pred


def _lgb(task, device, cat_idx):
    Est = lgb.LGBMClassifier if task == "clf" else lgb.LGBMRegressor
    dev = "gpu" if device == "gpu" else "cpu"
    cat_cols = [int(i) for i in cat_idx] if cat_idx else "auto"
    def fit(Xtr, ytr):
        m = Est(n_estimators=NTREES, learning_rate=LR, max_depth=DEPTH, num_leaves=LEAVES,
                device=dev, random_state=0, verbosity=-1)
        m.fit(_cat_frame(Xtr, cat_idx), ytr, categorical_feature=cat_cols)
        return m
    def pred(m, Xte):
        return m.predict(_cat_frame(Xte, cat_idx))
    return fit, pred


def _cat(task, device, cat_idx):
    Est = cat.CatBoostClassifier if task == "clf" else cat.CatBoostRegressor
    task_type = "GPU" if device == "gpu" else "CPU"
    def fit(Xtr, ytr):
        m = Est(iterations=NTREES, learning_rate=LR, depth=DEPTH, task_type=task_type,
                cat_features=list(cat_idx) or None, random_state=0, verbose=0)
        m.fit(_cat_frame(Xtr, cat_idx, as_int=True), ytr)
        return m
    def pred(m, Xte):
        out = m.predict(_cat_frame(Xte, cat_idx, as_int=True))
        return np.asarray(out).ravel()
    return fit, pred


def _hgb(task, device, cat_idx):
    # scikit-learn HistGradientBoosting: CPU-only (ignores `device`), native
    # categorical handling and missing-value support. High-cardinality
    # categoricals (> max_bins) raise at fit and are caught per-model upstream.
    Est = HistGradientBoostingClassifier if task == "clf" else HistGradientBoostingRegressor
    cat_feat = [int(i) for i in cat_idx] if cat_idx else None
    def fit(Xtr, ytr):
        m = Est(max_iter=NTREES, learning_rate=LR, max_depth=DEPTH,
                max_leaf_nodes=LEAVES, categorical_features=cat_feat, random_state=0)
        m.fit(_cat_frame(Xtr, cat_idx, as_int=True), ytr)
        return m
    def pred(m, Xte):
        return m.predict(_cat_frame(Xte, cat_idx, as_int=True))
    return fit, pred


def model_builders():
    b = {"YABT": _yabt}
    if HAS_XGB:
        b["XGBoost"] = _xgb
    if HAS_LGB:
        b["LightGBM"] = _lgb
    if HAS_CAT:
        b["CatBoost"] = _cat
    if HAS_HGB:
        b["HistGBM"] = _hgb
    return b


def score(task, y_true, y_pred):
    if task == "clf":
        return accuracy_score(y_true, np.asarray(y_pred).ravel())
    return r2_score(y_true, y_pred)


def run_dataset(tag, did, name, task, seeds, max_rows, devices):
    """Return {model: {device: {scores: [...], times: [...]}}} for one dataset."""
    X, y, cat_idx, _ = load_dataset(did, task, max_rows=max_rows, seed=0)
    builders = model_builders()
    out = {m: {dev: {"scores": [], "times": []} for dev in devices} for m in builders}

    for seed in seeds:
        strat = y if task == "clf" else None
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=0.3, random_state=seed, stratify=strat)
        Xtr = Xtr.reset_index(drop=True)
        Xte = Xte.reset_index(drop=True)
        for dev in devices:
            for mname, builder in builders.items():
                fit, pred = builder(task, dev, cat_idx)
                try:
                    t0 = time.time()
                    model = fit(Xtr, ytr)
                    dt = time.time() - t0
                    sc = score(task, yte, pred(model, Xte))
                    out[mname][dev]["scores"].append(float(sc))
                    out[mname][dev]["times"].append(float(dt))
                except Exception as e:
                    print(f"      {mname}@{dev} FAILED on {name}: {str(e)[:70]}")
    return out, X.shape, cat_idx


def aggregate(scores):
    s = np.asarray(scores, dtype=float)
    return (float(s.mean()), float(s.std())) if len(s) else (float("nan"), float("nan"))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", default="num_reg",
                    choices=list(SUITES) + ["all"], help="benchmark suite (default num_reg)")
    ap.add_argument("--seeds", type=int, default=3, help="random splits per dataset")
    ap.add_argument("--max-rows", type=int, default=50000,
                    help="subsample cap per dataset (Grinsztajn medium regime)")
    ap.add_argument("--max-datasets", type=int, default=None,
                    help="limit number of datasets (smallest first)")
    ap.add_argument("--device", default="cpu", choices=["cpu", "gpu", "both"],
                    help="device for all models; 'both' times each on CPU and GPU")
    ap.add_argument("--list", action="store_true", help="list datasets and exit")
    ap.add_argument("--out", default=None, help="output JSON path")
    args = ap.parse_args()

    tags = list(SUITES) if args.suite == "all" else [args.suite]

    if args.list:
        for t in tags:
            print(f"\n{t} (suite {SUITES[t][0]}):")
            for _, did, nm, tk in list_datasets(t):
                print(f"  did={did:6d}  {tk}  {nm}")
        return

    devices = ["cpu", "gpu"] if args.device == "both" else [args.device]
    if "gpu" in devices and not HAS_CUDA:
        print("⚠️  GPU requested but CUDA not available; dropping GPU.")
        devices = [d for d in devices if d != "gpu"] or ["cpu"]

    seeds = list(range(args.seeds))
    print(f"Models: {', '.join(model_builders())}")
    print(f"Config: {NTREES} trees, lr={LR}, depth={DEPTH} | seeds={seeds} | "
          f"max_rows={args.max_rows} | devices={', '.join(devices)}\n")

    results = {}
    for tag in tags:
        datasets = list_datasets(tag)
        if args.max_datasets:
            datasets = datasets[:args.max_datasets]
        metric = "accuracy" if SUITES[tag][1] == "clf" else "R²"
        print(f"{'='*70}\nSUITE {tag}  (metric: {metric})\n{'='*70}")
        for _, did, name, task in datasets:
            print(f"  • {name} (did={did}) ...", flush=True)
            try:
                per_model, shape, cat_idx = run_dataset(
                    tag, did, name, task, seeds, args.max_rows, devices)
            except Exception as e:
                print(f"      load FAILED: {str(e)[:80]}")
                continue
            primary = devices[0]
            row = {"n": shape[0], "p": shape[1], "n_cat": len(cat_idx), "models": {}}
            ranked = []
            for m, dd in per_model.items():
                row["models"][m] = {}
                for dev in devices:
                    mean, std = aggregate(dd[dev]["scores"])
                    tmean, _ = aggregate(dd[dev]["times"])
                    row["models"][m][dev] = {"mean": mean, "std": std, "time": tmean}
                ranked.append((m, row["models"][m][primary]["mean"]))
            results[f"{tag}:{name}"] = row
            ranked.sort(key=lambda r: (-r[1] if not np.isnan(r[1]) else 1))
            for i, (m, _) in enumerate(ranked):
                mark = "★" if i == 0 else " "
                parts = []
                for dev in devices:
                    d = row["models"][m][dev]
                    parts.append(f"{dev} {d['mean']:.4f}±{d['std']:.4f} {d['time']:.1f}s")
                print(f"      {mark} {m:<9} " + " | ".join(parts))
        print()

    _print_summary(results, tags, devices)

    out_path = Path(args.out) if args.out else Path(__file__).parent / "openml_benchmark_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\n✅ Results saved to {out_path}")


def _print_summary(results, tags, devices):
    """Per-suite mean score, avg rank, and per-device timing across datasets."""
    primary = devices[0]
    print(f"{'='*70}\nSUMMARY (mean score & rank by {primary}, time per device)\n{'='*70}")
    for tag in tags:
        rows = {k: v for k, v in results.items() if k.startswith(f"{tag}:")}
        if not rows:
            continue
        scores, ranks = {}, {}
        times = {dev: {} for dev in devices}
        for _, row in rows.items():
            valid = {m: row["models"][m][primary]["mean"]
                     for m in row["models"] if not np.isnan(row["models"][m][primary]["mean"])}
            order = sorted(valid, key=valid.get, reverse=True)
            for m, mdata in row["models"].items():
                if m in valid:
                    scores.setdefault(m, []).append(valid[m])
                    ranks.setdefault(m, []).append(order.index(m) + 1)
                for dev in devices:
                    t = mdata[dev]["time"]
                    if not np.isnan(t):
                        times[dev].setdefault(m, []).append(t)
        metric = "accuracy" if SUITES[tag][1] == "clf" else "R²"
        time_cols = "".join(f"{dev+'_time':>10}" for dev in devices)
        print(f"\n{tag} ({metric}, {len(rows)} datasets):")
        print(f"  {'model':<9}{'score':>8}{'rank':>7}{time_cols}")
        for m in sorted(scores, key=lambda x: np.mean(ranks[x])):
            tcells = "".join(
                f"{(np.mean(times[dev][m]) if times[dev].get(m) else float('nan')):>9.1f}s"
                for dev in devices)
            print(f"  {m:<9}{np.mean(scores[m]):>8.4f}{np.mean(ranks[m]):>7.2f}{tcells}")


if __name__ == "__main__":
    main()

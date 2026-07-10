"""A/B: native softmax multiclass vs One-vs-Rest on the TabArena multiclass tasks.

Uses the TabArena-Lite fold-0 train/test split of each multiclass task (all
cached locally by the TabArena run), an inner stratified 87.5/12.5 holdout for
early stopping (mirroring one bag fold of the TabArena protocol), and compares
test multiclass log loss and wall-clock fit time. XGBoost's official TabArena
default (bagged) scores are printed as a reference column.

Run with the tabarena venv (openml + yabt installed):
    /home/rafael/git/tabarena/.venv/bin/python benchmarks/tabarena/ab_softmax_multiclass.py
"""

from __future__ import annotations

import time

import numpy as np
import openml
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split

from yabt import YABTClassifier

TASK_IDS = [363612, 363613, 363614, 363615, 363616, 363618, 363619, 363620,
            363621, 363623, 363624, 363625, 363626, 363627, 363628, 363629,
            363630, 363631, 363632, 363671, 363672, 363673, 363674, 363675,
            363676, 363677, 363678, 363679, 363681, 363682, 363683, 363684,
            363685, 363686, 363689, 363691, 363693, 363694, 363696, 363697,
            363698, 363699, 363700, 363702, 363704, 363705, 363706, 363707,
            363708, 363711, 363712]


def load_task(tid: int):
    task = openml.tasks.get_task(tid, download_splits=True)
    dataset = task.get_dataset()
    X, y, cat_mask, _ = dataset.get_data(target=task.target_name)
    if y.dtype.name != "category" or len(y.unique()) <= 2:
        return None
    tr_idx, te_idx = task.get_train_test_split_indices(repeat=0, fold=0)
    return dataset.name, X, y, cat_mask, tr_idx, te_idx


def to_numeric(X: pd.DataFrame, cat_mask: list[bool]):
    """Categoricals -> integer codes (like the TabArena wrapper); returns
    (float32 matrix, categorical column indices)."""
    out = np.empty(X.shape, dtype=np.float32)
    cat_idx = []
    for j, c in enumerate(X.columns):
        col = X[c]
        if cat_mask[j] or col.dtype.name in ("category", "object"):
            out[:, j] = pd.Categorical(col).codes.astype(np.float32)
            cat_idx.append(j)
        else:
            out[:, j] = pd.to_numeric(col, errors="coerce").to_numpy(dtype=np.float32)
    return out, cat_idx


def main() -> None:
    rows = []
    for tid in TASK_IDS:
        try:
            loaded = load_task(tid)
        except Exception as e:
            print(f"skip {tid}: {e}")
            continue
        if loaded is None:
            continue
        name, X, y, cat_mask, tr_idx, te_idx = loaded
        Xn, cat_idx = to_numeric(X, cat_mask)
        yn = pd.Categorical(y).codes.astype(np.int64)
        Xtr, ytr = Xn[tr_idx], yn[tr_idx]
        Xte, yte = Xn[te_idx], yn[te_idx]
        Xfit, Xval, yfit, yval = train_test_split(
            Xtr, ytr, test_size=0.125, random_state=0, stratify=ytr)

        row = {"dataset": name, "n": len(tr_idx), "F": Xn.shape[1],
               "K": len(np.unique(yn))}
        for mode in ["ovr", "softmax"]:
            t0 = time.time()
            clf = YABTClassifier(multiclass=mode, n_estimators=10_000,
                                 early_stopping_rounds=50, seed=0)
            clf.fit(Xfit, yfit, eval_set=(Xval, yval),
                    categorical_features=cat_idx or None)
            row[f"{mode}_s"] = time.time() - t0
            proba = clf.predict_proba(Xte)
            row[f"{mode}_ll"] = log_loss(yte, proba, labels=np.arange(row["K"]))
        row["delta_ll"] = row["softmax_ll"] - row["ovr_ll"]
        row["speedup"] = row["ovr_s"] / row["softmax_s"]
        rows.append(row)
        print(f"{name:45s} K={row['K']:2d} ovr={row['ovr_ll']:.4f} ({row['ovr_s']:.0f}s) "
              f"softmax={row['softmax_ll']:.4f} ({row['softmax_s']:.0f}s) "
              f"delta={row['delta_ll']:+.4f} speedup={row['speedup']:.1f}x")

    df = pd.DataFrame(rows)
    print("\n=== summary ===")
    print(df.to_string(index=False))
    wins = int((df["delta_ll"] < 0).sum())
    print(f"\nsoftmax better log loss on {wins}/{len(df)} datasets; "
          f"mean delta {df['delta_ll'].mean():+.4f}; "
          f"median speedup {df['speedup'].median():.2f}x")


if __name__ == "__main__":
    main()

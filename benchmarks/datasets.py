"""OpenML loaders for the Grinsztajn et al. (2022) tabular benchmark suites.

These are the official OpenML benchmark suites used by "Why do tree-based models
still outperform deep learning on tabular data?" (NeurIPS 2022):

    337  Tabular benchmark numerical classification   (16 datasets)
    336  Tabular benchmark numerical regression       (19 datasets)
    334  Tabular benchmark categorical classification ( 7 datasets)
    335  Tabular benchmark categorical regression     (17 datasets)

The dataset ids below are resolved from those suites. They are hard-coded so a
benchmark run does not depend on the OpenML study endpoint being reachable; use
``refresh_suites()`` to re-resolve them if the suites change upstream.
"""

from __future__ import annotations

import numpy as np
import openml
import pandas as pd

openml.config.set_root_cache_directory("benchmarks/data_cache")

# suite tag -> (OpenML suite id, task string). Order roughly by dataset size.
SUITES = {
    "num_clf": (337, "clf"),
    "num_reg": (336, "reg"),
    "cat_clf": (334, "clf"),
    "cat_reg": (335, "reg"),
}

# suite tag -> list of (openml dataset id, friendly name, n_instances).
# Resolved from the suites above (see refresh_suites). Sorted small -> large so
# that capped runs cover the cheap datasets first.
SUITE_DATASETS: dict[str, list[tuple[int, str, int]]] = {
    "num_clf": [
        (45019, "Bioresponse", 3434),
        (44130, "eye_movements", 7608),
        (45026, "heloc", 10000),
        (44122, "pol", 10082),
        (44126, "bank-marketing", 10578),
        (45020, "default-of-credit-card-clients", 13272),
        (44125, "MagicTelescope", 13376),
        (44123, "house_16H", 13488),
        (44089, "credit", 16714),
        (45028, "california", 20634),
        (44120, "electricity", 38474),
        (45021, "jannis", 57580),
        (45022, "Diabetes130US", 71090),
        (44128, "MiniBooNE", 72998),
        (44121, "covertype", 566602),
        (44129, "Higgs", 940160),
    ],
    "num_reg": [
        (45033, "abalone", 4177),
        (44136, "wine_quality", 6497),
        (44132, "cpu_act", 8192),
        (45032, "yprop_4_1", 8885),
        (44145, "sulfur", 10081),
        (44141, "Brazilian_houses", 10692),
        (44137, "Ailerons", 13750),
        (44147, "MiamiHousing2016", 13932),
        (44133, "pol", 15000),
        (44134, "elevators", 16599),
        (44142, "Bike_Sharing_Demand", 17379),
        (44138, "houses", 20640),
        (44148, "superconduct", 21263),
        (44144, "house_sales", 21613),
        (44139, "house_16H", 22784),
        (44140, "diamonds", 53940),
        (44146, "medical_charges", 163065),
        (44143, "nyc-taxi-green-dec-2016", 581835),
        (45034, "delays_zurich_transport", 5465575),
    ],
    "cat_clf": [
        (45039, "compas-two-years", 4966),
        (44157, "eye_movements", 7608),
        (45036, "default-of-credit-card-clients", 13272),
        (44156, "electricity", 38474),
        (45035, "albert", 58252),
        (45038, "road-safety", 111762),
        (44159, "covertype", 423680),
    ],
    "cat_reg": [
        (44055, "analcatdata_supreme", 4052),
        (45042, "abalone", 4177),
        (44061, "Mercedes_Benz_Greener_Manufacturing", 4209),
        (44056, "visualizing_soil", 8641),
        (45041, "topo_2_1", 8885),
        (44062, "Brazilian_houses", 10692),
        (44063, "Bike_Sharing_Demand", 17379),
        (44066, "house_sales", 21613),
        (45043, "seattlecrime6", 52031),
        (44059, "diamonds", 53940),
        (45048, "medical_charges", 163065),
        (45046, "Allstate_Claims_Severity", 188318),
        (44069, "SGEMM_GPU_kernel_performance", 241600),
        (44068, "particulate-matter-ukair-2017", 394299),
        (44065, "nyc-taxi-green-dec-2016", 581835),
        (45047, "Airlines_DepDelay_1M", 1000000),
        (45045, "delays_zurich_transport", 5465575),
    ],
}


def refresh_suites() -> dict[str, list[tuple[int, str, int]]]:
    """Re-resolve SUITE_DATASETS from the live OpenML study endpoint.

    Returns the freshly resolved mapping (also handy for printing/pasting back
    into the literal above). Requires network access.
    """
    out: dict[str, list[tuple[int, str, int]]] = {}
    for tag, (sid, _task) in SUITES.items():
        suite = openml.study.get_suite(sid)
        tasks = openml.tasks.list_tasks(task_id=suite.tasks, output_format="dataframe")
        rows = [
            (int(r["did"]), str(r["name"]), int(r.get("NumberOfInstances", 0)))
            for _, r in tasks.iterrows()
        ]
        out[tag] = sorted(rows, key=lambda x: x[2])
    return out


def list_datasets(suite: str | None = None) -> list[tuple[str, int, str, str]]:
    """Return (suite_tag, dataset_id, name, task) for one or all suites."""
    tags = [suite] if suite else list(SUITES)
    out = []
    for tag in tags:
        task = SUITES[tag][1]
        for did, name, _n in SUITE_DATASETS[tag]:
            out.append((tag, did, name, task))
    return out


def load_dataset(dataset_id: int, task: str, *, max_rows: int | None = None, seed: int = 0):
    """Load one OpenML dataset by id.

    Returns ``(X, y, cat_idx, task)`` where:
      * ``X`` is a float32 DataFrame (categoricals label-encoded to integer codes,
        column names preserved),
      * ``y`` is int64 class codes (clf) or float32 (reg),
      * ``cat_idx`` is the list of categorical column indices,
      * ``task`` echoes the task string.

    Large datasets are uniformly subsampled to ``max_rows`` (Grinsztajn-style
    "medium" regime) using ``seed`` so every model sees identical rows.
    """
    ds = openml.datasets.get_dataset(
        dataset_id, download_data=True, download_qualities=False,
        download_features_meta_data=True,
    )
    X, y, cat_mask, _ = ds.get_data(target=ds.default_target_attribute, dataset_format="dataframe")

    # drop rows with missing target
    keep = ~pd.isna(y)
    X, y = X[keep], y[keep]

    cat_idx = [i for i, c in enumerate(cat_mask) if c]
    Xn = np.empty(X.shape, dtype=np.float32)
    for i, col in enumerate(X.columns):
        s = X[col]
        if i in cat_idx or s.dtype.name in ("category", "object"):
            codes = s.astype("category").cat.codes.to_numpy()
            # cat.codes uses -1 for missing; remap to a dedicated non-negative
            # bucket so codes stay contiguous 0..k (XGBoost rejects negatives).
            if (codes < 0).any():
                codes = codes.copy()
                codes[codes < 0] = codes.max() + 1
            Xn[:, i] = codes.astype(np.float32)
            if i not in cat_idx:
                cat_idx.append(i)
        else:
            Xn[:, i] = pd.to_numeric(s, errors="coerce").to_numpy(dtype=np.float32)
    cat_idx = sorted(set(cat_idx))

    # impute remaining NaNs (e.g. coerced numerics) with per-column means
    col_means = np.nanmean(Xn, axis=0)
    col_means = np.where(np.isnan(col_means), 0.0, col_means)
    nan_mask = np.isnan(Xn)
    if nan_mask.any():
        Xn[nan_mask] = np.take(col_means, np.where(nan_mask)[1])

    Xdf = pd.DataFrame(Xn, columns=[str(c) for c in X.columns], dtype=np.float32)

    if task == "clf":
        yv = pd.Series(y).astype("category").cat.codes.to_numpy().astype(np.int64)
        if len(np.unique(yv)) != 2:
            # binarize most-frequent vs rest (Grinsztajn convention)
            top = np.bincount(yv).argmax()
            yv = (yv == top).astype(np.int64)
    else:
        yv = pd.to_numeric(pd.Series(y), errors="coerce").to_numpy(dtype=np.float32)
        finite = np.isfinite(yv)
        if not finite.all():
            Xdf, yv = Xdf[finite].reset_index(drop=True), yv[finite]

    if max_rows and len(yv) > max_rows:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(len(yv), max_rows, replace=False))
        Xdf, yv = Xdf.iloc[idx].reset_index(drop=True), yv[idx]

    return Xdf, yv, cat_idx, task

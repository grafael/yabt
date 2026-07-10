"""YABT wrapped as an AutoGluon/TabArena model.

Lives in its own importable module (not the run script) because TabArena's
Ray-backed runs pickle the model class; only ``debug_mode=True`` tolerates
``__main__`` classes.

Preprocessing contract: AutoGluon hands us int/float/category columns
(``valid_raw_types``). We convert category columns to integer codes and pass
their column indices to YABT's ``categorical_features``, which target-encodes
them leakage-free (PermutationTargetEncoder); NaNs in numeric columns are left
in place — YABT's binner median-imputes them.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np
from autogluon.common.utils.resource_utils import ResourceManager
from autogluon.core.models import AbstractModel

if TYPE_CHECKING:
    import pandas as pd

    from tabarena.utils.config_utils import ConfigGenerator


class YABTModel(AbstractModel):
    """YABT (https://github.com/grafael/yabt) as a TabArena model."""

    ag_key = "YABT"
    ag_name = "YABT"
    seed_name = "seed"  # AutoGluon injects the framework seed here

    def _preprocess(self, X: pd.DataFrame, is_train: bool = False, **kwargs) -> np.ndarray:
        X = super()._preprocess(X, **kwargs)
        if is_train:
            cols = list(X.columns)
            self._cat_idx = [
                cols.index(c) for c in X.select_dtypes(include="category").columns
            ]
            self._cat_cols = [cols[i] for i in self._cat_idx]
        out = np.empty(X.shape, dtype=np.float32)
        for j, c in enumerate(X.columns):
            col = X[c]
            if c in self._cat_cols:
                # codes: NaN -> -1, which the target encoder treats as its own
                # category (a dedicated "missing" level).
                out[:, j] = col.cat.codes.to_numpy(dtype=np.float32)
            else:
                out[:, j] = col.to_numpy(dtype=np.float32, na_value=np.nan)
        return out

    def _fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
        time_limit: float | None = None,
        num_cpus: int = 1,
        num_gpus: float = 0,
        **kwargs,
    ) -> None:
        start = time.time()
        if self.problem_type == "regression":
            from yabt import YABTRegressor

            model_cls = YABTRegressor
        else:  # "binary" / "multiclass"
            from yabt import YABTClassifier

            model_cls = YABTClassifier

        X = self.preprocess(X, is_train=True)
        y = np.asarray(y)

        params = self._get_model_params()
        params.setdefault("device", "cuda" if num_gpus and num_gpus > 0 else "cpu")
        params.setdefault("c_grower_threads", int(num_cpus))

        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = (self.preprocess(X_val), np.asarray(y_val))
        else:
            # No validation split -> early stopping can't run; a raw 10k-tree
            # cap would be a budget blowout, so fall back to the library default.
            params["early_stopping_rounds"] = 0
            params["n_estimators"] = min(params.get("n_estimators", 500), 500)

        self.model = model_cls(**params)
        self.model.fit(X, y, eval_set=eval_set, categorical_features=self._cat_idx or None)
        self._fit_time = time.time() - start

    def _set_default_params(self) -> None:
        # n_estimators is a cap only: with a validation split, early stopping
        # picks the tree count, which is what makes the low learning rate safe.
        # lr/subsample/colsample/count-features are the proxy-sweep winner over
        # the 51 TabArena-Lite datasets (benchmarks/ab_tabarena_proxy.py,
        # "combo_cat": mean -3.3% metric error vs library defaults, 44/51 wins).
        defaults = {
            "n_estimators": 10_000,
            "early_stopping_rounds": 50,
            "learning_rate": 0.05,
            "subsample": 0.9,
            "colsample": 0.9,
            "cat_count_features": True,
            # High-cardinality pair conjunctions, gated by the parent-beating
            # correlation guard (seed-averaged A/B: Amazon -8%, no regressions).
            "cat_combinations": 16,
            # Vector scaling on the ES validation split (TabFM-style): proxy
            # A/B multiclass -3.3% mean (anneal -16%), binary/regression untouched.
            "calibrate_multiclass": True,
        }
        for param, val in defaults.items():
            self._set_default_param_value(param, val)

    def _get_default_auxiliary_params(self) -> dict:
        default_auxiliary_params = super()._get_default_auxiliary_params()
        default_auxiliary_params.update({"valid_raw_types": ["int", "float", "category"]})
        return default_auxiliary_params

    @classmethod
    def supported_problem_types(cls) -> list[str]:
        return ["binary", "multiclass", "regression"]

    def _get_default_resources(self) -> tuple[int, int]:
        num_cpus = ResourceManager.get_cpu_count(only_physical_cores=True)
        try:
            import torch

            num_gpus = 1 if torch.cuda.is_available() else 0
        except ImportError:
            num_gpus = 0
        return num_cpus, num_gpus

    @classmethod
    def config_generator(cls) -> ConfigGenerator:
        """Default config plus a random-search space over the core knobs."""
        from autogluon.common.space import Categorical, Int, Real

        from tabarena.utils.config_utils import ConfigGenerator

        search_space = {
            # n_estimators is not searched: the cap + early stopping applies to
            # every config.
            "learning_rate": Real(0.03, 0.3, log=True),
            "max_leaves": Int(15, 127),
            "reg_lambda": Real(0.1, 10.0, log=True),
            "min_samples_leaf": Int(5, 60),
            "subsample": Real(0.6, 1.0),
            "colsample": Real(0.6, 1.0),
            "neural_leaves": Categorical(True, False),
            "interaction_aware": Categorical(True, False),
            "min_split_gain_rel": Real(1e-6, 1e-2, log=True),
        }
        return ConfigGenerator(model_cls=cls, manual_configs=[{}], search_space=search_space)

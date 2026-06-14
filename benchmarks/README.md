# YABT Benchmark Suite

`openml_benchmark.py` is the standard tabular-GBM benchmark on the
**Grinsztajn et al. (2022) OpenML suites** (the suites used by *"Why do
tree-based models still outperform deep learning on tabular data?"*). This is
the one to cite. Real datasets, multiple seeds, native categorical handling,
same-device timing.

## Files

- **`openml_benchmark.py`** - Standard OpenML/Grinsztajn benchmark runner
- **`datasets.py`** - Grinsztajn suite definitions + OpenML loader
- **`openml_benchmark_results.json`** - Results from the standard benchmark

## Standard benchmark (openml_benchmark.py)

The four official OpenML benchmark suites (resolved in `datasets.py`):

| Suite      | OpenML id | Task                       | # datasets |
|------------|-----------|----------------------------|------------|
| `num_clf`  | 337       | numerical classification   | 16         |
| `num_reg`  | 336       | numerical regression       | 19         |
| `cat_clf`  | 334       | categorical classification | 7          |
| `cat_reg`  | 335       | categorical regression     | 17         |

**Protocol:** each dataset is subsampled to `--max-rows` (Grinsztajn "medium"
regime), split `--seeds` times into 70/30 train/test, and every model
(YABT, XGBoost, LightGBM, CatBoost) is fit with default hyper-parameters
(100 trees, lr 0.1, depth 6) on the **same device**. Metric is accuracy
(classification) or R² (regression), reported as mean ± std across seeds. Each
model gets native categorical handling (YABT target-encoding, XGBoost
`enable_categorical`, LightGBM `categorical_feature`, CatBoost `cat_features`).

```bash
# list datasets in a suite
python openml_benchmark.py --suite num_reg --list

# run numerical regression, 3 seeds (default)
python openml_benchmark.py --suite num_reg

# smaller/faster: cap rows and dataset count
python openml_benchmark.py --suite num_clf --max-rows 20000 --max-datasets 6

# everything (slow)
python openml_benchmark.py --suite all
```

Timing: every model runs on the same device. Use `--device gpu` for GPU-only,
or `--device both` to time each model on **CPU and GPU** in the same run (scores
should match; the summary prints a `cpu_time`/`gpu_time` column per model):
```bash
python openml_benchmark.py --suite num_reg --device gpu
python openml_benchmark.py --suite num_reg --device both
```
On small datasets GPU can be *slower* than CPU (kernel-launch overhead dominates);
the GPU advantage shows up on the large datasets — raise `--max-rows` to see it.

To re-resolve the dataset ids from the live OpenML study endpoint:
```python
from datasets import refresh_suites; print(refresh_suites())
```

## Results

Results are written to `openml_benchmark_results.json`, containing the scores
(accuracy / R²) and per-device timing for each model on each dataset.

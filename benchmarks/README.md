# YABT benchmarks: TabArena

YABT is benchmarked on [TabArena](https://github.com/autogluon/tabarena), the
living tabular-ML benchmark behind [tabarena.ai](https://tabarena.ai): 51
curated datasets (binary / multiclass / regression), an enforced protocol
(fixed splits, 8-fold bagging, per-dataset metrics: ROC AUC, log loss, RMSE),
and a leaderboard of 70+ methods spanning GBDTs, AutoML systems, and tabular
foundation models. Everything lives under `tabarena/`.

## Files (`tabarena/`)

- **`yabt_model.py`** — YABT wrapped as an AutoGluon `AbstractModel`, plus the
  TabArena default config and the HPO search space. The default config is the
  winner of the proxy sweep below, each piece confirmed on the full bagged
  protocol.
- **`run_tabarena.py`** — the benchmark runner. Results cache under
  `experiments/` (re-runs resume; delete a task dir to force a re-fit),
  leaderboard + plots under `eval/`.
- **`ab_smalldata.py`** — the band-restricted A/B loop, for the sub-2500-row
  datasets where YABT's leaderboard rank collapses (median 51/78 there against
  30/78 above it). One fold cannot resolve a candidate on 748 rows, so this
  fits every config on several outer CV splits per dataset and compares
  candidates *paired* on the same splits. The row band, device, split count and
  output tag are env-overridable (`YABT_AB_MIN_ROWS`, `YABT_AB_MAX_ROWS`,
  `YABT_AB_DEVICE`, `YABT_AB_SPLITS`, `YABT_AB_TAG`), so the same script holds
  the mid-size band on the GPU while the small band runs on the CPU:

  ```bash
  python benchmarks/tabarena/ab_smalldata.py baseline leaves4 caps_v2
  python benchmarks/tabarena/ab_smalldata.py --report      # paired comparison
  YABT_AB_MIN_ROWS=2500 YABT_AB_MAX_ROWS=9000 YABT_AB_DEVICE=cuda \
    YABT_AB_TAG=mid4 YABT_AB_SPLITS=4 python benchmarks/tabarena/ab_smalldata.py baseline leaves15
  ```

  `ab_smalldata_results.json` / `ab_mid4_results.json` hold the measurements
  behind the retuned `small_data_caps` default.
- **`ab_tabarena_proxy.py`** — fast config A/B loop: single (unbagged) fold-0
  fits over all 51 datasets, ~2–3 min per config on a free GPU. Single-fit
  deltas under ±5% are seed noise — confirm candidates with multi-seed means
  or a full bagged run before shipping them.
- **`ab_tabarena_proxy_results.json`** — proxy sweep measurements (17 configs).
- **`ab_softmax_multiclass.py`** — the softmax-vs-OvR multiclass A/B.

## Setup

The harness runs inside a TabArena environment (not YABT's own venv):

```bash
git clone https://github.com/autogluon/tabarena
cd tabarena && uv venv --python 3.12 .venv
VIRTUAL_ENV=$PWD/.venv uv pip install --prerelease=allow -e "./packages/tabarena[benchmark]" tabulate
VIRTUAL_ENV=$PWD/.venv uv pip install -e /path/to/yabt ninja
```

## Running

From `benchmarks/tabarena/`, with the TabArena venv's python:

```bash
python run_tabarena.py           # smoke run: 3 small datasets
python run_tabarena.py --full    # full TabArena-Lite (51 datasets, ~40 min on a 4090)
python run_tabarena.py --full --cpu   # CPU-only (the reported numbers; ~2 h on 8 cores)
python run_tabarena.py --full --n-configs 200   # adds the tuned-config HPO protocol
```

The first run downloads datasets from OpenML and the official leaderboard
baselines into `~/.cache/openml` / `~/.cache/tabarena`; both are persistent.

## Results

See the "Benchmark results" section of the top-level README for the current
leaderboard standing, and `eval/<run name>/` after a run for the full
leaderboard CSV, Pareto fronts, and win-rate matrix (`--cpu` writes to
`*_cpu/`, so GPU and CPU runs do not overwrite each other).

`--cpu` records a task that blows AutoGluon's 1-hour per-model budget as a
failure and continues, rather than aborting the sweep: on CPU, QSAR-TID-11
(1025 features) does not finish, and the leaderboard is then computed over the
remaining 50 datasets for every method.

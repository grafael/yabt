# YABT: Yet Another Boosting Tree

YABT is a gradient boosting library with multi-core CPU and CUDA backends
(`device="auto"` picks one). Like XGBoost or LightGBM, it trains a sequence of
small decision trees where each tree corrects the mistakes of the ones before
it. The difference is what YABT does
to each tree once it is built: it replaces each leaf's constant output with a
small learned model, and can optionally fine-tune the split points with
gradient descent (the same method used to train neural networks) so they land
on better values.

It exposes the scikit-learn API (`fit` / `predict` / `predict_proba`) and
works as a drop-in replacement for XGBoost/LightGBM-style estimators.

> **Experimental.** YABT is a research project, not a production library.
> On [TabArena](#benchmark-results-tabarena) its default config is the
> second-strongest GBDT default after CatBoost's, at the fastest train time in
> that group (CPU-to-CPU), but it has none of a mature library's hardening
> (distributed training, out-of-core training, ecosystem tooling). SciPy sparse
> input is accepted but densified, not exploited.

## How it works

YABT is gradient boosting: it builds a sequence of trees, each trained to
predict the error (the gradient of the loss) left over by all the trees before
it, and sums their contributions. Each tree is grown with the standard
histogram method — on an OpenMP C grower on CPU or a batched CUDA grower on
GPU — where continuous features are binned into small integer
buckets (0–255), bucketed error sums give a gain score for every candidate
split, and the highest-gain split wins until the leaf or depth budget is hit.
The twist is what sits in each leaf: instead of a single constant, YABT can use
neural leaves — a small linear model over the most useful features, so a leaf
captures trends inside its region (on by default, it helps at near-equal cost) —
and optionally refinement, which treats thresholds and leaf values as tunable
weights and takes a few gradient-descent steps off the coarse bin edges (off by
default via `refine_steps > 0`, since A/B tests show ~10% more training time for
negligible-to-negative accuracy gain). Then the next tree starts on the error
that is still left, and the loop repeats.

## Optional features

YABT ships several extra features you can toggle. Each was A/B-tested; the
sections below have the details and the measured trade-offs.

| Feature | Flag | Default | Summary |
|---|---|---|---|
| Neural leaves | `neural_leaves` | on | small per-leaf model instead of a constant |
| Interaction-aware splits | `interaction_aware` | on (n≥2000) | learned feature interactions steer tree growth |
| Product features | `product_features` | off | auto-built feature products for multiplicative targets |
| Differentiable refinement | `refine_steps` | off | gradient-descent polish of splits and leaves (set `refine_steps>0`) |
| Kernel splits | `kernel_splits` | off | non-linear RBF "blob" splits at a node |
| Stochastic routing | `stochastic_routing` | off | smooth, probabilistic predictions |
| Seed ensembling | `n_ensemble` | off (1) | average several seeds to cut variance |
| Small-data caps | `small_data_caps` | off | cap the tree budget below 2000 rows |
| Auto-tuning | `auto_tune` | off | picks hyperparameters per dataset before fitting |
| Adaptive features | `adaptive_features` | off | feature importance learned during training |
| GOSS sampling | `goss_enabled` | off | keep big-error rows, subsample the rest |
| Multi-task | `YABTMultiTaskRegressor` | | one shared tree structure across many targets |

## Installation

With [uv](https://docs.astral.sh/uv/) (recommended):

```bash
uv sync
```

Or with pip:

```bash
pip install -e .
```

### Native components

YABT is pure Python and works out of the box with the dependencies above; no
build step is required. The one native component is the optional **OpenMP C
grower** (`yabt/_cgrow/grow.c`), a multi-core drop-in for the single-threaded
Numba grower. It is *not* shipped as a precompiled binary — instead it is
compiled once, on demand, to a shared library cached next to the source the
first time a `fit` actually uses it (keyed by a hash of the source plus the
exact build command). This needs a C compiler with OpenMP support
(`gcc`/`clang` on Linux, Apple Clang + Homebrew `libomp` on macOS, MinGW/MSVC
on Windows).

If no working compiler is found, YABT silently falls back to the Numba grower —
a **performance** fallback, not a correctness one (single-threaded C is at
parity with Numba; the C grower's only edge is multi-core scaling).

To compile it deliberately ahead of time — e.g. in a Dockerfile or CI step, so
the cost and any toolchain problems surface up front rather than during the
first training run — run the prebuild diagnostic:

```bash
python -m yabt.grow_c
```

It reports the detected compilers and the cached library path on success, or
the per-candidate build errors on failure. To opt out of the C grower entirely,
pass `c_grower=False` to any estimator.

## Example

```python
from yabt import YABTClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1)

# The TabArena-winning recipe: a generous tree cap with early stopping on a
# validation split, a conservative learning rate, and light row/column
# subsampling. Neural leaves and interaction-aware splits are on by default.
clf = YABTClassifier(
    n_estimators=10_000,        # cap only; early stopping picks the count
    early_stopping_rounds=50,
    learning_rate=0.05,
    subsample=0.9,
    colsample=0.9,
)
clf.fit(X_train, y_train, eval_set=(X_val, y_val))
proba = clf.predict_proba(X_test)
```

Categorical columns are handled natively — pass their indices as
`categorical_features=[...]` to `fit` and they are target-encoded leakage-free
(see the `cat_*` parameters for count features, pair combinations, and
per-class multiclass encodings).

Per-row weights are supported on every estimator
(`fit(X, y, sample_weight=w)`): they scale the Newton gradients and Hessians,
so they weight the split gains, the leaf values and the per-leaf models alike.
Note that the binner's quantile edges, `min_samples_leaf` and
`leaf_net_min_samples` still count rows, so a zero weight silences a row's
influence on the objective without removing it from the data.

Fitted estimators pickle and unpickle to bit-identical predictions
(`tests/test_sklearn_api_surface.py`); there is no versioned on-disk format, so
a pickle is only readable by a compatible YABT/PyTorch.

### Missing values

NaNs get their own bin, above every real value of their feature: a column with
missing values in the training data gives up one bin of resolution and reserves
the top index for them, and the raw-space sentinel `impute` fills in is chosen
so `x <= threshold` is False for a missing value at every real threshold —
matching what the binned matrix says. Missingness is therefore an ordinary
extreme category the trees can split off, and no grower needs a NaN branch.

The previous behavior, median imputation, merged missing rows into the middle
of the distribution and destroyed informative missingness. Paired A/B over the
datasets that actually carry NaNs:

| dataset | rows | NaN | median impute | own NaN bin |
|---|--:|--:|--:|--:|
| polish_companies_bankruptcy | 5910 | 1.2% | 0.05847 | **0.05023** (-14.1%) |
| kick | 72983 | 0.06% | 0.22634 | **0.22522** (-0.5%) |
| APSFailure | 76000 | 8.3% | 0.00903 | 0.00901 (neutral) |
| hepatitis | 155 | 4.1% | 0.22310 | 0.22315 (neutral) |
| colic | 368 | 8.2% | 0.09381 | 0.09917 (+5.7%, within noise) |

(1 - ROC AUC, lower is better; APSFailure/hepatitis/colic re-measured at 12-40
fits each because the first 3-fold pass could not separate them from noise.) A
large win where missingness carries signal, neutral where it does not, and
exactly neutral on data with no NaNs at all — the bins there are unchanged.

A *learned per-split* direction (LightGBM/XGBoost style) is not worth the extra
machinery on top — XGBoost with its NaNs pre-replaced by this same kind of
sentinel scores 0.04119 on polish against 0.04142 for its own per-node
direction.

A NaN arriving at predict time in a column that had none during training has no
reserved bin; it lands in the top real bin, which still routes it consistently,
just not separably.

See `benchmarks/` for the [TabArena](https://github.com/autogluon/tabarena)
benchmark harness and results against XGBoost, LightGBM, CatBoost, and 70+
other methods.

## Benchmark results: TabArena

YABT is benchmarked on [TabArena](https://tabarena.ai) (TabArena-Lite: 51
curated datasets, the official protocol — fixed splits, 8-fold bagging, ROC
AUC / log loss / RMSE), against the public leaderboard of 70+ methods spanning
GBDTs, neural nets, AutoML systems, and tabular foundation models. The numbers
below are over the 50 datasets YABT completes on CPU within the protocol's
1-hour per-model budget; see [the caveat](#cpu-vs-gpu-and-the-wide-data-caveat)
for the one it does not.

**YABT's default configuration scores Elo 1267 (rank 37 of 78)** — the
strongest GBDT *default* on the leaderboard after CatBoost's, ahead of the
XGBoost / LightGBM / EBM defaults by a wide margin and ahead of several
*tuned* (200-config HPO) entries. YABT is trained **on CPU** here so the train
times are comparable with every other row (`run_tabarena.py --full --cpu`):

| # | Model | Elo | Median train s/1K rows |
|--:|---|--:|--:|
| 22 | CatBoost (default) | 1343 | 6.4 (CPU) |
| 27 | ChimeraBoost (tuned + ensembled) | 1328 | 1978.9 (CPU) |
| **37** | **YABT (default)** | **1267** | **1.7 (CPU)** |
| 38 | EBM (tuned + ensembled) | 1262 | 2892.5 (CPU) |
| 50 | XGBoost (default) | 1184 | 2.0 (CPU) |
| 55 | LightGBM (default) | 1150 | 2.1 (CPU) |

(The top of the table is tabular foundation models and 4-hour AutoML systems —
TabFM, TabPFN variants, AutoGluon — a different compute class than any single
default-config model.)

Against CatBoost's default head-to-head, YABT wins 17 of 50 datasets with a
median metric-error gap of +0.9% (regression +0.4%, binary +1.4%, multiclass
+4.5%); the remaining Elo gap is concentrated in a small tail of
small-or-noisy datasets where CatBoost's ordered boosting is strong.

### CPU vs GPU, and the wide-data caveat

Accuracy is device-neutral: over the 50 datasets both devices completed, the
median CPU/GPU metric-error ratio is 0.999 and CPU is better on 28 of 50
(Elo 1267 on CPU vs 1266 on GPU, same rank). Train time favors CPU at these
dataset sizes — median 1.7 s/1K rows on CPU against 3.3 on GPU, because
TabArena's datasets are small enough that GPU kernel-launch latency dominates
(anneal: 183 s on GPU, 22 s on CPU).

The exception is **wide** data, where the CPU grower's per-tree cost scales
with the feature count and the GPU pulls far ahead: hiva_agnostic (1617
features) is 13x slower on CPU, Bioresponse (1776) 5.8x, and **QSAR-TID-11
(1025 features) does not finish within TabArena's 1-hour per-model budget on
CPU at all** (94 s on GPU). That task is therefore excluded from the CPU
leaderboard above — for every method, so the 50-dataset comparison is
internally fair, but it is not the same task set as the published 51-dataset
board. On GPU, YABT ranks 32/78 on that task, slightly better than its overall
rank, so dropping it flatters YABT marginally. Use `device="cuda"` for wide
feature matrices.

The TabArena default config (defined in `benchmarks/tabarena/yabt_model.py`)
is `learning_rate=0.05`, `subsample=colsample=0.9`, a 10k-tree cap with
`early_stopping_rounds=50` on the fold's validation split,
`cat_count_features=True`, `cat_combinations=16`, and
`calibrate_multiclass=True`. Every piece was A/B-selected on a fold-0 proxy
sweep and confirmed on the full bagged protocol
(`benchmarks/tabarena/ab_tabarena_proxy.py` has the sweep harness and
measurements).

### Reproducing the numbers

The harness needs a TabArena environment (see `benchmarks/README.md` for
setup). Then, from `benchmarks/tabarena/`:

```bash
python run_tabarena.py                 # smoke run: 3 small datasets
python run_tabarena.py --full          # full TabArena-Lite, GPU if one is present
python run_tabarena.py --full --cpu    # CPU-only: the numbers reported above
```

Results cache under `experiments/` (re-runs resume where they left off); the
leaderboard CSV, Pareto fronts, and win-rate matrix land in `eval/`.

## Kernel-based splits

In addition to the usual axis-aligned splits (`x[f] <= t`), YABT can split
nodes by RBF similarity to a landmark point sampled from the node's own rows:
`exp(-gamma * ||x - c||^2) <= t`. This carves out spherical regions in
scale-normalized feature space, so a single split can capture structure that
axis-aligned trees need many splits to approximate. At each node both split
types compete on the same Newton gain criterion and the better one wins.

```python
clf = YABTClassifier(
    kernel_splits=True,      # enable RBF landmark splits (off by default)
    kernel_candidates=8,     # landmarks sampled per node
    kernel_gamma=0.0,        # RBF bandwidth; 0 = median-distance heuristic
    kernel_min_samples=64,   # only try kernel splits on nodes this large
)
```

`kernel_importance_weighting` ("node" or "ema") additionally weights the
distance by per-feature split gain, so important features dominate it. This
is experimental and off by default: in our A/B tests neither variant beat the
uniform distance, because gain-adaptive distances tend to select kernel
splits whose in-sample advantage does not generalize.

On a noisy XOR problem with depth-1 trees, axis-aligned boosting stays at
chance (about 50% accuracy, since the target is not additive in the features)
while kernel splits reach about 95%.

## Neural leaf networks

Instead of a constant value, each sufficiently populated leaf can hold a small
model over the tree's most informative features, fitted to the boosting Newton
objective at the current ensemble margin, so each leaf captures the residual
structure within its region. `leaf_net_hidden=0` (default) fits a closed-form
ridge-linear model per leaf; `>0` trains a small tanh MLP per leaf.

```python
reg = YABTRegressor(
    neural_leaves=True,      # per-leaf models (ON by default; set False to disable)
    leaf_net_hidden=0,       # 0 = ridge-linear leaves; >0 = tanh MLP width
    leaf_net_features=8,     # net inputs: top split features + strongest rest
    leaf_net_min_samples=50, # smaller leaves keep their constant value
)
```

Linear leaves match or beat constant leaves at near-equal training cost, with
the largest gains at small tree budgets (+2 to +3 points of accuracy/R² at 25
trees on california/digits in our A/Bs). The MLP variant did not outperform
linear leaves in benchmarks and trains several times slower, so prefer
`leaf_net_hidden=0` unless you are experimenting.

## Stochastic routing

Trees are trained with hard routing as usual, but at inference each internal
node can route probabilistically: a row goes left with probability
`sigmoid((threshold - x) / (tau * scale))`, and the prediction is the exact
expectation over all root-to-leaf paths. The result is a smooth (and
differentiable-in-x) prediction function instead of a piecewise-constant one.

```python
reg = YABTRegressor(
    stochastic_routing=True,  # soft inference (off by default)
    routing_tau=0.05,         # gate width relative to the feature's scale
)
```

In our A/Bs this helps on smooth or noisy targets (+1.5 accuracy points and
better log loss on a synthetic 30-dim task, +0.6 R² on Friedman #1) and hurts
when the target has genuine discontinuities (california housing, digits). So
it is off by default and worth a try when you believe the underlying function
is smooth, or when you need continuous/differentiable predictions downstream.
Soft inference costs roughly 3x hard inference (still milliseconds).

## Interaction-aware splits

YABT learns a feature-interaction matrix during training from
ancestor-descendant split pairs on tree paths (a split on B underneath a split
on A means B's effect is conditioned on A). With `interaction_aware=True`,
that matrix steers later tree growth: at each node, features that historically
interact with the features already on the node's path get a selection boost
capped at `1 + interaction_boost`. The boost only flips near-ties (split
acceptance and leaf ordering always use the true unboosted gain) and
interaction counts are measured against the background rate, so uniform noise
produces no steering.

```python
reg = YABTRegressor(
    interaction_aware=True,  # learned interactions guide growth (ON by default, n>=2000)
    interaction_boost=0.5,   # selection boost cap; try 1.0 for interaction-heavy data
)
```

In our A/Bs this is the strongest of the novel features on tabular data: +5 R²
points on a products-of-features regression in 30 dims, +1.2 accuracy points
on a 30-dim classification benchmark, small wins on friedman1/california, and
a -0.2 point cost on digits (which has no real feature interactions). Overhead
is about 10 to 20% training time. The learned pairs are also exposed directly
via `model.booster_.top_interactions(k)`.

It is automatically disabled for datasets under 2000 rows, where the
interaction counts are too noisy to trust and steering hurts more than it
helps (-2.5 R² points on a 442-row benchmark in our A/Bs). The
`interaction_aware=True` default still reflects this gate: it turns on once the
dataset is large enough.

## Auto-tuning

With `auto_tune=True`, YABT picks per-dataset hyperparameters before the final
fit via a bounded validation search over a small curated candidate set
(informed by the A/B results above, e.g. constant leaves for sharp-boundary
data, slower/deeper vs faster/shallower, stronger interaction boost). Each
candidate is scored on a held-out split at the deployment tree count, and the
winner is refit on all the data.

```python
clf = YABTClassifier(
    n_estimators=200,
    auto_tune=True,   # validation search before the final fit (off by default)
)
clf.fit(X, y)
print(clf.booster_.tuning_report_["selected"])  # which candidate won
```

This is a bounded search (a handful of fits), not an open-ended sweep, and it
is skipped automatically for datasets under 600 rows where a validation split
is too noisy to trust. In our A/Bs it improves or matches the default
everywhere (california +0.4 R², synthetic 30-dim +1.3 accuracy points,
friedman1/digits correctly left at the default) at a cost of roughly one fit
per candidate. Pass your own `eval_set` and it tunes against that instead of
an internal split. The chosen configuration is reported on
`booster_.tuning_report_`.

## Seed ensembling

`n_ensemble=k` fits `k` boosters that differ only in seed and averages them
(margins for regression and binary, probabilities for multiclass). It is pure
variance reduction, so it only does anything when training is stochastic —
with `subsample` and `colsample` both at 1.0 the members are identical and you
have paid `k` times over for one model.

```python
clf = YABTClassifier(n_ensemble=4, subsample=0.9, colsample=0.9)
```

On TabArena-Lite, `n_ensemble=4` is Elo 1289 against 1266 for a single fit
(rank 32 vs 36), at about 3.9x the train time. A local A/B over 11 datasets
agrees: median -1.4% metric error, better on 9 of 11. Off by default because
of the cost, not the accuracy.

`booster_` is then a `SeedEnsemble`; its members are in `booster_.members_`,
and attribute lookups (`booster_.binner`, `booster_.top_interactions()`) fall
through to the first member.

## Multi-task learning

`YABTMultiTaskRegressor` predicts several targets at once by growing one
shared tree structure for all of them: each split is chosen on the summed
per-target gain, and every leaf stores a value per target. Correlated targets
transfer through common splits and regularize one another; unrelated targets
fall back to roughly independent fits.

```python
from yabt import YABTMultiTaskRegressor

reg = YABTMultiTaskRegressor(n_estimators=200, max_leaves=16)
reg.fit(X, Y)          # Y is (n_samples, n_targets)
P = reg.predict(X)     # (n_samples, n_targets)
```

The structural sharing buys two things. First, efficiency: one model of
`n_estimators` trees instead of one model per target, about 3x faster to train
and 8x fewer trees on an 8-target problem, with proportional inference
savings. Second, accuracy in the data-scarce regime: when training data is
limited relative to the number of correlated targets, sharing splits acts as
regularization (+0.5 to +0.9 R² points across 4-8 correlated targets at a few
hundred rows in our A/Bs, growing with the number of targets). With abundant
data the shared structure is a mild constraint and independent models edge
ahead by a fraction of a point, and on uncorrelated targets the two are within
noise, so multi-task is the right tool when targets are related and data or
compute is the bottleneck. Numeric features only; the single-task extras
(kernel splits, neural leaves, soft routing, interaction-aware growth) do not
apply to this path.

## Parameters

All hyperparameters are passed as keyword arguments to the estimator
constructors (`YABTClassifier`, `YABTRegressor`, `YABTMultiTaskRegressor`).
`YABTMultiTaskRegressor` honors only the **Core tree / boosting** params plus
`early_stopping_rounds`, `seed`, and `device`; the single-task extras below do
not apply to its shared-structure path.

| Parameter | Default | Description |
|---|---|---|
| **Core tree / boosting** | | |
| `n_estimators` | `500` | Number of boosting iterations (trees). |
| `learning_rate` | `0.1` | Shrinkage applied to each tree's contribution. |
| `max_leaves` | `31` | Maximum number of leaves per tree. |
| `max_depth` | `64` | Maximum tree depth. |
| `reg_lambda` | `1.0` | L2 regularization on leaf weights. |
| `gamma` | `0.0` | Minimum loss reduction required to make a split. |
| `min_split_gain_rel` | `0.0` | Scale-invariant min-split-gain floor: per tree, an effective gamma of min_split_gain_rel * var(gradients) is added, refusing noise splits without a target-scale-dependent absolute threshold. |
| `min_child_weight` | `1e-3` | Minimum sum of Hessian (instance weight) allowed in a child. |
| `min_samples_leaf` | `20` | Minimum number of samples per leaf. |
| `subsample` | `1.0` | Row subsampling ratio drawn per tree. |
| `colsample` | `1.0` | Column (feature) subsampling ratio per tree. |
| `max_bins` | `256` | Number of histogram bins used to discretize features. |
| `small_data_caps` | `False` | Cap the tree budget to 16 leaves / depth 4 below 2000 rows, where the default 31-leaf budget can overfit. Caps only, never inflations. Off by default: across eight sub-2000-row datasets this is a median +0.36% metric error with a real regression tail (climate-model +6.6%, airfoil_self_noise +5.2%) even though it wins big where it lands (qsar-biodeg -7.7%), so `auto_tune` offers it as a candidate and deploys it only where a validation split says it helps. |
| `n_ensemble` | `1` | Fit this many boosters differing only in seed and average them. Pure variance reduction, so it does nothing unless training is stochastic (`subsample`/`colsample` < 1). TabArena-Lite at 4: +22 Elo (1266 -> 1289) for ~3.9x the train time, hence opt-in. |
| **Differentiable refinement** | | |
| `refine_steps` | `0` | Gradient-descent refinement steps applied to splits and leaves after each tree (0 disables; effective steps adapt to dataset size). Off by default: costs ~10% of fit time for negligible gain on real tabular data. Opt in with `refine_steps > 0`. |
| `refine_lr` | `0.02` | Learning rate for differentiable refinement. |
| `refine_min_gain` | `1e-4` | Skip refinement when the loss is already below this threshold. |
| `refit_every` | `0` | Refit all leaf values across the ensemble every N trees (0 disables). |
| `refit_steps` | `30` | Gradient steps per ensemble refit. |
| `refit_lr` | `0.05` | Learning rate for ensemble refit. |
| **Adaptive features & sampling** | | |
| `adaptive_features` | `False` | Learn feature importances during training and bias sampling toward them. |
| `feature_importance_alpha` | `0.1` | EMA smoothing factor for the learned feature importances. |
| `goss_enabled` | `False` | Gradient-based One-Side Sampling: keep large-gradient rows and subsample the rest. |
| `goss_ratio` | `0.9` | Fraction of large-gradient rows retained when GOSS is enabled. |
| **Interaction-aware growth** | | |
| `detect_interactions` | `False` | Track which feature pairs interact during training. |
| `interaction_aware` | `True` | Steer split selection toward features that interact with those already on the node's path. Only flips near-ties and never inflates the gain used to accept a split. On by default (A/B-verified on tabular data). |
| `interaction_boost` | `0.5` | Maximum multiplicative boost (capped at `1 + interaction_boost`) applied to near-tie gains by interaction steering. |
| **Product features** | | |
| `product_features` | `False` | Detect feature groups that drive the residual multiplicatively (via the magnitude signal corr(x^2, r^2)) and append their products as columns before training, so the greedy splitter can use interactions like x_i*x_j*x_k that have no marginal gain. A correlation guard keeps a product only when it beats its components, so data without multiplicative structure is left untouched. Off by default (A/B: large win on multiplicative targets, neutral elsewhere). |
| `product_max_features` | `5` | Number of top magnitude-signal features scanned for products. |
| `product_max_order` | `3` | Highest product order considered (3 = up to triple products). |
| `product_min_corr` | `0.03` | Absolute residual-correlation floor for a product to be kept. |
| `product_corr_gain` | `1.3` | A product is kept only if its residual correlation exceeds this factor times the best correlation of its component features. |
| **Kernel splits** | | |
| `kernel_splits` | `False` | Enable RBF landmark ("blob") splits for non-linear boundaries. |
| `kernel_candidates` | `8` | Number of candidate landmarks evaluated per node. |
| `kernel_gamma` | `0.0` | RBF bandwidth; 0 uses a per-landmark median-distance heuristic. |
| `kernel_min_samples` | `64` | Minimum node size for a kernel split to be considered. |
| `kernel_importance_weighting` | `False` | EXPERIMENTAL. Weight kernel distances by per-feature split gain. False = uniform (best overall in A/B tests); "node" / True = gains of the node being split; "ema" = EMA of root-level gains from previous iterations. |
| **Neural leaves** | | |
| `neural_leaves` | `True` | Fit a small per-leaf model instead of a constant value. On by default (linear leaves A/B-verified to win or tie at ~equal cost). |
| `leaf_net_hidden` | `0` | Hidden width; 0 = ridge-linear leaves, >0 = tanh MLP of this width. |
| `leaf_net_features` | `8` | Number of top tree-split features used as leaf-model inputs. |
| `leaf_net_l2` | `1.0` | L2 regularization for the leaf models. |
| `leaf_net_steps` | `40` | Adam steps per tree (MLP leaves only). |
| `leaf_net_lr` | `0.05` | Adam learning rate (MLP leaves only). |
| `leaf_net_min_samples` | `50` | Leaves smaller than this keep their constant value. |
| **Auto-tuning** | | |
| `auto_tune` | `False` | Search curated hyperparameter candidates on a validation split before the final fit (skipped for datasets with < 600 rows). |
| **Stochastic routing** | | |
| `stochastic_routing` | `False` | Use soft (expected-path) routing at inference; trees are still grown and trained hard, but predictions become smooth in X. |
| `routing_tau` | `0.05` | Gate width as a fraction of the split feature's scale. |
| **Growth strategy** | | |
| `levelwise` | `"auto"` | Breadth-first (level-wise) growth with sibling subtraction. "auto" enables it on CUDA when `max_leaves >= 16` (~1.8x faster), otherwise uses the best-first heap grower; it also falls back to the heap below 16 leaves or when `kernel_splits` is on. True/False force it on/off (True still skips kernel splits). |
| `numba_grower` | `"auto"` | Use the Numba-JIT compiled best-first grower on CPU (1.5-4x faster than the torch grower at identical accuracy). "auto" enables it on CPU for the axis-split path; it falls back to the torch grower on CUDA, on the level-wise path, or when `kernel_splits` is on. True/False force it on/off (True still falls back where unsupported). |
| `c_grower` | `"auto"` | Use the OpenMP-parallel C grower: the same leaf-wise kernel as the Numba grower but with the histogram build and split search threaded across cores (the Numba grower is single-threaded), bit-identical trees. "auto" uses it wherever the Numba grower runs when a C compiler is available and the problem is large enough to amortize threads, else falls back to Numba. True forces it on (still falls back if no compiler); False disables it. |
| `c_grower_threads` | `0` | OpenMP thread cap for the C grower; 0 picks a default (min(cores, 8), or `OMP_NUM_THREADS` if set above 1). |
| `sparse_hist` | `"auto"` | Sparse histogram build for the Numba grower: store each feature's non-modal bins and fill the modal bin by subtraction, making a histogram cost O(node_nnz + F) instead of O(node_rows * F). The win is on wide, sparse data (e.g. ~1.3x on Santander, 4991 features 97% zero), accuracy-neutral. "auto" uses it only when the data is dense enough below `sparse_hist_max_density` and rows are not subsampled; True/False force it (still requires the Numba grower). |
| `sparse_hist_max_density` | `0.5` | Max fraction of explicitly-stored cells for "auto" `sparse_hist` to engage; above this the dense builder is used (no sparsity to exploit). |
| **Training control** | | |
| `multiclass` | `"softmax"` | Multiclass strategy: "softmax" grows one tree per class per round on the joint softmax cross-entropy gradients (shared binning, joint early stopping on multiclass log loss); "ovr" trains one independent binary booster per class. The classifier falls back to OvR when an opt-in feature the softmax loop does not support is enabled (kernel splits, GOSS, adaptive/product features, refinement/refit, stochastic routing). Note the fallback costs multiclass accuracy, so prefer leaving those off unless you need them. |
| `early_stopping_rounds` | `50` | Stop if the eval metric does not improve for this many rounds (0 disables). The metric is computed on the `eval_set` passed to `fit`, or on a `validation_fraction` holdout carved automatically when no `eval_set` is given. |
| `validation_fraction` | `0.15` | Fraction of the training data carved off (stratified for the classifier, seeded by `seed`) to drive early stopping when `fit` is called without an `eval_set`, so the tree count adapts to the dataset instead of overfitting small data at the full `n_estimators` budget. Not carved -- the full data is trained on, as before -- when early stopping is disabled or could never trigger (`n_estimators` <= `early_stopping_rounds`), when set to 0, when the holdout would have fewer than 50 rows, or under `auto_tune` (which manages its own validation). |
| `seed` | `0` | Random seed. |
| `device` | `"auto"` | "auto" picks "cuda" when available, else "cpu"; may also be set to "cuda" or "cpu" explicitly. |
| `verbose` | `False` | Print per-iteration training progress. |
| `cat_smoothing` | `10.0` | Smoothing strength for the leakage-free target encoding of categorical columns (selected via the `categorical_features` argument to `fit`). |
| `cat_per_class` | `False` | Multiclass only: target-encode each categorical column once per class (one-vs-rest indicators), replacing the column with class 0's encoding and appending the rest, instead of encoding the raw class index (whose mean is ordinal noise). |
| `cat_count_features` | `False` | Append a log1p train-frequency column per categorical, so trees can split on how common a category is independently of its target rate. |
| `cat_combinations` | `0` | Target-encode up to this many categorical *pairs* (CatBoost-style feature combinations, strongest parent columns first) and append them as extra columns, so trees can split on conjunctions that neither parent captures alone. 0 disables. |
| `cat_combinations_min_card` | `8` | Only categorical columns with at least this many distinct training values are eligible as pair parents: small-cat conjunctions are reachable with two ordinary splits, so their pair encodings are noise columns, while high-cardinality conjunctions carry unique signal. |
| `calibrate_multiclass` | `False` | Vector-scale multiclass probabilities (per-class scale and bias on log p, re-softmaxed) fit on the `eval_set` after training. Bounded near identity and L2-pulled toward it, so a small validation set cannot push predictions far from the uncalibrated ones. Requires an `eval_set`; classifier-only, no effect on binary problems. |
| `svd_features` | `0` | Append this many PCA projections of the (encoded, standardized) feature matrix as extra columns. Axis-aligned splits cannot express linear combinations of features; the top principal directions hand the strongest ones to the grower as ordinary columns. 0 disables. |
| `svd_min_features` | `32` | Minimum encoded width for `svd_features` to engage: with few columns the top principal directions are near-copies of raw features and the extra columns just dilute sampling, while many correlated columns carry real linear structure. |

## License

MIT — see the [LICENSE](LICENSE) file for details.

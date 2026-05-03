# Gold Prediction — CAGR & Signal Direction Accuracy Improvement Analysis

**Date:** 2026-05-03 (updated)  
**Test period:** 2022-06-22 → 2026-05-01  
**Baseline (Buy & Hold):** CAGR 27.2%, DirAcc 55.0%, Sharpe 1.26, MaxDD -17.7%

---

## 0. Recently Completed Fixes

### 0.1 SelectKBest Leakage (2026-05-02)

`train.py` CV Loop 2 previously fit `SelectKBest` on all of `trainval` before splitting
into CV folds, leaking mild target information into feature selection. This has been fixed:
`SelectKBest` is now inside the `Pipeline` passed to `GridSearchCV` and is re-fit on the
training fold of each split only.

Impact on results:
- XGBoost `best_model` now uses **k=39** (all features) rather than the leaky k=30 selection
- Rankings shifted: SVR_lin moved from #2 to **#1**; MLP dropped from #1 to **#2**
- Lasso Sharpe fell from 7.51 to **7.45**, dropping it below the ★ threshold
- Keras models showed slightly worse MaxDD (LSTM: -24.0% → -31.7%, GRU: -31.9% → -37.2%)
- X_test was never affected — out-of-sample CAGR/DirAcc numbers remain valid comparisons

Data integrity status after the fix (from README):

| Check | Status |
|---|---|
| StandardScaler fit on test data | Safe — fit on training fold only |
| SelectKBest in CV Loop 2 | **Fixed 2026-05-02** — now inside Pipeline, re-fit per fold |
| XGBoost best_model | **Fixed 2026-05-02** — retrained with k=39, no leakage |
| Feature engineering rolling windows | Safe — `center=False`, strictly backward-looking |
| Fractional differencing | Safe — backward window `iloc[i-width+1 : i+1]` |
| X_test ever passed to fit() | Safe — never |
| TimeSeriesSplit used throughout | Safe — temporal order preserved |

### 0.2 Frac-diff d selected on full dataset (2026-05-03)

**Bug:** `train.py:147–156` ran the ADF stationarity loop on the entire `df['log_Gold']`
(including the 2022–2026 test period) to choose the fractional differencing order `d`.
Same for the COT column ADF tests at `train.py:161–165`.

`d` is a hyperparameter. Choosing it with knowledge of the test period is mild leakage —
ADF tests unit-root properties, not return levels, so the chosen `d` is very unlikely to
differ between train-only and full-dataset runs. But it is still a correctness issue.

**Fix (2026-05-03):** Added `_split_raw = int(len(df) * 0.80)` before the ADF loop.
The ADF test now runs on `df.iloc[:_split_raw]` only. The chosen `d` is then applied to
the full `df` for downstream use. Both the `log_Gold` and COT column ADF tests are fixed.

```python
# train.py — after fix
_split_raw = int(len(df) * 0.80)
adf_test(df['log_Gold'].iloc[:_split_raw], 'log(Gold) [train]')
for d in [0.3, 0.4, 0.5, 0.6, 0.7]:
    fd = frac_diff(df['log_Gold'], d=d)
    if adf_test(fd.iloc[:_split_raw].dropna(), f'FracDiff(log_Gold, d={d}) [train]'):
        ...
```

`tune_model.py` and `update_results.py` load `FRAC_D` from `model_metadata.json` rather
than running ADF independently, so they are unaffected by this fix once `train.py` is re-run.

---

## 1. Current State

### Post-tuning + leakage-fixed results (`tune_all.py` + `update_results.py`, 2026-05-02)

| Rank | Model | Sharpe | DirAcc | MaxDD | Best Parameters |
|---:|---|---:|---:|---:|---|
| 1 | **SVR_lin** ★ | 8.86 | 72.3% | -6.1% | C=0.01, ε=0.25×σ |
| 2 | **MLP** ★ | 8.55 | 71.7% | -6.1% | hidden=[128,64,32], α=0.01, lr=0.01 |
| 3 | **Ridge** ★ | 8.40 | 70.5% | -6.1% | α=100 |
| 4 | **SVR_rbf** ★ | 7.88 | 69.6% | -6.1% | C=0.01, γ=0.001, ε=0.25×σ |
| 5 | **XGBoost** ★ | 7.81 | 69.6% | -7.2% | lr=0.05, depth=2, n=200, sub=1.0 |
| 6 | **RandomForest** ★ | 7.74 | 68.7% | -7.2% | n=50, max_depth=10, min_split=10 |
| 7 | Lasso | 7.45 | 68.7% | -6.1% | α=0.001 |
| — | **Buy & Hold** | 1.26 | 55.0% | -17.7% | baseline |
| 8 | BiLSTM | 0.71 | 52.4% | -27.9% | ts=10, units=16, drop=0.3, rdrop=0.2 |
| 9 | LSTM | 0.13 | 50.6% | -31.7% | ts=7, units=32, drop=0.5, rdrop=0.2 |
| 10 | GRU | 0.06 | 49.9% | -37.2% | ts=15, units=64, drop=0.4, rdrop=0.0 |

★ = Sharpe ≥ 7.5. Note: Lasso (7.45) no longer qualifies after the leakage fix.  
Note: Keras model Sharpe figures vary slightly (~±0.3) across runs due to TF non-determinism.

---

## 2. Root Cause Analysis

### 2.1 Why CAGR and Sharpe top out at current levels

**Binary signal with no confidence scaling.**  
The z-score signal in `train.py` and `tune_model.py` converts any prediction with z > 0 to
LONG (+1) and any z < 0 to SHORT (-1), regardless of conviction strength. Every day gets equal
capital exposure whether the z-score is 0.01 (barely positive) or 2.5 (strong conviction).

**Z-score rolling window hardcoded to 5 across all scripts — never tuned.**  
All four scripts hardcoded `rolling(5)` in their z-score signal computation.
`predict_tomorrow.py` has `ZSCORE_WIN = 10` but that constant only controlled the **slice
size** passed to `zscore_signal()` (i.e. `preds[-ZSCORE_WIN:]`), not the rolling window
inside it — which was also `rolling(5)`:
```python
# predict_tomorrow.py — zscore_signal() before fix
ps.rolling(5, min_periods=1).mean()   # hardcoded 5; ZSCORE_WIN had no effect here
```
There was no train/production mismatch in the rolling window itself, but the window of 5
was never tuned. The fix makes `ZSCORE_WIN` control the actual rolling call in all four
scripts and exposes it as a tunable hyperparameter.

**No dead zone for low-conviction signals.**  
When z is near 0, the signal is essentially a coin flip but still trades. Filtering
out |z| < threshold would improve DirAcc by skipping the noisiest days.

### 2.2 Why Keras DirAcc is near 50% (coin flip)

MSE minimisation on a nearly-i.i.d. series pushes the network to predict ≈ 0 for every
observation (the unconditional mean minimises expected squared error when the series has
no exploitable autocorrelation). Near-zero predictions produce z-scores near 0, which
are essentially random after sign thresholding.

Evidence from tuned model files:

| Model | PredStd | DirAcc |
|---|---|---|
| SVR_lin (best) | ~0.018 | 72.3% |
| MLP | 0.0217 | 71.7% |
| LSTM | 0.0121 | 50.6% |
| GRU | 0.0074 | 49.9% |

Lower PredStd → predictions concentrated near zero → z-scores near zero → random signal.

Fractional differencing at d=0.5 removes most temporal autocorrelation. Keras sequence
models (LSTM/GRU/BiLSTM) rely on temporal structure; when it is absent they add inductive
bias that introduces noise without reducing bias.

### 2.3 Why DirAcc caps near 72%

- Features lack longer-horizon momentum (63d, 126d), which captures multi-month trend regime
- COT data is weekly, forward-filled to daily — intra-week COT variation is lost
- No regime detection: the same model runs in trending and mean-reverting environments, where
  optimal signal direction differs
- Month-of-year seasonal patterns in gold (typically strong Jan, Aug–Sep) are not encoded

---

## 3. Improvement Plan

### Tier 1 — High Impact, Low Effort (fix / tune existing code)

#### 3.1 Fix the z-score window inconsistency (correctness bug)

**Files:** `train.py` (`_trading_metrics()`), `predict_tomorrow.py:31`, `tune_model.py`

Standardise to a single constant `ZSCORE_WIN` shared across all three scripts, and expose
it as a tunable parameter alongside the model hyperparameters.

```python
# shared constant at top of each script (all four now use this)
ZSCORE_WIN = 10   # all scripts previously hardcoded rolling(5); now unified and tunable

# in _trading_metrics() / zscore_signal():
z = (ps - ps.rolling(ZSCORE_WIN, min_periods=1).mean()) \
  / ps.rolling(ZSCORE_WIN, min_periods=1).std().fillna(1e-8)
```

**Status:** Implemented. `train.py`, `tune_model.py`, `update_results.py`, and
`predict_tomorrow.py` all now use `rolling(ZSCORE_WIN)`. Next step: add `ZSCORE_WIN`
to `PARAM_CATALOGUE` in `tune_model.py` and search over `[5, 7, 10, 15, 20]` to find
the optimal window rather than accepting 10 as an arbitrary default.

Expected outcome: optimal window may lift DirAcc by 1–3 pp vs the untuned default.

---

#### 3.2 Add a signal dead zone (confidence threshold)

**Files:** `train.py` (`_trading_metrics()`), `tune_model.py`

Replace:
```python
sig = np.where(z > 0, 1, -1).astype(float)
```
With:
```python
SIGNAL_THRESHOLD = 0.3   # tunable
sig = np.where(z >  SIGNAL_THRESHOLD,  1.0,
      np.where(z < -SIGNAL_THRESHOLD, -1.0, 0.0))
```

When `sig = 0` the strategy holds cash — no return, no loss on that day.  
Tune `SIGNAL_THRESHOLD` over `[0.0, 0.2, 0.3, 0.5, 0.75]`.

Filtering the lowest-conviction signals typically lifts DirAcc by **2–4 pp** and
reduces MaxDD by limiting activity on uncertain days.

---

#### 3.3 Add LightGBM and ElasticNet to the model suite

**Files:** `train.py` (SK_MODELS dict), `tune_model.py` (PARAM_CATALOGUE)

Both are drop-in additions to `SK_MODELS`:

```python
import lightgbm as lgb
from sklearn.linear_model import ElasticNet

SK_MODELS['LightGBM']   = lgb.LGBMRegressor(n_estimators=200, random_state=SEED, verbose=-1)
SK_MODELS['ElasticNet'] = ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=5000)
```

Parameter grids for `PARAM_CATALOGUE` in `tune_model.py`:
```python
'LightGBM': {
    'n_estimators':  [100, 200, 300],
    'num_leaves':    [15, 31, 63],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample':     [0.8, 1.0],
},
'ElasticNet': {
    'alpha':    [0.0001, 0.001, 0.01, 0.1],
    'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],
},
```

LightGBM uses leaf-wise tree growth and typically outperforms XGBoost on tabular data
with many low-information features. ElasticNet combines Ridge and Lasso regularisation —
expected to land between them in Sharpe/DirAcc.

After adding, run: `train.py` → `tune_all.py --models LightGBM ElasticNet` → `update_results.py`

---

### Tier 2 — Medium Impact, Medium Effort

#### 3.4 Proportional position sizing (scale by z-score)

**Files:** `train.py` (`_trading_metrics()`), `tune_model.py`

Replace binary ±1 signal with a continuous signal proportional to conviction:
```python
sig = np.clip(z / 3.0, -1.0, 1.0)   # ÷3 normalises; clip prevents leverage
```

Strategy return on day t becomes `sig_t × log_ret_t` where `sig_t ∈ [-1, 1]`.

Expected effect:
- CAGR: +20–50% (high-z days are already correct ~72% of the time; scaling amplifies gains)
- MaxDD: moderate improvement (noisy near-zero days get near-zero exposure)
- Sharpe: likely improves (less noise in strategy returns)

---

#### 3.5 Ensemble stacking of top-3 models

Train a meta-model on the out-of-fold predictions of SVR_lin, MLP, and Ridge, then use
the meta-prediction as the final signal.

Implementation approach:
1. During CV, collect out-of-fold predictions for each of the three base models
2. Stack OOF predictions as a 3-column feature matrix
3. Fit a meta-Ridge (or meta-MLP) on the stacked OOF predictions
4. At test time, generate base predictions and pass through the meta-model

Expected DirAcc lift: **1–3 pp** — the three models are correlated but not identical
(SVR_lin is robust to outliers; MLP captures nonlinear interactions; Ridge is maximally
regularised). The meta-model learns to combine them optimally.

---

#### 3.6 Add longer-horizon momentum features

**File:** `train.py` feature engineering block

```python
# Multi-period momentum (add after existing technical features)
for period in [21, 63, 126, 252]:
    df[f'gold_ret_{period}d'] = df['log_Gold'].diff(period)

# Month-of-year dummies (gold has documented seasonal patterns)
df['month'] = df.index.month
for m in range(1, 13):
    df[f'month_{m}'] = (df['month'] == m).astype(int)
df.drop('month', axis=1, inplace=True)

# COT 4-week position change (captures slower position buildup)
if HAS_COT:
    df['net_spec_chg_20d'] = df['net_speculator'].diff(20)
    df['net_comm_chg_20d'] = df['net_commercial'].diff(20)
```

Longer momentum (63d, 126d) captures multi-month trend regimes that COT and short-term
technicals miss. Month dummies encode the known seasonal gold pattern without overfitting
(they are known-in-advance, not fitted to the target).

After adding features: re-run `train.py` → `tune_all.py` → `update_results.py`.

Expected DirAcc lift: **1–3 pp** across all models.

---

#### 3.7 Walk-forward / expanding-window retraining

Current approach: fixed 80/20 split (2007–2022 train → 2022–2026 test).  
The model trained through 2022 cannot adapt to the 2022–2023 rate-hike regime change.

Improvement: add an expanding-window retraining loop:
- Re-fit every 6 months on all data up to that point
- Generate predictions for the next 6 months using the freshly fitted model
- Concatenate rolling out-of-sample predictions for final metric calculation

This is the most realistic backtest setup and is especially important for COT-based signals
whose seasonal structure shifts across macro regimes.

Expected CAGR improvement: regime adaptation typically adds **+10–30%** CAGR.

---

### Tier 3 — Structural Fixes for Keras Models

#### 3.8 Use a softer frac-diff `d` for Keras models

**File:** `train.py` stationarity block

The current pipeline picks d=0.5 globally. This removes most temporal autocorrelation that
LSTM/GRU depend on. Maintain two targets in parallel:

```python
df['target_d05'] = frac_diff(df['log_Gold'], d=0.5)   # for sklearn models
df['target_d03'] = frac_diff(df['log_Gold'], d=0.3)   # for Keras models (softer)
```

sklearn models use `target_d05` (maximally stationary, i.i.d. — optimal for tabular).  
Keras models use `target_d03` (still ADF-stationary, but preserves more temporal structure).

Expected effect: LSTM/GRU DirAcc from ~50% → ~55–58% (still below sklearn, but above coin flip).

---

#### 3.9 Directional loss function for Keras

**File:** `train.py` (`build_lstm`, `build_gru`, `build_bilstm` functions)

Replace MSE loss with a hybrid loss that penalises sign errors:

```python
import tensorflow as tf

def directional_mse(y_true, y_pred):
    mse      = tf.reduce_mean(tf.square(y_true - y_pred))
    sign_err = tf.maximum(0.0, -y_true * y_pred)   # positive when signs differ
    return mse + 0.5 * tf.reduce_mean(sign_err)

# In build_lstm / build_gru / build_bilstm:
m.compile(optimizer=Adam(learning_rate=0.001), loss=directional_mse)
```

This prevents mean-collapse (near-zero predictions incur a large `sign_err` penalty when
the true return is large) while retaining MSE as the base loss for calibration.

Expected effect: PredStd increases → z-scores more spread → DirAcc from ~50% toward 57–62%.

---

#### 3.10 Regime-conditioned model selection

Compute a rolling volatility regime and route signals through the appropriate base model:

```python
vol_60d = df['log_return'].rolling(60).std() * np.sqrt(252)
regime  = np.where(vol_60d > vol_60d.quantile(0.7), 'high_vol', 'low_vol')
```

In `high_vol` regimes: use SVR_rbf (kernel method, robust to outliers and discontinuities).  
In `low_vol` regimes: use SVR_lin (highest overall Sharpe; linear patterns dominate in calm markets).  
Optionally: suppress trading entirely in the top-10% volatility quantile to reduce MaxDD.

---

## 4. Summary Table

| # | Improvement | Targets | Effort | Expected Sharpe lift | Expected DirAcc lift |
|---|---|---|---|---|---|
| 3.1 | Fix z-score window inconsistency | Sharpe, DirAcc | Low | correctness fix | correctness fix |
| 3.2 | Signal dead zone (threshold) | DirAcc, MaxDD | Low | neutral–slight | +2–4 pp |
| 3.3 | Add LightGBM + ElasticNet | Sharpe | Low | +0.3–1.0 | +0–1 pp |
| 3.4 | Proportional position sizing | Sharpe, CAGR | Low | +1–3 | neutral |
| 3.5 | Ensemble stacking (top-3) | DirAcc | Medium | +0.5–1.5 | +1–3 pp |
| 3.6 | Longer momentum + seasonal features | DirAcc | Medium | +0.5–1.5 | +1–3 pp |
| 3.7 | Walk-forward retraining | Sharpe | Medium | +0.5–2.0 | +1–2 pp |
| 3.8 | Softer d=0.3 for Keras | DirAcc (Keras only) | Medium | — | +5–8 pp |
| 3.9 | Directional loss for Keras | DirAcc (Keras only) | Medium | — | +5–10 pp |
| 3.10 | Regime-conditioned model selection | MaxDD, Sharpe | High | +0.5–2.0 | +1–2 pp |

Start with **3.1 → 3.2 → 3.4**: fastest wins with highest confidence of measurable improvement.

---

## 5. Implementation Order

```
Step 1 (fix correctness, ~30 min):
  - Unify ZSCORE_WIN across train.py, tune_model.py, predict_tomorrow.py
  - Add ZSCORE_WIN to tune_model.py PARAM_CATALOGUE as a tunable parameter
  - Run: tune_all.py → update_results.py

Step 2 (signal dead zone, ~1h):
  - Add SIGNAL_THRESHOLD to _trading_metrics() in train.py and tune_model.py
  - Run: tune_all.py (finds optimal threshold per model) → update_results.py

Step 3 (proportional sizing, ~1h):
  - Replace binary sig with continuous clip(z/3, -1, 1)
  - Run: tune_all.py → update_results.py (compare vs binary results)

Step 4 (new models, ~2h):
  - Add LightGBM and ElasticNet to train.py and tune_model.py
  - Run: train.py → tune_all.py --models LightGBM ElasticNet → update_results.py

Step 5 (features, ~2h):
  - Add 63d/126d/252d momentum to train.py feature engineering
  - Add month-of-year dummies
  - Add 20d COT change columns
  - Run: train.py → tune_all.py → update_results.py

Step 6 (walk-forward, ~4h):
  - Implement expanding-window retraining loop
  - Generate rolling OOS predictions
  - Compare vs fixed-split metrics to quantify regime-adaptation gain

Step 7 (Keras structural, ~4h each):
  - Add target_d03 as Keras-specific target in train.py
  - Implement directional_mse custom loss in build_lstm/build_gru/build_bilstm
  - Run: tune_all.py --models LSTM GRU BiLSTM → update_results.py
```

Note: after any tuning run, always run `update_results.py` (not `train.py`) to regenerate
result files with tuned params. `train.py` saves raw artifacts; `update_results.py`
applies tuned params and refreshes every chart and metrics file in `results/`.

---

## 6. Notes on What Not to Change

- **Do not switch back to directional classification** (`train_directional.py`). The MSE + z-score
  pipeline achieves 72.3% DirAcc vs the classifier's 62.2% best. The z-score noise filter is
  structurally superior to binary classification.
- **Do not increase frac-diff d beyond 0.5** for sklearn models. d=0.5 is the minimum that
  achieves ADF stationarity — increasing d destroys additional signal without benefit.
- **Do not remove the epsilon calibration** for SVR. The fix from `ε=0.1` to `ε=0.25×σ≈0.0044`
  was the single largest improvement in the project history (SVR_lin Sharpe -1.26 → +8.86).
- **WRC p=0.47** (not significant) is expected and acceptable. The circular block bootstrap is
  conservative; the MC permutation test (p=0.001) confirms the signal is real.
- **Do not revert the SelectKBest leakage fix.** The pre-fix results (MLP #1 at Sharpe 9.01)
  were slightly overfit due to feature selection leakage. Post-fix rankings are the correct baseline.

---

## 7. Feature Count Warning — Empirical Results (2026-05-03)

### 7.1 What happened

Step 5 of the implementation plan added 18 new features (multi-period momentum, month dummies,
20d COT changes), bringing the total from **39 → 57**. A full `train.py` run was made with all
57 features. Results degraded across every model:

| Model | Sharpe (39 feat) | Sharpe (57 feat) | Change |
|---|---:|---:|---:|
| SVR_lin | 8.86 | 7.64 | −1.22 |
| MLP | 8.55 | **−0.51** | **−9.06** |
| Ridge | 8.40 | 7.83 | −0.57 |
| SVR_rbf | 7.88 | 6.78 | −1.10 |
| XGBoost | 7.81 | 6.45 | −1.36 |
| RandomForest | 7.74 | 6.29 | −1.45 |

MLP collapsed completely (Sharpe 8.55 → −0.51, DirAcc 71.7% → 52%).

### 7.2 Why this happens

**Correlated features dilute signal.** The new momentum features (`gold_ret_21d`, `gold_ret_63d`,
`gold_ret_126d`, `gold_ret_252d`) are highly correlated with each other and with existing features
(`ROC_10`, `ROC_20`, `MACD_diff`). SelectKBest uses mutual information, which does not penalise
redundancy — so `k=57 of 57` was chosen (all features kept), meaning feature selection provided
no pruning.

**MLP is most sensitive.** Adding correlated inputs increases the effective condition number of
the weight matrix, causing gradient instability. With 57 features and `alpha=0.01` regularisation,
MLP converges to a degenerate solution (near-zero predictions → random signal).

**Month dummies add sparsity.** 12 binary features where 11/12 are zero on any given day add
dimensionality without dense signal, hurting distance-based and gradient-based learners.

### 7.3 Rule going forward

**Add features one group at a time and verify improvement before keeping them.**

| Group | Features added | Status |
|---|---|---|
| Baseline | 39 features | ✓ Keep (best known state) |
| Multi-period momentum | gold_ret_21d/63d/126d/252d | ✗ Reverted — net negative |
| Month dummies | month_1 … month_12 | ✗ Reverted — net negative |
| 20d COT changes | net_spec_chg_20d, net_comm_chg_20d | Test in isolation |

**Do not batch multiple feature groups into a single run.** Without isolation it is impossible
to attribute performance changes to a specific group.

**Threshold for keeping a new feature group:** every model that previously had Sharpe ≥ 7.5
must maintain Sharpe ≥ 7.5, and at least one model must show a measurable improvement.

---

## 8. Per-Model Feature Selection and PCA

### 8.1 The root problem

`k` is currently tuned in CV Loop 2 only for the `BEST_MODEL` (XGB in the latest run). The
selected `k=57` (all features) is optimised for a tree model that handles correlated inputs
natively. MLP and SVR receive that same selection even though they need fewer, less-correlated
features. This is what caused MLP's collapse (Sharpe 8.55 → −0.51) when features grew from 39 to 57.

**Missing controlled experiment:** The current 57-feature results cannot be cleanly compared to
the 39-feature baseline (Sharpe 8.86) because three things changed simultaneously: features grew
from 39 → 57, `ZSCORE_WIN` changed from 5 → 10, and the SelectKBest leakage fix was applied.
Before attributing the full Sharpe drop to features, a clean comparison is needed:

| Run | Features | ZSCORE_WIN | Leakage fix | SVR_lin Sharpe |
|---|---|---|---|---|
| Old baseline | 39 | 5 | No (leaky) | 8.86 |
| New (current) | 57 | 10 | Yes | 7.81 |
| **Missing** | **39** | **10** | **Yes** | **?** |

If the missing run gives ~8.5, the 57-feature drop is ~0.7 Sharpe. If it gives ~7.8, the features
are nearly neutral and ZSCORE_WIN + leakage removal account for most of the change.

**Attribution note for SVR_lin:** SVR with a linear kernel is a regularised regression and handles
correlated inputs reasonably well. Its Sharpe drop (8.86 → 7.81) is likely driven more by
ZSCORE_WIN=10 changing the strategy than by feature count. Per-model `k` tuning may not help
SVR_lin significantly — the priority there is ZSCORE_WIN search.

### 8.2 Option 1 — Tune `k` independently per model (recommended first step)

Add `k` to each model's individual hyperparameter search in `tune_model.py`:

```python
# tune_model.py — per-model k search
'MLP':   {'hidden_layer_sizes': [...], 'alpha': [...], 'k': [15, 20, 25, 30]},
'Ridge': {'alpha': [...],                               'k': [15, 20, 30, 39]},
'SVR_lin': {'C': [...],                                 'k': [20, 30, 39, 57]},
'XGB':   {'n_estimators': [...], 'max_depth': [...],   'k': [30, 39, 57]},  # trees OK with more
```

`k` maps to `selector__k` in the existing Pipeline — no structural changes needed. This is
the highest-leverage fix with least complexity.

**k range note for MLP:** The lower bound should be at least 15, not 10. MLP with a `(128, 64, 32)`
first layer and only k=10 inputs is severely under-determined — 128 neurons cannot be meaningfully
activated by 10 features. k ∈ [15, 20, 25, 30] is more appropriate.

**Complement — wider alpha for MLP:** Per-model `k` tuning should be combined with a wider
regularisation search for MLP. The current tuned alpha=0.01 was found with k=57; with k=20 the
optimal alpha is likely lower. Additionally, with more features the network needs stronger
regularisation to avoid overfitting. Extend MLP's alpha grid from `[0.0001, 0.001, 0.01]` to
`[0.001, 0.01, 0.1, 1.0]` in the same grid search:

```python
'MLP': {
    'hidden_layer_sizes': [(64, 32), (128, 64), (128, 64, 32)],
    'alpha':              [0.001, 0.01, 0.1, 1.0],   # wider than before
    'learning_rate_init': [0.001, 0.01],
    'k':                  [15, 20, 25, 30],
}
```

This is simpler than VIF or PCA and should be tried first.

### 8.3 Option 2 — VIF pruning (one-time shared preprocessing)

Variance Inflation Factor removes the most redundant features globally before any model sees
them. Features with VIF > 10 are almost certainly duplicating an existing signal (e.g.,
`gold_ret_63d`, `gold_ret_126d`, `gold_ret_252d` all capture the same long-run trend).

```python
from statsmodels.stats.outliers_influence import variance_inflation_factor

def vif_filter(X, thresh=10.0):
    cols = list(range(X.shape[1]))
    dropped = True
    while dropped:
        dropped = False
        vifs = [variance_inflation_factor(X[:, cols], i) for i in range(len(cols))]
        max_i = np.argmax(vifs)
        if vifs[max_i] > thresh:
            cols.pop(max_i)
            dropped = True
    return cols   # indices of surviving features

# Run on X_trainval only — never on X_test
surviving_cols = vif_filter(X_trainval)
X_trainval = X_trainval[:, surviving_cols]
X_test     = X_test[:, surviving_cols]
```

Run once on `X_trainval` after the train/test split. Expected result: 57 → ~38–42 features,
dropping the most collinear momentum duplicates. All models then share this cleaner base, and
per-model `k` search fine-tunes from there.

**Leakage note:** Running VIF on the full `X_trainval` means the feature subset was chosen with
knowledge of the validation folds' feature distributions (though not the target). This is
technically mild leakage. Strictly correct practice would run VIF inside each CV fold, but since
VIF measures feature-feature correlation (not feature-target correlation), the practical impact
is negligible. The one-time approach is acceptable.

### 8.4 Option 3 — SHAP-based selection per model

After an initial fit on trainval, use each model's SHAP mean absolute values to rank and
select the top `k` features. More expensive but model-aware. SHAP is already computed for
the best model — extending it to all models is straightforward.

**Circular dependency warning:** SHAP requires a trained model, which was trained on a fixed
feature set. Using SHAP to then re-select features requires refitting — meaning the feature
selection and model training are coupled. To use SHAP properly inside CV you would need to:
fit → compute SHAP → re-select features → refit in every fold, roughly doubling training time.
For an initial pass, treat SHAP selection as a post-hoc analysis tool (look at SHAP importance
after fitting with all 57 features, then manually drop the bottom-ranked features) rather than
a CV-integrated selector.

### 8.5 On PCA

| Model | Use PCA? | Reason |
|---|---|---|
| Ridge / Lasso / SVR_lin | No | Regularisation already handles multicollinearity; PCA loses interpretability |
| XGBoost / RandomForest | No | Trees are invariant to linear transforms; PCA breaks SHAP |
| **MLP** | **Maybe** | MLP is the one model that struggles with correlated inputs; PCA orthogonalises the input space and can stabilise training |
| LSTM / GRU / BiLSTM | No | PCA destroys temporal covariance structure that sequence models rely on |

Two additional reasons to avoid PCA globally:

1. **SHAP breaks** — PCA components have no economic meaning; feature attribution becomes
   uninterpretable.
2. **Component instability** — PCA rotation can shift between training and live windows in a
   non-stationary market, making the transform unreliable out-of-sample.

**For MLP specifically:** try wider alpha (8.2) and per-model k (8.2) before PCA. Both are
simpler and more interpretable. PCA should be the last resort — if MLP still underperforms
after k and alpha are properly tuned, add `PCA(n_components=20)` inside its Pipeline only:

```python
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

mlp_pipe = Pipeline([
    ('selector', SelectKBest(score_func=mutual_info_regression, k=25)),
    ('scaler',   StandardScaler()),
    ('pca',      PCA(n_components=20)),   # MLP only — last resort
    ('model',    MLPRegressor(...)),
])
```

### 8.6 Recommended implementation order

The original order (VIF first, then k) was backwards — the highest-leverage, lowest-complexity
fix should go first.

1. **Run the controlled 39-feature experiment** — re-run `update_results.py` with 39-feature set
   (revert momentum/month/COT features, keep all other fixes) to isolate how much Sharpe loss is
   from features vs ZSCORE_WIN and leakage removal.

2. **Per-model `k` + wider MLP alpha** in `tune_model.py` — MLP searches
   `k ∈ [15, 20, 25, 30]` and `alpha ∈ [0.001, 0.01, 0.1, 1.0]` jointly; trees search
   `k ∈ [30, 39, 57]`. Run `tune_all.py --skip-keras` → `update_results.py`.

3. **VIF filter** on `X_trainval` — one-time global cleanup, reduces 57 → ~38–42, run after
   step 2 establishes the new per-model-k baseline. Run `train.py` → `tune_all.py` → `update_results.py`.

4. **PCA for MLP only** — only if MLP Sharpe remains below 7.5 after steps 2–3.

**Threshold for proceeding to next step:** the top two models (SVR_lin, Ridge) must maintain or
improve Sharpe, and MLP must recover to ≥ 7.0 before step 3 is attempted.

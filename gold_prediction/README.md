# Gold Price Prediction — Project README

A full ML pipeline for daily gold (GC=F) price direction prediction.
Models predict fractionally differenced log-gold returns and convert predictions
to long/short trading signals via rolling z-score.

---

## Quick Start

```bash
# 1. Environment
source /Users/jixu/Documents/study/APS1052H/option_6/venv/bin/activate

# 2. Download fresh market data
python download_data.py

# 3. Full train (all models, ~15 min)
python train.py

# 4. Tune all models (parallel, ~2 h including Keras)
python tune_all.py

# 5. Regenerate all result files with tuned params
python update_results.py

# 6. Tomorrow's signal (no retrain needed)
python predict_tomorrow.py
```

---

## File Overview

| Script | Purpose | Runtime |
|---|---|---|
| `download_data.py` | Download raw OHLCV + COT data → `data/` | ~30s |
| `train.py` | Full training pipeline: features, CV, all models, save artifacts | ~15 min |
| `tune_model.py` | Fine-tune a single model with custom grid search | varies |
| `tune_all.py` | Fine-tune all models in parallel subprocesses | ~2 h |
| `update_results.py` | Re-fit all models with tuned params, regenerate every result file | ~2 min |
| `plot_tuned.py` | Quick equity + predicted-vs-actual charts from tuned params | ~1 min |
| `plot_all.py` | Regenerate individual charts from saved artifacts | ~1 min |
| `predict_tomorrow.py` | Load saved models → generate next-day signal for all models | ~2 min |

---

## Scripts

---

### `download_data.py`

Downloads all raw data via `yfinance` and the `cot_reports` library.

**Outputs** (written to `data/`):

| File | Source | Content |
|---|---|---|
| `gold_daily.csv` | yfinance `GC=F` | OHLCV daily gold futures |
| `dxy_daily.csv` | yfinance `DX-Y.NYB` | US Dollar Index |
| `vix_daily.csv` | yfinance `^VIX` | CBOE Volatility Index |
| `tnx_daily.csv` | yfinance `^TNX` | 10-year Treasury yield |
| `tyx_daily.csv` | yfinance `^TYX` | 30-year Treasury yield |
| `cot_gold_weekly.csv` | CFTC via `cot_reports` | Weekly COT positions for COMEX 100-oz gold |
| `merged_gold_dataset.csv` | All merged | Final merged dataset (COT forward-filled to daily) |

**COT filter:** `"GOLD - COMMODITY EXCHANGE INC."` only (excludes MINI, MICRO, GOLDMAN-SACHS contracts).

```bash
python download_data.py
```

---

### `train.py`

Full training pipeline. Runs all models, selects the best, saves artifacts (no charts).
Charts are generated separately by `update_results.py` or `plot_all.py`.

**Pipeline steps:**

| Step | Description |
|---|---|
| 1. Load data | Read `merged_gold_dataset.csv` |
| 2. Stationarity | ADF test; fractional differencing at `d=0.5` |
| 3. Feature engineering | 39 features: COT, macro, technical indicators, target lags |
| 4. Train/test split | 80/20 by date (~2007–2022 train, 2022–2026 test) |
| 5. CV Loop 1 | 7 sklearn models × 5-fold TSplit in parallel; LSTM/GRU/BiLSTM timestep search |
| 6. CV Loop 2 | GridSearchCV for best model; SelectKBest inside Pipeline (no leakage) |
| 7. Final model | Fit best model on full trainval, evaluate on test |
| 8. Trading metrics | Rolling z-score signal → CAGR, Sharpe, Profit Factor, Max DD, Signal Dir Acc |
| 9. All-model metrics | Fit and evaluate all models; save `all_model_metrics.csv` |
| 9a. SVR tuning | Dedicated C/gamma grid search for SVR_lin and SVR_rbf (always runs) |
| 10. Save models | Save all `.keras` / `.joblib` files + `model_metadata.json` |
| 11. SHAP | Compute and save SHAP values for best model |
| 12. Bias tests | Compute and save White Reality Check + MC Permutation Test data |

**Key config (top of file):**

```python
TIMESTEPS_CANDS = [5, 7, 10, 15]   # look-back windows searched for Keras models
N_CV_FOLDS      = 5                 # TimeSeriesSplit folds
N_JOBS          = -1                # parallel cores (-1 = all)
```

**SVR epsilon** is always auto-calibrated: `epsilon = 0.25 × std(y_trainval)` ≈ 0.0044.
The default `epsilon=0.1` is ~23× too large for the frac-diff series scale and causes
constant predictions (pred_std ≈ 0).

**CV Loop 2 — SelectKBest leakage fix (2026-05-02):** Feature selection is done inside
the `Pipeline` passed to `GridSearchCV`, so `SelectKBest` is re-fit on the training
fold of each CV split. Previously it was fit on all of trainval before splitting,
slightly inflating CV MSE estimates. X_test was never affected.

**Outputs:**

| Path | Content |
|---|---|
| `results/all_model_metrics.csv` | MSE, CAGR, Sharpe, PF, MaxDD, DirAcc per model |
| `saved_models/model_metadata.json` | Full pipeline config + best params + SVR tuning results |
| `saved_models/best_model_XGB.joblib` | Best CV model (XGBoost, k=39 all features) |
| `saved_models/model_*.joblib / *.keras` | All individual models |

```bash
python train.py
```

---

### `tune_model.py`

Fine-tunes a single model with full GridSearchCV and reports test-set trading metrics.

**Supported models and tunable parameters:**

| Model | Fixed | Tunable parameters (default grid) |
|---|---|---|
| `Ridge` | — | `alpha` [0.0001 → 100] |
| `Lasso` | — | `alpha` [0.00001 → 1] |
| `SVR_lin` | `epsilon=0.25×std`, `max_iter=10000` | `C` [0.01 → 1000] |
| `SVR_rbf` | `epsilon=0.25×std`, `kernel=rbf` | `C` [0.01 → 100], `gamma` [scale/0.001/0.01/0.1] |
| `XGBoost` | `random_state=42` | `n_estimators`, `max_depth`, `learning_rate`, `subsample` |
| `RandomForest` | `random_state=42` | `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf` |
| `MLP` | `max_iter=500` | `hidden_layer_sizes`, `alpha`, `learning_rate_init` |
| `LSTM` | `Adam(lr=0.001)`, `epochs=50`, `patience=10` | `timesteps`, `units`, `dropout`, `recurrent_dropout` |
| `GRU` | `Adam(lr=0.001)`, `epochs=50`, `patience=10` | `timesteps`, `units`, `dropout`, `recurrent_dropout` |
| `BiLSTM` | `Adam(lr=0.001)`, `epochs=60`, `patience=10` | `timesteps`, `units`, `dropout`, `recurrent_dropout` |

**Results saved to:** `saved_models/tuned_<ModelName>.json`

```bash
# Tune with default grid
python tune_model.py --model SVR_rbf

# Override specific grid values
python tune_model.py --model SVR_rbf --param C=[0.1,1,10,100] --param gamma=[0.001,0.01,0.1]
python tune_model.py --model XGBoost --param n_estimators=[100,200,300] --param max_depth=[2,3,4]
python tune_model.py --model LSTM    --param timesteps=[5,10,15] --param units=[32,64]

# Limit parallelism (useful when called inside tune_all.py)
python tune_model.py --model Ridge --n-jobs 2

# Print full parameter catalogue
python tune_model.py --help-params
```

---

### `tune_all.py`

Runs `tune_model.py` for all (or selected) models in **parallel subprocesses**.
Each model gets its own independent process and log file.

**Parallelism:**
- Sklearn models: `--jobs-per-model 2` cores each (default)
- Keras models: `--n-jobs 1`, TF manages its own thread pool internally
- No shared state between processes — results written to separate `tuned_<name>.json` files

**At completion:** prints a combined summary table sorted by Sharpe, then merges all
per-model results into `model_metadata.json`.

```bash
# Tune all 10 models in parallel
python tune_all.py

# Tune a subset
python tune_all.py --models SVR_lin SVR_rbf
python tune_all.py --models Ridge Lasso XGBoost RandomForest

# Skip Keras models (faster, sklearn only)
python tune_all.py --skip-keras

# Monitor individual model logs
tail -f /tmp/tune_gold_logs/tune_XGBoost.log
tail -f /tmp/tune_gold_logs/tune_LSTM.log
```

**Output files:**
- `/tmp/tune_gold_logs/tune_<name>.log` — per-model stdout during tuning
- `saved_models/tuned_<name>.json` — per-model best params + test metrics
- `saved_models/model_metadata.json` — merged at end

---

### `update_results.py`

Re-fits all 10 models using tuned params from `tuned_<name>.json`, then regenerates
every active result file in `results/` in a single consistent run.

**Run this after `tune_all.py` completes, or after any model change.**

**Outputs:**

| File | Description |
|---|---|
| `results/all_model_metrics.csv` | Updated metrics for all 10 models |
| `results/all_model_metrics_table.png` | Styled visual table sorted by Sharpe |
| `results/equity_curves_all_models.png` | 3-panel: gold price / equity curves / consensus |
| `results/predicted_vs_actual_all_models.png` | 10-panel predicted vs actual grid |
| `results/signal_timeline.png` | Per-model LONG/SHORT strips with equity overlay |
| `results/shap_summary.png` | SHAP beeswarm for best model (by Sharpe) |
| `results/bias_tests.png` | WRC + MC permutation test for best model |
| `results/final_summary.txt` | One-page text summary |

```bash
python update_results.py
```

---

### `plot_tuned.py`

Quick chart generation using tuned params — produces only the two main comparison charts
without recomputing SHAP or bias tests. Faster than `update_results.py`.

```bash
python plot_tuned.py
```

---

### `predict_tomorrow.py`

Loads saved models and generates the next-day trading signal for all models.
Does **not** require a retrain.

**What it does:**
1. Loads `model_metadata.json` for feature list, scaler params, timesteps
2. Downloads latest data (or uses existing CSVs)
3. Re-fits sklearn models on trainval
4. Loads Keras weights from `.keras` files
5. Computes rolling z-score signal over last `ZSCORE_WIN=10` rows
6. Weighted consensus using historical Sharpe as weights (positive-Sharpe models only)

```bash
python predict_tomorrow.py
```

---

## Data Flow

```
download_data.py
    └─→ data/merged_gold_dataset.csv
            │
            ▼
        train.py
            ├─→ saved_models/model_metadata.json
            ├─→ saved_models/best_model_XGB.joblib  (k=39, leakage-fixed)
            ├─→ saved_models/model_*.joblib / *.keras
            └─→ results/all_model_metrics.csv
            │
            ├── predict_tomorrow.py  (reads saved_models/, no retrain)
            ├── tune_all.py
            │       └── tune_model.py × 10 (parallel)
            │               └─→ saved_models/tuned_<name>.json
            └── update_results.py   ← run after tune_all.py
                    └─→ results/*.png + results/all_model_metrics.csv
```

---

## Models (tuned + leakage-fixed — 2026-05-02)

Results from `update_results.py` using tuned params from `tune_all.py`.
XGBoost uses all 39 features (k=39) after SelectKBest leakage fix.

| Rank | Model | Sharpe | DirAcc | MaxDD | Tuned Parameters |
|---:|---|---:|---:|---:|---|
| 1 | **SVR_lin** ★ | 8.86 | 72.3% | -6.1% | C=0.01, ε=0.25×σ |
| 2 | **MLP** ★ | 8.55 | 71.7% | -6.1% | hidden=[128,64,32], α=0.01, lr=0.01 |
| 3 | **Ridge** ★ | 8.40 | 70.5% | -6.1% | α=100 |
| 4 | **SVR_rbf** ★ | 7.88 | 69.6% | -6.1% | C=0.01, γ=0.001, ε=0.25×σ |
| 5 | **XGBoost** ★ | 7.81 | 69.6% | -7.2% | lr=0.05, depth=2, n=200, sub=1.0 |
| 6 | **RandomForest** ★ | 7.74 | 68.7% | -7.2% | n=50, max_depth=10, min_split=10 |
| 7 | **Lasso** ★ | 7.45 | 68.7% | -6.1% | α=0.001 |
| — | **Buy & Hold** | 1.26 | 55.0% | -17.7% | baseline |
| 8 | BiLSTM | 0.71 | 52.4% | -27.9% | ts=10, units=16, drop=0.3, rdrop=0.2 |
| 9 | LSTM | 0.13 | 50.6% | -31.7% | ts=7, units=32, drop=0.5, rdrop=0.2 |
| 10 | GRU | 0.06 | 49.9% | -37.2% | ts=15, units=64, drop=0.4, rdrop=0.0 |

★ = Sharpe ≥ 7.5. All 7 tuned sklearn/MLP models beat buy & hold by ≥6×.

> **SVR_lin now #1.** With epsilon calibrated to 0.25×σ ≈ 0.0044 and C=0.01,
> LinearSVR outperforms all other models. The default epsilon=0.1 was ~23× too large,
> producing constant predictions (pred_std ≈ 0) and Sharpe −1.26 before the fix.

> **Why linear/SVR models dominate:** Fractional differencing (d=0.5) removes most
> temporal structure — the resulting series is nearly i.i.d. Linear and kernel models
> are optimal for such data. Sequential models (LSTM/GRU/BiLSTM) add inductive bias
> that introduces noise rather than signal on this type of series.

> **Note on Keras variance:** Keras model Sharpe figures vary slightly between runs
> (~±0.3) due to TensorFlow non-determinism. Sklearn results are deterministic.

---

## Experiment: Directional Classification Pipeline (2026-05-02) — Abandoned

A classification pipeline (`train_directional.py`) was built to directly maximise
Signal Direction Accuracy. **Conclusion: inferior to the MSE + z-score pipeline.**

| Model | Sharpe | CAGR | DirAcc |
|---|---:|---:|---:|
| **RF_clf** | 4.49 | 128.9% | **62.2%** |
| **XGB_clf** | 3.02 | 76.6% | 56.2% |
| Buy & Hold | 1.26 | 27.2% | 55.0% |
| All others | < 0.5 | < 10% | < 49% | |

**Why it loses:** The z-score acts as a noise filter — it only trades when a prediction
deviates meaningfully from its recent mean. Binary classifiers treat every day equally,
discarding the confidence information encoded in prediction magnitude.

> Do not use `train_directional.py`. Results archived in `saved_models/directional/`.

---

## Data Integrity

| Check | Status |
|---|---|
| StandardScaler fit on test data | Safe — always fit on training fold only |
| SelectKBest in CV Loop 2 | Fixed 2026-05-02 — now inside Pipeline, re-fit per fold |
| XGBoost best_model retrained | Fixed 2026-05-02 — k=39 (all features), not leaky k=30 |
| Feature engineering (rolling windows) | Safe — all `center=False`, strictly backward-looking |
| Fractional differencing | Safe — backward window `iloc[i-width+1 : i+1]` |
| X_test ever passed to fit() | Safe — never |
| TimeSeriesSplit used throughout | Safe — temporal order preserved in all CV |

---

## Saved Files

### `saved_models/`

| File | Description |
|---|---|
| `model_metadata.json` | Full pipeline config: feature cols, best params, timesteps, dates |
| `best_model_XGB.joblib` | Best CV model (XGBoost, k=39 all features, leakage-fixed 2026-05-02) |
| `final_scaler.joblib` | StandardScaler fit on trainval for best model |
| `model_Ridge.joblib` | Ridge (unfitted stub — re-fitted in predict_tomorrow.py) |
| `model_Lasso.joblib` | Lasso |
| `model_RandomForest.joblib` | RandomForest |
| `model_XGBoost.joblib` | XGBoost |
| `model_MLP.joblib` | MLP |
| `model_SVR_lin.joblib` | LinearSVR |
| `model_SVR_rbf.joblib` | SVR (rbf) |
| `model_LSTM.keras` | LSTM (best_ts=7) |
| `model_GRU.keras` | GRU (best_ts=15) |
| `model_BiLSTM.keras` | BiLSTM (best_ts=10) |
| `all_models_scaler.joblib` | StandardScaler fit on trainval (used by all-model eval) |
| `tuned_<Name>.json` | Per-model tuning result (written by tune_model.py) |

### `results/`

| File | Description |
|---|---|
| `all_model_metrics.csv` | Sharpe, CAGR, DirAcc, PF, MaxDD, MSE per model |
| `all_model_metrics_table.png` | Styled visual table of metrics, sorted by Sharpe |
| `equity_curves_all_models.png` | 3-panel: gold price / equity curves / consensus bar |
| `predicted_vs_actual_all_models.png` | 10-panel predicted vs actual (DirAcc in titles) |
| `signal_timeline.png` | Per-model LONG/SHORT strips with equity overlay |
| `shap_summary.png` | SHAP beeswarm for best model (SVR_lin) |
| `bias_tests.png` | WRC + MC permutation test for best model |
| `final_summary.txt` | One-page text summary |

---

## Environment

```bash
# Python 3.11 (venv)
/Users/jixu/Documents/study/APS1052H/option_6/venv/bin/python3

# Key packages
yfinance, cot_reports, pandas, numpy, scikit-learn
xgboost, tensorflow, shap, pmdarima, matplotlib, joblib
```

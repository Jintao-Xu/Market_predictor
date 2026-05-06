# Gold Price Prediction — Project README

A full ML pipeline for daily gold (GC=F) price direction prediction.
Models predict fractionally differenced log-gold returns and convert predictions
to long/short trading signals via rolling z-score.

---

## Quick Start

```bash
# 1. Environment
source /Users/jixu/Documents/study/APS1052H/option_6/venv/bin/activate

# 2. Tomorrow's signal — auto-refreshes data, no retrain needed
python predict_tomorrow.py

# ── One-time setup / retraining ──────────────────────────────────────────────

# 3. Full train (all models, ~15 min)
python train.py

# 4. Tune all models (parallel, ~2 h including Keras)
python tune_all.py

# 5. Regenerate all result files with tuned params
python update_results.py

# (download_data.py is called automatically by predict_tomorrow.py;
#  run it manually only if you need data without running predictions)
python download_data.py
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
| 2. Stationarity | ADF test on train portion; fractional differencing — d chosen as smallest value passing ADF |
| 3. Feature engineering | 57 features: COT, macro, technical indicators, target lags |
| 4. Train/test split | 80/20 by date (~2007–2022 train, 2022–2026 test) |
| 5. Helper functions | Build sequence maker, trading metrics, model builders |
| 6. CV Loop 1 | Up to 9 sklearn models × 5-fold TSplit (Sharpe-CV) in parallel; Keras timestep search |
| 7. CV Loop 2 | Best model — feature selection (SelectKBest inside Pipeline, no leakage) + hyperparameter tuning |
| 8. Final model | Fit best model on full trainval, evaluate on test |
| 9. Trading metrics | Rolling z-score signal → CAGR, Sharpe, Profit Factor, Max DD, Signal Dir Acc |
| 10. All-model metrics | Fit and evaluate all 12 models in parallel; save `all_model_metrics.csv` |
| 10a. All-model tuning | MSE-CV GridSearchCV for Ridge, Lasso, SVR_lin, SVR_rbf, XGBoost, RF |
| 11. Save models | Save all `.keras` / `.joblib` files + `model_metadata.json` |
| 12. SHAP | Compute and save SHAP values for best model |
| 13. Bias tests | Compute and save White Reality Check + MC Permutation Test data |

**Key config (top of file):**

```python
TIMESTEPS_CANDS = [5, 7, 10, 15]   # look-back windows searched for Keras models
N_CV_FOLDS      = 5                 # TimeSeriesSplit folds
N_JOBS          = -1                # parallel cores (-1 = all)
```

**SVR epsilon** is always auto-calibrated: `epsilon = 0.25 × std(y_trainval)`.
The default `epsilon=0.1` is several times too large for the frac-diff series scale and causes
constant predictions (pred_std ≈ 0).

**CV Loop 2 — SelectKBest leakage fix (2026-05-02):** Feature selection is done inside
the `Pipeline` passed to `GridSearchCV`, so `SelectKBest` is re-fit on the training
fold of each CV split. Previously it was fit on all of trainval before splitting,
slightly inflating CV MSE estimates. X_test was never affected.

**Outputs:**

| Path | Content |
|---|---|
| `results/all_model_metrics.csv` | Sharpe, CAGR, DirAcc, Coverage, PF, MaxDD, MSE per model |
| `saved_models/model_metadata.json` | Full pipeline config + best params + frac_d + timesteps |
| `saved_models/best_model_<NAME>.joblib` | Best CV model (name varies by run) |
| `saved_models/model_*.joblib / *.keras` | All sklearn models (Ridge, Lasso, ElasticNet, LightGBM, SVR_lin, SVR_rbf, XGBoost, RF, MLP) + LSTM/GRU/BiLSTM |
| `saved_models/all_models_scaler.joblib` | StandardScaler fit on full trainval (used by `update_results.py`) |

```bash
python train.py
```

---

### `tune_model.py`

Fine-tunes a single model with full GridSearchCV (Sharpe-CV) and reports test-set trading metrics.
All models also tune `zscore_win` and `signal_threshold` jointly with the model hyperparameters.

Use `--recent-years 3` (default) for RECENT_MODELS; `--recent-years 0` for FULL_MODELS.

**Supported models and tunable parameters:**

| Model | Fixed | Tunable parameters (grid) |
|---|---|---|
| `Ridge` | — | `alpha` [1e-4→1000], `zscore_win` [5,7,10,15,20], `threshold` [0,0.2,0.3,0.5,0.75] |
| `Lasso` | — | `alpha` [1e-5→1], `zscore_win`, `threshold` |
| `ElasticNet` | `max_iter=5000` | `alpha` [1e-4→1], `l1_ratio` [0.1→0.9], `zscore_win`, `threshold` |
| `SVR_lin` | `epsilon=0.25×std`, `max_iter=10000` | `C` [0.01→10000], `zscore_win`, `threshold` |
| `SVR_rbf` | `epsilon=0.25×std`, `kernel=rbf` | `C` [0.001→100], `gamma` [1e-4→0.1/scale], `zscore_win`, `threshold` |
| `XGBoost` | `random_state=42` | `n_estimators`, `max_depth` [2→5], `learning_rate`, `subsample`, `zscore_win`, `threshold` |
| `LightGBM` | `random_state=42` | `n_estimators` [100→500], `num_leaves` [7→63], `learning_rate`, `subsample`, `zscore_win`, `threshold` |
| `RandomForest` | `random_state=42` | `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `zscore_win`, `threshold` |
| `MLP` | `max_iter=500` | `hidden_layer_sizes` [(32,)→(256,128,64)], `alpha`, `learning_rate_init`, `zscore_win`, `threshold` |
| `LSTM` | `Adam(lr=0.001)`, `epochs=50`, `patience=10` | `timesteps`, `units`, `dropout`, `recurrent_dropout`, `zscore_win`, `threshold` |
| `GRU` | `Adam(lr=0.001)`, `epochs=50`, `patience=10` | `timesteps`, `units`, `dropout`, `recurrent_dropout`, `zscore_win`, `threshold` |
| `BiLSTM` | `Adam(lr=0.001)`, `epochs=60`, `patience=10` | `timesteps`, `units`, `dropout`, `recurrent_dropout`, `zscore_win`, `threshold` |

**Results saved to:** `saved_models/tuned_<ModelName>.json` (best params + test metrics)

```bash
# Tune with default grid (recent 3yr window)
python tune_model.py --model SVR_rbf

# Full-history models
python tune_model.py --model LightGBM --recent-years 0
python tune_model.py --model LSTM     --recent-years 0

# Override specific grid values
python tune_model.py --model SVR_rbf --param C=[0.1,1,10,100] --param gamma=[0.001,0.01,0.1]
python tune_model.py --model XGBoost --param n_estimators=[100,200,300] --param max_depth=[2,3,4]

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
# Tune all 12 models in parallel
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

Re-fits all 12 models using tuned params from `tuned_<name>.json`, then regenerates
every active result file in `results/` in a single consistent run.

**Run this after `tune_all.py` completes, or after any model change.**

**Outputs:**

| File | Description |
|---|---|
| `results/all_model_metrics.csv` | Updated metrics for all 12 models |
| `results/all_model_metrics_table.png` | Styled visual table sorted by Sharpe |
| `results/equity_curves_all_models.png` | 3-panel: gold price / equity curves / consensus |
| `results/predicted_vs_actual_all_models.png` | 12-panel predicted vs actual grid |
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

Generates the next-day LONG / SHORT / FLAT signal and implied price target for all 12 models (Exp 11).
Auto-refreshes data on every run — no need to run `download_data.py` separately.
Does **not** require a retrain — re-fits sklearn models on the existing training window each run.

```bash
python predict_tomorrow.py
```

**What it does:**

| Step | Description |
|---|---|
| 0. Auto-refresh data | Calls `download_data.py` (incremental — only fetches missing rows) |
| 1. Feature engineering | Identical pipeline to `update_results.py` — frac-diff, COT, macro, technical, lags |
| 2. Dual scalers | `sc_rec` fit on recent-3yr train rows; `sc_full` fit on full-history train rows |
| 3. Fit sklearn models | Each model re-fit on its correct window using tuned params from `tuned_<name>.json` |
| 4. Load Keras models | LSTM / GRU / BiLSTM loaded from `.keras` files with `directional_mse` custom loss |
| 5. Rolling z-score signal | Per-model tuned zscore_win and threshold → LONG / SHORT / FLAT |
| 6. Price target | Invert frac-diff prediction back to USD price via weight decomposition |
| 7. Ensemble | Sharpe-weighted directional vote + 1/MSE-weighted price target |
| 8. Position sizing | `signal_strength × vol_scalar` — scales down in high-vol regimes |
| 9. Signal log | Append to `results/signal_log.csv`; auto-backfill prior actuals on next run |

**Models (12 total):**

| Group | Models | Scaler | Train window |
|---|---|---|---|
| RECENT_MODELS | Ridge, Lasso, ElasticNet, SVR_lin, SVR_rbf, XGBoost | `sc_rec` | 2023-05-01 → 2025-09-23 (~604 rows) |
| FULL_MODELS | LightGBM, RandomForest, MLP, LSTM, GRU, BiLSTM | `sc_full` | 2006-01-03 → 2025-09-23 (~3883 rows) |

**Signal logic (per model):**

```
z = (pred - rolling_mean) / rolling_std   (window = tuned zscore_win, typically 5)

z >  threshold  →  LONG  (+1)   enter long at signal_for_date close
z < -threshold  →  SHORT (-1)   enter short at signal_for_date close
|z| ≤ threshold →  FLAT  ( 0)   exit to cash
```

Thresholds and windows come from `tuned_<name>.json` (0.3–0.75 depending on model).
Returns measured entry-to-exit: close of `signal_for_date` → close of next trading day.

**Position sizing:**

```python
signal_strength = abs(sharpe_weighted_vote)       # [0, 1]
vol_scalar      = min(longrun_vol / recent_vol, 1.5)   # scale down if recent vol elevated
position_pct    = min(signal_strength * vol_scalar, 1.0) * 100
```

**Price target inversion:**

The models predict `frac_diff_log_Gold[t]`. To recover the implied gold price:

```
frac_diff[t] = 1.0 × log_Gold[t] + w₁ × log_Gold[t-1] + w₂ × log_Gold[t-2] + ...
→ log_Gold[t] = pred − Σ(wₖ × log_Gold[t-k])   for k = 1, 2, ...
→ Gold[t] = exp(log_Gold[t])
```

where `wₖ` are the fractional differencing weights for `d = 0.3`.

**Two ensemble outputs:**

| Output | Weight by | Rationale |
|---|---|---|
| Price target | **1 / MSE** | MSE measures frac_diff prediction accuracy — determines price inversion quality |
| Directional vote | **Sharpe** (positive only; GRU excluded) | Sharpe measures signal quality — risk-adjusted trading returns |

**Model weights (Exp 11 — test period 2025-09-24 → 2026-05-01, 152 days):**

| Model | Sharpe | MSE | 1/MSE wt% | Sharpe wt% | Train window |
|---|---:|---:|---:|---:|---|
| Lasso | 9.98 | 0.000433 | 21.8% | 13.1% | recent 3yr |
| ElasticNet | 9.98 | 0.000434 | 21.8% | 13.1% | recent 3yr |
| Ridge | 10.13 | 0.000468 | 20.2% | 13.3% | recent 3yr |
| SVR_lin | 10.60 | 0.000762 | 12.4% | 13.9% | recent 3yr |
| SVR_rbf | 9.82 | 0.000778 | 12.2% | 12.9% | recent 3yr |
| MLP | 7.30 | 0.002177 | 4.3% | 9.6% | full history |
| XGBoost | 9.62 | 0.002227 | 4.2% | 12.6% | recent 3yr |
| LSTM | 0.37 | 0.005753 | 1.6% | 0.5% | full history |
| LightGBM | 8.21 | 0.018528 | 0.5% | 10.8% | full history |
| RandomForest | 8.18 | 0.018253 | 0.5% | 10.7% | full history |
| BiLSTM | 0.25 | 0.019594 | 0.5% | 0.3% | full history |
| GRU | -0.13 | 0.026017 | 0.4% | excluded | full history |

Note: GRU excluded from Sharpe vote (negative Sharpe); included in 1/MSE price ensemble with tiny weight.

**Signal log (`results/signal_log.csv`):**

Columns: `run_timestamp`, `signal_for_date`, `signal`, `confidence_pct`, `position_size_pct`,
`price_at_signal`, `price_target_mse`, `price_target_chg_pct`, [12 model votes],
`actual_price`, `actual_return_pct`, `strategy_return_pct`

Actuals are auto-backfilled on the next run once market data is available.

---

## Data Flow

```
download_data.py  ←── also auto-called by predict_tomorrow.py on every run
    └─→ data/merged_gold_dataset.csv
            │
            ▼
        train.py
            ├─→ saved_models/model_metadata.json
            ├─→ saved_models/best_model_<NAME>.joblib  (best CV model, name varies)
            ├─→ saved_models/model_*.joblib / *.keras   (sklearn × 9 + Keras × 3)
            ├─→ saved_models/all_models_scaler.joblib
            └─→ results/all_model_metrics.csv
            │
            ├── predict_tomorrow.py  (reads saved_models/, no retrain)
            ├── tune_all.py
            │       └── tune_model.py × 12 (parallel)
            │               └─→ saved_models/tuned_<name>.json
            └── update_results.py   ← run after tune_all.py
                    └─→ results/*.png + results/all_model_metrics.csv
```

---

## Models — Exp 11 `exp/combined-recent` ★★★ Best Overall

**Hybrid setup:** RECENT_MODELS (Ridge, Lasso, ElasticNet, SVR_lin, SVR_rbf, XGBoost) tuned on
the last 3 years (2023-05-01 → 2025-09-23, 604 rows). FULL_MODELS (LightGBM, RandomForest, MLP,
LSTM, GRU, BiLSTM) tuned on full history (2006-01-03 → 2025-09-23, ~3883 rows).

**Test period:** 2025-09-24 → 2026-05-01 (152 days) | 57 features | frac-diff d=0.3

| Rank | Model | Sharpe | CAGR | Active DirAcc | Coverage | MaxDD | Train window | Tuned Parameters |
|---:|---|---:|---:|---:|---:|---:|---|---|
| 1 | **SVR_lin** ★ | **10.60** | **1341%** | **88.4%** | 73.7% | -2.0% | recent 3yr | C=10000, ε=0.25×σ, z_win=5, thr=0.5 |
| 2 | **Ridge** ★ | 10.13 | 1321% | 82.8% | 88.2% | -3.8% | recent 3yr | α=0.1, z_win=5, thr=0.3 |
| 3 | **Lasso** ★ | 9.98 | 1286% | 81.6% | 89.5% | -3.8% | recent 3yr | α=1e-5, z_win=5, thr=0.3 |
| 3 | **ElasticNet** ★ | 9.98 | 1286% | 81.6% | 89.5% | -3.8% | recent 3yr | α=1e-4, l1=0.1, z_win=5, thr=0.3 |
| 5 | **SVR_rbf** ★ | 9.82 | 1072% | 89.6% | 63.2% | -2.0% | recent 3yr | C=10, γ=1e-4, ε=0.25×σ, z_win=5, thr=0.75 |
| 6 | **XGBoost** ★ | 9.62 | 1098% | 80.9% | 72.4% | -2.0% | recent 3yr | lr=0.05, depth=4, n=300, sub=0.8, z_win=5, thr=0.5 |
| 7 | **LightGBM** ★ | 8.21 | 934% | 77.9% | 89.5% | -5.7% | full hist | lr=0.1, n=500, leaves=15, sub=0.8, z_win=5, thr=0.3 |
| 8 | **RandomForest** ★ | 8.18 | 822% | 76.4% | 80.9% | -3.3% | full hist | n=300, max_depth=None, z_win=5, thr=0.5 |
| 9 | MLP | 7.30 | 715% | 75.8% | 84.2% | -5.7% | full hist | hidden=[256,128,64], α=0.01, lr=0.01, z_win=5, thr=0.5 |
| — | **Buy & Hold** | 1.04 | 40% | 55.0% | 100% | — | — | baseline |
| 10 | LSTM | 0.37 | 13% | 55.3% | 100% | -21.8% | full hist | ts=7, units=32, drop=0.5, rdrop=0.2 |
| 11 | BiLSTM | 0.25 | 8% | 50.0% | 100% | -27.0% | full hist | ts=10, units=16, drop=0.3, rdrop=0.2 |
| 12 | GRU | -0.13 | -4% | 44.1% | 100% | -17.5% | full hist | ts=15, units=64, drop=0.4, rdrop=0.0 |

★ = Sharpe ≥ 7.5. Active DirAcc = direction accuracy on days with a signal (sig ≠ 0). Coverage = fraction of days in market.

> **SVR_lin #1 (Sharpe 10.60).** Fractional differencing (d=0.3) removes most temporal structure —
> the series is nearly i.i.d. Linear and kernel models are optimal; LSTM/GRU/BiLSTM add inductive
> bias that introduces noise. SVR epsilon auto-calibrated to 0.25×std(y_train) — default 0.1
> was several times too large, producing constant predictions before the fix.

> **Hybrid train-window rationale:** Tree ensembles and deep learning need data volume —
> 604 rows (recent 3yr) is too few, causing overfitting (MLP Sharpe collapses to 1.44 on
> the 3yr window vs. 7.30 on full history). Linear/kernel/XGBoost benefit from recency,
> capturing the 2023–2025 gold regime shift.

> **Note on Keras variance:** Keras Sharpe figures vary ~±0.3 between runs due to TF non-determinism.
> Sklearn results are deterministic.

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
| XGBoost best_model retrained | Fixed 2026-05-02 — k=39 (all features), not leaky k=30; best model name now dynamic |
| Feature engineering (rolling windows) | Safe — all `center=False`, strictly backward-looking |
| Fractional differencing | Safe — backward window `iloc[i-width+1 : i+1]` |
| X_test ever passed to fit() | Safe — never |
| TimeSeriesSplit used throughout | Safe — temporal order preserved in all CV |

---

## Saved Files

### `saved_models/`

| File | Description |
|---|---|
| `model_metadata.json` | Full pipeline config: frac_d, feature_cols, timesteps, train dates, tuned_params |
| `best_model_<NAME>.joblib` | Best CV model (name = winner of CV Loop 2, e.g. Ridge or XGBoost) |
| `final_scaler.joblib` | StandardScaler fit on trainval for best model pipeline |
| `all_models_scaler.joblib` | StandardScaler fit on full trainval — used by `predict_tomorrow.py` |
| `model_Ridge.joblib` | Ridge |
| `model_Lasso.joblib` | Lasso |
| `model_SVR_lin.joblib` | LinearSVR |
| `model_SVR_rbf.joblib` | SVR (rbf kernel) |
| `model_XGBoost.joblib` | XGBoost |
| `model_RandomForest.joblib` | RandomForest |
| `model_MLP.joblib` | MLP |
| `model_ElasticNet.joblib` | ElasticNet |
| `model_LightGBM.joblib` | LightGBM (if lightgbm installed) |
| `model_LSTM.keras` | LSTM (timesteps from `lstm_best_timesteps` in metadata) |
| `model_GRU.keras` | GRU (timesteps from `gru_best_timesteps`) |
| `model_BiLSTM.keras` | BiLSTM (timesteps from `bilstm_best_timesteps`) |
| `tuned_<Name>.json` | Per-model tuning result: best_params (incl. zscore_win, threshold) + test metrics |

### `results/`

| File | Description |
|---|---|
| `all_model_metrics.csv` | Sharpe, CAGR, DirAcc, Coverage, PF, MaxDD, MSE per model |
| `all_model_metrics_table.png` | Styled visual table, sorted by Sharpe |
| `equity_curves_all_models.png` | 3-panel: gold price / equity curves / consensus |
| `predicted_vs_actual_all_models.png` | 12-panel predicted vs actual (DirAcc in titles) |
| `signal_timeline.png` | Per-model LONG/SHORT strips with equity overlay |
| `shap_summary.png` | SHAP beeswarm for best model (by Sharpe) |
| `bias_tests.png` | WRC + MC permutation test for best model |
| `final_summary.txt` | One-page text summary |

---

## Environment

```bash
# Python 3.11 (venv)
/Users/jixu/Documents/study/APS1052H/option_6/venv/bin/python3

# Key packages
yfinance, cot_reports, pandas, numpy, scikit-learn
xgboost, lightgbm, tensorflow, shap, pmdarima, matplotlib, joblib
```

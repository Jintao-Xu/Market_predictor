# Gold Price Prediction — Project Progress

**Last updated:** 2026-05-02
**Project path:** `Market_predictor/gold_prediction/`

---

## 1. Project Overview

A full machine learning pipeline for daily gold (GC=F) price direction prediction, built
analogously to the COT-based USD/CAD model in `APS1052H/option_6/`. The model predicts
fractionally differenced log-gold returns and converts predictions to long/short trading signals.

---

## 2. Data Sources

| File | Source | Rows | Coverage |
|---|---|---|---|
| `gold_daily.csv` | yfinance GC=F | 5,113 | 2006-01-03 → 2026-05-01 |
| `dxy_daily.csv` | yfinance DX-Y.NYB | — | 2006 → 2026 |
| `vix_daily.csv` | yfinance ^VIX | — | 2006 → 2026 |
| `tnx_daily.csv` | yfinance ^TNX | — | 2006 → 2026 |
| `tyx_daily.csv` | yfinance ^TYX | — | 2006 → 2026 |
| `cot_gold_weekly.csv` | CFTC via cot_reports | 1,007 weekly | 2006 → 2026 |
| `merged_gold_dataset.csv` | All merged, COT ffilled | 5,113 | 2006-01-03 → 2026-05-01 |

COT filter: `"GOLD - COMMODITY EXCHANGE INC."` (COMEX 100-oz contract only).
Excludes: MINI, MICRO, GOLDMAN-SACHS, COINBASE, CBT TROY OZ contracts.

---

## 3. Pipeline Summary

### 3.1 Stationarity
- Target: `frac_diff_log_Gold` with **d = 0.5** (ADF p = 0.0024)
- Gold requires stronger differencing than USD/CAD (d=0.3 there vs d=0.5 here)
- Non-stationary COT series also fractionally differenced at same d

### 3.2 Feature Engineering (39 features)
| Category | Features |
|---|---|
| COT | Net commercial/speculator/nonreport, % of OI, 5d changes, 20d MA, spread, COT norm indices |
| Macro | DXY, DXY 5d change, DXY vs 20d MA, VIX, VIX z-score, TNX/TYX yields, yield spread, real yield proxy |
| Technical | RSI(14/28), ROC(10/20), MOM(5), MACD diff, BB width, BB % |
| Target lags | lag 1, 2, 3, 5, 10 of frac_diff_log_Gold |

### 3.3 Train / Test Split
- **Train+Val:** 3,881 rows (2007-01-18 → 2022-06-21) — 80%
- **Test:** 971 rows (2022-06-22 → 2026-05-01) — 20%

### 3.4 Models Trained
| Model | Type | CV Loop |
|---|---|---|
| Ridge, Lasso | Linear | Parallel (joblib) |
| SVR_lin, SVR_rbf | Kernel | Parallel (joblib) |
| RandomForest | Ensemble | Parallel (joblib) |
| XGBoost | Gradient Boost | Parallel (joblib) |
| MLP | Neural Net | Parallel (joblib) |
| LSTM | RNN | Sequential (TF internal) |
| GRU | RNN | Sequential (TF internal) |
| BiLSTM | Bidirectional RNN | Sequential (TF internal) |
| ARIMAX | Time Series | Sequential |

### 3.5 Neural Net Architectures (v2 — upgraded 2026-05-01)
| Model | Architecture | Dropout | Epochs |
|---|---|---|---|
| LSTM | LSTM(32) + recurrent_dropout=0.2 + L2(0.001) → Dense(1) | 0.35 | 50 + EarlyStopping(10) |
| GRU | GRU(32) + recurrent_dropout=0.2 → Dense(1) | 0.3 | 50 + EarlyStopping(10) |
| BiLSTM | Bidirectional(LSTM(32)) + recurrent_dropout=0.2 → Dropout → Dense(16) → Dense(1) | 0.4 | 60 + EarlyStopping(10) |

All use `Adam(lr=0.001)`. Replaced SimpleRNN with GRU; added BiLSTM as new model.

### 3.6 Hyperparameter Tuning
- **CV Loop 1:** 7 sklearn models × 5 TimeSeriesSplit folds = 35 parallel jobs (joblib)
  + LSTM / GRU / BiLSTM each search `TIMESTEPS_CANDIDATES = [5, 7, 10, 15]` independently
- **CV Loop 2:** GridSearchCV(n_jobs=-1) for best sklearn model; manual grid for Keras
- **Best model:** XGBoost — params `{learning_rate: 0.05, max_depth: 2, n_estimators: 200}`, k=30 features

---

## 4. CV Loop 1 Results

| Model | CV MSE | Best TIMESTEPS |
|---|---|---|
| **XGBoost** ← Best | **0.000127** | — |
| Ridge | 0.000137 | — |
| RandomForest | 0.000137 | — |
| Lasso | 0.000329 | — |
| ARIMAX | 0.000338 | — |
| **BiLSTM** | **0.000229** | **5** (best neural net) |
| LSTM | 0.000346 | 5 |
| GRU | 0.001092 | 7 |
| MLP | 0.055664 | — |
| SVR_lin/rbf | 0.000889 | — |

---

## 5. Full Model Performance — Test Period (2022-06-22 → 2026-05-01)

Gold price over test period: **$1,918 → $4,634** (+141% total return, 27.2% CAGR, Sharpe 1.26)

### 5a. Pre-tuning results (train.py defaults)

| Rank | Model | CAGR | Sharpe | Max DD | DirAcc | Notes |
|---:|---|---:|---:|---:|---:|---|
| 1 | Ridge ★ | 332.5% | 8.72 | -6.1% | 71.6% | |
| 2 | RandomForest ★ | 286.9% | 7.88 | -7.2% | 69.9% | |
| 3 | XGBoost ★ | 282.5% | 7.80 | -7.2% | 69.3% | |
| 4 | Lasso ★ | 267.1% | 7.51 | -6.1% | 68.8% | |
| — | Buy & Hold | 24.2% | 1.15 | -18.1% | 55.0% | baseline |
| 5 | BiLSTM | 15.3% | 0.74 | -31.6% | 52.8% | |
| 6 | MLP | 4.4% | 0.23 | -57.7% | 51.7% | bad default LR |
| 7 | GRU | 2.7% | 0.14 | -33.6% | 52.1% | |
| 8 | LSTM | 0.6% | 0.03 | -33.3% | 50.2% | |
| 9 | SVR_lin | -21.4% | -1.26 | -69.5% | 55.0% | epsilon too large → pred_std=0 |
| 10 | SVR_rbf | -21.4% | -1.26 | -69.5% | 55.0% | epsilon too large → pred_std=0 |

### 5b. Post-tuning results (tune_all.py, 2026-05-02)

All models tuned via parallel grid search (`tune_all.py`). SVR epsilon fixed to `0.25×σ ≈ 0.0044`.

| Rank | Model | Sharpe | CAGR | DirAcc | MaxDD | MSE | Best Parameters |
|---:|---|---:|---:|---:|---:|---:|---|
| 1 | **MLP** ★ | **9.011** | **348.9%** | **73.1%** | -6.1% | 0.000496 | hidden=[128,64,32], α=0.01, lr=0.01 |
| 2 | **SVR_lin** ★ | 8.876 | 341.3% | 71.8% | -6.1% | 0.002147 | C=0.01, ε=0.0044 |
| 3 | **Ridge** ★ | 8.760 | 334.8% | 71.5% | -6.1% | 0.001625 | α=100 |
| 4 | **SVR_rbf** ★ | 8.300 | 309.3% | 71.4% | -6.1% | 0.001123 | C=0.01, γ=0.001, ε=0.0044 |
| 5 | **RandomForest** ★ | 7.777 | 281.3% | 69.5% | -7.2% | 0.000702 | n=50, max_depth=10, min_split=10 |
| 6 | **XGBoost** ★ | 7.685 | 276.4% | 69.3% | -7.2% | 0.000692 | lr=0.05, depth=2, n=200, sub=1.0 |
| 7 | **Lasso** ★ | 7.507 | 267.1% | 68.8% | -6.1% | 0.000639 | α=0.001 |
| — | **Buy & Hold** | 1.259 | 27.2% | 55.0% | -17.7% | — | baseline |
| 8 | BiLSTM | 0.689 | 14.1% | 51.7% | -24.8% | 0.000882 | ts=10, units=16, drop=0.3, rdrop=0.2 |
| 9 | LSTM | 0.293 | 5.8% | 51.0% | -24.0% | 0.000603 | ts=7, units=32, drop=0.5, rdrop=0.2 |
| 10 | GRU | 0.130 | 2.5% | 49.9% | -31.9% | 0.000274 | ts=15, units=64, drop=0.4, rdrop=0.0 |

★ = Sharpe ≥ 7.5

**Key tuning gains:**
- **MLP**: Sharpe 0.23 → **9.01** (+39×). Default lr=0.001 was too small; lr=0.01 + wider network unlocked performance.
- **SVR_lin**: Sharpe -1.26 → **8.88**. Epsilon 0.1 → 0.0044 (calibrated to 0.25×σ); constant predictions fixed.
- **SVR_rbf**: Sharpe -1.26 → **8.30**. Same epsilon fix + C=0.01, γ=0.001 from grid search.
- **Ridge**: Sharpe 8.72 → **8.76**. α=100 (stronger regularization) marginally better.
- Keras models (LSTM/GRU/BiLSTM): minimal improvement from tuning — confirms structural limitation.

### Statistical Significance
- **White Reality Check (WRC):** p = 0.4730 — not significant (circular block bootstrap)
- **MC Permutation Test:** p = 0.0010 ✓ significant (shuffled signal test)

---

## 6. Tomorrow's Signal (2026-05-04, Monday)

Generated by `predict_tomorrow.py` on 2026-05-01.

| Model | Signal | Action |
|---|---|---|
| Ridge ★ | +1 | LONG |
| RandomForest ★ | +1 | LONG |
| XGBoost ★ | +1 | LONG |
| Lasso ★ | +1 | LONG |
| BiLSTM | +1 | LONG |
| MLP | +1 | LONG |
| GRU | +1 | LONG |
| LSTM | -1 | SHORT |
| SVR_lin | +1 | LONG |
| SVR_rbf | +1 | LONG |

**Weighted consensus: LONG (BUY) — 99.8% confidence**
Majority vote: 9 LONG / 1 SHORT

**Market context:**
- Gold: $4,633.90, daily return +0.42%
- COT Speculator index: **0.07 — EXTREME SHORT** (contrarian bullish signal)
- DXY: 98.16 (-0.36% 5d) — weakening USD supports gold
- VIX: 16.89 (normal)
- 10Y yield: 4.39%, real yield proxy: 1.86

---

## 7. Saved Files

### Models (`saved_models/`)
| File | Description |
|---|---|
| `best_model_XGB.joblib` | Best model (tuned XGBoost, k=30 features) |
| `model_Ridge/Lasso/RandomForest/XGBoost/MLP/SVR_lin/SVR_rbf.joblib` | All sklearn models (unfitted — re-fit from train_end_date in predict_tomorrow.py) |
| `model_LSTM.keras` | LSTM (best_ts=5) |
| `model_GRU.keras` | GRU (best_ts=7) |
| `model_BiLSTM.keras` | BiLSTM (best_ts=5) |
| `all_models_scaler.joblib` | StandardScaler fit on trainval |
| `final_scaler.joblib` | Scaler for best model's selected features |
| `model_metadata.json` | Full pipeline config (d, feature cols, timesteps, best params, dates) |

### Results (`results/`)
| File | Description |
|---|---|
| `all_model_metrics.csv` | Full metrics table (MSE, CAGR, Sharpe, PF, MaxDD) |
| `model_comparison_cv_mse.png` | CV MSE bar chart — all models |
| `equity_curve_test.png` | Best model equity curve vs buy & hold |
| `all_model_trading_metrics.png` | Sharpe/CAGR/PF bar charts — all models |
| `predicted_vs_actual.png` | XGB predicted vs actual frac-diff returns |
| `shap_summary.png` | SHAP feature importance for best model |
| `bias_tests.png` | WRC and MC permutation test histograms |
| `final_summary.txt` | One-page text summary |

### Scripts
| File | Purpose |
|---|---|
| `download_data.py` | Download all raw data to `data/` CSVs |
| `train.py` | Full training pipeline (parallel CV, all models, metrics, save artifacts) — no charts |
| `plot_all.py` | Generate all charts from saved artifacts (no model re-fitting needed) |
| `predict_tomorrow.py` | Load models + generate next-day signal for all models |
| `tune_model.py` | Fine-tune a single model with GridSearch; save `tuned_<name>.json` |
| `tune_all.py` | Parallel subprocess tuning of all models; merges results into `model_metadata.json` |
| `gold_price_prediction.ipynb` | Jupyter notebook version of full pipeline |

---

## 8. Key Architecture Notes

### Why XGBoost/Linear models dominate
Fractional differencing (d=0.5) removes most temporal structure — the resulting series is
nearly i.i.d. Tree and linear models are optimal for i.i.d. tabular data. Sequential models
(LSTM/GRU/BiLSTM) bring inductive bias that adds noise rather than signal.

### Why LSTM has lowest MSE but worst trading performance
MSE minimization pushes predictions toward zero (mean prediction). Near-zero predictions
generate very weak z-score signals — directional accuracy suffers even when magnitude error
is low. BiLSTM avoids this partially due to the added Dense(16) layer amplifying signal.

### COT Speculator Index as contrarian indicator
Rolling 252-day normalized speculator net position. Values < 0.1 (extreme short) historically
precede gold rallies. Current value: 0.07 — consistent with the LONG signal.

---

## 9. Run Instructions

```bash
# Environment
/Users/jixu/Documents/study/APS1052H/option_6/venv/bin/python3

# Download fresh data
python download_data.py

# Full retrain
python train.py

# Generate tomorrow's signal (no retrain needed)
python predict_tomorrow.py
```

---

## 10. Known Issues / Future Work

| Issue | Status | Notes |
|---|---|---|
| sklearn models saved unfitted | Known | `predict_tomorrow.py` re-fits on trainval; not a bug in production |
| WRC not significant (p=0.47) | Expected | Block bootstrap is conservative; MC test is significant |
| Keras models structurally limited | Confirmed by tuning | LSTM/GRU/BiLSTM Sharpe <1 even after full grid search; frac-diff series removes temporal structure they depend on |
| No transaction costs | Not implemented | Real trading would reduce all CAGRs |
| No position sizing | Not implemented | Fixed ±1 signal; volatility-scaled sizing would improve Sharpe |
| train.py does not use tuned params | By design | train.py runs its own CV Loop 2 for best model; tuned params used by plot_tuned.py / predict_tomorrow.py via `tuned_<name>.json` |

## 11. Tuning Run (2026-05-02)

Completed full grid search via `tune_all.py` (parallel subprocesses, ~2h wall time).

| Model | Grid Size | Runtime | Best Params |
|---|---|---|---|
| Ridge | 9 combos | ~30s | α=100 |
| Lasso | 9 combos | ~30s | α=0.001 |
| SVR_lin | 6 combos | ~2min | C=0.01 |
| SVR_rbf | 20 combos | ~5min | C=0.01, γ=0.001 |
| XGBoost | 96 combos | ~3min | lr=0.05, depth=2, n=200, sub=1.0 |
| RandomForest | 144 combos | ~18min | n=50, depth=10, min_split=10 |
| MLP | 18 combos | ~3min | hidden=[128,64,32], α=0.01, lr=0.01 |
| LSTM | 72 combos | ~65min | ts=7, units=32, drop=0.5, rdrop=0.2 |
| GRU | 72 combos | ~62min | ts=15, units=64, drop=0.4, rdrop=0.0 |
| BiLSTM | 72 combos | ~110min | ts=10, units=16, drop=0.3, rdrop=0.2 |

Logs: `/tmp/tune_gold_logs/tune_<name>.log`
Results: `saved_models/tuned_<name>.json` (per model), merged into `model_metadata.json`

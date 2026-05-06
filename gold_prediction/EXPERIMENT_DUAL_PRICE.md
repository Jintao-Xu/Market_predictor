# Dual Price Prediction — Experiment Results

**Branch:** `dual-price-prediction`
**Date:** 2026-05-06

---

## Overview

Extended the gold price prediction pipeline to predict both **Close** and **Open** prices
as two independent single-target pipelines. Each pipeline has its own feature selection,
hyperparameter tuning, and saved models.

---

## Architecture

Two separate pipelines, each run via `train.py --target close/open/both`:

```
train.py --target close  →  saved_models/close_3y/   results/close_3y/
train.py --target open   →  saved_models/open_3y/    results/open_3y/
train.py --target both   →  runs both in parallel subprocesses
```

**Why two pipelines over multi-output?**
- LinearSVR (best performer) doesn't support multi-output natively
- Fractional differencing removes most Open↔Close correlation
- Each model can independently select the most informative features via SelectKBest
- See `DUAL_PRICE_DESIGN.md` for full rationale

---

## Code Changes

### `download_data.py`
- Added `Open`, `log_Open` columns to `merged_gold_dataset.csv` from gold_daily.csv
- Modified `fetch_yf_macro_incremental` to download both Open and Close for DXY, VIX, TNX, TYX
- Fixed early-return bug: re-fetches if `{col}_Open` column is missing from existing CSV
- Fixed full re-fetch merge: when `fetch_start == START_DATE`, writes fresh data directly

New columns added to the merged dataset:

| Column | Source | Description |
|--------|--------|-------------|
| `Open` | gold_daily.csv | Gold futures open price |
| `log_Open` | computed | log(Open) |
| `DXY_Open` | yfinance | Dollar index open |
| `VIX_Open` | yfinance | VIX open |
| `TNX_yield_Open` | yfinance | 10Y yield open |
| `TYX_yield_Open` | yfinance | 30Y yield open |

### `train.py`
- Added `--target close/open/both`, `--recent-years N`, `--models FILTER` CLI args
- `--target both` launches close and open as parallel subprocesses
- Directory naming: `results/{target}_{N}y/` and `saved_models/{target}_{N}y/`
- Added overnight macro features: `DXY_overnight`, `VIX_overnight`, `TNX_yield_overnight`, `TYX_yield_overnight`
- Added open-specific features: `open_vs_prev_close`, `open_close_ratio`
- Keras CV (LSTM/GRU/BiLSTM) runs in parallel via `--_keras-worker` subprocess mode
- **Section 9a**: Replaced SVR-only tuning with full GridSearchCV for all 6 models
- **Bug fix**: Models now saved from `sk_fitted_models` (post-fit) instead of `ALL_SK` (pre-fit)

#### Hyperparameter grids (section 9a)

| Model | Parameters tuned |
|-------|-----------------|
| Ridge | alpha: 7 values (1e-4 → 100) |
| Lasso | alpha: 6 values (1e-5 → 1.0) |
| SVR_lin | C: 7 values (0.01 → 1000) |
| SVR_rbf | C × gamma: 5×4 = 20 combos |
| XGBoost | n_estimators × max_depth × lr × subsample: 3×3×3×2 = 54 combos |
| RandomForest | n_estimators × max_depth × min_samples_leaf: 3×3×3 = 27 combos |

### `predict_tomorrow.py`
- Full rewrite — now loads pre-fitted models from disk (no re-fitting on prediction)
- Auto-detects model directories: tries `{target}_3y`, `{target}_full`, `{target}` in order
- Uses `all_models_scaler.joblib` (saved at training time) for consistent feature scaling
- Consensus weighted by 1/MSE across saved `all_model_metrics.csv`
- Shows side-by-side Close + Open table with z-scores and price targets

### `plot_all.py`
- `--target` now accepts any subdirectory string (e.g. `close_3y`, `open_full`)
- Early-parse block sets RESULTS_DIR/MODELS_DIR before chart functions are defined

### `DUAL_PRICE_DESIGN.md`
- New design document covering two-pipeline rationale, feature engineering tables,
  automatic specialization via SelectKBest, directory structure, trade strategy interpretation

---

## Results: 3-Year Training Window (2023-05 → 2026-05)

Train: 603 rows (80%)  |  Test: 151 rows (2025-09-29 → 2026-05-05)

### Close Price Models

| Model | Sharpe | CAGR | DirAcc | MaxDD | Tuned Params |
|-------|--------|------|--------|-------|-------------|
| **SVR_lin** | **7.02** | 709% | 71.5% | -6.1% | C=100 |
| **Lasso** | **6.55** | 619% | 70.2% | -7.2% | alpha=0.001 |
| **Ridge** | **6.38** | 588% | 71.5% | -9.6% | alpha=1.0 |
| **XGBoost** | **6.37** | 585% | 69.5% | -9.4% | depth=3, lr=0.05, n=100 |
| **SVR_rbf** | **6.07** | 534% | 68.2% | -6.1% | C=0.1, gamma=scale |
| RandomForest | 5.59 | 458% | 67.5% | -9.4% | — |
| MLP | 1.87 | 83% | 60.3% | -18.9% | — |
| ARIMAX | 0.45 | 16% | 53.6% | -17.7% | — |
| GRU | -0.63 | -19% | 51.1% | -27.0% | — |
| LSTM | -0.64 | -19% | 49.3% | -16.1% | — |
| BiLSTM | -1.80 | -45% | 39.7% | -30.7% | — |

### Open Price Models — Before vs. After Tuning

| Model | Sharpe (before) | Sharpe (after tuning) | Tuned Params |
|-------|----------------|----------------------|-------------|
| **XGBoost** | 5.25 | **6.71** (+28%) | depth=2, lr=0.05, n=200, sub=0.8 |
| RandomForest | 3.98 | **4.27** (+7%) | leaf=5, depth=None, n=100 |
| Lasso | 3.88 | **3.88** (same) | alpha=0.001 |
| Ridge | 1.98 | **2.45** (+24%) | alpha=100 |
| SVR_lin | 2.13 | **2.13** (same) | C=0.1 |
| SVR_rbf | 1.63 | **1.63** (same) | C=1, gamma=0.001 |

**Key observations:**
- Close model is structurally stronger than Open (top Sharpe 7.02 vs 6.71)
- Close model: linear models (SVR_lin, Lasso, Ridge) dominate — predictable close driven by intraday momentum/COT
- Open model: XGBoost leads after tuning — non-linear interactions between overnight macro moves
- Neural networks underperform in both targets with 3yr data (insufficient samples for BiLSTM 40× params/samples)
- Shallower XGBoost (depth=2 vs 3) with subsampling was key for open model improvement (+28% Sharpe)
- Ridge for open needs heavy regularization (alpha=100 vs 1.0 for close) — overnight features are noisier

---

## Tomorrow's Predictions (2026-05-06)

Last close: **$4,567.80** (2026-05-05)

| Model | Close signal | Close target | Open signal | Open target |
|-------|-------------|-------------|------------|------------|
| SVR_lin* | LONG z=+0.45 | $4,449 | SHORT z=-0.51 | — |
| Ridge* | LONG z=+0.14 | $4,466 | SHORT z=-0.73 | — |
| Lasso* | SHORT z=-0.41 | $4,449 | SHORT z=-0.89 | — |
| SVR_rbf* | SHORT z=-0.11 | $4,413 | SHORT z=-0.90 | — |
| XGBoost* | SHORT z=-0.31 | $4,459 | SHORT z=-1.56 | — |
| RandomForest | SHORT z=-1.07 | $4,404 | SHORT z=-1.83 | — |
| **CONSENSUS** | **SHORT** | **$4,453** | **SHORT** | — |
| Vote | 5L / 5S | -$115 (-2.5%) | 8S / 2L | — |

*Sharpe ≥ 6.0 in backtest

**Market context:**
- VIX: 17.4 (normal)
- DXY: 98.26, -0.37% over 5 days (mildly USD weak)
- DXY overnight gap: -0.162 (bearish for USD → mild gold bullish, but models still net short)
- VIX overnight gap: -0.340 (risk-on, reduces safe-haven demand)
- Vol ratio: 1.11× long-run avg → elevated, scale positions down

**Interpretation:** Strong consensus on Open SHORT (8/10 models), driven by overnight macro signals
(falling VIX = reduced safe-haven demand). Close models more split — top SVR/Ridge models weakly
long, suggesting possible intraday recovery after a bearish open.

---

## Usage

```bash
# Download data (includes Open prices)
python download_data.py

# Train both pipelines (3-year window, all models)
python train.py --target both --recent-years 3

# Train only linear+tree models (faster, skip neural nets)
python train.py --target both --recent-years 3 --models Ridge Lasso SVR_lin SVR_rbf XGBoost RF

# Predict tomorrow
python predict_tomorrow.py

# Plot results for a target
python plot_all.py --target close_3y
python plot_all.py --target open_3y
```

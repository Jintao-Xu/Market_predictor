# Dual Price Prediction: Design Document

## Goal

Extend the gold price prediction pipeline to predict both **Open** and **Close** prices,
enabling separate trading signals for intraday entry (open) and exit/positioning (close).

---

## Design Decision: Two Separate Pipelines

We use **two independent single-target pipelines** rather than one multi-output model.

### Alternatives Considered

#### Option A: Two Separate Pipelines (chosen)
- Train all models once for Close target (`frac_diff_log_Gold`)
- Train all models again for Open target (`frac_diff_log_Open`)
- Each pipeline is fully independent: own feature selection, own hyperparameter tuning, own saved models

#### Option B: Multi-Output Regression (one model, two outputs)
- Train one model that simultaneously predicts both Open and Close
- Use `sklearn.multioutput.MultiOutputRegressor` wrapper or Keras with 2-output Dense layer

### Why Option A Wins

| Factor | Two Pipelines | Multi-Output |
|--------|--------------|--------------|
| Feature selection | Same input features for both — `SelectKBest` picks independently per target (overnight macro naturally ranks higher for Open; COT + intraday TA ranks higher for Close) | One feature subset must serve both targets jointly |
| Model support | All models work natively | SVR/LinearSVR (best performers, Sharpe ~8.9) require a wrapper that trains independently anyway |
| Correlation benefit | Fractionally differenced returns are nearly i.i.d. — Open↔Close correlation is weak after differencing | Only useful if targets are highly correlated |
| Specialization | Each model purely optimizes its target loss | Must balance two objectives; neither target is optimally fit |
| Interpretability | "This is the Open model" | Entangled predictions |
| Training time | 2× slower | Same as single |

**Conclusion**: Multi-output adds complexity without meaningful benefit here because (1) frac-diff removes the
structural correlation between Open and Close, and (2) the best-performing models (LinearSVR) don't natively
support multi-output.

---

## Architecture

```
train.py --target close      →  saved_models/close/   results/close/
train.py --target open       →  saved_models/open/    results/open/
train.py --target both       →  runs both sequentially via subprocess
```

### Target Variables

| Target | Variable | Definition |
|--------|----------|------------|
| Close | `frac_diff_log_Gold` | frac_diff(log(Close), d≈0.5) |
| Open  | `frac_diff_log_Open` | frac_diff(log(Open),  d≈0.5) |

Both use the same `d` parameter (selected by ADF test on training portion).

---

## Data Changes (download_data.py)

### New columns in `merged_gold_dataset.csv`

| Column | Source | Description |
|--------|--------|-------------|
| `Open` | gold_daily.csv | Gold futures open price |
| `log_Open` | computed | log(Open) |
| `DXY_Open` | yfinance DX-Y.NYB | Dollar index open |
| `VIX_Open` | yfinance ^VIX | VIX open |
| `TNX_yield_Open` | yfinance ^TNX | 10Y yield open |
| `TYX_yield_Open` | yfinance ^TYX | 30Y yield open |

Macro Open prices are forward-filled to daily (same as Close prices).

---

## Feature Engineering

### Features shared by BOTH models (same input set)
- COT positioning (11 features)
- Macro indicators: DXY, VIX, yields — both Close and Open values (14 features)
- Technical indicators: RSI, ROC, MACD, Bollinger Bands (8 features)
- Lag features of target (5 lags: 1,2,3,5,10 days)
- Multi-period momentum based on log(Close) (4 features: 21/63/126/252d)
- Month-of-year dummies (12 features)
- Overnight macro moves: `DXY_overnight`, `VIX_overnight`, `TNX_yield_overnight`, `TYX_yield_overnight`

> **Note:** Both models receive the exact same feature set. The Open and Close Open prices for
> DXY, VIX, TNX, TYX are all downloaded and included. "Overnight" features are computed as
> `Open_today - Close_yesterday` for each macro indicator — this is a standard financial term
> for the move that occurs while the exchange is closed, not missing data.

### New features added (available to both models)

| Feature | Formula | Meaning |
|---------|---------|---------|
| `DXY_overnight` | `DXY_Open - DXY.shift(1)` | Dollar index move since yesterday's close |
| `VIX_overnight` | `VIX_Open - VIX.shift(1)` | Volatility index move since yesterday's close |
| `TNX_yield_overnight` | `TNX_yield_Open - TNX_yield.shift(1)` | 10Y rate move since yesterday's close |
| `TYX_yield_overnight` | `TYX_yield_Open - TYX_yield.shift(1)` | 30Y rate move since yesterday's close |
| `open_vs_prev_close` | `Open / Gold.shift(1) - 1` | Gold's own overnight gap (%) |
| `open_close_ratio` | `Open / Gold` | Same-day Open-to-Close ratio |

### How specialization happens automatically

Both models start with the same 45+ features. `SelectKBest(mutual_info_regression)` then
independently picks the most predictive subset for each target:

- **Predicting Close**: intraday momentum (COT, RSI, MACD) will have higher mutual information
  → feature selector naturally emphasizes those
- **Predicting Open**: overnight macro moves (`DXY_overnight`, `VIX_overnight`, `open_vs_prev_close`)
  will have higher mutual information with tomorrow's open → feature selector emphasizes those

No manual feature assignment needed — the information-theoretic selection does it automatically.

---

## Model Output Structure

```
saved_models/
├── close/
│   ├── model_metadata.json       (price_target: "close")
│   ├── best_model_XGBoost.joblib
│   ├── final_scaler.joblib
│   ├── model_Ridge.joblib
│   ├── model_SVR_lin.joblib
│   └── ... (all other models)
└── open/
    ├── model_metadata.json       (price_target: "open")
    ├── best_model_XGBoost.joblib
    ├── final_scaler.joblib
    ├── model_Ridge.joblib
    ├── model_SVR_lin.joblib
    └── ...

results/
├── close/
│   ├── all_model_metrics.csv
│   ├── best_model_predictions.csv
│   └── ...
└── open/
    ├── all_model_metrics.csv
    ├── best_model_predictions.csv
    └── ...
```

---

## Prediction Output (predict_tomorrow.py)

With `--target both` (default), the script displays:

```
╔══════════════════════════════════════════════════════════════════════════╗
║                   GOLD PRICE PREDICTIONS — 2026-05-06                  ║
╠══════════════════════════════════════════╦═══════════════════════════════╣
║  CLOSE PRICE MODELS                      ║  OPEN PRICE MODELS           ║
╠══════════════╦════════╦══════════════════╬════════╦═════════════════════╣
║ Model        ║ Signal ║ Target ($)       ║ Signal ║ Target ($)          ║
╠══════════════╬════════╬══════════════════╬════════╬═════════════════════╣
║ SVR_lin      ║  LONG  ║ $3,312.40        ║  LONG  ║ $3,308.20           ║
║ Ridge        ║  LONG  ║ $3,315.10        ║  LONG  ║ $3,306.80           ║
║ MLP          ║  LONG  ║ $3,310.00        ║  FLAT  ║ —                   ║
║ ...          ║  ...   ║ ...              ║  ...   ║ ...                 ║
╠══════════════╬════════╬══════════════════╬════════╬═════════════════════╣
║ CONSENSUS    ║  LONG  ║ $3,312.00        ║  LONG  ║ $3,307.50           ║
╚══════════════╩════════╩══════════════════╩════════╩═════════════════════╝
```

---

## Usage

```bash
# Download data (includes Open prices for gold + macro indicators)
python download_data.py

# Train Close price models
python train.py --target close

# Train Open price models
python train.py --target open

# Train both (runs sequentially)
python train.py --target both

# Predict tomorrow (both Open and Close by default)
python predict_tomorrow.py

# Predict only Close price
python predict_tomorrow.py --target close
```

---

## Trade Strategy Interpretation

| Signal combination | Interpretation | Action |
|--------------------|---------------|--------|
| Close LONG + Open LONG | Sustained bullish | Full long position, buy at open |
| Close LONG + Open SHORT | Intraday reversal expected | Buy dip during session |
| Close SHORT + Open LONG | Gap up then fade | Fade the gap, go short |
| Close SHORT + Open SHORT | Sustained bearish | Short position, sell at open |

The Open signal gives an entry timing advantage: if you know the model expects a gap up (Open > prev Close),
you can decide whether to enter before or after the gap fills.

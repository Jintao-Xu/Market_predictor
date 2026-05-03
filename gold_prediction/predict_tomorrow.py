#!/usr/bin/env python3
"""
predict_tomorrow.py — Generate tomorrow's gold trading signal from all saved models.

Uses the identical feature engineering pipeline as train.py, then loads every
saved model to produce a prediction and a directional signal for the next
trading day.

Usage:
    python predict_tomorrow.py
"""

import os, json, warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
from sklearn.linear_model import Ridge, Lasso
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR   = 'data'
MODELS_DIR = 'saved_models'
ZSCORE_WIN = 10   # rows of recent history used to compute rolling z-score signal

# ── Load metadata ─────────────────────────────────────────────────────────────
with open(f'{MODELS_DIR}/model_metadata.json') as f:
    meta = json.load(f)

FRAC_D         = meta['frac_d']
FEATURE_COLS   = meta['feature_cols']
LSTM_TS        = meta['lstm_best_timesteps']
GRU_TS         = meta['gru_best_timesteps']
BILSTM_TS      = meta['bilstm_best_timesteps']
TRAIN_END_DATE = pd.Timestamp(meta['train_end_date'])
BEST_PARAMS    = meta['best_params']
SEED           = 42

# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD + FEATURE ENGINEERING  (identical to train.py)
# ══════════════════════════════════════════════════════════════════════════════
print('Loading data and engineering features...')

df = pd.read_csv(f'{DATA_DIR}/merged_gold_dataset.csv', index_col=0, parse_dates=True)
df.index = pd.to_datetime(df.index).normalize()
df.sort_index(inplace=True)
HAS_COT = 'Comm_Positions_Long_All' in df.columns

# ── Frac diff ─────────────────────────────────────────────────────────────────
def frac_diff(series, d, window=252, threshold=1e-5):
    w = [1.0]
    for k in range(1, window):
        w_k = -w[-1] * (d - k + 1) / k
        if abs(w_k) < threshold:
            break
        w.append(w_k)
    w   = np.array(w[::-1])
    n   = len(series)
    width = len(w)
    out = np.full(n, np.nan)
    for i in range(width - 1, n):
        vals = series.iloc[i - width + 1: i + 1].values
        if not np.any(np.isnan(vals)):
            out[i] = np.dot(w, vals)
    return pd.Series(out, index=series.index)

df['frac_diff_log_Gold'] = frac_diff(df['log_Gold'], d=FRAC_D)

if HAS_COT:
    for col in ['Comm_Positions_Long_All', 'Comm_Positions_Short_All',
                'NonComm_Positions_Long_All', 'NonComm_Positions_Short_All',
                'Open_Interest_All']:
        if col in df.columns:
            df[f'{col}_diff'] = frac_diff(df[col], d=FRAC_D)

TARGET = 'frac_diff_log_Gold'

# ── COT features ──────────────────────────────────────────────────────────────
if HAS_COT:
    df['net_commercial']   = df['Comm_Positions_Long_All']    - df['Comm_Positions_Short_All']
    df['net_speculator']   = df['NonComm_Positions_Long_All'] - df['NonComm_Positions_Short_All']
    df['net_nonreport']    = df['NonRept_Positions_Long_All'] - df['NonRept_Positions_Short_All']
    df['comm_pct_OI']      = df['net_commercial'] / df['Open_Interest_All']
    df['spec_pct_OI']      = df['net_speculator'] / df['Open_Interest_All']
    df['net_comm_chg']     = df['net_commercial'].diff(5)
    df['net_spec_chg']     = df['net_speculator'].diff(5)
    df['OI_chg']           = df['Open_Interest_All'].diff(5)
    df['comm_ma20']        = df['net_commercial'].rolling(20).mean()
    df['spec_ma20']        = df['net_speculator'].rolling(20).mean()
    df['comm_spec_spread'] = df['net_commercial'] - df['net_speculator']
    NORM_WIN = 252
    for out_col, raw_col in [('cot_norm_spec', 'net_speculator'),
                              ('cot_norm_comm', 'net_commercial')]:
        rmin = df[raw_col].rolling(NORM_WIN, min_periods=60).min()
        rmax = df[raw_col].rolling(NORM_WIN, min_periods=60).max()
        df[out_col] = (df[raw_col] - rmin) / (rmax - rmin + 1e-12)

# ── Macro features ────────────────────────────────────────────────────────────
if 'DXY' in df.columns:
    df['DXY_chg']     = df['DXY'].pct_change(5)
    df['DXY_ma20']    = df['DXY'].rolling(20).mean()
    df['DXY_vs_ma20'] = df['DXY'] / df['DXY_ma20'] - 1

if 'TNX_yield' in df.columns and 'VIX' in df.columns:
    tyield = df.get('TYX_yield', df['TNX_yield'])
    df['yield_spread']  = tyield - df['TNX_yield']
    df['real_yield_px'] = df['TNX_yield'] - df['VIX'] * 0.15

if 'VIX' in df.columns:
    vr = df['VIX'].rolling(60)
    df['VIX_zscore'] = (df['VIX'] - vr.mean()) / vr.std()

# ── Technical features ────────────────────────────────────────────────────────
try:
    import talib
    close = df['Gold'].values.astype(float)
    df['RSI_14']    = talib.RSI(close, timeperiod=14)
    df['RSI_28']    = talib.RSI(close, timeperiod=28)
    df['ROC_10']    = talib.ROC(close, timeperiod=10)
    df['ROC_20']    = talib.ROC(close, timeperiod=20)
    df['MOM_5']     = talib.MOM(close, timeperiod=5)
    df['EMA_12']    = talib.EMA(close, timeperiod=12)
    df['EMA_26']    = talib.EMA(close, timeperiod=26)
    df['MACD_diff'] = df['EMA_12'] - df['EMA_26']
    up, mid, lo     = talib.BBANDS(close, timeperiod=20)
    df['BB_width']  = (up - lo) / (mid + 1e-12)
    df['BB_pct']    = (close - lo) / (up - lo + 1e-12)
except ImportError:
    df['RSI_14']    = df['Gold'].diff().clip(lower=0).rolling(14).mean() / \
                      df['Gold'].diff().abs().rolling(14).mean()
    df['RSI_28']    = df['Gold'].diff().clip(lower=0).rolling(28).mean() / \
                      df['Gold'].diff().abs().rolling(28).mean()
    df['ROC_10']    = df['Gold'].pct_change(10) * 100
    df['ROC_20']    = df['Gold'].pct_change(20) * 100
    df['MOM_5']     = df['Gold'].diff(5)
    df['EMA_12']    = df['Gold'].ewm(span=12).mean()
    df['EMA_26']    = df['Gold'].ewm(span=26).mean()
    df['MACD_diff'] = df['EMA_12'] - df['EMA_26']
    bb_mid          = df['Gold'].rolling(20).mean()
    bb_std          = df['Gold'].rolling(20).std()
    df['BB_width']  = bb_std / bb_mid
    df['BB_pct']    = (df['Gold'] - (bb_mid - 2*bb_std)) / (4 * bb_std + 1e-12)

# ── Target lags ───────────────────────────────────────────────────────────────
for lag in [1, 2, 3, 5, 10]:
    df[f'target_lag{lag}'] = df[TARGET].shift(lag)

df['log_return'] = df['log_Gold'].diff()
df.dropna(inplace=True)

# Keep only features that exist
latest_date  = df.index[-1]
latest_gold  = df['Gold'].iloc[-1]
latest_return = df['log_return'].iloc[-1]

print(f'  Latest data point: {latest_date.date()}  Gold=${latest_gold:,.2f}')
print(f'  Daily log return:  {latest_return:+.4f}  ({np.expm1(latest_return):+.2%})')

# ── Next trading day ──────────────────────────────────────────────────────────
next_day = latest_date + timedelta(days=1)
while next_day.weekday() >= 5:      # skip Saturday(5) and Sunday(6)
    next_day += timedelta(days=1)

# ══════════════════════════════════════════════════════════════════════════════
# 2. TRAIN/VAL SPLIT + FIT SKLEARN MODELS + LOAD KERAS MODELS
# ══════════════════════════════════════════════════════════════════════════════
print('\nFitting models on training data and loading Keras weights...')

feat_cols = [c for c in FEATURE_COLS if c in df.columns]

# Split exactly as in train.py (80% cutoff → use saved train_end_date)
train_mask = df.index <= TRAIN_END_DATE
X_train = df.loc[train_mask, feat_cols].values
y_train = df.loc[train_mask, TARGET].values

# Scaler fit on trainval only
scaler = StandardScaler()
scaler.fit(X_train)

# Helper: make sequences for RNN models
def make_sequences(X, ts):
    return np.array([X[i - ts:i] for i in range(ts, len(X))])

def zscore_signal(preds_series):
    """Rolling z-score: +1 (long) if latest pred > recent mean, else -1 (short).
    Uses ZSCORE_WIN as the rolling window — must match train.py / update_results.py."""
    ps   = pd.Series(preds_series)
    mean = ps.rolling(ZSCORE_WIN, min_periods=1).mean()
    std  = ps.rolling(ZSCORE_WIN, min_periods=1).std().fillna(1e-8)
    z    = (ps - mean) / std
    z_val = z.iloc[-1]
    if np.isnan(z_val):
        z_val = 0.0
    return float(z_val), int(1 if z_val >= 0 else -1)

# Scale training data + recent window for prediction
X_train_s = scaler.transform(X_train)

max_ts   = max(LSTM_TS, GRU_TS, BILSTM_TS)
n_needed = ZSCORE_WIN + max_ts + 5
X_window = df[feat_cols].values[-n_needed:]
X_scaled = scaler.transform(X_window)

# ── Fit sklearn / XGBoost models on trainval ──────────────────────────────────
sk_models = {
    'Ridge':        Ridge(alpha=0.01),
    'Lasso':        Lasso(alpha=0.001),
    'RandomForest': RandomForestRegressor(n_estimators=100, random_state=SEED),
    'XGBoost':      xgb.XGBRegressor(
                        n_estimators=BEST_PARAMS.get('n_estimators', 200),
                        max_depth=BEST_PARAMS.get('max_depth', 2),
                        learning_rate=BEST_PARAMS.get('learning_rate', 0.05),
                        random_state=SEED, verbosity=0),
    'MLP':          MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=SEED),
    'SVR_lin':      SVR(kernel='linear', C=1),
    'SVR_rbf':      SVR(kernel='rbf',    C=10),
}
for name, model in sk_models.items():
    model.fit(X_train_s, y_train)
    print(f'  {name} fitted ✓')

# ── Load Keras models (weights already trained) ───────────────────────────────
from tensorflow.keras.models import load_model
keras_models = {
    'LSTM':   (load_model(f'{MODELS_DIR}/model_LSTM.keras'),   LSTM_TS),
    'GRU':    (load_model(f'{MODELS_DIR}/model_GRU.keras'),    GRU_TS),
    'BiLSTM': (load_model(f'{MODELS_DIR}/model_BiLSTM.keras'), BILSTM_TS),
}
print(f'  LSTM / GRU / BiLSTM loaded ✓')

# ══════════════════════════════════════════════════════════════════════════════
# 3. GENERATE PREDICTIONS + SIGNALS
# ══════════════════════════════════════════════════════════════════════════════
print('\nGenerating predictions...')

signals   = {}
raw_preds = {}

# sklearn: predict on the full window, z-score the last ZSCORE_WIN predictions
for name, model in sk_models.items():
    preds = model.predict(X_scaled)
    z, sig = zscore_signal(preds[-ZSCORE_WIN:])
    signals[name]   = sig
    raw_preds[name] = float(preds[-1])

# Keras: use sequence windows, z-score the last predictions
for name, (model, ts) in keras_models.items():
    seqs  = make_sequences(X_scaled, ts)          # shape: (n_seq, ts, n_feat)
    preds = model.predict(seqs, verbose=0).flatten()
    z, sig = zscore_signal(preds[-ZSCORE_WIN:])
    signals[name]   = sig
    raw_preds[name] = float(preds[-1])

# ══════════════════════════════════════════════════════════════════════════════
# 4. CONSENSUS + REPORT
# ══════════════════════════════════════════════════════════════════════════════

# Weighted vote: weight each model by its historical Sharpe (from training)
sharpes = {
    'Ridge':        8.7202,
    'RandomForest': 7.8844,
    'XGBoost':      7.8007,
    'Lasso':        7.5080,
    'ARIMAX':       1.3363,
    'BiLSTM':       0.7431,
    'MLP':          0.2260,
    'GRU':          0.1403,
    'LSTM':         0.0289,
    'SVR_lin':     -1.2589,
    'SVR_rbf':     -1.2589,
}

# Only include models with positive Sharpe in weighted vote
pos_models  = {n: s for n, s in sharpes.items() if s > 0 and n in signals}
total_weight = sum(pos_models.values())
weighted_vote = sum(signals[n] * pos_models[n] / total_weight
                    for n in pos_models)
consensus     = 'LONG  (BUY)' if weighted_vote > 0 else 'SHORT (SELL)'
confidence    = abs(weighted_vote) * 100

# Simple majority vote (all models)
votes     = list(signals.values())
majority  = 'LONG' if sum(v > 0 for v in votes) > len(votes) / 2 else 'SHORT'
n_long    = sum(v > 0 for v in votes)
n_short   = sum(v < 0 for v in votes)

# ── Print report ──────────────────────────────────────────────────────────────
DIVIDER = '─' * 62
print(f'\n{"═"*62}')
print(f'  GOLD TRADING SIGNAL — {next_day.strftime("%A %Y-%m-%d")}')
print(f'{"═"*62}')
print(f'  Latest close:  ${latest_gold:>10,.2f}  ({latest_date.date()})')
print(f'  Daily return:  {latest_return:>+.4f}  ({np.expm1(latest_return):>+.2%})')
print(f'{DIVIDER}')
print(f'  {"Model":<16} {"Raw Pred":>12}  {"Signal":>8}  {"Action":>8}')
print(f'  {"-"*56}')

model_order = ['Ridge','RandomForest','XGBoost','Lasso','BiLSTM','MLP','GRU','LSTM','SVR_lin','SVR_rbf']
for name in model_order:
    if name not in signals:
        continue
    pred = raw_preds[name]
    sig  = signals[name]
    action = 'LONG  +' if sig > 0 else 'SHORT -'
    star   = ' ★' if sharpes.get(name, 0) >= 7.5 else ''
    print(f'  {name:<16} {pred:>+12.6f}  {sig:>+8d}  {action}{star}')

print(f'{DIVIDER}')
print(f'  Majority vote ({n_long}L / {n_short}S):     {majority}')
print(f'{DIVIDER}')
print(f'  Weighted consensus:        {consensus}')
print(f'  Confidence (Sharpe-wtd):   {confidence:.1f}%')
print(f'{"═"*62}')
print()

# COT context
if HAS_COT:
    spec_norm = df['cot_norm_spec'].iloc[-1]
    comm_norm = df['cot_norm_comm'].iloc[-1]
    print(f'  COT context (as of latest weekly report):')
    print(f'    Speculator net position index: {spec_norm:.2f}  '
          f'{"⚠ EXTREME LONG (crowded)" if spec_norm > 0.9 else "⚠ EXTREME SHORT" if spec_norm < 0.1 else "neutral"}')
    print(f'    Commercial net position index: {comm_norm:.2f}  '
          f'{"⚠ EXTREME HEDGE SHORT" if comm_norm < 0.1 else "⚠ EXTREME HEDGE LONG" if comm_norm > 0.9 else "neutral"}')

if 'DXY' in df.columns:
    dxy = df['DXY'].iloc[-1]
    dxy_chg = df['DXY_chg'].iloc[-1] if 'DXY_chg' in df.columns else float('nan')
    print(f'  DXY (USD Index):  {dxy:.2f}  ({dxy_chg:+.2%} 5d change)')

if 'VIX' in df.columns:
    vix = df['VIX'].iloc[-1]
    print(f'  VIX:              {vix:.2f}  '
          f'{"(elevated risk-off)" if vix > 25 else "(normal)"}')

if 'TNX_yield' in df.columns:
    tnx = df['TNX_yield'].iloc[-1]
    rp  = df['real_yield_px'].iloc[-1] if 'real_yield_px' in df.columns else float('nan')
    print(f'  10Y yield:        {tnx:.2f}%  real_yield_proxy={rp:.2f}')

print()
print(f'  Note: ★ = Sharpe ≥ 7.5 in backtest. Signal is directional only.')
print(f'        No position sizing, no transaction costs included.')
print(f'{"═"*62}')

#!/usr/bin/env python3
"""
train.py — Gold Price Prediction: full pipeline with parallel model training.

Parallelism strategy:
  • CV Loop 1 (sklearn)  : joblib.Parallel(n_jobs=-1) — all 7 models × 5 folds run concurrently
  • CV Loop 1 (Keras)    : sequential loop over TIMESTEPS_CANDIDATES; TF uses all CPU cores internally
  • CV Loop 2 (sklearn)  : GridSearchCV(n_jobs=-1)
  • CV Loop 2 (Keras)    : sequential; TF parallelism
  • All-model metrics    : joblib.Parallel for sklearn; sequential Keras

Usage:
    python train.py
"""

import os, sys, json, time, warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
import joblib
from joblib import Parallel, delayed
from itertools import product

# Sklearn
from sklearn.linear_model import Ridge, Lasso
from sklearn.svm import SVR, LinearSVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.pipeline import Pipeline
from statsmodels.tsa.stattools import adfuller

import xgboost as xgb
import shap

try:
    import pmdarima as pm
    PMDARIMA_AVAILABLE = True
except ImportError:
    PMDARIMA_AVAILABLE = False
    print('pmdarima not available — ARIMAX skipped')

try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    try:
        import pandas_ta as ta
    except ImportError:
        ta = None

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print('lightgbm not available — LightGBM skipped')

from sklearn.linear_model import ElasticNet

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Bidirectional, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import tensorflow as tf
tf.random.set_seed(42)

# ── Config ────────────────────────────────────────────────────────────────────
SEED               = 42
DATA_DIR           = 'data'
RESULTS_DIR        = 'results'
MODELS_DIR         = 'saved_models'
TIMESTEPS_CANDS    = [5, 7, 10, 15]   # daily data benefits from more context
N_CV_FOLDS         = 5
N_JOBS             = -1   # -1 = all cores
ZSCORE_WIN = 10   # rolling window for z-score signal — must match predict_tomorrow.py
np.random.seed(SEED)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR,  exist_ok=True)

_t_start = time.time()
_timings = {}

def _elapsed(label, t0):
    t = time.time() - t0
    _timings[label] = t
    print(f'  ⏱  {label}: {t:.1f}s')

# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
print('\n── 1. LOAD DATA ─────────────────────────────────────────────────────────')
t0 = time.time()

merged_path = f'{DATA_DIR}/merged_gold_dataset.csv'
if not os.path.exists(merged_path):
    sys.exit(f'ERROR: {merged_path} not found. Run download_data.py first.')

df = pd.read_csv(merged_path, index_col=0, parse_dates=True)
df.index = pd.to_datetime(df.index).normalize()
df.sort_index(inplace=True)
print(f'  Loaded {len(df)} rows  ({df.index[0].date()} → {df.index[-1].date()})')
print(f'  Columns: {list(df.columns)}')
HAS_COT = 'Comm_Positions_Long_All' in df.columns
_elapsed('1. Load data', t0)

# ══════════════════════════════════════════════════════════════════════════════
# 2. STATIONARITY + FRACTIONAL DIFFERENCING
# ══════════════════════════════════════════════════════════════════════════════
print('\n── 2. STATIONARITY + FRAC DIFF ─────────────────────────────────────────')
t0 = time.time()

def adf_test(series, name=''):
    result = adfuller(series.dropna())
    pval   = result[1]
    stat   = 'STAT ✓' if pval < 0.05 else 'NON-STAT ✗'
    print(f'  {name:<40} p={pval:.4f}  {stat}')
    return pval < 0.05

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

# Use training portion only for d selection — avoids test-period leakage in ADF.
# _split_raw approximates the 80/20 split before dropna; ADF is insensitive to
# a few rows' difference, so this is precise enough for hyperparameter selection.
_split_raw = int(len(df) * 0.80)

adf_test(df['log_Gold'].iloc[:_split_raw], 'log(Gold) [train]')

FRAC_D = None
for d in [0.3, 0.4, 0.5, 0.6, 0.7]:
    fd = frac_diff(df['log_Gold'], d=d)
    if adf_test(fd.iloc[:_split_raw].dropna(), f'FracDiff(log_Gold, d={d}) [train]'):
        df['frac_diff_log_Gold'] = fd
        FRAC_D = d
        break
if FRAC_D is None:
    FRAC_D = 0.7
    df['frac_diff_log_Gold'] = frac_diff(df['log_Gold'], d=FRAC_D)

TARGET = 'frac_diff_log_Gold'
print(f'  → TARGET = "{TARGET}"  (d={FRAC_D})')

# Fractionally difference non-stationary COT series (ADF on train portion only)
if HAS_COT:
    for col in ['Comm_Positions_Long_All', 'Comm_Positions_Short_All',
                'NonComm_Positions_Long_All', 'NonComm_Positions_Short_All',
                'Open_Interest_All']:
        if col in df.columns and not adf_test(df[col].iloc[:_split_raw].dropna(), col + ' [train]'):
            df[f'{col}_diff'] = frac_diff(df[col], d=FRAC_D)

_elapsed('2. Stationarity + FracDiff', t0)

# ══════════════════════════════════════════════════════════════════════════════
# 3. FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════
print('\n── 3. FEATURE ENGINEERING ───────────────────────────────────────────────')
t0 = time.time()

# COT features
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

# Macro features
if 'DXY' in df.columns:
    df['DXY_chg']      = df['DXY'].pct_change(5)
    df['DXY_ma20']     = df['DXY'].rolling(20).mean()
    df['DXY_vs_ma20']  = df['DXY'] / df['DXY_ma20'] - 1

if 'TNX_yield' in df.columns and 'VIX' in df.columns:
    tyield = df.get('TYX_yield', df['TNX_yield'])
    df['yield_spread']  = tyield - df['TNX_yield']
    df['real_yield_px'] = df['TNX_yield'] - df['VIX'] * 0.15

if 'VIX' in df.columns:
    vr = df['VIX'].rolling(60)
    df['VIX_zscore'] = (df['VIX'] - vr.mean()) / vr.std()

# Technical features
close = df['Gold'].values.astype(float)
if TALIB_AVAILABLE:
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
else:
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

# Target lags
for lag in [1, 2, 3, 5, 10]:
    df[f'target_lag{lag}'] = df[TARGET].shift(lag)

df['log_return'] = df['log_Gold'].diff()

# Multi-period momentum features
for period in [21, 63, 126, 252]:
    df[f'gold_ret_{period}d'] = df['log_Gold'].diff(period)

# Month-of-year dummies (gold documented seasonal patterns)
for m_num in range(1, 13):
    df[f'month_{m_num}'] = (df.index.month == m_num).astype(int)

# COT 20-day position changes (captures slower buildup vs existing 5-day)
if HAS_COT:
    df['net_spec_chg_20d'] = df['net_speculator'].diff(20)
    df['net_comm_chg_20d'] = df['net_commercial'].diff(20)

df.dropna(inplace=True)
print(f'  Feature matrix: {df.shape[0]} rows × {df.shape[1]} cols')
_elapsed('3. Feature engineering', t0)

# ══════════════════════════════════════════════════════════════════════════════
# 4. TRAIN / TEST SPLIT
# ══════════════════════════════════════════════════════════════════════════════
print('\n── 4. TRAIN / TEST SPLIT ────────────────────────────────────────────────')
t0 = time.time()

_cot_feats = ([
    'NonComm_Positions_Long_All', 'NonRept_Positions_Short_All',
    'net_commercial', 'net_speculator', 'net_nonreport',
    'comm_pct_OI', 'spec_pct_OI',
    'net_comm_chg', 'net_spec_chg', 'OI_chg',
    'comm_ma20', 'spec_ma20', 'comm_spec_spread',
    'cot_norm_spec', 'cot_norm_comm',
    'Comm_Positions_Long_All_diff', 'Comm_Positions_Short_All_diff',
    'NonComm_Positions_Short_All_diff', 'Open_Interest_All_diff',
] if HAS_COT else [])

_macro_feats = [
    'DXY', 'DXY_chg', 'DXY_ma20', 'DXY_vs_ma20',
    'VIX', 'VIX_zscore', 'TNX_yield', 'TYX_yield',
    'yield_spread', 'real_yield_px',
]
_ta_feats = [
    'RSI_14', 'RSI_28', 'ROC_10', 'ROC_20', 'MOM_5',
    'MACD_diff', 'BB_width', 'BB_pct',
]
_lag_feats      = [f'target_lag{l}' for l in [1, 2, 3, 5, 10]]
_momentum_feats = [f'gold_ret_{p}d' for p in [21, 63, 126, 252]]
_month_feats    = [f'month_{m}' for m in range(1, 13)]
_cot20_feats    = (['net_spec_chg_20d', 'net_comm_chg_20d'] if HAS_COT else [])

FEATURE_COLS = [c for c in (_cot_feats + _macro_feats + _ta_feats + _lag_feats
                             + _momentum_feats + _month_feats + _cot20_feats)
                if c in df.columns]
assert TARGET not in FEATURE_COLS, 'LEAKAGE!'

X      = df[FEATURE_COLS].values
y      = df[TARGET].values
dates  = df.index
n      = len(X)
split  = int(n * 0.80)

X_trainval, y_trainval = X[:split],  y[:split]
X_test,     y_test     = X[split:],  y[split:]
dates_test             = dates[split:]
log_ret_test           = df['log_return'].values[split:]

print(f'  Train+Val: {split} rows  ({dates[0].date()} → {dates[split-1].date()})')
print(f'  Test:      {len(X_test)} rows  ({dates_test[0].date()} → {dates_test[-1].date()})')
print(f'  Features:  {len(FEATURE_COLS)}')

# SVR epsilon anchored to target std (default ε=0.1 is 5.7× too large for frac-diff returns)
SVR_EPSILON = round(0.25 * float(np.std(y_trainval)), 6)
print(f'  Target std: {np.std(y_trainval):.6f}  →  SVR_EPSILON = {SVR_EPSILON:.6f} (0.25×std)')
_elapsed('4. Train/test split', t0)

# ══════════════════════════════════════════════════════════════════════════════
# 5. HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════
def make_sequences(X, y, ts):
    Xs, ys = [], []
    for i in range(ts, len(X)):
        Xs.append(X[i - ts:i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)

def directional_mse(y_true, y_pred):
    """Custom loss: MSE + sign-error penalty to prevent Keras mean-collapse on i.i.d. series."""
    mse      = tf.reduce_mean(tf.square(y_true - y_pred))
    sign_err = tf.maximum(0.0, -y_true * y_pred)
    return mse + 0.5 * tf.reduce_mean(sign_err)

def build_lstm(units=32, dropout=0.35, recurrent_dropout=0.2, n_features=1, timesteps=5):
    """Improved LSTM: recurrent dropout + L2 regularization + tuned lr."""
    m = Sequential([
        LSTM(units, input_shape=(timesteps, n_features),
             dropout=dropout,
             recurrent_dropout=recurrent_dropout,
             kernel_regularizer=l2(0.001)),
        Dense(1)
    ])
    m.compile(optimizer=Adam(learning_rate=0.001), loss=directional_mse)
    return m

def build_gru(units=32, dropout=0.3, recurrent_dropout=0.2, n_features=1, timesteps=5):
    """GRU: better gating than SimpleRNN, lighter than LSTM, less vanishing gradient."""
    m = Sequential([
        GRU(units, input_shape=(timesteps, n_features),
            dropout=dropout,
            recurrent_dropout=recurrent_dropout),
        Dense(1)
    ])
    m.compile(optimizer=Adam(learning_rate=0.001), loss=directional_mse)
    return m

def build_bilstm(units=32, dropout=0.4, recurrent_dropout=0.2, n_features=1, timesteps=5):
    """BiLSTM: gold-specific upgrade — macro regime shifts have both fwd/bwd patterns."""
    m = Sequential([
        Bidirectional(LSTM(units, recurrent_dropout=recurrent_dropout),
                      input_shape=(timesteps, n_features)),
        Dropout(dropout),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    m.compile(optimizer=Adam(learning_rate=0.001), loss=directional_mse)
    return m

# ══════════════════════════════════════════════════════════════════════════════
# 6. CV LOOP 1 — MODEL SELECTION (PARALLEL)
# ══════════════════════════════════════════════════════════════════════════════
print('\n── 5. CV LOOP 1 — MODEL SELECTION ──────────────────────────────────────')
t0 = time.time()

tscv    = TimeSeriesSplit(n_splits=N_CV_FOLDS)
results = {}

# ── sklearn: run all models × all folds in parallel ──────────────────────────
SK_MODELS = {
    'Ridge':   Ridge(alpha=1.0),
    'Lasso':   Lasso(alpha=1.0),
    'SVR_lin': LinearSVR(C=100, epsilon=SVR_EPSILON, max_iter=5000, dual=True),
    'SVR_rbf': SVR(kernel='rbf', C=0.1, epsilon=SVR_EPSILON, gamma=0.01),
    'XGB':     xgb.XGBRegressor(n_estimators=100, max_depth=3,
                                 learning_rate=0.05, random_state=SEED, verbosity=0),
    'RF':      RandomForestRegressor(n_estimators=100, random_state=SEED),
    'MLP':     MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=SEED),
}
SK_MODELS['ElasticNet'] = ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=5000)
if LIGHTGBM_AVAILABLE:
    SK_MODELS['LightGBM'] = lgb.LGBMRegressor(n_estimators=200, random_state=SEED, verbose=-1)

def _cv_one_fold(model_name, model, X_tr, y_tr, X_va, y_va):
    """Fit one model on one fold. Returns (model_name, mse).
    loky backend pickles a fresh copy into each subprocess — no manual copy needed."""
    sc = StandardScaler()
    model.fit(sc.fit_transform(X_tr), y_tr)
    preds = model.predict(sc.transform(X_va))
    return model_name, mean_squared_error(y_va, preds)

print(f'  Running {len(SK_MODELS)} sklearn models × {N_CV_FOLDS} folds in parallel (n_jobs={N_JOBS})...')
fold_jobs = [
    delayed(_cv_one_fold)(name, model, X_trainval[tr], y_trainval[tr],
                                       X_trainval[va], y_trainval[va])
    for name, model in SK_MODELS.items()
    for tr, va in tscv.split(X_trainval)
]
sk_results_flat = Parallel(n_jobs=N_JOBS, backend='loky', verbose=0)(fold_jobs)

# Aggregate per-model
for name in SK_MODELS:
    results[name] = [mse for (n, mse) in sk_results_flat if n == name]
    print(f'  {name:<22} CV MSE: {np.mean(results[name]):.8f} ± {np.std(results[name]):.8f}')

_elapsed('5a. sklearn CV Loop 1', t0)

# ── LSTM: loop over TIMESTEPS_CANDIDATES (TF uses its own parallelism) ────────
print(f'\n  LSTM — searching timesteps {TIMESTEPS_CANDS}...')
t0_lstm = time.time()
lstm_ts_results = {}

for ts in TIMESTEPS_CANDS:
    fold_mses = []
    sc = StandardScaler()
    for tr, va in tscv.split(X_trainval):
        X_tr_s = sc.fit_transform(X_trainval[tr])
        X_va_s = sc.transform(X_trainval[va])
        Xtr_sq, ytr_sq = make_sequences(X_tr_s, y_trainval[tr], ts)
        Xva_sq, yva_sq = make_sequences(X_va_s, y_trainval[va], ts)
        if len(Xva_sq) == 0:
            continue
        m = build_lstm(units=32, n_features=X_trainval.shape[1], timesteps=ts)
        m.fit(Xtr_sq, ytr_sq, epochs=30, batch_size=32, verbose=0,
              callbacks=[EarlyStopping(patience=5, restore_best_weights=True)])
        fold_mses.append(mean_squared_error(yva_sq, m.predict(Xva_sq, verbose=0).flatten()))
    lstm_ts_results[ts] = fold_mses
    print(f'    LSTM ts={ts:>2d}  CV MSE: {np.mean(fold_mses):.8f}')

LSTM_BEST_TS = min(lstm_ts_results, key=lambda t: np.mean(lstm_ts_results[t]))
results['LSTM'] = lstm_ts_results[LSTM_BEST_TS]
print(f'  {"LSTM":<22} best_ts={LSTM_BEST_TS}  CV MSE: {np.mean(results["LSTM"]):.8f}')
_elapsed('5b. LSTM CV Loop 1', t0_lstm)

# ── GRU ───────────────────────────────────────────────────────────────────────
print(f'\n  GRU — searching timesteps {TIMESTEPS_CANDS}...')
t0_gru = time.time()
gru_ts_results = {}

for ts in TIMESTEPS_CANDS:
    fold_mses = []
    sc = StandardScaler()
    for tr, va in tscv.split(X_trainval):
        X_tr_s = sc.fit_transform(X_trainval[tr])
        X_va_s = sc.transform(X_trainval[va])
        Xtr_sq, ytr_sq = make_sequences(X_tr_s, y_trainval[tr], ts)
        Xva_sq, yva_sq = make_sequences(X_va_s, y_trainval[va], ts)
        if len(Xva_sq) == 0:
            continue
        m = build_gru(units=32, n_features=X_trainval.shape[1], timesteps=ts)
        m.fit(Xtr_sq, ytr_sq, epochs=50, batch_size=32, verbose=0,
              callbacks=[EarlyStopping(patience=10, restore_best_weights=True)])
        fold_mses.append(mean_squared_error(yva_sq, m.predict(Xva_sq, verbose=0).flatten()))
    gru_ts_results[ts] = fold_mses
    print(f'    GRU ts={ts:>2d}  CV MSE: {np.mean(fold_mses):.8f}')

GRU_BEST_TS = min(gru_ts_results, key=lambda t: np.mean(gru_ts_results[t]))
results['GRU'] = gru_ts_results[GRU_BEST_TS]
print(f'  {"GRU":<22} best_ts={GRU_BEST_TS}  CV MSE: {np.mean(results["GRU"]):.8f}')
_elapsed('5c. GRU CV Loop 1', t0_gru)

# ── BiLSTM ────────────────────────────────────────────────────────────────────
print(f'\n  BiLSTM — searching timesteps {TIMESTEPS_CANDS}...')
t0_bi = time.time()
bilstm_ts_results = {}

for ts in TIMESTEPS_CANDS:
    fold_mses = []
    sc = StandardScaler()
    for tr, va in tscv.split(X_trainval):
        X_tr_s = sc.fit_transform(X_trainval[tr])
        X_va_s = sc.transform(X_trainval[va])
        Xtr_sq, ytr_sq = make_sequences(X_tr_s, y_trainval[tr], ts)
        Xva_sq, yva_sq = make_sequences(X_va_s, y_trainval[va], ts)
        if len(Xva_sq) == 0:
            continue
        m = build_bilstm(units=32, n_features=X_trainval.shape[1], timesteps=ts)
        m.fit(Xtr_sq, ytr_sq, epochs=60, batch_size=32, verbose=0,
              callbacks=[EarlyStopping(patience=10, restore_best_weights=True)])
        fold_mses.append(mean_squared_error(yva_sq, m.predict(Xva_sq, verbose=0).flatten()))
    bilstm_ts_results[ts] = fold_mses
    print(f'    BiLSTM ts={ts:>2d}  CV MSE: {np.mean(fold_mses):.8f}')

BILSTM_BEST_TS = min(bilstm_ts_results, key=lambda t: np.mean(bilstm_ts_results[t]))
results['BiLSTM'] = bilstm_ts_results[BILSTM_BEST_TS]
print(f'  {"BiLSTM":<22} best_ts={BILSTM_BEST_TS}  CV MSE: {np.mean(results["BiLSTM"]):.8f}')
_elapsed('5d. BiLSTM CV Loop 1', t0_bi)

# ── ARIMAX ────────────────────────────────────────────────────────────────────
if PMDARIMA_AVAILABLE:
    t0_arima = time.time()
    print('\n  ARIMAX CV...')
    arimax_mses = []
    sc = StandardScaler()
    for tr, va in tscv.split(X_trainval):
        try:
            X_tr_s = sc.fit_transform(X_trainval[tr])
            X_va_s = sc.transform(X_trainval[va])
            arima_m = pm.auto_arima(y_trainval[tr], exogenous=X_tr_s, seasonal=False,
                                    stepwise=True, suppress_warnings=True,
                                    max_p=3, max_q=3, max_d=1,
                                    error_action='ignore', trace=False)
            preds = arima_m.predict(n_periods=len(va), exogenous=X_va_s)
            arimax_mses.append(mean_squared_error(y_trainval[va], preds))
        except Exception as e:
            print(f'    fold failed: {e}')
    if arimax_mses:
        results['ARIMAX'] = arimax_mses
        print(f'  {"ARIMAX":<22} CV MSE: {np.mean(arimax_mses):.8f}')
    _elapsed('5d. ARIMAX CV Loop 1', t0_arima)

# ── Pick best model ───────────────────────────────────────────────────────────
mean_mses       = {n: np.mean(v) for n, v in results.items()}
BEST_MODEL_NAME = min(mean_mses, key=mean_mses.get)

print(f'\n  Model ranking (CV MSE):')
for name, mse in sorted(mean_mses.items(), key=lambda x: x[1]):
    mark = ' ← BEST' if name == BEST_MODEL_NAME else ''
    print(f'    {name:<22} {mse:.8f}{mark}')

# Resolve TIMESTEPS
if BEST_MODEL_NAME == 'LSTM':
    TIMESTEPS = LSTM_BEST_TS
elif BEST_MODEL_NAME == 'GRU':
    TIMESTEPS = GRU_BEST_TS
elif BEST_MODEL_NAME == 'BiLSTM':
    TIMESTEPS = BILSTM_BEST_TS
else:
    TIMESTEPS = LSTM_BEST_TS   # used for all-model Keras comparison

# Save CV MSE data for plot_all.py
cv_mse_data = {'best_model': BEST_MODEL_NAME,
               'models': {n: float(v) for n, v in mean_mses.items()}}
with open(f'{RESULTS_DIR}/cv_mse.json', 'w') as f:
    json.dump(cv_mse_data, f, indent=2)
print(f'  CV MSE data saved → {RESULTS_DIR}/cv_mse.json')

_elapsed('5. CV Loop 1 total', t0)

# ══════════════════════════════════════════════════════════════════════════════
# 7. CV LOOP 2 — FEATURE SELECTION + HYPERPARAMETER TUNING (PARALLEL)
# ══════════════════════════════════════════════════════════════════════════════
print(f'\n── 6. CV LOOP 2 — HYPERPARAM TUNING ({BEST_MODEL_NAME}) ─────────────────')
t0 = time.time()

tscv2 = TimeSeriesSplit(n_splits=N_CV_FOLDS)
k_values = [10, 20, 30, len(FEATURE_COLS)]

best_score, best_params, best_k, best_feature_mask = np.inf, {}, len(FEATURE_COLS), None

if BEST_MODEL_NAME in ['XGB', 'XGBoost']:
    param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [2, 3, 4],
                  'learning_rate': [0.01, 0.05, 0.1]}
    def _make_model(p): return xgb.XGBRegressor(**p, random_state=SEED, verbosity=0)
elif BEST_MODEL_NAME == 'Ridge':
    param_grid = {'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]}
    def _make_model(p): return Ridge(**p)
elif BEST_MODEL_NAME == 'Lasso':
    param_grid = {'alpha': [0.0001, 0.001, 0.01, 0.1, 1.0]}
    def _make_model(p): return Lasso(**p)
elif BEST_MODEL_NAME in ['SVR_rbf', 'SVR_lin']:
    eps_candidates = [round(f * SVR_EPSILON, 6) for f in [0.1, 0.25, 0.5, 1.0, 2.0]]
    if BEST_MODEL_NAME == 'SVR_lin':
        param_grid = {'C': [1, 10, 100], 'epsilon': eps_candidates}
        def _make_model(p): return LinearSVR(max_iter=5000, dual=True, **p)
    else:
        param_grid = {'C': [0.1, 1, 10], 'epsilon': eps_candidates, 'gamma': ['scale', 0.01, 0.1]}
        def _make_model(p): return SVR(kernel='rbf', **p)
elif BEST_MODEL_NAME == 'RF':
    param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [3, 5, None]}
    def _make_model(p): return RandomForestRegressor(**p, random_state=SEED)
elif BEST_MODEL_NAME == 'MLP':
    param_grid = {'hidden_layer_sizes': [(32,), (64, 32), (128, 64)], 'alpha': [0.001, 0.01]}
    def _make_model(p): return MLPRegressor(**p, max_iter=500, random_state=SEED)
elif BEST_MODEL_NAME in ['LSTM', 'GRU', 'BiLSTM']:
    param_grid = {'timesteps': TIMESTEPS_CANDS, 'units': [16, 32, 64], 'dropout': [0.2, 0.3, 0.4]}
    _make_model = None   # handled separately
elif BEST_MODEL_NAME == 'ElasticNet':
    param_grid = {'alpha': [0.0001, 0.001, 0.01, 0.1], 'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]}
    def _make_model(p): return ElasticNet(max_iter=5000, **p)
elif BEST_MODEL_NAME == 'LightGBM':
    param_grid = {'n_estimators': [100, 200], 'num_leaves': [15, 31], 'learning_rate': [0.01, 0.05, 0.1]}
    def _make_model(p): return lgb.LGBMRegressor(random_state=SEED, verbose=-1, **p)
else:
    param_grid = {}
    _make_model = None

if param_grid and BEST_MODEL_NAME not in ['LSTM', 'GRU', 'BiLSTM']:
    # SelectKBest is included in the Pipeline so GridSearchCV re-fits it on
    # each training fold only — avoids leaking validation-fold label info
    # into feature selection (previously selector was fit on all X_trainval).
    if BEST_MODEL_NAME == 'Ridge':
        base = Ridge()
    elif BEST_MODEL_NAME == 'Lasso':
        base = Lasso()
    elif BEST_MODEL_NAME == 'SVR_rbf':
        base = SVR(kernel='rbf', epsilon=SVR_EPSILON)
    elif BEST_MODEL_NAME == 'SVR_lin':
        base = LinearSVR(epsilon=SVR_EPSILON, max_iter=5000, dual=True)
    elif BEST_MODEL_NAME in ['XGB', 'XGBoost']:
        base = xgb.XGBRegressor(random_state=SEED, verbosity=0)
    elif BEST_MODEL_NAME == 'RF':
        base = RandomForestRegressor(random_state=SEED)
    elif BEST_MODEL_NAME == 'MLP':
        base = MLPRegressor(max_iter=500, random_state=SEED)
    elif BEST_MODEL_NAME == 'ElasticNet':
        base = ElasticNet(max_iter=5000)
    elif BEST_MODEL_NAME == 'LightGBM' and LIGHTGBM_AVAILABLE:
        base = lgb.LGBMRegressor(random_state=SEED, verbose=-1)
    else:
        base = Ridge()

    pipe = Pipeline([
        ('selector', SelectKBest(score_func=mutual_info_regression)),
        ('scaler',   StandardScaler()),
        ('model',    base),
    ])
    pg_prefixed = {f'model__{k}': v for k, v in param_grid.items()}
    pg_prefixed['selector__k'] = [min(k, X_trainval.shape[1]) for k in k_values]
    gs = GridSearchCV(pipe, pg_prefixed, cv=tscv2,
                      scoring='neg_mean_squared_error', n_jobs=N_JOBS, refit=True)
    gs.fit(X_trainval, y_trainval)
    best_score  = -gs.best_score_
    best_params = {k.replace('model__', ''): v
                   for k, v in gs.best_params_.items()
                   if k.startswith('model__')}
    best_k      = gs.best_params_['selector__k']
    # Re-fit selector on full trainval with best k to get the feature mask
    best_selector = SelectKBest(score_func=mutual_info_regression,
                                k=best_k).fit(X_trainval, y_trainval)
    best_feature_mask = best_selector.get_support()
    print(f'  Best k={best_k}  params={best_params}  CV MSE={best_score:.8f}')

elif BEST_MODEL_NAME in ['LSTM', 'GRU', 'BiLSTM']:
    # Sequential Keras grid (TF handles its own thread parallelism)
    if BEST_MODEL_NAME == 'LSTM':
        build_fn = build_lstm
    elif BEST_MODEL_NAME == 'GRU':
        build_fn = build_gru
    else:
        build_fn = build_bilstm
    for k in k_values:
        for vals in product(*param_grid.values()):
            params = dict(zip(param_grid.keys(), vals))
            ts     = params['timesteps']
            fold_mses = []
            for tr, va in tscv2.split(X_trainval):
                # Selector fit on training fold only — no leakage into val fold
                selector = SelectKBest(score_func=mutual_info_regression,
                                       k=min(k, X_trainval.shape[1]))
                X_tr_sel = selector.fit_transform(X_trainval[tr], y_trainval[tr])
                X_va_sel = selector.transform(X_trainval[va])
                sc = StandardScaler()
                Xtr = sc.fit_transform(X_tr_sel)
                Xva = sc.transform(X_va_sel)
                Xtr_sq, ytr_sq = make_sequences(Xtr, y_trainval[tr], ts)
                Xva_sq, yva_sq = make_sequences(Xva, y_trainval[va], ts)
                if len(Xva_sq) == 0:
                    continue
                m = build_fn(units=params['units'], dropout=params['dropout'],
                             n_features=X_tr_sel.shape[1], timesteps=ts)
                m.fit(Xtr_sq, ytr_sq, epochs=20, batch_size=32, verbose=0,
                      callbacks=[EarlyStopping(patience=3, restore_best_weights=True)])
                fold_mses.append(mean_squared_error(yva_sq,
                                 m.predict(Xva_sq, verbose=0).flatten()))
            score = np.mean(fold_mses) if fold_mses else np.inf
            if score < best_score:
                best_score, best_params, best_k = score, params, k
    if 'timesteps' in best_params:
        TIMESTEPS = best_params['timesteps']
    # Re-fit selector on full trainval with best k for final feature mask
    best_selector = SelectKBest(score_func=mutual_info_regression,
                                k=best_k).fit(X_trainval, y_trainval)
    best_feature_mask = best_selector.get_support()
    print(f'  Best params={best_params}  CV MSE={best_score:.8f}')

if best_feature_mask is not None:
    selected_features = [FEATURE_COLS[i] for i, s in enumerate(best_feature_mask) if s]
else:
    selected_features = FEATURE_COLS
    best_k = len(FEATURE_COLS)

print(f'  Best k={best_k}  params={best_params}  TIMESTEPS={TIMESTEPS}')
_elapsed('6. CV Loop 2', t0)

# ══════════════════════════════════════════════════════════════════════════════
# 8. FINAL MODEL — FIT ON FULL TRAINVAL, EVALUATE ON TEST
# ══════════════════════════════════════════════════════════════════════════════
print(f'\n── 7. FINAL MODEL: {BEST_MODEL_NAME} ────────────────────────────────────')
t0 = time.time()

# Apply feature selection
X_tv_f = X_trainval[:, best_feature_mask] if best_feature_mask is not None else X_trainval
X_te_f = X_test[:,    best_feature_mask] if best_feature_mask is not None else X_test
feat_names_final = selected_features

final_scaler = StandardScaler()
X_tv_s = final_scaler.fit_transform(X_tv_f)
X_te_s = final_scaler.transform(X_te_f)

def _build_final(name, params, n_features):
    if name in ['XGB', 'XGBoost']:
        return xgb.XGBRegressor(n_estimators=params.get('n_estimators', 100),
                                 max_depth=params.get('max_depth', 3),
                                 learning_rate=params.get('learning_rate', 0.05),
                                 random_state=SEED, verbosity=0)
    if name == 'Ridge':    return Ridge(alpha=params.get('alpha', 1.0))
    if name == 'Lasso':    return Lasso(alpha=params.get('alpha', 0.01))
    if name == 'SVR_rbf':  return SVR(kernel='rbf', **{k: v for k, v in params.items()})
    if name == 'SVR_lin':  return LinearSVR(max_iter=5000, dual=True, **{k: v for k, v in params.items()})
    if name == 'MLP':
        return MLPRegressor(hidden_layer_sizes=params.get('hidden_layer_sizes', (64,32)),
                            alpha=params.get('alpha', 0.01), max_iter=500, random_state=SEED)
    if name == 'RF':
        return RandomForestRegressor(n_estimators=params.get('n_estimators', 100),
                                     max_depth=params.get('max_depth', None), random_state=SEED)
    return Ridge()

if BEST_MODEL_NAME in ['LSTM', 'GRU', 'BiLSTM']:
    if BEST_MODEL_NAME == 'LSTM':
        build_fn = build_lstm
    elif BEST_MODEL_NAME == 'GRU':
        build_fn = build_gru
    else:
        build_fn = build_bilstm
    units       = best_params.get('units', 32)
    dropout     = best_params.get('dropout', 0.35)
    print(f'  Building {BEST_MODEL_NAME}(ts={TIMESTEPS}, units={units}, dropout={dropout})')
    final_model = build_fn(units=units, dropout=dropout,
                           n_features=X_tv_f.shape[1], timesteps=TIMESTEPS)
    Xtv_sq, ytv_sq = make_sequences(X_tv_s, y_trainval, TIMESTEPS)
    Xte_sq, yte_sq = make_sequences(X_te_s, y_test,     TIMESTEPS)
    final_model.fit(Xtv_sq, ytv_sq, epochs=50, batch_size=32, verbose=1,
                    callbacks=[EarlyStopping(patience=5, restore_best_weights=True)])
    y_pred_test = final_model.predict(Xte_sq, verbose=0).flatten()
    test_mse    = mean_squared_error(yte_sq, y_pred_test)
elif BEST_MODEL_NAME == 'ARIMAX' and PMDARIMA_AVAILABLE:
    final_model = pm.auto_arima(y_trainval, exogenous=X_tv_s, seasonal=False,
                                stepwise=True, suppress_warnings=True,
                                max_p=3, max_q=3, max_d=1, error_action='ignore')
    y_pred_test = final_model.predict(n_periods=len(y_test), exogenous=X_te_s)
    test_mse    = mean_squared_error(y_test, y_pred_test)
else:
    final_model = _build_final(BEST_MODEL_NAME, best_params, X_tv_f.shape[1])
    final_model.fit(X_tv_s, y_trainval)
    y_pred_test = final_model.predict(X_te_s)
    test_mse    = mean_squared_error(y_test, y_pred_test)

n_pred = len(y_pred_test)
print(f'  Test MSE:  {test_mse:.8f}')
print(f'  Test RMSE: {np.sqrt(test_mse):.8f}')

# Save best model predictions for plot_all.py
pd.DataFrame({
    'date':    dates_test[:n_pred].strftime('%Y-%m-%d'),
    'y_test':  y_test[:n_pred],
    'y_pred':  y_pred_test,
    'log_ret': log_ret_test[:n_pred],
}).to_csv(f'{RESULTS_DIR}/best_model_predictions.csv', index=False)
print(f'  Best model predictions saved → {RESULTS_DIR}/best_model_predictions.csv')
_elapsed('7. Final model', t0)

# ══════════════════════════════════════════════════════════════════════════════
# 9. TRADING METRICS
# ══════════════════════════════════════════════════════════════════════════════
print('\n── 8. TRADING METRICS ───────────────────────────────────────────────────')
t0 = time.time()

pred     = y_pred_test
log_ret  = log_ret_test[:n_pred]
tdates   = dates_test[:n_pred]

ps          = pd.Series(pred)
pred_z      = (ps - ps.rolling(5, min_periods=1).mean()) / ps.rolling(5, min_periods=1).std().fillna(1e-8)
signal      = np.where(pred_z > 0, 1, -1).astype(float)
strat_ret   = signal * log_ret
equity_strat = np.exp(np.cumsum(strat_ret))
equity_bnh   = np.exp(np.cumsum(log_ret))

def profit_factor(lr):
    r = pd.Series(lr)
    g, l = r[r>0].sum(), abs(r[r<0].sum())
    return float(g/l) if l > 0 else np.inf

def cagr(eq, n_days):
    return float(eq[-1] ** (252/n_days) - 1)

def sharpe(lr, ann=252):
    r = np.exp(lr) - 1
    return float(r.mean() / r.std() * np.sqrt(ann))

def max_dd(eq):
    peak = np.maximum.accumulate(eq)
    return float(((eq - peak)/peak).min())

pf_s   = profit_factor(strat_ret)
cagr_s = cagr(equity_strat, len(strat_ret))
sr_s   = sharpe(strat_ret)
mdd_s  = max_dd(equity_strat)

print(f'  {"Metric":<20} {"Strategy":>12} {"Buy&Hold":>12}')
print(f'  {"-"*46}')
print(f'  {"CAGR":<20} {cagr_s:>12.2%} {cagr(equity_bnh, len(log_ret)):>12.2%}')
print(f'  {"Sharpe":<20} {sr_s:>12.4f} {sharpe(log_ret):>12.4f}')
print(f'  {"Profit Factor":<20} {pf_s:>12.4f} {profit_factor(log_ret):>12.4f}')
print(f'  {"Max Drawdown":<20} {mdd_s:>12.2%} {max_dd(equity_bnh):>12.2%}')

# Save strategy data for plot_all.py
pd.DataFrame({
    'date':         tdates.strftime('%Y-%m-%d'),
    'equity_strat': equity_strat,
    'equity_bnh':   equity_bnh,
    'strat_ret':    strat_ret,
    'signal':       signal,
}).to_csv(f'{RESULTS_DIR}/best_model_strategy.csv', index=False)
print(f'  Strategy data saved → {RESULTS_DIR}/best_model_strategy.csv')
_elapsed('8. Trading metrics', t0)

# ══════════════════════════════════════════════════════════════════════════════
# 10. ALL-MODEL TRADING METRICS (PARALLEL SKLEARN)
# ══════════════════════════════════════════════════════════════════════════════
print('\n── 9. ALL-MODEL TRADING METRICS ─────────────────────────────────────────')
t0 = time.time()

def _trading_metrics(preds, log_rets, y_true):
    n          = len(preds)
    ps         = pd.Series(preds)
    z          = (ps - ps.rolling(ZSCORE_WIN, min_periods=1).mean()) / ps.rolling(ZSCORE_WIN, min_periods=1).std().fillna(1e-8)
    sig        = np.where(z > 0, 1, -1).astype(float)
    strat      = sig * np.array(log_rets[:n])
    eq         = np.exp(np.cumsum(strat))
    market_dir = np.sign(np.array(log_rets[:n]))
    dir_acc    = float(np.mean(sig == market_dir))   # signal vs actual market direction
    return dict(
        MSE    = mean_squared_error(y_true[:n], preds),
        CAGR   = float(eq[-1] ** (252/n) - 1),
        Sharpe = float(np.mean(strat)/(np.std(strat)+1e-12)*np.sqrt(252)),
        PF     = float(strat[strat>0].sum()/abs(strat[strat<0].sum()))
                 if strat[strat<0].sum() != 0 else np.inf,
        MaxDD  = float(((eq - np.maximum.accumulate(eq))/np.maximum.accumulate(eq)).min()),
        DirAcc = dir_acc,
        equity = eq,
        preds  = np.array(preds),
    )

_sc  = StandardScaler()
_Xtv = _sc.fit_transform(X_trainval)
_Xte = _sc.transform(X_test)

# ── Dedicated SVR tuning (always runs, independent of best model) ─────────────
print('\n── 9a. SVR HYPERPARAMETER TUNING ────────────────────────────────────────')
t0_svr = time.time()
_tscv_svr = TimeSeriesSplit(n_splits=N_CV_FOLDS)

# SVR_lin: tune C (epsilon fixed at SVR_EPSILON)
_svr_lin_grid = GridSearchCV(
    Pipeline([('sc', StandardScaler()),
              ('m',  LinearSVR(epsilon=SVR_EPSILON, max_iter=10000, dual=True))]),
    param_grid={'m__C': [0.1, 1, 10, 100, 1000]},
    cv=_tscv_svr, scoring='neg_mean_squared_error', n_jobs=N_JOBS, refit=True)
_svr_lin_grid.fit(X_trainval, y_trainval)
SVR_LIN_BEST_C = _svr_lin_grid.best_params_['m__C']
SVR_LIN_BEST_MSE = -_svr_lin_grid.best_score_
print(f'  SVR_lin best: C={SVR_LIN_BEST_C}  CV MSE={SVR_LIN_BEST_MSE:.8f}')

# SVR_rbf: tune C and gamma (epsilon fixed at SVR_EPSILON)
_svr_rbf_grid = GridSearchCV(
    Pipeline([('sc', StandardScaler()),
              ('m',  SVR(kernel='rbf', epsilon=SVR_EPSILON))]),
    param_grid={'m__C':     [0.01, 0.1, 1, 10, 100],
                'm__gamma': ['scale', 0.001, 0.01, 0.1]},
    cv=_tscv_svr, scoring='neg_mean_squared_error', n_jobs=N_JOBS, refit=True)
_svr_rbf_grid.fit(X_trainval, y_trainval)
SVR_RBF_BEST_C     = _svr_rbf_grid.best_params_['m__C']
SVR_RBF_BEST_GAMMA = _svr_rbf_grid.best_params_['m__gamma']
SVR_RBF_BEST_MSE   = -_svr_rbf_grid.best_score_
print(f'  SVR_rbf best: C={SVR_RBF_BEST_C}  gamma={SVR_RBF_BEST_GAMMA}  CV MSE={SVR_RBF_BEST_MSE:.8f}')
_elapsed('9a. SVR tuning', t0_svr)

ALL_SK = {
    'Ridge':        Ridge(alpha=best_params.get('alpha', 0.01) if BEST_MODEL_NAME=='Ridge' else 0.01),
    'Lasso':        Lasso(alpha=0.001),
    'SVR_lin':      LinearSVR(C=SVR_LIN_BEST_C, epsilon=SVR_EPSILON, max_iter=10000, dual=True),
    'SVR_rbf':      SVR(kernel='rbf', C=SVR_RBF_BEST_C, epsilon=SVR_EPSILON, gamma=SVR_RBF_BEST_GAMMA),
    'XGBoost':      xgb.XGBRegressor(n_estimators=100, max_depth=3,
                                      learning_rate=0.05, random_state=SEED, verbosity=0),
    'RandomForest': RandomForestRegressor(n_estimators=100, random_state=SEED),
    'MLP':          MLPRegressor(hidden_layer_sizes=(64,32), max_iter=500, random_state=SEED),
}

def _fit_predict(name, model, Xtv, Xte):
    model.fit(Xtv, y_trainval)
    return name, model.predict(Xte), model

print(f'  Training {len(ALL_SK)} sklearn models in parallel...')
sk_preds = Parallel(n_jobs=N_JOBS, backend='loky')(
    delayed(_fit_predict)(name, model, _Xtv, _Xte)
    for name, model in ALL_SK.items()
)

all_metrics = {}
sk_fitted_models = {}
for name, preds, fitted_model in sk_preds:
    all_metrics[name] = _trading_metrics(preds, log_ret_test, y_test)
    sk_fitted_models[name] = fitted_model
    print(f'  {name} ✓')

# LSTM — use its own best timesteps
Xtr_sq_l, ytr_sq_l = make_sequences(_Xtv, y_trainval, LSTM_BEST_TS)
Xte_sq_l, yte_sq_l = make_sequences(_Xte, y_test,     LSTM_BEST_TS)
_lstm2 = build_lstm(units=32, n_features=_Xtv.shape[1], timesteps=LSTM_BEST_TS)
_lstm2.fit(Xtr_sq_l, ytr_sq_l, epochs=50, batch_size=32, verbose=0,
           callbacks=[EarlyStopping(patience=10, restore_best_weights=True)])
all_metrics['LSTM'] = _trading_metrics(
    _lstm2.predict(Xte_sq_l, verbose=0).flatten(), log_ret_test[LSTM_BEST_TS:], y_test[LSTM_BEST_TS:])
print(f'  LSTM (ts={LSTM_BEST_TS}) ✓')

# GRU — use its own best timesteps
Xtr_sq_g, ytr_sq_g = make_sequences(_Xtv, y_trainval, GRU_BEST_TS)
Xte_sq_g, yte_sq_g = make_sequences(_Xte, y_test,     GRU_BEST_TS)
_gru2 = build_gru(units=32, n_features=_Xtv.shape[1], timesteps=GRU_BEST_TS)
_gru2.fit(Xtr_sq_g, ytr_sq_g, epochs=50, batch_size=32, verbose=0,
          callbacks=[EarlyStopping(patience=10, restore_best_weights=True)])
all_metrics['GRU'] = _trading_metrics(
    _gru2.predict(Xte_sq_g, verbose=0).flatten(), log_ret_test[GRU_BEST_TS:], y_test[GRU_BEST_TS:])
print(f'  GRU (ts={GRU_BEST_TS}) ✓')

# BiLSTM — use its own best timesteps
Xtr_sq_b, ytr_sq_b = make_sequences(_Xtv, y_trainval, BILSTM_BEST_TS)
Xte_sq_b, yte_sq_b = make_sequences(_Xte, y_test,     BILSTM_BEST_TS)
_bilstm2 = build_bilstm(units=32, n_features=_Xtv.shape[1], timesteps=BILSTM_BEST_TS)
_bilstm2.fit(Xtr_sq_b, ytr_sq_b, epochs=60, batch_size=32, verbose=0,
             callbacks=[EarlyStopping(patience=10, restore_best_weights=True)])
all_metrics['BiLSTM'] = _trading_metrics(
    _bilstm2.predict(Xte_sq_b, verbose=0).flatten(), log_ret_test[BILSTM_BEST_TS:], y_test[BILSTM_BEST_TS:])
print(f'  BiLSTM (ts={BILSTM_BEST_TS}) ✓')

if PMDARIMA_AVAILABLE:
    try:
        _arima2 = pm.auto_arima(y_trainval, exogenous=_Xtv, seasonal=False,
                                stepwise=True, suppress_warnings=True,
                                max_p=3, max_q=3, max_d=1, error_action='ignore', trace=False)
        all_metrics['ARIMAX'] = _trading_metrics(
            _arima2.predict(n_periods=len(y_test), exogenous=_Xte), log_ret_test, y_test)
        print('  ARIMAX ✓')
    except Exception as e:
        print(f'  ARIMAX failed: {e}')

# Summary table
metrics_df = pd.DataFrame({
    n: {k: v for k, v in m.items() if k not in ('equity', 'preds')}
    for n, m in all_metrics.items()
}).T.sort_values('Sharpe', ascending=False)

print(f'\n  All-Model Metrics (sorted by Sharpe):')
print(metrics_df[['MSE','CAGR','Sharpe','PF','MaxDD','DirAcc']].to_string(
    float_format=lambda x: f'{x:.4f}'))
metrics_df[['MSE','CAGR','Sharpe','PF','MaxDD','DirAcc']].to_csv(f'{RESULTS_DIR}/all_model_metrics.csv')

_elapsed('9. All-model metrics', t0)

# ══════════════════════════════════════════════════════════════════════════════
# 9b. PREDICTED VS ACTUAL — ALL MODELS  (Signal Dir Acc in subtitle)
# ══════════════════════════════════════════════════════════════════════════════
print('\n── 9b. SAVE ALL-MODEL PREDICTIONS ──────────────────────────────────────')
t0_pva = time.time()

# Save all model predictions + y_test + log_return for plot_all.py
_pred_df = pd.DataFrame({'date': dates_test.strftime('%Y-%m-%d'),
                          'y_test': y_test, 'log_return': log_ret_test})
for _nm in all_metrics:
    _p = all_metrics[_nm]['preds']
    _col = pd.Series(np.nan, index=range(len(dates_test)), dtype=float)
    _col.iloc[:len(_p)] = _p
    _pred_df[f'pred_{_nm}'] = _col.values
_pred_df.to_csv(f'{RESULTS_DIR}/test_predictions.csv', index=False)
print(f'  All-model predictions saved → {RESULTS_DIR}/test_predictions.csv')
_elapsed('9b. Save predictions', t0_pva)

# ══════════════════════════════════════════════════════════════════════════════
# 11. SAVE MODELS
# ══════════════════════════════════════════════════════════════════════════════
print('\n── 10. SAVE MODELS ──────────────────────────────────────────────────────')
t0 = time.time()

if BEST_MODEL_NAME in ['LSTM', 'GRU', 'BiLSTM']:
    final_model.save(f'{MODELS_DIR}/best_model_{BEST_MODEL_NAME}.keras')
elif BEST_MODEL_NAME == 'ARIMAX':
    joblib.dump(final_model, f'{MODELS_DIR}/best_model_ARIMAX.joblib')
else:
    joblib.dump(final_model, f'{MODELS_DIR}/best_model_{BEST_MODEL_NAME}.joblib')

joblib.dump(final_scaler, f'{MODELS_DIR}/final_scaler.joblib')
joblib.dump(_sc,          f'{MODELS_DIR}/all_models_scaler.joblib')

for name, model in ALL_SK.items():
    joblib.dump(model, f'{MODELS_DIR}/model_{name}.joblib')

_lstm2.save(f'{MODELS_DIR}/model_LSTM.keras')
_gru2.save(f'{MODELS_DIR}/model_GRU.keras')
_bilstm2.save(f'{MODELS_DIR}/model_BiLSTM.keras')

meta = {
    'target':             TARGET,
    'frac_d':             FRAC_D,
    'best_model':         BEST_MODEL_NAME,
    'best_params':        {k: (int(v) if isinstance(v, (np.integer,)) else
                               list(v) if isinstance(v, tuple) else v)
                           for k, v in best_params.items()},
    'best_k':             best_k,
    'feature_cols':       FEATURE_COLS,
    'selected_features':  selected_features,
    'timesteps_candidates':  TIMESTEPS_CANDS,
    'lstm_best_timesteps':   LSTM_BEST_TS,
    'gru_best_timesteps':    GRU_BEST_TS,
    'bilstm_best_timesteps': BILSTM_BEST_TS,
    'timesteps':             TIMESTEPS,
    'has_cot':            HAS_COT,
    'train_end_date':     str(dates[split-1].date()),
    'test_start_date':    str(dates_test[0].date()),
    'test_mse':           float(test_mse),
    'cagr':               float(cagr_s),
    'sharpe':             float(sr_s),
    'profit_factor':      float(pf_s),
    'max_drawdown':       float(mdd_s),
    'svr_epsilon':        float(SVR_EPSILON),
    'svr_lin_best_C':     float(SVR_LIN_BEST_C),
    'svr_rbf_best_C':     float(SVR_RBF_BEST_C),
    'svr_rbf_best_gamma': str(SVR_RBF_BEST_GAMMA),
}
with open(f'{MODELS_DIR}/model_metadata.json', 'w') as f:
    json.dump(meta, f, indent=2)

for fname in sorted(os.listdir(MODELS_DIR)):
    kb = os.path.getsize(f'{MODELS_DIR}/{fname}') / 1024
    print(f'  {fname:<40} {kb:>8.1f} KB')
_elapsed('10. Save models', t0)

# ══════════════════════════════════════════════════════════════════════════════
# 12. SHAP — BEST MODEL (beeswarm) + ALL MODELS (bar grid)
# ══════════════════════════════════════════════════════════════════════════════
print('\n── 11. SHAP — ALL MODELS ────────────────────────────────────────────────')
t0 = time.time()

# ── Best model: full beeswarm ────────────────────────────────────────────────
if BEST_MODEL_NAME in ['XGBoost', 'RF', 'RandomForest']:
    _exp_best = shap.TreeExplainer(final_model)
    _sv_best  = _exp_best.shap_values(X_te_s)
else:
    _proxy = Ridge(alpha=0.01)
    _proxy.fit(X_tv_s, y_trainval)
    _bg        = shap.kmeans(X_tv_s, 20)
    _exp_best  = shap.KernelExplainer(_proxy.predict, _bg)
    _sv_best   = _exp_best.shap_values(X_te_s[:100])

np.save(f'{RESULTS_DIR}/shap_best_values.npy', _sv_best)
np.save(f'{RESULTS_DIR}/shap_best_X.npy', X_te_s[:len(_sv_best)])
print(f'  Best model SHAP saved → {RESULTS_DIR}/shap_best_values.npy')

# ── All models: compute SHAP values ─────────────────────────────────────────
print('  Computing SHAP for all models...')
SHAP_N  = 200   # test samples
SHAP_BG = 50    # background samples
X_shap_test = _Xte[:SHAP_N]
X_shap_bg   = _Xtv[:SHAP_BG]

all_shap_vals = {}   # name -> np.ndarray (n_samples, n_features)

# Tree models
for _name in ['XGBoost', 'RandomForest']:
    if _name in sk_fitted_models:
        try:
            _sv = shap.TreeExplainer(sk_fitted_models[_name]).shap_values(X_shap_test)
            all_shap_vals[_name] = _sv
            print(f'    {_name} ✓ (TreeExplainer)')
        except Exception as e:
            print(f'    {_name} SHAP failed: {e}')

# Linear models (Ridge, Lasso, LinearSVR)
for _name in ['Ridge', 'Lasso', 'SVR_lin']:
    if _name in sk_fitted_models:
        try:
            _sv = shap.LinearExplainer(sk_fitted_models[_name], X_shap_bg).shap_values(X_shap_test)
            all_shap_vals[_name] = _sv
            print(f'    {_name} ✓ (LinearExplainer)')
        except Exception as e:
            print(f'    {_name} SHAP failed: {e}')

# SVR_rbf + MLP (PermutationExplainer)
for _name in ['SVR_rbf', 'MLP']:
    if _name in sk_fitted_models:
        try:
            _exp = shap.PermutationExplainer(sk_fitted_models[_name].predict, X_shap_bg)
            _sv  = _exp(X_shap_test).values
            all_shap_vals[_name] = _sv
            print(f'    {_name} ✓ (PermutationExplainer)')
        except Exception as e:
            print(f'    {_name} SHAP failed: {e}')

# Keras models: PermutationExplainer with 2D wrapper (context tail prepended)
def _keras_2d_predictor(model, context_tail, ts):
    """Accept 2D (n, n_feat) input; prepend fixed context to form sequences."""
    def predict(X_2d):
        if hasattr(X_2d, 'values'):
            X_2d = X_2d.values
        if context_tail.shape[0] == 0:
            seqs = X_2d.reshape(-1, ts, X_2d.shape[1])
        else:
            seqs = np.array([np.vstack([context_tail, row.reshape(1, -1)])
                             for row in X_2d])
        return model.predict(seqs, verbose=0).flatten()
    return predict

for _name, _model, _ts in [('LSTM',   _lstm2,   LSTM_BEST_TS),
                             ('GRU',    _gru2,    GRU_BEST_TS),
                             ('BiLSTM', _bilstm2, BILSTM_BEST_TS)]:
    try:
        _ctx    = _Xtv[-(_ts - 1):] if _ts > 1 else np.empty((0, _Xtv.shape[1]))
        _pred_fn = _keras_2d_predictor(_model, _ctx, _ts)
        _exp    = shap.PermutationExplainer(_pred_fn, X_shap_bg)
        _sv     = _exp(X_shap_test[:50]).values   # 50 samples — speed
        all_shap_vals[_name] = _sv
        print(f'    {_name} ✓ (PermutationExplainer wrapper, n=50)')
    except Exception as e:
        print(f'    {_name} SHAP failed: {e}')

# Save all SHAP values for plot_all.py
np.savez(f'{RESULTS_DIR}/shap_all_models.npz',
         **{name: sv for name, sv in all_shap_vals.items()})
print(f'  All-model SHAP saved → {RESULTS_DIR}/shap_all_models.npz')
_elapsed('11. SHAP all models', t0)

# ══════════════════════════════════════════════════════════════════════════════
# 13. WRC + MCPT
# ══════════════════════════════════════════════════════════════════════════════
print('\n── 12. SELECTION BIAS TESTS ─────────────────────────────────────────────')
t0 = time.time()

def white_reality_check(sr, n_boot=1000, block=10, seed=42):
    rng = np.random.default_rng(seed)
    obs = sr.mean()
    n   = len(sr)
    nulls = []
    for _ in range(n_boot):
        blocks = []
        while sum(len(b) for b in blocks) < n:
            s = rng.integers(0, n)
            blocks.append(np.take(sr, range(s, s+block), mode='wrap'))
        nulls.append(np.concatenate(blocks)[:n].mean())
    nulls = np.array(nulls)
    return obs, nulls, float(np.mean(nulls >= obs))

def mc_permutation(sig, lr, n_perm=1000, seed=42):
    rng = np.random.default_rng(seed)
    obs = (sig * lr).mean()
    perms = [(rng.permutation(sig) * lr).mean() for _ in range(n_perm)]
    perms = np.array(perms)
    return obs, perms, float((np.sum(perms >= obs) + 1) / (n_perm + 1))

obs_wrc, null_wrc, wrc_p = white_reality_check(strat_ret)
obs_mc,  null_mc,  mc_p  = mc_permutation(signal, log_ret)

print(f'  WRC p={wrc_p:.4f}  {"✓ significant" if wrc_p < 0.05 else "✗ not significant"}')
print(f'  MC  p={mc_p:.4f}   {"✓ significant" if mc_p  < 0.05 else "✗ not significant"}')

bias_data = {'obs_wrc': float(obs_wrc), 'wrc_p': float(wrc_p),
             'obs_mc':  float(obs_mc),  'mc_p':  float(mc_p),
             'null_wrc': null_wrc.tolist(), 'null_mc': null_mc.tolist()}
with open(f'{RESULTS_DIR}/bias_test_data.json', 'w') as f:
    json.dump(bias_data, f)
print(f'  Bias test data saved → {RESULTS_DIR}/bias_test_data.json')
_elapsed('12. Bias tests', t0)

# ══════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
total = time.time() - _t_start
print(f'\n{"="*60}')
print(f'  FINAL SUMMARY')
print(f'{"="*60}')
print(f'  Best model:        {BEST_MODEL_NAME}')
print(f'  Features (k):      {best_k} of {len(FEATURE_COLS)}')
print(f'  FracDiff d:        {FRAC_D}')
print(f'  TIMESTEPS:         {TIMESTEPS}  (LSTM={LSTM_BEST_TS}, GRU={GRU_BEST_TS}, BiLSTM={BILSTM_BEST_TS})')
print(f'  Test MSE:          {test_mse:.8f}')
print(f'  CAGR:              {cagr_s:.2%}')
print(f'  Sharpe Ratio:      {sr_s:.4f}')
print(f'  Profit Factor:     {pf_s:.4f}')
print(f'  Max Drawdown:      {mdd_s:.2%}')
print(f'  WRC p-value:       {wrc_p:.4f}  {"✓" if wrc_p < 0.05 else "✗"}')
print(f'  MC  p-value:       {mc_p:.4f}   {"✓" if mc_p  < 0.05 else "✗"}')
print(f'{"="*60}')
print(f'  Total runtime:     {total:.0f}s')
print(f'\n  Cell timings:')
for label, t in sorted(_timings.items(), key=lambda x: -x[1]):
    print(f'    {label:<40} {t:>6.1f}s')

# Save final summary
summary_lines = [
    f'Best model:        {BEST_MODEL_NAME}',
    f'Features (k):      {best_k} of {len(FEATURE_COLS)}',
    f'FracDiff d:        {FRAC_D}',
    f'TIMESTEPS:         {TIMESTEPS}  (LSTM={LSTM_BEST_TS}, GRU={GRU_BEST_TS}, BiLSTM={BILSTM_BEST_TS})',
    f'Test MSE:          {test_mse:.8f}',
    f'CAGR:              {cagr_s:.2%}',
    f'Sharpe Ratio:      {sr_s:.4f}',
    f'Profit Factor:     {pf_s:.4f}',
    f'Max Drawdown:      {mdd_s:.2%}',
    f'WRC p-value:       {wrc_p:.4f}',
    f'MC  p-value:       {mc_p:.4f}',
    f'Total runtime:     {total:.0f}s',
]
with open(f'{RESULTS_DIR}/final_summary.txt', 'w') as f:
    f.write('\n'.join(summary_lines))

print(f'\n  Results → {RESULTS_DIR}/')
print(f'  Models  → {MODELS_DIR}/')

#!/usr/bin/env python3
"""
train_directional.py — DEPRECATED / DO NOT USE IN PRODUCTION

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FINDING (2026-05-02): This classification approach is INFERIOR to the MSE
pipeline in train.py. Do not use for trading signals or further development.

Results on test set (2022-06-22 → 2026-05-01):
  Best classifier  → RF_clf:  Sharpe 4.49,  DirAcc 62.2%
  MSE pipeline     → MLP:     Sharpe 9.01,  DirAcc 73.1%
  MSE pipeline     → SVR_lin: Sharpe 8.88,  DirAcc 71.8%

WHY THE MSE PIPELINE WINS:
  Predicting a continuous frac-diff value and thresholding via rolling z-score
  extracts substantially more signal than directly classifying binary direction.
  The z-score acts as a noise filter — it only generates a signal when the
  prediction deviates meaningfully from its recent mean, avoiding overtrading
  on marginal predictions. Direct binary classification treats every day as
  equally signal-worthy, which on a near-i.i.d. frac-diff series causes the
  classifier to overfit noise.

  Additionally, MSE regression preserves the magnitude of predictions, which
  the z-score uses implicitly. Classification collapses this to a 1-bit output,
  discarding confidence information.

This script is kept as a research artifact only.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Classification pipeline optimizing Signal Direction Accuracy (archived).

Unlike train.py (which minimizes MSE of frac-diff predictions and converts via z-score),
this pipeline directly predicts the *direction* of the next day's log return:
    y_dir = 1 if log_return > 0  (UP)
    y_dir = 0 if log_return <= 0 (DOWN)

Signal: +1 if predicted UP, -1 if predicted DOWN.
DirAcc = fraction of test days where signal matches actual return direction.

CV metric: balanced_accuracy_score (handles mild class imbalance in binary direction).

Outputs saved to:
    results/directional/   — charts and metrics CSV
    saved_models/directional/ — fitted models + metadata

Usage:
    python train_directional.py
"""

import os, sys, json, time, warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
import joblib
from joblib import Parallel, delayed
from itertools import product

# Sklearn — classifiers
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.pipeline import Pipeline
from statsmodels.tsa.stattools import adfuller

import xgboost as xgb

try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    try:
        import pandas_ta as ta
    except ImportError:
        ta = None

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Bidirectional, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import tensorflow as tf
tf.random.set_seed(42)

# ── Config ────────────────────────────────────────────────────────────────────
SEED          = 42
DATA_DIR      = 'data'
RESULTS_DIR   = 'results/directional'
MODELS_DIR    = 'saved_models/directional'
N_CV_FOLDS    = 5
N_JOBS        = -1
TIMESTEPS_CANDS = [5, 7, 10, 15]
np.random.seed(SEED)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR,  exist_ok=True)

_t_start = time.time()

# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA  (identical to train.py)
# ══════════════════════════════════════════════════════════════════════════════
print('\n── 1. LOAD DATA ─────────────────────────────────────────────────────────')
merged_path = f'{DATA_DIR}/merged_gold_dataset.csv'
if not os.path.exists(merged_path):
    sys.exit(f'ERROR: {merged_path} not found. Run download_data.py first.')

df = pd.read_csv(merged_path, index_col=0, parse_dates=True)
df.index = pd.to_datetime(df.index).normalize()
df.sort_index(inplace=True)
print(f'  Loaded {len(df)} rows  ({df.index[0].date()} → {df.index[-1].date()})')
HAS_COT = 'Comm_Positions_Long_All' in df.columns

# ══════════════════════════════════════════════════════════════════════════════
# 2. STATIONARITY + FRACTIONAL DIFFERENCING  (same as train.py)
# ══════════════════════════════════════════════════════════════════════════════
print('\n── 2. STATIONARITY ──────────────────────────────────────────────────────')

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

adf_test(df['log_Gold'], 'log(Gold)')

FRAC_D = None
for d in [0.3, 0.4, 0.5, 0.6, 0.7]:
    fd = frac_diff(df['log_Gold'], d=d)
    if adf_test(fd.dropna(), f'FracDiff(log_Gold, d={d})'):
        df['frac_diff_log_Gold'] = fd
        FRAC_D = d
        break
if FRAC_D is None:
    FRAC_D = 0.7
    df['frac_diff_log_Gold'] = frac_diff(df['log_Gold'], d=FRAC_D)

TARGET = 'frac_diff_log_Gold'
print(f'  → Frac-diff target: d={FRAC_D} (used only for lag features)')

if HAS_COT:
    for col in ['Comm_Positions_Long_All', 'Comm_Positions_Short_All',
                'NonComm_Positions_Long_All', 'NonComm_Positions_Short_All',
                'Open_Interest_All']:
        if col in df.columns and not adf_test(df[col].dropna(), col):
            df[f'{col}_diff'] = frac_diff(df[col], d=FRAC_D)

# ══════════════════════════════════════════════════════════════════════════════
# 3. FEATURE ENGINEERING  (same as train.py)
# ══════════════════════════════════════════════════════════════════════════════
print('\n── 3. FEATURE ENGINEERING ───────────────────────────────────────────────')

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

for lag in [1, 2, 3, 5, 10]:
    df[f'target_lag{lag}'] = df[TARGET].shift(lag)

df['log_return'] = df['log_Gold'].diff()
df.dropna(inplace=True)
print(f'  Feature matrix: {df.shape[0]} rows × {df.shape[1]} cols')

# ══════════════════════════════════════════════════════════════════════════════
# 4. TRAIN / TEST SPLIT
# ══════════════════════════════════════════════════════════════════════════════
print('\n── 4. TRAIN / TEST SPLIT ────────────────────────────────────────────────')

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
_lag_feats = [f'target_lag{l}' for l in [1, 2, 3, 5, 10]]

FEATURE_COLS = [c for c in (_cot_feats + _macro_feats + _ta_feats + _lag_feats)
                if c in df.columns]

X      = df[FEATURE_COLS].values
y_cont = df[TARGET].values          # continuous frac-diff (for lag features + Keras compat)
y_dir  = (df['log_return'].values > 0).astype(int)  # DIRECTIONAL TARGET: 1=UP, 0=DOWN
dates  = df.index
n      = len(X)
split  = int(n * 0.80)

X_trainval = X[:split]
y_tv_dir   = y_dir[:split]
y_tv_cont  = y_cont[:split]
X_test     = X[split:]
y_te_dir   = y_dir[split:]
dates_test = dates[split:]
log_ret_test = df['log_return'].values[split:]

up_pct = 100 * y_te_dir.mean()
print(f'  Train+Val: {split} rows  ({dates[0].date()} → {dates[split-1].date()})')
print(f'  Test:      {len(X_test)} rows  ({dates_test[0].date()} → {dates_test[-1].date()})')
print(f'  Features:  {len(FEATURE_COLS)}')
print(f'  Test UP%: {up_pct:.1f}%  (buy-and-hold dir acc baseline: {up_pct:.1f}%)')

# ══════════════════════════════════════════════════════════════════════════════
# 5. HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def trading_metrics_from_signal(signal, log_rets, name=''):
    """Compute Sharpe, CAGR, PF, MaxDD, DirAcc from a ±1 signal array."""
    signal = np.array(signal, dtype=float)
    log_rets = np.array(log_rets[:len(signal)])
    strat  = signal * log_rets
    eq     = np.exp(np.cumsum(strat))
    n_days = len(strat)
    cagr   = (eq[-1] ** (252 / n_days) - 1) * 100
    sharpe = (strat.mean() / (strat.std() + 1e-12)) * np.sqrt(252)
    wins   = strat[strat > 0].sum()
    losses = -strat[strat < 0].sum()
    pf     = wins / (losses + 1e-12)
    roll_max = np.maximum.accumulate(eq)
    max_dd = float(np.min((eq - roll_max) / (roll_max + 1e-12)))
    mkt_dir = np.sign(log_rets)
    dir_acc = float(np.mean(signal == mkt_dir))
    return dict(Sharpe=sharpe, CAGR=cagr, PF=pf, MaxDD=max_dd,
                DirAcc=dir_acc, equity=eq)

def make_sequences(X, y, ts):
    Xs, ys = [], []
    for i in range(ts, len(X)):
        Xs.append(X[i - ts:i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)

def build_lstm_clf(units=32, dropout=0.35, recurrent_dropout=0.2, n_features=1, timesteps=5):
    m = Sequential([
        LSTM(units, input_shape=(timesteps, n_features),
             dropout=dropout, recurrent_dropout=recurrent_dropout,
             kernel_regularizer=l2(0.001)),
        Dense(1, activation='sigmoid')
    ])
    m.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return m

def build_gru_clf(units=32, dropout=0.3, recurrent_dropout=0.2, n_features=1, timesteps=5):
    m = Sequential([
        GRU(units, input_shape=(timesteps, n_features),
            dropout=dropout, recurrent_dropout=recurrent_dropout),
        Dense(1, activation='sigmoid')
    ])
    m.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return m

def build_bilstm_clf(units=32, dropout=0.4, recurrent_dropout=0.2, n_features=1, timesteps=5):
    m = Sequential([
        Bidirectional(LSTM(units, recurrent_dropout=recurrent_dropout),
                      input_shape=(timesteps, n_features)),
        Dropout(dropout),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    m.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return m

# ══════════════════════════════════════════════════════════════════════════════
# 6. CV LOOP — CLASSIFICATION MODEL SELECTION
# ══════════════════════════════════════════════════════════════════════════════
print('\n── 5. CV LOOP — CLASSIFICATION MODELS ──────────────────────────────────')
t0 = time.time()

tscv = TimeSeriesSplit(n_splits=N_CV_FOLDS)

# Classification analogs of the regression models
CLF_MODELS = {
    'LogReg_L2':   LogisticRegression(C=1.0, max_iter=1000, random_state=SEED),
    'LogReg_L1':   LogisticRegression(C=1.0, l1_ratio=1.0, solver='saga',
                                       max_iter=2000, random_state=SEED),
    'SVC_lin':     SVC(kernel='linear', C=1.0, probability=True, random_state=SEED),
    'SVC_rbf':     SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=SEED),
    'RF_clf':      RandomForestClassifier(n_estimators=100, random_state=SEED),
    'XGB_clf':     xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05,
                                      random_state=SEED, verbosity=0, eval_metric='logloss'),
    'MLP_clf':     MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=SEED),
}

def _cv_clf_fold(model_name, model, X_tr, y_tr, X_va, y_va):
    sc = StandardScaler()
    model.fit(sc.fit_transform(X_tr), y_tr)
    preds = model.predict(sc.transform(X_va))
    return model_name, balanced_accuracy_score(y_va, preds)

print(f'  Running {len(CLF_MODELS)} classifiers × {N_CV_FOLDS} folds in parallel...')
fold_jobs = [
    delayed(_cv_clf_fold)(name, model, X_trainval[tr], y_tv_dir[tr],
                                       X_trainval[va], y_tv_dir[va])
    for name, model in CLF_MODELS.items()
    for tr, va in tscv.split(X_trainval)
]
cv_results_flat = Parallel(n_jobs=N_JOBS, backend='loky', verbose=0)(fold_jobs)

cv_scores = {}
for name in CLF_MODELS:
    scores = [s for (n, s) in cv_results_flat if n == name]
    cv_scores[name] = float(np.mean(scores))
    print(f'  {name:<22} CV BalAcc: {np.mean(scores):.4f} ± {np.std(scores):.4f}')

print(f'\n  Elapsed: {time.time()-t0:.1f}s')

# ══════════════════════════════════════════════════════════════════════════════
# 7. GRID SEARCH — TUNE EACH CLASSIFIER
# ══════════════════════════════════════════════════════════════════════════════
print('\n── 6. GRID SEARCH — TUNE CLASSIFIERS ───────────────────────────────────')
t0 = time.time()

PARAM_GRIDS = {
    'LogReg_L2': {'m__C': [0.001, 0.01, 0.1, 1, 10, 100]},
    'LogReg_L1': {'m__C': [0.001, 0.01, 0.1, 1, 10, 100]},
    'SVC_lin':   {'m__C': [0.01, 0.1, 1, 10, 100]},
    'SVC_rbf':   {'m__C': [0.1, 1, 10, 100], 'm__gamma': ['scale', 0.001, 0.01, 0.1]},
    'RF_clf':    {'m__n_estimators': [50, 100, 200],
                  'm__max_depth': [None, 5, 10],
                  'm__min_samples_split': [2, 5, 10]},
    'XGB_clf':   {'m__n_estimators': [50, 100, 200],
                  'm__max_depth': [2, 3, 4],
                  'm__learning_rate': [0.01, 0.05, 0.1],
                  'm__subsample': [0.8, 1.0]},
    'MLP_clf':   {'m__hidden_layer_sizes': [(64,32), (128,64,32), (64,32,16)],
                  'm__alpha': [0.0001, 0.001, 0.01],
                  'm__learning_rate_init': [0.001, 0.01]},
}

BASE_CLFS = {
    'LogReg_L2': LogisticRegression(max_iter=1000, random_state=SEED),
    'LogReg_L1': LogisticRegression(l1_ratio=1.0, solver='saga', max_iter=2000, random_state=SEED),
    'SVC_lin':   SVC(kernel='linear', probability=True, random_state=SEED),
    'SVC_rbf':   SVC(kernel='rbf', probability=True, random_state=SEED),
    'RF_clf':    RandomForestClassifier(random_state=SEED),
    'XGB_clf':   xgb.XGBClassifier(random_state=SEED, verbosity=0, eval_metric='logloss'),
    'MLP_clf':   MLPClassifier(max_iter=500, random_state=SEED),
}

tuned_clfs = {}
tuned_params = {}
for name in CLF_MODELS:
    pipe = Pipeline([('sc', StandardScaler()), ('m', BASE_CLFS[name])])
    gs = GridSearchCV(pipe, PARAM_GRIDS[name], cv=tscv, scoring='balanced_accuracy',
                      n_jobs=N_JOBS, refit=True)
    gs.fit(X_trainval, y_tv_dir)
    tuned_clfs[name]  = gs.best_estimator_
    tuned_params[name] = {k.replace('m__', ''): v
                          for k, v in gs.best_params_.items()}
    print(f'  {name:<22} best BalAcc(CV): {gs.best_score_:.4f}  params: {tuned_params[name]}')

print(f'  Elapsed: {time.time()-t0:.1f}s')

# ══════════════════════════════════════════════════════════════════════════════
# 8. KERAS CLASSIFIERS — full grid search (timesteps × units × dropout)
# ══════════════════════════════════════════════════════════════════════════════
print('\n── 7. KERAS CLASSIFIERS — FULL GRID SEARCH ──────────────────────────────')
t0 = time.time()

KERAS_GRIDS = {
    # Reduced grid — Keras classifiers topped out at CV BalAcc ~0.51-0.52 in full run;
    # narrowed to promising region to save ~90 min while still covering key axes.
    'LSTM_clf':    {'timesteps': [7, 10, 15], 'units': [32, 64],
                    'dropout': [0.2, 0.35],   'recurrent_dropout': [0.0, 0.2]},
    'GRU_clf':     {'timesteps': [5, 7, 15],  'units': [32, 64],
                    'dropout': [0.2, 0.3],    'recurrent_dropout': [0.0, 0.2]},
    'BiLSTM_clf':  {'timesteps': [10, 15],    'units': [16, 32],
                    'dropout': [0.4, 0.5],    'recurrent_dropout': [0.0, 0.2]},
}

keras_best = {}

for model_name, build_fn in [('LSTM_clf', build_lstm_clf),
                              ('GRU_clf',  build_gru_clf),
                              ('BiLSTM_clf', build_bilstm_clf)]:
    grid  = KERAS_GRIDS[model_name]
    combos = list(product(grid['timesteps'], grid['units'],
                          grid['dropout'], grid['recurrent_dropout']))
    print(f'\n  {model_name} — {len(combos)} combos × {N_CV_FOLDS} folds...')
    best_acc, best_combo = -np.inf, None

    for ts, units, drop, rdrop in combos:
        fold_accs = []
        for tr, va in tscv.split(X_trainval):
            sc = StandardScaler()
            X_tr_s = sc.fit_transform(X_trainval[tr])
            X_va_s = sc.transform(X_trainval[va])
            Xtr_sq, ytr_sq = make_sequences(X_tr_s, y_tv_dir[tr], ts)
            Xva_sq, yva_sq = make_sequences(X_va_s, y_tv_dir[va], ts)
            if len(Xva_sq) == 0:
                continue
            m = build_fn(units=units, dropout=drop, recurrent_dropout=rdrop,
                         n_features=X_trainval.shape[1], timesteps=ts)
            m.fit(Xtr_sq, ytr_sq, epochs=50, batch_size=32, verbose=0,
                  callbacks=[EarlyStopping(patience=10, restore_best_weights=True)])
            preds = (m.predict(Xva_sq, verbose=0).ravel() > 0.5).astype(int)
            fold_accs.append(balanced_accuracy_score(yva_sq, preds))
        mean_acc = float(np.mean(fold_accs)) if fold_accs else 0.0
        if mean_acc > best_acc:
            best_acc, best_combo = mean_acc, (ts, units, drop, rdrop)

    ts, units, drop, rdrop = best_combo
    keras_best[model_name] = {
        'timesteps': ts, 'units': units, 'dropout': drop,
        'recurrent_dropout': rdrop, 'cv_balacc': best_acc,
    }
    print(f'  → Best: ts={ts}, units={units}, drop={drop}, rdrop={rdrop}  '
          f'CV BalAcc={best_acc:.4f}')

print(f'\n  Elapsed: {time.time()-t0:.1f}s')

# ══════════════════════════════════════════════════════════════════════════════
# 9. FIT FINAL MODELS ON FULL TRAINVAL + EVALUATE ON TEST
# ══════════════════════════════════════════════════════════════════════════════
print('\n── 8. FINAL FIT + TEST EVALUATION ───────────────────────────────────────')

sc_final = StandardScaler()
sc_final.fit(X_trainval)
X_tv_s = sc_final.transform(X_trainval)
X_te_s = sc_final.transform(X_test)

all_metrics   = {}
all_signals   = {}
all_probs     = {}   # probability of UP for each model

# ── sklearn classifiers ───────────────────────────────────────────────────────
for name, clf in tuned_clfs.items():
    clf.fit(X_trainval, y_tv_dir)          # re-fit (GridSearchCV already fit, but refit=True)
    pred_class = clf.predict(X_test)
    if hasattr(clf, 'predict_proba'):
        prob_up = clf.predict_proba(X_test)[:, 1]
    else:
        # RidgeClassifier, LinearSVC don't have predict_proba
        prob_up = clf.decision_function(X_test)
        prob_up = (prob_up - prob_up.min()) / (prob_up.max() - prob_up.min() + 1e-12)

    signal = np.where(pred_class == 1, 1, -1).astype(float)
    m      = trading_metrics_from_signal(signal, log_ret_test)
    all_metrics[name]  = m
    all_signals[name]  = signal
    all_probs[name]    = prob_up
    print(f'  {name:<22} Sharpe={m["Sharpe"]:6.3f}  CAGR={m["CAGR"]:7.1f}%  '
          f'DirAcc={m["DirAcc"]*100:.1f}%  MaxDD={m["MaxDD"]*100:.1f}%')

# ── Keras classifiers ─────────────────────────────────────────────────────────
print()
keras_fitted = {}
for model_name, build_fn in [('LSTM_clf', build_lstm_clf),
                              ('GRU_clf',  build_gru_clf),
                              ('BiLSTM_clf', build_bilstm_clf)]:
    kp   = keras_best[model_name]
    ts, units = kp['timesteps'], kp['units']
    drop, rdrop = kp['dropout'], kp['recurrent_dropout']

    Xtv_sq, ytv_sq = make_sequences(X_tv_s, y_tv_dir, ts)
    Xte_sq, yte_sq = make_sequences(X_te_s, y_te_dir, ts)
    log_ret_keras  = log_ret_test[ts:]

    m = build_fn(units=units, dropout=drop, recurrent_dropout=rdrop,
                 n_features=X_trainval.shape[1], timesteps=ts)
    m.fit(Xtv_sq, ytv_sq, epochs=50, batch_size=32, verbose=0,
          callbacks=[EarlyStopping(patience=10, restore_best_weights=True)],
          validation_split=0.1)

    prob_up   = m.predict(Xte_sq, verbose=0).ravel()
    pred_class = (prob_up > 0.5).astype(int)
    signal     = np.where(pred_class == 1, 1, -1).astype(float)
    met        = trading_metrics_from_signal(signal, log_ret_keras)

    all_metrics[model_name]  = met
    all_signals[model_name]  = signal
    all_probs[model_name]    = prob_up
    keras_best[model_name]['model'] = m
    keras_fitted[model_name] = m

    print(f'  {model_name:<22} Sharpe={met["Sharpe"]:6.3f}  CAGR={met["CAGR"]:7.1f}%  '
          f'DirAcc={met["DirAcc"]*100:.1f}%  MaxDD={met["MaxDD"]*100:.1f}%')

# Buy & Hold baseline
bnh_equity = np.exp(np.cumsum(log_ret_test))
bnh_n = len(log_ret_test)
bnh_cagr = (bnh_equity[-1] ** (252 / bnh_n) - 1) * 100
bnh_sharpe = (log_ret_test.mean() / (log_ret_test.std() + 1e-12)) * np.sqrt(252)
bnh_dirac = float(np.mean(np.sign(log_ret_test) == np.sign(log_ret_test)))  # trivially 1.0 vs itself
bnh_up_acc = float(np.mean(log_ret_test > 0))   # actual % of up days = the meaningful baseline
print(f'\n  Buy & Hold baseline:  Sharpe={bnh_sharpe:.3f}  CAGR={bnh_cagr:.1f}%  '
      f'Up-day %={bnh_up_acc*100:.1f}%')

# ══════════════════════════════════════════════════════════════════════════════
# 10. SAVE METRICS + MODELS
# ══════════════════════════════════════════════════════════════════════════════
print('\n── 9. SAVE ──────────────────────────────────────────────────────────────')

def _json_safe(obj):
    """Convert numpy scalars to native Python types for JSON serialization."""
    if isinstance(obj, (np.integer,)):  return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, np.ndarray):     return obj.tolist()
    raise TypeError(f'Not serializable: {type(obj)}')

rows = []
for name, m in all_metrics.items():
    p = {k: v for k, v in (tuned_params.get(name) or keras_best.get(name, {})).items()
         if k != 'model'}
    rows.append({
        'Model': name,
        'Sharpe': round(m['Sharpe'], 4),
        'CAGR':   round(m['CAGR'], 2),
        'DirAcc': round(m['DirAcc'], 4),
        'PF':     round(m['PF'], 4),
        'MaxDD':  round(m['MaxDD'], 4),
        'Params': json.dumps(p, default=_json_safe),
    })
metrics_df = pd.DataFrame(rows).sort_values('Sharpe', ascending=False)
metrics_df.to_csv(f'{RESULTS_DIR}/directional_model_metrics.csv', index=False)
print(f'  Saved → {RESULTS_DIR}/directional_model_metrics.csv')

# Save signals + predictions
sig_df = pd.DataFrame({'date': dates_test}, index=range(len(dates_test)))
for name, sig in all_signals.items():
    n_ = min(len(sig), len(dates_test))
    col = pd.Series([np.nan]*len(dates_test))
    col.iloc[:n_] = sig[:n_]
    sig_df[f'signal_{name}'] = col.values
sig_df.to_csv(f'{RESULTS_DIR}/directional_signals.csv', index=False)

# Save metadata
meta = {
    'pipeline':   'directional_classification',
    'target':     'log_return_direction (1=UP, 0=DOWN)',
    'cv_metric':  'balanced_accuracy',
    'features':   FEATURE_COLS,
    'frac_d':     FRAC_D,
    'split_date': str(dates[split].date()),
    'test_up_pct': float(bnh_up_acc),
    'tuned_params': tuned_params,
    'keras_best':   {k: {kk: vv for kk, vv in v.items() if kk != 'model'}
                     for k, v in keras_best.items()},
}
with open(f'{MODELS_DIR}/directional_metadata.json', 'w') as f:
    json.dump(meta, f, indent=2)

# Save sklearn models
for name, clf in tuned_clfs.items():
    joblib.dump(clf, f'{MODELS_DIR}/clf_{name}.joblib')
for name, m in keras_fitted.items():
    ts = keras_best[name]['timesteps']
    m.save(f'{MODELS_DIR}/clf_{name}_ts{ts}.keras')
joblib.dump(sc_final, f'{MODELS_DIR}/directional_scaler.joblib')
print('  Models saved.')

# ══════════════════════════════════════════════════════════════════════════════
# 11. CHARTS
# ══════════════════════════════════════════════════════════════════════════════
print('\n── 10. CHARTS ────────────────────────────────────────────────────────────')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

STRONG_COLOR = {'LogReg_L2': '#DAA520', 'LogReg_L1': '#228B22',
                'SVC_lin':   '#1E90FF', 'SVC_rbf':   '#9932CC',
                'RF_clf':    '#FF8C00', 'XGB_clf':   '#DC143C',
                'MLP_clf':   '#00CED1'}
KERAS_COLOR  = {'LSTM_clf': '#888888', 'GRU_clf': '#AAAAAA', 'BiLSTM_clf': '#CCCCCC'}
SHARPE_THRESH = 7.5

# ── Chart 1: DirAcc bar chart ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))
names   = metrics_df['Model'].tolist()
daccs   = metrics_df['DirAcc'].tolist()
colors  = [STRONG_COLOR.get(n, KERAS_COLOR.get(n, '#888888')) for n in names]
bars    = ax.bar(names, [d*100 for d in daccs], color=colors, edgecolor='black', linewidth=0.5)
ax.axhline(bnh_up_acc * 100, color='gray', linestyle='--', lw=1.5, label=f'Up-day % baseline ({bnh_up_acc*100:.1f}%)')
ax.axhline(50, color='black', linestyle=':', lw=1, label='50% (coin flip)')
for bar, d in zip(bars, daccs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f'{d*100:.1f}%', ha='center', va='bottom', fontsize=8)
ax.set_ylabel('Signal Direction Accuracy (%)')
ax.set_title('Directional Pipeline — Signal Direction Accuracy (test set)')
ax.legend()
plt.xticks(rotation=25, ha='right')
plt.tight_layout()
plt.savefig(f'{RESULTS_DIR}/directional_dirac_bar.png', dpi=150)
plt.close()
print(f'  Saved → {RESULTS_DIR}/directional_dirac_bar.png')

# ── Chart 2: Equity curves ────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=False)

# Panel 1: Gold price
ax = axes[0]
gold_test = df['Gold'].values[split:]
ax.plot(dates_test, gold_test, color='goldenrod', lw=1.5)
ax.set_ylabel('Gold Price (USD)')
ax.set_title('Gold Price — Test Period')
ax.grid(alpha=0.3)

# Panel 2: Equity curves — strong models
ax = axes[1]
bnh_eq = np.exp(np.cumsum(log_ret_test))
ax.plot(dates_test, bnh_eq, color='gray', lw=1.5, ls='--', label=f'Buy & Hold (Sharpe {bnh_sharpe:.2f})')
for name in metrics_df['Model']:
    m = all_metrics[name]
    sig = all_signals[name]
    n_ = min(len(sig), len(dates_test))
    eq = m['equity']
    d_plot = dates_test[:n_]
    color = STRONG_COLOR.get(name, KERAS_COLOR.get(name, '#AAAAAA'))
    lw = 1.8 if m['Sharpe'] >= SHARPE_THRESH else 0.8
    ls = '-' if m['Sharpe'] >= SHARPE_THRESH else '--'
    alpha = 1.0 if m['Sharpe'] >= SHARPE_THRESH else 0.5
    label = f'{name} (Sharpe {m["Sharpe"]:.2f}, DirAcc {m["DirAcc"]*100:.1f}%)'
    ax.plot(d_plot, eq, color=color, lw=lw, ls=ls, alpha=alpha, label=label)
ax.set_ylabel('Portfolio Value (×)')
ax.set_title('Directional Pipeline — Equity Curves (★ = Sharpe ≥ 7.5)')
ax.legend(fontsize=7, ncol=2)
ax.grid(alpha=0.3)

# Panel 3: DirAcc vs MSE pipeline comparison
ax = axes[2]
# Load MSE pipeline metrics for comparison
mse_csv = 'results/all_model_metrics.csv'
model_map = {   # directional name → MSE pipeline name
    'LogReg_L2': 'Ridge', 'LogReg_L1': 'Lasso',
    'SVC_lin': 'SVR_lin', 'SVC_rbf': 'SVR_rbf',
    'RF_clf': 'RandomForest', 'XGB_clf': 'XGBoost', 'MLP_clf': 'MLP',
    'LSTM_clf': 'LSTM', 'GRU_clf': 'GRU', 'BiLSTM_clf': 'BiLSTM',
}
dir_accs_dir = {n: all_metrics[n]['DirAcc']*100 for n in all_metrics}
dir_accs_mse = {}
if os.path.exists(mse_csv):
    mse_df = pd.read_csv(mse_csv)
    if 'DirAcc' in mse_df.columns:
        for d_name, m_name in model_map.items():
            row = mse_df[mse_df['Model'] == m_name]
            if not row.empty:
                dir_accs_mse[d_name] = float(row['DirAcc'].values[0]) * 100

x    = np.arange(len(all_metrics))
w    = 0.35
keys = list(all_metrics.keys())
ax.bar(x - w/2, [dir_accs_dir.get(k, 0) for k in keys], w,
       label='Directional pipeline', color='steelblue', alpha=0.8, edgecolor='black', lw=0.5)
if dir_accs_mse:
    ax.bar(x + w/2, [dir_accs_mse.get(k, 0) for k in keys], w,
           label='MSE pipeline (tuned)', color='coral', alpha=0.8, edgecolor='black', lw=0.5)
ax.axhline(bnh_up_acc * 100, color='gray', ls='--', lw=1.2, label=f'Up-day % ({bnh_up_acc*100:.1f}%)')
ax.set_xticks(x)
ax.set_xticklabels(keys, rotation=25, ha='right')
ax.set_ylabel('Signal Direction Accuracy (%)')
ax.set_title('DirAcc Comparison: Directional vs MSE Pipeline')
ax.legend(fontsize=8)
ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(f'{RESULTS_DIR}/directional_equity_curves.png', dpi=150)
plt.close()
print(f'  Saved → {RESULTS_DIR}/directional_equity_curves.png')

# ══════════════════════════════════════════════════════════════════════════════
# 12. SUMMARY TABLE
# ══════════════════════════════════════════════════════════════════════════════
total_time = time.time() - _t_start
print(f'\n{"="*100}')
print(f'  {"Model":<22} {"Sharpe":>8} {"CAGR":>8} {"DirAcc":>8} {"MaxDD":>8}  Optimized Parameters')
print(f'  {"-"*98}')
for _, row in metrics_df.iterrows():
    star = ' ★' if row['Sharpe'] >= SHARPE_THRESH else '  '
    print(f'  {row["Model"]:<22}{star} {row["Sharpe"]:>7.3f} {row["CAGR"]:>7.1f}% '
          f'{row["DirAcc"]*100:>7.1f}% {row["MaxDD"]*100:>7.1f}%  {row["Params"]}')
print(f'  {"-"*98}')
print(f'  {"Buy & Hold (up-day%)":<24} {bnh_sharpe:>7.3f} {bnh_cagr:>7.1f}% '
      f'{bnh_up_acc*100:>7.1f}%          (baseline)')
print(f'{"="*100}')
print(f'\n  Total runtime: {total_time/60:.1f} min')
print(f'  Results → {RESULTS_DIR}/')
print(f'  Models  → {MODELS_DIR}/')

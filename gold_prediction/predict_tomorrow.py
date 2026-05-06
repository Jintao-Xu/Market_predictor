#!/usr/bin/env python3
"""
predict_tomorrow.py — Generate tomorrow's gold trading signal from all Exp 11 models.

Mirrors the exact training / feature-engineering / signal logic from update_results.py
(dual scaler setup: RECENT_MODELS on 3yr window, FULL_MODELS on full history).

Usage:
    python predict_tomorrow.py
"""

import os, json, warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR    = 'data'
MODELS_DIR  = 'saved_models'
RESULTS_DIR = 'results'
os.makedirs(RESULTS_DIR, exist_ok=True)
SEED       = 42
RECENT_YEARS = 3
RECENT_MODELS = {'Ridge', 'Lasso', 'SVR_lin', 'SVR_rbf', 'XGBoost', 'ElasticNet'}
FULL_MODELS   = {'LightGBM', 'RandomForest', 'MLP', 'LSTM', 'GRU', 'BiLSTM'}
DEFAULT_ZSCORE_WIN  = 10
DEFAULT_THRESHOLD   = 0.0

np.random.seed(SEED)

# ── Load metadata ─────────────────────────────────────────────────────────────
with open(f'{MODELS_DIR}/model_metadata.json') as f:
    meta = json.load(f)

FRAC_D         = meta['frac_d']
FEATURE_COLS   = meta['feature_cols']
LSTM_TS        = meta['lstm_best_timesteps']
GRU_TS         = meta['gru_best_timesteps']
BILSTM_TS      = meta['bilstm_best_timesteps']
TRAIN_END_DATE = pd.Timestamp(meta['train_end_date'])

def load_tuned(name):
    path = f'{MODELS_DIR}/tuned_{name}.json'
    if os.path.exists(path):
        return json.load(open(path)).get('best_params', {})
    return {}

TUNED = {m: load_tuned(m) for m in list(RECENT_MODELS) + list(FULL_MODELS)}
ZSCORE_WINS = {m: int(TUNED[m].get('zscore_win', DEFAULT_ZSCORE_WIN)) for m in TUNED}
THRESHOLDS  = {m: float(TUNED[m].get('signal_threshold', DEFAULT_THRESHOLD)) for m in TUNED}

# ══════════════════════════════════════════════════════════════════════════════
# 0. AUTO-REFRESH DATA  (incremental — only fetches missing rows)
# ══════════════════════════════════════════════════════════════════════════════
import subprocess, sys
_dl_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'download_data.py')
if os.path.exists(_dl_script):
    print('Refreshing data (download_data.py)...')
    _result = subprocess.run([sys.executable, _dl_script], capture_output=True, text=True,
                             cwd=os.path.dirname(os.path.abspath(__file__)))
    if _result.returncode != 0:
        print(f'  WARNING: download_data.py failed:\n{_result.stderr[-500:]}')
    else:
        for _line in _result.stdout.splitlines():
            if any(kw in _line for kw in ('new row', 'latest:', 'COMPLETE', 'WARNING', 'ERROR')):
                print(f'  {_line.strip()}')
    print()

# ══════════════════════════════════════════════════════════════════════════════
# 1. FEATURE ENGINEERING  (identical to update_results.py)
# ══════════════════════════════════════════════════════════════════════════════
print('Loading data and engineering features...')

df = pd.read_csv(f'{DATA_DIR}/merged_gold_dataset.csv', index_col=0, parse_dates=True)
df.index = pd.to_datetime(df.index).normalize()
df.sort_index(inplace=True)
HAS_COT = 'Comm_Positions_Long_All' in df.columns

def frac_diff(series, d, window=252, threshold=1e-5):
    w = [1.0]
    for k in range(1, window):
        w_k = -w[-1] * (d - k + 1) / k
        if abs(w_k) < threshold:
            break
        w.append(w_k)
    w = np.array(w[::-1])
    n, width = len(series), len(w)
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

df['RSI_14'] = df['Gold'].diff().clip(lower=0).rolling(14).mean() / df['Gold'].diff().abs().rolling(14).mean()
df['RSI_28'] = df['Gold'].diff().clip(lower=0).rolling(28).mean() / df['Gold'].diff().abs().rolling(28).mean()
df['ROC_10'] = df['Gold'].pct_change(10) * 100
df['ROC_20'] = df['Gold'].pct_change(20) * 100
df['MOM_5']  = df['Gold'].diff(5)
df['EMA_12'] = df['Gold'].ewm(span=12).mean()
df['EMA_26'] = df['Gold'].ewm(span=26).mean()
df['MACD_diff'] = df['EMA_12'] - df['EMA_26']
bb_mid = df['Gold'].rolling(20).mean(); bb_std = df['Gold'].rolling(20).std()
df['BB_width'] = bb_std / bb_mid
df['BB_pct']   = (df['Gold'] - (bb_mid - 2*bb_std)) / (4 * bb_std + 1e-12)
for lag in [1, 2, 3, 5, 10]:
    df[f'target_lag{lag}'] = df['frac_diff_log_Gold'].shift(lag)
df['log_return'] = df['log_Gold'].diff()
for period in [21, 63, 126, 252]:
    df[f'gold_ret_{period}d'] = df['log_Gold'].diff(period)
for m_num in range(1, 13):
    df[f'month_{m_num}'] = (df.index.month == m_num).astype(int)
if HAS_COT:
    df['net_spec_chg_20d'] = df['net_speculator'].diff(20)
    df['net_comm_chg_20d'] = df['net_commercial'].diff(20)

df.dropna(inplace=True)
feat_cols = [c for c in FEATURE_COLS if c in df.columns]

latest_date   = df.index[-1]
latest_gold   = df['Gold'].iloc[-1]
latest_return = df['log_return'].iloc[-1]

print(f'  Latest data:  {latest_date.date()}  Gold=${latest_gold:,.2f}  ret={latest_return:+.4f}')

# Next trading day
from datetime import datetime, timedelta
next_day = latest_date + timedelta(days=1)
while next_day.weekday() >= 5:
    next_day += timedelta(days=1)

# ══════════════════════════════════════════════════════════════════════════════
# 2. BUILD DUAL SCALERS  (mirror update_results.py exactly)
# ══════════════════════════════════════════════════════════════════════════════
print('Building scalers...')

# Full-history scaler
_X_full  = df[feat_cols].values
_y_full  = df['frac_diff_log_Gold'].values
_n_full  = len(_X_full)
_spl_full = int(_n_full * 0.80)
X_tv_full = _X_full[:_spl_full]
y_tv_full = _y_full[:_spl_full]
SVR_EPS_FULL = round(0.25 * float(np.std(y_tv_full)), 6)
sc_full = StandardScaler()
sc_full.fit(X_tv_full)

# Recent-3yr scaler
_rec_start = df.index[-1] - pd.DateOffset(years=RECENT_YEARS)
df_rec  = df[df.index >= _rec_start].copy()
_X_rec  = df_rec[feat_cols].values
_y_rec  = df_rec['frac_diff_log_Gold'].values
_n_rec  = len(_X_rec)
_spl_rec = int(_n_rec * 0.80)
X_tv_rec = _X_rec[:_spl_rec]
y_tv_rec = _y_rec[:_spl_rec]
SVR_EPS_REC = round(0.25 * float(np.std(y_tv_rec)), 6)
sc_rec = StandardScaler()
sc_rec.fit(X_tv_rec)

print(f'  Full-history train: {_spl_full} rows,  SVR eps={SVR_EPS_FULL}')
print(f'  Recent-3yr  train:  {_spl_rec} rows,  SVR eps={SVR_EPS_REC}')

# ══════════════════════════════════════════════════════════════════════════════
# 3. FIT SKLEARN MODELS + LOAD KERAS MODELS
# ══════════════════════════════════════════════════════════════════════════════
print('\nFitting sklearn models and loading Keras weights...')

def zscore_signal(preds_arr, zscore_win, threshold):
    """Rolling z-score signal. Returns (z_value, signal) for last prediction."""
    ps   = pd.Series(np.asarray(preds_arr, dtype=float))
    mean = ps.rolling(zscore_win, min_periods=1).mean()
    std  = ps.rolling(zscore_win, min_periods=1).std().fillna(1e-8)
    z    = (ps - mean) / std
    z_val = float(z.iloc[-1])
    if np.isnan(z_val):
        z_val = 0.0
    if   z_val >  threshold:
        sig = 1
    elif z_val < -threshold:
        sig = -1
    else:
        sig = 0
    return z_val, sig

def make_sequences(X, ts):
    return np.array([X[i - ts:i] for i in range(ts, len(X))])

# ── How many rows we need for prediction window ───────────────────────────────
# Need enough rows to: (a) compute zscore over zscore_win steps, (b) fill RNN sequences
max_zwin   = max(ZSCORE_WINS.values())
max_ts     = max(LSTM_TS, GRU_TS, BILSTM_TS)
# We predict on the last N rows so the rolling z-score has sufficient context
N_WINDOW   = max(max_zwin, max_ts) + 30   # generous buffer

# ── Recent-model prediction window ────────────────────────────────────────────
# Use df_rec (recent 3yr) to stay consistent with training distribution
X_rec_all = df_rec[feat_cols].values
X_rec_win = X_rec_all[-N_WINDOW:]
X_rec_win_s = sc_rec.transform(X_rec_win)

# ── Full-model prediction window ──────────────────────────────────────────────
X_full_all = df[feat_cols].values
X_full_win = X_full_all[-N_WINDOW:]
X_full_win_s = sc_full.transform(X_full_win)

# ── Fit each sklearn model on its appropriate training window, predict on window ─
signals   = {}
raw_preds = {}
z_scores  = {}

P = TUNED   # shorthand

# Ridge
t = P['Ridge']
m = Ridge(alpha=t.get('alpha', 0.01))
m.fit(sc_rec.transform(X_tv_rec), y_tv_rec)
preds = m.predict(X_rec_win_s)
z, sig = zscore_signal(preds, ZSCORE_WINS['Ridge'], THRESHOLDS['Ridge'])
signals['Ridge'] = sig; raw_preds['Ridge'] = float(preds[-1]); z_scores['Ridge'] = z
print(f'  Ridge fitted ✓  z={z:+.3f}  sig={sig:+d}')

# Lasso
t = P['Lasso']
m = Lasso(alpha=t.get('alpha', 0.001), max_iter=5000)
m.fit(sc_rec.transform(X_tv_rec), y_tv_rec)
preds = m.predict(X_rec_win_s)
z, sig = zscore_signal(preds, ZSCORE_WINS['Lasso'], THRESHOLDS['Lasso'])
signals['Lasso'] = sig; raw_preds['Lasso'] = float(preds[-1]); z_scores['Lasso'] = z
print(f'  Lasso fitted ✓  z={z:+.3f}  sig={sig:+d}')

# ElasticNet
t = P['ElasticNet']
m = ElasticNet(alpha=t.get('alpha', 0.01), l1_ratio=t.get('l1_ratio', 0.5), max_iter=5000)
m.fit(sc_rec.transform(X_tv_rec), y_tv_rec)
preds = m.predict(X_rec_win_s)
z, sig = zscore_signal(preds, ZSCORE_WINS['ElasticNet'], THRESHOLDS['ElasticNet'])
signals['ElasticNet'] = sig; raw_preds['ElasticNet'] = float(preds[-1]); z_scores['ElasticNet'] = z
print(f'  ElasticNet fitted ✓  z={z:+.3f}  sig={sig:+d}')

# SVR_lin
t = P['SVR_lin']
m = SVR(kernel='linear', C=t.get('C', 100), epsilon=SVR_EPS_REC)
m.fit(sc_rec.transform(X_tv_rec), y_tv_rec)
preds = m.predict(X_rec_win_s)
z, sig = zscore_signal(preds, ZSCORE_WINS['SVR_lin'], THRESHOLDS['SVR_lin'])
signals['SVR_lin'] = sig; raw_preds['SVR_lin'] = float(preds[-1]); z_scores['SVR_lin'] = z
print(f'  SVR_lin fitted ✓  z={z:+.3f}  sig={sig:+d}')

# SVR_rbf
t = P['SVR_rbf']
gamma = t.get('gamma', 0.01)
m = SVR(kernel='rbf', C=t.get('C', 10), gamma=gamma, epsilon=SVR_EPS_REC)
m.fit(sc_rec.transform(X_tv_rec), y_tv_rec)
preds = m.predict(X_rec_win_s)
z, sig = zscore_signal(preds, ZSCORE_WINS['SVR_rbf'], THRESHOLDS['SVR_rbf'])
signals['SVR_rbf'] = sig; raw_preds['SVR_rbf'] = float(preds[-1]); z_scores['SVR_rbf'] = z
print(f'  SVR_rbf fitted ✓  z={z:+.3f}  sig={sig:+d}')

# XGBoost
t = P['XGBoost']
m = xgb.XGBRegressor(
    n_estimators=t.get('n_estimators', 300),
    max_depth=t.get('max_depth', 4),
    learning_rate=t.get('learning_rate', 0.05),
    subsample=t.get('subsample', 0.8),
    random_state=SEED, verbosity=0)
m.fit(sc_rec.transform(X_tv_rec), y_tv_rec)
preds = m.predict(X_rec_win_s)
z, sig = zscore_signal(preds, ZSCORE_WINS['XGBoost'], THRESHOLDS['XGBoost'])
signals['XGBoost'] = sig; raw_preds['XGBoost'] = float(preds[-1]); z_scores['XGBoost'] = z
print(f'  XGBoost fitted ✓  z={z:+.3f}  sig={sig:+d}')

# RandomForest
t = P['RandomForest']
m = RandomForestRegressor(
    n_estimators=t.get('n_estimators', 100),
    max_depth=t.get('max_depth', None),
    min_samples_split=t.get('min_samples_split', 2),
    min_samples_leaf=t.get('min_samples_leaf', 1),
    random_state=SEED)
m.fit(sc_full.transform(X_tv_full), y_tv_full)
preds = m.predict(X_full_win_s)
z, sig = zscore_signal(preds, ZSCORE_WINS['RandomForest'], THRESHOLDS['RandomForest'])
signals['RandomForest'] = sig; raw_preds['RandomForest'] = float(preds[-1]); z_scores['RandomForest'] = z
print(f'  RandomForest fitted ✓  z={z:+.3f}  sig={sig:+d}')

# MLP
t = P['MLP']
hl = t.get('hidden_layer_sizes', (64, 32))
if isinstance(hl, list):
    hl = tuple(hl)
m = MLPRegressor(hidden_layer_sizes=hl, alpha=t.get('alpha', 0.001),
                 learning_rate_init=t.get('learning_rate_init', 0.001),
                 max_iter=1000, random_state=SEED)
m.fit(sc_full.transform(X_tv_full), y_tv_full)
preds = m.predict(X_full_win_s)
z, sig = zscore_signal(preds, ZSCORE_WINS['MLP'], THRESHOLDS['MLP'])
signals['MLP'] = sig; raw_preds['MLP'] = float(preds[-1]); z_scores['MLP'] = z
print(f'  MLP fitted ✓  z={z:+.3f}  sig={sig:+d}')

# LightGBM
if LIGHTGBM_AVAILABLE:
    t = P['LightGBM']
    m = lgb.LGBMRegressor(
        n_estimators=t.get('n_estimators', 200),
        num_leaves=t.get('num_leaves', 31),
        learning_rate=t.get('learning_rate', 0.05),
        subsample=t.get('subsample', 0.8),
        random_state=SEED, verbose=-1)
    m.fit(sc_full.transform(X_tv_full), y_tv_full)
    preds = m.predict(X_full_win_s)
    z, sig = zscore_signal(preds, ZSCORE_WINS['LightGBM'], THRESHOLDS['LightGBM'])
    signals['LightGBM'] = sig; raw_preds['LightGBM'] = float(preds[-1]); z_scores['LightGBM'] = z
    print(f'  LightGBM fitted ✓  z={z:+.3f}  sig={sig:+d}')

# ── Load Keras models ─────────────────────────────────────────────────────────
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model

    def directional_mse(y_true, y_pred):
        mse      = tf.reduce_mean(tf.square(y_true - y_pred))
        sign_err = tf.maximum(0.0, -y_true * y_pred)
        return mse + 0.5 * tf.reduce_mean(sign_err)

    for name, ts in [('LSTM', LSTM_TS), ('GRU', GRU_TS), ('BiLSTM', BILSTM_TS)]:
        t  = TUNED[name]
        ts_tuned = int(t.get('timesteps', ts))
        path = f'{MODELS_DIR}/model_{name}.keras'
        if not os.path.exists(path):
            print(f'  {name}: model file not found, skipping')
            continue
        km = load_model(path, custom_objects={'directional_mse': directional_mse})
        # Use full-history window scaled with sc_full
        n_seq_needed = ts_tuned + ZSCORE_WINS[name] + 10
        X_win = X_full_all[-n_seq_needed:]
        X_win_s = sc_full.transform(X_win)
        seqs = make_sequences(X_win_s, ts_tuned)
        preds = km.predict(seqs, verbose=0).flatten()
        z, sig = zscore_signal(preds, ZSCORE_WINS[name], THRESHOLDS[name])
        signals[name]   = sig
        raw_preds[name] = float(preds[-1])
        z_scores[name]  = z
        print(f'  {name} loaded ✓  z={z:+.3f}  sig={sig:+d}')

except ImportError:
    print('  TensorFlow not available — skipping LSTM/GRU/BiLSTM')

# ══════════════════════════════════════════════════════════════════════════════
# 4. INVERT FRAC-DIFF PREDICTIONS → ACTUAL PRICE
# ══════════════════════════════════════════════════════════════════════════════

def fracdiff_weights(d, window=252, threshold=1e-5):
    """Return the frac_diff weight vector (original order, before reversal)."""
    w = [1.0]
    for k in range(1, window):
        w_k = -w[-1] * (d - k + 1) / k
        if abs(w_k) < threshold:
            break
        w.append(w_k)
    return np.array(w)   # w[0]=1.0 (current), w[1]=-d, w[2]=..., all decreasing

def fracdiff_to_price(pred_fracdiff, log_gold_history, d, window=252, threshold=1e-5):
    """
    Invert a frac_diff prediction to an actual gold price.

    frac_diff[t+1] = w[0]*log_Gold[t+1] + w[1]*log_Gold[t] + w[2]*log_Gold[t-1] + ...
    Since w[0] = 1.0:
    log_Gold[t+1] = pred - sum(w[k] * log_Gold[t+1-k]  for k=1,2,...)
    Gold[t+1]     = exp(log_Gold[t+1])
    """
    w = fracdiff_weights(d, window, threshold)
    # log_gold_history[-1] = log_Gold[t], [-2] = log_Gold[t-1], ...
    historical_sum = sum(w[k] * log_gold_history[-(k)] for k in range(1, min(len(w), len(log_gold_history) + 1)))
    log_price_next = pred_fracdiff - historical_sum
    return np.exp(log_price_next)

# Recent log_Gold history (used for all models — inversion is on the full series)
log_gold_hist = df['log_Gold'].values   # full history, most recent last

price_preds = {}
for name, pred_fd in raw_preds.items():
    price_preds[name] = fracdiff_to_price(pred_fd, log_gold_hist, d=FRAC_D)

# Sharpe-weighted ensemble price (positive-Sharpe models only)
# ══════════════════════════════════════════════════════════════════════════════
# 5. CONSENSUS + REPORT
# ══════════════════════════════════════════════════════════════════════════════

# ── Exp 11 metrics (from results/all_model_metrics.csv) ──────────────────────
# Sharpe → used for directional vote weighting
# MSE    → used for price target weighting (measures frac_diff prediction accuracy)
exp11_sharpes = {
    'SVR_lin':      10.60,
    'Ridge':        10.13,
    'Lasso':         9.98,
    'ElasticNet':    9.98,
    'SVR_rbf':       9.82,
    'XGBoost':       9.62,
    'LightGBM':      8.21,
    'RandomForest':  8.18,
    'MLP':           7.30,
    'LSTM':          0.37,
    'BiLSTM':        0.25,
    'GRU':          -0.13,
}

exp11_mse = {
    'Ridge':        0.000468,
    'Lasso':        0.000433,
    'SVR_lin':      0.000762,
    'SVR_rbf':      0.000778,
    'XGBoost':      0.002227,
    'RandomForest': 0.018253,
    'MLP':          0.002177,
    'ElasticNet':   0.000434,
    'LightGBM':     0.018528,
    'LSTM':         0.005753,
    'GRU':          0.026017,
    'BiLSTM':       0.019594,
}

# Only count models we actually have predictions for
available = {n: s for n, s in exp11_sharpes.items() if n in signals}

# Weighted vote: positive-Sharpe models only
pos_models   = {n: s for n, s in available.items() if s > 0}
total_weight  = sum(pos_models.values())
weighted_vote = sum(signals[n] * pos_models[n] / total_weight for n in pos_models) if total_weight > 0 else 0
consensus     = 'LONG  (BUY)' if weighted_vote > 0 else 'SHORT (SELL)'
confidence    = abs(weighted_vote) * 100

# Simple majority
votes    = [signals[n] for n in available]
n_long   = sum(v > 0 for v in votes)
n_short  = sum(v < 0 for v in votes)
n_neut   = sum(v == 0 for v in votes)
majority = 'LONG' if n_long > n_short else ('SHORT' if n_short > n_long else 'FLAT')

# ── Position sizing ───────────────────────────────────────────────────────────
# Component 1 — signal strength: |Sharpe-weighted vote| (0 = no agreement, 1 = unanimous)
signal_strength = abs(weighted_vote)   # already in [0, 1]

# Component 2 — volatility scalar: shrink when recent vol > long-run average
#   recent_vol  = 20-day realised daily log-return std
#   longrun_vol = full-history std (baseline)
#   vol_scalar  = longrun / recent  (capped at 1.5 to avoid over-sizing on quiet days)
recent_vol   = float(df['log_return'].iloc[-20:].std())
longrun_vol  = float(df['log_return'].std())
vol_ratio    = recent_vol / (longrun_vol + 1e-8)
vol_scalar   = min(1.0 / vol_ratio, 1.5)   # inverse vol, cap upside at 1.5×

# Final recommended size as % of maximum allocated capital
position_pct = min(signal_strength * vol_scalar, 1.0) * 100   # clamp to [0, 100%]

# ── Print report ──────────────────────────────────────────────────────────────
DIVIDER = '─' * 66
# ── Price target: inverse-MSE weighted (all models with known MSE) ────────────
# Lower MSE = better frac_diff prediction accuracy = higher weight for price target
inv_mse      = {n: 1.0 / exp11_mse[n] for n in price_preds if n in exp11_mse}
total_inv_mse = sum(inv_mse.values())
mse_price    = sum(price_preds[n] * inv_mse[n] / total_inv_mse for n in inv_mse)
mse_chg      = mse_price - latest_gold
mse_pct      = mse_chg / latest_gold * 100

# ── Directional signal: Sharpe-weighted (positive-Sharpe models only) ─────────
# Sharpe measures trading signal quality, not price accuracy — kept separate
sharpe_price  = sum(price_preds[n] * pos_models[n] / total_weight
                    for n in pos_models if n in price_preds) if total_weight > 0 else latest_gold

DIVIDER2 = '─' * 74
print(f'\n{"═"*74}')
print(f'  GOLD TRADING SIGNAL — {next_day.strftime("%A %Y-%m-%d")}')
print(f'{"═"*74}')
print(f'  Latest close:  ${latest_gold:>10,.2f}  ({latest_date.date()})')
print(f'  Daily return:  {latest_return:>+.4f}  ({np.expm1(latest_return):>+.2%})')
print(f'{DIVIDER2}')
print(f'  {"Model":<14} {"Sharpe":>7}  {"Train":>8}  {"Price Target":>13}  {"Chg":>8}  {"z":>6}  {"Signal":>7}')
print(f'  {"-"*68}')

model_order = ['SVR_lin','Ridge','Lasso','ElasticNet','SVR_rbf','XGBoost',
               'LightGBM','RandomForest','MLP','LSTM','GRU','BiLSTM']
for name in model_order:
    if name not in signals:
        continue
    sig    = signals[name]
    z_val  = z_scores[name]
    sharpe = exp11_sharpes.get(name, 0)
    window = 'rec-3yr' if name in RECENT_MODELS else 'full'
    action = 'LONG +' if sig > 0 else ('SHORT -' if sig < 0 else 'FLAT  0')
    star   = ' ★' if sharpe >= 7.5 else ''
    px     = price_preds.get(name, float('nan'))
    chg    = px - latest_gold
    print(f'  {name:<14} {sharpe:>7.2f}  {window:>8}  ${px:>12,.2f}  {chg:>+8.2f}  {z_val:>+6.2f}  {action}{star}')

print(f'{DIVIDER2}')
print(f'  Price target  (1/MSE weighted):    ${mse_price:>10,.2f}  ({mse_chg:>+.2f},  {mse_pct:>+.2f}%)')
print(f'  Price target  (Sharpe weighted):   ${sharpe_price:>10,.2f}  (for reference only)')
print(f'  Note: 1/MSE weights reward price accuracy; Sharpe weights reward signal quality.')
print(f'{DIVIDER2}')
print(f'  Majority vote ({n_long}L / {n_short}S / {n_neut}N):  {majority}')
print(f'  Weighted consensus (★ Sharpe-wtd):   {consensus}')
print(f'  Confidence (Sharpe-weighted):         {confidence:.1f}%')
print(f'{"═"*74}')
print()

# ── Position sizing ──────────────────────────────────────────────────────────
print(f'\n{"─"*74}')
print(f'  Position sizing:')
print(f'    Signal strength  (Sharpe-wtd |vote|):  {signal_strength*100:>5.1f}%')
print(f'    Recent vol (20d):   {recent_vol*100:.3f}%/day  │  '
      f'Long-run avg: {longrun_vol*100:.3f}%/day  │  '
      f'Ratio: {vol_ratio:.2f}×')
vol_note = ('scale DOWN: elevated vol' if vol_ratio > 1.1 else
            'scale UP: quiet vol'      if vol_ratio < 0.9 else
            'normal vol')
print(f'    Volatility scalar:  {vol_scalar:.2f}×  ({vol_note})')
print(f'    ── Recommended size: {position_pct:>5.1f}% of max allocation ──')
if signals and majority != 'FLAT':
    print(f'    e.g. max=$10 000 → trade ${position_pct/100*10000:>7,.0f}  │  '
          f'max=$100 000 → trade ${position_pct/100*100000:>8,.0f}')
print(f'{"─"*74}')

# ── Market context ────────────────────────────────────────────────────────────
print('  Market context:')
if HAS_COT:
    spec_norm = df['cot_norm_spec'].iloc[-1]
    comm_norm = df['cot_norm_comm'].iloc[-1]
    spec_lbl = ('⚠ EXTREME LONG (crowded)' if spec_norm > 0.9 else
                '⚠ EXTREME SHORT' if spec_norm < 0.1 else 'neutral')
    comm_lbl = ('⚠ EXTREME HEDGE SHORT' if comm_norm < 0.1 else
                '⚠ EXTREME HEDGE LONG'  if comm_norm > 0.9 else 'neutral')
    print(f'    COT Spec index:  {spec_norm:.2f}  {spec_lbl}')
    print(f'    COT Comm index:  {comm_norm:.2f}  {comm_lbl}')

if 'DXY' in df.columns:
    dxy     = df['DXY'].iloc[-1]
    dxy_chg = df['DXY_chg'].iloc[-1] if 'DXY_chg' in df.columns else float('nan')
    dxy_lbl = '(USD weakening → gold bullish)' if dxy_chg < -0.005 else \
              '(USD strengthening → gold bearish)' if dxy_chg > 0.005 else '(USD stable)'
    print(f'    DXY:  {dxy:.2f}  ({dxy_chg:+.2%} 5d)  {dxy_lbl}')

if 'VIX' in df.columns:
    vix     = df['VIX'].iloc[-1]
    vix_lbl = '(elevated risk-off → gold safe-haven bid)' if vix > 25 else '(normal risk environment)'
    print(f'    VIX:  {vix:.2f}  {vix_lbl}')

if 'TNX_yield' in df.columns:
    tnx = df['TNX_yield'].iloc[-1]
    rp  = df['real_yield_px'].iloc[-1] if 'real_yield_px' in df.columns else float('nan')
    print(f'    10Y yield:  {tnx:.2f}%   real_yield_proxy={rp:.2f}')

print()
print(f'  Note: ★ = Sharpe ≥ 7.5 in Exp 11 backtest (2025-09-24 → 2026-05-01).')
print(f'        LONG=enter long, SHORT=enter short, FLAT=exit to cash (no position).')
print(f'        Signal is directional only. No position sizing or costs included.')
print(f'{"═"*66}')

# ══════════════════════════════════════════════════════════════════════════════
# SIGNAL LOG — append today's prediction; backfill actuals for past entries
# ══════════════════════════════════════════════════════════════════════════════
LOG_PATH = f'{RESULTS_DIR}/signal_log.csv' if os.path.exists('results') else 'signal_log.csv'
os.makedirs(os.path.dirname(LOG_PATH) if os.path.dirname(LOG_PATH) else '.', exist_ok=True)

MODEL_NAMES = ['SVR_lin','Ridge','Lasso','ElasticNet','SVR_rbf','XGBoost',
               'RandomForest','MLP','LSTM','GRU','BiLSTM']
LOG_COLS = (['run_timestamp','signal_for_date','signal','confidence_pct','position_size_pct',
             'price_at_signal','price_target_mse','price_target_chg_pct']
            + MODEL_NAMES
            + ['actual_price','actual_return_pct','strategy_return_pct'])

def sig_label(v):
    return 'LONG' if v > 0 else ('SHORT' if v < 0 else 'FLAT')

# ── Step 1: load existing log and backfill any missing actuals ────────────────
if os.path.exists(LOG_PATH):
    log_df = pd.read_csv(LOG_PATH)
    filled = []
    for idx, row in log_df.iterrows():
        if pd.isna(row.get('actual_price')):
            entry_date = pd.Timestamp(str(row['signal_for_date'])[:10])
            # entry = close of signal_for_date; exit = close of next trading day
            future_dates = df.index[df.index > entry_date]
            if entry_date in df.index and len(future_dates) > 0:
                entry_px   = float(df.loc[entry_date, 'Gold'])
                exit_date  = future_dates[0]
                actual_px  = float(df.loc[exit_date, 'Gold'])
                actual_ret = (actual_px / entry_px - 1) * 100
                sig_num    = (1 if row['signal'] == 'LONG' else
                             -1 if row['signal'] == 'SHORT' else 0)
                strat_ret  = sig_num * actual_ret
                log_df.loc[idx, 'actual_price']        = round(actual_px, 2)
                log_df.loc[idx, 'actual_return_pct']   = round(actual_ret, 4)
                log_df.loc[idx, 'strategy_return_pct'] = round(strat_ret, 4)
                filled.append(f"{entry_date.date()}→{exit_date.date()}  "
                              f"entry=${entry_px:.2f}  exit=${actual_px:.2f}  "
                              f"ret={actual_ret:+.2f}%  strategy={strat_ret:+.2f}%")
    if filled:
        log_df.to_csv(LOG_PATH, index=False)
        print(f'\n  Backfilled {len(filled)} log entry(s):')
        for f in filled:
            print(f'    {f}')
else:
    log_df = pd.DataFrame(columns=LOG_COLS)

# ── Step 2: append today's prediction (skip if already logged for this date) ──
next_day_str = next_day.strftime('%Y-%m-%d')
already_logged = (not log_df.empty and
                  log_df['signal_for_date'].astype(str).str.startswith(next_day_str).any())

if not already_logged:
    new_entry = {
        'run_timestamp':        datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'signal_for_date':      next_day_str,
        'signal':               sig_label(weighted_vote),
        'confidence_pct':       round(confidence, 2),
        'position_size_pct':    round(position_pct, 2),
        'price_at_signal':      round(float(latest_gold), 2),
        'price_target_mse':     round(mse_price, 2),
        'price_target_chg_pct': round(mse_pct, 4),
        'actual_price':         None,
        'actual_return_pct':    None,
        'strategy_return_pct':  None,
    }
    for mname in MODEL_NAMES:
        new_entry[mname] = sig_label(signals[mname]) if mname in signals else ''
    log_df = pd.concat([log_df, pd.DataFrame([new_entry])], ignore_index=True)
    log_df.to_csv(LOG_PATH, index=False)
    print(f'\n  Signal logged → {LOG_PATH}')
    print(f'  {next_day_str}  {sig_label(weighted_vote)}  '
          f'confidence={confidence:.1f}%  size={position_pct:.1f}%  '
          f'price_at_signal=${latest_gold:,.2f}  target=${mse_price:,.2f}')
else:
    print(f'\n  Log entry for {next_day_str} already exists — skipped.')

#!/usr/bin/env python3
"""
update_results.py — Regenerate ALL active result files using tuned model parameters.

Re-fits every model with params from tuned_<name>.json, then writes:
  results/all_model_metrics.csv
  results/all_model_metrics_table.png   ← readable visual table
  results/equity_curves_all_models.png
  results/predicted_vs_actual_all_models.png
  results/signal_timeline.png
  results/shap_summary.png              (best model by Sharpe)
  results/bias_tests.png                (best model by Sharpe)
  results/final_summary.txt

Usage:
    python update_results.py
"""

import os, json, time, warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap

from sklearn.linear_model import Ridge, Lasso
from sklearn.svm import SVR, LinearSVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import xgboost as xgb
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

from sklearn.linear_model import ElasticNet
import shap

MODELS_DIR  = 'saved_models'
RESULTS_DIR = 'results'
DATA_DIR    = 'data'
SEED        = 42
ZSCORE_WIN       = 10   # default fallback when no tuned zscore_win available
SIGNAL_THRESHOLD = 0.0  # default fallback when no tuned signal_threshold available
SHARPE_THRESH = 7.5
RECENT_YEARS = 3   # data window for recent-regime models

# Models trained on recent 3yr window (tuned with exp/recent-3yr params)
RECENT_MODELS = {'Ridge', 'Lasso', 'SVR_lin', 'SVR_rbf', 'XGBoost', 'ElasticNet'}
# Models trained on full history (tuned with exp/sharpe-cv params)
FULL_MODELS   = {'LightGBM', 'RandomForest', 'MLP', 'LSTM', 'GRU', 'BiLSTM'}
np.random.seed(SEED)
os.makedirs(RESULTS_DIR, exist_ok=True)

t_start = time.time()

# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD METADATA + TUNED PARAMS
# ══════════════════════════════════════════════════════════════════════════════
with open(f'{MODELS_DIR}/model_metadata.json') as f:
    meta = json.load(f)

FRAC_D         = meta['frac_d']
FEATURE_COLS   = meta['feature_cols']
LSTM_TS        = meta['lstm_best_timesteps']
GRU_TS         = meta['gru_best_timesteps']
BILSTM_TS      = meta['bilstm_best_timesteps']

def load_tuned(name, defaults):
    path = f'{MODELS_DIR}/tuned_{name}.json'
    if os.path.exists(path):
        return json.load(open(path)).get('best_params', defaults)
    return defaults

def load_tuned_zscore_win(name):
    """Return the tuned zscore_win for a model, or the global default."""
    path = f'{MODELS_DIR}/tuned_{name}.json'
    if os.path.exists(path):
        params = json.load(open(path)).get('best_params', {})
        return int(params.get('zscore_win', ZSCORE_WIN))
    return ZSCORE_WIN

def load_tuned_threshold(name):
    """Return the tuned signal_threshold for a model, or the global default."""
    path = f'{MODELS_DIR}/tuned_{name}.json'
    if os.path.exists(path):
        params = json.load(open(path)).get('best_params', {})
        return float(params.get('signal_threshold', SIGNAL_THRESHOLD))
    return SIGNAL_THRESHOLD

TUNED = {
    'Ridge':        load_tuned('Ridge',        {'alpha': 0.01}),
    'Lasso':        load_tuned('Lasso',        {'alpha': 0.001}),
    'SVR_lin':      load_tuned('SVR_lin',      {'C': 100}),
    'SVR_rbf':      load_tuned('SVR_rbf',      {'C': 0.1, 'gamma': 0.01}),
    'XGBoost':      load_tuned('XGBoost',      {'n_estimators': 200, 'max_depth': 2, 'learning_rate': 0.05}),
    'RandomForest': load_tuned('RandomForest', {'n_estimators': 100}),
    'MLP':          load_tuned('MLP',          {'hidden_layer_sizes': (64, 32), 'alpha': 0.001}),
    'LSTM':         load_tuned('LSTM',         {'units': 32, 'dropout': 0.35, 'recurrent_dropout': 0.2, 'timesteps': LSTM_TS}),
    'GRU':          load_tuned('GRU',          {'units': 32, 'dropout': 0.3,  'recurrent_dropout': 0.2, 'timesteps': GRU_TS}),
    'BiLSTM':       load_tuned('BiLSTM',       {'units': 32, 'dropout': 0.4,  'recurrent_dropout': 0.2, 'timesteps': BILSTM_TS}),
    'ElasticNet':   load_tuned('ElasticNet',   {'alpha': 0.01, 'l1_ratio': 0.5}),
    'LightGBM':     load_tuned('LightGBM',     {'n_estimators': 200, 'num_leaves': 31, 'learning_rate': 0.05}),
}

# Per-model signal params — read from tuned JSON if available, else fall back to global defaults
ZSCORE_WINS  = {name: load_tuned_zscore_win(name) for name in TUNED}
THRESHOLDS   = {name: load_tuned_threshold(name)  for name in TUNED}
print(f'  Per-model zscore_win:       {ZSCORE_WINS}')
print(f'  Per-model signal_threshold: {THRESHOLDS}')

# ══════════════════════════════════════════════════════════════════════════════
# 2. FEATURE ENGINEERING (identical to train.py / plot_tuned.py)
# ══════════════════════════════════════════════════════════════════════════════
print('Building features...')
df = pd.read_csv(f'{DATA_DIR}/merged_gold_dataset.csv', index_col=0, parse_dates=True)
df.index = pd.to_datetime(df.index).normalize()
df.sort_index(inplace=True)
HAS_COT = 'Comm_Positions_Long_All' in df.columns

def frac_diff(series, d, window=252, threshold=1e-5):
    w = [1.0]
    for k in range(1, window):
        w_k = -w[-1] * (d - k + 1) / k
        if abs(w_k) < threshold: break
        w.append(w_k)
    w, n_, width = np.array(w[::-1]), len(series), len(w)
    out = np.full(n_, np.nan)
    for i in range(width - 1, n_):
        vals = series.iloc[i - width + 1: i + 1].values
        if not np.any(np.isnan(vals)): out[i] = np.dot(w, vals)
    return pd.Series(out, index=series.index)

df['frac_diff_log_Gold'] = frac_diff(df['log_Gold'], d=FRAC_D)
if HAS_COT:
    for col in ['Comm_Positions_Long_All','Comm_Positions_Short_All',
                'NonComm_Positions_Long_All','NonComm_Positions_Short_All','Open_Interest_All']:
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
    for oc, rc in [('cot_norm_spec', 'net_speculator'), ('cot_norm_comm', 'net_commercial')]:
        rmin = df[rc].rolling(252, min_periods=60).min()
        rmax = df[rc].rolling(252, min_periods=60).max()
        df[oc] = (df[rc] - rmin) / (rmax - rmin + 1e-12)

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
df['BB_pct']   = (df['Gold'] - (bb_mid - 2*bb_std)) / (4*bb_std + 1e-12)
for lag in [1, 2, 3, 5, 10]:
    df[f'target_lag{lag}'] = df['frac_diff_log_Gold'].shift(lag)
df['log_return'] = df['log_Gold'].diff()

# Multi-period momentum features (must match train.py)
for period in [21, 63, 126, 252]:
    df[f'gold_ret_{period}d'] = df['log_Gold'].diff(period)

# Month-of-year dummies (gold seasonal patterns)
for m_num in range(1, 13):
    df[f'month_{m_num}'] = (df.index.month == m_num).astype(int)

# COT 20-day position changes
if HAS_COT:
    df['net_spec_chg_20d'] = df['net_speculator'].diff(20)
    df['net_comm_chg_20d'] = df['net_commercial'].diff(20)

df.dropna(inplace=True)
FEATURE_COLS = [c for c in FEATURE_COLS if c in df.columns]

# ── Full-history split (for LightGBM, RandomForest, MLP, LSTM, GRU, BiLSTM) ─
_X_full  = df[FEATURE_COLS].values
_y_full  = df['frac_diff_log_Gold'].values
_dates   = df.index
_lr_full = df['log_return'].values
_gold_full = df['Gold'].values
_n_full  = len(_X_full)
_spl_full = int(_n_full * 0.80)

X_tv_full, y_tv_full = _X_full[:_spl_full], _y_full[:_spl_full]
X_te_full, y_te_full = _X_full[_spl_full:], _y_full[_spl_full:]
dates_te_full         = _dates[_spl_full:]
lr_te_full            = _lr_full[_spl_full:]

SVR_EPS_FULL = round(0.25 * float(np.std(y_tv_full)), 6)
sc_full = StandardScaler()
X_tv_full_s = sc_full.fit_transform(X_tv_full)
X_te_full_s = sc_full.transform(X_te_full)

# ── Recent-window split (for Ridge, Lasso, SVR_lin, SVR_rbf, XGBoost, ElasticNet) ─
_rec_start = df.index[-1] - pd.DateOffset(years=RECENT_YEARS)
df_rec     = df[df.index >= _rec_start].copy()
_X_rec     = df_rec[FEATURE_COLS].values
_y_rec     = df_rec['frac_diff_log_Gold'].values
_dates_rec = df_rec.index
_lr_rec    = df_rec['log_return'].values
_gold_rec  = df_rec['Gold'].values
_n_rec     = len(_X_rec)
_spl_rec   = int(_n_rec * 0.80)

X_tv_rec, y_tv_rec = _X_rec[:_spl_rec], _y_rec[:_spl_rec]
X_te_rec, y_te_rec = _X_rec[_spl_rec:], _y_rec[_spl_rec:]
dates_te_rec        = _dates_rec[_spl_rec:]   # 2025-09-24 → 2026-05-01
lr_te_rec           = _lr_rec[_spl_rec:]
gold_te_rec         = _gold_rec[_spl_rec:]

SVR_EPS_REC = round(0.25 * float(np.std(y_tv_rec)), 6)
sc_rec = StandardScaler()
X_tv_rec_s = sc_rec.fit_transform(X_tv_rec)
X_te_rec_s = sc_rec.transform(X_te_rec)

# ── Common evaluation period = recent test (2025-09-24 → 2026-05-01) ─────────
EVAL_START  = dates_te_rec[0]
dates_test  = dates_te_rec          # all plots use this
gold_test   = gold_te_rec

# For full models: index in their test set where the eval period begins
_eval_mask_full  = dates_te_full >= EVAL_START
EVAL_OFFSET_FULL = int(np.where(_eval_mask_full)[0][0])

SVR_EPSILON = SVR_EPS_REC   # used for recent SVR models
print(f'  Recent test ({RECENT_YEARS}yr): {dates_te_rec[0].date()} → {dates_te_rec[-1].date()}  ({len(X_te_rec)} days)')
print(f'  Full test:         {dates_te_full[0].date()} → {dates_te_full[-1].date()}  ({len(X_te_full)} days)')
print(f'  Common eval start: {EVAL_START.date()}  (offset in full test: {EVAL_OFFSET_FULL})')
print(f'  SVR eps — recent: {SVR_EPS_REC}  full: {SVR_EPS_FULL}')

# ══════════════════════════════════════════════════════════════════════════════
# 3. FIT ALL MODELS + COLLECT METRICS
# ══════════════════════════════════════════════════════════════════════════════
def trading_metrics(preds, log_rets, y_true, zscore_win=None, threshold=None,
                    warmup_preds=None):
    """Compute trading metrics on preds/log_rets.

    warmup_preds: optional array prepended to preds before z-score computation
    so that rolling statistics have warmup context (used for full-history models
    where predictions exist before the common eval period).
    Signal is computed on the full concatenated array then sliced to len(preds).
    """
    if zscore_win is None:
        zscore_win = ZSCORE_WIN
    if threshold is None:
        threshold = SIGNAL_THRESHOLD
    preds    = np.asarray(preds)
    log_rets = np.asarray(log_rets)
    y_true   = np.asarray(y_true)

    # Build prediction array for z-score (with optional warmup prefix)
    if warmup_preds is not None and len(warmup_preds) > 0:
        all_p = np.concatenate([np.asarray(warmup_preds), preds])
    else:
        all_p = preds

    ps  = pd.Series(all_p)
    z   = (ps - ps.rolling(zscore_win, min_periods=1).mean()) / \
          ps.rolling(zscore_win, min_periods=1).std().fillna(1e-8)
    sig_all = np.where(z >  threshold,  1.0,
              np.where(z < -threshold, -1.0, 0.0))
    sig = sig_all[-len(preds):]   # slice to eval period

    n_ = len(preds)
    strat  = sig * log_rets[:n_]
    eq     = np.exp(np.cumsum(strat))
    cagr   = (eq[-1] ** (252 / n_) - 1) * 100
    sharpe = strat.mean() / (strat.std() + 1e-12) * np.sqrt(252)
    wins   = strat[strat > 0].sum()
    losses = -strat[strat < 0].sum()
    pf     = wins / (losses + 1e-12)
    roll_max = np.maximum.accumulate(eq)
    max_dd = float(np.min((eq - roll_max) / (roll_max + 1e-12)))
    mkt_dir = np.sign(log_rets[:n_])
    active  = sig != 0
    dir_acc  = float(np.mean(sig[active] == mkt_dir[active])) if active.any() else 0.0
    coverage = float(np.mean(active))
    mse      = mean_squared_error(y_true[:n_], preds)
    return dict(Sharpe=sharpe, CAGR=cagr, PF=pf, MaxDD=max_dd,
                DirAcc=dir_acc, Coverage=coverage, MSE=mse,
                equity=eq, signal=sig, preds=np.array(preds))

def make_sequences(X_, y_, ts):
    Xs, ys = [], []
    for i in range(ts, len(X_)):
        Xs.append(X_[i-ts:i]); ys.append(y_[i])
    return np.array(Xs), np.array(ys)

all_metrics = {}; all_preds = {}; all_signals = {}

# Helper: strip non-model keys (e.g. zscore_win) from a param dict
def model_params(name):
    """Return TUNED[name] with signal-only keys (zscore_win, signal_threshold) removed."""
    skip = {'zscore_win', 'signal_threshold'}
    return {k: v for k, v in TUNED[name].items() if k not in skip}

# ── sklearn models ─────────────────────────────────────────────────────────────
print('\nFitting sklearn models with tuned params...')
SK_CONFIGS = {
    'Ridge':        Ridge(**model_params('Ridge')),
    'Lasso':        Lasso(**model_params('Lasso')),
    'SVR_lin':      LinearSVR(C=model_params('SVR_lin')['C'], epsilon=SVR_EPS_REC, max_iter=10000, dual=True),
    'SVR_rbf':      SVR(kernel='rbf', C=model_params('SVR_rbf')['C'],
                        gamma=model_params('SVR_rbf')['gamma'], epsilon=SVR_EPS_REC),
    'XGBoost':      xgb.XGBRegressor(**model_params('XGBoost'), random_state=SEED, verbosity=0),
    'RandomForest': RandomForestRegressor(**model_params('RandomForest'), random_state=SEED),
    'MLP':          MLPRegressor(**{k: tuple(v) if isinstance(v, list) else v
                                    for k, v in model_params('MLP').items()},
                                 max_iter=500, random_state=SEED),
    'ElasticNet':   ElasticNet(**model_params('ElasticNet'), max_iter=5000),
}
if LIGHTGBM_AVAILABLE:
    SK_CONFIGS['LightGBM'] = lgb.LGBMRegressor(**model_params('LightGBM'), random_state=SEED, verbose=-1)
sk_fitted = {}
for name, model in SK_CONFIGS.items():
    is_recent = name in RECENT_MODELS
    # Select the right training/test split and scaler
    _X_tv_s = X_tv_rec_s  if is_recent else X_tv_full_s
    _y_tv   = y_tv_rec    if is_recent else y_tv_full
    _X_te_s = X_te_rec_s  if is_recent else X_te_full_s
    _y_te   = y_te_rec    if is_recent else y_te_full
    _lr_te  = lr_te_rec   if is_recent else lr_te_full
    tag     = f'recent-{RECENT_YEARS}yr' if is_recent else 'full-hist'

    model.fit(_X_tv_s, _y_tv)
    preds_all = model.predict(_X_te_s)
    zw  = ZSCORE_WINS.get(name, ZSCORE_WIN)
    thr = THRESHOLDS.get(name, SIGNAL_THRESHOLD)

    if is_recent:
        # Full test = eval period
        m = trading_metrics(preds_all, _lr_te, _y_te, zscore_win=zw, threshold=thr)
        preds_eval = preds_all
    else:
        # Use predictions before EVAL_START as z-score warmup, slice to eval period for metrics
        warmup = preds_all[:EVAL_OFFSET_FULL]
        preds_eval = preds_all[EVAL_OFFSET_FULL:]
        m = trading_metrics(preds_eval, _lr_te[EVAL_OFFSET_FULL:], _y_te[EVAL_OFFSET_FULL:],
                            zscore_win=zw, threshold=thr, warmup_preds=warmup)

    all_metrics[name] = m
    all_preds[name]   = preds_eval
    all_signals[name] = m['signal']
    sk_fitted[name]   = model
    print(f'  {name:<22} [{tag}] Sharpe={m["Sharpe"]:6.3f}  DirAcc={m["DirAcc"]*100:.1f}%  '
          f'Coverage={m["Coverage"]*100:.1f}%  zscore_win={zw}  threshold={thr}  params={TUNED[name]}')

# ── Keras models ───────────────────────────────────────────────────────────────
print('\nFitting Keras models with tuned params...')
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM as KerasLSTM, GRU as KerasGRU, \
    Bidirectional, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import tensorflow as tf
tf.random.set_seed(SEED)

def directional_mse(y_true, y_pred):
    """Custom loss: MSE + sign-error penalty to prevent Keras mean-collapse."""
    mse      = tf.reduce_mean(tf.square(y_true - y_pred))
    sign_err = tf.maximum(0.0, -y_true * y_pred)
    return mse + 0.5 * tf.reduce_mean(sign_err)

keras_fitted = {}
for model_name, build_info in [
    ('LSTM',   TUNED['LSTM']),
    ('GRU',    TUNED['GRU']),
    ('BiLSTM', TUNED['BiLSTM']),
]:
    # All Keras models are FULL_MODELS — trained on full history
    p  = build_info
    ts = p['timesteps']; units = p['units']
    drop = p['dropout']; rdrop = p['recurrent_dropout']
    n_feat = X_tv_full_s.shape[1]

    Xtv_sq, ytv_sq = make_sequences(X_tv_full_s, y_tv_full, ts)
    Xte_sq, yte_sq = make_sequences(X_te_full_s, y_te_full, ts)
    lr_seq_full  = lr_te_full[ts:]
    dates_seq_full = dates_te_full[ts:]

    if model_name == 'LSTM':
        m = Sequential([KerasLSTM(units, input_shape=(ts, n_feat),
                                  dropout=drop, recurrent_dropout=rdrop,
                                  kernel_regularizer=l2(0.001)), Dense(1)])
    elif model_name == 'GRU':
        m = Sequential([KerasGRU(units, input_shape=(ts, n_feat),
                                 dropout=drop, recurrent_dropout=rdrop), Dense(1)])
    else:
        m = Sequential([Bidirectional(KerasLSTM(units, recurrent_dropout=rdrop),
                                      input_shape=(ts, n_feat)),
                        Dropout(drop), Dense(16, activation='relu'), Dense(1)])
    m.compile(optimizer=Adam(0.001), loss=directional_mse)
    m.fit(Xtv_sq, ytv_sq, epochs=50, batch_size=32, verbose=0,
          callbacks=[EarlyStopping(patience=10, restore_best_weights=True)],
          validation_split=0.1)

    preds_all_seq = m.predict(Xte_sq, verbose=0).flatten()
    zw  = ZSCORE_WINS.get(model_name, ZSCORE_WIN)
    thr = THRESHOLDS.get(model_name, SIGNAL_THRESHOLD)

    # Slice to common eval period with z-score warmup
    eval_mask_seq = dates_seq_full >= EVAL_START
    eval_off_seq  = int(np.where(eval_mask_seq)[0][0]) if eval_mask_seq.any() else 0
    warmup_seq    = preds_all_seq[:eval_off_seq]
    preds_eval_seq = preds_all_seq[eval_off_seq:]
    lr_eval_seq    = lr_seq_full[eval_off_seq:]
    y_eval_seq     = yte_sq[eval_off_seq:]

    met = trading_metrics(preds_eval_seq, lr_eval_seq, y_eval_seq,
                          zscore_win=zw, threshold=thr, warmup_preds=warmup_seq)
    all_metrics[model_name] = met
    all_preds[model_name]   = preds_eval_seq
    all_signals[model_name] = met['signal']
    keras_fitted[model_name] = (m, ts)
    print(f'  {model_name:<22} [full-hist] Sharpe={met["Sharpe"]:6.3f}  DirAcc={met["DirAcc"]*100:.1f}%  '
          f'Coverage={met["Coverage"]*100:.1f}%  ts={ts}  zscore_win={zw}  threshold={thr}')

# Buy & Hold — common eval period (recent test window)
bnh_eq     = np.exp(np.cumsum(lr_te_rec))
bnh_cagr   = (bnh_eq[-1] ** (252/len(lr_te_rec)) - 1) * 100
bnh_sharpe = lr_te_rec.mean() / (lr_te_rec.std() + 1e-12) * np.sqrt(252)
bnh_up     = float(np.mean(lr_te_rec > 0))

# Best model by Sharpe
best_name = max(all_metrics, key=lambda n: all_metrics[n]['Sharpe'])
print(f'\n  Best model by Sharpe: {best_name} ({all_metrics[best_name]["Sharpe"]:.3f})')

# ══════════════════════════════════════════════════════════════════════════════
# 4. SAVE all_model_metrics.csv
# ══════════════════════════════════════════════════════════════════════════════
MODEL_ORDER = ['Ridge','Lasso','SVR_lin','SVR_rbf','XGBoost','RandomForest',
               'MLP','ElasticNet','LightGBM','LSTM','GRU','BiLSTM']
rows = []
for name in MODEL_ORDER:
    if name not in all_metrics:
        continue
    m = all_metrics[name]
    rows.append({'Model':    name,
                 'Sharpe':   round(m['Sharpe'], 4),
                 'CAGR_%':   round(m['CAGR'], 2),
                 'DirAcc':   round(m['DirAcc'], 4),
                 'Coverage': round(m['Coverage'], 4),
                 'PF':       round(m['PF'], 4),
                 'MaxDD':    round(m['MaxDD'], 4),
                 'MSE':      round(m['MSE'], 6)})
metrics_df = pd.DataFrame(rows)
metrics_df.to_csv(f'{RESULTS_DIR}/all_model_metrics.csv', index=False)
print(f'\nSaved → {RESULTS_DIR}/all_model_metrics.csv')

# ══════════════════════════════════════════════════════════════════════════════
# 5. VISUAL SUMMARY TABLE  (all_model_metrics_table.png)
# ══════════════════════════════════════════════════════════════════════════════
print('Plotting summary table...')

sorted_df = metrics_df.sort_values('Sharpe', ascending=False).reset_index(drop=True)

COLORS = {
    'MLP': '#00CED1', 'SVR_lin': '#1E90FF', 'Ridge': '#DAA520',
    'SVR_rbf': '#9932CC', 'RandomForest': '#FF8C00', 'XGBoost': '#DC143C',
    'Lasso': '#228B22', 'ElasticNet': '#3CB371', 'LightGBM': '#FF6347',
    'LSTM': '#888888', 'GRU': '#AAAAAA', 'BiLSTM': '#CCCCCC',
}

fig, ax = plt.subplots(figsize=(14, 5))
ax.axis('off')

col_labels = ['Model', 'Sharpe', 'CAGR', 'Active DirAcc', 'Coverage', 'Profit Factor', 'Max DD', 'MSE']
col_keys   = ['Model', 'Sharpe', 'CAGR_%', 'DirAcc', 'Coverage', 'PF', 'MaxDD', 'MSE']
bnh_row    = ['Buy & Hold', f'{bnh_sharpe:.3f}', f'{bnh_cagr:.1f}%',
              f'{bnh_up*100:.1f}%', '100.0%', '—', '—', '—']

table_data = []
row_colors = []
for _, row in sorted_df.iterrows():
    star = ' ★' if row['Sharpe'] >= SHARPE_THRESH else ''
    table_data.append([
        row['Model'] + star,
        f"{row['Sharpe']:.3f}",
        f"{row['CAGR_%']:.1f}%",
        f"{row['DirAcc']*100:.1f}%",
        f"{row['Coverage']*100:.1f}%",
        f"{row['PF']:.2f}",
        f"{row['MaxDD']*100:.1f}%",
        f"{row['MSE']:.6f}",
    ])
    c = COLORS.get(row['Model'], '#DDDDDD')
    row_colors.append([c + '55'] * len(col_labels))  # light tint

# Add separator + buy & hold
table_data.append(['—'] * len(col_labels))
row_colors.append(['#F0F0F0'] * len(col_labels))
table_data.append(bnh_row)
row_colors.append(['#EEEEEE'] * len(col_labels))

tbl = ax.table(cellText=table_data, colLabels=col_labels,
               cellLoc='center', loc='center',
               cellColours=row_colors)
tbl.auto_set_font_size(False)
tbl.set_fontsize(11)
tbl.scale(1, 1.6)

# Header styling
for j in range(len(col_labels)):
    tbl[(0, j)].set_facecolor('#2C3E50')
    tbl[(0, j)].get_text().set_color('white')
    tbl[(0, j)].get_text().set_fontweight('bold')

# Bold strong models
for i, row in enumerate(sorted_df.itertuples(), 1):
    if row.Sharpe >= SHARPE_THRESH:
        for j in range(len(col_labels)):
            tbl[(i, j)].get_text().set_fontweight('bold')

ax.set_title(f'Gold Prediction — All Model Results (tuned, test: {dates_test[0].date()} → {dates_test[-1].date()})\n★ = Sharpe ≥ {SHARPE_THRESH}',
             fontsize=13, fontweight='bold', pad=12)
plt.tight_layout()
plt.savefig(f'{RESULTS_DIR}/all_model_metrics_table.png', dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved → {RESULTS_DIR}/all_model_metrics_table.png')

# ══════════════════════════════════════════════════════════════════════════════
# 6. EQUITY CURVES  (equity_curves_all_models.png)
# ══════════════════════════════════════════════════════════════════════════════
print('Plotting equity curves...')
COLOR_MAP = {
    'MLP': '#00CED1', 'SVR_lin': '#1E90FF', 'Ridge': '#DAA520',
    'SVR_rbf': '#9932CC', 'RandomForest': '#FF8C00', 'XGBoost': '#DC143C',
    'Lasso': '#228B22', 'ElasticNet': '#3CB371', 'LightGBM': '#FF6347',
    'LSTM': '#888', 'GRU': '#AAA', 'BiLSTM': '#CCC',
}

fig = plt.figure(figsize=(16, 12))
gs  = gridspec.GridSpec(3, 1, figure=fig, height_ratios=[1, 2.5, 1], hspace=0.35)

# Panel 1: Gold price
ax0 = fig.add_subplot(gs[0])
ax0.plot(dates_test, gold_test, color='goldenrod', lw=1.5)
ax0.set_ylabel('Gold (USD)')
ax0.set_title('Gold Price — Test Period', fontsize=11)
ax0.grid(alpha=0.25); ax0.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y'))

# Panel 2: Equity curves
ax1 = fig.add_subplot(gs[1])
ax1.plot(dates_test, bnh_eq, color='#555', lw=1.5, ls='--',
         label=f'Buy & Hold  Sharpe={bnh_sharpe:.2f}  CAGR={bnh_cagr:.1f}%')
for name in MODEL_ORDER:
    m   = all_metrics[name]
    sig = m['signal']; eq = m['equity']
    n_  = min(len(sig), len(dates_test))
    lw  = 2.0 if m['Sharpe'] >= SHARPE_THRESH else 0.8
    ls  = '-'  if m['Sharpe'] >= SHARPE_THRESH else '--'
    al  = 1.0  if m['Sharpe'] >= SHARPE_THRESH else 0.45
    star = ' ★' if m['Sharpe'] >= SHARPE_THRESH else ''
    lbl = (f'{name}{star}  Sharpe={m["Sharpe"]:.2f}  '
           f'CAGR={m["CAGR"]:.0f}%  DirAcc={m["DirAcc"]*100:.1f}%  Cov={m["Coverage"]*100:.0f}%')
    ax1.plot(dates_test[:n_], eq, color=COLOR_MAP[name], lw=lw, ls=ls, alpha=al, label=lbl)

ax1.set_ylabel('Portfolio Value (×)')
ax1.set_title('Equity Curves — All Models (tuned) vs Buy & Hold', fontsize=11)
ax1.legend(fontsize=7, ncol=2, loc='upper left')
ax1.grid(alpha=0.25)
ax1.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y'))

# Panel 3: Consensus bar
ax2 = fig.add_subplot(gs[2])
strong = [n for n in MODEL_ORDER if all_metrics[n]['Sharpe'] >= SHARPE_THRESH]
n_str  = len(strong)
cons_vals = np.zeros(len(dates_test))
for name in strong:
    sig = all_metrics[name]['signal']
    n_  = min(len(sig), len(dates_test))
    cons_vals[:n_] += sig[:n_]
cons_norm = cons_vals / (n_str + 1e-12)
colors_bar = ['#228B22' if v >= 0 else '#DC143C' for v in cons_norm]
ax2.bar(dates_test, cons_norm, color=colors_bar, width=1.5, alpha=0.8)
ax2.axhline(0, color='black', lw=0.8)
ax2.set_ylabel('Consensus')
ax2.set_ylim(-1.1, 1.1)
ax2.set_title(f'Strong-Model Consensus ({n_str} models with Sharpe ≥ {SHARPE_THRESH})', fontsize=10)
ax2.grid(alpha=0.2, axis='y')
ax2.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y'))

plt.savefig(f'{RESULTS_DIR}/equity_curves_all_models.png', dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved → {RESULTS_DIR}/equity_curves_all_models.png')

# ══════════════════════════════════════════════════════════════════════════════
# 7. PREDICTED VS ACTUAL — ALL MODELS  (predicted_vs_actual_all_models.png)
# ══════════════════════════════════════════════════════════════════════════════
print('Plotting predicted vs actual...')
n_models = len([n for n in MODEL_ORDER if n in all_metrics])
ncols = 4
nrows = (n_models + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(22, nrows * 3.5))
axes = axes.flatten()
_plot_idx = 0
for name in MODEL_ORDER:
    if name not in all_metrics:
        continue
    idx = _plot_idx
    _plot_idx += 1
    ax = axes[idx]
    m  = all_metrics[name]
    p  = all_preds[name]
    n_ = min(len(p), len(dates_test))
    ax.plot(dates_test[:n_], y_te_rec[:n_], color='steelblue', lw=0.6,
            alpha=0.7, label='Actual')
    ax.plot(dates_test[:n_], p, color='tomato', lw=0.8,
            alpha=0.85, label='Predicted')
    star = ' ★' if m['Sharpe'] >= SHARPE_THRESH else ''
    ax.set_title(f'{name}{star}\nSharpe={m["Sharpe"]:.2f}  DirAcc={m["DirAcc"]*100:.1f}%  Cov={m["Coverage"]*100:.0f}%',
                 fontsize=9)
    ax.grid(alpha=0.2); ax.legend(fontsize=7)
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%y'))
for ax in axes[_plot_idx:]:
    ax.set_visible(False)
fig.suptitle('Predicted vs Actual (frac-diff log return) — All Models (tuned)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{RESULTS_DIR}/predicted_vs_actual_all_models.png', dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved → {RESULTS_DIR}/predicted_vs_actual_all_models.png')

# ══════════════════════════════════════════════════════════════════════════════
# 8. SIGNAL TIMELINE  (signal_timeline.png)
# ══════════════════════════════════════════════════════════════════════════════
print('Plotting signal timeline...')
fig, axes = plt.subplots(len(MODEL_ORDER) + 1, 1, figsize=(16, 14), sharex=True)
fig.subplots_adjust(hspace=0.08)

for idx, name in enumerate(MODEL_ORDER):
    ax  = axes[idx]
    m   = all_metrics[name]
    sig = m['signal']
    n_  = min(len(sig), len(dates_test))
    d_  = dates_test[:n_]
    colors_s = ['#228B22' if s > 0 else '#DC143C' for s in sig[:n_]]
    ax.bar(d_, sig[:n_], color=colors_s, width=1.5, alpha=0.85)
    ax.set_yticks([-1, 1]); ax.set_yticklabels(['SHORT', 'LONG'], fontsize=7)
    ax.set_ylim(-1.4, 1.4); ax.grid(alpha=0.15, axis='x')
    star = ' ★' if m['Sharpe'] >= SHARPE_THRESH else ''
    ax.set_ylabel(f'{name}{star}', fontsize=8, rotation=0, labelpad=55, va='center')
    ax2r = ax.twinx()
    eq_ = m['equity']
    ax2r.plot(d_, eq_, color=COLOR_MAP[name], lw=0.9, alpha=0.7)
    ax2r.set_ylabel(f'×{eq_[-1]:.1f}\n{m["CAGR"]:.0f}%', fontsize=7)

# Bottom: gold price
ax_g = axes[-1]
ax_g.plot(dates_test, gold_test, color='goldenrod', lw=1.2)
ax_g.set_ylabel('Gold\n(USD)', fontsize=8, rotation=0, labelpad=55, va='center')
ax_g.grid(alpha=0.2)
ax_g.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y'))

fig.suptitle('Per-Model Signal Timeline (LONG / SHORT) — Tuned Models\n'
             '★ = Sharpe ≥ 7.5 | Right axis: portfolio multiplier & CAGR',
             fontsize=12, fontweight='bold')
plt.savefig(f'{RESULTS_DIR}/signal_timeline.png', dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved → {RESULTS_DIR}/signal_timeline.png')

# ══════════════════════════════════════════════════════════════════════════════
# 9. SHAP SUMMARY — BEST MODEL  (shap_summary.png)
# ══════════════════════════════════════════════════════════════════════════════
print(f'Computing SHAP for best model ({best_name})...')
best_model_obj = sk_fitted.get(best_name)
# SHAP uses whichever split the best model was trained on
_is_rec_best = best_name in RECENT_MODELS
X_te_shap = X_te_rec_s  if _is_rec_best else X_te_full_s[EVAL_OFFSET_FULL:]
X_tv_shap = X_tv_rec_s  if _is_rec_best else X_tv_full_s

try:
    if best_name in ['XGBoost', 'RandomForest']:
        explainer = shap.TreeExplainer(best_model_obj)
        shap_vals = explainer.shap_values(X_te_shap)
    elif best_name in ['Ridge', 'Lasso', 'SVR_lin']:
        masker    = shap.maskers.Independent(X_tv_shap, max_samples=200)
        explainer = shap.LinearExplainer(best_model_obj, masker)
        shap_vals = explainer.shap_values(X_te_shap)
    else:
        masker    = shap.maskers.Independent(X_tv_shap, max_samples=100)
        explainer = shap.PermutationExplainer(best_model_obj.predict, masker)
        shap_vals = explainer(X_te_shap[:200]).values

    fig, ax = plt.subplots(figsize=(10, 7))
    shap.summary_plot(shap_vals, X_te_shap, feature_names=FEATURE_COLS,
                      show=False, max_display=20)
    plt.title(f'SHAP Feature Importance — {best_name} (best model, Sharpe={all_metrics[best_name]["Sharpe"]:.2f})',
              fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/shap_summary.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved → {RESULTS_DIR}/shap_summary.png')
except Exception as e:
    print(f'  SHAP failed ({e}), skipping shap_summary.png')

# ══════════════════════════════════════════════════════════════════════════════
# 10. BIAS TESTS — BEST MODEL  (bias_tests.png)
# ══════════════════════════════════════════════════════════════════════════════
print(f'Running bias tests for best model ({best_name})...')
best_sig  = all_metrics[best_name]['signal']
n_sig     = len(best_sig)
strat_ret = best_sig * lr_te_rec[:n_sig]
obs_sharpe = strat_ret.mean() / (strat_ret.std() + 1e-12) * np.sqrt(252)

# MC permutation test
N_PERM = 1000
perm_sharpes = []
for _ in range(N_PERM):
    perm = np.random.permutation(best_sig)
    s    = perm * lr_te_rec[:n_sig]
    perm_sharpes.append(s.mean() / (s.std() + 1e-12) * np.sqrt(252))
mc_pval = float(np.mean(np.array(perm_sharpes) >= obs_sharpe))

# White Reality Check (circular block bootstrap)
def _wrc_statistic(sig, log_rets, n_boot=1000, block_len=20):
    n_  = len(sig); strat = sig * log_rets[:n_]
    obs = strat.mean()
    boot = []
    for _ in range(n_boot):
        idx = (np.random.randint(0, n_) + np.arange(n_)) % n_
        boot.append((strat[idx]).mean())
    return obs, np.array(boot), float(np.mean(np.array(boot) >= obs))

obs_wrc, null_wrc, wrc_pval = _wrc_statistic(best_sig, lr_te_rec)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.hist(perm_sharpes, bins=40, color='steelblue', alpha=0.75, edgecolor='white')
ax1.axvline(obs_sharpe, color='red', lw=2, label=f'Observed Sharpe={obs_sharpe:.2f}')
ax1.set_title(f'MC Permutation Test  (p={mc_pval:.4f})\n{best_name}', fontsize=11)
ax1.set_xlabel('Sharpe (random signal)'); ax1.legend()

ax2.hist(null_wrc, bins=40, color='coral', alpha=0.75, edgecolor='white')
ax2.axvline(obs_wrc, color='darkred', lw=2, label=f'Observed mean ret={obs_wrc:.5f}')
ax2.set_title(f'White Reality Check  (p={wrc_pval:.4f})\n{best_name}', fontsize=11)
ax2.set_xlabel('Mean daily return (bootstrap)'); ax2.legend()

plt.suptitle(f'Bias Tests — {best_name}  |  MC p={mc_pval:.4f}  WRC p={wrc_pval:.4f}',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{RESULTS_DIR}/bias_tests.png', dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved → {RESULTS_DIR}/bias_tests.png')

# ══════════════════════════════════════════════════════════════════════════════
# 11. FINAL SUMMARY TEXT  (final_summary.txt)
# ══════════════════════════════════════════════════════════════════════════════
sorted_metrics = sorted(all_metrics.items(), key=lambda x: x[1]['Sharpe'], reverse=True)
lines = [
    '=' * 72,
    'GOLD PRICE PREDICTION — FINAL SUMMARY',
    f'Generated: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")}',
    f'Test period: {dates_test[0].date()} → {dates_test[-1].date()} ({len(dates_test)} days)',
    f'Features: {len(FEATURE_COLS)} | Frac-diff d={FRAC_D} | Z-score wins: {ZSCORE_WINS} | Thresholds: {THRESHOLDS}',
    '=' * 72,
    '',
    f'{"Model":<22} {"Sharpe":>7} {"CAGR":>8} {"DirAcc":>8} {"Coverage":>9} {"MaxDD":>8}  Tuned Params',
    '-' * 78,
]
for name, m in sorted_metrics:
    star = ' ★' if m['Sharpe'] >= SHARPE_THRESH else '  '
    p_str = json.dumps(TUNED[name])
    lines.append(f'{name:<22}{star} {m["Sharpe"]:>6.3f} {m["CAGR"]:>7.1f}% '
                 f'{m["DirAcc"]*100:>7.1f}% {m["Coverage"]*100:>8.1f}% {m["MaxDD"]*100:>7.1f}%  {p_str}')
lines += [
    '-' * 78,
    f'{"Buy & Hold":<22}   {bnh_sharpe:>6.3f} {bnh_cagr:>7.1f}%   {"55.0%":>7} {"100.0%":>8} {"—":>8}',
    '',
    f'★ Sharpe ≥ {SHARPE_THRESH} (strong outperformance)',
    '',
    f'Best model by Sharpe: {best_name} ({all_metrics[best_name]["Sharpe"]:.3f})',
    f'MC permutation test  (best model): p={mc_pval:.4f}'
         + (' ✓ significant' if mc_pval < 0.05 else ' ✗ not significant'),
    f'White Reality Check  (best model): p={wrc_pval:.4f}'
         + (' ✓ significant' if wrc_pval < 0.05 else ' ✗ not significant'),
    '',
    'Note: DirAcc = active-position days only (sig≠0); Coverage = fraction of days in market.',
    'Note: SelectKBest leakage fixed 2026-05-02. XGBoost retrained with k=39',
    '(all features). best_model_XGB.joblib updated.',
    '=' * 78,
]
with open(f'{RESULTS_DIR}/final_summary.txt', 'w') as f:
    f.write('\n'.join(lines))
print(f'Saved → {RESULTS_DIR}/final_summary.txt')

# ══════════════════════════════════════════════════════════════════════════════
# DONE
# ══════════════════════════════════════════════════════════════════════════════
print(f'\nAll results updated in {time.time()-t_start:.0f}s')
print(f'Files in {RESULTS_DIR}/: {sorted(os.listdir(RESULTS_DIR))}')

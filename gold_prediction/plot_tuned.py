#!/usr/bin/env python3
"""
plot_tuned.py — Redraw charts using tuned model parameters from tune_all.py output.

Reads saved_models/tuned_<name>.json for best params, re-fits each model on
trainval with those params, generates predictions, then saves:
  results/predicted_vs_actual_all_models.png  (Signal Dir Acc in subtitles)
  results/equity_curves_all_models.png        (capital growth + consensus bar)

Also prints a full summary table: optimized params + all trading metrics.
"""

import os, json, warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import joblib

from sklearn.linear_model import Ridge, Lasso
from sklearn.svm import SVR, LinearSVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import xgboost as xgb

MODELS_DIR  = 'saved_models'
RESULTS_DIR = 'results'
DATA_DIR    = 'data'
SEED        = 42
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Load metadata ─────────────────────────────────────────────────────────────
with open(f'{MODELS_DIR}/model_metadata.json') as f:
    meta = json.load(f)

FRAC_D         = meta['frac_d']
FEATURE_COLS   = meta['feature_cols']
LSTM_TS        = meta['lstm_best_timesteps']
GRU_TS         = meta['gru_best_timesteps']
BILSTM_TS      = meta['bilstm_best_timesteps']
TRAIN_END_DATE = pd.Timestamp(meta['train_end_date'])

# ── Load per-model tuned params (fall back to defaults if not tuned yet) ──────
def load_tuned(name, defaults):
    path = f'{MODELS_DIR}/tuned_{name}.json'
    if os.path.exists(path):
        with open(path) as f:
            d = json.load(f)
        return d.get('best_params', defaults)
    print(f'  [warn] No tuned params for {name}, using defaults')
    return defaults

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
}

# ══════════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING (identical to train.py)
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
    w, n, width = np.array(w[::-1]), len(series), len(w)
    out = np.full(n, np.nan)
    for i in range(width - 1, n):
        vals = series.iloc[i - width + 1: i + 1].values
        if not np.any(np.isnan(vals)): out[i] = np.dot(w, vals)
    return pd.Series(out, index=series.index)

df['frac_diff_log_Gold'] = frac_diff(df['log_Gold'], d=FRAC_D)
if HAS_COT:
    for col in ['Comm_Positions_Long_All','Comm_Positions_Short_All',
                'NonComm_Positions_Long_All','NonComm_Positions_Short_All','Open_Interest_All']:
        if col in df.columns:
            df[f'{col}_diff'] = frac_diff(df[col], d=FRAC_D)
TARGET = 'frac_diff_log_Gold'

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
    for out_col, raw_col in [('cot_norm_spec','net_speculator'),('cot_norm_comm','net_commercial')]:
        rmin = df[raw_col].rolling(252, min_periods=60).min()
        rmax = df[raw_col].rolling(252, min_periods=60).max()
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

try:
    import talib
    close = df['Gold'].values.astype(float)
    df['RSI_14']    = talib.RSI(close, 14); df['RSI_28']    = talib.RSI(close, 28)
    df['ROC_10']    = talib.ROC(close, 10); df['ROC_20']    = talib.ROC(close, 20)
    df['MOM_5']     = talib.MOM(close, 5)
    df['EMA_12']    = talib.EMA(close, 12); df['EMA_26']    = talib.EMA(close, 26)
    df['MACD_diff'] = df['EMA_12'] - df['EMA_26']
    up, mid, lo     = talib.BBANDS(close, 20)
    df['BB_width']  = (up - lo) / (mid + 1e-12)
    df['BB_pct']    = (close - lo) / (up - lo + 1e-12)
except ImportError:
    df['RSI_14']    = df['Gold'].diff().clip(lower=0).rolling(14).mean() / df['Gold'].diff().abs().rolling(14).mean()
    df['RSI_28']    = df['Gold'].diff().clip(lower=0).rolling(28).mean() / df['Gold'].diff().abs().rolling(28).mean()
    df['ROC_10']    = df['Gold'].pct_change(10) * 100; df['ROC_20'] = df['Gold'].pct_change(20) * 100
    df['MOM_5']     = df['Gold'].diff(5)
    df['EMA_12']    = df['Gold'].ewm(span=12).mean(); df['EMA_26'] = df['Gold'].ewm(span=26).mean()
    df['MACD_diff'] = df['EMA_12'] - df['EMA_26']
    bb_mid = df['Gold'].rolling(20).mean(); bb_std = df['Gold'].rolling(20).std()
    df['BB_width']  = bb_std / bb_mid
    df['BB_pct']    = (df['Gold'] - (bb_mid - 2*bb_std)) / (4 * bb_std + 1e-12)

for lag in [1, 2, 3, 5, 10]:
    df[f'target_lag{lag}'] = df[TARGET].shift(lag)
df['log_return'] = df['log_Gold'].diff()
df.dropna(inplace=True)

feat_cols  = [c for c in FEATURE_COLS if c in df.columns]
train_mask = df.index <= TRAIN_END_DATE
X_all  = df[feat_cols].values
y_all  = df[TARGET].values
dates  = df.index

X_train = X_all[train_mask];  y_train = y_all[train_mask]
X_test  = X_all[~train_mask]; y_test  = y_all[~train_mask]
dates_test   = dates[~train_mask]
log_ret_test = df['log_return'].values[~train_mask]
gold_test    = df['Gold'].values[~train_mask]

SVR_EPS = round(0.25 * float(np.std(y_train)), 6)

scaler    = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)
X_full_s  = scaler.transform(X_all)

print(f'  Test: {dates_test[0].date()} → {dates_test[-1].date()}  ({len(dates_test)} days)')
print(f'  SVR epsilon: {SVR_EPS}')

# ══════════════════════════════════════════════════════════════════════════════
# BUILD MODELS WITH TUNED PARAMS
# ══════════════════════════════════════════════════════════════════════════════
def build_sklearn(name, params):
    if name == 'Ridge':
        return Ridge(**params)
    if name == 'Lasso':
        return Lasso(**params)
    if name == 'SVR_lin':
        return LinearSVR(C=params.get('C', 100), epsilon=SVR_EPS, max_iter=10000, dual=True)
    if name == 'SVR_rbf':
        return SVR(kernel='rbf', C=params.get('C', 0.1),
                   gamma=params.get('gamma', 0.01), epsilon=SVR_EPS)
    if name == 'XGBoost':
        return xgb.XGBRegressor(random_state=SEED, verbosity=0,
                                n_estimators=int(params.get('n_estimators', 200)),
                                max_depth=int(params.get('max_depth', 2)),
                                learning_rate=params.get('learning_rate', 0.05),
                                subsample=params.get('subsample', 1.0))
    if name == 'RandomForest':
        return RandomForestRegressor(random_state=SEED,
                                     n_estimators=int(params.get('n_estimators', 100)),
                                     max_depth=params.get('max_depth') or None,
                                     min_samples_split=int(params.get('min_samples_split', 2)),
                                     min_samples_leaf=int(params.get('min_samples_leaf', 1)))
    if name == 'MLP':
        hls = params.get('hidden_layer_sizes', (64, 32))
        if isinstance(hls, list): hls = tuple(hls)
        return MLPRegressor(hidden_layer_sizes=hls,
                            alpha=params.get('alpha', 0.001),
                            learning_rate_init=params.get('learning_rate_init', 0.001),
                            max_iter=500, random_state=SEED)
    raise ValueError(f'Unknown model: {name}')

# ══════════════════════════════════════════════════════════════════════════════
# PREDICT + METRICS
# ══════════════════════════════════════════════════════════════════════════════
def rolling_zscore_signal(preds, win=5):
    ps   = pd.Series(preds)
    mean = ps.rolling(win, min_periods=1).mean()
    std  = ps.rolling(win, min_periods=1).std().fillna(1e-8)
    return np.where(((ps - mean) / std).fillna(0) >= 0, 1, -1).astype(float)

def compute_metrics(preds, log_rets, y_true):
    n   = min(len(preds), len(log_rets), len(y_true))
    sig = rolling_zscore_signal(preds[:n])
    st  = sig * np.array(log_rets[:n])
    eq  = np.exp(np.cumsum(st))
    mkt = np.sign(np.array(log_rets[:n]))
    pf_val = st[st>0].sum() / abs(st[st<0].sum()) if st[st<0].sum() != 0 else np.inf
    return dict(
        MSE    = float(mean_squared_error(y_true[:n], preds[:n])),
        CAGR   = float(eq[-1] ** (252/n) - 1),
        Sharpe = float(np.mean(st) / (np.std(st) + 1e-12) * np.sqrt(252)),
        PF     = pf_val,
        MaxDD  = float(((eq - np.maximum.accumulate(eq)) / np.maximum.accumulate(eq)).min()),
        DirAcc = float(np.mean(sig == mkt)),
        equity = eq,
        signal = sig,
        preds  = np.array(preds[:n]),
    )

def make_sequences(X, ts):
    return np.array([X[i-ts:i] for i in range(ts, len(X))])

# ── Sklearn models ────────────────────────────────────────────────────────────
print('\nFitting sklearn models with tuned params...')
SK_NAMES = ['Ridge','Lasso','SVR_lin','SVR_rbf','XGBoost','RandomForest','MLP']
all_metrics = {}

for name in SK_NAMES:
    m = build_sklearn(name, TUNED[name])
    m.fit(X_train_s, y_train)
    preds = m.predict(X_test_s)
    all_metrics[name] = compute_metrics(preds, log_ret_test, y_test)
    sr = all_metrics[name]['Sharpe']
    da = all_metrics[name]['DirAcc']
    print(f'  {name:<14} Sharpe={sr:.3f}  DirAcc={da*100:.1f}%  params={TUNED[name]}')

# ── Keras models ──────────────────────────────────────────────────────────────
print('\nFitting Keras models with tuned params...')
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Bidirectional, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import tensorflow as tf
tf.random.set_seed(SEED)

def build_keras(name, params, n_feat, ts):
    u  = int(params.get('units', 32))
    dr = params.get('dropout', 0.35)
    rd = params.get('recurrent_dropout', 0.2)
    if name == 'LSTM':
        m = Sequential([LSTM(u, input_shape=(ts, n_feat), dropout=dr,
                             recurrent_dropout=rd, kernel_regularizer=l2(0.001)), Dense(1)])
    elif name == 'GRU':
        m = Sequential([GRU(u, input_shape=(ts, n_feat), dropout=dr,
                            recurrent_dropout=rd), Dense(1)])
    elif name == 'BiLSTM':
        m = Sequential([Bidirectional(LSTM(u, recurrent_dropout=rd),
                                      input_shape=(ts, n_feat)),
                        Dropout(dr), Dense(16, activation='relu'), Dense(1)])
    m.compile(optimizer=Adam(0.001), loss='mse')
    return m

KERAS_CFG = [
    ('LSTM',   int(TUNED['LSTM'].get('timesteps', LSTM_TS)),   50, 10),
    ('GRU',    int(TUNED['GRU'].get('timesteps', GRU_TS)),    50, 10),
    ('BiLSTM', int(TUNED['BiLSTM'].get('timesteps', BILSTM_TS)), 60, 10),
]

for name, ts, epochs, patience in KERAS_CFG:
    Xtr_sq = make_sequences(X_train_s, ts)
    ytr_sq = y_train[ts:]
    Xte_sq = make_sequences(X_test_s, ts)
    # Use full history for prediction context
    seqs_full = make_sequences(X_full_s, ts)
    n_tr_seqs = train_mask.sum() - ts
    preds_test = seqs_full[n_tr_seqs:]
    n = min(len(preds_test), len(log_ret_test))

    keras_m = build_keras(name, TUNED[name], X_train.shape[1], ts)
    keras_m.fit(Xtr_sq, ytr_sq, epochs=epochs, batch_size=32, verbose=0,
                callbacks=[EarlyStopping(patience=patience, restore_best_weights=True)])
    raw_preds = keras_m.predict(seqs_full, verbose=0).flatten()[n_tr_seqs:n_tr_seqs+n]
    all_metrics[name] = compute_metrics(raw_preds, log_ret_test[:n], y_test[:n])
    sr = all_metrics[name]['Sharpe']
    da = all_metrics[name]['DirAcc']
    print(f'  {name:<14} Sharpe={sr:.3f}  DirAcc={da*100:.1f}%  ts={ts}  params={TUNED[name]}')

# Buy & Hold
bnh_eq = np.exp(np.cumsum(log_ret_test))

# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY TABLE
# ══════════════════════════════════════════════════════════════════════════════
MODEL_ORDER = ['Ridge','RandomForest','XGBoost','Lasso','MLP','SVR_lin','SVR_rbf',
               'BiLSTM','GRU','LSTM']
MODEL_ORDER = [m for m in MODEL_ORDER if m in all_metrics]

bnh_cagr   = float(bnh_eq[-1] ** (252 / len(bnh_eq)) - 1)
bnh_sharpe = float(np.mean(log_ret_test) / (np.std(log_ret_test) + 1e-12) * np.sqrt(252))
bnh_dd     = float(((bnh_eq - np.maximum.accumulate(bnh_eq)) / np.maximum.accumulate(bnh_eq)).min())

print('\n' + '='*110)
print(f'  {"Model":<14} {"Sharpe":>8} {"CAGR":>8} {"DirAcc":>8} {"MaxDD":>8} {"MSE":>10}  '
      f'{"Optimized Parameters"}')
print('  ' + '-'*108)
for name in MODEL_ORDER:
    m = all_metrics[name]
    star = ' ★' if m['Sharpe'] >= 7.5 else '  '
    params_str = json.dumps(TUNED[name], separators=(',', ':'))
    print(f'  {name:<14}{star} {m["Sharpe"]:>7.3f}  {m["CAGR"]:>7.1%}  '
          f'{m["DirAcc"]*100:>6.1f}%  {m["MaxDD"]:>7.1%}  {m["MSE"]:>10.6f}  {params_str}')
print('  ' + '-'*108)
print(f'  {"Buy & Hold":<16} {bnh_sharpe:>7.3f}  {bnh_cagr:>7.1%}  {"55.0%":>7}  '
      f'{bnh_dd:>7.1%}  {"—":>10}  (baseline)')
print('='*110)

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — EQUITY CURVES (tuned)
# ══════════════════════════════════════════════════════════════════════════════
print('\nPlotting equity curves...')

SHARPES = {n: all_metrics[n]['Sharpe'] for n in all_metrics}
strong  = sorted([n for n in MODEL_ORDER if SHARPES[n] >= 7.5],
                 key=lambda x: -SHARPES[x])
STRONG_COLORS = ['#D4AF37','#2E7D32','#1565C0','#6A1B9A','#E65100','#00838F']
color_map = {m: STRONG_COLORS[i % len(STRONG_COLORS)] for i, m in enumerate(strong)}
for m in MODEL_ORDER:
    if m not in color_map:
        s = SHARPES[m]
        color_map[m] = '#888888' if s >= 1.0 else '#aaaaaa' if s >= 0 else '#cc4444'

fig = plt.figure(figsize=(16, 11))
gs  = gridspec.GridSpec(3, 1, height_ratios=[1, 2.5, 0.8], hspace=0.08)

ax_gold = fig.add_subplot(gs[0])
ax_gold.plot(dates_test, gold_test, color='#B8860B', lw=1.5)
ax_gold.set_ylabel('Gold Price (USD)', fontsize=10)
ax_gold.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))
ax_gold.set_title(
    f'Gold Price Prediction — Capital Growth (Tuned Parameters) vs Buy & Hold\n'
    f'Test: {dates_test[0].strftime("%Y-%m-%d")} → {dates_test[-1].strftime("%Y-%m-%d")}',
    fontsize=12, fontweight='bold', pad=8)
ax_gold.tick_params(labelbottom=False)
ax_gold.grid(axis='y', alpha=0.3)

ax_eq = fig.add_subplot(gs[1], sharex=ax_gold)
ax_eq.plot(dates_test, bnh_eq, color='#B8860B', lw=2.5, ls='--',
           label=f'Buy & Hold  (CAGR {bnh_cagr:+.1%} | Sharpe {bnh_sharpe:.2f})', zorder=5)
ax_eq.axhline(1.0, color='grey', lw=0.8, ls=':')

for name in MODEL_ORDER:
    if SHARPES[name] < 1.0:
        eq   = all_metrics[name]['equity']
        dt   = dates_test[:len(eq)]
        cagr = eq[-1] ** (252/len(eq)) - 1
        da   = all_metrics[name]['DirAcc']
        label = f'{name}  (CAGR {cagr:+.0%} | Sharpe {SHARPES[name]:.2f} | DirAcc {da*100:.1f}%)'
        ax_eq.plot(dt, eq, color=color_map[name], lw=0.8, alpha=0.55, label=label)

for name in MODEL_ORDER:
    if SHARPES[name] >= 7.5:
        eq   = all_metrics[name]['equity']
        dt   = dates_test[:len(eq)]
        cagr = eq[-1] ** (252/len(eq)) - 1
        da   = all_metrics[name]['DirAcc']
        label = (f'{name}  (CAGR {cagr:+.0%} | Sharpe {SHARPES[name]:.2f} | '
                 f'DirAcc {da*100:.1f}%)')
        ax_eq.plot(dt, eq, color=color_map[name], lw=2.2, label=label, zorder=10)

# Mid-tier models (0 < Sharpe < 7.5)
for name in MODEL_ORDER:
    s = SHARPES[name]
    if 1.0 <= s < 7.5:
        eq   = all_metrics[name]['equity']
        dt   = dates_test[:len(eq)]
        cagr = eq[-1] ** (252/len(eq)) - 1
        da   = all_metrics[name]['DirAcc']
        ax_eq.plot(dt, eq, color='#888888', lw=1.2, ls='-.', alpha=0.7,
                   label=f'{name}  (CAGR {cagr:+.0%} | Sharpe {s:.2f} | DirAcc {da*100:.1f}%)',
                   zorder=6)

ax_eq.set_ylabel('Portfolio Value ($1 start)', fontsize=10)
ax_eq.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:.1f}'))
ax_eq.legend(loc='upper left', fontsize=8, framealpha=0.9)
ax_eq.tick_params(labelbottom=False)
ax_eq.grid(alpha=0.25)

ax_vote = fig.add_subplot(gs[2], sharex=ax_gold)
vote_array = np.zeros(len(dates_test))
n_vote = 0
for name in MODEL_ORDER:
    sig = all_metrics[name]['signal']
    n   = min(len(sig), len(dates_test))
    vote_array[:n] += sig[:n]
    n_vote += 1
pct_long = (vote_array + n_vote) / (2 * n_vote)
cmap_rg = LinearSegmentedColormap.from_list('rg', ['#cc3333','#dddddd','#2a9d2a'])
for i in range(len(dates_test) - 1):
    ax_vote.axvspan(dates_test[i], dates_test[i+1],
                    facecolor=cmap_rg(pct_long[i]), alpha=0.85, linewidth=0)
ax_vote.axhline(0.5, color='black', lw=0.5, ls=':')
ax_vote.set_ylim(0, 1)
ax_vote.set_yticks([0, 0.5, 1])
ax_vote.set_yticklabels(['All SHORT', '50/50', 'All LONG'], fontsize=8)
ax_vote.set_ylabel('Model\nConsensus', fontsize=9)
ax_vote.set_xlabel('Date', fontsize=10)
ax_vote.grid(axis='x', alpha=0.2)

import matplotlib.dates as mdates
ax_vote.xaxis.set_major_locator(mdates.YearLocator())
ax_vote.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
fig.align_ylabels([ax_gold, ax_eq, ax_vote])
plt.savefig(f'{RESULTS_DIR}/equity_curves_all_models.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f'  Saved → {RESULTS_DIR}/equity_curves_all_models.png')

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — PREDICTED VS ACTUAL (tuned, Signal Dir Acc)
# ══════════════════════════════════════════════════════════════════════════════
print('Plotting predicted vs actual...')

PVA_ORDER = ['Ridge','RandomForest','XGBoost','Lasso','MLP',
             'SVR_lin','SVR_rbf','BiLSTM','GRU','LSTM']
PVA_ORDER = [m for m in PVA_ORDER if m in all_metrics]

nrows = (len(PVA_ORDER) + 1) // 2
fig2, axes = plt.subplots(nrows, 2, figsize=(16, 3.5 * nrows))
axes = axes.flatten()
fig2.suptitle(
    f'Predicted vs Actual — Frac-Diff Log Gold Returns  (Tuned Parameters)\n'
    f'Test: {dates_test[0].strftime("%Y-%m-%d")} → {dates_test[-1].strftime("%Y-%m-%d")}',
    fontsize=13, fontweight='bold')

for idx, name in enumerate(PVA_ORDER):
    ax  = axes[idx]
    m   = all_metrics[name]
    p   = m['preds']
    n   = len(p)
    dt  = dates_test[:n]
    star = ' ★' if m['Sharpe'] >= 7.5 else ''
    params_str = ', '.join(f'{k}={v}' for k, v in TUNED[name].items())
    title = (f'{name}{star} | MSE={m["MSE"]:.5f} | Sig Dir Acc={m["DirAcc"]*100:.1f}% | '
             f'Sharpe={m["Sharpe"]:.2f}\n[{params_str}]')
    ax.plot(dt, y_test[:n], color='#2c7bb6', lw=0.7, alpha=0.9, label='Actual')
    ax.plot(dt, p,          color='#d7191c', lw=0.7, alpha=0.8, ls='--', label='Predicted')
    ax.set_title(title, fontsize=7.5)
    ax.tick_params(labelsize=7)
    ax.set_ylabel('Frac-Diff\nReturn', fontsize=7)
    if idx == 0:
        ax.legend(fontsize=7, loc='upper left')
    ax.grid(alpha=0.2)

for idx in range(len(PVA_ORDER), len(axes)):
    axes[idx].set_visible(False)

plt.tight_layout()
plt.savefig(f'{RESULTS_DIR}/predicted_vs_actual_all_models.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f'  Saved → {RESULTS_DIR}/predicted_vs_actual_all_models.png')
print('\nDone.')

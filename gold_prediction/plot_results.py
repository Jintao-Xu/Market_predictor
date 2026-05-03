#!/usr/bin/env python3
"""
plot_results.py — Visualize equity curves and trading signals for all models.

Produces two figures saved to results/:
  1. equity_curves_all_models.png  — capital growth of every model vs buy & hold
  2. signal_timeline.png           — long/short action strips for every model + gold price
"""

import os, json, warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import joblib
import xgboost as xgb
from sklearn.linear_model import Ridge, Lasso
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

RESULTS_DIR = 'results'
MODELS_DIR  = 'saved_models'
DATA_DIR    = 'data'
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
BEST_PARAMS    = meta['best_params']
SEED           = 42

# ══════════════════════════════════════════════════════════════════════════════
# 1. FEATURE ENGINEERING (identical to train.py)
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
    df['RSI_14']    = df['Gold'].diff().clip(lower=0).rolling(14).mean() / df['Gold'].diff().abs().rolling(14).mean()
    df['RSI_28']    = df['Gold'].diff().clip(lower=0).rolling(28).mean() / df['Gold'].diff().abs().rolling(28).mean()
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

feat_cols = [c for c in FEATURE_COLS if c in df.columns]
train_mask = df.index <= TRAIN_END_DATE
X_all  = df[feat_cols].values
y_all  = df[TARGET].values
dates  = df.index

X_train = X_all[train_mask]
y_train = y_all[train_mask]
X_test  = X_all[~train_mask]
y_test  = y_all[~train_mask]
dates_test    = dates[~train_mask]
log_ret_test  = df['log_return'].values[~train_mask]
gold_test     = df['Gold'].values[~train_mask]

scaler   = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

print(f'  Test period: {dates_test[0].date()} → {dates_test[-1].date()}  ({len(dates_test)} days)')

# ══════════════════════════════════════════════════════════════════════════════
# 2. FIT SKLEARN MODELS + LOAD KERAS
# ══════════════════════════════════════════════════════════════════════════════
print('Fitting sklearn models...')

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
for name, m in sk_models.items():
    m.fit(X_train_s, y_train)
    print(f'  {name} ✓')

from tensorflow.keras.models import load_model as keras_load
print('Loading Keras models...')
keras_cfg = {
    'LSTM':   (keras_load(f'{MODELS_DIR}/model_LSTM.keras'),   LSTM_TS),
    'GRU':    (keras_load(f'{MODELS_DIR}/model_GRU.keras'),    GRU_TS),
    'BiLSTM': (keras_load(f'{MODELS_DIR}/model_BiLSTM.keras'), BILSTM_TS),
}
print('  LSTM / GRU / BiLSTM ✓')

# ══════════════════════════════════════════════════════════════════════════════
# 3. PREDICT + BUILD EQUITY CURVES
# ══════════════════════════════════════════════════════════════════════════════
print('Generating predictions and equity curves...')

def rolling_zscore_signal(preds, win=5):
    ps   = pd.Series(preds)
    mean = ps.rolling(win, min_periods=1).mean()
    std  = ps.rolling(win, min_periods=1).std().fillna(1e-8)
    z    = (ps - mean) / std
    z    = z.fillna(0)
    return np.where(z >= 0, 1, -1).astype(float)

def equity_curve(signal, log_rets):
    strat = signal * log_rets
    return np.exp(np.cumsum(strat))

def make_sequences(X, ts):
    return np.array([X[i - ts:i] for i in range(ts, len(X))])

# sklearn: predict on full test set
all_signals  = {}
all_equities = {}
all_dates    = {}

# Need extra history for sequence models — stack train tail + test
max_ts = max(LSTM_TS, GRU_TS, BILSTM_TS)
X_stack   = np.vstack([X_train_s[-max_ts:], X_test_s])
date_stack = np.concatenate([dates[train_mask][-max_ts:], dates_test])

for name, model in sk_models.items():
    preds  = model.predict(X_test_s)
    sig    = rolling_zscore_signal(preds)
    eq     = equity_curve(sig, log_ret_test)
    all_signals[name]  = sig
    all_equities[name] = eq
    all_dates[name]    = dates_test

# Keras
X_full_s = scaler.transform(X_all)
for name, (model, ts) in keras_cfg.items():
    # Predict over full history so z-score has context, then slice test
    seqs    = make_sequences(X_full_s, ts)
    preds_f = model.predict(seqs, verbose=0).flatten()
    # seqs[i] corresponds to df row (ts + i), so first test index in seqs:
    n_train_seqs = train_mask.sum() - ts
    preds_test   = preds_f[n_train_seqs:]
    # trim to match test length
    n = min(len(preds_test), len(log_ret_test))
    sig  = rolling_zscore_signal(preds_test[:n])
    eq   = equity_curve(sig, log_ret_test[:n])
    all_signals[name]  = sig
    all_equities[name] = eq
    all_dates[name]    = dates_test[:n]

# Buy & Hold
bnh_eq    = np.exp(np.cumsum(log_ret_test))
bnh_dates = dates_test

print(f'  Done. {len(all_signals)} models computed.')

# ══════════════════════════════════════════════════════════════════════════════
# 4. FIGURE 1: EQUITY CURVES
# ══════════════════════════════════════════════════════════════════════════════
print('Plotting Figure 1: Equity curves...')

# Model display order and style
MODEL_ORDER = ['Ridge','RandomForest','XGBoost','Lasso','ARIMAX',
               'BiLSTM','GRU','LSTM','MLP','SVR_lin','SVR_rbf']
MODEL_ORDER = [m for m in MODEL_ORDER if m in all_equities]

SHARPES = {
    'Ridge':8.72,'RandomForest':7.88,'XGBoost':7.80,'Lasso':7.51,
    'ARIMAX':1.34,'BiLSTM':0.74,'MLP':0.23,'GRU':0.14,'LSTM':0.03,
    'SVR_lin':-1.26,'SVR_rbf':-1.26,
}

# Color palette: top models gold/green tones, underperformers grey/red
def model_color(name):
    s = SHARPES.get(name, 0)
    if s >= 7.5:   return None          # assigned from strong palette below
    if s >= 1.0:   return '#888888'
    if s >= 0:     return '#aaaaaa'
    return '#cc4444'

STRONG_COLORS = ['#D4AF37','#2E7D32','#1565C0','#6A1B9A']   # gold, dark green, dark blue, purple
strong_models = [m for m in MODEL_ORDER if SHARPES.get(m, 0) >= 7.5]
color_map = {}
for i, m in enumerate(strong_models):
    color_map[m] = STRONG_COLORS[i % len(STRONG_COLORS)]
for m in MODEL_ORDER:
    if m not in color_map:
        color_map[m] = model_color(m)

fig = plt.figure(figsize=(16, 11))
gs  = gridspec.GridSpec(3, 1, height_ratios=[1, 2.5, 0.8], hspace=0.08)

# ── Top panel: Gold price ──────────────────────────────────────────────────
ax_gold = fig.add_subplot(gs[0])
ax_gold.plot(bnh_dates, gold_test, color='#B8860B', lw=1.5, label='Gold Price (GC=F)')
ax_gold.set_ylabel('Gold Price (USD)', fontsize=10)
ax_gold.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))
ax_gold.legend(loc='upper left', fontsize=9)
ax_gold.set_title('Gold Price Prediction — Capital Growth vs Buy & Hold  '
                  f'(Test: {dates_test[0].strftime("%Y-%m-%d")} → {dates_test[-1].strftime("%Y-%m-%d")})',
                  fontsize=13, fontweight='bold', pad=10)
ax_gold.tick_params(labelbottom=False)
ax_gold.grid(axis='y', alpha=0.3)

# ── Middle panel: Equity curves ────────────────────────────────────────────
ax_eq = fig.add_subplot(gs[1], sharex=ax_gold)

# B&H first (background reference)
ax_eq.plot(bnh_dates, bnh_eq, color='#B8860B', lw=2.5, ls='--',
           label=f'Buy & Hold  (CAGR +24.2%)', zorder=5)
ax_eq.axhline(1.0, color='grey', lw=0.8, ls=':')

# Underperforming models (behind)
for name in reversed(MODEL_ORDER):
    if SHARPES.get(name, 0) < 1.0:
        eq, dt = all_equities[name], all_dates[name]
        lw = 0.8
        ax_eq.plot(dt, eq, color=color_map[name], lw=lw, alpha=0.5)

# Strong models (on top)
for name in MODEL_ORDER:
    if SHARPES.get(name, 0) >= 7.5:
        eq, dt = all_equities[name], all_dates[name]
        final  = eq[-1]
        cagr   = final ** (252 / len(eq)) - 1
        sr     = SHARPES[name]
        label  = f'{name}  (CAGR +{cagr:.0%} | Sharpe {sr:.2f})'
        ax_eq.plot(dt, eq, color=color_map[name], lw=2.2,
                   label=label, zorder=10)

# ARIMAX separately
if 'ARIMAX' in all_equities:
    eq, dt = all_equities['ARIMAX'], all_dates['ARIMAX']
    cagr = eq[-1] ** (252 / len(eq)) - 1
    ax_eq.plot(dt, eq, color='#888888', lw=1.2, ls='-.', alpha=0.7,
               label=f'ARIMAX  (CAGR +{cagr:.0%})', zorder=6)

ax_eq.set_ylabel('Portfolio Value ($1 → ...)', fontsize=10)
ax_eq.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:.1f}'))
ax_eq.legend(loc='upper left', fontsize=8.5, framealpha=0.9)
ax_eq.tick_params(labelbottom=False)
ax_eq.grid(alpha=0.25)

# ── Bottom panel: consensus bar (how many models are LONG each day) ────────
ax_vote = fig.add_subplot(gs[2], sharex=ax_gold)

# Align all signals to the shortest common date range
common_dates = dates_test
vote_array = np.zeros(len(common_dates))
n_models   = 0
for name in MODEL_ORDER:
    sig = all_signals[name]
    dt  = all_dates[name]
    # Map onto common_dates
    aligned = np.zeros(len(common_dates))
    for i, d in enumerate(common_dates):
        idx = np.searchsorted(dt, d)
        if idx < len(sig):
            aligned[i] = sig[idx]
    vote_array += aligned
    n_models   += 1

pct_long = (vote_array + n_models) / (2 * n_models)   # normalise to [0,1]

# Color bar: green=mostly long, red=mostly short
cmap_rg = LinearSegmentedColormap.from_list('rg', ['#cc3333','#dddddd','#2a9d2a'])
for i in range(len(common_dates) - 1):
    ax_vote.axvspan(common_dates[i], common_dates[i+1],
                    facecolor=cmap_rg(pct_long[i]), alpha=0.85, linewidth=0)

ax_vote.axhline(0.5, color='black', lw=0.5, ls=':')
ax_vote.set_ylim(0, 1)
ax_vote.set_yticks([0, 0.5, 1])
ax_vote.set_yticklabels(['All SHORT', '50/50', 'All LONG'], fontsize=8)
ax_vote.set_ylabel('Model\nConsensus', fontsize=9)
ax_vote.set_xlabel('Date', fontsize=10)
ax_vote.grid(axis='x', alpha=0.2)

plt.setp(ax_vote.get_xticklabels(), rotation=0, fontsize=9)
fig.align_ylabels([ax_gold, ax_eq, ax_vote])
plt.savefig(f'{RESULTS_DIR}/equity_curves_all_models.png', dpi=150,
            bbox_inches='tight', facecolor='white')
plt.close()
print(f'  Saved → {RESULTS_DIR}/equity_curves_all_models.png')

# ══════════════════════════════════════════════════════════════════════════════
# 5. FIGURE 2: SIGNAL TIMELINE (per-model action strips)
# ══════════════════════════════════════════════════════════════════════════════
print('Plotting Figure 2: Signal timeline...')

DISPLAY_ORDER = ['Ridge','RandomForest','XGBoost','Lasso',
                 'BiLSTM','GRU','LSTM','MLP','SVR_lin','SVR_rbf']
DISPLAY_ORDER = [m for m in DISPLAY_ORDER if m in all_signals]

n_models = len(DISPLAY_ORDER)
fig2, axes = plt.subplots(n_models + 2, 1,
                          figsize=(16, n_models * 0.85 + 4.5),
                          gridspec_kw={'height_ratios': [2.5] + [1]*n_models + [0.6]})
fig2.subplots_adjust(hspace=0.06, left=0.12, right=0.97, top=0.95, bottom=0.05)
fig2.suptitle('Gold Trading Signals — Each Model\'s Long / Short Actions Over Time\n'
              f'Test Period: {dates_test[0].strftime("%Y-%m-%d")} → {dates_test[-1].strftime("%Y-%m-%d")}',
              fontsize=13, fontweight='bold')

# ── Gold price panel ──────────────────────────────────────────────────────────
ax0 = axes[0]
ax0.plot(bnh_dates, gold_test, color='#B8860B', lw=1.6)
ax0.fill_between(bnh_dates, gold_test, gold_test.min(), alpha=0.08, color='#B8860B')
ax0.set_ylabel('Gold\nPrice ($)', fontsize=9)
ax0.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))
ax0.tick_params(labelbottom=False)
ax0.grid(axis='y', alpha=0.3)
ax0.set_xlim(bnh_dates[0], bnh_dates[-1])

# Annotate price at start and end
ax0.annotate(f'${gold_test[0]:,.0f}', xy=(bnh_dates[0], gold_test[0]),
             xytext=(8, 6), textcoords='offset points', fontsize=8, color='#7a5c00')
ax0.annotate(f'${gold_test[-1]:,.0f}', xy=(bnh_dates[-1], gold_test[-1]),
             xytext=(-50, 6), textcoords='offset points', fontsize=8, color='#7a5c00')

# ── Per-model signal strips ───────────────────────────────────────────────────
for row, name in enumerate(DISPLAY_ORDER):
    ax = axes[row + 1]
    sig = all_signals[name]
    dt  = all_dates[name]

    sharpe = SHARPES.get(name, 0)
    star   = ' ★' if sharpe >= 7.5 else ''
    label  = f'{name}{star}\nSharpe {sharpe:.2f}'

    # Draw colored spans: green=LONG, red=SHORT
    changes = np.where(np.diff(sig, prepend=sig[0]))[0]
    boundaries = np.concatenate([changes, [len(sig)]])
    prev = 0
    for b in boundaries[1:]:
        if prev >= len(dt) or b > len(dt): break
        color = '#2ecc71' if sig[prev] > 0 else '#e74c3c'
        alpha = 0.85
        ax.axvspan(dt[prev], dt[min(b, len(dt)-1)],
                   facecolor=color, alpha=alpha, linewidth=0)
        prev = b

    # Equity curve overlaid as thin line
    eq = all_equities[name]
    eq_dates = all_dates[name]
    ax2 = ax.twinx()
    eq_color = color_map.get(name, '#444444')
    ax2.plot(eq_dates, eq, color=eq_color, lw=1.2, alpha=0.8)
    ax2.set_ylim(0, max(eq) * 1.1)
    final_val = eq[-1]
    cagr = final_val ** (252 / len(eq)) - 1
    ax2.set_ylabel(f'×{final_val:.1f}', fontsize=7.5, color=eq_color, rotation=0, labelpad=22)
    ax2.tick_params(right=False, labelright=False)

    ax.set_xlim(bnh_dates[0], bnh_dates[-1])
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_ylabel(label, fontsize=8, rotation=0, ha='right', va='center', labelpad=5)
    ax.tick_params(labelbottom=False)

    # Add CAGR annotation on right edge
    ax.text(1.001, 0.5, f'+{cagr:.0%}', transform=ax.transAxes,
            fontsize=7.5, va='center', color=eq_color,
            fontweight='bold' if sharpe >= 7.5 else 'normal')

# ── Legend row ────────────────────────────────────────────────────────────────
ax_leg = axes[-1]
ax_leg.set_visible(False)
long_patch  = mpatches.Patch(color='#2ecc71', alpha=0.85, label='LONG (Buy)')
short_patch = mpatches.Patch(color='#e74c3c', alpha=0.85, label='SHORT (Sell)')
star_note   = mpatches.Patch(color='none',    label='★ = Sharpe ≥ 7.5 (top model)')
fig2.legend(handles=[long_patch, short_patch, star_note],
            loc='lower center', ncol=3, fontsize=9,
            bbox_to_anchor=(0.5, 0.005), framealpha=0.9)

# Date ticks on last visible signal axis
axes[-2].tick_params(labelbottom=True, rotation=0)
import matplotlib.dates as mdates
axes[-2].xaxis.set_major_locator(mdates.YearLocator())
axes[-2].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

plt.savefig(f'{RESULTS_DIR}/signal_timeline.png', dpi=150,
            bbox_inches='tight', facecolor='white')
plt.close()
print(f'  Saved → {RESULTS_DIR}/signal_timeline.png')

print('\nDone. Two charts saved:')
print(f'  {RESULTS_DIR}/equity_curves_all_models.png')
print(f'  {RESULTS_DIR}/signal_timeline.png')

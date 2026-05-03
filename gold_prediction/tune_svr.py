#!/usr/bin/env python3
"""
tune_svr.py — Standalone SVR hyperparameter tuning.
Tunes C for SVR_lin (LinearSVR) and C + gamma for SVR_rbf.
Epsilon is fixed at 0.25 * target_std (calibrated to frac-diff series scale).
"""

import os, json, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, Lasso
from sklearn.svm import SVR, LinearSVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

DATA_DIR   = 'data'
MODELS_DIR = 'saved_models'

# ── Load metadata for feature list and split date ────────────────────────────
with open(f'{MODELS_DIR}/model_metadata.json') as f:
    meta = json.load(f)

FRAC_D         = meta['frac_d']
FEATURE_COLS   = meta['feature_cols']
TRAIN_END_DATE = pd.Timestamp(meta['train_end_date'])

# ── Rebuild features (same logic as train.py) ─────────────────────────────────
print('Loading and building features...')

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
    bb_mid = df['Gold'].rolling(20).mean()
    bb_std = df['Gold'].rolling(20).std()
    df['BB_width']  = bb_std / bb_mid
    df['BB_pct']    = (df['Gold'] - (bb_mid - 2*bb_std)) / (4 * bb_std + 1e-12)

for lag in [1, 2, 3, 5, 10]:
    df[f'target_lag{lag}'] = df[TARGET].shift(lag)
df.dropna(inplace=True)

feat_cols  = [c for c in FEATURE_COLS if c in df.columns]
train_mask = df.index <= TRAIN_END_DATE
X_trainval = df[feat_cols].values[train_mask]
y_trainval = df[TARGET].values[train_mask]

SVR_EPSILON = round(0.25 * float(np.std(y_trainval)), 6)
print(f'  target std={np.std(y_trainval):.6f}  SVR_EPSILON={SVR_EPSILON}  (0.25 × std)')
print(f'  trainval rows: {len(X_trainval)}')

# ── Grid search ───────────────────────────────────────────────────────────────
tscv = TimeSeriesSplit(n_splits=5)

# SVR_lin: tune C
print('\n── SVR_lin (LinearSVR) — tuning C ───────────────────────────────────────')
gs_lin = GridSearchCV(
    Pipeline([('sc', StandardScaler()),
              ('m',  LinearSVR(epsilon=SVR_EPSILON, max_iter=10000, dual=True))]),
    param_grid={'m__C': [0.1, 1, 10, 100, 1000]},
    cv=tscv, scoring='neg_mean_squared_error', n_jobs=-1, refit=True, verbose=2)
gs_lin.fit(X_trainval, y_trainval)

print(f'\n  Results:')
for params, mean, std in zip(gs_lin.cv_results_['params'],
                              gs_lin.cv_results_['mean_test_score'],
                              gs_lin.cv_results_['std_test_score']):
    print(f'    C={params["m__C"]:<8}  CV MSE={-mean:.8f} ± {std:.8f}')
print(f'\n  Best: C={gs_lin.best_params_["m__C"]}  CV MSE={-gs_lin.best_score_:.8f}')

# SVR_rbf: tune C and gamma
print('\n── SVR_rbf — tuning C and gamma ─────────────────────────────────────────')
gs_rbf = GridSearchCV(
    Pipeline([('sc', StandardScaler()),
              ('m',  SVR(kernel='rbf', epsilon=SVR_EPSILON))]),
    param_grid={'m__C':     [0.01, 0.1, 1, 10, 100],
                'm__gamma': ['scale', 0.001, 0.01, 0.1]},
    cv=tscv, scoring='neg_mean_squared_error', n_jobs=-1, refit=True, verbose=1)
gs_rbf.fit(X_trainval, y_trainval)

print(f'\n  Results (sorted by CV MSE):')
rbf_res = sorted(zip(gs_rbf.cv_results_['params'],
                     gs_rbf.cv_results_['mean_test_score']),
                 key=lambda x: -x[1])
for params, mean in rbf_res[:10]:
    print(f'    C={params["m__C"]:<6}  gamma={str(params["m__gamma"]):<8}  CV MSE={-mean:.8f}')
print(f'\n  Best: C={gs_rbf.best_params_["m__C"]}  gamma={gs_rbf.best_params_["m__gamma"]}  '
      f'CV MSE={-gs_rbf.best_score_:.8f}')

# ── Summary ───────────────────────────────────────────────────────────────────
print('\n' + '='*60)
print('  SVR TUNING RESULTS')
print('='*60)
print(f'  epsilon (shared)   = {SVR_EPSILON}  (0.25 × target_std)')
print(f'  SVR_lin best C     = {gs_lin.best_params_["m__C"]}')
print(f'  SVR_rbf best C     = {gs_rbf.best_params_["m__C"]}')
print(f'  SVR_rbf best gamma = {gs_rbf.best_params_["m__gamma"]}')
print('='*60)
print('\nPaste these into train.py ALL_SK:')
print(f'  SVR_lin: LinearSVR(C={gs_lin.best_params_["m__C"]}, epsilon=SVR_EPSILON, max_iter=10000, dual=True)')
print(f'  SVR_rbf: SVR(kernel="rbf", C={gs_rbf.best_params_["m__C"]}, '
      f'epsilon=SVR_EPSILON, gamma={repr(gs_rbf.best_params_["m__gamma"])})')

# ── Save results to metadata ──────────────────────────────────────────────────
meta['svr_epsilon']        = float(SVR_EPSILON)
meta['svr_lin_best_C']     = float(gs_lin.best_params_['m__C'])
meta['svr_rbf_best_C']     = float(gs_rbf.best_params_['m__C'])
meta['svr_rbf_best_gamma'] = str(gs_rbf.best_params_['m__gamma'])
with open(f'{MODELS_DIR}/model_metadata.json', 'w') as f:
    json.dump(meta, f, indent=2)
print(f'\n  Saved to {MODELS_DIR}/model_metadata.json')

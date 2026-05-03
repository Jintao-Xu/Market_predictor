#!/usr/bin/env python3
"""
tune_model.py — Train and fine-tune a single model.

Usage:
    python tune_model.py --model Ridge
    python tune_model.py --model SVR_rbf
    python tune_model.py --model XGBoost
    python tune_model.py --model LSTM
    python tune_model.py --help-params        # show all model parameter tables

Each model has a default search grid. You can override it with --param:
    python tune_model.py --model SVR_rbf --param C=[0.1,1,10] --param gamma=[0.01,0.1]
    python tune_model.py --model Ridge   --param alpha=[0.001,0.01,0.1,1,10]
    python tune_model.py --model XGBoost --param n_estimators=[100,200,300] --param max_depth=[2,3,4]
"""

import os, sys, json, time, argparse, warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
from itertools import product as iterproduct
from sklearn.linear_model import Ridge, Lasso
from sklearn.svm import SVR, LinearSVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
import xgboost as xgb

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

from sklearn.linear_model import ElasticNet

DATA_DIR   = 'data'
MODELS_DIR = 'saved_models'
SEED       = 42
N_JOBS     = -1
ZSCORE_WIN       = 10   # default fallback; tunable via OOF CV
SIGNAL_THRESHOLD = 0.0  # default fallback; tunable via OOF CV

# ══════════════════════════════════════════════════════════════════════════════
# PARAMETER CATALOGUE
# Each entry: { 'param_name': [candidate_values], ... }
# ══════════════════════════════════════════════════════════════════════════════
PARAM_CATALOGUE = {
    'Ridge': {
        'description': 'L2 linear regression. alpha controls regularisation strength.',
        'type': 'sklearn',
        'fixed': {},
        'grid': {
            'alpha':            [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 500.0, 1000.0],
            'zscore_win':       [5, 7, 10, 15, 20],
            'signal_threshold': [0.0, 0.2, 0.3, 0.5, 0.75],
        },
    },
    'Lasso': {
        'description': 'L1 linear regression (sparse). alpha controls sparsity.',
        'type': 'sklearn',
        'fixed': {},
        'grid': {
            'alpha':            [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0],
            'zscore_win':       [5, 7, 10, 15, 20],
            'signal_threshold': [0.0, 0.2, 0.3, 0.5, 0.75],
        },
    },
    'SVR_lin': {
        'description': 'LinearSVR (liblinear, O(n)). epsilon auto-set to 0.25×target_std.',
        'type': 'sklearn_pipeline',
        'fixed': {'epsilon': 'SVR_EPSILON', 'max_iter': 10000, 'dual': True},
        'grid': {
            'C':                [0.01, 0.1, 1, 10, 100, 1000, 5000, 10000],
            'zscore_win':       [5, 7, 10, 15, 20],
            'signal_threshold': [0.0, 0.2, 0.3, 0.5, 0.75],
        },
    },
    'SVR_rbf': {
        'description': 'RBF kernel SVR (libsvm). epsilon auto-set to 0.25×target_std.',
        'type': 'sklearn_pipeline',
        'fixed': {'epsilon': 'SVR_EPSILON', 'kernel': 'rbf'},
        'grid': {
            'C':                [0.001, 0.01, 0.1, 1, 10, 100],
            'gamma':            [0.0001, 0.001, 0.01, 0.1, 'scale'],
            'zscore_win':       [5, 7, 10, 15, 20],
            'signal_threshold': [0.0, 0.2, 0.3, 0.5, 0.75],
        },
    },
    'XGBoost': {
        'description': 'Gradient boosted trees. Key params: depth (complexity) and lr (speed).',
        'type': 'sklearn',
        'fixed': {'random_state': SEED, 'verbosity': 0},
        'grid': {
            'n_estimators':     [50, 100, 200, 300],
            'max_depth':        [2, 3, 4, 5],
            'learning_rate':    [0.01, 0.05, 0.1, 0.2],
            'subsample':        [0.8, 1.0],
            'zscore_win':       [5, 7, 10, 15, 20],
            'signal_threshold': [0.0, 0.2, 0.3, 0.5, 0.75],
        },
    },
    'RandomForest': {
        'description': 'Bagged decision trees. More trees = better but slower.',
        'type': 'sklearn',
        'fixed': {'random_state': SEED},
        'grid': {
            'n_estimators':      [50, 100, 200, 300],
            'max_depth':         [3, 5, 10, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf':  [1, 2, 4],
            'zscore_win':        [5, 7, 10, 15, 20],
            'signal_threshold':  [0.0, 0.2, 0.3, 0.5, 0.75],
        },
    },
    'MLP': {
        'description': 'Multi-layer perceptron. hidden_layer_sizes = (layer1, layer2, ...).',
        'type': 'sklearn',
        'fixed': {'max_iter': 500, 'random_state': SEED},
        'grid': {
            'hidden_layer_sizes': [(32,), (64,), (128,), (64, 32), (128, 64), (128, 64, 32), (256, 128, 64)],
            'alpha':              [0.0001, 0.001, 0.01],
            'learning_rate_init': [0.001, 0.01],
            'zscore_win':         [5, 7, 10, 15, 20],
            'signal_threshold':   [0.0, 0.2, 0.3, 0.5, 0.75],
        },
    },
    'LSTM': {
        'description': 'LSTM RNN. timesteps = look-back window. recurrent_dropout adds regularisation.',
        'type': 'keras',
        'fixed': {'optimizer': 'Adam(0.001)', 'loss': 'mse', 'epochs': 50, 'patience': 10},
        'grid': {
            'timesteps':         [5, 7, 10, 15],
            'units':             [16, 32, 64],
            'dropout':           [0.2, 0.35, 0.5],
            'recurrent_dropout': [0.0, 0.2],
            'zscore_win':        [5, 7, 10, 15, 20],
            'signal_threshold':  [0.0, 0.2, 0.3, 0.5, 0.75],
        },
    },
    'GRU': {
        'description': 'Gated Recurrent Unit. Lighter than LSTM, often similar performance.',
        'type': 'keras',
        'fixed': {'optimizer': 'Adam(0.001)', 'loss': 'mse', 'epochs': 50, 'patience': 10},
        'grid': {
            'timesteps':         [5, 7, 10, 15],
            'units':             [16, 32, 64],
            'dropout':           [0.2, 0.3, 0.4],
            'recurrent_dropout': [0.0, 0.2],
            'zscore_win':        [5, 7, 10, 15, 20],
            'signal_threshold':  [0.0, 0.2, 0.3, 0.5, 0.75],
        },
    },
    'BiLSTM': {
        'description': 'Bidirectional LSTM. Captures both past and future context in sequence.',
        'type': 'keras',
        'fixed': {'optimizer': 'Adam(0.001)', 'loss': 'mse', 'epochs': 60, 'patience': 10},
        'grid': {
            'timesteps':         [5, 7, 10, 15],
            'units':             [16, 32, 64],
            'dropout':           [0.3, 0.4, 0.5],
            'recurrent_dropout': [0.0, 0.2],
            'zscore_win':        [5, 7, 10, 15, 20],
            'signal_threshold':  [0.0, 0.2, 0.3, 0.5, 0.75],
        },
    },
    'LightGBM': {
        'description': 'LightGBM gradient boosting. Leaf-wise growth, faster than XGBoost on tabular data.',
        'type': 'sklearn',
        'fixed': {'random_state': SEED, 'verbose': -1},
        'grid': {
            'n_estimators':     [100, 200, 300, 500],
            'num_leaves':       [7, 15, 31, 63],
            'learning_rate':    [0.005, 0.01, 0.05, 0.1],
            'subsample':        [0.8, 1.0],
            'zscore_win':       [5, 7, 10, 15, 20],
            'signal_threshold': [0.0, 0.2, 0.3, 0.5, 0.75],
        },
    },
    'ElasticNet': {
        'description': 'Elastic Net (L1+L2 regularisation). l1_ratio=0 → Ridge, l1_ratio=1 → Lasso.',
        'type': 'sklearn',
        'fixed': {'max_iter': 5000},
        'grid': {
            'alpha':            [0.0001, 0.001, 0.01, 0.1, 1.0],
            'l1_ratio':         [0.1, 0.3, 0.5, 0.7, 0.9],
            'zscore_win':       [5, 7, 10, 15, 20],
            'signal_threshold': [0.0, 0.2, 0.3, 0.5, 0.75],
        },
    },
}

# ══════════════════════════════════════════════════════════════════════════════
# HELP: print parameter table
# ══════════════════════════════════════════════════════════════════════════════
def print_param_report():
    print('\n' + '='*70)
    print('  MODEL PARAMETER CATALOGUE')
    print('='*70)
    for name, cfg in PARAM_CATALOGUE.items():
        print(f'\n  {name}  [{cfg["type"]}]')
        print(f'  {cfg["description"]}')
        if cfg['fixed']:
            fixed_str = ', '.join(f'{k}={v}' for k, v in cfg['fixed'].items())
            print(f'  Fixed:  {fixed_str}')
        print(f'  Tunable parameters (default grid):')
        for param, vals in cfg['grid'].items():
            print(f'    {param:<25} {vals}')
    print('\n' + '='*70)
    print('  Override any parameter grid with --param name=[v1,v2,v3]')
    print('  Example: python tune_model.py --model SVR_rbf '
          '--param C=[0.1,1,10] --param gamma=[0.01,0.1]')
    print('='*70 + '\n')

# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING (shared setup)
# ══════════════════════════════════════════════════════════════════════════════
def load_data():
    with open(f'{MODELS_DIR}/model_metadata.json') as f:
        meta = json.load(f)

    FRAC_D         = meta['frac_d']
    FEATURE_COLS   = meta['feature_cols']
    TRAIN_END_DATE = pd.Timestamp(meta['train_end_date'])

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

    feat_cols  = [c for c in FEATURE_COLS if c in df.columns]
    train_mask = df.index <= TRAIN_END_DATE
    test_mask  = ~train_mask

    X_tv = df[feat_cols].values[train_mask]
    y_tv = df[TARGET].values[train_mask]
    X_te = df[feat_cols].values[test_mask]
    y_te = df[TARGET].values[test_mask]
    lr_tv = df['log_return'].values[train_mask]
    lr_te = df['log_return'].values[test_mask]
    dates_te = df.index[test_mask]

    SVR_EPS = round(0.25 * float(np.std(y_tv)), 6)
    print(f'  trainval: {len(X_tv)} rows | test: {len(X_te)} rows | '
          f'features: {len(feat_cols)} | SVR_epsilon: {SVR_EPS}')
    return X_tv, y_tv, X_te, y_te, lr_tv, lr_te, dates_te, feat_cols, SVR_EPS, meta

# ══════════════════════════════════════════════════════════════════════════════
# TRADING METRICS (signal dir acc)
# ══════════════════════════════════════════════════════════════════════════════
def trading_metrics(preds, log_rets, y_true, zscore_win=None, threshold=None):
    if zscore_win is None:
        zscore_win = ZSCORE_WIN
    if threshold is None:
        threshold = SIGNAL_THRESHOLD
    # Ensure numpy arrays so [-1] indexing always works (pandas Series would raise KeyError: -1)
    preds    = np.asarray(preds)
    log_rets = np.asarray(log_rets)
    y_true   = np.asarray(y_true)
    n   = min(len(preds), len(log_rets), len(y_true))
    ps  = pd.Series(preds[:n])
    z   = (ps - ps.rolling(zscore_win, min_periods=1).mean()) / ps.rolling(zscore_win, min_periods=1).std().fillna(1e-8)
    sig = np.where(z >  threshold,  1.0,
          np.where(z < -threshold, -1.0, 0.0))
    lr  = log_rets[:n]
    st  = sig * lr
    eq  = np.exp(np.cumsum(st))
    mkt = np.sign(lr)
    active = sig != 0
    return {
        'MSE':      float(mean_squared_error(y_true[:n], preds[:n])),
        'CAGR':     float(eq[-1] ** (252/n) - 1),
        'Sharpe':   float(np.mean(st) / (np.std(st) + 1e-12) * np.sqrt(252)),
        'PF':       float(st[st>0].sum() / abs(st[st<0].sum())) if st[st<0].sum() != 0 else np.inf,
        'MaxDD':    float(((eq - np.maximum.accumulate(eq)) / np.maximum.accumulate(eq)).min()),
        'DirAcc':   float(np.mean(sig[active] == mkt[active])) if active.any() else 0.0,
        'Coverage': float(np.mean(active)),
        'PredStd':  float(np.std(preds[:n])),
    }

# ══════════════════════════════════════════════════════════════════════════════
# SKLEARN TUNING
# ══════════════════════════════════════════════════════════════════════════════
def tune_sklearn(model_name, grid, X_tv, y_tv, X_te, y_te, lr_tv, lr_te, SVR_EPS):
    tscv = TimeSeriesSplit(n_splits=5)

    # Separate signal-layer params from model hyperparameters
    zscore_win_vals  = grid.pop('zscore_win',       [ZSCORE_WIN])
    threshold_vals   = grid.pop('signal_threshold', [SIGNAL_THRESHOLD])
    model_grid = grid  # remaining params are genuine model hyperparameters

    if model_name == 'Ridge':
        base = Ridge()
        pg   = {f'model__{k}': v for k, v in model_grid.items()}
    elif model_name == 'Lasso':
        base = Lasso()
        pg   = {f'model__{k}': v for k, v in model_grid.items()}
    elif model_name == 'SVR_lin':
        base = LinearSVR(epsilon=SVR_EPS, max_iter=10000, dual=True)
        pg   = {f'model__{k}': v for k, v in model_grid.items()}
    elif model_name == 'SVR_rbf':
        base = SVR(kernel='rbf', epsilon=SVR_EPS)
        pg   = {f'model__{k}': v for k, v in model_grid.items()}
    elif model_name == 'XGBoost':
        base = xgb.XGBRegressor(random_state=SEED, verbosity=0)
        pg   = {f'model__{k}': v for k, v in model_grid.items()}
    elif model_name == 'RandomForest':
        base = RandomForestRegressor(random_state=SEED)
        pg   = {f'model__{k}': v for k, v in model_grid.items()}
    elif model_name == 'MLP':
        base = MLPRegressor(max_iter=500, random_state=SEED)
        pg   = {f'model__{k}': v for k, v in model_grid.items()}
    elif model_name == 'ElasticNet':
        base = ElasticNet(max_iter=5000)
        pg   = {f'model__{k}': v for k, v in model_grid.items()}
    elif model_name == 'LightGBM':
        if not LIGHTGBM_AVAILABLE:
            raise ImportError('lightgbm not installed')
        base = lgb.LGBMRegressor(random_state=SEED, verbose=-1)
        pg   = {f'model__{k}': v for k, v in model_grid.items()}
    else:
        raise ValueError(f'Unknown sklearn model: {model_name}')

    pipe = Pipeline([('scaler', StandardScaler()), ('model', base)])
    total_model = 1
    for v in model_grid.values():
        total_model *= len(v)
    total_signal = len(zscore_win_vals) * len(threshold_vals)
    print(f'  Model grid size: {total_model} combinations × 5 folds = {total_model*5} fits')
    print(f'  Signal OOF grid: zscore_win={zscore_win_vals} × threshold={threshold_vals} = {total_signal} combos')

    t0 = time.time()
    gs = GridSearchCV(pipe, pg, cv=tscv, scoring='neg_mean_squared_error',
                      n_jobs=N_JOBS, refit=True, verbose=1)
    gs.fit(X_tv, y_tv)
    elapsed_gs = time.time() - t0

    # Print full model grid results sorted by CV MSE
    print(f'\n  All results (sorted by CV MSE):')
    res = sorted(zip(gs.cv_results_['params'], gs.cv_results_['mean_test_score']),
                 key=lambda x: -x[1])
    for params, score in res[:20]:
        clean = {k.replace('model__',''): v for k, v in params.items()}
        print(f'    {str(clean):<60}  CV MSE={-score:.8f}')
    if len(res) > 20:
        print(f'    ... ({len(res)-20} more)')

    best_model_params = {k.replace('model__', ''): v for k, v in gs.best_params_.items()}
    best_cv_mse = -gs.best_score_
    print(f'\n  Best model params : {best_model_params}')
    print(f'  Best CV MSE       : {best_cv_mse:.8f}')
    print(f'  GridSearchCV time : {elapsed_gs:.1f}s')

    # ── OOF CV to jointly pick best (zscore_win, signal_threshold) ──────────────
    # Collect OOF predictions from the best model via TimeSeriesSplit on trainval
    print(f'\n  Tuning zscore_win + signal_threshold via joint OOF CV Sharpe...')
    oof_preds    = np.full(len(y_tv), np.nan)
    oof_log_rets = np.full(len(y_tv), np.nan)
    best_pipe_clone = gs.best_estimator_
    for tr_idx, va_idx in tscv.split(X_tv):
        fold_pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('model',  best_pipe_clone.named_steps['model'].__class__(
                **{k: v for k, v in best_pipe_clone.named_steps['model'].get_params().items()}
            ))
        ])
        fold_pipe.fit(X_tv[tr_idx], y_tv[tr_idx])
        oof_preds[va_idx]    = fold_pipe.predict(X_tv[va_idx])
        oof_log_rets[va_idx] = lr_tv[va_idx]

    valid_mask     = ~np.isnan(oof_preds)
    oof_preds_v    = oof_preds[valid_mask]
    oof_log_rets_v = oof_log_rets[valid_mask]
    oof_y_true_v   = y_tv[valid_mask]

    best_signal_sharpe = -np.inf
    best_zscore_win    = zscore_win_vals[0]
    best_threshold     = threshold_vals[0]
    for zw in zscore_win_vals:
        for thr in threshold_vals:
            m_oof = trading_metrics(oof_preds_v, oof_log_rets_v, oof_y_true_v,
                                    zscore_win=zw, threshold=thr)
            print(f'    zw={zw:>3}  thr={thr:.2f}  OOF Sharpe={m_oof["Sharpe"]:+.4f}')
            if m_oof['Sharpe'] > best_signal_sharpe:
                best_signal_sharpe = m_oof['Sharpe']
                best_zscore_win    = zw
                best_threshold     = thr

    print(f'  Best: zscore_win={best_zscore_win}  threshold={best_threshold}'
          f'  (OOF Sharpe={best_signal_sharpe:.4f})')

    best_params = {**best_model_params, 'zscore_win': best_zscore_win,
                   'signal_threshold': best_threshold}
    elapsed = time.time() - t0
    print(f'  Total elapsed   : {elapsed:.1f}s')

    # Evaluate on test set using best model + best signal params
    y_pred = gs.best_estimator_.predict(X_te)
    m = trading_metrics(y_pred, lr_te, y_te, zscore_win=best_zscore_win, threshold=best_threshold)
    print(f'\n  Test metrics (zscore_win={best_zscore_win}, threshold={best_threshold}):')
    for k, v in m.items():
        print(f'    {k:<12} {v:.6f}')

    return best_params, best_cv_mse, m

# ══════════════════════════════════════════════════════════════════════════════
# KERAS TUNING
# ══════════════════════════════════════════════════════════════════════════════
def tune_keras(model_name, grid, X_tv, y_tv, X_te, y_te, lr_tv, lr_te):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, GRU, Bidirectional, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.regularizers import l2
    import tensorflow as tf
    tf.random.set_seed(SEED)

    def directional_mse(y_true, y_pred):
        """Custom loss: MSE + sign-error penalty to prevent mean-collapse."""
        mse      = tf.reduce_mean(tf.square(y_true - y_pred))
        sign_err = tf.maximum(0.0, -y_true * y_pred)
        return mse + 0.5 * tf.reduce_mean(sign_err)

    tscv = TimeSeriesSplit(n_splits=5)

    def make_sequences(X, y, ts):
        Xs, ys = [], []
        for i in range(ts, len(X)):
            Xs.append(X[i-ts:i]); ys.append(y[i])
        return np.array(Xs), np.array(ys)

    def build(name, units, dropout, recurrent_dropout, n_feat, ts):
        if name == 'LSTM':
            m = Sequential([LSTM(units, input_shape=(ts, n_feat),
                                 dropout=dropout, recurrent_dropout=recurrent_dropout,
                                 kernel_regularizer=l2(0.001)), Dense(1)])
        elif name == 'GRU':
            m = Sequential([GRU(units, input_shape=(ts, n_feat),
                                dropout=dropout, recurrent_dropout=recurrent_dropout), Dense(1)])
        elif name == 'BiLSTM':
            m = Sequential([Bidirectional(LSTM(units, recurrent_dropout=recurrent_dropout),
                                          input_shape=(ts, n_feat)),
                            Dropout(dropout), Dense(16, activation='relu'), Dense(1)])
        m.compile(optimizer=Adam(0.001), loss=directional_mse)
        return m

    epochs  = PARAM_CATALOGUE[model_name]['fixed']['epochs']
    patience= PARAM_CATALOGUE[model_name]['fixed']['patience']

    # Separate signal-layer params from Keras architecture params
    zscore_win_vals = grid.pop('zscore_win',       [ZSCORE_WIN])
    threshold_vals  = grid.pop('signal_threshold', [SIGNAL_THRESHOLD])

    param_names = list(grid.keys())
    param_vals  = list(grid.values())
    combos      = list(iterproduct(*param_vals))
    total_signal = len(zscore_win_vals) * len(threshold_vals)
    print(f'  Model grid size: {len(combos)} combinations × 5 folds = {len(combos)*5} fits')
    print(f'  Signal OOF grid: zscore_win={zscore_win_vals} × threshold={threshold_vals} = {total_signal} combos')

    sc = StandardScaler()
    X_tv_s = sc.fit_transform(X_tv)
    X_te_s = sc.transform(X_te)

    all_results = []
    t0 = time.time()

    for combo in combos:
        params = dict(zip(param_names, combo))
        ts     = params['timesteps']
        units  = params.get('units', 32)
        drop   = params.get('dropout', 0.3)
        rdrop  = params.get('recurrent_dropout', 0.2)

        fold_mses = []
        for tr, va in tscv.split(X_tv_s):
            Xtr, ytr = make_sequences(X_tv_s[tr], y_tv[tr], ts)
            Xva, yva = make_sequences(X_tv_s[va], y_tv[va], ts)
            if len(Xva) == 0: continue
            m = build(model_name, units, drop, rdrop, X_tv.shape[1], ts)
            m.fit(Xtr, ytr, epochs=epochs, batch_size=32, verbose=0,
                  callbacks=[EarlyStopping(patience=patience, restore_best_weights=True)])
            fold_mses.append(mean_squared_error(yva, m.predict(Xva, verbose=0).flatten()))
        cv_mse = np.mean(fold_mses) if fold_mses else np.inf
        all_results.append((params, cv_mse))
        print(f'  {str(params):<60}  CV MSE={cv_mse:.8f}')

    all_results.sort(key=lambda x: x[1])
    best_arch_params, best_cv_mse = all_results[0]
    elapsed_arch = time.time() - t0

    print(f'\n  All results (sorted):')
    for params, mse in all_results:
        mark = ' ← BEST' if params == best_arch_params else ''
        print(f'    {str(params):<60}  CV MSE={mse:.8f}{mark}')

    print(f'\n  Best arch params : {best_arch_params}')
    print(f'  Best CV MSE      : {best_cv_mse:.8f}')
    print(f'  Arch search time : {elapsed_arch:.1f}s')

    # ── Collect OOF preds from best arch to tune zscore_win ──────────────────
    print(f'\n  Tuning zscore_win via OOF CV Sharpe...')
    ts_best   = best_arch_params['timesteps']
    units_best = best_arch_params.get('units', 32)
    drop_best  = best_arch_params.get('dropout', 0.3)
    rdrop_best = best_arch_params.get('recurrent_dropout', 0.2)

    oof_preds    = []
    oof_log_rets = []
    oof_y_true   = []
    for tr, va in tscv.split(X_tv_s):
        Xtr, ytr = make_sequences(X_tv_s[tr], y_tv[tr], ts_best)
        Xva, yva = make_sequences(X_tv_s[va], y_tv[va], ts_best)
        if len(Xva) == 0: continue
        fold_m = build(model_name, units_best, drop_best, rdrop_best, X_tv.shape[1], ts_best)
        fold_m.fit(Xtr, ytr, epochs=epochs, batch_size=32, verbose=0,
                   callbacks=[EarlyStopping(patience=patience, restore_best_weights=True)])
        fold_preds = fold_m.predict(Xva, verbose=0).flatten()
        oof_preds.extend(fold_preds)
        # va indices for log_rets: sequences start at ts_best within va split
        va_global = va[ts_best:ts_best + len(fold_preds)]
        oof_log_rets.extend(lr_tv[va_global])
        oof_y_true.extend(yva)

    oof_preds    = np.array(oof_preds)
    oof_log_rets = np.array(oof_log_rets)
    oof_y_true   = np.array(oof_y_true)

    best_signal_sharpe = -np.inf
    best_zscore_win    = zscore_win_vals[0]
    best_threshold     = threshold_vals[0]
    for zw in zscore_win_vals:
        for thr in threshold_vals:
            m_oof = trading_metrics(oof_preds, oof_log_rets, oof_y_true,
                                    zscore_win=zw, threshold=thr)
            print(f'    zw={zw:>3}  thr={thr:.2f}  OOF Sharpe={m_oof["Sharpe"]:+.4f}')
            if m_oof['Sharpe'] > best_signal_sharpe:
                best_signal_sharpe = m_oof['Sharpe']
                best_zscore_win    = zw
                best_threshold     = thr

    print(f'  Best: zscore_win={best_zscore_win}  threshold={best_threshold}'
          f'  (OOF Sharpe={best_signal_sharpe:.4f})')

    best_params = {**best_arch_params, 'zscore_win': best_zscore_win,
                   'signal_threshold': best_threshold}
    elapsed = time.time() - t0
    print(f'  Total elapsed   : {elapsed:.1f}s')

    # Final fit on full trainval and evaluate on test
    final_m = build(model_name, units_best, drop_best, rdrop_best, X_tv.shape[1], ts_best)
    Xtv_sq, ytv_sq = make_sequences(X_tv_s, y_tv, ts_best)
    Xte_sq, yte_sq = make_sequences(X_te_s, y_te, ts_best)
    final_m.fit(Xtv_sq, ytv_sq, epochs=epochs, batch_size=32, verbose=0,
                callbacks=[EarlyStopping(patience=patience, restore_best_weights=True)])
    y_pred = final_m.predict(Xte_sq, verbose=0).flatten()
    m = trading_metrics(y_pred, lr_te[ts_best:], y_te[ts_best:],
                        zscore_win=best_zscore_win, threshold=best_threshold)
    print(f'\n  Test metrics (zscore_win={best_zscore_win}, threshold={best_threshold}):')
    for k, v in m.items():
        print(f'    {k:<12} {v:.6f}')

    return best_params, best_cv_mse, m

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def parse_param_override(raw):
    """Parse '--param key=[v1,v2,v3]' into {key: [v1,v2,v3]}."""
    overrides = {}
    for item in raw:
        k, _, v = item.partition('=')
        vals = json.loads(v)
        if not isinstance(vals, list):
            vals = [vals]
        overrides[k.strip()] = vals
    return overrides

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train and fine-tune a single model.')
    parser.add_argument('--model', type=str, choices=list(PARAM_CATALOGUE.keys()),
                        help='Model to tune')
    parser.add_argument('--param', type=str, action='append', default=[],
                        metavar='name=[v1,v2,v3]',
                        help='Override a parameter grid (repeatable)')
    parser.add_argument('--help-params', action='store_true',
                        help='Print full parameter catalogue and exit')
    parser.add_argument('--n-jobs', type=int, default=-1,
                        help='Parallel jobs for sklearn GridSearchCV (default: -1 = all cores). '
                             'Set to 1 or 2 when running multiple models in parallel via tune_all.py.')
    args = parser.parse_args()
    N_JOBS = args.n_jobs

    if args.help_params:
        print_param_report()
        sys.exit(0)

    if not args.model:
        print_param_report()
        parser.print_help()
        sys.exit(0)

    print_param_report()

    model_name = args.model
    cfg        = PARAM_CATALOGUE[model_name]

    # Merge default grid with any overrides
    grid = dict(cfg['grid'])
    if args.param:
        overrides = parse_param_override(args.param)
        grid.update(overrides)
        print(f'  Overridden params: {overrides}')

    print(f'\n{"="*60}')
    print(f'  TUNING: {model_name}')
    print(f'  Type  : {cfg["type"]}')
    print(f'  Grid  : {grid}')
    print(f'{"="*60}\n')

    print('Loading data...')
    X_tv, y_tv, X_te, y_te, lr_tv, lr_te, dates_te, feat_cols, SVR_EPS, meta = load_data()

    if cfg['type'] in ('sklearn', 'sklearn_pipeline'):
        best_params, best_cv_mse, test_m = tune_sklearn(
            model_name, grid, X_tv, y_tv, X_te, y_te, lr_tv, lr_te, SVR_EPS)
    elif cfg['type'] == 'keras':
        best_params, best_cv_mse, test_m = tune_keras(
            model_name, grid, X_tv, y_tv, X_te, y_te, lr_tv, lr_te)

    # Save per-model result (safe for parallel runs — no shared file writes)
    result = {'model': model_name, 'best_params': best_params,
              'cv_mse': best_cv_mse, **test_m}
    per_model_path = f'{MODELS_DIR}/tuned_{model_name}.json'
    with open(per_model_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f'\n  Results saved to {per_model_path}')
    print('\nDone.')

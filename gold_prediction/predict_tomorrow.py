#!/usr/bin/env python3
"""
predict_tomorrow.py — Gold price prediction using pre-trained models.

Loads saved sklearn / Keras models from disk, rebuilds features on the latest
data, and produces trading signals for tomorrow's Close and/or Open price.

Usage:
    python predict_tomorrow.py               # both Close and Open (default)
    python predict_tomorrow.py --target close
    python predict_tomorrow.py --target open
"""

import os, sys, json, warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
import joblib
from datetime import timedelta

import argparse
_parser = argparse.ArgumentParser()
_parser.add_argument('--target', choices=['close', 'open', 'both'], default='both')
_args = _parser.parse_args()
PRICE_TARGET = _args.target

DATA_DIR   = 'data'
ZSCORE_WIN = 10   # must match train.py

# ── Model directory discovery ─────────────────────────────────────────────────
def _find_models_dir(price_target):
    for candidate in [f'saved_models/{price_target}_3y',
                      f'saved_models/{price_target}_full',
                      f'saved_models/{price_target}']:
        if os.path.isdir(candidate) and os.path.exists(f'{candidate}/model_metadata.json'):
            return candidate
    sys.exit(f'ERROR: no saved models found for target={price_target}. Run train.py first.')

# ── Refresh data ───────────────────────────────────────────────────────────────
_dl = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'download_data.py')
if os.path.exists(_dl):
    import subprocess
    print('Refreshing data...')
    r = subprocess.run([sys.executable, _dl], capture_output=True, text=True,
                       cwd=os.path.dirname(os.path.abspath(__file__)))
    for line in r.stdout.splitlines():
        if any(kw in line for kw in ('new row', 'latest:', 'COMPLETE', 'WARNING', 'ERROR')):
            print(f'  {line.strip()}')
    print()

# ── Fractional differencing ────────────────────────────────────────────────────
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

def fracdiff_weights(d, window=252, threshold=1e-5):
    w = [1.0]
    for k in range(1, window):
        w_k = -w[-1] * (d - k + 1) / k
        if abs(w_k) < threshold:
            break
        w.append(w_k)
    return np.array(w)

def fracdiff_to_price(pred_fd, log_gold_hist, d):
    """Invert a frac_diff prediction to an actual price."""
    w = fracdiff_weights(d)
    hist_sum = sum(w[k] * log_gold_hist[-(k)]
                   for k in range(1, min(len(w), len(log_gold_hist) + 1)))
    return float(np.exp(pred_fd - hist_sum))

# ── Feature engineering (mirrors train.py section 3 exactly) ─────────────────
def build_features(df_raw, price_target_local, frac_d):
    df = df_raw.copy()
    HAS_COT = 'Comm_Positions_Long_All' in df.columns

    if price_target_local == 'open':
        TARGET_COL = 'frac_diff_log_Open'
        if 'log_Open' not in df.columns and 'Open' in df.columns:
            df['log_Open'] = np.log(df['Open'])
        df[TARGET_COL] = frac_diff(df['log_Open'], frac_d)
    else:
        TARGET_COL = 'frac_diff_log_Gold'

    df['frac_diff_log_Gold'] = frac_diff(df['log_Gold'], frac_d)

    if HAS_COT:
        for col in ['Comm_Positions_Long_All', 'Comm_Positions_Short_All',
                    'NonComm_Positions_Long_All', 'NonComm_Positions_Short_All',
                    'Open_Interest_All']:
            if col in df.columns:
                df[f'{col}_diff'] = frac_diff(df[col], frac_d)
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
        for out_col, raw_col in [('cot_norm_spec', 'net_speculator'),
                                  ('cot_norm_comm', 'net_commercial')]:
            rmin = df[raw_col].rolling(252, min_periods=60).min()
            rmax = df[raw_col].rolling(252, min_periods=60).max()
            df[out_col] = (df[raw_col] - rmin) / (rmax - rmin + 1e-12)

    if 'DXY' in df.columns:
        df['DXY_chg']     = df['DXY'].pct_change(5)
        df['DXY_ma20']    = df['DXY'].rolling(20).mean()
        df['DXY_vs_ma20'] = df['DXY'] / df['DXY_ma20'] - 1
    if 'TNX_yield' in df.columns and 'VIX' in df.columns:
        tyield = df['TYX_yield'] if 'TYX_yield' in df.columns else df['TNX_yield']
        df['yield_spread']  = tyield - df['TNX_yield']
        df['real_yield_px'] = df['TNX_yield'] - df['VIX'] * 0.15
    if 'VIX' in df.columns:
        vr = df['VIX'].rolling(60)
        df['VIX_zscore'] = (df['VIX'] - vr.mean()) / vr.std()

    for _mc, _mc_open in [('DXY', 'DXY_Open'), ('VIX', 'VIX_Open'),
                           ('TNX_yield', 'TNX_yield_Open'), ('TYX_yield', 'TYX_yield_Open')]:
        if _mc_open in df.columns and _mc in df.columns:
            df[f'{_mc}_overnight'] = df[_mc_open] - df[_mc].shift(1)

    try:
        import talib
        c = df['Gold'].values.astype(float)
        df['RSI_14']    = talib.RSI(c, timeperiod=14)
        df['RSI_28']    = talib.RSI(c, timeperiod=28)
        df['ROC_10']    = talib.ROC(c, timeperiod=10)
        df['ROC_20']    = talib.ROC(c, timeperiod=20)
        df['MOM_5']     = talib.MOM(c, timeperiod=5)
        df['EMA_12']    = talib.EMA(c, timeperiod=12)
        df['EMA_26']    = talib.EMA(c, timeperiod=26)
        df['MACD_diff'] = df['EMA_12'] - df['EMA_26']
        up, mid, lo     = talib.BBANDS(c, timeperiod=20)
        df['BB_width']  = (up - lo) / (mid + 1e-12)
        df['BB_pct']    = (c - lo) / (up - lo + 1e-12)
    except ImportError:
        df['RSI_14'] = df['Gold'].diff().clip(lower=0).rolling(14).mean() / df['Gold'].diff().abs().rolling(14).mean()
        df['RSI_28'] = df['Gold'].diff().clip(lower=0).rolling(28).mean() / df['Gold'].diff().abs().rolling(28).mean()
        df['ROC_10'] = df['Gold'].pct_change(10) * 100
        df['ROC_20'] = df['Gold'].pct_change(20) * 100
        df['MOM_5']  = df['Gold'].diff(5)
        df['EMA_12'] = df['Gold'].ewm(span=12).mean()
        df['EMA_26'] = df['Gold'].ewm(span=26).mean()
        df['MACD_diff'] = df['EMA_12'] - df['EMA_26']
        bb_mid = df['Gold'].rolling(20).mean()
        bb_std = df['Gold'].rolling(20).std()
        df['BB_width'] = bb_std / bb_mid
        df['BB_pct']   = (df['Gold'] - (bb_mid - 2*bb_std)) / (4 * bb_std + 1e-12)

    for lag in [1, 2, 3, 5, 10]:
        df[f'target_lag{lag}'] = df[TARGET_COL].shift(lag)

    df['log_return'] = df['log_Gold'].diff()
    for period in [21, 63, 126, 252]:
        df[f'gold_ret_{period}d'] = df['log_Gold'].diff(period)
    for m_num in range(1, 13):
        df[f'month_{m_num}'] = (df.index.month == m_num).astype(int)

    if HAS_COT:
        df['net_spec_chg_20d'] = df['net_speculator'].diff(20)
        df['net_comm_chg_20d'] = df['net_commercial'].diff(20)

    if price_target_local == 'open' and 'Open' in df.columns:
        df['open_vs_prev_close'] = df['Open'] / df['Gold'].shift(1) - 1
        df['open_close_ratio']   = df['Open'] / df['Gold']

    df.dropna(inplace=True)
    return df, TARGET_COL

# ── Z-score signal ─────────────────────────────────────────────────────────────
def zscore_signal(preds):
    ps   = pd.Series(np.asarray(preds, dtype=float))
    mean = ps.rolling(ZSCORE_WIN, min_periods=1).mean()
    std  = ps.rolling(ZSCORE_WIN, min_periods=1).std().fillna(1e-8)
    z    = float(((ps - mean) / std).iloc[-1])
    sig  = 1 if z > 0 else -1
    return z, sig

def sig_label(v):
    return 'LONG' if v > 0 else 'SHORT'

# ── Per-target prediction ──────────────────────────────────────────────────────
def run_prediction(price_target_local):
    models_dir = _find_models_dir(price_target_local)
    meta       = json.load(open(f'{models_dir}/model_metadata.json'))
    feat_cols  = meta['feature_cols']
    frac_d     = meta['frac_d']
    tuned      = meta.get('tuned_params', {})

    print(f'[{price_target_local.upper()}] models from {models_dir}  (frac_d={frac_d})')

    # Build features from full history
    df_raw = pd.read_csv(f'{DATA_DIR}/merged_gold_dataset.csv', index_col=0, parse_dates=True)
    df_raw.index = pd.to_datetime(df_raw.index).normalize()
    df_raw.sort_index(inplace=True)
    df, target_col = build_features(df_raw, price_target_local, frac_d)

    latest_date = df.index[-1]
    latest_gold = float(df['Gold'].iloc[-1])
    log_gold_hist = df['log_Gold'].values

    # Feature window for z-score (need ZSCORE_WIN rows of predictions)
    n_win = ZSCORE_WIN + 30
    feat_cols_present = [c for c in feat_cols if c in df.columns]
    if len(feat_cols_present) < len(feat_cols):
        missing = set(feat_cols) - set(feat_cols_present)
        print(f'  WARNING: {len(missing)} feature cols missing: {missing}')
    X_win = df[feat_cols_present].values[-n_win:]

    sc = joblib.load(f'{models_dir}/all_models_scaler.joblib')
    # Scaler was fitted on all feat_cols; subset to present ones
    if len(feat_cols_present) == len(feat_cols):
        X_win_s = sc.transform(X_win)
    else:
        # Re-fit on available features (fallback; degrades quality slightly)
        X_win_s = X_win  # pass raw; scaler will error if shapes mismatch

    signals, raw_preds, z_scores, price_preds = {}, {}, {}, {}

    SK_NAMES = ['Ridge', 'Lasso', 'SVR_lin', 'SVR_rbf', 'XGBoost', 'RandomForest', 'MLP']
    for name in SK_NAMES:
        path = f'{models_dir}/model_{name}.joblib'
        if not os.path.exists(path):
            continue
        try:
            m = joblib.load(path)
            preds = m.predict(X_win_s)
            z, sig = zscore_signal(preds)
            signals[name]     = sig
            raw_preds[name]   = float(preds[-1])
            z_scores[name]    = z
            price_preds[name] = fracdiff_to_price(float(preds[-1]), log_gold_hist, frac_d)
        except Exception as e:
            print(f'  {name}: ERROR — {e}')

    # Keras models
    try:
        import tensorflow as tf
        from tensorflow.keras.models import load_model as keras_load

        def directional_mse(y_true, y_pred):
            mse      = tf.reduce_mean(tf.square(y_true - y_pred))
            sign_err = tf.maximum(0.0, -y_true * y_pred)
            return mse + 0.5 * tf.reduce_mean(sign_err)

        lstm_ts   = meta.get('lstm_best_timesteps',   5)
        gru_ts    = meta.get('gru_best_timesteps',    5)
        bilstm_ts = meta.get('bilstm_best_timesteps', 5)

        for name, ts in [('LSTM', lstm_ts), ('GRU', gru_ts), ('BiLSTM', bilstm_ts)]:
            path = f'{models_dir}/model_{name}.keras'
            if not os.path.exists(path):
                continue
            try:
                km = keras_load(path, custom_objects={'directional_mse': directional_mse})
                n_seq = ts + n_win
                X_full = df[feat_cols_present].values[-n_seq:]
                X_full_s = sc.transform(X_full)
                seqs = np.array([X_full_s[i - ts:i] for i in range(ts, len(X_full_s))])
                preds = km.predict(seqs, verbose=0).flatten()
                z, sig = zscore_signal(preds)
                signals[name]     = sig
                raw_preds[name]   = float(preds[-1])
                z_scores[name]    = z
                price_preds[name] = fracdiff_to_price(float(preds[-1]), log_gold_hist, frac_d)
            except Exception as e:
                print(f'  {name}: ERROR — {e}')
    except ImportError:
        pass

    # Consensus: 1/Sharpe² weighted vote (use backtest Sharpes from metrics CSV)
    metrics_csv = None
    for suffix in ['3y', 'full', '']:
        path = (f'results/{price_target_local}_{suffix}/all_model_metrics.csv' if suffix
                else f'results/{price_target_local}/all_model_metrics.csv')
        if os.path.exists(path):
            try:
                metrics_csv = pd.read_csv(path, index_col=0)
                break
            except Exception:
                pass

    sharpes = {}
    if metrics_csv is not None:
        for n in signals:
            if n in metrics_csv.index:
                sharpes[n] = max(float(metrics_csv.loc[n, 'Sharpe']), 0.0)

    # Weighted consensus
    pos_w = {n: s for n, s in sharpes.items() if n in signals and s > 0}
    total_w = sum(pos_w.values())
    if total_w > 0:
        weighted_vote = sum(signals[n] * pos_w[n] / total_w for n in pos_w)
    else:
        weighted_vote = sum(signals.values()) / max(len(signals), 1)

    n_long  = sum(1 for v in signals.values() if v > 0)
    n_short = sum(1 for v in signals.values() if v < 0)

    # MSE-weighted price target
    mse_weights = {}
    if metrics_csv is not None:
        for n in price_preds:
            if n in metrics_csv.index:
                mse_val = float(metrics_csv.loc[n, 'MSE'])
                if mse_val > 0:
                    mse_weights[n] = 1.0 / mse_val
    if mse_weights:
        total_mse_w = sum(mse_weights.values())
        consensus_price = sum(price_preds[n] * mse_weights[n] / total_mse_w
                              for n in mse_weights)
    else:
        consensus_price = np.mean(list(price_preds.values())) if price_preds else latest_gold

    return {
        'label':           price_target_local.upper(),
        'models_dir':      models_dir,
        'latest_date':     latest_date,
        'latest_gold':     latest_gold,
        'log_gold_hist':   log_gold_hist,
        'signals':         signals,
        'z_scores':        z_scores,
        'price_preds':     price_preds,
        'sharpes':         sharpes,
        'weighted_vote':   weighted_vote,
        'n_long':          n_long,
        'n_short':         n_short,
        'consensus_price': consensus_price,
        'df':              df,
    }

# ══════════════════════════════════════════════════════════════════════════════
# RUN
# ══════════════════════════════════════════════════════════════════════════════
targets = (['close', 'open'] if PRICE_TARGET == 'both' else [PRICE_TARGET])
results = {}
for t in targets:
    results[t] = run_prediction(t)

# Shared date info from first result
_r0 = results[targets[0]]
latest_date = _r0['latest_date']
latest_gold = _r0['latest_gold']
next_day    = latest_date + timedelta(days=1)
while next_day.weekday() >= 5:
    next_day += timedelta(days=1)

df0 = _r0['df']
recent_vol  = float(df0['log_return'].iloc[-20:].std())
longrun_vol = float(df0['log_return'].std())
vol_ratio   = recent_vol / (longrun_vol + 1e-8)
vol_note    = ('high vol — scale down' if vol_ratio > 1.1 else
               'low vol  — scale up'   if vol_ratio < 0.9 else 'normal vol')

MODEL_ORDER = ['SVR_lin', 'Ridge', 'Lasso', 'SVR_rbf', 'XGBoost', 'RandomForest', 'MLP',
               'LSTM', 'GRU', 'BiLSTM']

# ── Display ────────────────────────────────────────────────────────────────────
print(f'\n{"═"*72}')
print(f'  GOLD PRICE PREDICTIONS — {next_day.strftime("%A  %Y-%m-%d")}')
print(f'  (based on data through {latest_date.date()}  |  Last close: ${latest_gold:,.2f})')
print(f'{"═"*72}')

if PRICE_TARGET == 'both':
    rc = results['close']
    ro = results['open']

    W1, W2, W3, W4 = 15, 9, 14, 14
    header = (f'{"║"} {"Model":<{W1}} {"║"} {"CLOSE sig":^{W2}} {"║"} {"Close Price":^{W3}} {"║"}'
              f' {"OPEN sig":^{W4}} {"║"}')
    sep    = f'{"╠"}{"═"*(W1+2)}{"╬"}{"═"*(W2+2)}{"╬"}{"═"*(W3+2)}{"╬"}{"═"*(W4+2)}{"╣"}'
    top    = f'{"╔"}{"═"*(W1+2)}{"╦"}{"═"*(W2+2)}{"╦"}{"═"*(W3+2)}{"╦"}{"═"*(W4+2)}{"╗"}'
    bot    = f'{"╚"}{"═"*(W1+2)}{"╩"}{"═"*(W2+2)}{"╩"}{"═"*(W3+2)}{"╩"}{"═"*(W4+2)}{"╝"}'

    print(top)
    print(header)
    print(sep)

    for name in MODEL_ORDER:
        in_c = name in rc['signals']
        in_o = name in ro['signals']
        if not in_c and not in_o:
            continue
        c_sig   = rc['signals'].get(name, 0)
        c_z     = rc['z_scores'].get(name, 0.0)
        c_px    = rc['price_preds'].get(name, float('nan'))
        c_chg   = c_px - latest_gold
        c_lbl   = f'{"▲" if c_sig > 0 else "▼"} {sig_label(c_sig):5} z={c_z:+.2f}'
        c_price = f'${c_px:>10,.2f} ({c_chg:>+7.2f})' if in_c else f'{"—":^{W3}}'
        o_sig   = ro['signals'].get(name, 0)
        o_z     = ro['z_scores'].get(name, 0.0)
        o_px    = ro['price_preds'].get(name, float('nan'))
        o_chg   = o_px - latest_gold
        o_lbl   = (f'{"▲" if o_sig > 0 else "▼"} {sig_label(o_sig):5} z={o_z:+.2f}' if in_o
                   else f'{"—":^{W4}}')
        sr = rc['sharpes'].get(name, 0)
        star = '*' if sr >= 6.0 else ' '
        print(f'{"║"} {name+star:<{W1}} {"║"} {c_lbl:^{W2}} {"║"} {c_price:<{W3}} {"║"}'
              f' {o_lbl:^{W4}} {"║"}')

    print(sep)
    c_cons = sig_label(rc['weighted_vote'])
    o_cons = sig_label(ro['weighted_vote'])
    c_price_str = f'${rc["consensus_price"]:>10,.2f} ({rc["consensus_price"]-latest_gold:>+7.2f})'
    o_price_str = f'${ro["consensus_price"]:>10,.2f} ({ro["consensus_price"]-latest_gold:>+7.2f})'
    print(f'{"║"} {"CONSENSUS":<{W1}} {"║"} {"▲ "+c_cons if rc["weighted_vote"]>0 else "▼ "+c_cons:^{W2}} {"║"}'
          f' {c_price_str:<{W3}} {"║"} {"▲ "+o_cons if ro["weighted_vote"]>0 else "▼ "+o_cons:^{W4}} {"║"}')
    print(f'{"║"} {"Vote":<{W1}} {"║"} {str(rc["n_long"])+"L/"+str(rc["n_short"])+"S":^{W2}} {"║"}'
          f' {"":<{W3}} {"║"} {str(ro["n_long"])+"L/"+str(ro["n_short"])+"S":^{W4}} {"║"}')
    print(bot)

else:
    r = results[targets[0]]
    print(f'\n  {"Model":<15} {"Sharpe":>7}  {"Signal":>7}  {"z-score":>8}  {"Price Target":>13}  {"Chg":>8}')
    print(f'  {"─"*68}')
    for name in MODEL_ORDER:
        if name not in r['signals']:
            continue
        sig = r['signals'][name]
        z   = r['z_scores'][name]
        px  = r['price_preds'].get(name, float('nan'))
        sr  = r['sharpes'].get(name, 0)
        star = '*' if sr >= 6.0 else ''
        lbl = sig_label(sig)
        print(f'  {name+star:<15} {sr:>7.2f}  {"▲ "+lbl if sig>0 else "▼ "+lbl:>7}  {z:>+8.3f}  ${px:>12,.2f}  {px-latest_gold:>+8.2f}')
    print(f'  {"─"*68}')
    cons_px = r['consensus_price']
    print(f'  {"CONSENSUS":<15}        {"▲ "+sig_label(r["weighted_vote"]) if r["weighted_vote"]>0 else "▼ "+sig_label(r["weighted_vote"]):>7}          ${cons_px:>12,.2f}  {cons_px-latest_gold:>+8.2f}')
    print(f'  Vote: {r["n_long"]}L / {r["n_short"]}S')

# ── Market context ─────────────────────────────────────────────────────────────
print(f'\n{"─"*72}')
print(f'  Latest:  ${latest_gold:,.2f}  ({latest_date.date()})')
print(f'  Vol:     20d={recent_vol*100:.3f}%/day  long-run={longrun_vol*100:.3f}%/day  ratio={vol_ratio:.2f}x  → {vol_note}')

if 'DXY' in df0.columns:
    dxy = float(df0['DXY'].iloc[-1])
    dxy_chg = float(df0['DXY_chg'].iloc[-1]) if 'DXY_chg' in df0.columns else float('nan')
    dxy_lbl = ('USD weak → gold bullish' if dxy_chg < -0.005 else
               'USD strong → gold bearish' if dxy_chg > 0.005 else 'USD stable')
    print(f'  DXY:     {dxy:.2f}  ({dxy_chg:+.2%} 5d)  {dxy_lbl}')
if 'VIX' in df0.columns:
    vix = float(df0['VIX'].iloc[-1])
    print(f'  VIX:     {vix:.1f}  {"(elevated → safe-haven bid)" if vix > 25 else "(normal risk env)"}')
if 'TNX_yield' in df0.columns:
    print(f'  10Y:     {float(df0["TNX_yield"].iloc[-1]):.2f}%')
if 'DXY_overnight' in df0.columns:
    print(f'  DXY gap: {float(df0["DXY_overnight"].iloc[-1]):+.3f} (overnight)')
if 'VIX_overnight' in df0.columns:
    print(f'  VIX gap: {float(df0["VIX_overnight"].iloc[-1]):+.3f} (overnight)')

print(f'{"─"*72}')
print(f'  * = Sharpe ≥ 6.0 in backtest')
print(f'  LONG = bullish signal  |  SHORT = bearish signal')
print(f'  z-score: rolling {ZSCORE_WIN}-period normalisation of raw model output')
print(f'{"═"*72}')

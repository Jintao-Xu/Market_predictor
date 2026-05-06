#!/usr/bin/env python3
"""
plot_all.py — Generate all result charts from saved training artifacts.

Reads data saved by train.py from results/ and saved_models/, produces:
  model_comparison_cv_mse.png      — CV MSE bar chart (all models)
  predicted_vs_actual.png          — Best model predicted vs actual
  equity_curve_test.png            — Best model equity curve vs buy & hold
  all_model_trading_metrics.png    — Sharpe / CAGR / PF bars (all models)
  predicted_vs_actual_all_models.png — 10-panel predicted vs actual grid
  equity_curves_all_models.png     — Capital growth + consensus bar (all models)
  signal_timeline.png              — Per-model long/short action strips
  shap_summary.png                 — Best model SHAP beeswarm
  shap_all_models.png              — All-model SHAP bar grid (top 15 features)
  bias_tests.png                   — White Reality Check + MC permutation test

Run after train.py has completed:
    python plot_all.py

Optional flags:
    python plot_all.py --only cv_mse pva_best equity_best   # subset of charts
    python plot_all.py --dpi 200                             # higher resolution
"""

import os, sys, json, argparse, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
from matplotlib.colors import LinearSegmentedColormap

RESULTS_DIR = 'results'
MODELS_DIR  = 'saved_models'

# Allow --target close/open to be parsed early so path constants are set
# before any chart functions are defined (they close over RESULTS_DIR/MODELS_DIR).
import sys as _sys
for _i, _a in enumerate(_sys.argv[:-1]):
    if _a == '--target':
        _t = _sys.argv[_i + 1]
        RESULTS_DIR = f'results/{_t}'
        MODELS_DIR  = f'saved_models/{_t}'
        break

ALL_CHARTS = ['cv_mse', 'pva_best', 'equity_best',
              'trading_metrics', 'pva_all', 'equity_all',
              'signal_timeline', 'shap_best', 'shap_all', 'bias_tests']

# ── Load shared artifacts ─────────────────────────────────────────────────────
def load_meta():
    with open(f'{MODELS_DIR}/model_metadata.json') as f:
        return json.load(f)

def load_metrics_df():
    return pd.read_csv(f'{RESULTS_DIR}/all_model_metrics.csv', index_col=0)

def load_predictions():
    df = pd.read_csv(f'{RESULTS_DIR}/test_predictions.csv', parse_dates=['date'])
    return df

def load_best_predictions():
    df = pd.read_csv(f'{RESULTS_DIR}/best_model_predictions.csv', parse_dates=['date'])
    return df

def load_strategy():
    df = pd.read_csv(f'{RESULTS_DIR}/best_model_strategy.csv', parse_dates=['date'])
    return df

def rolling_zscore_signal(preds, win=5):
    ps   = pd.Series(preds)
    mean = ps.rolling(win, min_periods=1).mean()
    std  = ps.rolling(win, min_periods=1).std().fillna(1e-8)
    return np.where(((ps - mean) / std).fillna(0) >= 0, 1, -1).astype(float)

def equity_curve(signal, log_rets):
    return np.exp(np.cumsum(signal * np.array(log_rets)))

def model_color(name, sharpe, strong_palette):
    if name in strong_palette:
        return strong_palette[name]
    if sharpe >= 1.0:   return '#888888'
    if sharpe >= 0:     return '#aaaaaa'
    return '#cc4444'

# ══════════════════════════════════════════════════════════════════════════════
# 1. CV MSE bar chart
# ══════════════════════════════════════════════════════════════════════════════
def plot_cv_mse(dpi):
    path = f'{RESULTS_DIR}/cv_mse.json'
    if not os.path.exists(path):
        print(f'  [skip] {path} not found'); return
    with open(path) as f:
        data = json.load(f)
    best   = data['best_model']
    mses   = data['models']
    names  = list(mses.keys())
    vals   = [mses[n] for n in names]
    colors = ['tomato' if n == best else 'steelblue' for n in names]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].bar(names, vals, color=colors)
    axes[0].set_yscale('log')
    axes[0].set_title('CV MSE — all models (log scale)')
    axes[0].tick_params(axis='x', rotation=45)
    non_keras = {n: mses[n] for n in names if n not in ['LSTM','GRU','BiLSTM']}
    axes[1].bar(non_keras.keys(), non_keras.values(),
                color=['tomato' if n == best else 'steelblue' for n in non_keras])
    axes[1].set_title('CV MSE — sklearn + ARIMAX (linear scale)')
    axes[1].tick_params(axis='x', rotation=45)
    plt.suptitle(f'CV MSE — Best model: {best}', fontsize=12, fontweight='bold')
    plt.tight_layout()
    out = f'{RESULTS_DIR}/model_comparison_cv_mse.png'
    plt.savefig(out, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'  Saved → {out}')

# ══════════════════════════════════════════════════════════════════════════════
# 2. Best model predicted vs actual
# ══════════════════════════════════════════════════════════════════════════════
def plot_pva_best(dpi):
    path = f'{RESULTS_DIR}/best_model_predictions.csv'
    if not os.path.exists(path):
        print(f'  [skip] {path} not found'); return
    meta = load_meta()
    df   = load_best_predictions()

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(df['date'], df['y_test'], label='Actual',    lw=1.2, color='#2c7bb6')
    ax.plot(df['date'], df['y_pred'], label='Predicted', lw=1.2, color='#d7191c',
            ls='--', alpha=0.85)
    ax.set_title(f'{meta["best_model"]} — Predicted vs Actual (Test)',
                 fontsize=11, fontweight='bold')
    ax.set_ylabel('Frac-Diff Log Return')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.tight_layout()
    out = f'{RESULTS_DIR}/predicted_vs_actual.png'
    plt.savefig(out, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'  Saved → {out}')

# ══════════════════════════════════════════════════════════════════════════════
# 3. Best model equity curve
# ══════════════════════════════════════════════════════════════════════════════
def plot_equity_best(dpi):
    path = f'{RESULTS_DIR}/best_model_strategy.csv'
    if not os.path.exists(path):
        print(f'  [skip] {path} not found'); return
    meta = load_meta()
    df   = load_strategy()

    cagr_s = float(meta['cagr'])
    sr_s   = float(meta['sharpe'])
    pf_s   = float(meta['profit_factor'])
    mdd_s  = float(meta['max_drawdown'])

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    axes[0].plot(df['date'], df['equity_strat'], label='Strategy',      color='steelblue', lw=1.5)
    axes[0].plot(df['date'], df['equity_bnh'],   label='Buy&Hold Gold', color='gold',      lw=1.5, ls='--')
    axes[0].set_title(
        f'{meta["best_model"]} Equity Curve | CAGR={cagr_s:.1%} | '
        f'Sharpe={sr_s:.2f} | PF={pf_s:.2f} | MDD={mdd_s:.2%}',
        fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Portfolio Value')
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.25)
    axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:.1f}'))
    axes[1].bar(df['date'], df['strat_ret'],
                color=['steelblue' if r >= 0 else 'tomato' for r in df['strat_ret']], width=1)
    axes[1].axhline(0, color='black', lw=0.5)
    axes[1].set_title('Daily Strategy Log Returns')
    axes[1].set_ylabel('Log Return')
    axes[1].grid(alpha=0.2)
    axes[1].xaxis.set_major_locator(mdates.YearLocator())
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.tight_layout()
    out = f'{RESULTS_DIR}/equity_curve_test.png'
    plt.savefig(out, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'  Saved → {out}')

# ══════════════════════════════════════════════════════════════════════════════
# 4. All-model trading metrics bar charts
# ══════════════════════════════════════════════════════════════════════════════
def plot_trading_metrics(dpi):
    path = f'{RESULTS_DIR}/all_model_metrics.csv'
    if not os.path.exists(path):
        print(f'  [skip] {path} not found'); return
    meta       = load_meta()
    metrics_df = load_metrics_df()
    mnames     = metrics_df.index.tolist()

    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    for ax, metric, col in zip(axes[:3],
                                ['Sharpe','CAGR','PF'],
                                ['steelblue','seagreen','darkorange']):
        vals = metrics_df[metric].values.astype(float)
        ax.bar(mnames, vals, color=[col if v > 0 else 'tomato' for v in vals])
        ax.set_title(f'{metric} by Model', fontsize=10)
        ax.tick_params(axis='x', rotation=45)
        ax.axhline(0, color='black', lw=0.5)
    # DirAcc panel
    if 'DirAcc' in metrics_df.columns:
        vals = metrics_df['DirAcc'].values.astype(float) * 100
        axes[3].bar(mnames, vals, color=['#D4AF37' if v > 65 else 'steelblue' for v in vals])
        axes[3].axhline(55, color='tomato', lw=1, ls='--', label='Market base rate (55%)')
        axes[3].set_title('Signal Dir Acc (%)', fontsize=10)
        axes[3].tick_params(axis='x', rotation=45)
        axes[3].legend(fontsize=8)

    plt.suptitle(
        f'All-Model Trading Metrics | LSTM ts={meta.get("lstm_best_timesteps","?")}  '
        f'GRU ts={meta.get("gru_best_timesteps","?")}  '
        f'BiLSTM ts={meta.get("bilstm_best_timesteps","?")}',
        fontsize=11, fontweight='bold', y=1.01)
    plt.tight_layout()
    out = f'{RESULTS_DIR}/all_model_trading_metrics.png'
    plt.savefig(out, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'  Saved → {out}')

# ══════════════════════════════════════════════════════════════════════════════
# 5. All-model predicted vs actual grid
# ══════════════════════════════════════════════════════════════════════════════
def plot_pva_all(dpi):
    path = f'{RESULTS_DIR}/test_predictions.csv'
    if not os.path.exists(path):
        print(f'  [skip] {path} not found'); return
    pred_df    = load_predictions()
    metrics_df = load_metrics_df()
    dates      = pred_df['date']
    y_test     = pred_df['y_test'].values

    pred_cols  = [c for c in pred_df.columns if c.startswith('pred_')]
    model_names = [c.replace('pred_', '') for c in pred_cols]

    ORDER = ['Ridge','RandomForest','XGBoost','Lasso','ARIMAX',
             'BiLSTM','GRU','LSTM','MLP','SVR_lin','SVR_rbf']
    ORDER = [m for m in ORDER if m in model_names] + \
            [m for m in model_names if m not in ORDER]

    nrows = (len(ORDER) + 1) // 2
    fig, axes = plt.subplots(nrows, 2, figsize=(16, 3.5 * nrows))
    axes = axes.flatten()
    fig.suptitle(
        f'Predicted vs Actual — Frac-Diff Log Gold Returns\n'
        f'Test: {dates.iloc[0].strftime("%Y-%m-%d")} → {dates.iloc[-1].strftime("%Y-%m-%d")}',
        fontsize=13, fontweight='bold')

    for idx, name in enumerate(ORDER):
        ax    = axes[idx]
        preds = pred_df[f'pred_{name}'].dropna().values
        n     = len(preds)
        dt    = dates.iloc[:n]
        mrow  = metrics_df.loc[name] if name in metrics_df.index else {}
        sharpe = float(mrow.get('Sharpe', 0)) if isinstance(mrow, pd.Series) else 0
        dacc   = float(mrow.get('DirAcc', 0)) if isinstance(mrow, pd.Series) else 0
        mse    = float(mrow.get('MSE', 0))    if isinstance(mrow, pd.Series) else 0
        star   = ' ★' if sharpe >= 7.5 else ''
        title  = (f'{name}{star} | MSE={mse:.5f} | '
                  f'Sig Dir Acc={dacc*100:.1f}% | Sharpe={sharpe:.2f}')
        ax.plot(dt, y_test[:n], color='#2c7bb6', lw=0.7, alpha=0.9, label='Actual')
        ax.plot(dt, preds,      color='#d7191c', lw=0.7, alpha=0.8, ls='--', label='Predicted')
        ax.set_title(title, fontsize=8.5)
        ax.tick_params(labelsize=7)
        ax.set_ylabel('Frac-Diff\nReturn', fontsize=7)
        if idx == 0:
            ax.legend(fontsize=7, loc='upper left')
        ax.grid(alpha=0.2)
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    for idx in range(len(ORDER), len(axes)):
        axes[idx].set_visible(False)
    plt.tight_layout()
    out = f'{RESULTS_DIR}/predicted_vs_actual_all_models.png'
    plt.savefig(out, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'  Saved → {out}')

# ══════════════════════════════════════════════════════════════════════════════
# 6. All-model equity curves + consensus bar
# ══════════════════════════════════════════════════════════════════════════════
def plot_equity_all(dpi):
    path = f'{RESULTS_DIR}/test_predictions.csv'
    if not os.path.exists(path):
        print(f'  [skip] {path} not found'); return

    pred_df    = load_predictions()
    metrics_df = load_metrics_df()
    dates      = pred_df['date'].values
    log_rets   = pred_df['log_return'].values
    y_test     = pred_df['y_test'].values

    # Need gold price — load from data
    try:
        gdf = pd.read_csv('data/merged_gold_dataset.csv', index_col=0, parse_dates=True)
        gdf.index = pd.to_datetime(gdf.index).normalize()
        gold_all = gdf['Gold']
    except Exception:
        gold_all = None

    model_names = [c.replace('pred_', '') for c in pred_df.columns if c.startswith('pred_')]
    ORDER = ['Ridge','RandomForest','XGBoost','Lasso','ARIMAX',
             'BiLSTM','GRU','LSTM','MLP','SVR_lin','SVR_rbf']
    ORDER = [m for m in ORDER if m in model_names]

    # Compute equity + signal per model
    equities = {}
    signals  = {}
    sharpes  = {}
    for name in ORDER:
        preds = pred_df[f'pred_{name}'].dropna().values
        n     = len(preds)
        sig   = rolling_zscore_signal(preds)
        eq    = equity_curve(sig, log_rets[:n])
        equities[name] = (eq, dates[:n])
        signals[name]  = sig
        sharpes[name]  = float(metrics_df.loc[name, 'Sharpe']) \
                         if name in metrics_df.index else 0

    bnh_eq = equity_curve(np.ones(len(log_rets)), log_rets)
    bnh_cagr = float(bnh_eq[-1] ** (252/len(bnh_eq)) - 1)

    # Color map
    STRONG_COLORS = ['#D4AF37','#2E7D32','#1565C0','#6A1B9A','#E65100','#00838F']
    strong = sorted([n for n in ORDER if sharpes[n] >= 7.5], key=lambda x: -sharpes[x])
    pal    = {m: STRONG_COLORS[i % len(STRONG_COLORS)] for i, m in enumerate(strong)}
    cmap   = {m: model_color(m, sharpes[m], pal) for m in ORDER}

    fig = plt.figure(figsize=(16, 12))
    has_gold = gold_all is not None
    ratios   = [1, 2.5, 0.8] if has_gold else [2.5, 0.8]
    n_rows   = 3 if has_gold else 2
    gs       = gridspec.GridSpec(n_rows, 1, height_ratios=ratios, hspace=0.08)

    row = 0
    if has_gold:
        ax_gold = fig.add_subplot(gs[row]); row += 1
        mask = (gdf.index >= pd.Timestamp(dates[0])) & (gdf.index <= pd.Timestamp(dates[-1]))
        ax_gold.plot(gdf.index[mask], gold_all[mask], color='#B8860B', lw=1.5)
        ax_gold.set_ylabel('Gold Price (USD)', fontsize=10)
        ax_gold.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))
        ax_gold.tick_params(labelbottom=False)
        ax_gold.grid(axis='y', alpha=0.3)
        start_str = pd.Timestamp(dates[0]).strftime('%Y-%m-%d')
        end_str   = pd.Timestamp(dates[-1]).strftime('%Y-%m-%d')
        ax_gold.set_title(
            f'Gold Price Prediction — Capital Growth vs Buy & Hold\n'
            f'Test: {start_str} → {end_str}',
            fontsize=12, fontweight='bold', pad=8)

    ax_eq = fig.add_subplot(gs[row]); row += 1
    ax_eq.plot(dates, bnh_eq, color='#B8860B', lw=2.5, ls='--',
               label=f'Buy & Hold  (CAGR {bnh_cagr:+.1%})', zorder=5)
    ax_eq.axhline(1.0, color='grey', lw=0.8, ls=':')

    # Underperformers first (background)
    for name in ORDER:
        if sharpes[name] < 1.0:
            eq, dt = equities[name]
            cagr   = eq[-1] ** (252/len(eq)) - 1
            da     = float(metrics_df.loc[name, 'DirAcc']) if name in metrics_df.index else 0
            label  = (f'{name}  (CAGR {cagr:+.0%} | Sharpe {sharpes[name]:.2f} | '
                      f'DirAcc {da*100:.1f}%)')
            ax_eq.plot(dt, eq, color=cmap[name], lw=0.8, alpha=0.55, label=label)
    # Mid-tier
    for name in ORDER:
        if 1.0 <= sharpes[name] < 7.5:
            eq, dt = equities[name]
            cagr   = eq[-1] ** (252/len(eq)) - 1
            da     = float(metrics_df.loc[name, 'DirAcc']) if name in metrics_df.index else 0
            label  = (f'{name}  (CAGR {cagr:+.0%} | Sharpe {sharpes[name]:.2f} | '
                      f'DirAcc {da*100:.1f}%)')
            ax_eq.plot(dt, eq, color=cmap[name], lw=1.2, ls='-.', alpha=0.75,
                       label=label, zorder=6)
    # Strong models on top
    for name in ORDER:
        if sharpes[name] >= 7.5:
            eq, dt = equities[name]
            cagr   = eq[-1] ** (252/len(eq)) - 1
            da     = float(metrics_df.loc[name, 'DirAcc']) if name in metrics_df.index else 0
            label  = (f'{name}  (CAGR {cagr:+.0%} | Sharpe {sharpes[name]:.2f} | '
                      f'DirAcc {da*100:.1f}%)')
            ax_eq.plot(dt, eq, color=cmap[name], lw=2.2, label=label, zorder=10)

    ax_eq.set_ylabel('Portfolio Value ($1 start)', fontsize=10)
    ax_eq.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:.1f}'))
    ax_eq.legend(loc='upper left', fontsize=8, framealpha=0.9)
    ax_eq.tick_params(labelbottom=False)
    ax_eq.grid(alpha=0.25)
    if not has_gold:
        ax_eq.set_title(
            f'Capital Growth vs Buy & Hold | Test: '
            f'{pd.Timestamp(dates[0]).strftime("%Y-%m-%d")} → '
            f'{pd.Timestamp(dates[-1]).strftime("%Y-%m-%d")}',
            fontsize=12, fontweight='bold')

    # Consensus bar
    ax_vote = fig.add_subplot(gs[row])
    vote = np.zeros(len(dates))
    n_v  = 0
    for name in ORDER:
        sig = signals[name]
        n   = min(len(sig), len(dates))
        vote[:n] += sig[:n]
        n_v += 1
    pct_long = (vote + n_v) / (2 * n_v)
    cmap_rg  = LinearSegmentedColormap.from_list('rg', ['#cc3333','#dddddd','#2a9d2a'])
    dt_arr   = pd.to_datetime(dates)
    for i in range(len(dt_arr) - 1):
        ax_vote.axvspan(dt_arr[i], dt_arr[i+1],
                        facecolor=cmap_rg(pct_long[i]), alpha=0.85, linewidth=0)
    ax_vote.axhline(0.5, color='black', lw=0.5, ls=':')
    ax_vote.set_ylim(0, 1)
    ax_vote.set_yticks([0, 0.5, 1])
    ax_vote.set_yticklabels(['All SHORT', '50/50', 'All LONG'], fontsize=8)
    ax_vote.set_ylabel('Model\nConsensus', fontsize=9)
    ax_vote.set_xlabel('Date', fontsize=10)
    ax_vote.grid(axis='x', alpha=0.2)
    ax_vote.xaxis.set_major_locator(mdates.YearLocator())
    ax_vote.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    out = f'{RESULTS_DIR}/equity_curves_all_models.png'
    plt.savefig(out, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'  Saved → {out}')

# ══════════════════════════════════════════════════════════════════════════════
# 7. Signal timeline (per-model long/short strips)
# ══════════════════════════════════════════════════════════════════════════════
def plot_signal_timeline(dpi):
    path = f'{RESULTS_DIR}/test_predictions.csv'
    if not os.path.exists(path):
        print(f'  [skip] {path} not found'); return

    pred_df    = load_predictions()
    metrics_df = load_metrics_df()
    dates      = pd.to_datetime(pred_df['date'].values)
    log_rets   = pred_df['log_return'].values

    model_names = [c.replace('pred_', '') for c in pred_df.columns if c.startswith('pred_')]
    ORDER = ['Ridge','RandomForest','XGBoost','Lasso',
             'BiLSTM','GRU','LSTM','MLP','SVR_lin','SVR_rbf']
    ORDER = [m for m in ORDER if m in model_names]

    STRONG_COLORS = ['#D4AF37','#2E7D32','#1565C0','#6A1B9A','#E65100','#00838F']
    sharpes = {m: float(metrics_df.loc[m, 'Sharpe']) if m in metrics_df.index else 0
               for m in ORDER}
    strong  = sorted([n for n in ORDER if sharpes[n] >= 7.5], key=lambda x: -sharpes[x])
    pal     = {m: STRONG_COLORS[i % len(STRONG_COLORS)] for i, m in enumerate(strong)}
    cmap    = {m: model_color(m, sharpes[m], pal) for m in ORDER}

    n_m   = len(ORDER)
    fig2, axes = plt.subplots(n_m + 2, 1,
                              figsize=(16, n_m * 0.85 + 4.5),
                              gridspec_kw={'height_ratios': [2.5] + [1]*n_m + [0.6]})
    fig2.subplots_adjust(hspace=0.06, left=0.12, right=0.97, top=0.95, bottom=0.05)
    fig2.suptitle(
        f"Gold Trading Signals — Long/Short Actions Over Time\n"
        f"Test: {dates[0].strftime('%Y-%m-%d')} → {dates[-1].strftime('%Y-%m-%d')}",
        fontsize=13, fontweight='bold')

    # Gold price (try loading)
    try:
        gdf = pd.read_csv('data/merged_gold_dataset.csv', index_col=0, parse_dates=True)
        gdf.index = pd.to_datetime(gdf.index).normalize()
        mask  = (gdf.index >= dates[0]) & (gdf.index <= dates[-1])
        gold_dt = gdf.index[mask]
        gold_px = gdf['Gold'].values[mask]
    except Exception:
        gold_dt, gold_px = dates, np.ones(len(dates))

    ax0 = axes[0]
    ax0.plot(gold_dt, gold_px, color='#B8860B', lw=1.6)
    ax0.fill_between(gold_dt, gold_px, gold_px.min(), alpha=0.08, color='#B8860B')
    ax0.set_ylabel('Gold\nPrice ($)', fontsize=9)
    ax0.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))
    ax0.tick_params(labelbottom=False)
    ax0.grid(axis='y', alpha=0.3)
    ax0.set_xlim(dates[0], dates[-1])
    ax0.annotate(f'${gold_px[0]:,.0f}', xy=(gold_dt[0], gold_px[0]),
                 xytext=(8, 6), textcoords='offset points', fontsize=8, color='#7a5c00')
    ax0.annotate(f'${gold_px[-1]:,.0f}', xy=(gold_dt[-1], gold_px[-1]),
                 xytext=(-50, 6), textcoords='offset points', fontsize=8, color='#7a5c00')

    for row, name in enumerate(ORDER):
        ax  = axes[row + 1]
        p   = pred_df[f'pred_{name}'].dropna().values
        n   = len(p)
        sig = rolling_zscore_signal(p)
        dt  = dates[:n]
        eq  = equity_curve(sig, log_rets[:n])
        cagr   = eq[-1] ** (252/n) - 1
        da     = float(metrics_df.loc[name,'DirAcc']) if name in metrics_df.index else 0
        star   = ' ★' if sharpes[name] >= 7.5 else ''
        label  = f'{name}{star}\nSharpe {sharpes[name]:.2f}\nDirAcc {da*100:.1f}%'

        changes = np.where(np.diff(sig, prepend=sig[0]))[0]
        bounds  = np.concatenate([changes, [len(sig)]])
        prev    = 0
        for b in bounds[1:]:
            if prev >= len(dt) or b > len(dt): break
            color = '#2ecc71' if sig[prev] > 0 else '#e74c3c'
            ax.axvspan(dt[prev], dt[min(b, len(dt)-1)],
                       facecolor=color, alpha=0.85, linewidth=0)
            prev = b

        ax2 = ax.twinx()
        ax2.plot(dt, eq, color=cmap[name], lw=1.2, alpha=0.8)
        ax2.set_ylim(0, max(eq) * 1.1)
        ax2.set_ylabel(f'×{eq[-1]:.1f}', fontsize=7.5, color=cmap[name], rotation=0, labelpad=22)
        ax2.tick_params(right=False, labelright=False)
        ax.text(1.001, 0.5, f'{cagr:+.0%}', transform=ax.transAxes,
                fontsize=7.5, va='center', color=cmap[name],
                fontweight='bold' if sharpes[name] >= 7.5 else 'normal')

        ax.set_xlim(dates[0], dates[-1])
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_ylabel(label, fontsize=7.5, rotation=0, ha='right', va='center', labelpad=5)
        ax.tick_params(labelbottom=False)

    ax_leg = axes[-1]
    ax_leg.set_visible(False)
    fig2.legend(
        handles=[mpatches.Patch(color='#2ecc71', alpha=0.85, label='LONG (Buy)'),
                 mpatches.Patch(color='#e74c3c', alpha=0.85, label='SHORT (Sell)'),
                 mpatches.Patch(color='none',    label='★ = Sharpe ≥ 7.5')],
        loc='lower center', ncol=3, fontsize=9,
        bbox_to_anchor=(0.5, 0.005), framealpha=0.9)

    axes[-2].tick_params(labelbottom=True)
    axes[-2].xaxis.set_major_locator(mdates.YearLocator())
    axes[-2].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    out = f'{RESULTS_DIR}/signal_timeline.png'
    plt.savefig(out, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'  Saved → {out}')

# ══════════════════════════════════════════════════════════════════════════════
# 8. Best model SHAP beeswarm
# ══════════════════════════════════════════════════════════════════════════════
def plot_shap_best(dpi):
    import shap
    sv_path = f'{RESULTS_DIR}/shap_best_values.npy'
    X_path  = f'{RESULTS_DIR}/shap_best_X.npy'
    if not os.path.exists(sv_path):
        print(f'  [skip] {sv_path} not found'); return
    meta          = load_meta()
    sv            = np.load(sv_path)
    X_te          = np.load(X_path)
    feat_names    = meta.get('selected_features', meta.get('feature_cols', []))

    plt.figure(figsize=(10, 7))
    shap.summary_plot(sv, X_te, feature_names=feat_names, show=False)
    plt.title(f'SHAP Feature Importance — {meta["best_model"]} (best model)',
              fontsize=11, fontweight='bold')
    plt.tight_layout()
    out = f'{RESULTS_DIR}/shap_summary.png'
    plt.savefig(out, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'  Saved → {out}')

# ══════════════════════════════════════════════════════════════════════════════
# 9. All-model SHAP bar grid
# ══════════════════════════════════════════════════════════════════════════════
def plot_shap_all(dpi):
    npz_path = f'{RESULTS_DIR}/shap_all_models.npz'
    if not os.path.exists(npz_path):
        print(f'  [skip] {npz_path} not found'); return
    meta       = load_meta()
    metrics_df = load_metrics_df()
    feat_cols  = meta.get('feature_cols', [])
    data       = np.load(npz_path, allow_pickle=True)
    all_shap   = {k: data[k] for k in data.files}

    ORDER = ['Ridge','Lasso','RandomForest','XGBoost',
             'SVR_lin','SVR_rbf','MLP','BiLSTM','GRU','LSTM']
    ORDER = [m for m in ORDER if m in all_shap]

    ncols = 2
    nrows = (len(ORDER) + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4.5 * nrows))
    axes = axes.flatten()
    fig.suptitle('SHAP Feature Importance — All Models  (Top 15, Mean |SHAP Value|)',
                 fontsize=13, fontweight='bold')

    for idx, name in enumerate(ORDER):
        ax       = axes[idx]
        sv       = all_shap[name]
        mean_abs = np.abs(sv).mean(axis=0)
        k        = min(15, len(mean_abs))
        top      = np.argsort(mean_abs)[::-1][:k]
        sharpe   = float(metrics_df.loc[name, 'Sharpe']) if name in metrics_df.index else 0
        da       = float(metrics_df.loc[name, 'DirAcc']) if name in metrics_df.index else 0
        star     = ' ★' if sharpe >= 7.5 else ''
        color    = '#D4AF37' if sharpe >= 7.5 else 'steelblue'
        labels   = [feat_cols[i] if i < len(feat_cols) else str(i) for i in top[::-1]]
        values   = mean_abs[top[::-1]]
        ax.barh(labels, values, color=color)
        ax.set_title(f'{name}{star}  Sharpe={sharpe:.2f}  DirAcc={da*100:.1f}%', fontsize=9)
        ax.set_xlabel('Mean |SHAP value|', fontsize=8)
        ax.tick_params(axis='y', labelsize=7)

    for idx in range(len(ORDER), len(axes)):
        axes[idx].set_visible(False)
    plt.tight_layout()
    out = f'{RESULTS_DIR}/shap_all_models.png'
    plt.savefig(out, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'  Saved → {out}')

# ══════════════════════════════════════════════════════════════════════════════
# 10. Bias tests (WRC + MC permutation)
# ══════════════════════════════════════════════════════════════════════════════
def plot_bias_tests(dpi):
    path = f'{RESULTS_DIR}/bias_test_data.json'
    if not os.path.exists(path):
        print(f'  [skip] {path} not found'); return
    with open(path) as f:
        d = json.load(f)
    null_wrc = np.array(d['null_wrc'])
    null_mc  = np.array(d['null_mc'])

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    axes[0].hist(null_wrc, bins=50, color='steelblue', alpha=0.7)
    axes[0].axvline(d['obs_wrc'], color='tomato', lw=2,
                    label=f'obs={d["obs_wrc"]:.5f}')
    axes[0].set_title(f'White Reality Check  p={d["wrc_p"]:.4f}', fontsize=11)
    axes[0].set_xlabel('Mean strategy return (bootstrap)')
    axes[0].legend()
    axes[1].hist(null_mc, bins=50, color='steelblue', alpha=0.7)
    axes[1].axvline(d['obs_mc'], color='tomato', lw=2,
                    label=f'obs={d["obs_mc"]:.5f}')
    axes[1].set_title(f'MC Permutation Test  p={d["mc_p"]:.4f}', fontsize=11)
    axes[1].set_xlabel('Mean shuffled return')
    axes[1].legend()
    plt.suptitle('Selection Bias Tests — Best Model Strategy', fontsize=12, fontweight='bold')
    plt.tight_layout()
    out = f'{RESULTS_DIR}/bias_tests.png'
    plt.savefig(out, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'  Saved → {out}')

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
CHART_FNS = {
    'cv_mse':         plot_cv_mse,
    'pva_best':       plot_pva_best,
    'equity_best':    plot_equity_best,
    'trading_metrics':plot_trading_metrics,
    'pva_all':        plot_pva_all,
    'equity_all':     plot_equity_all,
    'signal_timeline':plot_signal_timeline,
    'shap_best':      plot_shap_best,
    'shap_all':       plot_shap_all,
    'bias_tests':     plot_bias_tests,
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate all result charts.')
    parser.add_argument('--only', nargs='+', choices=ALL_CHARTS,
                        help='Generate only these charts (default: all)')
    parser.add_argument('--dpi', type=int, default=150,
                        help='Output resolution (default: 150)')
    parser.add_argument('--target', type=str,
                        help='Subdirectory to plot, e.g. close_3y, open_full, close_full')
    args = parser.parse_args()

    to_run = args.only or ALL_CHARTS
    print(f'\nGenerating {len(to_run)} chart(s)  [dpi={args.dpi}]\n')

    for key in to_run:
        print(f'── {key}')
        try:
            CHART_FNS[key](args.dpi)
        except Exception as e:
            print(f'  [error] {e}')

    print('\nDone.')

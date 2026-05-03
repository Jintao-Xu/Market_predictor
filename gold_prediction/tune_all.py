#!/usr/bin/env python3
"""
tune_all.py — Tune all (or selected) models in parallel subprocesses.

Each model runs as an independent process (tune_model.py --model <name>)
with its own log file. Sklearn models each get --n-jobs 2 so they don't
fight for cores; Keras models run single-threaded (TF manages its own cores).

Usage:
    python tune_all.py                          # tune all 10 models
    python tune_all.py --models SVR_lin SVR_rbf # tune only SVR models
    python tune_all.py --models Ridge XGBoost LSTM
    python tune_all.py --skip-keras             # sklearn + ARIMAX only
    python tune_all.py --jobs-per-model 4       # cores per sklearn model (default 2)
"""

import os, sys, json, time, subprocess, argparse
from pathlib import Path
from datetime import datetime

MODELS_DIR = 'saved_models'
LOG_DIR    = '/tmp/tune_gold_logs'
PYTHON     = sys.executable   # same venv that launched this script

SKLEARN_MODELS = ['Ridge', 'Lasso', 'SVR_lin', 'SVR_rbf',
                  'XGBoost', 'RandomForest', 'MLP']
KERAS_MODELS   = ['LSTM', 'GRU', 'BiLSTM']
ALL_MODELS     = SKLEARN_MODELS + KERAS_MODELS

# Metrics to show in the summary table
SUMMARY_COLS = ['cv_mse', 'Sharpe', 'DirAcc', 'CAGR', 'MaxDD', 'PredStd']

# ── Helpers ───────────────────────────────────────────────────────────────────
def tail(path, n=8):
    """Return last n lines of a file, or empty string if missing."""
    try:
        lines = Path(path).read_text().splitlines()
        return '\n'.join(lines[-n:])
    except Exception:
        return ''

def read_result(model_name):
    """Load per-model tuning result JSON, or None if not finished yet."""
    path = f'{MODELS_DIR}/tuned_{model_name}.json'
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None

def status_line(name, proc, start_time):
    """One-line status for a running/finished process."""
    elapsed = time.time() - start_time
    rc = proc.poll()
    if rc is None:
        state = f'RUNNING  {elapsed:5.0f}s'
    elif rc == 0:
        state = f'DONE     {elapsed:5.0f}s'
    else:
        state = f'FAILED({rc}) {elapsed:5.0f}s'
    return f'  {name:<14} {state}'

def print_status(procs, starts):
    """Print live status table."""
    print('\n' + '─'*45)
    for name, proc in procs.items():
        print(status_line(name, proc, starts[name]))
    print('─'*45)

def print_summary(models):
    """Print final results table for all finished models."""
    results = {}
    for name in models:
        r = read_result(name)
        if r:
            results[name] = r

    if not results:
        print('  No results found.')
        return

    # Header
    col_w = 10
    header = f'  {"Model":<14}' + ''.join(f'{c:>{col_w}}' for c in SUMMARY_COLS)
    print('\n' + '='*70)
    print('  TUNING RESULTS SUMMARY')
    print('='*70)
    print(header)
    print('  ' + '-'*68)

    # Sort by Sharpe descending
    sorted_results = sorted(results.items(),
                            key=lambda x: x[1].get('Sharpe', -999), reverse=True)
    for name, r in sorted_results:
        row = f'  {name:<14}'
        for col in SUMMARY_COLS:
            val = r.get(col, float('nan'))
            if col == 'DirAcc':
                row += f'{val*100:>{col_w}.1f}%'[:-1].rjust(col_w)   # show as %
            elif isinstance(val, float):
                row += f'{val:>{col_w}.5f}'
            else:
                row += f'{str(val):>{col_w}}'
        star = ' ★' if r.get('Sharpe', 0) >= 7.5 else ''
        print(row + star)

    print('='*70)
    print('\n  Best params per model:')
    for name, r in sorted_results:
        bp = r.get('best_params', {})
        print(f'    {name:<14} {bp}')

    # Merge all into model_metadata.json
    try:
        meta_path = f'{MODELS_DIR}/model_metadata.json'
        with open(meta_path) as f:
            meta = json.load(f)
        for name, r in results.items():
            meta[f'tuned_{name}'] = r
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)
        print(f'\n  All results merged into {meta_path}')
    except Exception as e:
        print(f'  Could not merge into metadata: {e}')

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tune all models in parallel.')
    parser.add_argument('--models', nargs='+', choices=ALL_MODELS,
                        default=None, help='Subset of models to tune (default: all)')
    parser.add_argument('--skip-keras', action='store_true',
                        help='Skip LSTM / GRU / BiLSTM')
    parser.add_argument('--jobs-per-model', type=int, default=2,
                        help='n_jobs for sklearn GridSearchCV per model (default: 2). '
                             'Total cores used ≈ n_models × jobs_per_model.')
    args = parser.parse_args()

    models_to_run = args.models or ALL_MODELS
    if args.skip_keras:
        models_to_run = [m for m in models_to_run if m not in KERAS_MODELS]

    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    print(f'\n{"="*60}')
    print(f'  tune_all.py — parallel model tuning')
    print(f'  Models    : {models_to_run}')
    print(f'  Jobs/model: {args.jobs_per_model} (sklearn); TF-managed (Keras)')
    print(f'  Logs      : {LOG_DIR}/')
    print(f'  Started   : {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'{"="*60}\n')

    # ── Launch all models as subprocesses ─────────────────────────────────────
    procs  = {}
    starts = {}
    logs   = {}

    for name in models_to_run:
        log_path = f'{LOG_DIR}/tune_{name}.log'
        log_file = open(log_path, 'w')
        n_jobs   = 1 if name in KERAS_MODELS else args.jobs_per_model
        cmd = [PYTHON, 'tune_model.py', '--model', name,
               '--n-jobs', str(n_jobs)]
        proc = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            cwd=os.getcwd(),
        )
        procs[name]  = proc
        starts[name] = time.time()
        logs[name]   = log_path
        print(f'  Launched {name:<14} PID={proc.pid}  log → {log_path}')

    print(f'\n  {len(procs)} processes running. Monitoring...\n')

    # ── Monitor loop ──────────────────────────────────────────────────────────
    poll_interval = 15   # seconds between status prints
    last_print    = time.time()

    while True:
        alive = [p for p in procs.values() if p.poll() is None]
        done  = [n for n, p in procs.items() if p.poll() == 0]
        failed= [n for n, p in procs.items() if p.poll() not in (None, 0)]

        now = time.time()
        if now - last_print >= poll_interval:
            print_status(procs, starts)
            # Show last lines of any recently finished model
            for name in done:
                result = read_result(name)
                if result:
                    sharpe = result.get('Sharpe', '?')
                    dacc   = result.get('DirAcc', 0)
                    bp     = result.get('best_params', {})
                    print(f'    {name} done → Sharpe={sharpe:.3f}  '
                          f'DirAcc={dacc*100:.1f}%  best={bp}')
            if failed:
                for name in failed:
                    rc = procs[name].poll()
                    print(f'    {name} FAILED (exit {rc}) — tail of log:')
                    for line in tail(logs[name], 6).splitlines():
                        print(f'      {line}')
            last_print = now

        if not alive:
            break
        time.sleep(3)

    # ── Final status ──────────────────────────────────────────────────────────
    print_status(procs, starts)

    failed = [n for n, p in procs.items() if p.poll() not in (None, 0)]
    if failed:
        print(f'\n  FAILED models: {failed}')
        for name in failed:
            print(f'\n  ── {name} log (last 15 lines) ──')
            print(tail(logs[name], 15))

    # ── Summary table ─────────────────────────────────────────────────────────
    print_summary(models_to_run)

    total = max(time.time() - s for s in starts.values())
    print(f'\n  Total wall-clock time: {total:.0f}s')
    print(f'  Individual logs: {LOG_DIR}/')
    print('\nDone.')

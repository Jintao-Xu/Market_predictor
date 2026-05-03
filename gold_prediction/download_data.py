"""
download_data.py — Download all data sources for the gold prediction model.

Downloads and saves:
  gold_daily.csv          — Gold futures (GC=F) daily OHLCV
  dxy_daily.csv           — USD Index (DX-Y.NYB)
  vix_daily.csv           — CBOE VIX
  tnx_daily.csv           — US 10Y Treasury yield (^TNX)
  tyx_daily.csv           — US 30Y Treasury yield (^TYX)
  cot_gold_weekly.csv     — CFTC COT gold futures (weekly, Tuesday release)
  merged_gold_dataset.csv — All sources merged on trading-day index (COT forward-filled)

Run once before opening the notebook:
    python download_data.py
"""

import os
import datetime
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yfinance as yf

START_DATE = datetime.datetime(2006, 1, 1)
END_DATE   = datetime.datetime.now()

DATA_DIR = 'data'
os.makedirs(DATA_DIR, exist_ok=True)

# ── 1. Gold futures ───────────────────────────────────────────────────────────
print('Downloading Gold futures (GC=F)...')
gold_raw = yf.download('GC=F', start=START_DATE, end=END_DATE, interval='1d', auto_adjust=True)
# yfinance may return MultiIndex columns — flatten to single level
if isinstance(gold_raw.columns, pd.MultiIndex):
    gold_raw.columns = gold_raw.columns.get_level_values(0)
gold_raw.index = pd.to_datetime(gold_raw.index).normalize()
gold_raw.dropna(subset=['Close'], inplace=True)
gold = gold_raw[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
gold.columns = ['Open', 'High', 'Low', 'Gold', 'Volume']
gold['log_Gold'] = np.log(gold['Gold'])
gold.to_csv(f'{DATA_DIR}/gold_daily.csv')
print(f'  gold_daily.csv: {len(gold)} rows  ({gold.index[0].date()} → {gold.index[-1].date()})')

# ── 2. Macro indicators ───────────────────────────────────────────────────────
macro_tickers = {
    'DX-Y.NYB': ('DXY',       'dxy_daily.csv'),
    '^VIX':     ('VIX',       'vix_daily.csv'),
    '^TNX':     ('TNX_yield', 'tnx_daily.csv'),
    '^TYX':     ('TYX_yield', 'tyx_daily.csv'),
}

macro_frames = {}
for ticker, (col_name, filename) in macro_tickers.items():
    print(f'Downloading {col_name} ({ticker})...')
    try:
        raw = yf.download(ticker, start=START_DATE, end=END_DATE, interval='1d', auto_adjust=True)
        if raw.empty:
            print(f'  WARNING: {ticker} returned empty data')
            continue
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        s = raw[['Close']].copy()
        s.index = pd.to_datetime(s.index).normalize()
        s.columns = [col_name]
        s.dropna(inplace=True)
        s.to_csv(f'{DATA_DIR}/{filename}')
        macro_frames[col_name] = s
        print(f'  {filename}: {len(s)} rows  ({s.index[0].date()} → {s.index[-1].date()})')
    except Exception as e:
        print(f'  ERROR: {e}')

# ── 3. COT data (CFTC gold futures) ──────────────────────────────────────────
print('\nDownloading CFTC COT data (gold futures)...')

COT_COL_MAP = {
    'Market and Exchange Names':            'Market_and_Exchange_Names',
    'Commercial Positions-Long (All)':      'Comm_Positions_Long_All',
    'Commercial Positions-Short (All)':     'Comm_Positions_Short_All',
    'Noncommercial Positions-Long (All)':   'NonComm_Positions_Long_All',
    'Noncommercial Positions-Short (All)':  'NonComm_Positions_Short_All',
    'Open Interest (All)':                  'Open_Interest_All',
    'Nonreportable Positions-Long (All)':   'NonRept_Positions_Long_All',
    'Nonreportable Positions-Short (All)':  'NonRept_Positions_Short_All',
    'As of Date in Form YYMMDD':            'As_of_Date_in_Form_YYMMDD',
    'As of Date in Form YYYY-MM-DD':        'As_of_Date_in_Form_YYYY_MM_DD',
}

cot_gold = None
try:
    import cot_reports as cot
    year_end = datetime.datetime.now().year
    frames = []
    for y in range(2006, year_end + 1):
        try:
            df_y = cot.cot_year(year=y, cot_report_type='legacy_fut', store_txt=False, verbose=False)
            df_y = df_y.rename(columns={k: v for k, v in COT_COL_MAP.items() if k in df_y.columns})
            frames.append(df_y)
            print(f'  {y} ✓', end='\r')
        except Exception as e:
            print(f'  {y} skip: {e}')

    if frames:
        cot_raw = pd.concat(frames, ignore_index=True)
        print(f'\n  Loaded {len(frames)} years of COT data')

        # Filter to COMEX full-size gold futures only
        # "GOLD - COMMODITY EXCHANGE INC." is the standard 100-oz COMEX contract
        # Exclude: MINI GOLD, MICRO GOLD, GOLDMAN-SACHS INDEX, COINBASE, CBT (100 TROY OZ)
        names = cot_raw['Market_and_Exchange_Names'].astype(str)
        gold_mask   = names.str.contains('GOLD', case=False, na=False)
        comex_mask  = names.str.contains('COMMODITY EXCHANGE', case=False, na=False)
        exclude_mask = (names.str.contains('MINI',    case=False, na=False) |
                        names.str.contains('MICRO',   case=False, na=False) |
                        names.str.contains('GOLDMAN', case=False, na=False) |
                        names.str.contains('COINBASE',case=False, na=False) |
                        names.str.contains('TROY OZ', case=False, na=False))
        cot_gold_raw = cot_raw[gold_mask & comex_mask & ~exclude_mask].copy()

        print(f'  Contract names found:')
        for name in cot_gold_raw['Market_and_Exchange_Names'].unique():
            print(f'    {name}')

        # Parse date
        if 'As_of_Date_in_Form_YYYY_MM_DD' in cot_gold_raw.columns:
            cot_gold_raw['Date'] = pd.to_datetime(cot_gold_raw['As_of_Date_in_Form_YYYY_MM_DD'], errors='coerce')
        else:
            cot_gold_raw['Date'] = pd.to_datetime(
                cot_gold_raw['As_of_Date_in_Form_YYMMDD'].astype(str).str.replace('.0', '', regex=False),
                format='%y%m%d', errors='coerce')

        cot_gold_raw = (cot_gold_raw
                        .dropna(subset=['Date'])
                        .drop_duplicates(subset=['Date'], keep='last')
                        .set_index('Date')
                        .sort_index())

        keep_cols = [
            'Comm_Positions_Long_All', 'Comm_Positions_Short_All',
            'NonComm_Positions_Long_All', 'NonComm_Positions_Short_All',
            'Open_Interest_All',
            'NonRept_Positions_Long_All', 'NonRept_Positions_Short_All',
        ]
        cot_gold = cot_gold_raw[[c for c in keep_cols if c in cot_gold_raw.columns]]
        cot_gold.to_csv(f'{DATA_DIR}/cot_gold_weekly.csv')
        print(f'  cot_gold_weekly.csv: {len(cot_gold)} rows  ({cot_gold.index[0].date()} → {cot_gold.index[-1].date()})')
    else:
        print('  No COT frames loaded')

except ImportError:
    print('  cot_reports not installed — skipping COT download')
    print('  Install with: pip install cot_reports')
except Exception as e:
    print(f'  COT download error: {e}')

# ── 4. Merge all sources ──────────────────────────────────────────────────────
print('\nMerging all sources...')
df = gold[['Gold', 'log_Gold']].copy()

# COT: forward-fill weekly data to daily
if cot_gold is not None and len(cot_gold) > 0:
    cot_daily = cot_gold.reindex(df.index, method='ffill')
    df = df.join(cot_daily, how='left')
    print(f'  COT columns joined ({len(cot_gold.columns)} cols, ffilled to daily)')

# Macro: left join + ffill
for col_name, series in macro_frames.items():
    series.index = pd.to_datetime(series.index).normalize()
    df = df.join(series, how='left')
    df[col_name] = df[col_name].ffill()
    print(f'  {col_name} joined: {df[col_name].notna().sum()} non-null rows')

df.dropna(subset=['Gold'], inplace=True)
df.to_csv(f'{DATA_DIR}/merged_gold_dataset.csv')
print(f'\nmerged_gold_dataset.csv: {df.shape[0]} rows × {df.shape[1]} cols')
print(f'Date range: {df.index[0].date()} → {df.index[-1].date()}')
print(f'Columns: {list(df.columns)}')

# ── Summary ───────────────────────────────────────────────────────────────────
print('\n' + '=' * 55)
print('  DOWNLOAD COMPLETE — files in data/')
print('=' * 55)
for fname in sorted(os.listdir(DATA_DIR)):
    fpath = f'{DATA_DIR}/{fname}'
    size_kb = os.path.getsize(fpath) / 1024
    print(f'  {fname:<35} {size_kb:>8.1f} KB')

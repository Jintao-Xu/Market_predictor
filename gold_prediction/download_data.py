"""
download_data.py — Download / incrementally update all data sources.

On first run: downloads full history from 2006-01-01.
On subsequent runs: only fetches rows newer than the latest date already on disk,
then appends and re-merges.  Typically completes in a few seconds.

Outputs (written to data/):
  gold_daily.csv          — Gold futures (GC=F) daily OHLCV
  dxy_daily.csv           — USD Index (DX-Y.NYB)
  vix_daily.csv           — CBOE VIX
  tnx_daily.csv           — US 10Y Treasury yield (^TNX)
  tyx_daily.csv           — US 30Y Treasury yield (^TYX)
  cot_gold_weekly.csv     — CFTC COT gold futures (weekly, Tuesday release)
  merged_gold_dataset.csv — All sources merged on trading-day index (COT forward-filled)

Usage:
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
TODAY      = datetime.datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

DATA_DIR = 'data'
os.makedirs(DATA_DIR, exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def last_date_in_csv(path):
    """Return the latest index date in an existing CSV, or None if missing/empty."""
    if os.path.exists(path):
        try:
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            if not df.empty:
                return pd.Timestamp(df.index[-1]).normalize()
        except Exception:
            pass
    return None


def fetch_yf_incremental(ticker, csv_path, col_name, extra_cols=None):
    """
    Download only rows newer than the latest date already in csv_path.
    Returns a DataFrame with the full (existing + new) series.

    extra_cols: dict of {new_col: transform_fn(df)} added after download (e.g. log_Gold).
    """
    last = last_date_in_csv(csv_path)
    fetch_start = (last + datetime.timedelta(days=1)) if last is not None else START_DATE

    # Already up to date
    if last is not None and fetch_start > TODAY:
        print(f'  {col_name}: already up to date ({last.date()})')
        return pd.read_csv(csv_path, index_col=0, parse_dates=True)

    print(f'  {col_name}: fetching {fetch_start.date()} → today', end=' ... ')
    raw = yf.download(ticker, start=fetch_start, end=TODAY + datetime.timedelta(days=1),
                      interval='1d', auto_adjust=True, progress=False)

    if raw.empty:
        print('no new data')
        return pd.read_csv(csv_path, index_col=0, parse_dates=True) if last else pd.DataFrame()

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw.index = pd.to_datetime(raw.index).normalize()
    raw.dropna(subset=['Close'], inplace=True)

    # Apply extra derived columns before selecting
    if extra_cols:
        for new_col, fn in extra_cols.items():
            raw[new_col] = fn(raw)

    # Keep only columns that exist in raw
    keep = [c for c in (['Close'] + list(extra_cols.keys() if extra_cols else []))
            if c in raw.columns]
    new_rows = raw[keep].copy()
    new_rows.columns = ([col_name] + list(extra_cols.keys() if extra_cols else []))

    # Load existing and append
    if last is not None:
        existing = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        # Drop any overlap (yfinance sometimes re-returns the last known date)
        new_rows = new_rows[new_rows.index > last]
        combined = pd.concat([existing, new_rows])
    else:
        combined = new_rows

    combined = combined[~combined.index.duplicated(keep='last')].sort_index()
    combined.to_csv(csv_path)
    n_new = len(new_rows)
    print(f'{n_new} new row(s) → {len(combined)} total  '
          f'({combined.index[0].date()} → {combined.index[-1].date()})')
    return combined


def fetch_yf_macro_incremental(ticker, col_name, csv_path):
    """Incremental fetch for a single-column macro series (DXY, VIX, TNX, TYX)."""
    last = last_date_in_csv(csv_path)
    fetch_start = (last + datetime.timedelta(days=1)) if last is not None else START_DATE

    if last is not None and fetch_start > TODAY:
        existing = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        if f'{col_name}_Open' in existing.columns and existing[f'{col_name}_Open'].notna().any():
            print(f'  {col_name}: already up to date ({last.date()})')
            return existing
        # Open column missing from cached CSV — re-fetch full history to add it
        fetch_start = START_DATE
        print(f'  {col_name}: re-fetching to add Open column', end=' ... ')
    else:
        print(f'  {col_name}: fetching {fetch_start.date()} → today', end=' ... ')
    try:
        raw = yf.download(ticker, start=fetch_start,
                          end=TODAY + datetime.timedelta(days=1),
                          interval='1d', auto_adjust=True, progress=False)
        if raw.empty:
            print('no new data')
            return pd.read_csv(csv_path, index_col=0, parse_dates=True) if last else pd.DataFrame()

        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        cols_to_save = [c for c in ['Open', 'Close'] if c in raw.columns]
        s = raw[cols_to_save].copy()
        s.index = pd.to_datetime(s.index).normalize()
        s.columns = [f'{col_name}_Open' if c == 'Open' else col_name for c in cols_to_save]
        s.dropna(subset=[col_name], inplace=True)

        if last is not None and fetch_start != START_DATE:
            # Incremental append: only add rows newer than last cached date
            existing = pd.read_csv(csv_path, index_col=0, parse_dates=True)
            new_rows = s[s.index > last]
            s = pd.concat([existing, new_rows])
        # If fetch_start == START_DATE (full re-fetch to add Open column), s already
        # contains the full history — just write it directly without merging old rows.

        s = s[~s.index.duplicated(keep='last')].sort_index()
        s.to_csv(csv_path)
        cutoff = last if last is not None else pd.Timestamp('1900-01-01')
        n_new  = int((s.index > cutoff).sum())
        print(f'{n_new} new row(s) → {len(s)} total  ({s.index[0].date()} → {s.index[-1].date()})')
        return s
    except Exception as e:
        print(f'ERROR: {e}')
        return pd.read_csv(csv_path, index_col=0, parse_dates=True) if last else pd.DataFrame()


# ══════════════════════════════════════════════════════════════════════════════
# 1. Gold futures
# ══════════════════════════════════════════════════════════════════════════════
print('Gold futures (GC=F):')
gold_last = last_date_in_csv(f'{DATA_DIR}/gold_daily.csv')
fetch_start_gold = (gold_last + datetime.timedelta(days=1)) if gold_last else START_DATE

if gold_last is not None and fetch_start_gold > TODAY:
    print(f'  already up to date ({gold_last.date()})')
    gold = pd.read_csv(f'{DATA_DIR}/gold_daily.csv', index_col=0, parse_dates=True)
else:
    print(f'  fetching {fetch_start_gold.date()} → today', end=' ... ')
    gold_raw = yf.download('GC=F', start=fetch_start_gold,
                           end=TODAY + datetime.timedelta(days=1),
                           interval='1d', auto_adjust=True, progress=False)
    if isinstance(gold_raw.columns, pd.MultiIndex):
        gold_raw.columns = gold_raw.columns.get_level_values(0)
    gold_raw.index = pd.to_datetime(gold_raw.index).normalize()
    gold_raw.dropna(subset=['Close'], inplace=True)

    new_gold = gold_raw[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    new_gold.columns = ['Open', 'High', 'Low', 'Gold', 'Volume']
    new_gold['log_Gold'] = np.log(new_gold['Gold'])

    if gold_last is not None:
        existing_gold = pd.read_csv(f'{DATA_DIR}/gold_daily.csv', index_col=0, parse_dates=True)
        new_gold = new_gold[new_gold.index > gold_last]
        gold = pd.concat([existing_gold, new_gold])
    else:
        gold = new_gold

    gold = gold[~gold.index.duplicated(keep='last')].sort_index()
    gold.to_csv(f'{DATA_DIR}/gold_daily.csv')
    print(f'{len(new_gold)} new row(s) → {len(gold)} total  '
          f'({gold.index[0].date()} → {gold.index[-1].date()})')

# ══════════════════════════════════════════════════════════════════════════════
# 2. Macro indicators
# ══════════════════════════════════════════════════════════════════════════════
print('\nMacro indicators:')
macro_tickers = {
    'DX-Y.NYB': ('DXY',       'dxy_daily.csv'),
    '^VIX':     ('VIX',       'vix_daily.csv'),
    '^TNX':     ('TNX_yield', 'tnx_daily.csv'),
    '^TYX':     ('TYX_yield', 'tyx_daily.csv'),
}

macro_frames = {}
for ticker, (col_name, filename) in macro_tickers.items():
    s = fetch_yf_macro_incremental(ticker, col_name, f'{DATA_DIR}/{filename}')
    if s is not None and not s.empty:
        macro_frames[col_name] = s

# ══════════════════════════════════════════════════════════════════════════════
# 3. COT data — only re-download years not yet fully in the file
# ══════════════════════════════════════════════════════════════════════════════
print('\nCFTC COT data:')

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
cot_csv  = f'{DATA_DIR}/cot_gold_weekly.csv'

try:
    import cot_reports as cot

    current_year = TODAY.year
    cot_last = last_date_in_csv(cot_csv)

    # Always re-download the current year (weekly data may have new entries)
    # and the previous year if we have never downloaded it
    if cot_last is not None:
        start_year = cot_last.year   # re-download from last year to catch late releases
        print(f'  Existing COT up to {cot_last.date()} — refreshing from {start_year}')
    else:
        start_year = 2006
        print(f'  No existing COT — full download from {start_year}')

    frames = []
    for y in range(start_year, current_year + 1):
        try:
            df_y = cot.cot_year(year=y, cot_report_type='legacy_fut',
                                store_txt=False, verbose=False)
            df_y = df_y.rename(columns={k: v for k, v in COT_COL_MAP.items()
                                        if k in df_y.columns})
            frames.append(df_y)
            print(f'  {y} ✓', end='\r')
        except Exception as e:
            print(f'  {y} skip: {e}')

    if frames:
        cot_new_raw = pd.concat(frames, ignore_index=True)

        # Filter to COMEX full-size gold futures only
        names        = cot_new_raw['Market_and_Exchange_Names'].astype(str)
        gold_mask    = names.str.contains('GOLD',              case=False, na=False)
        comex_mask   = names.str.contains('COMMODITY EXCHANGE', case=False, na=False)
        exclude_mask = (names.str.contains('MINI',     case=False, na=False) |
                        names.str.contains('MICRO',    case=False, na=False) |
                        names.str.contains('GOLDMAN',  case=False, na=False) |
                        names.str.contains('COINBASE', case=False, na=False) |
                        names.str.contains('TROY OZ',  case=False, na=False))
        cot_new_raw = cot_new_raw[gold_mask & comex_mask & ~exclude_mask].copy()

        # Parse date
        if 'As_of_Date_in_Form_YYYY_MM_DD' in cot_new_raw.columns:
            cot_new_raw['Date'] = pd.to_datetime(
                cot_new_raw['As_of_Date_in_Form_YYYY_MM_DD'], errors='coerce')
        else:
            cot_new_raw['Date'] = pd.to_datetime(
                cot_new_raw['As_of_Date_in_Form_YYMMDD'].astype(str)
                                 .str.replace('.0', '', regex=False),
                format='%y%m%d', errors='coerce')

        cot_new_raw = (cot_new_raw
                       .dropna(subset=['Date'])
                       .set_index('Date')
                       .sort_index())

        keep_cols = [
            'Comm_Positions_Long_All', 'Comm_Positions_Short_All',
            'NonComm_Positions_Long_All', 'NonComm_Positions_Short_All',
            'Open_Interest_All',
            'NonRept_Positions_Long_All', 'NonRept_Positions_Short_All',
        ]
        cot_new = cot_new_raw[[c for c in keep_cols if c in cot_new_raw.columns]]

        # Merge with existing (if any)
        if cot_last is not None:
            existing_cot = pd.read_csv(cot_csv, index_col=0, parse_dates=True)
            # Only keep rows from existing that are before start_year
            existing_cot = existing_cot[existing_cot.index.year < start_year]
            cot_gold = pd.concat([existing_cot, cot_new])
        else:
            cot_gold = cot_new

        cot_gold = cot_gold[~cot_gold.index.duplicated(keep='last')].sort_index()
        cot_gold.to_csv(cot_csv)
        print(f'\n  cot_gold_weekly.csv: {len(cot_gold)} rows  '
              f'({cot_gold.index[0].date()} → {cot_gold.index[-1].date()})')

except ImportError:
    print('  cot_reports not installed — skipping (pip install cot_reports)')
    if os.path.exists(cot_csv):
        cot_gold = pd.read_csv(cot_csv, index_col=0, parse_dates=True)
        print(f'  Loaded existing {cot_csv}')
except Exception as e:
    print(f'  COT error: {e}')
    if os.path.exists(cot_csv):
        cot_gold = pd.read_csv(cot_csv, index_col=0, parse_dates=True)

# ══════════════════════════════════════════════════════════════════════════════
# 4. Re-merge all sources from CSVs
# ══════════════════════════════════════════════════════════════════════════════
print('\nMerging all sources...')

# Always load from CSVs so merge is consistent regardless of full/incremental run
gold_df = pd.read_csv(f'{DATA_DIR}/gold_daily.csv', index_col=0, parse_dates=True)
gold_df.index = pd.to_datetime(gold_df.index).normalize()

df = gold_df[['Open', 'Gold', 'log_Gold']].copy()
df['log_Open'] = np.log(df['Open'])

if cot_gold is not None and len(cot_gold) > 0:
    cot_gold.index = pd.to_datetime(cot_gold.index).normalize()
    cot_daily = cot_gold.reindex(df.index, method='ffill')
    df = df.join(cot_daily, how='left')
    print(f'  COT columns joined ({len(cot_gold.columns)} cols, ffilled to daily)')

for col_name, series in macro_frames.items():
    series.index = pd.to_datetime(series.index).normalize()
    df = df.join(series, how='left')
    df[col_name] = df[col_name].ffill()
    open_col = f'{col_name}_Open'
    if open_col in df.columns:
        df[open_col] = df[open_col].ffill()
        print(f'  {col_name} (+ Open) joined: {df[col_name].notna().sum()} non-null rows')
    else:
        print(f'  {col_name} joined: {df[col_name].notna().sum()} non-null rows')

df.dropna(subset=['Gold'], inplace=True)
df.to_csv(f'{DATA_DIR}/merged_gold_dataset.csv')
print(f'\nmerged_gold_dataset.csv: {df.shape[0]} rows × {df.shape[1]} cols')
print(f'Date range: {df.index[0].date()} → {df.index[-1].date()}')

# ── Summary ───────────────────────────────────────────────────────────────────
print('\n' + '=' * 55)
print('  UPDATE COMPLETE — files in data/')
print('=' * 55)
for fname in sorted(os.listdir(DATA_DIR)):
    if fname.endswith('.csv'):
        fpath  = f'{DATA_DIR}/{fname}'
        last   = last_date_in_csv(fpath)
        size   = os.path.getsize(fpath) / 1024
        print(f'  {fname:<35} {size:>7.1f} KB   latest: {last.date() if last else "—"}')

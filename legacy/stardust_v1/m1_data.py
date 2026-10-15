"""
Legacy Stardust 1.x - M1 - Market data fetching & cleaning

Mechanically extracted from stardust_core.py lines 553-756 by
legacy/tools/split_monolith.py. The code inside run() is the original,
byte-for-byte; only the input/publish scaffolding around it is generated.
Do not hand-edit the extracted block - fix bugs in STAR2.0 (star/) instead.
"""
from ._bootstrap import ensure_bootstrap

ensure_bootstrap()

#: data this stage reads from earlier stages (its upstream contract)
INPUTS = (
    'BENCHMARK',
    'END_DATE',
    'FACTOR_TICKERS',
    'GLOBAL_SEED',
    'MISSING_THRESHOLD',
    'START_DATE',
    'status_callback',
    'TICKERS',
    'VOLUME_MIN',
)

#: legacy loop variables that leak in from earlier stages; pulled only so the
#: split behaves exactly like the monolith's single shared scope
CARRYOVER = ()

#: data this stage produces that a later stage reads (downstream contract)
OUTPUTS = (
    'clean_tickers',
    'close_df',
    'factor_index_ret',
    'log_return',
    'log_return_aligned',
)

#: everything bound at stage scope; all of it is published to ctx
_PUBLISHED = ('_close', '_coverage', '_factor_download_tickers', '_good_data', '_raw', '_t', 'ax', 'bad_tickers', 'clean_tickers', 'close_df', 'common_dates', 'e', 'factor_index_ret', 'factor_raw_data', 'fig', 'log_return', 'log_return_aligned', 'low_vol_tickers', 'median_vol', 'missing_pct', 'raw_data', 't', 'volume_df')


def run(ctx: dict) -> None:
    # ---- inputs from upstream stages (generated) ----
    BENCHMARK = ctx['BENCHMARK']
    END_DATE = ctx['END_DATE']
    FACTOR_TICKERS = ctx['FACTOR_TICKERS']
    GLOBAL_SEED = ctx['GLOBAL_SEED']
    MISSING_THRESHOLD = ctx['MISSING_THRESHOLD']
    START_DATE = ctx['START_DATE']
    status_callback = ctx['status_callback']
    TICKERS = ctx['TICKERS']
    VOLUME_MIN = ctx['VOLUME_MIN']
    # ---- original code: stardust_core.py lines 553-756 ----
    # ============================================================
    # MODULE 1: Data Fetching & Cleaning  (UPGRADED)
    # ------------------------------------------------------------
    # Changes from original:
    #   - Section 1G added: downloads factor index series
    #     (benchmark + smallcap + midcap) for Module 2
    #   - Everything else identical to original
    #
    # Reads from Module 0:
    #   TICKERS, START_DATE, END_DATE, GLOBAL_SEED,
    #   BENCHMARK, FACTOR_TICKERS
    #
    # Produces:
    #   close_df           → daily close prices (clean)
    #   log_return         → daily log returns  (clean)
    #   clean_tickers      → validated ticker list
    #   factor_index_ret   → daily returns of factor indices
    #   log_return_aligned → log_return trimmed to factor dates
    # ============================================================

    import pandas as pd
    import numpy as np
    import yfinance as yf
    import matplotlib.pyplot as plt

    status_callback("Module 1: Fetching market data...")

    # -----------------------------------------------------------
    # 1A. Confirm configuration from Module 0
    # -----------------------------------------------------------

    print(f"Data window  : {START_DATE.date()} → {END_DATE.date()}")
    print(f"Tickers      : {len(TICKERS)} stocks")
    print(f"Random seed  : {GLOBAL_SEED}")

    # -----------------------------------------------------------
    # 1B. Download OHLCV data
    # -----------------------------------------------------------

    raw_data  = yf.download(
        TICKERS,
        start       = START_DATE,
        end         = END_DATE,
        auto_adjust = True,
        progress    = True
    )

    close_df  = raw_data['Close'].copy()
    volume_df = raw_data['Volume'].copy()

    print(f"\nDownloaded: {close_df.shape[0]} trading days "
        f"× {close_df.shape[1]} tickers")

    # -----------------------------------------------------------
    # 1C. NaN check on RAW data BEFORE forward fill
    # -----------------------------------------------------------



    missing_pct   = close_df.isna().mean()
    bad_tickers   = missing_pct[missing_pct > MISSING_THRESHOLD].index.tolist()
    clean_tickers = [t for t in TICKERS if t not in bad_tickers]

    if bad_tickers:
        print(f"\nDropped tickers (>{MISSING_THRESHOLD*100:.0f}% missing):")
        for t in bad_tickers:
            print(f"  {t:20s}: {missing_pct[t]*100:.1f}% missing")
    else:
        print(f"\nAll {len(TICKERS)} tickers passed missing-data check.")

    close_df  = close_df[clean_tickers].copy()
    volume_df = volume_df[clean_tickers].copy()

    close_df = close_df.ffill(limit=2).dropna()

    print(f"Clean data: {close_df.shape[0]} trading days "
        f"× {close_df.shape[1]} tickers")

    # -----------------------------------------------------------
    # 1D. Volume sanity check
    # -----------------------------------------------------------


    volume_df  = volume_df.loc[close_df.index]
    median_vol = volume_df.median()

    low_vol_tickers = median_vol[median_vol < VOLUME_MIN].index.tolist()

    if low_vol_tickers:
        print(f"\nLow volume warning (median < {VOLUME_MIN:,} shares/day):")
        for t in low_vol_tickers:
            print(f"  {t:20s}: {median_vol[t]:,.0f} shares/day "
                f"— stale prices possible, review before investing.")
    else:
        print(f"All stocks have adequate volume "
            f"(> {VOLUME_MIN:,} shares/day).")

    # -----------------------------------------------------------
    # 1E. Compute log returns
    # -----------------------------------------------------------

    log_return = np.log(close_df / close_df.shift(1))
    log_return = log_return.dropna()

    print(f"\nLog return shape : {log_return.shape}  "
        f"(rows=trading days, cols=stocks)")
    print(f"Date range       : {log_return.index[0].date()} "
        f"→ {log_return.index[-1].date()}")
    print(f"Clean tickers    : "
        f"{[t.replace('.NS','') for t in clean_tickers]}")

    # -----------------------------------------------------------
    # 1F. Sanity check plot
    # -----------------------------------------------------------

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(log_return.index, log_return.values,
            alpha=0.4, linewidth=0.5)
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_title('Daily Log Returns — All Tickers\n'
                'Look for flat lines (stale data) or extreme spikes (unadjusted splits)')
    ax.set_xlabel('Date')
    ax.set_ylabel('Log Return')
    plt.tight_layout()
    plt.show()

    # -----------------------------------------------------------
    # 1G. Download factor index series
    #
    #     WHY HERE AND NOT MODULE 2?
    #     Module 1 owns all data fetching. Module 2 receives
    #     clean aligned data and computes — it downloads nothing.
    #
    #     We download per-ticker with individual try/except so
    #     one failed index doesn't block the whole pipeline.
    #     ^NSEI (benchmark) is mandatory — assertion will stop
    #     execution if it fails. SIZE/MID indices are optional
    #     but strongly recommended — without them the SIZE factor
    #     degrades to zero and MOM carries the full factor load.
    #
    #     Confirmed working yfinance tickers (as of May 2026):
    #       ^NSEI    — Nifty 50
    #       ^CNXSC   — Nifty Smallcap 100
    #       ^CNXMDCP — Nifty Midcap 100
    # -----------------------------------------------------------

    print("\nDownloading factor index series...")

    _factor_download_tickers = [BENCHMARK] + list(FACTOR_TICKERS.values())
    _good_data = {}

    for _t in _factor_download_tickers:
        try:
            _raw = yf.download(_t, start=START_DATE, end=END_DATE,
                            auto_adjust=True, progress=False)
            if not _raw.empty:
                _close = _raw['Close']
                _good_data[_t] = (_close.squeeze()
                                if hasattr(_close, 'squeeze')
                                else _close)
                print(f"  ✓ {_t}")
            else:
                print(f"  ✗ {_t} — empty response, skipping")
        except Exception as e:
            print(f"  ✗ {_t} — failed ({e}), skipping")

    assert BENCHMARK in _good_data, (
        f"Benchmark {BENCHMARK} failed to download — "
        "check internet connection and rerun Module 1."
    )

    factor_raw_data  = pd.DataFrame(_good_data)
    factor_index_ret = np.log(
        factor_raw_data / factor_raw_data.shift(1)
    ).dropna()

    # Align to trading days present in log_return
    common_dates       = log_return.index.intersection(factor_index_ret.index)
    factor_index_ret   = factor_index_ret.loc[common_dates]
    log_return_aligned = log_return.loc[common_dates]

    _coverage = len(common_dates) / len(log_return) * 100
    print(f"\n  Factor index dates : {len(factor_index_ret)} days "
        f"({_coverage:.1f}% overlap with stock returns)")
    print(f"  Indices available  : {list(factor_index_ret.columns)}")

    if _coverage < 80:
        print("  ⚠ Low overlap — check FACTOR_TICKERS in Module 0.")
    else:
        print("  ✓ Factor data ready for Module 2.")

    # -----------------------------------------------------------
    # MODULE 1 OUTPUTS — passed into Module 2
    # -----------------------------------------------------------
    # close_df           → pd.DataFrame : daily close prices (clean)
    # log_return         → pd.DataFrame : daily log returns  (clean)
    # clean_tickers      → list[str]    : tickers that passed validation
    # factor_index_ret   → pd.DataFrame : factor index daily returns
    #                                     columns: BENCHMARK + FACTOR_TICKERS
    # log_return_aligned → pd.DataFrame : log_return trimmed to factor dates
    #
    # CRITICAL: log_return column order is the alignment anchor.
    # Never sort or reindex log_return after this point.
    # -----------------------------------------------------------
    # ---- publish (generated) ----
    _scope = locals()
    ctx.update({k: _scope[k] for k in _PUBLISHED if k in _scope})

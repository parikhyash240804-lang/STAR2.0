"""
Legacy Stardust 1.x - M2 (2B.5-2B.7) - Fundamentals, signals, systematic Black-Litterman views

Mechanically extracted from stardust_core.py lines 1142-1673 by
legacy/tools/split_monolith.py. The code inside run() is the original,
byte-for-byte; only the input/publish scaffolding around it is generated.
Do not hand-edit the extracted block - fix bugs in STAR2.0 (star/) instead.
"""
from ._bootstrap import ensure_bootstrap

ensure_bootstrap()

#: data this stage reads from earlier stages (its upstream contract)
INPUTS = (
    'log_return',
    'MIN_PEER_GROUP',
    'n_stk',
    'RISK_FREE',
    'SCORE_CEILING',
    'SCORE_FLOOR',
    'SECTOR_MULTIPLIERS',
    'SIGNAL_LOOKBACKS',
    'SIGNAL_WEIGHTS',
    'TRADING_DAYS',
    'VIEW_CEILING',
    'VIEW_FLOOR',
    'WINSOR_CAP',
)

#: legacy loop variables that leak in from earlier stages; pulled only so the
#: split behaves exactly like the monolith's single shared scope
CARRYOVER = (
    'BL_VIEWS',
)

#: data this stage produces that a later stage reads (downstream contract)
OUTPUTS = (
    '_manual_overrides',
    'BL_VIEWS',
)

#: everything bound at stage scope; all of it is published to ctx
_PUBLISHED = ('BL_VIEWS', '_PE_MAX', '_PE_MIN', '_ROE_MAX', '_ROE_MIN', '_c', '_cnt', '_col', '_cum_12m', '_cum_1m', '_defaults', '_e', '_eps', '_eps_rev_raw', '_f', '_fetch_fail', '_fetch_ok', '_field_labels', '_fields', '_fund_raw', '_hist', '_ind', '_info', '_invalid_manual', '_k', '_listed', '_manual_overrides', '_mean', '_mom_lookback', '_more', '_mult', '_n_after_sec', '_n_before', '_n_default', '_n_filled', '_n_valid', '_names', '_pe', '_peer_vals', '_peers', '_q_last', '_q_prev', '_raw', '_rec', '_rev', '_rev_lookback', '_roe', '_roe_hi', '_roe_lo', '_rolling_vol', '_s', '_score', '_score_to_view', '_sec', '_sec_medians', '_sector_counts', '_sectors', '_sig_cols', '_small_sectors', '_sorted_views', '_src', '_std', '_systematic_views', '_t', '_tickers_in_sec', '_ts', '_univ_mean', '_univ_median', '_univ_std', '_unmapped_sectors', '_v', '_val', '_view_vals', '_vol_lookback', '_wt_vec', '_z', 'composite_adj', 'composite_raw', 'composite_s', 'fundamentals_df', 'k', 'peer_groups', 's', 'sector_map', 'sig_earnings_yld', 'sig_eps_rev', 'sig_low_vol', 'sig_momentum', 'sig_roe', 'signals_df', 't', 'ts', 'v', 'zscores_df')


def run(ctx: dict) -> None:
    # ---- inputs from upstream stages (generated) ----
    log_return = ctx['log_return']
    MIN_PEER_GROUP = ctx['MIN_PEER_GROUP']
    n_stk = ctx['n_stk']
    RISK_FREE = ctx['RISK_FREE']
    SCORE_CEILING = ctx['SCORE_CEILING']
    SCORE_FLOOR = ctx['SCORE_FLOOR']
    SECTOR_MULTIPLIERS = ctx['SECTOR_MULTIPLIERS']
    SIGNAL_LOOKBACKS = ctx['SIGNAL_LOOKBACKS']
    SIGNAL_WEIGHTS = ctx['SIGNAL_WEIGHTS']
    TRADING_DAYS = ctx['TRADING_DAYS']
    VIEW_CEILING = ctx['VIEW_CEILING']
    VIEW_FLOOR = ctx['VIEW_FLOOR']
    WINSOR_CAP = ctx['WINSOR_CAP']
    # ---- read from upstream, then rebound here (generated) ----
    if 'BL_VIEWS' in ctx: BL_VIEWS = ctx['BL_VIEWS']
    # ---- original code: stardust_core.py lines 1142-1673 ----
    # ============================================================
    # MODULE 2 — NEW SECTIONS 2B.5 / 2B.6 / 2B.7
    # ------------------------------------------------------------
    # INSERT LOCATION: After section 2E (arith_factor computed)
    #                  Before section 2F (Black-Litterman)
    #
    # These three sections replace the static BL_VIEWS you typed
    # manually. They auto-generate views from 5 quantitative
    # signals scored within sector peer groups.
    #
    # WHAT CHANGES IN EXISTING CODE:
    #   - Section 2F (BL) is UNCHANGED — it still reads BL_VIEWS
    #   - But BL_VIEWS is now POPULATED HERE automatically
    #   - Manual entries in BL_VIEWS (Module 0) override signals
    #
    # DEPENDENCIES FROM MODULE 0:
    #   SIGNAL_WEIGHTS, SIGNAL_LOOKBACKS, VIEW_FLOOR, VIEW_CEILING
    #   MIN_PEER_GROUP, SECTOR_MULTIPLIERS, BL_VIEWS (overrides)
    #   RISK_FREE, TRADING_DAYS
    #
    # DEPENDENCIES FROM EARLIER MODULE 2:
    #   log_return         — price returns (already computed)
    #   close_df           — prices (from Module 1)
    #   arith_factor       — factor returns (from section 2E)
    #   _factor_betas      — OLS betas (from section 2B)
    #   BENCHMARK          — from Module 0
    # ============================================================

    import numpy  as np
    import pandas as pd
    import yfinance as yf
    import warnings
    warnings.filterwarnings('ignore')

    # ============================================================
    # 2B.5  FUNDAMENTAL DATA FETCH
    # ------------------------------------------------------------
    # Fetches three things per ticker in ONE .info call:
    #   1. Sector classification  (for peer-group z-scoring)
    #   2. Trailing PE ratio      (earnings yield signal)
    #   3. Return on Equity       (profitability signal)
    #
    # Then ONE .quarterly_earnings call per ticker:
    #   4. Last 4 quarters EPS    (earnings revision signal)
    #
    # FILL HIERARCHY (in order):
    #   valid value → sector median → universe median → default
    #
    # VALIDATION RULES per field:
    #   trailingPE  : must be > 0 and < 200
    #                 (negative PE = loss-making, not a value signal)
    #                 (PE > 200 = speculative, distorts yield calc)
    #   returnOnEquity: must be > -1.0 and < 5.0
    #                 (ROE > 500% = leverage distortion, discard)
    #   sector      : any non-empty string is valid
    #                 unknown → tagged 'Unknown' → gets multiplier 1.0
    # ============================================================

    print(f"\n{'─'*58}")
    print(f"  2B.5  Fetching fundamental data for {n_stk} stocks...")
    print(f"{'─'*58}")

    # ── Storage ─────────────────────────────────────────────────
    _fund_raw = {}     # raw values before fill
    _sectors  = {}     # sector per ticker (filled separately)

    _PE_MIN,   _PE_MAX  = 0.0,  200.0
    _ROE_MIN, _ROE_MAX  = -1.0, 5.0

    _fetch_ok   = 0
    _fetch_fail = 0

    for _t in log_return.columns:
        _rec = {
            'trailingPE'      : np.nan,
            'returnOnEquity'  : np.nan,
            'sector'          : 'Unknown',
            'industry'        : 'Unknown',
        }
        try:
            _info = yf.Ticker(_t).info

            # Sector
            _sec = _info.get('sector', '') or ''
            _ind = _info.get('industry', '') or ''
            _rec['sector']   = _sec if _sec else 'Unknown'
            _rec['industry'] = _ind if _ind else 'Unknown'

            # Trailing PE — validate range
            _pe = _info.get('trailingPE', None)
            if _pe is not None and _PE_MIN < float(_pe) < _PE_MAX:
                _rec['trailingPE'] = float(_pe)

            # ROE — validate range
            _roe = _info.get('returnOnEquity', None)
            if _roe is not None and _ROE_MIN < float(_roe) < _ROE_MAX:
                _rec['returnOnEquity'] = float(_roe)

            _fetch_ok += 1

        except Exception as _e:
            _fetch_fail += 1

        _fund_raw[_t] = _rec
        _sectors[_t]  = _rec['sector']

    # ── EPS revision fetch (separate call — quarterly_earnings) ─
    _eps_rev_raw = {}

    for _t in log_return.columns:
        try:
            _hist = yf.Ticker(_t).quarterly_earnings
            if _hist is not None and len(_hist) >= 2:
                # Most recent 2 quarters EPS
                _eps = _hist['Earnings'].dropna()
                if len(_eps) >= 2:
                    _q_last = float(_eps.iloc[0])
                    _q_prev = float(_eps.iloc[1])
                    # Direction × magnitude (capped to avoid outliers)
                    if _q_prev != 0:
                        _rev = (_q_last - _q_prev) / abs(_q_prev)
                        _eps_rev_raw[_t] = np.clip(_rev, -2.0, 2.0)
                    else:
                        _eps_rev_raw[_t] = 0.0
                else:
                    _eps_rev_raw[_t] = np.nan
            else:
                _eps_rev_raw[_t] = np.nan
        except Exception:
            _eps_rev_raw[_t] = np.nan

    # ── Build fundamentals DataFrame ────────────────────────────
    fundamentals_df = pd.DataFrame(_fund_raw).T.reindex(log_return.columns)
    fundamentals_df['eps_revision'] = pd.Series(
        _eps_rev_raw
    ).reindex(log_return.columns)

    # ── Coverage report ─────────────────────────────────────────
    _fields = ['trailingPE', 'returnOnEquity', 'eps_revision']
    _field_labels = {
        'trailingPE'     : 'Trailing PE',
        'returnOnEquity' : 'ROE',
        'eps_revision'   : 'EPS revision',
    }
    print(f"\n  Fetch results:")
    print(f"    Successful .info calls : {_fetch_ok} / {n_stk}")
    print(f"    Failed .info calls     : {_fetch_fail} / {n_stk}")
    print(f"\n  Field coverage (before fill):")
    for _f in _fields:
        _n_valid = fundamentals_df[_f].notna().sum()
        print(f"    {_field_labels[_f]:<20} {_n_valid:>3}/{n_stk} "
            f"({_n_valid/n_stk*100:.0f}%)")

    # ── Sector classification table ─────────────────────────────
    _sector_counts = fundamentals_df['sector'].value_counts()
    print(f"\n  Sector classification (check for mismatches):")
    for _sec, _cnt in _sector_counts.items():
        _tickers_in_sec = [
            t.replace('.NS','')
            for t in fundamentals_df.index
            if fundamentals_df.loc[t, 'sector'] == _sec
        ]
        _listed = ', '.join(_tickers_in_sec[:6])
        _more   = f' +{len(_tickers_in_sec)-6} more' if len(_tickers_in_sec) > 6 else ''
        print(f"    {_sec:<28} ({_cnt:>2} stocks) {_listed}{_more}")

    # ── Missing value fill: sector median → universe median ─────
    print(f"\n  Filling missing values...")

    for _f in _fields:
        _n_before = fundamentals_df[_f].isna().sum()
        if _n_before == 0:
            continue

        # Sector median fill
        _sec_medians = (fundamentals_df
                        .groupby('sector')[_f]
                        .transform('median'))
        fundamentals_df[_f] = fundamentals_df[_f].fillna(_sec_medians)

        # Universe median fill for remaining
        _univ_median = fundamentals_df[_f].median()
        _n_after_sec = fundamentals_df[_f].isna().sum()
        fundamentals_df[_f] = fundamentals_df[_f].fillna(_univ_median)

        # Conservative defaults if everything is NaN
        _defaults = {
            'trailingPE'     : 20.0,   # ~market PE
            'returnOnEquity' : 0.12,   # ~12% — typical Indian corp
            'eps_revision'   : 0.0,    # neutral
        }
        fundamentals_df[_f] = fundamentals_df[_f].fillna(_defaults[_f])

        _n_filled = _n_before - _n_after_sec
        _n_default = _n_after_sec
        if _n_before > 0:
            print(f"    {_field_labels[_f]:<20} "
                f"filled {_n_filled} from sector median, "
                f"{_n_default} from universe median")

    # Winsorise ROE at 5th/95th percentile — prevents one
    # extremely high-ROE NBFC from dominating the signal
    _roe_lo = fundamentals_df['returnOnEquity'].quantile(0.05)
    _roe_hi = fundamentals_df['returnOnEquity'].quantile(0.95)
    fundamentals_df['returnOnEquity'] = fundamentals_df['returnOnEquity'].clip(
        _roe_lo, _roe_hi
    )

    # Compute earnings yield from PE
    # earnings_yield = 1 / PE — avoid division by zero
    fundamentals_df['earnings_yield'] = (
        1.0 / fundamentals_df['trailingPE'].replace(0, np.nan)
    ).fillna(1.0 / 20.0)

    print(f"\n  ✓ Fundamental data ready.")
    print(f"{'─'*58}")

    # ============================================================
    # 2B.6  SIGNAL COMPUTATION
    # ------------------------------------------------------------
    # Computes 5 raw signals per stock.
    # No z-scoring yet — raw values only.
    # Z-scoring happens in 2B.7 within sector peer groups.
    #
    # SIGNALS:
    #
    #   momentum  = cumulative 12M return minus last 1M
    #               This is the Carhart UMD factor in stock form.
    #               Subtracting 1M avoids short-term reversal.
    #               Uses the full log_return (5Y) for maximum
    #               lookback — rolling(252) needs at least 200 obs.
    #
    #   low_vol   = rolling 6M annualised volatility, sign-flipped
    #               Lower vol → higher score → low-vol anomaly.
    #               Particularly strong in India because retail
    #               investors systematically overpay for high-
    #               beta lottery stocks, leaving low-vol
    #               systematically underpriced.
    #
    #   earnings_yield = 1 / trailing PE (from fundamentals_df)
    #               Higher yield → cheaper → value signal.
    #               More stable than raw PE for cross-stock ranking.
    #
    #   roe       = return on equity (from fundamentals_df)
    #               Winsorised in 2B.5 to remove outliers.
    #               Proxies for FF5 RMW (Robust Minus Weak).
    #
    #   eps_rev   = direction + magnitude of last 2Q EPS change
    #               +2.0 = large improvement, -2.0 = large decline
    #               Most forward-looking signal; also noisiest.
    #               Low weight (0.10) reflects this.
    # ============================================================

    print(f"\n{'─'*58}")
    print(f"  2B.6  Computing signals for {n_stk} stocks...")

    # ── Signal 1: Momentum (12M - 1M) ───────────────────────────
    _mom_lookback = SIGNAL_LOOKBACKS.get('momentum', 252)
    _rev_lookback = SIGNAL_LOOKBACKS.get('reversal', 21)

    _cum_12m = log_return.rolling(
        _mom_lookback, min_periods=int(_mom_lookback * 0.8)
    ).sum()
    _cum_1m  = log_return.rolling(
        _rev_lookback, min_periods=int(_rev_lookback * 0.7)
    ).sum()

    # Take last row = most recent trailing signal value per stock
    sig_momentum = (_cum_12m - _cum_1m).iloc[-1].rename('momentum')

    # ── Signal 2: Low Volatility (sign-flipped) ──────────────────
    _vol_lookback = SIGNAL_LOOKBACKS.get('low_vol', 126)
    _rolling_vol  = (log_return
                    .rolling(_vol_lookback, min_periods=int(_vol_lookback * 0.7))
                    .std()
                    .iloc[-1]
                    * np.sqrt(TRADING_DAYS))      # annualise

    # Sign-flip: lower vol → higher score
    sig_low_vol = (-_rolling_vol).rename('low_vol')

    # ── Signal 3: Earnings Yield (from fundamentals) ─────────────
    sig_earnings_yld = fundamentals_df['earnings_yield'].rename('earnings_yld')

    # ── Signal 4: ROE (from fundamentals) ────────────────────────
    sig_roe = fundamentals_df['returnOnEquity'].rename('roe')

    # ── Signal 5: EPS Revision (from fundamentals) ───────────────
    sig_eps_rev = fundamentals_df['eps_revision'].rename('eps_revision')

    # ── Combine into signals DataFrame ───────────────────────────
    signals_df = pd.DataFrame({
        'momentum'    : sig_momentum,
        'low_vol'     : sig_low_vol,
        'earnings_yld': sig_earnings_yld,
        'roe'         : sig_roe,
        'eps_revision': sig_eps_rev,
    }).reindex(log_return.columns)

    # Check for any all-NaN columns — fallback to zero
    for _col in signals_df.columns:
        if signals_df[_col].isna().all():
            print(f"  ⚠ Signal '{_col}' is all-NaN — setting to 0.")
            signals_df[_col] = 0.0

    print(f"\n  Raw signal ranges (before z-scoring):")
    for _col in signals_df.columns:
        _s = signals_df[_col].dropna()
        print(f"    {_col:<18} "
            f"min={_s.min():+.3f}  "
            f"mean={_s.mean():+.3f}  "
            f"max={_s.max():+.3f}  "
            f"({_s.notna().sum()}/{n_stk} valid)")

    print(f"\n  ✓ All 5 signals computed.")
    print(f"{'─'*58}")

    # ============================================================
    # 2B.7  SCORE, BLEND, GENERATE BL_VIEWS
    # ------------------------------------------------------------
    # Four sub-steps:
    #
    #   A. Build sector peer groups from fundamentals_df
    #   B. Z-score each signal within sector peer groups
    #   C. Blend z-scores → composite score per stock
    #   D. Map composite → view return → populate BL_VIEWS
    #
    # OUTPUT: BL_VIEWS dict (overrides Module 0 manual entries)
    # The rest of section 2F runs exactly as before.
    # ============================================================

    print(f"\n{'─'*58}")
    print(f"  2B.7  Generating systematic BL views...")

    # ── Sub-step A: Sector peer groups ──────────────────────────
    sector_map   = fundamentals_df['sector'].to_dict()  # {ticker: sector}
    peer_groups  = {}                                    # {sector: [tickers]}

    for _t, _sec in sector_map.items():
        peer_groups.setdefault(_sec, []).append(_t)

    # Log any sectors that are too small for peer-group z-scoring
    _small_sectors = {
        s: ts for s, ts in peer_groups.items()
        if len(ts) < MIN_PEER_GROUP
    }
    if _small_sectors:
        print(f"\n  ⚠ Sectors with < {MIN_PEER_GROUP} stocks "
            f"(universe z-score used for these):")
        for _s, _ts in _small_sectors.items():
            _names = [t.replace('.NS','') for t in _ts]
            print(f"    {_s:<28} : {', '.join(_names)}")

    # ── Sub-step B: Sector-relative z-scoring ───────────────────
    #
    # For each signal:
    #   1. Compute mean and std within each sector
    #   2. Z-score = (stock_value - sector_mean) / sector_std
    #   3. If sector too small → use universe mean/std instead
    #   4. Winsorise at ±2.5 to remove extreme outliers
    #


    zscores_df = pd.DataFrame(index=signals_df.index,
                            columns=signals_df.columns,
                            dtype=float)

    for _col in signals_df.columns:
        _raw = signals_df[_col].copy()

        # Universe-level fallback stats (used for small sectors)
        _univ_mean = _raw.mean()
        _univ_std  = _raw.std()
        if _univ_std == 0 or np.isnan(_univ_std):
            _univ_std = 1.0

        for _t in log_return.columns:
            _sec   = sector_map.get(_t, 'Unknown')
            _peers = peer_groups.get(_sec, [])

            # Decide scope: sector or universe
            if len(_peers) >= MIN_PEER_GROUP:
                _peer_vals = _raw.loc[_peers].dropna()
                _mean = _peer_vals.mean()
                _std  = _peer_vals.std()
                if np.isnan(_std) or _std == 0:
                    _mean, _std = _univ_mean, _univ_std
            else:
                _mean, _std = _univ_mean, _univ_std

            _val = _raw.get(_t, np.nan)
            if np.isnan(_val):
                zscores_df.loc[_t, _col] = 0.0
            else:
                _z = (_val - _mean) / _std
                zscores_df.loc[_t, _col] = np.clip(_z, -WINSOR_CAP, WINSOR_CAP)

    zscores_df = zscores_df.astype(float)

    # ── Sub-step C: Composite score ──────────────────────────────
    #
    # composite = weighted sum of z-scores
    # Then apply sector multiplier from Module 0.
    #
    # Sector multiplier > 1.0 = bullish tilt for that sector
    # Sector multiplier < 1.0 = bearish tilt
    # Default = 1.0 if sector not in SECTOR_MULTIPLIERS
    #
    # SIGNAL_WEIGHTS validated in Module 0 to sum to 1.0

    # Build weight vector aligned to zscores_df columns
    _sig_cols = list(signals_df.columns)   # order matters
    _wt_vec   = np.array([
        SIGNAL_WEIGHTS.get(_c, 0.0) for _c in _sig_cols
    ])

    composite_raw = zscores_df[_sig_cols].values @ _wt_vec
    composite_s   = pd.Series(composite_raw, index=zscores_df.index)

    # Apply sector multipliers
    _unmapped_sectors = set()
    composite_adj = composite_s.copy()

    for _t in log_return.columns:
        _sec  = sector_map.get(_t, 'Unknown')
        _mult = SECTOR_MULTIPLIERS.get(_sec, None)
        if _mult is None:
            _unmapped_sectors.add(_sec)
            _mult = 1.0
        composite_adj[_t] = composite_s[_t] * _mult

    if _unmapped_sectors:
        print(f"\n  ⚠ Sectors not in SECTOR_MULTIPLIERS "
            f"(using 1.0x neutral):")
        for _s in sorted(_unmapped_sectors):
            _ts = [t.replace('.NS','') for t in log_return.columns
                if sector_map.get(t) == _s]
            print(f"    {_s:<28} : {', '.join(_ts)}")
        print(f"    → Add to SECTOR_MULTIPLIERS in Module 0 to tilt.")

    # ── Sub-step D: Map composite → view return → BL_VIEWS ───────
    #
    # Linear map: composite clipped to [-3, +3] → [VIEW_FLOOR, VIEW_CEILING]
    # Using fixed [-3, +3] range (not sample min/max) to avoid
    # extreme compression when all stocks score similarly.
    #
    # Then check for manual overrides from Module 0 BL_VIEWS dict.
    # Manual entries replace the systematic view for that ticker.



    def _score_to_view(score: float) -> float:
        """Map composite score in [-3,3] to excess return view in [VIEW_FLOOR, VIEW_CEILING].
        Views are stored as excess returns (above RISK_FREE) to match the BL prior Pi."""
        _clipped = np.clip(score, SCORE_FLOOR, SCORE_CEILING)
        _norm    = (_clipped - SCORE_FLOOR) / (SCORE_CEILING - SCORE_FLOOR)
        _total   = VIEW_FLOOR + _norm * (VIEW_CEILING - VIEW_FLOOR)
        return _total - RISK_FREE


    # Generate systematic views for ALL stocks in universe
    _systematic_views = {
        _t: _score_to_view(composite_adj[_t])
        for _t in log_return.columns
    }

    # Apply manual overrides from Module 0
    # Manual entries in BL_VIEWS take priority over signals
    _manual_overrides = {
        k: v for k, v in BL_VIEWS.items()
        if k in log_return.columns
    }
    _invalid_manual = {
        k for k in BL_VIEWS.keys()
        if k not in log_return.columns
    }
    if _invalid_manual:
        print(f"\n  ⚠ Manual overrides not in universe (ignored):")
        for _k in _invalid_manual:
            print(f"    {_k}")

    # Final BL_VIEWS = systematic + manual overrides
    BL_VIEWS = {**_systematic_views, **_manual_overrides}

    # ── Print generated view table ───────────────────────────────
    print(f"\n  Generated views ({len(BL_VIEWS)} stocks):")
    print(f"  {'Ticker':<14} {'Sector':<22} {'Score':>7} "
        f"{'Mult':>5} {'View%':>7} {'Source':>10}")
    print(f"  {'─'*68}")

    _sorted_views = sorted(BL_VIEWS.items(), key=lambda x: -x[1])
    for _t, _v in _sorted_views:
        _sec   = sector_map.get(_t, 'Unknown')[:20]
        _score = composite_adj.get(_t, 0.0)
        _mult  = SECTOR_MULTIPLIERS.get(sector_map.get(_t,''), 1.0)
        _src   = 'OVERRIDE' if _t in _manual_overrides else 'signal'
        print(f"  {_t.replace('.NS',''):<14} "
            f"{_sec:<22} "
            f"{_score:>+6.2f}  "
            f"{_mult:>4.2f}x "
            f"{_v*100:>6.1f}%  "
            f"{_src:>10}")

    # ── Summary statistics ───────────────────────────────────────
    _view_vals = np.array(list(BL_VIEWS.values()))
    print(f"\n  View summary:")
    print(f"    Mean  : {_view_vals.mean()*100:+.1f}%")
    print(f"    Min   : {_view_vals.min()*100:+.1f}%  "
        f"({min(BL_VIEWS, key=BL_VIEWS.get).replace('.NS','')})")
    print(f"    Max   : {_view_vals.max()*100:+.1f}%  "
        f"({max(BL_VIEWS, key=BL_VIEWS.get).replace('.NS','')})")
    print(f"    Stdev : {_view_vals.std()*100:.1f}%  "
        f"(spread — higher = more differentiation)")
    if len(_manual_overrides) > 0:
        print(f"    Manual overrides applied : {len(_manual_overrides)} stocks")
        for _k, _v in _manual_overrides.items():
            print(f"      {_k.replace('.NS',''):<16} "
                f"→ {_v*100:.1f}% (overrides signal)")

    print(f"\n  ✓ BL_VIEWS populated — section 2F will use "
        f"{len(BL_VIEWS)} systematic views.")
    print(f"{'─'*58}")

    # ── Expose diagnostic variables for inspection ───────────────
    # signals_df     → raw signal values per stock (pre z-score)
    # zscores_df     → z-scored signals per stock (post peer-group)
    # composite_s    → raw composite score (pre multiplier)
    # composite_adj  → final composite score (post multiplier)
    # sector_map     → {ticker: sector} classification
    # peer_groups    → {sector: [tickers]} grouping
    # fundamentals_df→ PE, ROE, EPS revision per stock

    # ---- publish (generated) ----
    _scope = locals()
    ctx.update({k: _scope[k] for k in _PUBLISHED if k in _scope})

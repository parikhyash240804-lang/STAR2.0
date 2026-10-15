"""
Legacy Stardust 1.x - M2 (2A-2E) - Factor construction, factor covariance, baselines, factor-model returns

Mechanically extracted from stardust_core.py lines 757-1141 by
legacy/tools/split_monolith.py. The code inside run() is the original,
byte-for-byte; only the input/publish scaffolding around it is generated.
Do not hand-edit the extracted block - fix bugs in STAR2.0 (star/) instead.
"""
from ._bootstrap import ensure_bootstrap

ensure_bootstrap()

#: data this stage reads from earlier stages (its upstream contract)
INPUTS = (
    'BENCHMARK',
    'FACTOR_HALFLIFE',
    'factor_index_ret',
    'log_return',
    'log_return_aligned',
    'LOW_R2_FALLBACK_THRESHOLD',
    'MKT_FLOOR',
    'R2_MIN_THRESHOLD',
    'RISK_FREE',
    'SIGNAL_LOOKBACKS',
    'SIZE_FLOOR',
    'SIZE_REGIME',
    'SIZE_REGIME_SCALAR',
    'status_callback',
    'TRADING_DAYS',
)

#: legacy loop variables that leak in from earlier stages; pulled only so the
#: split behaves exactly like the monolith's single shared scope
CARRYOVER = ()

#: data this stage produces that a later stage reads (downstream contract)
OUTPUTS = (
    '_cov_lw_daily',
    '_cov_method',
    '_factor_r2',
    'arith_equal',
    'arith_ewma',
    'arith_factor',
    'cov_annual',
    'cov_daily',
    'n_stk',
)

#: everything bound at stage scope; all of it is published to ctx
_PUBLISHED = ('B', 'D_daily', 'F_daily', 'X', 'X_t', '_cov_factor_daily', '_cov_lw_daily', '_cov_method', '_factor_betas', '_factor_exp_rets', '_factor_names', '_factor_r2', '_failed', '_frac_low_r2', '_low_r2', '_mom_mean', '_mom_raw', '_mom_std', '_mom_zscore', '_non_mkt', '_resid_df', '_residuals', '_roll_12m', '_roll_1m', '_size_candidates', '_size_col', '_stock_ret', 'arith_equal', 'arith_ewma', 'arith_factor', 'avg_r2', 'b', 'c', 'cov_annual', 'cov_daily', 'daily_rf', 'ewma_log_mean', 'f', 'factor_premiums', 'factors', 'log_mean_annual', 'lw', 'mkt_factor', 'model', 'mom_factor', 'n_obs', 'n_stk', 'ratio', 'size_factor', 't', 'ticker', 'v', 'y')


def run(ctx: dict) -> None:
    # ---- inputs from upstream stages (generated) ----
    BENCHMARK = ctx['BENCHMARK']
    FACTOR_HALFLIFE = ctx['FACTOR_HALFLIFE']
    factor_index_ret = ctx['factor_index_ret']
    log_return = ctx['log_return']
    log_return_aligned = ctx['log_return_aligned']
    LOW_R2_FALLBACK_THRESHOLD = ctx['LOW_R2_FALLBACK_THRESHOLD']
    MKT_FLOOR = ctx['MKT_FLOOR']
    R2_MIN_THRESHOLD = ctx['R2_MIN_THRESHOLD']
    RISK_FREE = ctx['RISK_FREE']
    SIGNAL_LOOKBACKS = ctx['SIGNAL_LOOKBACKS']
    SIZE_FLOOR = ctx['SIZE_FLOOR']
    SIZE_REGIME = ctx['SIZE_REGIME']
    SIZE_REGIME_SCALAR = ctx['SIZE_REGIME_SCALAR']
    status_callback = ctx['status_callback']
    TRADING_DAYS = ctx['TRADING_DAYS']
    # ---- original code: stardust_core.py lines 757-1141 ----
    # ============================================================
    # MODULE 2: Covariance & Return Estimation  (BL-MF UPGRADE)
    # ------------------------------------------------------------
    # Full replacement of original Module 2.
    # Supports five return estimation methods via RETURN_METHOD
    # in Module 0. Covariance upgraded to factor-structured
    # decomposition (replaces pure Ledoit-Wolf sample covariance).
    #
    # RETURN METHODS:
    #   'equal'  → 5Y equal-weighted mean  (original, biased)
    #   'ewma'   → exponentially weighted mean
    #   'bl'     → Black-Litterman posterior  ← RECOMMENDED
    #   'factor' → 3-factor OLS expected returns
    #   'hybrid' → 0.5 × BL + 0.5 × factor
    #
    # COVARIANCE:
    #   Factor-structured covariance (BL-MF approach):
    #     Cov = B × F × B.T + D
    #   where B = factor loadings (n_stocks × 3)
    #         F = factor covariance (3 × 3)
    #         D = diagonal residual variances
    #   Falls back to Ledoit-Wolf if factor data unavailable.
    #
    # Reads from Module 0:
    #   TRADING_DAYS, SHRINK_FACTOR, RISK_FREE, BENCHMARK
    #   RETURN_METHOD, BL_DELTA, BL_TAU, BL_VIEWS,
    #   BL_VIEW_CONFIDENCE, FACTOR_HALFLIFE, FACTOR_TICKERS
    #
    # Reads from Module 1:
    #   log_return, log_return_aligned, factor_index_ret
    #
    # Produces (SAME INTERFACE AS ORIGINAL — no downstream changes):
    #   cov_daily     → daily covariance (factor-structured)
    #   cov_annual    → annualized covariance
    #   arith_returns → expected returns (method-selected)
    #
    # All downstream modules (3, 4, 5, 6) work without changes.
    # ============================================================

    import numpy as np
    import pandas as pd
    import statsmodels.api as sm
    from sklearn.covariance import LedoitWolf

    status_callback("Module 2: Estimating covariance & returns...")

    # -----------------------------------------------------------
    # Dependency check
    # -----------------------------------------------------------


    n_stk = len(log_return.columns)
    n_obs = len(log_return)

    # ============================================================
    # 2A. Factor construction
    #
    #     Three factors for Indian equity:
    #
    #     MKT  = Nifty 50 daily excess return over risk-free
    #            The dominant factor — explains 60-70% of
    #            individual stock variance in Indian equities.
    #
    #     SIZE = Nifty Smallcap 100 minus Nifty 50 daily return
    #            Captures the small-cap premium.
    #            Positive = small-caps outperforming large-caps.
    #            IIM-A research confirms this premium is real
    #            and persistent in India (1996–2024 data).
    #
    #     MOM  = 12-1 month momentum (Carhart 1997)
    #            Return from t-252 to t-21 (skip last month).
    #            Cross-sectionally demeaned and z-scored daily.
    #            Computed on FULL log_return (5 years) for maximum
    #            lookback — then aligned to factor dates.
    #            This is the fix for the MOM bug in the MF notebook.
    #
    #     All factors share the same date index as
    #     log_return_aligned (Module 1 section 1G).
    # ============================================================

    print(f"{'─'*58}")
    print(f"  Building factor series...")

    daily_rf = RISK_FREE / TRADING_DAYS

    # --- MKT factor ---
    if BENCHMARK in factor_index_ret.columns:
        mkt_factor = factor_index_ret[BENCHMARK] - daily_rf
    else:
        mkt_factor = factor_index_ret.iloc[:, 0] - daily_rf
        print(f"  ⚠ {BENCHMARK} not found — using {factor_index_ret.columns[0]}")
    mkt_factor.name = 'MKT'

    # --- SIZE factor ---
    _non_mkt = [c for c in factor_index_ret.columns if c != BENCHMARK]
    if _non_mkt:
        # Prefer smallcap (^CNXSC) over midcap for SIZE
        # FIXED parentheses vs MF notebook bug
        _size_candidates = [c for c in _non_mkt
                            if ('SC' in c.upper() or 'CNXSC' in c)
                            and c != BENCHMARK]
        _size_col = _size_candidates[0] if _size_candidates else _non_mkt[0]
        size_factor = factor_index_ret[_size_col] - factor_index_ret[BENCHMARK]
        print(f"  SIZE factor source : {_size_col} minus {BENCHMARK}")
    else:
        size_factor = pd.Series(
            np.zeros(len(mkt_factor)),
            index=mkt_factor.index, name='SIZE'
        )
        print("  ⚠ No secondary index — SIZE factor set to zero.")
        print("    Add ^CNXSC to FACTOR_TICKERS in Module 0 for better estimates.")
    size_factor.name = 'SIZE'

    # --- MOM factor ---
    # CRITICAL FIX: compute on full log_return (5Y), THEN align.
    # Using log_return_aligned (shorter, trimmed to factor dates)
    # means rolling(252) produces NaN for the early months,
    # which fillna(0) hides. Result: MOM is silently near-zero.
    # Fix: use the full series, then reindex at the end.
    _roll_12m   = log_return.rolling( SIGNAL_LOOKBACKS['momentum'], min_periods=int(SIGNAL_LOOKBACKS['momentum'] * 0.8)).sum()
    _roll_1m    = log_return.rolling( SIGNAL_LOOKBACKS['reversal'], min_periods=int(SIGNAL_LOOKBACKS['reversal'] * 0.7)).sum()
    _mom_raw    = _roll_12m - _roll_1m     # skip last month (t-21 to t-0)

    # Cross-sectional z-score each day → captures relative momentum
    _mom_mean   = _mom_raw.mean(axis=1)
    _mom_std    = _mom_raw.std(axis=1).replace(0, np.nan)
    _mom_zscore = _mom_raw.sub(_mom_mean, axis=0).div(_mom_std, axis=0)

    # Daily MOM factor = equal-weighted mean of z-scores
    # Align to factor dates AFTER full computation
    mom_factor = (_mom_zscore
                .mean(axis=1)
                .reindex(mkt_factor.index)
                .fillna(0))
    mom_factor.name = 'MOM'

    # Combine into factor DataFrame, aligned to factor dates
    factors = pd.DataFrame({
        'MKT'  : mkt_factor,
        'SIZE' : size_factor,
        'MOM'  : mom_factor,
    }).dropna()

    # Align stock returns to factor dates for regression
    _stock_ret = log_return_aligned.reindex(factors.index).dropna()
    factors    = factors.reindex(_stock_ret.index)

    print(f"  Factor series length : {len(factors)} days")
    print(f"  MKT  ann. premium    : {mkt_factor.mean()*TRADING_DAYS*100:+.2f}%")
    print(f"  SIZE ann. premium    : {size_factor.mean()*TRADING_DAYS*100:+.2f}%")
    print(f"  MOM  daily mean      : {mom_factor.mean():+.4f} (z-score)")
    print(f"{'─'*58}")

    # ============================================================
    # 2B. Factor-structured covariance matrix (BL-MF approach)
    #
    #     Standard Ledoit-Wolf estimates the full (n×n) covariance
    #     matrix from raw price history. With 72 stocks and 5 years
    #     of data, every entry carries estimation noise from
    #     company-specific events that will not repeat.
    #
    #     Factor-structured decomposition:
    #       Cov = B × F × B.T + D
    #
    #     B (n × k) = factor loadings per stock, estimated via OLS
    #     F (k × k) = factor covariance (just 3×3 — very stable)
    #     D (n × n) = diagonal residual variances (idiosyncratic)
    #
    #     Benefits over raw Ledoit-Wolf:
    #     1. Off-diagonal entries driven by shared factor exposure
    #        only — not by accidental co-movement in the sample
    #     2. Factor covariance (3×3) is estimated from hundreds
    #        of observations per parameter — far more reliable
    #        than the 72×72 sample matrix
    #     3. Positive definiteness guaranteed by construction
    #     4. Matches the BL-MF paper (2025) approach for
    #        simultaneous return and covariance improvement
    #
    #     Fallback: if factor regression fails for too many stocks,
    #     falls back to standard Ledoit-Wolf (original behavior).
    # ============================================================

    print(f"\n  Running OLS factor regressions ({n_stk} stocks)...")

    X = sm.add_constant(factors)    # adds intercept column (alpha)

    _factor_betas = {}    # loadings per stock (for diagnostics)
    _factor_r2    = {}    # R² per stock
    _residuals    = {}    # OLS residuals per stock

    _low_r2 = []
    _failed = []

    for ticker in log_return.columns:
        y   = _stock_ret[ticker].dropna()
        X_t = X.reindex(y.index).dropna()
        y   = y.reindex(X_t.index)

        if len(y) < 60:
            _failed.append(ticker)
            _factor_betas[ticker] = {'const': 0, 'MKT': 1, 'SIZE': 0, 'MOM': 0}
            _factor_r2[ticker]    = 0.0
            _residuals[ticker]    = y - y.mean()
            continue

        model = sm.OLS(y, X_t, missing='drop').fit()
        _factor_betas[ticker] = model.params.to_dict()
        _factor_r2[ticker]    = model.rsquared
        _residuals[ticker]    = model.resid

        if model.rsquared < R2_MIN_THRESHOLD:
            _low_r2.append(ticker)

    if _failed:
        print(f"  ⚠ {len(_failed)} stocks had < 60 obs — using MKT-only beta")
    if _low_r2:
        print(f"  ⚠ {len(_low_r2)} stocks with R² < 0.15 "
            f"(factor model explains little variance):")
        for t in _low_r2:
            print(f"    {t.replace('.NS',''):<20} R²={_factor_r2[t]:.3f}")
    else:
        avg_r2 = np.mean(list(_factor_r2.values()))
        print(f"  ✓ Average R² across stocks: {avg_r2:.3f}")

    # Build B matrix (n_stocks × 3) — factor loadings only, no intercept
    _factor_names = ['MKT', 'SIZE', 'MOM']
    B = np.array([
        [_factor_betas[t].get(f, 0.0) for f in _factor_names]
        for t in log_return.columns
    ])  # shape: (n_stocks, 3)

    # Factor covariance F (3×3) — estimated from factor returns directly
    # Using daily factor returns — very stable (hundreds of obs per param)
    F_daily = factors[_factor_names].cov().values   # (3, 3)

    # Residual variances D — diagonal only (idiosyncratic)
    _resid_df = pd.DataFrame(
        {t: _residuals.get(t, pd.Series(dtype=float))
        for t in log_return.columns}
    ).dropna()

    D_daily = np.diag(np.var(_resid_df.values, axis=0, ddof=1))  # (n, n)

    # Factor-structured daily covariance
    _cov_factor_daily = B @ F_daily @ B.T + D_daily   # (n, n)
    _cov_factor_daily = pd.DataFrame(
        _cov_factor_daily,
        index=log_return.columns, columns=log_return.columns
    )

    # Ledoit-Wolf fallback — always compute for comparison
    lw = LedoitWolf()
    lw.fit(log_return.values)
    _cov_lw_daily = pd.DataFrame(
        lw.covariance_,
        index=log_return.columns, columns=log_return.columns
    )

    # Decide which to use
    # If factor model R² is good (most stocks > 0.15), use factor structure.
    # Otherwise fall back to Ledoit-Wolf.

    _frac_low_r2 = len(_low_r2) / n_stk
    if _frac_low_r2 > LOW_R2_FALLBACK_THRESHOLD:
        print(f"\n  ⚠ >40% of stocks have low R² — using Ledoit-Wolf fallback.")
        print(f"  Tip: increase LOOKBACK_Y or check FACTOR_TICKERS in Module 0.")
        cov_daily = _cov_lw_daily
        _cov_method = 'Ledoit-Wolf (fallback)'
    else:
        cov_daily   = _cov_factor_daily
        _cov_method = 'Factor-structured (B×F×Bᵀ + D)'

    cov_annual = cov_daily * TRADING_DAYS

    print(f"\n  Covariance method    : {_cov_method}")
    print(f"  Matrix shape         : {cov_daily.shape}")
    print(f"  Ledoit-Wolf shrinkage: {lw.shrinkage_:.4f} (for reference)")
    print(f"  Factor-LW max diff   : "
        f"{np.abs(_cov_factor_daily.values - _cov_lw_daily.values).max():.4e}")

    # ============================================================
    # 2C. Covariance reliability check
    # ============================================================

    ratio = n_obs / n_stk
    print(f"\nCovariance reliability:")
    print(f"  Observations  : {n_obs:,} trading days")
    print(f"  Stocks        : {n_stk}")
    print(f"  Ratio         : {ratio:.1f}x  (obs per stock)")
    if ratio >= 10:
        print(f"  ✓ Excellent — matrix is highly reliable")
    elif ratio >= 5:
        print(f"  ✓ Acceptable — matrix is usable")
    elif ratio >= 3:
        print(f"  ⚠ Marginal — factor structure is critical at this ratio.")
    else:
        print(f"  ✗ Unreliable — reduce stocks or increase LOOKBACK_Y.")

    # ============================================================
    # 2D. Historical baselines (always computed for comparison)
    #
    #     Equal-weighted and EWMA baselines are computed regardless
    #     of RETURN_METHOD. They appear in the comparison table
    #     in section 2G so you can see how BL and factor estimates
    #     differ from raw history.
    # ============================================================

    log_mean_annual = log_return.mean() * TRADING_DAYS
    arith_equal     = np.exp(log_mean_annual) - 1

    ewma_log_mean = (log_return
                    .ewm(halflife=FACTOR_HALFLIFE)
                    .mean()
                    .iloc[-1] * TRADING_DAYS)
    arith_ewma    = np.exp(ewma_log_mean) - 1

    print(f"\n{'─'*58}")
    print(f"  Historical baselines: equal + EWMA computed.")
    print(f"  EWMA halflife: {FACTOR_HALFLIFE} days")
    print(f"{'─'*58}")

    # ============================================================
    # 2E. Factor-model expected returns (OLS per stock)
    #
    #     For each stock:
    #       E[r_i] = rf + β_MKT × premium_MKT
    #                   + β_SIZE × premium_SIZE
    #                   + β_MOM  × premium_MOM
    #
    #     Factor premiums = EWMA-weighted realized mean of each
    #     factor over the estimation window. This is the standard
    #     approach when no separate premium model is available.
    #
    #     Alpha (intercept) is intentionally excluded.
    #     Persistent alpha over 5 years is very rare — including
    #     it leads to massive overfit in the return estimates.
    #
    #     R² diagnostic: stocks with R² < 0.15 have returns
    #     driven mostly by idiosyncratic events. Factor estimates
    #     for these stocks should be viewed with skepticism —
    #     they will be pulled toward the mean by James-Stein
    #     shrinkage in section 2I anyway.
    # ============================================================
    # Full 5-year mean
    factor_premiums = factors[_factor_names].mean() * TRADING_DAYS

    # MKT floor
    factor_premiums['MKT']  = max(factor_premiums['MKT'],  MKT_FLOOR)
    factor_premiums['SIZE'] = max(factor_premiums['SIZE'], SIZE_FLOOR)

    # Adaptive SIZE scalar from Module 0 regime detection
    # SIZE_REGIME_SCALAR was computed live from recent market data
    factor_premiums['SIZE'] = factor_premiums['SIZE'] * SIZE_REGIME_SCALAR

    print(f"\n  Factor premiums (regime-adjusted):")
    print(f"  MKT  raw={factors['MKT'].mean()*TRADING_DAYS*100:+.2f}%  "
        f"→ floored={factor_premiums['MKT']*100:+.2f}%")
    print(f"  SIZE raw={factors['SIZE'].mean()*TRADING_DAYS*100:+.2f}%  "
        f"→ after {SIZE_REGIME} scalar "
        f"({SIZE_REGIME_SCALAR:.2f}x)="
        f"{factor_premiums['SIZE']*100:+.2f}%")
    print(f"  MOM  {factor_premiums['MOM']:+.4f}")


    # Per-stock factor expected returns
    # E[r_i] = Rf + β_MKT × premium_MKT + β_SIZE × premium_SIZE + β_MOM × premium_MOM
    _factor_exp_rets = {}
    for ticker in log_return.columns:
        b = _factor_betas[ticker]
        _factor_exp_rets[ticker] = (
            RISK_FREE
            + b.get('MKT',  0) * factor_premiums['MKT']
            + b.get('SIZE', 0) * factor_premiums['SIZE']
            + b.get('MOM',  0) * factor_premiums['MOM']
        )

    arith_factor = pd.Series(_factor_exp_rets).reindex(log_return.columns)

    print(f"\n  Factor expected returns (top 5 / bottom 5):")
    for t, v in arith_factor.nlargest(5).items():
        print(f"    {t.replace('.NS',''):<16} {v*100:+.2f}%")
    print(f"    ...")
    for t, v in arith_factor.nsmallest(5).items():
        print(f"    {t.replace('.NS',''):<16} {v*100:+.2f}%")

    # ---- publish (generated) ----
    _scope = locals()
    ctx.update({k: _scope[k] for k in _PUBLISHED if k in _scope})

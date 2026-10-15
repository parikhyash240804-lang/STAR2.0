"""
Legacy Stardust 1.x - M2 (2F-2K) - Black-Litterman returns, method selection, shrinkage, checks

Mechanically extracted from stardust_core.py lines 1674-1965 by
legacy/tools/split_monolith.py. The code inside run() is the original,
byte-for-byte; only the input/publish scaffolding around it is generated.
Do not hand-edit the extracted block - fix bugs in STAR2.0 (star/) instead.
"""
from ._bootstrap import ensure_bootstrap

ensure_bootstrap()

#: data this stage reads from earlier stages (its upstream contract)
INPUTS = (
    '_cov_lw_daily',
    '_manual_overrides',
    'arith_equal',
    'arith_ewma',
    'arith_factor',
    'BL_DELTA',
    'BL_TAU',
    'BL_VIEW_CONFIDENCE',
    'BL_VIEWS',
    'FACTOR_HALFLIFE',
    'log_return',
    'n_stk',
    'RETURN_CAP',
    'RISK_FREE',
    'SHRINK_FACTOR',
    'TRADING_DAYS',
)

#: legacy loop variables that leak in from earlier stages; pulled only so the
#: split behaves exactly like the monolith's single shared scope
CARRYOVER = (
    'cov_annual',
    'cov_daily',
    'RETURN_METHOD',
)

#: data this stage produces that a later stage reads (downstream contract)
OUTPUTS = (
    'active_label',
    'arith_bl',
    'arith_returns',
    'cov_annual',
    'cov_daily',
    'Pi_s',
    'RETURN_METHOD',
)

#: everything bound at stage scope; all of it is published to ctx
_PUBLISHED = ('M1_inv', 'Omega', 'Omega_inv', 'P', 'Pi', 'Pi_s', 'Q', 'RETURN_METHOD', 'Sigma', '_aligned', '_all_tickers', '_diag_vals', '_eigenvalues', '_hybrid', '_info', '_invalid', '_is_symmetric', '_mcap', '_mcap_series', '_methods', '_n_views', '_selected', '_suspicious', '_t', '_valid_views', '_vtickers', 'act', 'active_label', 'arith_bl', 'arith_returns', 'bl', 'capped', 'conf', 'cov_annual', 'cov_daily', 'eq', 'ew', 'fac', 'grand_mean', 'i', 'k', 'mu_bl', 'ret', 't', 'tau', 'tau_Sigma_inv', 'ticker', 'v', 'w', 'w_mkt')


def run(ctx: dict) -> None:
    # ---- inputs from upstream stages (generated) ----
    _cov_lw_daily = ctx['_cov_lw_daily']
    _manual_overrides = ctx['_manual_overrides']
    arith_equal = ctx['arith_equal']
    arith_ewma = ctx['arith_ewma']
    arith_factor = ctx['arith_factor']
    BL_DELTA = ctx['BL_DELTA']
    BL_TAU = ctx['BL_TAU']
    BL_VIEW_CONFIDENCE = ctx['BL_VIEW_CONFIDENCE']
    BL_VIEWS = ctx['BL_VIEWS']
    FACTOR_HALFLIFE = ctx['FACTOR_HALFLIFE']
    log_return = ctx['log_return']
    n_stk = ctx['n_stk']
    RETURN_CAP = ctx['RETURN_CAP']
    RISK_FREE = ctx['RISK_FREE']
    SHRINK_FACTOR = ctx['SHRINK_FACTOR']
    TRADING_DAYS = ctx['TRADING_DAYS']
    # ---- read from upstream, then rebound here (generated) ----
    if 'cov_annual' in ctx: cov_annual = ctx['cov_annual']
    if 'cov_daily' in ctx: cov_daily = ctx['cov_daily']
    if 'RETURN_METHOD' in ctx: RETURN_METHOD = ctx['RETURN_METHOD']
    # ---- imports made upstream, repeated locally (generated) ----
    import numpy as np
    import pandas as pd
    # ---- original code: stardust_core.py lines 1674-1965 ----
    # ============================================================
    # 2F. Black-Litterman expected returns
    #
    #     BL blends two information sources in a Bayesian way:
    #
    #     PRIOR (Pi) — market-implied equilibrium returns
    #       Reverse-optimized from Nifty 500 market-cap weights.
    #       We approximate market-cap weights using equal weights
    #       as a starting point — you can replace with actual
    #       free-float market cap weights from NSE if available.
    #       Formula: Pi = delta × Sigma × w_market
    #
    #     POSTERIOR (mu_bl) — blends Pi with your views
    #       If BL_VIEWS is empty: mu_bl = Pi exactly.
    #       If views supplied: tilts toward them by BL_VIEW_CONFIDENCE.
    #
    #     OMEGA — view uncertainty matrix (diagonal)
    #       omega_ii = ((1-conf)/conf) × tau × P_i @ Sigma @ P_i.T
    #       High conf → small omega → views dominate the posterior.
    #       Low conf  → large omega → prior dominates.
    #
    #     The equilibrium prior is forward-looking and anchored
    #     to market prices, not to the historical bull run.
    #     This is the core reason BL outperforms raw history.
    # ============================================================

    print(f"\n{'─'*58}")
    print(f"  Computing Black-Litterman posterior...")

    # Market-cap proxy weights
    # Using equal weights as approximation — replace with actual
    # NSE free-float market cap weights if you have them.
    # For a 72-stock portfolio equal weights give a reasonable
    # diversified prior — better than concentrating the prior
    # in whichever stocks happened to be large in 2019.
    import yfinance as yf

    _mcap = {}
    for _t in log_return.columns:
        try:
            _info = yf.Ticker(_t).fast_info
            _mcap[_t] = getattr(_info, 'market_cap', None) or 1.0
        except:
            _mcap[_t] = 1.0

    _mcap_series = pd.Series(_mcap).reindex(log_return.columns).fillna(1.0)
    w_mkt = (_mcap_series / _mcap_series.sum()).values
    print(f"  Top 5 by market cap weight:")
    for t, w in _mcap_series.nlargest(5).items():
        print(f"    {t.replace('.NS',''):<16} {w/_mcap_series.sum()*100:.2f}%")

    Sigma = cov_annual.values           # (n × n) — annual scale for BL
    Pi    = BL_DELTA * Sigma @ w_mkt   # (n,) equilibrium returns
    Pi_s  = pd.Series(Pi, index=cov_annual.index)

    print(f"  Equilibrium prior (Pi):")
    print(f"    Mean : {Pi.mean()*100:+.2f}%  "
        f"Min : {Pi.min()*100:+.2f}%  "
        f"Max : {Pi.max()*100:+.2f}%")

    # Views
    if BL_VIEWS:
        _valid_views = {k: v for k, v in BL_VIEWS.items()
                        if k in cov_annual.index}
        _invalid = set(BL_VIEWS.keys()) - set(_valid_views.keys())
        if _invalid:
            print(f"  ⚠ Views ignored (ticker not in universe): {_invalid}")

        if _valid_views:
            _vtickers  = list(_valid_views.keys())
            _n_views   = len(_vtickers)
            _all_tickers = list(cov_annual.index)

            P = np.zeros((_n_views, n_stk))
            Q = np.zeros(_n_views)
            for i, (t, ret) in enumerate(_valid_views.items()):
                P[i, _all_tickers.index(t)] = 1.0
                Q[i] = ret  # already excess return (systematic views converted in _score_to_view)
                            # manual overrides from Module 0 are entered as total returns → subtract rf
                if t in _manual_overrides:
                    Q[i] = ret - RISK_FREE

            conf  = BL_VIEW_CONFIDENCE
            tau   = BL_TAU
            Omega = np.diag([
                ((1 - conf) / conf) * tau * float(P[i] @ Sigma @ P[i].T)
                for i in range(_n_views)
            ])

            # BL closed-form posterior mean
            tau_Sigma_inv = np.linalg.inv(tau * Sigma)
            Omega_inv     = np.linalg.inv(Omega)
            M1_inv = np.linalg.inv(tau_Sigma_inv + P.T @ Omega_inv @ P)
            mu_bl  = M1_inv @ (tau_Sigma_inv @ Pi + P.T @ Omega_inv @ Q)

            print(f"\n  {_n_views} view(s) incorporated:")
            for t, ret in _valid_views.items():
                print(f"    {t.replace('.NS',''):<20} "
                    f"view={ret*100:+.1f}%  "
                    f"prior={Pi_s[t]*100:+.1f}%")
            print(f"  View confidence : {conf*100:.0f}%")
        else:
            mu_bl = Pi.copy()
            print(f"  No valid views — using pure equilibrium Pi")
    else:
        mu_bl = Pi.copy()
        print(f"  No views supplied — using pure equilibrium Pi")
        print(f"  (Add views in BL_VIEWS dict in Module 0 to tilt)")

    arith_bl = pd.Series(mu_bl, index=cov_annual.index).reindex(
        log_return.columns
    )

    print(f"\n  BL posterior:")
    print(f"    Mean : {arith_bl.mean()*100:+.2f}%  "
        f"Min : {arith_bl.min()*100:+.2f}%  "
        f"Max : {arith_bl.max()*100:+.2f}%")
    print(f"{'─'*58}")

    # ============================================================
    # 2G. Method selection
    #
    #     RETURN_METHOD from Module 0 controls which estimate
    #     becomes arith_returns — the variable consumed by all
    #     downstream modules (3, 4, 5, 6).
    #
    #     CRITICAL: arith_returns must NEVER be sorted.
    #     Its index must match log_return.columns exactly.
    #     All downstream modules index-pair arith_returns.values
    #     with weight vectors — any reordering silently assigns
    #     the wrong return to the wrong stock.
    # ============================================================

    _hybrid = (
        0.5 * arith_bl.reindex(log_return.columns)
        + 0.5 * arith_factor.reindex(log_return.columns)
    )

    _methods = {
        'equal'  : (arith_equal,  'Equal-weighted 5-year mean'),
        'ewma'   : (arith_ewma,   f'EWMA (halflife={FACTOR_HALFLIFE} days)'),
        'bl'     : (arith_bl,     'Black-Litterman posterior'),
        'factor' : (arith_factor, '3-factor OLS (MKT, SIZE, MOM)'),
        'hybrid' : (_hybrid,      'Hybrid (50% BL + 50% factor)'),
    }

    if RETURN_METHOD not in _methods:
        print(f"⚠ Unknown RETURN_METHOD '{RETURN_METHOD}' — defaulting to 'bl'")
        RETURN_METHOD = 'bl'

    _selected, active_label = _methods[RETURN_METHOD]
    arith_returns = _selected.reindex(log_return.columns).copy()

    # ============================================================
    # 2H. Comparison table
    #
    #     All five estimates side-by-side.
    #     Large divergence between methods = that stock's
    #     return estimate is uncertain — optimizer weights
    #     for it should be viewed with skepticism.
    #     '◄ ACTIVE' marks the method selected.
    # ============================================================

    print(f"\nReturn method selected : {active_label}")
    print(f"\n{'Ticker':<16} {'Equal%':>8} {'EWMA%':>8} "
        f"{'Factor%':>9} {'BL%':>8} {'Active%':>9}")
    print("─" * 62)

    for ticker in log_return.columns:
        eq  = arith_equal.get(ticker,  np.nan) * 100
        ew  = arith_ewma.get(ticker,   np.nan) * 100
        fac = arith_factor.get(ticker, np.nan) * 100
        bl  = arith_bl.get(ticker,     np.nan) * 100
        act = arith_returns.get(ticker, np.nan) * 100
        print(f"  {ticker.replace('.NS',''):<14} "
            f"{eq:>7.1f}% {ew:>7.1f}% "
            f"{fac:>8.1f}% {bl:>7.1f}% "
            f"{act:>8.1f}%")

    print(f"\n  Active method : {active_label}")

    # ============================================================
    # 2I. Sanity checks on covariance matrix
    # ============================================================

    _is_symmetric = np.allclose(cov_daily, cov_daily.T)
    _diag_vals    = np.diag(cov_daily.values)
    _eigenvalues  = np.linalg.eigvalsh(cov_daily.values)

    print(f"\nCovariance sanity checks:")
    print(f"  Symmetric matrix         : {_is_symmetric}")
    print(f"  All variances positive   : {(_diag_vals > 0).all()}")
    print(f"  All eigenvalues positive : {(_eigenvalues > 0).all()}")
    print(f"  Min eigenvalue           : {_eigenvalues.min():.2e}")

    if not (_eigenvalues > 0).all():
        print(f"  ⚠ Negative eigenvalue — Cholesky may fail in Module 5.")
        print(f"  Falling back to Ledoit-Wolf for stability...")
        cov_daily  = _cov_lw_daily
        cov_annual = cov_daily * TRADING_DAYS
        print(f"  ✓ Ledoit-Wolf covariance substituted.")
    else:
        print(f"  ✓ Cholesky decomposition will succeed in Module 5.")

    _suspicious = arith_returns[(arith_returns > 2.0) | (arith_returns < -0.8)]
    if not _suspicious.empty:
        print(f"\n  ⚠ Suspicious returns — check these tickers:")
        for t, v in _suspicious.items():
            print(f"    {t}: {v*100:+.1f}%")
    else:
        print(f"  ✓ All return estimates look reasonable.")

    # ============================================================
    # 2J. James-Stein shrinkage on return estimates
    #
    #     Pulls extreme estimates toward the cross-sectional mean.
    #     Applied AFTER method selection, on the unsorted
    #     arith_returns — critical for alignment.
    #
    #     For BL and factor returns, shrinkage has less effect
    #     than on raw historical returns (they're already more
    #     moderate). SHRINK_FACTOR=0.1 is appropriate for BL.
    #     Increase to 0.3 if optimizer concentrates too heavily.
    # ============================================================

    grand_mean    = arith_returns.mean()
    arith_returns = ((1 - SHRINK_FACTOR) * arith_returns
                    + SHRINK_FACTOR * grand_mean)

    print(f"\nPost-shrinkage returns "
        f"(JS factor={SHRINK_FACTOR}, method={RETURN_METHOD}):")
    print(f"{'Ticker':<16} {'Return%':>10}")
    print("─" * 28)

    for ticker, ret in arith_returns.sort_values(ascending=False).items():
        print(f"  {ticker.replace('.NS',''):<14} {ret*100:>+9.2f}%")

    # Sanity cap — no stock should expect >20% from a factor model
    # without analyst views. Caps prevent optimizer over-concentrating
    # in high-beta cyclicals driven purely by SIZE loading.
    # 20% annual — adjust if needed
    arith_returns = arith_returns.clip(upper=RETURN_CAP)

    capped = arith_returns[arith_returns >= RETURN_CAP]
    if not capped.empty:
        print(f"\n  ⚠ {len(capped)} stock(s) capped at {RETURN_CAP*100:.0f}%:")
        for t, v in capped.items():
            print(f"    {t.replace('.NS',''):<16} capped at {RETURN_CAP*100:.0f}%")

    # ============================================================
    # 2K. Alignment guarantee — hard assertion before Module 3
    # ============================================================

    _aligned = list(arith_returns.index) == list(log_return.columns)
    print(f"\nAlignment check:")
    print(f"  arith_returns.index == log_return.columns : {_aligned}")

    assert _aligned, (
        "CRITICAL ALIGNMENT FAILURE.\n"
        "arith_returns index does not match log_return columns.\n"
        "arith_returns must never be sorted or reindexed.\n"
        "Rerun Module 2 from scratch."
    )
    print(f"  ✓ Alignment confirmed — safe to proceed to Module 3.")

    # ============================================================
    # MODULE 2 OUTPUTS — passed into Modules 3, 4, 5, 6
    # ============================================================
    # cov_daily     → pd.DataFrame : daily covariance
    #                 (factor-structured or Ledoit-Wolf fallback)
    # cov_annual    → pd.DataFrame : annualized covariance
    # arith_returns → pd.Series    : expected returns (method-selected,
    #                                James-Stein shrunk, NEVER sorted)
    #
    # Diagnostic variables (inspection only, not consumed downstream):
    #   arith_equal   → equal-weighted historical baseline
    #   arith_ewma    → EWMA historical baseline
    #   arith_factor  → 3-factor OLS expected returns
    #   arith_bl      → Black-Litterman posterior returns
    #   Pi_s          → BL equilibrium prior (pd.Series)
    #   active_label  → string name of active method
    #   _factor_betas → dict of per-stock betas
    #   _factor_r2    → dict of per-stock R²
    #   _cov_method   → which covariance was used
    #   factor_premiums → annualized factor premiums
    #
    # ALIGNMENT GUARANTEE:
    #   arith_returns.index
    #   == log_return.columns
    #   == cov_daily.index == cov_daily.columns
    #   == cov_annual.index == cov_annual.columns
    # ============================================================
    # ---- publish (generated) ----
    _scope = locals()
    ctx.update({k: _scope[k] for k in _PUBLISHED if k in _scope})

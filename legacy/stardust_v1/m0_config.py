"""
Legacy Stardust 1.x - M0 - Global configuration (+ adaptive SIZE regime detection)

Mechanically extracted from stardust_core.py lines 28-552 by
legacy/tools/split_monolith.py. The code inside run() is the original,
byte-for-byte; only the input/publish scaffolding around it is generated.
Do not hand-edit the extracted block - fix bugs in STAR2.0 (star/) instead.
"""
from ._bootstrap import ensure_bootstrap

ensure_bootstrap()

#: data this stage reads from earlier stages (its upstream contract)
INPUTS = (
    'params',
    'tickers',
)

#: legacy loop variables that leak in from earlier stages; pulled only so the
#: split behaves exactly like the monolith's single shared scope
CARRYOVER = (
    'output_dir',
)

#: data this stage produces that a later stage reads (downstream contract)
OUTPUTS = (
    'BENCHMARK',
    'BL_DELTA',
    'BL_TAU',
    'BL_VIEW_CONFIDENCE',
    'BL_VIEWS',
    'END_DATE',
    'FACTOR_HALFLIFE',
    'FACTOR_TICKERS',
    'GAIN_THRESHOLD_1',
    'GAIN_THRESHOLD_2',
    'GLOBAL_SEED',
    'INITIAL_VALUE',
    'LOOKBACK_Y',
    'LOW_R2_FALLBACK_THRESHOLD',
    'MAX_ASSETS',
    'MIN_PEER_GROUP',
    'MISSING_THRESHOLD',
    'MKT_FLOOR',
    'output_dir',
    'R2_MIN_THRESHOLD',
    'REBALANCE_FREQ',
    'REBALANCE_PERIODS_PER_YEAR',
    'RETURN_CAP',
    'RETURN_METHOD',
    'RISK_FREE',
    'ROLLING_WINDOW_YRS',
    'SCORE_CEILING',
    'SCORE_FLOOR',
    'SECTOR_MULTIPLIERS',
    'SHRINK_FACTOR',
    'SIGNAL_LOOKBACKS',
    'SIGNAL_WEIGHTS',
    'SIZE_FLOOR',
    'SIZE_REGIME',
    'SIZE_REGIME_SCALAR',
    'START_DATE',
    'STRESS_SCENARIOS',
    'T_DIST_DF',
    'TICKERS',
    'TRADING_DAYS',
    'TRANSACTION_COST',
    'VIEW_CEILING',
    'VIEW_FLOOR',
    'VOLUME_MIN',
    'WF_N_STARTS',
    'WINSOR_CAP',
)

#: everything bound at stage scope; all of it is published to ctx
_PUBLISHED = ('BENCHMARK', 'BL_DELTA', 'BL_TAU', 'BL_VIEWS', 'BL_VIEW_CONFIDENCE', 'CUSTOM_DROP_PCT', 'CUSTOM_NAME', 'CUSTOM_SCENARIO_ON', 'END_DATE', 'FACTOR_HALFLIFE', 'FACTOR_TICKERS', 'GAIN_THRESHOLD_1', 'GAIN_THRESHOLD_2', 'GLOBAL_SEED', 'INITIAL_VALUE', 'LOOKBACK_Y', 'LOW_R2_FALLBACK_THRESHOLD', 'MAX_ASSETS', 'MAX_STOCKS', 'MIN_PEER_GROUP', 'MIN_STOCKS', 'MISSING_THRESHOLD', 'MKT_FLOOR', 'R2_MIN_THRESHOLD', 'REBALANCE_FREQ', 'REBALANCE_PERIODS_PER_YEAR', 'RETURN_CAP', 'RETURN_METHOD', 'RISK_FREE', 'ROLLING_WINDOW_YRS', 'SCORE_CEILING', 'SCORE_FLOOR', 'SECTOR_MULTIPLIERS', 'SHRINK_FACTOR', 'SIGNAL_LOOKBACKS', 'SIGNAL_WEIGHTS', 'SIZE_FLOOR', 'SIZE_REGIME', 'SIZE_REGIME_SCALAR', 'START_DATE', 'STRESS_SCENARIOS', 'TICKERS', 'TRADING_DAYS', 'TRANSACTION_COST', 'T_DIST_DF', 'VIEW_CEILING', 'VIEW_FLOOR', 'VOLUME_MIN', 'WF_N_STARTS', 'WINSOR_CAP', '_REGIME_INTERP', '_detect_size_regime', '_size_spread', '_tag', '_wt_sum', 'drop', 'mult', 'name', 'output_dir', 'sec', 'sig', 'wt')


def run(ctx: dict) -> None:
    # ---- inputs from upstream stages (generated) ----
    params = ctx['params']
    tickers = ctx['tickers']
    # ---- read from upstream, then rebound here (generated) ----
    if 'output_dir' in ctx: output_dir = ctx['output_dir']
    # ---- original code: stardust_core.py lines 28-552 ----
    
    # ============================================================
    # MODULE 0 — Global Configuration
    # ------------------------------------------------------------
    # THE ONLY CELL YOU NEED TO EDIT.
    # All other modules read from these variables automatically.
    #
    # Sections:
    #   0A  Stock universe
    #   0B  Universe validation 
    #   0C  Date range
    #   0D  Risk & return settings
    #   0E  Optimizer settings
    #   0F  Simulation settings
    #   0G  Simulation distribution
    #   0H  Stress scenarios
    #   0I  Benchmark
    #   0J  Configuration summary (auto-print)
    #   0K  Return estimation method
    #   0L  Black-Litterman parameters
    #         BL_VIEWS is now an OVERRIDE dict only —
    #         systematic views are auto-generated in 2B.7
    #         Add a ticker here ONLY for high-conviction
    #         manual overrides that bypass the signal model
    #   0M  Factor model parameters
    #   0N  Adaptive SIZE regime (auto-detected)
    #   0O  Multi-factor signal configuration (NEW)
    #   0P  Sector macro multipliers (NEW)
    #   0Q  Configuration summary — BL + factor + signals
    # ============================================================

    import datetime as dt
    import pandas   as pd

    import numpy    as np
    import yfinance as yf

    import pathlib
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_dir = pathlib.Path(output_dir)

    # -----------------------------------------------------------
    # 0A. Stock universe
    #     All tickers must be .NS (NSE) format
    #     Remove any stock by deleting its line
    #     Add new stocks by adding a new line
    #     Sector classification is fetched automatically in 2B.5
    #     — no manual sector tags needed here
    # -----------------------------------------------------------
    SIGNAL_WEIGHTS = params.get('SIGNAL_WEIGHTS', {
        'momentum'    : 0.30,
        'low_vol'     : 0.20,
        'earnings_yld': 0.20,
        'roe'         : 0.20,
        'eps_revision': 0.10,
        })
    CUSTOM_DROP_PCT    = params.get('CUSTOM_DROP_PCT',    25)
    CUSTOM_NAME        = params.get('CUSTOM_NAME',        'India Recession')

    VOLUME_MIN        = params.get('VOLUME_MIN', 10_000)
    BL_VIEWS           = params.get('BL_VIEWS', {})
    SECTOR_MULTIPLIERS = params.get('SECTOR_MULTIPLIERS', {})
    STRESS_SCENARIOS   = params.get('STRESS_SCENARIOS', {})
    TICKERS           = tickers
    BENCHMARK = params.get('BENCHMARK', '^NSEI')
    VIEW_FLOOR         = params.get('VIEW_FLOOR',          0.08)
    VIEW_CEILING       = params.get('VIEW_CEILING',        0.22)
    WF_N_STARTS        = params.get('WF_N_STARTS',         8)
    LOOKBACK_Y        = params.get('LOOKBACK_Y',        5)
    RISK_FREE         = params.get('RISK_FREE',         0.065)
    INITIAL_VALUE     = params.get('INITIAL_VALUE',     1_000_000)
    RETURN_METHOD     = params.get('RETURN_METHOD',     'ewma')
    SHRINK_FACTOR     = params.get('SHRINK_FACTOR',     0.00)
    T_DIST_DF         = params.get('T_DIST_DF',         6)
    if T_DIST_DF >= 500:          # treat as Normal per template documentation
        T_DIST_DF = np.inf

    BL_DELTA          = params.get('BL_DELTA',          0.50)
    BL_TAU            = params.get('BL_TAU',            0.10)
    BL_VIEW_CONFIDENCE= params.get('BL_VIEW_CONFIDENCE',0.40)
    MAX_ASSETS        = params.get('MAX_ASSETS',        25)
    RETURN_CAP        = params.get('RETURN_CAP',        0.30)
    ROLLING_WINDOW_YRS= params.get('ROLLING_WINDOW_YRS',3)
    REBALANCE_FREQ    = params.get('REBALANCE_FREQ',    '3ME')
    TRANSACTION_COST  = params.get('TRANSACTION_COST',  0.0035)
    GAIN_THRESHOLD_1  = params.get('GAIN_THRESHOLD_1',  0.20)
    GAIN_THRESHOLD_2  = params.get('GAIN_THRESHOLD_2',  0.30)
    MISSING_THRESHOLD = params.get('MISSING_THRESHOLD', 0.05)

    # -----------------------------------------------------------
    # 0B. Universe validation
    #     Minimum 3 : covariance matrix needs at least 3 stocks
    #     Maximum 100: beyond this mean-variance breaks down
    # -----------------------------------------------------------
    if ROLLING_WINDOW_YRS >= LOOKBACK_Y:
        print(f"⚠ ROLLING_WINDOW_YRS ({ROLLING_WINDOW_YRS}) must be less than "
            f"LOOKBACK_Y ({LOOKBACK_Y}). Auto-adjusting to {LOOKBACK_Y - 1}.")
        ROLLING_WINDOW_YRS = LOOKBACK_Y - 1


    MIN_STOCKS = 3
    MAX_STOCKS = 100

    assert len(TICKERS) >= MIN_STOCKS, (
        f"Need at least {MIN_STOCKS} stocks. "
        f"You provided {len(TICKERS)}."
    )
    assert len(TICKERS) <= MAX_STOCKS, (
        f"Maximum {MAX_STOCKS} stocks supported. "
        f"You provided {len(TICKERS)}."
    )

    # -----------------------------------------------------------
    # 0C. Date range
    #     END_DATE   : today automatically
    #     LOOKBACK_Y : years of history (5 recommended)
    # -----------------------------------------------------------

    END_DATE   = pd.Timestamp.now()
    START_DATE = END_DATE - pd.DateOffset(years=LOOKBACK_Y)

    # -----------------------------------------------------------
    # 0D. Risk & return settings
    #     RISK_FREE    : RBI repo rate — update when it changes
    #                    Current: 6.5% (as of April 2026)
    #     TRADING_DAYS : NSE trading days per year — do not change
    # -----------------------------------------------------------

    TRADING_DAYS = 252

    # -----------------------------------------------------------
    # 0E. Optimizer settings
    #     SHRINK_FACTOR : James-Stein return shrinkage
    #                     0.05 = mild — correct when using BL/hybrid
    #                     (BL already moderates extremes; don't
    #                     double-shrink by using 0.2 here)
    # -----------------------------------------------------------

    # -----------------------------------------------------------
    # 0F. Simulation settings
    # -----------------------------------------------------------

    GLOBAL_SEED   = 42
    np.random.seed(GLOBAL_SEED)



    # -----------------------------------------------------------
    # 0G. Simulation distribution
    #     T_DIST_DF : degrees of freedom for Student's t
    #                 4-6 is standard for equity fat tails
    #                 np.inf = normal distribution
    # -----------------------------------------------------------

    # -----------------------------------------------------------
    # 0H. Stress scenarios
    #     Format: 'Scenario Name' : market_drop_as_decimal
    # -----------------------------------------------------------


    CUSTOM_SCENARIO_ON = True

    if CUSTOM_SCENARIO_ON:
        STRESS_SCENARIOS[CUSTOM_NAME] = -abs(CUSTOM_DROP_PCT / 100)

    # -----------------------------------------------------------
    # 0I. Benchmark
    # -----------------------------------------------------------


    # -----------------------------------------------------------
    # 0J. Configuration summary
    # -----------------------------------------------------------

    print("=" * 55)
    print(f"{'PORTFOLIO CONFIGURATION SUMMARY':^55}")
    print("=" * 55)
    print(f"  {'Stocks in universe':<30} {len(TICKERS):>21}")
    print(f"  {'Lookback period':<30} {f'{LOOKBACK_Y} years':>21}")
    print(f"  {'Start date':<30} {START_DATE.strftime('%d %b %Y'):>21}")
    print(f"  {'End date':<30} {END_DATE.strftime('%d %b %Y'):>21}")
    print(f"  {'─'*51}")
    print(f"  {'Risk-free rate':<30} {f'{RISK_FREE*100:.1f}%':>21}")
    print(f"  {'Trading days / year':<30} {TRADING_DAYS:>21}")
    print(f"  {'─'*51}")
    print(f"  {'Shrinkage factor':<30} {SHRINK_FACTOR:>21.2f}")
    print(f"  {'─'*51}")
    print(f"  {'Initial investment':<30} {'₹'+f'{INITIAL_VALUE:,.0f}':>21}")
    print(f"  {'Random seed':<30} {GLOBAL_SEED:>21}")
    print(f"  {'─'*51}")
    print(f"  {'Simulation distribution':<30} "
        f"{'Normal (GBM)' if T_DIST_DF == np.inf else f't-dist (df={T_DIST_DF})':>21}")
    print(f"  {'─'*51}")
    print(f"  {'Benchmark':<30} {BENCHMARK:>21}")
    print(f"  {'Stress scenarios':<30} {len(STRESS_SCENARIOS):>21}")
    for name, drop in STRESS_SCENARIOS.items():
        print(f"    {name:<28} {f'{drop*100:.0f}%':>21}")
    print("=" * 55)
    print(f"\n  ✓ Universe validated: {len(TICKERS)} stocks "
        f"(min={MIN_STOCKS}, max={MAX_STOCKS})")
    print(f"  ✓ Global seed set: {GLOBAL_SEED}")
    print(f"  ✓ Configuration ready — run Modules 1 through 6 in order.")
    print("=" * 55)

    # -----------------------------------------------------------
    # 0K. Return estimation method
    #
    #     'equal'  → 5-year equal-weighted historical mean
    #     'ewma'   → exponentially weighted mean (6M bias)
    #     'bl'     → Black-Litterman posterior only
    #     'factor' → 7-factor model only
    #     'hybrid' → 0.5 × BL + 0.5 × factor  ← RECOMMENDED
    # -----------------------------------------------------------


    # -----------------------------------------------------------
    # 0L. Black-Litterman parameters
    #
    #     BL_DELTA : market risk aversion (2.0 for India)
    #     BL_TAU   : prior uncertainty (0.05 = He & Litterman)
    #
    #     BL_VIEWS : MANUAL OVERRIDE DICT ONLY
    #     ─────────────────────────────────────────────────────
    #     Systematic views are now AUTO-GENERATED in section
    #     2B.7 using 5 quantitative signals (momentum, low-vol,
    #     earnings yield, ROE, EPS revision) scored within
    #     sector peer groups.
    #
    #     Only add a ticker here if you want to OVERRIDE the
    #     systematic signal with your own high-conviction view.
    #     Leave empty ({}) to use pure systematic views.
    #
    #     Example override:
    #         'RELIANCE.NS' : 0.20   # override signal — very
    #                                 # bullish on Jio+retail
    #
    #     BL_VIEW_CONFIDENCE : trust in views vs equilibrium
    #         0.35 = moderate conviction (recommended)
    #         Higher → views dominate more over prior
    # -----------------------------------------------------------


    # -----------------------------------------------------------
    # 0M. Factor model parameters
    #
    #     FACTOR_HALFLIFE : EWMA halflife in trading days
    #                       used for EWMA return baseline only
    #                       factor premiums use full 5Y mean
    #
    #     FACTOR_TICKERS  : index proxies for SIZE factor
    #                       MKT is always BENCHMARK (^NSEI)
    #
    #     RETURN_CAP      : no stock gets expected return > this
    #                       prevents optimizer concentrating in
    #                       high-SIZE-beta cyclicals (NATIONALUM
    #                       etc.) that the factor model inflates
    # -----------------------------------------------------------

    FACTOR_HALFLIFE = 252 #originally 252 or 1 year

    FACTOR_TICKERS = {
        'size' : 'NIFTY_MIDCAP_100.NS',
    }

    MKT_FLOOR  = 0.03   # 3%  — minimum annual market risk premium
    SIZE_FLOOR = 0.00   # 0%  — SIZE_REGIME_SCALAR handles the floor

    # -----------------------------------------------------------
    # 0N. Adaptive SIZE regime — auto-detected from live data
    #
    #     Fetches 6M midcap vs largecap return spread and maps
    #     it to a SIZE premium scalar applied in section 2E.
    #
    #     WHY THIS EXISTS:
    #     The factor model estimates SIZE premium from 5Y history.
    #     That history includes a strong midcap bull run (2021-26).
    #     If the current regime has shifted (RBI cutting rates,
    #     FII flows favouring largecap), blindly using 8%+ SIZE
    #     premium causes the model to overpile into cyclical
    #     midcaps. This scalar corrects for that in real time.
    #
    #     Scalar mapping:
    #       spread > +8%          bull_midcap   1.00x  (full premium)
    #       spread +3% to +8%     neutral_high  0.70x
    #       spread -3% to +3%     neutral       0.50x
    #       spread -8% to -3%     rate_cut      0.25x
    #       spread < -8%          bear_midcap   0.10x
    #
    #     Falls back to neutral (0.50x) if data fetch fails.
    # -----------------------------------------------------------

    def _detect_size_regime(
        midcap_ticker  : str = 'NIFTY_MIDCAP_100.NS',
        largecap_ticker: str = '^NSEI',
        lookback_days  : int = 126, #originally 126 or 6mnths
    ) -> tuple:
        """
        Returns (scalar, regime_name, spread_pct).
        Never raises — always returns a valid scalar.
        """
        _REGIME_MAP = [
            ( 0.08,  float('inf'), 'bull_midcap',  1.00),
            ( 0.03,  0.08,         'neutral_high', 0.70),
            (-0.03,  0.03,         'neutral',      0.50),
            (-0.08, -0.03,         'rate_cut',     0.25),
            (float('-inf'), -0.08, 'bear_midcap',  0.10),
        ]
        try:
            _end   = pd.Timestamp.now()
            _start = _end - pd.DateOffset(days=lookback_days * 2)

            _mid = yf.download(
                midcap_ticker, start=_start, end=_end,
                auto_adjust=True, progress=False
            )['Close'].squeeze()

            _lg = yf.download(
                largecap_ticker, start=_start, end=_end,
                auto_adjust=True, progress=False
            )['Close'].squeeze()

            _common = _mid.index.intersection(_lg.index)
            _mid    = _mid.loc[_common].iloc[-lookback_days:]
            _lg     = _lg.loc[_common].iloc[-lookback_days:]

            if len(_mid) < 60:
                raise ValueError("Insufficient data")

            _spread = float(
                _mid.iloc[-1] / _mid.iloc[0] - 1
            - (_lg.iloc[-1]  / _lg.iloc[0]  - 1)
            )

            for lo, hi, name, scalar in _REGIME_MAP:
                if lo <= _spread < hi:
                    return scalar, name, _spread

            return 0.50, 'neutral', _spread

        except Exception as _e:
            print(f"  ⚠ SIZE regime detection failed ({_e})"
                f" — falling back to neutral (0.50x)")
            return 0.50, 'neutral', 0.0


    SIZE_REGIME_SCALAR, SIZE_REGIME, _size_spread = _detect_size_regime(
        midcap_ticker   = 'NIFTY_MIDCAP_100.NS',
        largecap_ticker = '^NSEI',
        lookback_days   = 126,
    )

    _REGIME_INTERP = {
        'bull_midcap'  : 'Midcap outperforming — full SIZE premium used',
        'neutral_high' : 'Mild midcap lead — modest SIZE tilt',
        'neutral'      : 'No clear regime — balanced 0.50x assumption',
        'rate_cut'     : 'Largecap leading — SIZE premium haircut (0.25x)',
        'bear_midcap'  : 'Midcap underperforming — minimal SIZE (0.10x)',
    }

    print(f"\n{'─'*55}")
    print(f"  SIZE regime auto-detection")
    print(f"  6M midcap vs largecap spread : {_size_spread*100:+.2f}%")
    print(f"  Detected regime              : {SIZE_REGIME}")
    print(f"  SIZE scalar                  : {SIZE_REGIME_SCALAR:.2f}x")
    print(f"  Interpretation               : "
        f"{_REGIME_INTERP.get(SIZE_REGIME, 'Unknown')}")
    print(f"{'─'*55}\n")

    # -----------------------------------------------------------
    # 0O. Multi-factor signal configuration (NEW)
    #
    #     Controls the 5 quantitative signals used to auto-generate
    #     BL_VIEWS in section 2B.7. Each signal is z-scored within
    #     sector peer groups before blending, so weights are
    #     directly comparable.
    #
    #     SIGNAL_WEIGHTS must sum to 1.0 (enforced in 2B.7).
    #
    #     WHY THESE WEIGHTS?
    #       Momentum (0.30) : strongest anomaly in Indian equities,
    #                         NSE data shows 15-18% annual premium
    #       Low Vol  (0.20) : counterbalances SIZE beta tilt;
    #                         low-vol stocks systematically
    #                         overpay vs high-beta in India
    #       Earn Yld (0.20) : value signal; more stable than P/B
    #                         for Indian companies on yfinance
    #       ROE      (0.20) : FF5 RMW proxy; HDFC fund tilts here
    #       EPS Rev  (0.10) : most forward-looking but noisiest;
    #                         Indian markets under-react to EPS
    #                         surprises → extra edge when correct
    #
    #     SIGNAL_LOOKBACKS: trading days used per signal
    #       Momentum  : 252 days (1Y) minus 21 days (1M reversal)
    #       Low Vol   : 126 days (6M rolling std)
    #       Others    : determined by yfinance data availability
    #
    #     VIEW_FLOOR / VIEW_CEILING:
    #       All generated views are clipped to this range.
    #       Composite score of -3 → VIEW_FLOOR
    #       Composite score of +3 → VIEW_CEILING
    #       Prevents extreme views from dominating BL posterior.
    # -----------------------------------------------------------


    # Validate weights sum to 1.0
    _wt_sum = sum(SIGNAL_WEIGHTS.values())
    assert abs(_wt_sum - 1.0) < 1e-9, (
        f"SIGNAL_WEIGHTS must sum to 1.0 — currently {_wt_sum:.4f}. "
        f"Adjust values in section 0O."
    )

    SIGNAL_LOOKBACKS = {
        'momentum' : 252,    # 12M window for momentum signal
        'reversal' : 21,     # 1M window subtracted from momentum
        'low_vol'  : 126,    # 6M window for realised volatility
    }


    # Minimum peer group size for sector-relative z-scoring.
    # If a sector has fewer stocks than this threshold,
    # falls back to universe-level z-scoring for that signal.
    MIN_PEER_GROUP = 3
    SCORE_FLOOR   = -3.0
    SCORE_CEILING =  3.0

    WINSOR_CAP = 2.5
    # -----------------------------------------------------------
    # 0P. Sector macro multipliers (NEW)
    #
    #     Applied to the composite signal score AFTER sector-
    #     relative z-scoring. This is where YOUR macro judgment
    #     enters the systematic signal pipeline.
    #
    #     Multiplier > 1.0 → bullish tilt (views pushed higher)
    #     Multiplier < 1.0 → bearish tilt (views pulled lower)
    #     Multiplier = 1.0 → neutral (pure signal, no tilt)
    #
    #     Keys must match yfinance sector names exactly.
    #     Run section 2B.5 once to see which sectors your
    #     tickers map to, then update this dict accordingly.
    #
    #     If a stock's sector is not listed here, it gets 1.0
    #     (neutral) automatically — no error, just logged.
    #
    #     CURRENT THESIS (May 2026, targeting HDFC Ret. Fund):
    #       Bullish  : Financials (rate cut cycle)
    #                  Healthcare (domestic + US generics)
    #                  Industrials (govt capex, order books)
    #                  Energy (renewables push, 500GW target)
    #       Neutral  : Consumer (rural recovery, urban drag)
    #                  Utilities (regulated, low upside)
    #       Cautious : Technology (US demand weakness, AI disruption)
    #                  Basic Materials (China dumping, one-time tariff)
    #                  Consumer Cyclical (EV transition uncertainty)
    #
    #     Update this section when your macro thesis changes.
    #     No other code changes needed — everything downstream
    #     picks up the new multipliers automatically.
    # -----------------------------------------------------------



    # -----------------------------------------------------------
    # 0Q. Final configuration summary — BL + factor + signals
    # -----------------------------------------------------------

    print(f"  {'─'*51}")
    print(f"  {'Return method':<35} {RETURN_METHOD:>16}")
    print(f"  {'BL delta (risk aversion)':<35} {BL_DELTA:>16.1f}")
    print(f"  {'BL tau (prior weight)':<35} {BL_TAU:>16.3f}")
    print(f"  {'BL view confidence':<35} {BL_VIEW_CONFIDENCE:>16.0%}")
    print(f"  {'Manual BL overrides':<35} {len(BL_VIEWS):>16}")
    print(f"  {'Return cap (max any stock)':<35} {RETURN_CAP:>16.0%}")
    print(f"  {'Factor EWMA halflife':<35} {f'{FACTOR_HALFLIFE} days':>16}")
    print(f"  {'─'*51}")
    print(f"  {'Signal weights':}")
    for sig, wt in SIGNAL_WEIGHTS.items():
        print(f"    {sig:<30} {wt:>6.0%}")
    print(f"  {'View floor / ceiling':<35} "
        f"{VIEW_FLOOR*100:.0f}% – {VIEW_CEILING*100:.0f}%")
    print(f"  {'Min peer group size':<35} {MIN_PEER_GROUP:>16}")
    print(f"  {'─'*51}")
    print(f"  {'Sector multipliers':}")
    for sec, mult in SECTOR_MULTIPLIERS.items():
        _tag = ('↑ bullish' if mult > 1.0
                else '↓ cautious' if mult < 1.0
                else '→ neutral')
        print(f"    {sec:<28} {mult:.2f}x  {_tag}")
    print(f"  {'─'*51}")
    print(f"  ✓ BL-Factor config ready — Module 2 will use: {RETURN_METHOD}")
    print(f"  ✓ Systematic views will be generated in section 2B.7")
    if BL_VIEWS:
        print(f"  ⚠ {len(BL_VIEWS)} manual override(s) will replace "
            f"signal views for those tickers")
    else:
        print(f"  ✓ No manual overrides — pure systematic signal views")
    print(f"{'=' * 55}")
    #OTHERS
    LOW_R2_FALLBACK_THRESHOLD = 0.40
    R2_MIN_THRESHOLD = 0.15

    #MODULE 5F
    ###
    #BACKTEST
    # -----------------------------------------------------------
    # 0R. Walk-Forward Backtest Settings
    #     ROLLING_WINDOW_YRS : in-sample training window
    #                          must be < LOOKBACK_Y (currently 5)
    #                          3 years recommended for 5Y data
    #     REBALANCE_FREQ     : how often to rebalance
    #                          '3ME' = quarterly
    #                          '6ME' = semi-annual
    #                          'ME'  = monthly
    #     WF_N_STARTS        : optimizer restarts per window
    #                          lower than N_STARTS (Module 3)
    #                          because it runs many windows
    # -----------------------------------------------------------

    REBALANCE_PERIODS_PER_YEAR = {
        'ME' : 12,   # monthly
        '3ME': 4,    # quarterly
        'QE' : 4,    # quarterly (alias)
        '6ME': 2,    # semi-annual
    }

    # ---- publish (generated) ----
    _scope = locals()
    ctx.update({k: _scope[k] for k in _PUBLISHED if k in _scope})

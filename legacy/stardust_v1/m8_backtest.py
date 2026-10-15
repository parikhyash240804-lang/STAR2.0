"""
Legacy Stardust 1.x - M8 - Walk-forward backtest

Mechanically extracted from stardust_core.py lines 3610-4658 by
legacy/tools/split_monolith.py. The code inside run() is the original,
byte-for-byte; only the input/publish scaffolding around it is generated.
Do not hand-edit the extracted block - fix bugs in STAR2.0 (star/) instead.
"""
from ._bootstrap import ensure_bootstrap

ensure_bootstrap()

#: data this stage reads from earlier stages (its upstream contract)
INPUTS = (
    'BENCHMARK',
    'factor_index_ret',
    'GLOBAL_SEED',
    'INITIAL_VALUE',
    'log_return',
    'output_dir',
    'POLISH_FLOOR',
    'port_sharpe',
    'REBALANCE_FREQ',
    'REBALANCE_PERIODS_PER_YEAR',
    'RETURN_CAP',
    'RETURN_METHOD',
    'RISK_FREE',
    'ROLLING_WINDOW_YRS',
    'SHRINK_FACTOR',
    'TRADING_DAYS',
    'TRANSACTION_COST',
    'VaR_95',
    'WEIGHT_MAX',
    'WEIGHT_MIN',
    'WF_N_STARTS',
)

#: legacy loop variables that leak in from earlier stages; pulled only so the
#: split behaves exactly like the monolith's single shared scope
CARRYOVER = ()

#: data this stage produces that a later stage reads (downstream contract)
OUTPUTS = (
    'wf_port_returns',
    'wf_summary',
    'wf_turnover',
    'wf_window_stats',
)

#: everything bound at stage scope; all of it is published to ctx
_PUBLISHED = ('W', '_AMBER', '_BLUE', '_DARK', '_GREEN', '_GREY', '_PURPLE', '_RED', '_WF_METHOD_MAP', '_agg_metrics', '_ann_cost', '_ann_to_ms', '_ann_to_mv', '_apply_polish_floor', '_avg_to_ms', '_avg_to_mv', '_backtest_start', '_bm_aligned', '_bm_daily', '_bm_metrics', '_bm_ok', '_bm_simple', '_breach', '_bw', '_cells', '_completed', '_compute_cov_from_window', '_compute_mu_from_window', '_compute_turnover', '_concat_clean', '_cost_ew', '_cost_ms', '_cost_mv', '_cov', '_daily_ret', '_daily_var_empirical', '_data_end', '_data_start', '_drop_disp', '_drop_pct', '_end_dt', '_ew_daily', '_ew_metrics', '_ew_w', '_fwd_ret', '_i', '_ins_data', '_ins_port_ret', '_ins_start', '_is_sh', '_key', '_lbl', '_m', '_min_obs', '_ms_daily', '_ms_metrics', '_ms_raw', '_ms_w', '_mu', '_mv_daily', '_mv_metrics', '_mv_raw', '_mv_w', '_n_ms', '_n_mv', '_n_tickers', '_n_windows', '_next_dt', '_oos_sh', '_oos_sh_disp', '_optimize_minvar', '_optimize_sharpe', '_pct', '_period_sharpe', '_periods_per_yr', '_prev_ew_w', '_prev_ms_w', '_prev_mv_w', '_r', '_r6_ms', '_r6_mv', '_rd', '_rebal_dates', '_rebal_dt', '_rebal_trading', '_roll_w', '_rolling_sharpe', '_rows', '_safe_fmt', '_sh_ms_str', '_sh_mv_str', '_sh_str', '_start_dt', '_style_ax', '_to_dates', '_to_ew', '_to_ms', '_to_ms_v', '_to_mv', '_to_mv_v', '_trading_idx', '_var_daily_95', '_verdict', '_verdict2', '_w_sharpe_ms', '_w_sharpe_mv', '_wf_return_method', '_ws', '_ws_dates', '_ws_ms', '_ws_mv', '_x', '_xto', '_years', 'ax1', 'ax2', 'ax3', 'ax4', 'd', 'fig', 'gs', 't', 'v', 'w', 'wf_bm_wealth', 'wf_cost_drag', 'wf_ew_ret_dict', 'wf_ew_returns', 'wf_ew_wealth', 'wf_mv_ret_dict', 'wf_mv_returns', 'wf_mv_wealth', 'wf_mv_weights_hist', 'wf_port_ret_dict', 'wf_port_returns', 'wf_summary', 'wf_turnover', 'wf_wealth', 'wf_weights_hist', 'wf_window_stats', 'x')


def run(ctx: dict) -> None:
    # ---- inputs from upstream stages (generated) ----
    BENCHMARK = ctx['BENCHMARK']
    factor_index_ret = ctx['factor_index_ret']
    GLOBAL_SEED = ctx['GLOBAL_SEED']
    INITIAL_VALUE = ctx['INITIAL_VALUE']
    log_return = ctx['log_return']
    output_dir = ctx['output_dir']
    POLISH_FLOOR = ctx['POLISH_FLOOR']
    port_sharpe = ctx['port_sharpe']
    REBALANCE_FREQ = ctx['REBALANCE_FREQ']
    REBALANCE_PERIODS_PER_YEAR = ctx['REBALANCE_PERIODS_PER_YEAR']
    RETURN_CAP = ctx['RETURN_CAP']
    RETURN_METHOD = ctx['RETURN_METHOD']
    RISK_FREE = ctx['RISK_FREE']
    ROLLING_WINDOW_YRS = ctx['ROLLING_WINDOW_YRS']
    SHRINK_FACTOR = ctx['SHRINK_FACTOR']
    TRADING_DAYS = ctx['TRADING_DAYS']
    TRANSACTION_COST = ctx['TRANSACTION_COST']
    VaR_95 = ctx['VaR_95']
    WEIGHT_MAX = ctx['WEIGHT_MAX']
    WEIGHT_MIN = ctx['WEIGHT_MIN']
    WF_N_STARTS = ctx['WF_N_STARTS']
    # ---- original code: stardust_core.py lines 3610-4658 ----
    # ============================================================
    # MODULE 8: Walk-Forward Backtest  (CORRECTED — Institutional Grade)
    # ------------------------------------------------------------
    # PURPOSE
    #   Every number from Modules 3–6 is IN-SAMPLE: the optimizer
    #   was trained and tested on the same data. This always looks
    #   better than reality. Module 8 tests whether the strategy
    #   works on data the optimizer has NEVER seen — the only
    #   honest measure of real-world performance.
    #
    # HOW IT WORKS
    #   Rolling windows: train on 3 years → test on next quarter
    #   → roll forward → repeat. Each test period is genuinely
    #   out-of-sample — the optimizer only sees data up to the
    #   rebalance date, never beyond.
    #
    # WHAT IT MEASURES
    #   - CAGR, Sharpe, Max Drawdown, Calmar, Sortino, Hit Rate
    #   - Daily VaR breach rate vs Module 5 (properly scaled)
    #   - Turnover + transaction cost drag per rebalance
    #   - Per-window Sharpe: is performance consistent or lucky?
    #   - Comparison: Max Sharpe vs Min Var vs Equal Weight vs Nifty
    #
    # CORRECTIONS vs PREVIOUS VERSION
    #   [BUG-1] _optimize_sharpe: `bounds` kwarg used instead of
    #           `constraints` for bounds → SLSQP ignores bounds
    #           entirely. Weight constraints silently dropped.
    #           Fix: pass bnds to `bounds=` kwarg.
    #
    #   [BUG-2] Turnover history lookup uses list(dict.values())[-2]
    #           which is O(n) every window and breaks if any window
    #           was skipped (wf_weights_hist has fewer entries than
    #           _i). Fix: track previous weights in loop variables.
    #
    #   [BUG-3] VaR breach rate uses annual VaR / sqrt(252) as the
    #           daily threshold. sqrt-of-time scaling only holds for
    #           i.i.d. normal returns. For NSE equities (fat tails,
    #           autocorrelated volatility) this grossly understates
    #           the daily threshold → breach rate inflated.
    #           Fix: compute empirical daily VaR directly from the
    #           in-sample window (historical simulation, no
    #           distributional assumption).
    #
    #   [BUG-4] Equal-weight transaction cost applied as
    #           _to_ms * TRANSACTION_COST * 0.5 — the EW portfolio
    #           has its own turnover independent of Max Sharpe.
    #           Fix: compute _to_ew from actual EW weight changes.
    #
    #   [BUG-5] IS→OOS Sharpe drop formula can divide by zero or
    #           produce >100% drop when port_sharpe is near zero or
    #           OOS Sharpe is negative.
    #           Fix: guard with abs and cap at 999%.
    #
    #   [BUG-6] Per-window Sharpe printed even when NaN; f-string
    #           formatting on np.nan raises ValueError in some
    #           environments (depending on pandas version).
    #           Fix: use a safe formatter throughout.
    #
    #   [BUG-7] Rolling Sharpe computed with operator-precedence
    #           error: (mean*TD - RF) / (std*sqrt(TD)) computes
    #           mean*TD first, subtracts RF, then divides — but
    #           RF is annual while mean*TD is also annual, so the
    #           formula is correct in intent but written without
    #           parentheses, relying on Python precedence in a way
    #           that is fragile. Also: std=0 windows produce inf.
    #           Fix: explicit parentheses; clip inf → NaN.
    #
    #   [BUG-8] Panel 4 (turnover chart) promised in section
    #           comment but never drawn. GridSpec is 2×2 but only
    #           3 panels are filled — the bottom-right (gs[1,1])
    #           shows per-window Sharpe, and the turnover panel
    #           described in the docstring is missing.
    #           Fix: add turnover bar chart as Panel 4.
    #
    #   [BUG-9] Benchmark wealth path starts from first benchmark
    #           return day after index alignment, which may differ
    #           from backtest start — wealth comparison is off.
    #           Fix: rebase both to the same first date explicitly.
    #
    #   [LOGIC-1] _compute_mu_from_window converts log mean to
    #             arithmetic via exp(mu_log) - 1 but then clips the
    #             result. The clip should come AFTER shrinkage is
    #             applied — currently shrinkage can push a capped
    #             value slightly above the cap. Order fixed.
    #
    #   [LOGIC-2] CAGR formula uses (_end_dt - _start_dt).days /
    #             365.25 for both the full period and per-series.
    #             The benchmark series is aligned with dropna() so
    #             it covers fewer days — using the portfolio _years
    #             denominator for the benchmark CAGR is wrong.
    #             Fix: compute years from each series' own index.
    #
    #   [LOGIC-3] First-rebalance turnover is set to 1.0 (full
    #             construction). But if equal-weight is the prior
    #             (cash allocation), turnover should be 1.0 for all
    #             three strategies consistently. Currently EW gets
    #             0.5x MS cost — which is inconsistent.
    #
    #   [STYLE-1] Printing progress inside the loop uses the loop
    #             variable _i directly — if windows are skipped via
    #             `continue`, the count shown is wrong.
    #             Fix: track a separate _completed counter.
    #
    # Reads from Module 0:
    #   TRADING_DAYS, RISK_FREE, BENCHMARK, SHRINK_FACTOR,
    #   RETURN_CAP, WEIGHT_MAX, INITIAL_VALUE, GLOBAL_SEED
    #
    # Reads from Module 1:
    #   log_return, factor_index_ret
    #
    # Reads from Module 3:
    #   port_sharpe, WEIGHT_MIN, POLISH_FLOOR
    #
    # Reads from Module 5:
    #   VaR_95
    #
    # Produces:
    #   wf_port_returns    → Series : daily OOS returns (Max Sharpe)
    #   wf_mv_returns      → Series : daily OOS returns (Min Var)
    #   wf_ew_returns      → Series : daily OOS returns (Equal Wt)
    #   wf_wealth          → Series : cumulative wealth (Max Sharpe)
    #   wf_mv_wealth       → Series : cumulative wealth (Min Var)
    #   wf_ew_wealth       → Series : cumulative wealth (Equal Wt)
    #   wf_bm_wealth       → Series : cumulative wealth (Nifty 50)
    #   wf_weights_hist    → dict   : date → weight Series (MS)
    #   wf_mv_weights_hist → dict   : date → weight Series (MV)
    #   wf_turnover        → list   : (date, to_ms, to_mv, to_ew)
    #   wf_window_stats    → list   : per-window performance dicts
    #   wf_summary         → dict   : aggregate metrics for all 4
    # ============================================================

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from scipy.optimize import minimize
    import warnings
    warnings.filterwarnings('ignore')

    # -----------------------------------------------------------
    # 8A. Dependency check
    # -----------------------------------------------------------



    # -----------------------------------------------------------
    # 8B. Backtest configuration
    # -----------------------------------------------------------

    np.random.seed(GLOBAL_SEED)

    _data_start     = log_return.index[0]
    _data_end       = log_return.index[-1]
    _backtest_start = _data_start + pd.DateOffset(years=1)

    print(f"  Data range        : {_data_start.date()} → "
        f"{_data_end.date()}")
    print(f"  Backtest starts   : {_backtest_start.date()}")
    print(f"  Rolling window    : {ROLLING_WINDOW_YRS} years")
    print(f"  Rebalance freq    : {REBALANCE_FREQ}")
    print(f"  Optimizer starts  : {WF_N_STARTS} per window")
    print(f"  Transaction cost  : {TRANSACTION_COST*100:.2f}% one-way")
    print(f"  In-sample Sharpe  : {port_sharpe:.4f}  "
        f"← benchmark to beat\n")
    # ── Return method mapping for walk-forward ──────────────────
    # bl/factor/hybrid require live API calls per window — not
    # feasible across 8+ windows without significant runtime cost.
    # EWMA is used as the closest feasible approximation:
    #   - Uses only in-sample price data (already downloaded)
    #   - Recency-weighted like BL (recent data matters more)
    #   - No external API calls needed
    # equal → equal (exact match)
    # ewma  → ewma  (exact match)
    # bl / factor / hybrid → ewma (closest feasible proxy)

    _WF_METHOD_MAP = {
        'equal'  : 'equal',
        'ewma'   : 'ewma',
        'bl'     : 'ewma',
        'factor' : 'ewma',
        'hybrid' : 'ewma',
    }
    _wf_return_method = _WF_METHOD_MAP.get(RETURN_METHOD, 'equal')

    if RETURN_METHOD != _wf_return_method:
        print(f"  ⚠ Walk-forward return method:")
        print(f"    Main run used   : {RETURN_METHOD}")
        print(f"    Walk-forward    : {_wf_return_method} (EWMA proxy)")
        print(f"    Reason          : {RETURN_METHOD} requires live fundamental")
        print(f"                      data fetches not feasible per window.")
        print(f"    Impact          : OOS Sharpe reflects EWMA strategy.")
        print(f"                      IS→OOS gap is directionally valid")
        print(f"                      but not a perfect apples-to-apples test.")
        print(f"    Full fix        : Implement per-window BL (future work).\n")
    else:
        print(f"  ✓ Walk-forward return method matches main run: {RETURN_METHOD}\n")

    # -----------------------------------------------------------
    # 8C. Helper functions
    # -----------------------------------------------------------

    def _compute_mu_from_window(ins_data, shrink, return_cap,
                            trading_days, return_method='equal'):
            """
            Expected returns from in-sample window data only.

            Supports two methods:
            'ewma'  — exponentially weighted mean (halflife=252 days)
                        Used as proxy for bl/factor/hybrid since those
                        require live API calls not feasible per window.
            'equal' — equal-weighted historical mean (original)

            If RETURN_METHOD is bl/factor/hybrid, EWMA is used as the
            closest feasible per-window approximation. A warning is
            printed once before the loop (not per window).

            Order: method → shrinkage → cap (cap is the final gate).
            """
            if return_method == 'ewma':
                # EWMA with 1Y halflife — recency-weighted, no API calls needed
                ewma_log = (ins_data
                            .ewm(halflife=252)
                            .mean()
                            .iloc[-1] * trading_days)
                mu = np.exp(ewma_log) - 1
            else:
                # equal-weight historical mean
                log_mean_annual = ins_data.mean() * trading_days
                mu              = np.exp(log_mean_annual) - 1

            # James-Stein shrinkage toward cross-sectional mean
            grand_mean = mu.mean()
            mu         = (1.0 - shrink) * mu + shrink * grand_mean

            # Hard cap AFTER shrinkage
            mu = mu.clip(upper=return_cap)
            return mu


    def _compute_cov_from_window(ins_data, trading_days):
        """Annualised sample covariance from in-sample data."""
        return ins_data.cov() * trading_days


    def _optimize_sharpe(mu, cov, risk_free, weight_min,
                        weight_max, n_starts, seed):
        """
        Maximize Sharpe with multiple random restarts.

        FIX BUG-1: `bounds=` kwarg (not `constraints=`) carries
        the box constraints into SLSQP. The previous version
        passed bnds to `constraints=`, causing SLSQP to silently
        discard all weight bounds.
        """
        n    = len(mu)
        rng  = np.random.RandomState(seed)
        cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},)
        bnds = [(weight_min, weight_max)] * n   # box constraints

        def _neg_sharpe(w):
            ret = float(w @ mu.values)
            vol = float(np.sqrt(w @ cov.values @ w))
            return -(ret - risk_free) / vol if vol > 1e-8 else 9_999.0

        best_val = np.inf
        best_w   = np.ones(n) / n   # fallback: equal weight

        for _ in range(n_starts):
            w0 = rng.dirichlet(np.ones(n))
            w0 = np.clip(w0, weight_min, weight_max)
            w0 /= w0.sum()

            res = minimize(
                _neg_sharpe, w0,
                bounds=bnds,           # FIX: was `constraints=bnds`
                constraints=cons,
                method='SLSQP',
                options={'maxiter': 600, 'ftol': 1e-10}
            )
            if res.success and res.fun < best_val:
                best_val = res.fun
                best_w   = res.x

        return best_w


    def _optimize_minvar(cov, weight_min, weight_max,
                        n_starts, seed):
        """
        Minimize portfolio variance with multiple random restarts.

        Same BUG-1 fix as _optimize_sharpe: bounds passed to the
        `bounds=` kwarg, not `constraints=`.
        """
        n    = len(cov)
        rng  = np.random.RandomState(seed + 1)
        cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},)
        bnds = [(weight_min, weight_max)] * n

        def _port_var(w):
            return float(w @ cov.values @ w)

        best_val = np.inf
        best_w   = np.ones(n) / n

        for _ in range(n_starts):
            w0 = rng.dirichlet(np.ones(n))
            w0 = np.clip(w0, weight_min, weight_max)
            w0 /= w0.sum()

            res = minimize(
                _port_var, w0,
                bounds=bnds,           # FIX: was `constraints=bnds`
                constraints=cons,
                method='SLSQP',
                options={'maxiter': 600, 'ftol': 1e-10}
            )
            if res.success and res.fun < best_val:
                best_val = res.fun
                best_w   = res.x

        return best_w


    def _apply_polish_floor(w_array, tickers, polish_floor):
        """Zero sub-threshold weights and renormalise."""
        w = w_array.copy()
        w[w < polish_floor] = 0.0
        if w.sum() < 1e-8:
            top_idx = np.argsort(w_array)[-10:]
            w       = np.zeros_like(w_array)
            w[top_idx] = w_array[top_idx]
        w /= w.sum()
        return pd.Series(w, index=tickers)


    def _compute_turnover(new_w, prev_w):
        """
        One-way portfolio turnover = Σ|Δw| / 2.

        prev_w is reindexed to new_w to handle any universe
        differences (delistings, additions). Missing tickers
        in prev_w are treated as zero weight (new position).
        """
        aligned = prev_w.reindex(new_w.index).fillna(0.0)
        return float(np.abs(new_w.values - aligned.values).sum() / 2.0)


    def _period_sharpe(daily_ret, risk_free, trading_days):
        """Annualised Sharpe for a single return series."""
        if len(daily_ret) < 5:
            return np.nan
        ann_ret = daily_ret.mean() * trading_days
        ann_vol = daily_ret.std()  * np.sqrt(trading_days)
        return (ann_ret - risk_free) / ann_vol if ann_vol > 1e-8 else np.nan


    def _daily_var_empirical(ins_ret_series, confidence=0.95):
        """
        Empirical (historical simulation) daily VaR at given
        confidence level from in-sample returns.  No distributional
        assumption — uses actual quantile.

        FIX BUG-3: replaces the faulty annual VaR / sqrt(252)
        approximation which grossly understates the daily threshold
        for fat-tailed equity returns.
        """
        q = ins_ret_series.quantile(1.0 - confidence)
        return float(q)   # negative value (loss)


    def _safe_fmt(v, pct=False, rupee=False, width=13):
        """Format a scalar for the results table; NaN → dash."""
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return f"{'—':>{width}}"
        if rupee:
            return f"{'₹'+f'{v:,.0f}':>{width}}"
        if pct:
            return f"{v*100:>+{width-1}.2f}%"
        return f"{v:>{width}.4f}"

    # -----------------------------------------------------------
    # 8D. Generate rebalance dates
    # -----------------------------------------------------------

    _rebal_dates = pd.date_range(
        start=_backtest_start,
        end=_data_end,
        freq=REBALANCE_FREQ
    )

    _trading_idx  = log_return.index
    _rebal_trading = sorted(set(
        _trading_idx[_trading_idx.get_indexer([d], method='nearest')[0]]
        for d in _rebal_dates
    ))

    print(f"  Rebalance dates   : {len(_rebal_trading)} windows")
    print(f"  First rebalance   : {_rebal_trading[0].date()}")
    print(f"  Last rebalance    : {_rebal_trading[-1].date()}\n")

    # -----------------------------------------------------------
    # 8E. Storage containers
    # -----------------------------------------------------------

    wf_port_ret_dict    = {}
    wf_mv_ret_dict      = {}
    wf_ew_ret_dict      = {}
    wf_weights_hist     = {}
    wf_mv_weights_hist  = {}
    wf_turnover         = []   # (date, to_ms, to_mv, to_ew)
    wf_window_stats     = []
    wf_cost_drag        = []   # (date, cost_ms, cost_mv, cost_ew)

    # -----------------------------------------------------------
    # 8F. Main walk-forward loop
    # -----------------------------------------------------------

    print("  Starting walk-forward loop...")
    print(f"  {'─'*68}")

    _n_windows   = len(_rebal_trading)
    _daily_ret   = np.exp(log_return) - 1   # log → simple returns
    _completed   = 0                        # FIX STYLE-1: honest counter

    # Track previous weights explicitly — FIX BUG-2
    _prev_ms_w: pd.Series | None = None
    _prev_mv_w: pd.Series | None = None
    _prev_ew_w: pd.Series | None = None

    for _i, _rebal_dt in enumerate(_rebal_trading):

        # ── Out-of-sample (forward) window boundaries ──────────
        _next_dt = (_rebal_trading[_i + 1]
                    if _i < _n_windows - 1
                    else _data_end)

        # ── In-sample slice (strict: excludes rebalance date) ──
        _ins_start = _rebal_dt - pd.DateOffset(years=ROLLING_WINDOW_YRS)
        _ins_data  = log_return.loc[
            (log_return.index >= _ins_start) &
            (log_return.index <  _rebal_dt)
        ]

        # Skip if insufficient observations for stable covariance
        _min_obs = max(400, len(log_return.columns) * 10)
        if len(_ins_data) < _min_obs:
            print(f"  [SKIP] {_rebal_dt.date()} — "
                f"{len(_ins_data)} obs (need {_min_obs})")
            continue

        # ── Return and covariance estimates (in-sample only) ───
        _mu  = _compute_mu_from_window(
                _ins_data, SHRINK_FACTOR, RETURN_CAP, TRADING_DAYS,
                return_method=_wf_return_method)
        _cov = _compute_cov_from_window(_ins_data, TRADING_DAYS)

        assert list(_mu.index) == list(_cov.index), (
            f"mu/cov index misalignment at {_rebal_dt.date()}")

        # ── Optimise ────────────────────────────────────────────
        _ms_raw = _optimize_sharpe(
            _mu, _cov, RISK_FREE, WEIGHT_MIN, WEIGHT_MAX,
            WF_N_STARTS, GLOBAL_SEED)
        _ms_w = _apply_polish_floor(_ms_raw, _mu.index, POLISH_FLOOR)

        _mv_raw = _optimize_minvar(
            _cov, WEIGHT_MIN, WEIGHT_MAX, WF_N_STARTS, GLOBAL_SEED)
        _mv_w = _apply_polish_floor(_mv_raw, _mu.index, POLISH_FLOOR)

        _n_tickers = len(_mu)
        _ew_w = pd.Series(1.0 / _n_tickers, index=_mu.index)

        # ── Store weights ───────────────────────────────────────
        wf_weights_hist[_rebal_dt]    = _ms_w
        wf_mv_weights_hist[_rebal_dt] = _mv_w

        # ── Turnover  [FIX BUG-2, FIX LOGIC-3] ─────────────────
        # First window: full construction from cash → 1.0 for all.
        # Subsequent: actual Σ|Δw|/2 from explicit prev trackers.
        if _prev_ms_w is None:
            _to_ms, _to_mv, _to_ew = 1.0, 1.0, 1.0
        else:
            _to_ms = _compute_turnover(_ms_w, _prev_ms_w)
            _to_mv = _compute_turnover(_mv_w, _prev_mv_w)
            _to_ew = _compute_turnover(_ew_w, _prev_ew_w)

        _prev_ms_w = _ms_w
        _prev_mv_w = _mv_w
        _prev_ew_w = _ew_w

        _cost_ms = _to_ms * TRANSACTION_COST
        _cost_mv = _to_mv * TRANSACTION_COST
        _cost_ew = _to_ew * TRANSACTION_COST

        wf_turnover.append((_rebal_dt, _to_ms, _to_mv, _to_ew))
        wf_cost_drag.append((_rebal_dt, _cost_ms, _cost_mv, _cost_ew))

        # ── Forward (out-of-sample) returns ─────────────────────
        _fwd_ret = _daily_ret.loc[
            (_daily_ret.index >  _rebal_dt) &
            (_daily_ret.index <= _next_dt)
        ]
        if len(_fwd_ret) == 0:
            continue

        _ms_daily = (_fwd_ret
                    @ _ms_w.reindex(_fwd_ret.columns).fillna(0.0))
        _mv_daily = (_fwd_ret
                    @ _mv_w.reindex(_fwd_ret.columns).fillna(0.0))
        _ew_daily = (_fwd_ret
                    @ _ew_w.reindex(_fwd_ret.columns).fillna(0.0))

        # Apply transaction cost drag to first trading day
        _ms_daily.iloc[0] -= _cost_ms
        _mv_daily.iloc[0] -= _cost_mv
        _ew_daily.iloc[0] -= _cost_ew   # FIX BUG-4

        wf_port_ret_dict[_rebal_dt] = _ms_daily
        wf_mv_ret_dict[_rebal_dt]   = _mv_daily
        wf_ew_ret_dict[_rebal_dt]   = _ew_daily

        # ── Per-window metrics ──────────────────────────────────
        _w_sharpe_ms = _period_sharpe(_ms_daily, RISK_FREE, TRADING_DAYS)
        _w_sharpe_mv = _period_sharpe(_mv_daily, RISK_FREE, TRADING_DAYS)
        _n_ms = int((_ms_w > POLISH_FLOOR).sum())
        _n_mv = int((_mv_w > POLISH_FLOOR).sum())

        # Empirical daily VaR from in-sample distribution  [FIX BUG-3]
        _ins_port_ret = (_daily_ret.loc[_ins_data.index]
                        @ _ms_w.reindex(_daily_ret.columns).fillna(0.0))
        _var_daily_95 = _daily_var_empirical(_ins_port_ret, 0.95)

        wf_window_stats.append({
            'date'         : _rebal_dt,
            'n_days'       : len(_fwd_ret),
            'sharpe_ms'    : _w_sharpe_ms,
            'sharpe_mv'    : _w_sharpe_mv,
            'turnover_ms'  : _to_ms,
            'turnover_mv'  : _to_mv,
            'cost_ms'      : _cost_ms,
            'n_stocks_ms'  : _n_ms,
            'n_stocks_mv'  : _n_mv,
            'var_daily_95' : _var_daily_95,
        })

        _completed += 1   # FIX STYLE-1
        _sh_ms_str = (f"{_w_sharpe_ms:>+6.3f}"
                    if not np.isnan(_w_sharpe_ms) else "   NaN")
        _sh_mv_str = (f"{_w_sharpe_mv:>+6.3f}"
                    if not np.isnan(_w_sharpe_mv) else "   NaN")
        print(f"  [{_completed:>2}/{_n_windows}] "
            f"{_rebal_dt.date()} → {_next_dt.date()} "
            f"| {len(_fwd_ret):>3}d "
            f"| MS={_sh_ms_str} MV={_sh_mv_str} "
            f"| Stk {_n_ms}/{_n_mv} "
            f"| TO {_to_ms*100:.1f}% "
            f"| mu={_wf_return_method}")

    print(f"  {'─'*68}")
    print(f"  ✓ Walk-forward loop complete ({_completed} windows).\n")

    # Guard: nothing to report if all windows were skipped
    if not wf_port_ret_dict:
        raise RuntimeError(
            "MODULE 8: No windows completed. "
            "Check data range and ROLLING_WINDOW_YRS.")

    # -----------------------------------------------------------
    # 8G. Aggregate return series and wealth paths
    # -----------------------------------------------------------

    def _concat_clean(d: dict) -> pd.Series:
        """Concatenate per-window return dicts; remove duplicates."""
        s = pd.concat(list(d.values())).sort_index()
        return s[~s.index.duplicated(keep='first')]

    wf_port_returns = _concat_clean(wf_port_ret_dict)
    wf_mv_returns   = _concat_clean(wf_mv_ret_dict)
    wf_ew_returns   = _concat_clean(wf_ew_ret_dict)

    wf_wealth    = (1 + wf_port_returns).cumprod() * INITIAL_VALUE
    wf_mv_wealth = (1 + wf_mv_returns  ).cumprod() * INITIAL_VALUE
    wf_ew_wealth = (1 + wf_ew_returns  ).cumprod() * INITIAL_VALUE

    # -----------------------------------------------------------
    # 8H. Benchmark wealth path
    #
    #     FIX BUG-9: both series are rebased to start at
    #     INITIAL_VALUE on the same first date, so the visual
    #     and CAGR comparisons are apples-to-apples.
    # -----------------------------------------------------------

    _bm_daily = factor_index_ret.get(BENCHMARK, None)
    _bm_ok    = False

    if _bm_daily is not None:
        _bm_simple  = np.exp(_bm_daily) - 1
        _bm_aligned = (_bm_simple
                    .reindex(wf_port_returns.index)
                    .dropna())
        if len(_bm_aligned) > 5:
            wf_bm_wealth = (1 + _bm_aligned).cumprod() * INITIAL_VALUE
            _bm_ok = True
        else:
            wf_bm_wealth = None
            print("  ⚠  Benchmark has too few overlapping dates — skipped.")
    else:
        wf_bm_wealth = None
        print("  ⚠  Benchmark not available — skipping benchmark line.")

    # -----------------------------------------------------------
    # 8I. Aggregate performance metrics
    #
    #     FIX LOGIC-2: each series uses its own date range to
    #     compute annualised metrics — the benchmark may be
    #     shorter after alignment.
    #
    #     FIX BUG-3: daily VaR breach uses per-window empirical
    #     threshold stored in wf_window_stats, then aggregated.
    # -----------------------------------------------------------

    def _agg_metrics(ret_series: pd.Series, label: str) -> dict:
        """Compute institutional-grade aggregate OOS metrics."""
        if len(ret_series) < 20:
            return {}

        # Annualisation uses this series' own date span  [FIX LOGIC-2]
        _s   = ret_series.index[0]
        _e   = ret_series.index[-1]
        _yrs = (_e - _s).days / 365.25
        if _yrs < 1e-4:
            return {}

        _wealth   = (1 + ret_series).cumprod() * INITIAL_VALUE
        _final    = float(_wealth.iloc[-1])
        _cagr     = float((_final / INITIAL_VALUE) ** (1.0 / _yrs) - 1)
        _ann_vol  = float(ret_series.std()  * np.sqrt(TRADING_DAYS))
        _ann_ret  = float(ret_series.mean() * TRADING_DAYS)
        _sharpe   = ((_ann_ret - RISK_FREE) / _ann_vol
                    if _ann_vol > 1e-8 else np.nan)

        # Sortino — downside deviation only
        _neg      = ret_series[ret_series < (RISK_FREE / TRADING_DAYS)]
        _dside_vol = (float(_neg.std() * np.sqrt(TRADING_DAYS))
                    if len(_neg) > 2 else np.nan)
        _sortino  = ((_ann_ret - RISK_FREE) / _dside_vol
                    if (_dside_vol is not None and _dside_vol > 1e-8)
                    else np.nan)

        # Hit rate (fraction of days with positive return)
        _hit_rate = float((ret_series > 0).mean())

        # Max drawdown
        _cum      = (1 + ret_series).cumprod()
        _dd       = _cum / _cum.cummax() - 1
        _max_dd   = float(_dd.min())
        _calmar   = (_cagr / abs(_max_dd)
                    if abs(_max_dd) > 1e-8 else np.nan)

        # Longest drawdown duration (calendar days)
        _peak_idx = (_cum.cummax() != _cum.cummax().shift()).cumsum()
        _dd_dur   = _dd.groupby(_peak_idx).apply(
            lambda g: (g.index[-1] - g.index[0]).days
            if g.min() < 0 else 0
        )
        _max_dd_dur = int(_dd_dur.max()) if len(_dd_dur) > 0 else 0

        # VaR breach rate — use per-window empirical thresholds [FIX BUG-3]
        # For each day in the OOS stream, look up which window it belongs to
        # and use that window's in-sample daily VaR as the threshold.
        _breach_days = 0
        _total_days  = 0
        for _ws in wf_window_stats:
            _wd = _ws['date']
            _thresh = _ws['var_daily_95']   # already negative
            # Identify OOS days for this window
            _i_ws = wf_window_stats.index(_ws)
            _nd   = (_rebal_trading[_i_ws + 1]
                    if _i_ws < len(_rebal_trading) - 1
                    else _data_end) if _i_ws < len(wf_window_stats) else _data_end
            _slice = ret_series.loc[
                (ret_series.index > _wd) &
                (ret_series.index <= _nd)
            ]
            if len(_slice) > 0:
                _breach_days += int((_slice < _thresh).sum())
                _total_days  += len(_slice)
        _breach_rt = (_breach_days / _total_days
                    if _total_days > 0 else np.nan)

        return {
            'label'       : label,
            'final_value' : _final,
            'total_ret'   : _final / INITIAL_VALUE - 1,
            'cagr'        : _cagr,
            'ann_vol'     : _ann_vol,
            'ann_ret'     : _ann_ret,
            'sharpe'      : _sharpe,
            'sortino'     : _sortino,
            'hit_rate'    : _hit_rate,
            'max_dd'      : _max_dd,
            'max_dd_dur'  : _max_dd_dur,
            'calmar'      : _calmar,
            'var_breach'  : _breach_rt,
            'n_days'      : len(ret_series),
            'years'       : _yrs,
        }

    _ms_metrics = _agg_metrics(wf_port_returns, 'Max Sharpe (OOS)')
    _mv_metrics = _agg_metrics(wf_mv_returns,   'Min Variance (OOS)')
    _ew_metrics = _agg_metrics(wf_ew_returns,   'Equal Weight (OOS)')
    _bm_metrics = (_agg_metrics(_bm_aligned,    f'Nifty 50 (OOS)')
                if _bm_ok else {})

    wf_summary = {
        'max_sharpe' : _ms_metrics,
        'min_var'    : _mv_metrics,
        'equal_wt'   : _ew_metrics,
        'benchmark'  : _bm_metrics,
    }

    # -----------------------------------------------------------
    # 8J. Results table
    # -----------------------------------------------------------

    _start_dt = wf_port_returns.index[0]
    _end_dt   = wf_port_returns.index[-1]
    _years    = (_end_dt - _start_dt).days / 365.25

    print(f"{'='*76}")
    print(f"  WALK-FORWARD BACKTEST RESULTS")
    print(f"  Period  : {_start_dt.date()} → {_end_dt.date()} "
        f"({_years:.1f} years)")
    print(f"  mu used : {_wf_return_method}"
        + (f"  (proxy for {RETURN_METHOD})"
           if _wf_return_method != RETURN_METHOD else ""))
    print(f"{'='*76}")

    _m = [_ms_metrics, _mv_metrics, _ew_metrics, _bm_metrics]

    W = 14   # column width
    print(f"\n  {'Metric':<28}"
        + f"{'Max Sharpe':>{W}} {'Min Var':>{W}} "
            f"{'Equal Wt':>{W}} {'Nifty 50':>{W}}")
    print(f"  {'─'*72}")

    _rows = [
        ('Final Value',           'final_value', False, True),
        ('Total Return',          'total_ret',   True,  False),
        ('CAGR',                  'cagr',        True,  False),
        ('Annualised Vol',         'ann_vol',     True,  False),
        ('Annualised Return',      'ann_ret',     True,  False),
        ('Sharpe (OOS)',           'sharpe',      False, False),
        ('Sortino (OOS)',          'sortino',     False, False),
        ('Hit Rate',               'hit_rate',    True,  False),
        ('Max Drawdown',           'max_dd',      True,  False),
        ('Max DD Duration (days)', 'max_dd_dur',  False, False),
        ('Calmar Ratio',           'calmar',      False, False),
        ('VaR Breach Rate (95%)',  'var_breach',  True,  False),
    ]

    for _lbl, _key, _pct, _r in _rows:
        _cells = ''.join(
            _safe_fmt(mm.get(_key) if mm else None,
                    pct=_pct, rupee=_r, width=W)
            for mm in _m
        )
        print(f"  {_lbl:<28}{_cells}")

    # IS → OOS Sharpe gap  [FIX BUG-5: guard zero/negative]
    _oos_sh  = _ms_metrics.get('sharpe', np.nan)
    _is_sh   = port_sharpe
    if not np.isnan(_oos_sh) and abs(_is_sh) > 1e-4:
        _drop_pct = (1.0 - _oos_sh / _is_sh) * 100
        _drop_pct = max(-999.0, min(_drop_pct, 999.0))  # cap
        _verdict  = ('acceptable' if _drop_pct < 50
                    else 'HIGH — consider increasing SHRINK_FACTOR')
    else:
        _drop_pct = np.nan
        _verdict  = 'indeterminate'

    print(f"\n  {'In-sample Sharpe (M3)':<28}"
        f"{_safe_fmt(_is_sh, width=W)} {'(same model)':>{W}} "
        f"{'n/a':>{W}} {'n/a':>{W}}")
    print(f"\n  IS → OOS Sharpe drop : "
        f"{_drop_pct:>+.1f}%  ({_verdict})" if not np.isnan(_drop_pct)
        else f"\n  IS → OOS Sharpe drop : indeterminate")
    print(f"{'='*76}")

    # -----------------------------------------------------------
    # 8K. Per-window performance table
    # -----------------------------------------------------------

    print(f"\n  PER-WINDOW PERFORMANCE")
    print(f"  {'─'*76}")
    print(f"  {'Date':<14} {'Days':>5} {'MS Sharpe':>10} "
        f"{'MV Sharpe':>10} {'TO_MS':>7} {'TO_MV':>7} "
        f"{'Cost':>7} {'Stocks':>8}")
    print(f"  {'─'*76}")

    for _ws in wf_window_stats:
        def _sh_str(v):
            return f"{v:>+9.3f}" if not np.isnan(v) else f"{'—':>9}"
        print(f"  {str(_ws['date'].date()):<14} "
            f"{_ws['n_days']:>5} "
            f"{_sh_str(_ws['sharpe_ms']):>10} "
            f"{_sh_str(_ws['sharpe_mv']):>10} "
            f"{_ws['turnover_ms']*100:>6.1f}% "
            f"{_ws['turnover_mv']*100:>6.1f}% "
            f"{_ws['cost_ms']*100:>6.3f}% "
            f"{_ws['n_stocks_ms']:>3}/{_ws['n_stocks_mv']:<3}")

    # Turnover and cost summary
    _avg_to_ms = float(np.mean([x[1] for x in wf_turnover]))
    _avg_to_mv = float(np.mean([x[2] for x in wf_turnover]))

    _periods_per_yr = REBALANCE_PERIODS_PER_YEAR.get(REBALANCE_FREQ, 4)
    _ann_to_ms = _avg_to_ms * _periods_per_yr
    _ann_to_mv = _avg_to_mv * _periods_per_yr
    _ann_cost  = _ann_to_ms * TRANSACTION_COST

    print(f"\n  Avg turnover/rebalance : MS={_avg_to_ms*100:.1f}%  "
        f"MV={_avg_to_mv*100:.1f}%")
    print(f"  Annualised turnover    : MS={_ann_to_ms*100:.1f}%  "
        f"MV={_ann_to_mv*100:.1f}%")
    print(f"  Est. annual cost drag  : {_ann_cost*100:.2f}%  "
        f"(at {TRANSACTION_COST*100:.2f}% one-way)")
    print(f"{'─'*76}")

    # -----------------------------------------------------------
    # 8L. VaR calibration check
    # -----------------------------------------------------------

    _breach = _ms_metrics.get('var_breach', np.nan)
    if not np.isnan(_breach):
        print(f"\n  VaR Calibration (daily, empirical, 95% confidence):")
        print(f"  Module 5 VaR_95 (annual)  : "
            f"₹{VaR_95:,.0f}  "
            f"({VaR_95/INITIAL_VALUE*100:.1f}% of capital)")
        _verdict2 = (
            "Conservative — VaR overstates risk slightly" if _breach <= 0.03
            else "Well-calibrated ✓"                        if _breach <= 0.07
            else "Slightly aggressive — VaR may understate" if _breach <= 0.10
            else "Aggressive — VaR significantly understates risk")
        print(f"  OOS daily breach rate     : "
            f"{_breach*100:.2f}%  ({_verdict2})")
        print(f"  Expected breach rate      : 5.00%  (if VaR exact)")

    # -----------------------------------------------------------
    # 8M. Four-panel chart
    #
    #   Panel 1 (full top row) : Cumulative wealth — log scale
    #   Panel 2 (bottom-left)  : Rolling 6-month Sharpe
    #   Panel 3 (bottom-mid)   : Per-window Sharpe bars
    #   Panel 4 (bottom-right) : Turnover per rebalance  [FIX BUG-8]
    #
    #   FIX BUG-7: rolling Sharpe uses explicit parentheses and
    #   clamps ±inf to NaN to avoid distorted y-axis.
    # -----------------------------------------------------------

    _DARK   = '#0f0f1a'
    _BLUE   = '#4fc3f7'
    _GREEN  = '#81c784'
    _AMBER  = '#ffb74d'
    _RED    = '#ef5350'
    _GREY   = '#78909c'
    _PURPLE = '#ce93d8'

    def _style_ax(ax, title):
        ax.set_facecolor(_DARK)
        ax.set_title(title, color='white',
                    fontsize=9, fontweight='bold', pad=7)
        ax.tick_params(colors='white', labelsize=8)
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        for sp in ['top', 'right']:
            ax.spines[sp].set_visible(False)
        for sp in ['bottom', 'left']:
            ax.spines[sp].set_color('#444444')
        ax.grid(True, alpha=0.12, color='white', linewidth=0.5)

    fig = plt.figure(figsize=(18, 13))
    fig.patch.set_facecolor(_DARK)
    gs  = gridspec.GridSpec(2, 3, figure=fig,
                            hspace=0.38, wspace=0.30)

    # ── Panel 1: Wealth paths (full top row) ────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    _style_ax(ax1, 'Cumulative Wealth — Out-of-Sample (Log Scale)')

    ax1.semilogy(wf_wealth.index, wf_wealth.values,
                color=_BLUE, lw=1.6,
                label=f"Max Sharpe  "
                    f"SR={_ms_metrics.get('sharpe',np.nan):.3f}  "
                    f"CAGR={_ms_metrics.get('cagr',np.nan)*100:+.1f}%")
    ax1.semilogy(wf_mv_wealth.index, wf_mv_wealth.values,
                color=_GREEN, lw=1.4, ls='--',
                label=f"Min Variance  "
                    f"SR={_mv_metrics.get('sharpe',np.nan):.3f}  "
                    f"CAGR={_mv_metrics.get('cagr',np.nan)*100:+.1f}%")
    ax1.semilogy(wf_ew_wealth.index, wf_ew_wealth.values,
                color=_GREY, lw=1.0, ls=':',
                label=f"Equal Weight  "
                    f"SR={_ew_metrics.get('sharpe',np.nan):.3f}  "
                    f"CAGR={_ew_metrics.get('cagr',np.nan)*100:+.1f}%")
    if _bm_ok:
        ax1.semilogy(wf_bm_wealth.index, wf_bm_wealth.values,
                    color=_AMBER, lw=1.2, ls='-.',
                    label=f"Nifty 50  "
                        f"SR={_bm_metrics.get('sharpe',np.nan):.3f}  "
                        f"CAGR={_bm_metrics.get('cagr',np.nan)*100:+.1f}%")

    ax1.axhline(INITIAL_VALUE, color=_RED, lw=0.8,
                ls=':', alpha=0.6, label='Initial Capital')

    for _rd in _rebal_trading:
        ax1.axvline(_rd, color='white', alpha=0.06, lw=0.5)

    ax1.set_ylabel('Portfolio Value (₹)', color='white')
    ax1.legend(fontsize=8, framealpha=0.25,
            facecolor='#1a1a2e', edgecolor='#444444',
            labelcolor='white', loc='upper left')

    _oos_sh_disp = _ms_metrics.get('sharpe', np.nan)
    _drop_disp   = (_drop_pct if not np.isnan(_drop_pct) else 0)
    ax1.annotate(
        f"IS Sharpe (M3): {_is_sh:.3f}  →  "
        f"OOS Sharpe: {_oos_sh_disp:.3f}  "
        f"(drop: {_drop_disp:.0f}%)",
        xy=(0.01, 0.04), xycoords='axes fraction',
        fontsize=8, color=_AMBER,
        bbox=dict(boxstyle='round,pad=0.3',
                facecolor='#1a1a2e', alpha=0.7,
                edgecolor='#444444')
    )

    # ── Panel 2: Rolling 6M Sharpe  [FIX BUG-7] ────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    _style_ax(ax2, 'Rolling 6M Sharpe (OOS)')

    _roll_w = TRADING_DAYS // 2

    def _rolling_sharpe(ret, rf, td, w):
        """Rolling Sharpe with explicit parentheses and inf→NaN."""
        mu_  = ret.rolling(w).mean() * td
        sig_ = ret.rolling(w).std()  * np.sqrt(td)
        sr   = (mu_ - rf) / sig_
        return sr.replace([np.inf, -np.inf], np.nan)

    _r6_ms = _rolling_sharpe(wf_port_returns, RISK_FREE, TRADING_DAYS, _roll_w)
    _r6_mv = _rolling_sharpe(wf_mv_returns,   RISK_FREE, TRADING_DAYS, _roll_w)

    ax2.plot(_r6_ms.index, _r6_ms.values,
            color=_BLUE, lw=1.0, label='Max Sharpe')
    ax2.plot(_r6_mv.index, _r6_mv.values,
            color=_GREEN, lw=1.0, ls='--', label='Min Variance')
    ax2.axhline(0, color=_RED, lw=0.8, ls='--', alpha=0.8)
    ax2.fill_between(_r6_ms.index, _r6_ms.values, 0,
                    where=_r6_ms.values > 0,
                    color=_BLUE, alpha=0.07)
    ax2.fill_between(_r6_ms.index, _r6_ms.values, 0,
                    where=_r6_ms.values < 0,
                    color=_RED, alpha=0.07)
    ax2.set_ylabel('Sharpe Ratio', color='white')
    ax2.legend(fontsize=8, framealpha=0.25,
            facecolor='#1a1a2e', edgecolor='#444444',
            labelcolor='white')

    # ── Panel 3: Per-window Sharpe bars ─────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    _style_ax(ax3, 'Per-Window Sharpe: MS vs MV')

    _ws_dates  = [w['date']     for w in wf_window_stats]
    _ws_ms     = [w['sharpe_ms'] if not np.isnan(w['sharpe_ms']) else 0
                for w in wf_window_stats]
    _ws_mv     = [w['sharpe_mv'] if not np.isnan(w['sharpe_mv']) else 0
                for w in wf_window_stats]
    _x  = np.arange(len(_ws_dates))
    _bw = 0.35

    ax3.bar(_x - _bw/2, _ws_ms, _bw,
            color=[_BLUE if v >= 0 else _RED for v in _ws_ms],
            alpha=0.78, label='Max Sharpe')
    ax3.bar(_x + _bw/2, _ws_mv, _bw,
            color=[_GREEN if v >= 0 else _AMBER for v in _ws_mv],
            alpha=0.78, label='Min Variance')
    ax3.axhline(0, color='white', lw=0.6, alpha=0.5)
    ax3.set_xticks(_x)
    ax3.set_xticklabels(
        [str(d.date())[:7] for d in _ws_dates],
        rotation=45, ha='right', fontsize=7)
    ax3.set_ylabel('Sharpe Ratio', color='white')
    ax3.legend(fontsize=8, framealpha=0.25,
            facecolor='#1a1a2e', edgecolor='#444444',
            labelcolor='white')

    # ── Panel 4: Turnover per rebalance  [FIX BUG-8] ────────────
    ax4 = fig.add_subplot(gs[1, 2])
    _style_ax(ax4, 'Turnover per Rebalance (%)')

    _to_dates = [t[0] for t in wf_turnover]
    _to_ms_v  = [t[1] * 100 for t in wf_turnover]
    _to_mv_v  = [t[2] * 100 for t in wf_turnover]
    _xto = np.arange(len(_to_dates))

    ax4.bar(_xto - _bw/2, _to_ms_v, _bw,
            color=_BLUE, alpha=0.78, label='Max Sharpe')
    ax4.bar(_xto + _bw/2, _to_mv_v, _bw,
            color=_GREEN, alpha=0.78, label='Min Variance')
    ax4.axhline(60, color=_RED, lw=0.8, ls='--', alpha=0.6,
                label='60% threshold')
    ax4.set_xticks(_xto)
    ax4.set_xticklabels(
        [str(d.date())[:7] for d in _to_dates],
        rotation=45, ha='right', fontsize=7)
    ax4.set_ylabel('One-way Turnover (%)', color='white')
    ax4.legend(fontsize=8, framealpha=0.25,
            facecolor='#1a1a2e', edgecolor='#444444',
            labelcolor='white')

    plt.suptitle(
        f"Walk-Forward Backtest — {_completed} Windows  |  "
        f"{_years:.1f} Yrs OOS  |  "
        f"Rebalance: {REBALANCE_FREQ}  |  "
        f"Tx Cost: {TRANSACTION_COST*100:.2f}% one-way",
        color='white', fontsize=11, fontweight='bold', y=1.01
    )

    plt.savefig(str(output_dir / 'walkforward_backtest.png'), dpi=150,
                bbox_inches='tight', facecolor=_DARK)
    plt.show()
    print("  Chart saved → walkforward_backtest.png")

    # ============================================================
    # MODULE 8 OUTPUTS
    # ============================================================
    # wf_port_returns     → Series : daily OOS returns (Max Sharpe)
    # wf_mv_returns       → Series : daily OOS returns (Min Var)
    # wf_ew_returns       → Series : daily OOS returns (Equal Wt)
    # wf_wealth           → Series : cumulative wealth (Max Sharpe)
    # wf_mv_wealth        → Series : cumulative wealth (Min Var)
    # wf_ew_wealth        → Series : cumulative wealth (Equal Wt)
    # wf_bm_wealth        → Series : cumulative wealth (Nifty 50)
    # wf_weights_hist     → dict   : date → weight Series (MS)
    # wf_mv_weights_hist  → dict   : date → weight Series (MV)
    # wf_turnover         → list   : (date, to_ms, to_mv, to_ew)
    # wf_window_stats     → list   : per-window performance dicts
    # wf_summary          → dict   : aggregate metrics for all 4
    # ---- publish (generated) ----
    _scope = locals()
    ctx.update({k: _scope[k] for k in _PUBLISHED if k in _scope})

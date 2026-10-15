"""
Legacy Stardust 1.x - M3 - Portfolio optimizer (max Sharpe / min variance)

Mechanically extracted from stardust_core.py lines 1966-2297 by
legacy/tools/split_monolith.py. The code inside run() is the original,
byte-for-byte; only the input/publish scaffolding around it is generated.
Do not hand-edit the extracted block - fix bugs in STAR2.0 (star/) instead.
"""
from ._bootstrap import ensure_bootstrap

ensure_bootstrap()

#: data this stage reads from earlier stages (its upstream contract)
INPUTS = (
    'arith_returns',
    'cov_annual',
    'GLOBAL_SEED',
    'RISK_FREE',
    'status_callback',
)

#: legacy loop variables that leak in from earlier stages; pulled only so the
#: split behaves exactly like the monolith's single shared scope
CARRYOVER = ()

#: data this stage produces that a later stage reads (downstream contract)
OUTPUTS = (
    'minvar_return',
    'minvar_sharpe',
    'minvar_volatility',
    'minvar_weights',
    'N_STARTS',
    'n_stocks',
    'optimal_weights',
    'POLISH_FLOOR',
    'port_return',
    'port_sharpe',
    'port_volatility',
    'WEIGHT_MAX',
    'WEIGHT_MIN',
)

#: everything bound at stage scope; all of it is published to ctx
_PUBLISHED = ('N_STARTS', 'POLISH_FLOOR', 'WEIGHT_MAX', 'WEIGHT_MIN', 'bar', 'best_result', 'best_sharpe', 'best_var_result', 'best_var_val', 'bounds', 'comparison', 'constraints', 'diff', 'equal_weight', 'i', 'marker', 'minvar_return', 'minvar_sharpe', 'minvar_volatility', 'minvar_weights', 'n_stocks', 'negative_sharpe', 'optimal_weights', 'port_return', 'port_sharpe', 'port_volatility', 'portfolio_variance', 'res', 'row', 'ticker', 'w', 'w0', 'weights_sorted')


def run(ctx: dict) -> None:
    # ---- inputs from upstream stages (generated) ----
    arith_returns = ctx['arith_returns']
    cov_annual = ctx['cov_annual']
    GLOBAL_SEED = ctx['GLOBAL_SEED']
    RISK_FREE = ctx['RISK_FREE']
    status_callback = ctx['status_callback']
    # ---- original code: stardust_core.py lines 1966-2297 ----
    # ============================================================
    # MODULE 3: Portfolio Optimizer
    # ------------------------------------------------------------
    # Reads from Module 0: RISK_FREE, GLOBAL_SEED
    # Reads from Module 1: clean_tickers
    # Reads from Module 2: arith_returns, cov_annual, cov_daily
    #
    # Corrections applied:
    #   - Issue 9  : Inputs correctly aligned — arith_returns
    #                never sorted so cov_annual and arith_returns
    #                share exact same index order
    #   - Issue 10 : Multiple random starts — N_STARTS scales
    #                with stock count to avoid local optima
    #   - Issue 11 : Long-only explicitly documented
    #   - Issue 12 : Min Variance also uses multiple starts
    #   - Dynamic  : Weight bounds computed from stock count —
    #                works correctly for 3 to 100 stocks
    # ============================================================

    import numpy as np
    import pandas as pd
    from scipy.optimize import minimize

    status_callback("Module 3: Running optimisation...")

    # -----------------------------------------------------------
    # 3A. Dynamic weight bounds
    #     Allows for allocation of 0 to underperforming stocks
    # -----------------------------------------------------------

    n_stocks     = len(arith_returns)
    equal_weight = 1.0 / n_stocks

    WEIGHT_MIN = 0.00   # allow zero — optimizer decides who gets excluded
    WEIGHT_MAX = min(0.40, equal_weight * 4)
    POLISH_FLOOR = 0.01  # 1% floor applied only to final selected stocks

    # Hard feasibility assertions — these should never fail
    # if Module 0 universe validation passed
    assert n_stocks * WEIGHT_MIN <= 1.0, (
        f"Floor infeasible: {n_stocks} × {WEIGHT_MIN*100:.2f}% = "
        f"{n_stocks*WEIGHT_MIN*100:.1f}% > 100%. "
        f"Reduce stock count or lower WEIGHT_MIN."
    )
    assert WEIGHT_MIN < WEIGHT_MAX, (
        f"Floor ({WEIGHT_MIN*100:.2f}%) ≥ cap ({WEIGHT_MAX*100:.2f}%). "
        f"Logic error — check stock count."
    )

    print(f"Dynamic weight bounds for {n_stocks} stocks:")
    print(f"  Equal weight per stock : {equal_weight*100:.2f}%")
    print(f"  Floor  (min per stock) : {WEIGHT_MIN*100:.2f}%")
    print(f"  Cap    (max per stock) : {WEIGHT_MAX*100:.2f}%")
    print(f"  Feasibility check      : {n_stocks} × {WEIGHT_MIN*100:.2f}% "
        f"= {n_stocks*WEIGHT_MIN*100:.1f}% ✓")

    bounds      = [(WEIGHT_MIN, WEIGHT_MAX)] * n_stocks
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}

    # -----------------------------------------------------------
    # 3B. N_STARTS scales with stock count
    #
    #     WHY MULTIPLE STARTS?
    #     SLSQP is a local optimizer — it finds the nearest
    #     local maximum from the starting point. With a single
    #     equal-weight start you have no guarantee of finding
    #     the global maximum. With 24 stocks and complex bounds,
    #     multiple local optima exist — especially when some
    #     stocks have similar risk/return profiles.
    #
    #     More stocks = more complex landscape = more starts needed.
    #     Above 50 stocks each run is slower so we reduce starts
    #     to keep total runtime reasonable (~2-3 seconds).
    # -----------------------------------------------------------

    if n_stocks <= 15:
        N_STARTS = 30
    elif n_stocks <= 50:
        N_STARTS = 50
    else:
        N_STARTS = 20

    print(f"  Optimizer starts       : {N_STARTS}")

    # -----------------------------------------------------------
    # 3C. Alignment verification
    #
    #     Hard stop before optimization. If arith_returns was
    #     accidentally sorted anywhere in Module 2, this catches
    #     it before producing wrong weights silently.
    # -----------------------------------------------------------

    assert list(arith_returns.index) == list(cov_annual.index), (
        "ALIGNMENT ERROR: arith_returns and cov_annual indices differ.\n"
        "arith_returns must never be sorted or reindexed.\n"
        "Rerun Module 2."
    )
    print("\nAlignment check passed — safe to optimize.")

    # -----------------------------------------------------------
    # 3D. Objective function — negative Sharpe ratio
    #
    #     scipy.minimize always minimizes.
    #     Minimizing negative Sharpe = maximizing Sharpe.
    #
    #     Uses arithmetic returns — what investors actually earn.
    #     Log mean underestimates return by ~0.5 × variance.
    #     For a 30% vol stock this is ~4.5% per year — enough
    #     to meaningfully change optimizer weights.
    #
    #     Port vol guard (< 1e-10): prevents division by zero
    #     if optimizer tries a degenerate weight vector during
    #     early random starts.
    # -----------------------------------------------------------

    def negative_sharpe(weights, arith_ret, cov_ann, rf):
        port_return = np.sum(arith_ret.values * weights)
        port_vol    = np.sqrt(weights.T @ cov_ann.values @ weights)
        if port_vol < 1e-10:
            return 0.0
        return -(port_return - rf) / port_vol

    # -----------------------------------------------------------
    # 3E. Max Sharpe optimizer — multiple random starts
    # -----------------------------------------------------------

    best_sharpe = -np.inf
    best_result = None
    np.random.seed(GLOBAL_SEED)

    print(f"\nMax Sharpe optimizer — {N_STARTS} random starts...")

    for i in range(N_STARTS):
        # Random Dirichlet start, clipped and renormalized to bounds
        w0  = np.random.dirichlet(np.ones(n_stocks))
        w0  = np.clip(w0, WEIGHT_MIN, WEIGHT_MAX)
        w0 /= w0.sum()

        res = minimize(
            negative_sharpe,
            w0,
            args        = (arith_returns, cov_annual, RISK_FREE),
            method      = 'SLSQP',
            bounds      = bounds,
            constraints = constraints,
            options     = {'maxiter': 1000, 'ftol': 1e-12}
        )

        if res.success and (-res.fun) > best_sharpe:
            best_sharpe = -res.fun
            best_result = res

    if best_result is None:
        print("WARNING: No optimizer run converged. Using equal weights.")
        optimal_weights = pd.Series(1/n_stocks, index=arith_returns.index)
    else:
        print(f"Best Sharpe across {N_STARTS} starts: {best_sharpe:.4f}")
        optimal_weights = pd.Series(best_result.x, index=arith_returns.index)

    # -----------------------------------------------------------
    # 3F. Portfolio metrics — Max Sharpe
    # -----------------------------------------------------------

    port_return     = np.sum(arith_returns.values * optimal_weights.values)
    port_volatility = np.sqrt(
        optimal_weights.values.T @ cov_annual.values @ optimal_weights.values
    )
    port_sharpe     = (port_return - RISK_FREE) / port_volatility

    # Display only — sort a copy, NEVER reassign optimal_weights
    weights_sorted = optimal_weights.sort_values(ascending=False)

    print("\n" + "=" * 48)
    print(f"{'OPTIMAL PORTFOLIO — MAX SHARPE':^48}")
    print("=" * 48)
    for ticker, w in weights_sorted.items():
        bar = "█" * int(w * 100)
        print(f"  {ticker.replace('.NS',''):12s}  {w*100:5.2f}%  {bar}")
    print("=" * 48)
    print(f"  {'Expected Return':<24}: {port_return*100:+.2f}%")
    print(f"  {'Annual Volatility':<24}: {port_volatility*100:.2f}%")
    print(f"  {'Sharpe Ratio':<24}: {port_sharpe:.4f}")
    print(f"  {'Risk-Free Rate':<24}: {RISK_FREE*100:.1f}%")
    print(f"  {'Weight floor':<24}: {WEIGHT_MIN*100:.2f}%")
    print(f"  {'Weight cap':<24}: {WEIGHT_MAX*100:.2f}%")
    print("=" * 48)

    # -----------------------------------------------------------
    # 3G. Min Variance optimizer — multiple starts
    #
    #     FIX Issue 12 — same multiple-start approach as Max Sharpe.
    #
    #     WHY MIN VARIANCE?
    #     Max Sharpe depends on return estimates which are noisy.
    #     Small changes in expected returns can swing weights
    #     dramatically. Min Variance only uses the covariance
    #     matrix — which is far more stable and predictable.
    #     Research shows Min Var portfolios often outperform
    #     Max Sharpe out-of-sample because they avoid the
    #     noisy return estimation problem entirely.
    #
    #     Uses same dynamic bounds as Max Sharpe — consistent.
    # -----------------------------------------------------------

    def portfolio_variance(weights, cov_ann):
        return weights.T @ cov_ann.values @ weights

    best_var_val    = np.inf
    best_var_result = None

    print(f"\nMin Variance optimizer — {N_STARTS} random starts...")

    for i in range(N_STARTS):
        w0  = np.random.dirichlet(np.ones(n_stocks))
        w0  = np.clip(w0, WEIGHT_MIN, WEIGHT_MAX)
        w0 /= w0.sum()

        res = minimize(
            portfolio_variance,
            w0,
            args        = (cov_annual,),
            method      = 'SLSQP',
            bounds      = bounds,
            constraints = constraints,
            options     = {'maxiter': 1000, 'ftol': 1e-12}
        )

        if res.success and res.fun < best_var_val:
            best_var_val    = res.fun
            best_var_result = res

    if best_var_result is None:
        print("WARNING: Min-Var optimizer did not converge.")
        minvar_weights = pd.Series(1/n_stocks, index=arith_returns.index)
    else:
        minvar_weights = pd.Series(best_var_result.x, index=arith_returns.index)

    minvar_return     = np.sum(arith_returns.values * minvar_weights.values)
    minvar_volatility = np.sqrt(
        minvar_weights.values.T @ cov_annual.values @ minvar_weights.values
    )
    minvar_sharpe     = (minvar_return - RISK_FREE) / minvar_volatility

    print(f"Min-Var best portfolio variance: {best_var_val:.6f}")

    print("\n" + "=" * 48)
    print(f"{'OPTIMAL PORTFOLIO — MIN VARIANCE':^48}")
    print("=" * 48)
    for ticker, w in minvar_weights.sort_values(ascending=False).items():
        bar = "█" * int(w * 100)
        print(f"  {ticker.replace('.NS',''):12s}  {w*100:5.2f}%  {bar}")
    print("=" * 48)
    print(f"  {'Expected Return':<24}: {minvar_return*100:+.2f}%")
    print(f"  {'Annual Volatility':<24}: {minvar_volatility*100:.2f}%")
    print(f"  {'Sharpe Ratio':<24}: {minvar_sharpe:.4f}")
    print("=" * 48)

    # -----------------------------------------------------------
    # 3H. Side-by-side comparison
    #
    #     ▲ = Max Sharpe weights significantly MORE
    #         → return estimate is driving this allocation
    #         → treat with caution — historical return may not repeat
    #     ▼ = Min Var weights significantly MORE
    #         → stock genuinely reduces portfolio risk
    #         → higher conviction hold regardless of return estimate
    #     ≈ = Both models agree (difference < 1%)
    #         → highest conviction — robust to estimation error
    # -----------------------------------------------------------

    comparison = pd.DataFrame({
        'Max Sharpe %' : (optimal_weights * 100).round(2),
        'Min Var %'    : (minvar_weights * 100).round(2),
        'Difference %' : ((optimal_weights - minvar_weights) * 100).round(2)
    }, index=arith_returns.index).sort_values('Max Sharpe %', ascending=False)

    print("\n" + "=" * 62)
    print(f"{'PORTFOLIO COMPARISON':^62}")
    print("=" * 62)
    print(f"  {'Metric':<25} {'Max Sharpe':>15} {'Min Variance':>15}")
    print("-" * 62)
    print(f"  {'Expected Return':<25} {port_return*100:>14.2f}% "
        f"{minvar_return*100:>14.2f}%")
    print(f"  {'Annual Volatility':<25} {port_volatility*100:>14.2f}% "
        f"{minvar_volatility*100:>14.2f}%")
    print(f"  {'Sharpe Ratio':<25} {port_sharpe:>15.4f} "
        f"{minvar_sharpe:>15.4f}")
    print("=" * 62)
    print(f"\n  {'Ticker':<14} {'Max Sharpe':>10} {'Min Var':>10} {'Diff':>10}")
    print("-" * 62)
    for ticker, row in comparison.iterrows():
        diff   = row['Difference %']
        marker = '▲' if diff > 1 else ('▼' if diff < -1 else '≈')
        print(f"  {ticker.replace('.NS',''):<14} "
            f"{row['Max Sharpe %']:>9.2f}% "
            f"{row['Min Var %']:>9.2f}% "
            f"{marker}{abs(diff):>8.2f}%")
    print("=" * 62)
    print("\n  ▲ return-driven  ▼ risk-driven  ≈ high conviction")

    # -----------------------------------------------------------
    # 3I. Final alignment verification before passing downstream
    # -----------------------------------------------------------

    assert list(optimal_weights.index) == list(arith_returns.index), \
        "optimal_weights misaligned with arith_returns."
    assert list(minvar_weights.index) == list(arith_returns.index), \
        "minvar_weights misaligned with arith_returns."

    print(f"\n  ✓ optimal_weights aligned with arith_returns")
    print(f"  ✓ minvar_weights  aligned with arith_returns")
    print(f"  ✓ Safe to proceed to Module 4.")

    # -----------------------------------------------------------
    # MODULE 3 OUTPUTS — passed into Modules 4, 5, 6
    # -----------------------------------------------------------
    # optimal_weights   → pd.Series : Max Sharpe weights
    # minvar_weights    → pd.Series : Min Variance weights
    # port_return       → float     : Max Sharpe annualized return
    # port_volatility   → float     : Max Sharpe annualized vol
    # port_sharpe       → float     : Max Sharpe Sharpe ratio
    # minvar_return     → float     : Min Var annualized return
    # minvar_volatility → float     : Min Var annualized vol
    # minvar_sharpe     → float     : Min Var Sharpe ratio
    # WEIGHT_MIN        → float     : dynamic floor (used in M4)
    # WEIGHT_MAX        → float     : dynamic cap  (used in M4)
    # N_STARTS          → int       : starts used  (documented)
    #
    # All weight series share exact same index order as
    # arith_returns, cov_daily, and cov_annual.
    # -----------------------------------------------------------

    # ---- publish (generated) ----
    _scope = locals()
    ctx.update({k: _scope[k] for k in _PUBLISHED if k in _scope})

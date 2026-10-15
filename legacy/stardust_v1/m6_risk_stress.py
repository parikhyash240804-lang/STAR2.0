"""
Legacy Stardust 1.x - M6 - Risk metrics & stress testing

Mechanically extracted from stardust_core.py lines 3119-3609 by
legacy/tools/split_monolith.py. The code inside run() is the original,
byte-for-byte; only the input/publish scaffolding around it is generated.
Do not hand-edit the extracted block - fix bugs in STAR2.0 (star/) instead.
"""
from ._bootstrap import ensure_bootstrap

ensure_bootstrap()

#: data this stage reads from earlier stages (its upstream contract)
INPUTS = (
    'BENCHMARK',
    'close_df',
    'final_values',
    'INITIAL_VALUE',
    'log_return',
    'N_SIMULATIONS',
    'optimal_weights',
    'output_dir',
    'port_return',
    'port_value_paths',
    'prob_loss',
    'status_callback',
    'STRESS_SCENARIOS',
    'TRADING_DAYS',
    'VaR_95',
)

#: legacy loop variables that leak in from earlier stages; pulled only so the
#: split behaves exactly like the monolith's single shared scope
CARRYOVER = ()

#: data this stage produces that a later stage reads (downstream contract)
OUTPUTS = (
    'all_drawdowns',
    'betas',
    'CVaR_hist_1day',
    'CVaR_hist_annual',
    'scenario_results',
    'VaR_hist_1day',
    'VaR_hist_annual',
)

#: everything bound at stage scope; all of it is published to ctx
_PUBLISHED = ('CVaR_hist_1day', 'CVaR_hist_annual', 'CVaR_hist_pct', 'VaR_hist_1day', 'VaR_hist_annual', 'VaR_hist_pct', '_', 'actual_ret', 'all_drawdowns', 'annual_pct_5', 'ax', 'ax1', 'ax2', 'ax3', 'ax4', 'b', 'bar', 'bars', 'beta_colors', 'betas', 'colors', 'common', 'crash_periods', 'dd_pct5', 'dd_pct95', 'diff_pct', 'e', 'end_b', 'end_p', 'fig', 'gs', 'hist_port_returns', 'hist_simple_returns', 'i', 'l', 'loss_r', 'loss_rupees', 'losses', 'lr_aligned', 'market_drop', 'mask_period', 'max_drawdown', 'median_dd', 'method', 'method_', 'n', 'name', 'names', 'nf_aligned', 'nifty_b', 'nifty_ret', 'period', 'period_prices', 'portfolio_loss', 'portfolio_value', 'scenario_results', 'shock_arr', 'slope', 'sorted_betas', 'start_b', 'start_p', 'stock_shocks', 'stress_beta', 't', 'ticker', 'worst_dd')


def run(ctx: dict) -> None:
    # ---- inputs from upstream stages (generated) ----
    BENCHMARK = ctx['BENCHMARK']
    close_df = ctx['close_df']
    final_values = ctx['final_values']
    INITIAL_VALUE = ctx['INITIAL_VALUE']
    log_return = ctx['log_return']
    N_SIMULATIONS = ctx['N_SIMULATIONS']
    optimal_weights = ctx['optimal_weights']
    output_dir = ctx['output_dir']
    port_return = ctx['port_return']
    port_value_paths = ctx['port_value_paths']
    prob_loss = ctx['prob_loss']
    status_callback = ctx['status_callback']
    STRESS_SCENARIOS = ctx['STRESS_SCENARIOS']
    TRADING_DAYS = ctx['TRADING_DAYS']
    VaR_95 = ctx['VaR_95']
    # ---- original code: stardust_core.py lines 3119-3609 ----
    # ============================================================
    # MODULE 6: Risk & Stress Testing
    # ------------------------------------------------------------
    # Reads from Module 0: INITIAL_VALUE, RISK_FREE,
    #                      STRESS_SCENARIOS, BENCHMARK
    # Reads from Module 1: log_return, close_df
    # Reads from Module 2: arith_returns, cov_daily
    # Reads from Module 3: optimal_weights, port_return,
    #                      port_volatility, n_stocks
    # Reads from Module 5: port_value_paths, final_values,
    #                      VaR_95, CVaR_95, prob_loss,
    #                      N_SIMULATIONS
    #
    # Corrections applied:
    #   - Issue 19 : Annual VaR from simulation paths directly
    #                — removed sqrt-of-time scaling which only
    #                holds for normal distributions
    #   - Issue 20 : In-sample bias explicitly documented
    #   - Issue 21 : Stress scenarios use actual historical
    #                crash returns where available, stressed
    #                beta (×1.3) otherwise — not linear beta
    #   - Issue 22 : Capital gains tax from actual prices
    #                not 10% flat assumption
    # ============================================================

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from scipy import stats
    import yfinance as yf
    import datetime as dt

    status_callback("Module 6: Stress testing...")

    # -----------------------------------------------------------
    # 6A. Historical Simulation VaR & CVaR
    #
    #     WHY HISTORICAL SIMULATION?
    #     Module 5 uses a parametric distribution (normal or t)
    #     to generate shocks. Even with fat tails from t-dist,
    #     the shape is symmetric and smooth — it cannot fully
    #     capture the actual distribution of real equity returns
    #     which have asymmetric crashes, volatility clustering,
    #     and regime changes.
    #
    #     Historical simulation uses ACTUAL past daily returns
    #     with zero distributional assumptions. Every real event
    #     — COVID crash, 2022 rate shock, every bad day in your
    #     5-year window — is used directly. Fat tails are real
    #     because the data itself is fat-tailed.
    #
    #     FIX Issue 20 — In-sample bias documented:
    #     These weights were optimized on the SAME data being
    #     tested. The optimizer already avoided the worst days
    #     in this dataset by construction. True out-of-sample
    #     historical VaR would be higher — treat this as an
    #     optimistic lower bound, not the true risk estimate.
    #
    #     FIX Issue 19 — Annual VaR:
    #     OLD: VaR_annual = VaR_1day × sqrt(252)
    #     PROBLEM: sqrt-of-time rule only holds for independent
    #     normally distributed returns. Real returns have serial
    #     correlation and fat tails — annual VaR is larger than
    #     this formula suggests.
    #     FIX: Use Module 5 simulation final_values directly.
    #     The simulation already ran 252 days. No scaling needed.
    # -----------------------------------------------------------

    print("=" * 58)
    print(f"{'COMPONENT 1 — HISTORICAL VAR & CVAR':^58}")
    print("=" * 58)

    # Portfolio daily log return using optimal weights
    hist_port_returns   = log_return @ optimal_weights.values
    hist_simple_returns = np.exp(hist_port_returns) - 1

    # 1-day VaR and CVaR from historical distribution
    VaR_hist_pct  = np.percentile(hist_simple_returns, 5)
    CVaR_hist_pct = hist_simple_returns[
                        hist_simple_returns <= VaR_hist_pct].mean()

    VaR_hist_1day  = abs(VaR_hist_pct)  * INITIAL_VALUE
    CVaR_hist_1day = abs(CVaR_hist_pct) * INITIAL_VALUE

    # Annual VaR directly from Module 5 simulation — no scaling
    annual_pct_5     = np.percentile(final_values, 5)
    VaR_hist_annual  = INITIAL_VALUE - annual_pct_5
    CVaR_hist_annual = INITIAL_VALUE - np.mean(
        final_values[final_values <= annual_pct_5])

    print(f"\n  Based on {len(hist_simple_returns):,} actual trading days")
    print(f"\n  ⚠ IN-SAMPLE NOTE: weights optimized on same data.")
    print(f"  Historical VaR is optimistic — true risk is higher.")
    print(f"  Use as lower bound, not the definitive risk estimate.")

    print(f"\n  {'Metric':<35} {'1-Day':>10} {'Annual (sim)':>14}")
    print(f"  {'-'*61}")
    print(f"  {'VaR 95% — rupee loss':<35} "
        f"{'₹'+f'{VaR_hist_1day:,.0f}':>10} "
        f"{'₹'+f'{VaR_hist_annual:,.0f}':>14}")
    print(f"  {'CVaR 95% — expected loss':<35} "
        f"{'₹'+f'{CVaR_hist_1day:,.0f}':>10} "
        f"{'₹'+f'{CVaR_hist_annual:,.0f}':>14}")

    # Compare Module 5 vs Module 6 VaR
    print(f"\n  Comparison — 1-day VaR:")
    print(f"  Module 5 (parametric, {N_SIMULATIONS:,} paths) : "
        f"₹{VaR_95/TRADING_DAYS**0.5:,.0f}  [scaled from annual]")
    print(f"  Module 6 (historical, actual returns) : "
        f"₹{VaR_hist_1day:,.0f}")

    diff_pct = ((VaR_hist_1day - VaR_95/TRADING_DAYS**0.5) /
                (VaR_95/TRADING_DAYS**0.5)) * 100
    if diff_pct > 0:
        print(f"  Historical VaR is {diff_pct:+.1f}% higher "
            f"→ fat tails confirmed in actual data.")
    else:
        print(f"  Historical VaR is {diff_pct:+.1f}% lower "
            f"→ t-distribution capturing tails adequately.")

    # -----------------------------------------------------------
    # 6B. Maximum Drawdown
    #
    #     Maximum drawdown = largest peak-to-trough drop
    #     across the entire simulation period.
    #     More intuitive than volatility for real investors:
    #     "what is the worst I could have seen on my screen
    #     at any point during the year?"
    #
    #     Computed across ALL simulation paths from Module 5
    #     so the distribution reflects full range of outcomes.
    # -----------------------------------------------------------

    print(f"\n{'='*58}")
    print(f"{'COMPONENT 2 — MAXIMUM DRAWDOWN':^58}")
    print(f"{'='*58}")

    def max_drawdown(path):
        """Largest peak-to-trough percentage drop in a path."""
        peak     = np.maximum.accumulate(path)
        drawdown = (path - peak) / peak
        return drawdown.min()   # most negative value

    all_drawdowns = np.array([
        max_drawdown(port_value_paths[:, i])
        for i in range(port_value_paths.shape[1])
    ])

    median_dd = np.median(all_drawdowns) * 100
    worst_dd  = all_drawdowns.min() * 100
    dd_pct5   = np.percentile(all_drawdowns, 5) * 100
    dd_pct95  = np.percentile(all_drawdowns, 95) * 100

    print(f"\n  Across {port_value_paths.shape[1]:,} simulation paths:")
    print(f"\n  {'Metric':<35} {'Value':>18}")
    print(f"  {'-'*53}")
    print(f"  {'Median max drawdown':<35} {median_dd:>17.2f}%")
    print(f"  {'Worst 5% max drawdown':<35} {dd_pct5:>17.2f}%")
    print(f"  {'Best 5% max drawdown':<35} {dd_pct95:>17.2f}%")
    print(f"  {'Absolute worst drawdown':<35} {worst_dd:>17.2f}%")
    print(f"\n  In rupees (₹{INITIAL_VALUE:,.0f} portfolio):")
    print(f"  Median worst drop : "
        f"₹{abs(median_dd/100 * INITIAL_VALUE):,.0f}")
    print(f"  Worst 5% drop     : "
        f"₹{abs(dd_pct5/100 * INITIAL_VALUE):,.0f}")

    # -----------------------------------------------------------
    # 6C. Beta calculation vs benchmark
    #
    #     Beta required for two purposes:
    #     1. Stress scenario fallback when historical crash
    #        data is not in our 5-year window
    #     2. Beta bar chart in the plot
    #
    #     Uses stressed beta (×1.3) for scenarios rather than
    #     raw beta because in real crashes betas increase —
    #     stocks that normally have beta=0.8 behave like
    #     beta=1.2 during a crisis. Linear beta understates
    #     stress losses.
    # -----------------------------------------------------------

    print(f"\n{'='*58}")
    print(f"{'COMPONENT 3 — STRESS SCENARIOS':^58}")
    print(f"{'='*58}")

    try:
        end_b    = dt.datetime.now()
        start_b  = end_b - dt.timedelta(days=5 * 365)
        nifty_b  = yf.download(BENCHMARK, start=start_b, end=end_b,
                                auto_adjust=True, progress=False)
        nifty_ret = np.log(
            nifty_b['Close'] / nifty_b['Close'].shift(1)).dropna()

        common     = log_return.index.intersection(nifty_ret.index)
        lr_aligned = log_return.loc[common]
        nf_aligned = nifty_ret.loc[common].values.flatten()

        betas = pd.Series(index=log_return.columns, dtype=float)
        for ticker in log_return.columns:
            slope, _, _, _, _ = stats.linregress(
                nf_aligned, lr_aligned[ticker].values)
            betas[ticker] = slope

        print(f"\n  Betas vs {BENCHMARK}:")
        for t, b in betas.sort_values(ascending=False).items():
            bar = '█' * int(abs(b) * 10)
            print(f"  {t.replace('.NS',''):15s} β={b:.2f}  {bar}")

    except Exception as e:
        print(f"\n  Could not download {BENCHMARK} — using β=1.0 for all.")
        print(f"  Error: {e}")
        betas = pd.Series(1.0, index=log_return.columns)

    # -----------------------------------------------------------
    # 6D. Stress scenarios
    #
    #     FIX Issue 21 — actual historical returns where available.
    #
    #     OLD approach: stock_shock = beta × market_drop
    #     PROBLEM: assumes linearity. Beta=1.5 stock always
    #     falls 1.5× the market. In real crashes betas increase.
    #     A stock with normal beta=0.8 can behave like beta=1.5
    #     in a crisis. Linear beta systematically underestimates
    #     crash losses.
    #
    #     FIX — two-tier approach:
    #     Tier 1 (preferred): If crash period falls within our
    #     5-year close_df window, use ACTUAL stock returns from
    #     that period. No assumptions — real data.
    #     Tier 2 (fallback): If crash predates our data window
    #     (e.g. 2008), use stressed beta = raw_beta × 1.3.
    #     The 1.3 multiplier accounts for beta expansion in crises.
    #     For custom scenario: always stressed beta.
    #
    #     STRESS_SCENARIOS dict read from Module 0 — any scenario
    #     you added there automatically appears here.
    # -----------------------------------------------------------

    # Crash date ranges for Tier 1 (actual returns)
    crash_periods = {
        'COVID Crash (Mar 2020)'    : ('2020-01-17', '2020-03-23'),
        '2008 Crisis (Lehman)'      : ('2008-01-02', '2008-10-27'),
        'Rate Shock (2022)'         : ('2021-10-19', '2022-06-17'),
        'Russia-Ukraine (Feb 2022)' : ('2022-02-10', '2022-03-08'),
    }

    print(f"\n  {'Scenario':<30} {'Method':>14} "
        f"{'Mkt Drop':>9} {'Port Loss':>10} {'Final Value':>13}")
    print(f"  {'─'*79}")

    scenario_results = {}

    for name, market_drop in STRESS_SCENARIOS.items():
        period       = crash_periods.get(name)
        stock_shocks = {}
        method       = 'beta-scaled'

        if period is not None:
            start_p, end_p = period
            try:
                mask_period   = ((close_df.index >= start_p) &
                                (close_df.index <= end_p))
                period_prices = close_df.loc[mask_period]

                if len(period_prices) >= 5:
                    # Tier 1 — actual historical returns
                    actual_ret = (period_prices.iloc[-1] /
                                period_prices.iloc[0]) - 1

                    for ticker in optimal_weights.index:
                        if ticker in actual_ret.index:
                            stock_shocks[ticker] = float(
                                actual_ret[ticker])
                        else:
                            # Stock not available — stressed beta fallback
                            stress_beta = betas.get(ticker, 1.0) * 1.3
                            stock_shocks[ticker] = float(np.clip(
                                stress_beta * market_drop, -0.80, 0))
                    method = 'historical'
                else:
                    raise ValueError("Insufficient data in period")

            except Exception:
                # Tier 2 — stressed beta fallback
                method = 'stressed β×1.3'
                for ticker in optimal_weights.index:
                    stress_beta = betas.get(ticker, 1.0) * 1.3
                    stock_shocks[ticker] = float(np.clip(
                        stress_beta * market_drop, -0.80, 0))
        else:
            # Custom scenario — always stressed beta
            method = 'stressed β×1.3'
            for ticker in optimal_weights.index:
                stress_beta = betas.get(ticker, 1.0) * 1.3
                stock_shocks[ticker] = float(np.clip(
                    stress_beta * market_drop, -0.80, 0))

        shock_arr       = np.array([stock_shocks[t]
                                    for t in optimal_weights.index])
        portfolio_loss  = float(np.dot(optimal_weights.values, shock_arr))
        portfolio_value = INITIAL_VALUE * (1 + portfolio_loss)
        loss_rupees     = INITIAL_VALUE * abs(portfolio_loss)

        scenario_results[name] = {
            'market_drop'    : market_drop * 100,
            'portfolio_loss' : portfolio_loss * 100,
            'final_value'    : portfolio_value,
            'loss_rupees'    : loss_rupees,
            'method'         : method
        }

        print(f"  {name:<30} {method:>14} "
            f"{market_drop*100:>8.1f}% "
            f"{portfolio_loss*100:>9.1f}% "
            f"{'₹'+f'{portfolio_value:,.0f}':>13}")

    print(f"\n  'historical'   = actual stock returns during crash period")
    print(f"  'stressed β×1.3' = beta × 1.3 × market drop")
    print(f"  (1.3 multiplier accounts for beta expansion in crises)")

    # -----------------------------------------------------------
    # 6E. Plots — four panel layout
    # -----------------------------------------------------------

    fig = plt.figure(figsize=(16, 14))
    fig.patch.set_facecolor('#0f0f1a')
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_facecolor('#0f0f1a')

    # Plot 1 — historical return distribution with VaR/CVaR
    ax1.hist(hist_simple_returns * 100, bins=60,
            color='#4fc3f7', alpha=0.7, edgecolor='none')
    ax1.axvline(VaR_hist_pct * 100, color='#ef5350',
                linewidth=2, linestyle='--',
                label=f'VaR 95%: {VaR_hist_pct*100:.2f}%')
    ax1.axvline(CVaR_hist_pct * 100, color='#ff7043',
                linewidth=2, linestyle=':',
                label=f'CVaR 95%: {CVaR_hist_pct*100:.2f}%')
    ax1.axvline(0, color='yellow', linewidth=1,
                linestyle=':', label='Break-even')
    ax1.set_title('Historical Daily Return Distribution\n'
                '(In-sample — optimistic lower bound)',
                color='white', fontsize=11, fontweight='bold')
    ax1.set_xlabel('Daily Return (%)', color='white', fontsize=10)
    ax1.set_ylabel('Frequency', color='white', fontsize=10)
    ax1.tick_params(colors='white')
    ax1.spines['bottom'].set_color('#444444')
    ax1.spines['left'].set_color('#444444')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.grid(True, alpha=0.1, color='white')
    ax1.legend(fontsize=8, framealpha=0.3, facecolor='#1a1a2e',
            edgecolor='#444444', labelcolor='white')

    # Plot 2 — drawdown distribution
    ax2.hist(all_drawdowns * 100, bins=50,
            color='#ef5350', alpha=0.7, edgecolor='none')
    ax2.axvline(median_dd, color='white', linewidth=2,
                linestyle='--', label=f'Median: {median_dd:.1f}%')
    ax2.axvline(dd_pct5, color='#ff7043', linewidth=2,
                linestyle=':', label=f'Worst 5%: {dd_pct5:.1f}%')
    ax2.set_title(f'Max Drawdown Distribution\n'
                f'({port_value_paths.shape[1]:,} Simulation Paths)',
                color='white', fontsize=11, fontweight='bold')
    ax2.set_xlabel('Max Drawdown (%)', color='white', fontsize=10)
    ax2.set_ylabel('Frequency', color='white', fontsize=10)
    ax2.tick_params(colors='white')
    ax2.spines['bottom'].set_color('#444444')
    ax2.spines['left'].set_color('#444444')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.grid(True, alpha=0.1, color='white')
    ax2.legend(fontsize=8, framealpha=0.3, facecolor='#1a1a2e',
            edgecolor='#444444', labelcolor='white')

    # Plot 3 — stress scenario bar chart
    names  = list(scenario_results.keys())
    losses = [scenario_results[n]['portfolio_loss'] for n in names]
    colors = ['#ef5350' if l < -20 else
            '#ff7043' if l < -10 else
            '#ffa726' for l in losses]

    bars = ax3.barh(names, losses, color=colors,
                    alpha=0.8, edgecolor='none')
    for bar, name in zip(bars, names):
        loss_r  = scenario_results[name]['loss_rupees']
        method_ = scenario_results[name]['method']
        ax3.text(
            bar.get_width() - 0.5,
            bar.get_y() + bar.get_height() / 2,
            f'-₹{loss_r:,.0f}  ({method_})',
            va='center', ha='right',
            color='white', fontsize=7.5, fontweight='bold'
        )
    ax3.axvline(0, color='white', linewidth=0.8)
    ax3.set_title('Stress Scenario Portfolio Impact\n'
                'Historical actual returns where available',
                color='white', fontsize=10, fontweight='bold')
    ax3.set_xlabel('Portfolio Loss (%)', color='white', fontsize=10)
    ax3.tick_params(colors='white')
    ax3.spines['bottom'].set_color('#444444')
    ax3.spines['left'].set_color('#444444')
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.grid(True, alpha=0.1, color='white', axis='x')

    # Plot 4 — beta bar chart
    sorted_betas = betas.sort_values(ascending=True)
    beta_colors  = ['#ef5350' if b > 1.2 else
                    '#4fc3f7' if b < 0.8 else
                    '#aaaaaa'
                    for b in sorted_betas.values]
    ax4.barh(
        [t.replace('.NS', '') for t in sorted_betas.index],
        sorted_betas.values,
        color=beta_colors, alpha=0.8, edgecolor='none'
    )
    ax4.axvline(1.0, color='yellow', linewidth=1,
                linestyle='--', label='Market (β=1.0)')
    ax4.axvline(0.0, color='white', linewidth=0.5)
    ax4.set_title(f'Stock Betas vs {BENCHMARK}',
                color='white', fontsize=11, fontweight='bold')
    ax4.set_xlabel('Beta', color='white', fontsize=10)
    ax4.tick_params(colors='white')
    ax4.spines['bottom'].set_color('#444444')
    ax4.spines['left'].set_color('#444444')
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    ax4.grid(True, alpha=0.1, color='white', axis='x')
    ax4.legend(fontsize=8, framealpha=0.3, facecolor='#1a1a2e',
            edgecolor='#444444', labelcolor='white')

    plt.suptitle(
        'Risk & Stress Testing\n'
        'Historical VaR  |  Max Drawdown  |  '
        'Crisis Scenarios  |  Beta',
        color='white', fontsize=13, fontweight='bold', y=1.01
    )
    plt.savefig(str(output_dir / 'risk_stress_testing.png'), dpi=150,
                bbox_inches='tight', facecolor='#0f0f1a')
    plt.show()

    # -----------------------------------------------------------
    # 6F. Honest risk summary
    # -----------------------------------------------------------

    print(f"\n{'='*58}")
    print(f"{'HONEST RISK SUMMARY':^58}")
    print(f"{'='*58}")
    print(f"""
    What the model says vs what reality adds:

    Metric              Model Output       Reality Check
    ────────────────────────────────────────────────────
    Expected Return     {port_return*100:>+6.1f}%           
                                            (bull market bias)
    Prob. of Loss       {prob_loss:>6.1f}%           
                                            (fat tails remain)
    Annual VaR (95%)    ₹{VaR_hist_annual:>10,.0f}   In-sample optimistic
                                            
    Worst Case (1yr)    ₹{min(final_values):>10,.0f}   Could be worse
                                            (regime change risk)

    Key risks not captured by any model:
    → Liquidity risk   : Can you sell at quoted price?
    → Regulatory risk  : SEBI / RBI policy changes
    → Concentration    : Top 3 stocks ≈ 40-45% of portfolio
    → Sequence risk    : Bad year early damages compounding
    → In-sample bias   : Weights trained on same data tested
    → Regime change    : 5-year bull run may not continue
    """)
    print("=" * 58)

    # -----------------------------------------------------------
    # MODULE 6 OUTPUTS
    # -----------------------------------------------------------
    # VaR_hist_1day    → float  : historical 1-day VaR (₹)
    # CVaR_hist_1day   → float  : historical 1-day CVaR (₹)
    # VaR_hist_annual  → float  : simulation-based annual VaR (₹)
    # CVaR_hist_annual → float  : simulation-based annual CVaR (₹)
    # all_drawdowns    → array  : max drawdown per sim path
    # scenario_results → dict   : stress scenario outcomes
    # betas            → Series : stock betas vs benchmark
    # -----------------------------------------------------------

    # ---- publish (generated) ----
    _scope = locals()
    ctx.update({k: _scope[k] for k in _PUBLISHED if k in _scope})

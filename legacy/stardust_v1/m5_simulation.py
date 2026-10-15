"""
Legacy Stardust 1.x - M5 - Forward simulation (Cholesky t-distribution Monte Carlo)

Mechanically extracted from stardust_core.py lines 2702-3118 by
legacy/tools/split_monolith.py. The code inside run() is the original,
byte-for-byte; only the input/publish scaffolding around it is generated.
Do not hand-edit the extracted block - fix bugs in STAR2.0 (star/) instead.
"""
from ._bootstrap import ensure_bootstrap

ensure_bootstrap()

#: data this stage reads from earlier stages (its upstream contract)
INPUTS = (
    'arith_returns',
    'cov_daily',
    'GAIN_THRESHOLD_1',
    'GAIN_THRESHOLD_2',
    'GLOBAL_SEED',
    'INITIAL_VALUE',
    'n_stocks',
    'optimal_weights',
    'output_dir',
    'port_return',
    'status_callback',
    'T_DIST_DF',
    'TRADING_DAYS',
)

#: legacy loop variables that leak in from earlier stages; pulled only so the
#: split behaves exactly like the monolith's single shared scope
CARRYOVER = ()

#: data this stage produces that a later stage reads (downstream contract)
OUTPUTS = (
    'CVaR_95',
    'final_values',
    'N_SIMULATIONS',
    'port_value_paths',
    'prob_loss',
    'VaR_95',
)

#: everything bound at stage scope; all of it is published to ctx
_PUBLISHED = ('CVaR_95', 'L', 'N_SIMULATIONS', 'VaR_95', 'Z', 'Z_raw', '_sim_header', 'ax', 'ax1', 'ax2', 'ax3', 'cell', 'col', 'corr_shocks', 'daily_arith', 'daily_drift', 'daily_log', 'daily_var', 'days_axis', 'dist_label', 'fig', 'final_values', 'gap', 'gs', 'lbl', 'max_val', 'mean_val', 'median_val', 'min_val', 'n_plot', 'percentile_25', 'percentile_5', 'percentile_75', 'percentile_95', 'plot_idx', 'port_cumret', 'port_daily_drift', 'port_daily_std', 'port_daily_var', 'port_log', 'port_value_paths', 'prob_gain_20', 'prob_gain_30', 'prob_loss', 'row', 't_std', 't_variance', 'table', 'table_data', 'val')


def run(ctx: dict) -> None:
    # ---- inputs from upstream stages (generated) ----
    arith_returns = ctx['arith_returns']
    cov_daily = ctx['cov_daily']
    GAIN_THRESHOLD_1 = ctx['GAIN_THRESHOLD_1']
    GAIN_THRESHOLD_2 = ctx['GAIN_THRESHOLD_2']
    GLOBAL_SEED = ctx['GLOBAL_SEED']
    INITIAL_VALUE = ctx['INITIAL_VALUE']
    n_stocks = ctx['n_stocks']
    optimal_weights = ctx['optimal_weights']
    output_dir = ctx['output_dir']
    port_return = ctx['port_return']
    status_callback = ctx['status_callback']
    T_DIST_DF = ctx['T_DIST_DF']
    TRADING_DAYS = ctx['TRADING_DAYS']
    # ---- original code: stardust_core.py lines 2702-3118 ----
    # ============================================================
    # MODULE 5: Forward Simulation — Cholesky Monte Carlo
    # ------------------------------------------------------------
    # Reads from Module 0: INITIAL_VALUE, RISK_FREE,
    #                      GLOBAL_SEED, T_DIST_DF
    # Reads from Module 2: arith_returns, cov_daily
    # Reads from Module 3: optimal_weights, n_stocks
    #
    # Corrections applied:
    #   - Issue 16 : arith_returns correctly ordered — drift
    #                per stock matches the correct stock
    #   - Issue 17 : GBM constant-vol limitation documented
    #   - Issue 18 : N_SIMULATIONS scales with stock count
    #                for reliable tail estimates
    #   - Issue 28 : Uses GLOBAL_SEED from Module 0
    #   - NEW      : Student's t-distribution replaces normal
    #                for fat-tailed realistic simulation
    #                Normalized to preserve covariance structure
    # ============================================================

    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from scipy.stats import t as t_dist

    status_callback("Module 5: Running Monte Carlo simulation...")

    # -----------------------------------------------------------
    # 5A. N_SIMULATIONS scales with stock count
    #
    #     WHY SCALE?
    #     With 1,000 paths the 5th percentile VaR is based on
    #     only 50 observations — statistically unreliable.
    #     The confidence interval around that estimate is wide
    #     enough to make the number meaningless.
    #
    #     With fewer stocks each simulation path is faster to
    #     compute — so we can afford more paths for better
    #     tail accuracy. With many stocks each path is slower
    #     so we reduce count to keep runtime reasonable.
    #
    #     At 10,000 paths: 5th percentile uses 500 obs — stable.
    #     At 20,000 paths: 5th percentile uses 1000 obs — very stable.
    # -----------------------------------------------------------

    if n_stocks <= 15:
        N_SIMULATIONS = 20_000
    elif n_stocks <= 50:
        N_SIMULATIONS = 10_000
    else:
        N_SIMULATIONS = 5_000

    print(f"Simulation settings:")
    print(f"  Stocks       : {n_stocks}")
    print(f"  Paths        : {N_SIMULATIONS:,}")
    print(f"  Days         : {TRADING_DAYS}")
    print(f"  Distribution : "
        f"{'Normal (GBM)' if T_DIST_DF == np.inf else f't-distribution (df={T_DIST_DF})'}")
    print(f"  Initial val  : ₹{INITIAL_VALUE:,.0f}")

    # -----------------------------------------------------------
    # 5B. Cholesky decomposition
    #
    #     WHY CHOLESKY?
    #     Your 24 stocks are correlated — HDFC and ICICI move
    #     together, ITC moves differently from TCS. Generating
    #     independent random returns per stock ignores these
    #     relationships entirely — the simulation is unrealistic
    #     and understates diversification benefit.
    #
    #     Cholesky takes cov_daily and finds matrix L such that:
    #       L @ L.T = cov_daily
    #     When you multiply L by independent random draws, the
    #     result has exactly the correlation structure of your
    #     real stocks — correlated in the right way, right amount.
    #
    #     WHY cov_daily NOT cov_annual?
    #     Simulation steps one day at a time over 252 days.
    #     Daily covariance = correct scale for one day.
    #     Annual covariance would overstate daily moves by 252×.
    #
    #     Ledoit-Wolf in Module 2 guarantees all eigenvalues
    #     are positive — so Cholesky cannot fail here.
    # -----------------------------------------------------------

    L = np.linalg.cholesky(cov_daily.values)
    print(f"\nCholesky decomposition:")
    print(f"  L shape         : {L.shape}")
    print(f"  L @ L.T ≈ cov   : {np.allclose(L @ L.T, cov_daily.values)}")

    if not np.allclose(L @ L.T, cov_daily.values):
        raise ValueError(
            "Cholesky verification failed — L @ L.T does not equal cov_daily.\n"
            "Check Module 2 Ledoit-Wolf computation."
        )
    print(f"  ✓ Cholesky verified — simulation is correctly structured.")

    # -----------------------------------------------------------
    # 5C. Daily drift per stock — Ito corrected
    #
    #     FIX Issue 16 — arith_returns correctly ordered.
    #     Each stock gets its own drift matched by index position.
    #     Before the fix: sorted arith_returns meant Stock A got
    #     Stock Z's drift. Every simulation path was wrong.
    #
    #     GBM drift formula:
    #       drift_i = (arith_return_i / 252) - 0.5 × variance_i
    #
    #     The -0.5 × variance term is the Ito correction.
    #     Without it: simulation overestimates returns because
    #     GBM with log returns has a systematic upward bias.
    #     With it: simulation correctly represents what an
    #     investor actually receives in compounded returns.
    #
    #     FIX Issue 17 — GBM limitation documented:
    #     This simulation assumes volatility and correlations
    #     are constant throughout the year. In reality:
    #     - Volatility clusters (GARCH effect — high vol
    #       periods follow high vol periods)
    #     - Correlations spike during market crashes
    #     - Returns have fat tails beyond what t-dist captures
    #     Module 6 historical VaR partially corrects for this
    #     by using actual past returns with real distributional
    #     properties.
    # -----------------------------------------------------------

    daily_arith  = arith_returns.values / TRADING_DAYS
    daily_var    = np.diag(cov_daily.values)
    daily_drift  = daily_arith - 0.5 * daily_var

    port_daily_drift = np.dot(optimal_weights.values, daily_drift)
    port_daily_var   = (optimal_weights.values.T @
                        cov_daily.values @
                        optimal_weights.values)
    port_daily_std   = np.sqrt(port_daily_var)

    print(f"\nPortfolio daily stats:")
    print(f"  Daily drift (Ito corrected) : {port_daily_drift*100:.4f}%")
    print(f"  Daily std                   : {port_daily_std*100:.4f}%")
    print(f"  Annualized drift check      : {port_daily_drift*252*100:.2f}%")
    print(f"  Module 3 port_return        : {port_return*100:.2f}%")
    gap = abs(port_daily_drift*TRADING_DAYS - port_return)
    if gap < 0.02:
        print(f"  ✓ Drift matches Module 3 return (gap={gap*100:.2f}%)")
    else:
        print(f"  ⚠ Drift gap={gap*100:.2f}% — check arith_returns alignment")

    # -----------------------------------------------------------
    # 5D. NEW — Student's t-distribution shocks
    #
    #     WHY t-DISTRIBUTION?
    #     Standard GBM uses normal distribution for daily shocks.
    #     Real equity returns have fat tails — extreme moves
    #     happen far more frequently than normal predicts.
    #     The 2020 COVID crash, 2008 Lehman collapse, and 2022
    #     rate shock all produced moves that were 5-8 standard
    #     deviations under a normal assumption — events that
    #     should happen once in thousands of years but actually
    #     happen every decade.
    #
    #     Student's t-distribution with low degrees of freedom
    #     produces fatter tails — more frequent extreme moves
    #     that better match real market behaviour.
    #
    #     T_DIST_DF from Module 0:
    #     df=3  : very fat tails (emerging markets, crypto)
    #     df=5  : fat tails — standard for equity returns
    #     df=10 : mild fat tails
    #     df=∞  : normal distribution (original GBM)
    #
    #     NORMALIZATION — critical step:
    #     t-distribution with df degrees of freedom has variance
    #     df/(df-2), not 1. If you multiply L (from cov_daily)
    #     by unnormalized t shocks, the resulting portfolio has
    #     variance = cov_daily × df/(df-2) — inflated by the
    #     t-variance. Dividing by sqrt(df/(df-2)) restores unit
    #     variance so the covariance structure is preserved.
    #
    #     Without normalization: simulation overstates volatility
    #     by ~29% (df=5), making risk metrics meaningless.
    # -----------------------------------------------------------

    np.random.seed(GLOBAL_SEED)
    print(f"\nRunning {N_SIMULATIONS:,} simulations × {TRADING_DAYS} days...")

    if T_DIST_DF == np.inf:
        # Standard normal — original GBM
        Z = np.random.standard_normal((TRADING_DAYS, n_stocks, N_SIMULATIONS))
        print(f"  Using normal distribution (T_DIST_DF=inf)")
    else:
        # Student's t — fat tails
        # t variance = df/(df-2), normalize to unit variance
        t_variance    = T_DIST_DF / (T_DIST_DF - 2)
        t_std         = np.sqrt(t_variance)
        Z_raw         = t_dist.rvs(
                            df   = T_DIST_DF,
                            size = (TRADING_DAYS, n_stocks, N_SIMULATIONS)
                        )
        Z             = Z_raw / t_std   # normalize to unit variance
        print(f"  Using t-distribution (df={T_DIST_DF})")
        print(f"  t variance = {t_variance:.4f}  "
            f"→ normalized by {t_std:.4f}")
        print(f"  Tail weight vs normal: "
            f"{t_dist.pdf(0, df=T_DIST_DF) / 0.3989:.3f}x at peak, "
            f"heavier tails by construction")

    # -----------------------------------------------------------
    # 5E. Run simulation — fully vectorized
    #
    #     Shape guide:
    #     Z           : (N_DAYS, n_stocks, N_SIMULATIONS)
    #     corr_shocks : (N_DAYS, n_stocks, N_SIMULATIONS)
    #                   — shocks correlated via Cholesky
    #     daily_log   : (N_DAYS, n_stocks, N_SIMULATIONS)
    #                   — drift + correlated shock per stock per day
    #     port_log    : (N_DAYS, N_SIMULATIONS)
    #                   — weighted portfolio log return per day
    #     port_cumret : (N_DAYS, N_SIMULATIONS)
    #                   — cumulative portfolio return path
    # -----------------------------------------------------------

    corr_shocks      = np.einsum('ij,djk->dik', L, Z)
    daily_log        = daily_drift[np.newaxis, :, np.newaxis] + corr_shocks
    port_log         = np.einsum('i,dij->dj', optimal_weights.values, daily_log)
    port_cumret      = np.exp(np.cumsum(port_log, axis=0))
    port_value_paths = INITIAL_VALUE * port_cumret

    print(f"  Simulation complete. Paths shape: {port_value_paths.shape}")

    # -----------------------------------------------------------
    # 5F. Summary statistics
    #
    #     With scaled N_SIMULATIONS, percentile estimates are
    #     statistically reliable:
    #     At 10,000 paths: 5th pct = 500 observations
    #     At 20,000 paths: 5th pct = 1,000 observations
    # -----------------------------------------------------------

    final_values  = port_value_paths[-1, :]

    median_val    = np.median(final_values)
    mean_val      = np.mean(final_values)
    percentile_5  = np.percentile(final_values, 5)
    percentile_25 = np.percentile(final_values, 25)
    percentile_75 = np.percentile(final_values, 75)
    percentile_95 = np.percentile(final_values, 95)
    prob_loss     = np.mean(final_values < INITIAL_VALUE) * 100
    prob_gain_20  = np.mean(final_values > INITIAL_VALUE * (1+ GAIN_THRESHOLD_1)) * 100
    prob_gain_30  = np.mean(final_values > INITIAL_VALUE * (1+ GAIN_THRESHOLD_2)) * 100
    max_val       = final_values.max()
    min_val       = final_values.min()

    VaR_95  = INITIAL_VALUE - percentile_5
    CVaR_95 = INITIAL_VALUE - np.mean(
                final_values[final_values <= percentile_5])

    dist_label = ('Normal' if T_DIST_DF == np.inf
                else f't-dist df={T_DIST_DF}')

    print(f"\n{'='*55}")
    _sim_header = f'SIMULATION SUMMARY — ₹{INITIAL_VALUE:,.0f} PORTFOLIO'
    print(f'{_sim_header:^55}')
    print(f"  {N_SIMULATIONS:,} paths  |  Distribution: {dist_label}")
    print(f"{'='*55}")
    print(f"  {'Metric':<30} {'Value':>21}")
    print(f"  {'-'*51}")
    print(f"  {'Median final value':<30} {'₹'+f'{median_val:,.0f}':>21}")
    print(f"  {'Mean final value':<30} {'₹'+f'{mean_val:,.0f}':>21}")
    print(f"  {'Best 5% (95th pct)':<30} {'₹'+f'{percentile_95:,.0f}':>21}")
    print(f"  {'Worst 5% (5th pct)':<30} {'₹'+f'{percentile_5:,.0f}':>21}")
    print(f"  {'Best case':<30} {'₹'+f'{max_val:,.0f}':>21}")
    print(f"  {'Worst case':<30} {'₹'+f'{min_val:,.0f}':>21}")
    print(f"  {'-'*51}")
    print(f"  {'Probability of loss':<30} {prob_loss:>20.1f}%")
    print(f"  {'Probability of >{GAIN_THRESHOLD_1*100:.0f}% gain':<30} {prob_gain_20:>20.1f}%")
    print(f"  {'Probability of >{GAIN_THRESHOLD_2*100:.0f}% gain':<30} {prob_gain_30:>20.1f}%")
    print(f"  {'-'*51}")
    print(f"  {'VaR 95% (1 year)':<30} {'₹'+f'{VaR_95:,.0f}':>21}")
    print(f"  {'CVaR 95% (1 year)':<30} {'₹'+f'{CVaR_95:,.0f}':>21}")
    print(f"{'='*55}")
    print(f"\n  Note: t-distribution captures fat tails better than")
    print(f"  normal GBM. Historical VaR in Module 6 uses actual")
    print(f"  past returns — compare both for full risk picture.")

    # -----------------------------------------------------------
    # 5G. Three plots
    # -----------------------------------------------------------

    fig = plt.figure(figsize=(16, 14))
    fig.patch.set_facecolor('#0f0f1a')
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])
    for ax in [ax1, ax2, ax3]:
        ax.set_facecolor('#0f0f1a')

    days_axis = np.arange(1, TRADING_DAYS + 1)

    # Plot 1 — spaghetti paths (subsample for speed)
    n_plot   = min(500, N_SIMULATIONS)
    plot_idx = np.random.choice(N_SIMULATIONS, size=n_plot, replace=False)
    ax1.plot(days_axis,
            port_value_paths[:, plot_idx] / 1e5,
            color='#4fc3f7', alpha=0.04, linewidth=0.5)
    ax1.plot(days_axis,
            np.percentile(port_value_paths, 50, axis=1) / 1e5,
            color='white', linewidth=2, label='Median', zorder=5)
    ax1.plot(days_axis,
            np.percentile(port_value_paths, 5, axis=1) / 1e5,
            color='#ef5350', linewidth=1.5, linestyle='--',
            label='5th percentile (worst 5%)', zorder=5)
    ax1.plot(days_axis,
            np.percentile(port_value_paths, 95, axis=1) / 1e5,
            color='#66bb6a', linewidth=1.5, linestyle='--',
            label='95th percentile (best 5%)', zorder=5)
    ax1.axhline(INITIAL_VALUE / 1e5, color='yellow',
                linewidth=1, linestyle=':', label='Initial investment', zorder=4)

    ax1.set_title(
        f'Monte Carlo — {N_SIMULATIONS:,} Paths  |  '
        f'Distribution: {dist_label}  |  1 Year Forward',
        color='white', fontsize=12, fontweight='bold', pad=10)
    ax1.set_xlabel('Trading Days', color='white', fontsize=10)
    ax1.set_ylabel('Portfolio Value (₹ Lakhs)', color='white', fontsize=10)
    ax1.tick_params(colors='white')
    ax1.spines['bottom'].set_color('#444444')
    ax1.spines['left'].set_color('#444444')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.grid(True, alpha=0.1, color='white')
    ax1.legend(fontsize=9, framealpha=0.3, facecolor='#1a1a2e',
            edgecolor='#444444', labelcolor='white')

    # Plot 2 — final value distribution
    ax2.hist(final_values / 1e5, bins=80,
            color='#4fc3f7', alpha=0.7, edgecolor='none')
    for val, lbl, col in [
        (percentile_5,  '5th pct',  '#ef5350'),
        (median_val,    'Median',   'white'),
        (percentile_95, '95th pct', '#66bb6a'),
        (INITIAL_VALUE, 'Invested', 'yellow'),
    ]:
        ax2.axvline(val / 1e5, color=col, linewidth=1.5,
                    linestyle='--', label=lbl)

    ax2.set_title('Distribution of Final Portfolio Values',
                color='white', fontsize=11, fontweight='bold')
    ax2.set_xlabel('Final Value (₹ Lakhs)', color='white', fontsize=10)
    ax2.set_ylabel('Frequency', color='white', fontsize=10)
    ax2.tick_params(colors='white')
    ax2.spines['bottom'].set_color('#444444')
    ax2.spines['left'].set_color('#444444')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.grid(True, alpha=0.1, color='white')
    ax2.legend(fontsize=8, framealpha=0.3, facecolor='#1a1a2e',
            edgecolor='#444444', labelcolor='white')

    # Plot 3 — summary table
    ax3.axis('off')
    table_data = [
        ['Metric',              'Value'],
        ['Paths',               f'{N_SIMULATIONS:,}'],
        ['Distribution',        dist_label],
        ['Initial Investment',  f'₹{INITIAL_VALUE:,.0f}'],
        ['Median Outcome',      f'₹{median_val:,.0f}'],
        ['Mean Outcome',        f'₹{mean_val:,.0f}'],
        ['Best 5% Scenario',    f'₹{percentile_95:,.0f}'],
        ['Worst 5% Scenario',   f'₹{percentile_5:,.0f}'],
        ['Prob. of Any Loss',   f'{prob_loss:.1f}%'],
        [f'Prob. of >{GAIN_THRESHOLD_1*100:.0f}% Gain', f'{prob_gain_20:.1f}%'],
        [f'Prob. of >{GAIN_THRESHOLD_2*100:.0f}% Gain', f'{prob_gain_30:.1f}%'],
        ['VaR 95% (1yr)',       f'₹{VaR_95:,.0f}'],
        ['CVaR 95% (1yr)',      f'₹{CVaR_95:,.0f}'],
    ]

    table = ax3.table(
        cellText  = table_data[1:],
        colLabels = table_data[0],
        loc       = 'center',
        cellLoc   = 'center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.8)
    for (row, col), cell in table.get_celld().items():
        cell.set_facecolor('#1a1a2e')
        cell.set_text_props(color='white')
        cell.set_edgecolor('#444444')
        if row == 0:
            cell.set_facecolor('#2d2d4e')
            cell.set_text_props(color='white', fontweight='bold')

    ax3.set_title('Summary Statistics',
                color='white', fontsize=11, fontweight='bold')

    plt.suptitle(
        f'Monte Carlo Forward Simulation  —  Cholesky Correlated Returns\n'
        f'₹{INITIAL_VALUE:,.0f} Initial  |  {N_SIMULATIONS:,} Paths  |  '
        f'{TRADING_DAYS} Trading Days  |  {dist_label}',
        color='white', fontsize=12, fontweight='bold', y=1.01
    )
    plt.savefig(str(output_dir / 'monte_carlo_simulation.png'), dpi=150,
                bbox_inches='tight', facecolor='#0f0f1a')
    plt.show()

    # -----------------------------------------------------------
    # MODULE 5 OUTPUTS — passed into Module 6
    # -----------------------------------------------------------
    # port_value_paths → np.ndarray (N_DAYS, N_SIMULATIONS)
    # final_values     → np.ndarray (N_SIMULATIONS,)
    # VaR_95           → float : 1-year VaR from simulation
    # CVaR_95          → float : 1-year CVaR from simulation
    # prob_loss        → float : probability of loss %
    # N_SIMULATIONS    → int   : path count (used in Module 6)
    # -----------------------------------------------------------
    # ---- publish (generated) ----
    _scope = locals()
    ctx.update({k: _scope[k] for k in _PUBLISHED if k in _scope})

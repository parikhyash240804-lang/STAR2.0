"""
Legacy Stardust 1.x - M4 - Efficient frontier (random-weights Monte Carlo)

Mechanically extracted from stardust_core.py lines 2298-2701 by
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
    'minvar_return',
    'minvar_sharpe',
    'minvar_volatility',
    'n_stocks',
    'output_dir',
    'port_return',
    'port_sharpe',
    'port_volatility',
    'RISK_FREE',
    'status_callback',
    'WEIGHT_MAX',
    'WEIGHT_MIN',
)

#: legacy loop variables that leak in from earlier stages; pulled only so the
#: split behaves exactly like the monolith's single shared scope
CARRYOVER = ()

#: data this stage produces that a later stage reads (downstream contract)
OUTPUTS = ()

#: everything bound at stage scope; all of it is published to ctx
_PUBLISHED = ('HIGH', 'LOW', 'N_BATCH', 'N_PORTFOLIOS', 'OPTIMIZER_VALIDATION_TOLERANCE', 'accept_rate', 'ax', 'best_bnd_sh', 'bounded_ret', 'bounded_sharpe', 'bounded_vol', 'bounded_w', 'cbar', 'clipped', 'cov_vals', 'diff', 'fig', 'half', 'idx', 'n_valid', 'normed', 'raw_batch', 'raw_weights', 'ret_vals', 'row_sums', 'scatter', 'sim_returns', 'sim_sharpes', 'sim_vols', 'unbounded_gap', 'valid_mask', 'w_concentrated', 'w_uniform', 'x_cml', 'x_max', 'y_cml')


def run(ctx: dict) -> None:
    # ---- inputs from upstream stages (generated) ----
    arith_returns = ctx['arith_returns']
    cov_annual = ctx['cov_annual']
    GLOBAL_SEED = ctx['GLOBAL_SEED']
    minvar_return = ctx['minvar_return']
    minvar_sharpe = ctx['minvar_sharpe']
    minvar_volatility = ctx['minvar_volatility']
    n_stocks = ctx['n_stocks']
    output_dir = ctx['output_dir']
    port_return = ctx['port_return']
    port_sharpe = ctx['port_sharpe']
    port_volatility = ctx['port_volatility']
    RISK_FREE = ctx['RISK_FREE']
    status_callback = ctx['status_callback']
    WEIGHT_MAX = ctx['WEIGHT_MAX']
    WEIGHT_MIN = ctx['WEIGHT_MIN']
    # ---- original code: stardust_core.py lines 2298-2701 ----
    # ============================================================
    # MODULE 4: Efficient Frontier — Random Weights Monte Carlo
    # ------------------------------------------------------------
    # Reads from Module 0: RISK_FREE, GLOBAL_SEED
    # Reads from Module 2: arith_returns, cov_annual
    # Reads from Module 3: optimal_weights, minvar_weights,
    #                      port_return, port_volatility,
    #                      port_sharpe, minvar_return,
    #                      minvar_volatility, minvar_sharpe,
    #                      WEIGHT_MIN, WEIGHT_MAX
    #
    # Corrections applied:
    #   - Issue 13 : arith_returns correctly ordered — frontier
    #                return calculations are now accurate
    #   - Issue 14 : Mixed Dirichlet sampling — alpha=1 and
    #                alpha=0.5 for better edge coverage
    #   - Issue 15 : CML formula verified — units consistent
    #   - Issue 28 : Uses GLOBAL_SEED from Module 0
    #   - Dynamic  : N_PORTFOLIOS scales with stock count
    #   - Dynamic  : LOW, HIGH use WEIGHT_MIN, WEIGHT_MAX
    #                not hardcoded values
    # ============================================================

    import numpy as np
    import matplotlib.pyplot as plt
    OPTIMIZER_VALIDATION_TOLERANCE = 0.05 #For Optimizer

    status_callback("Module 4: Building efficient frontier...")
    # -----------------------------------------------------------
    # 4A. N_PORTFOLIOS scales with stock count
    #
    #     More stocks = larger feasible region = more portfolios
    #     needed to get a smooth frontier. But above 60 stocks
    #     each portfolio computation is slower — reduce count
    #     to keep runtime reasonable.
    #
    #     These values produce a smooth, dense frontier cloud
    #     at each stock count range.
    # -----------------------------------------------------------

    if n_stocks <= 10:
        N_PORTFOLIOS = 50_000
    elif n_stocks <= 30:
        N_PORTFOLIOS = 110_000
    elif n_stocks <= 60:
        N_PORTFOLIOS = 80_000
    else:
        N_PORTFOLIOS = 50_000

    LOW  = WEIGHT_MIN   # use dynamic bounds from Module 3
    HIGH = WEIGHT_MAX   # not hardcoded 0.02 / 0.15

    cov_vals = cov_annual.values    # annual — for Sharpe calculation
    ret_vals = arith_returns.values # correctly ordered — never sorted

    print(f"Efficient Frontier settings:")
    print(f"  Stocks         : {n_stocks}")
    print(f"  Portfolios     : {N_PORTFOLIOS:,}")
    print(f"  Bounds (LOW)   : {LOW*100:.2f}%")
    print(f"  Bounds (HIGH)  : {HIGH*100:.2f}%")

    # -----------------------------------------------------------
    # 4B. Generate unbounded portfolios — full opportunity set
    #
    #     PURPOSE: show the complete risk-return landscape
    #     including portfolios outside your weight constraints.
    #     This is intentional — the unbounded cloud shows where
    #     unconstrained portfolios can reach, making your
    #     optimizer's bounded solution visually interpretable.
    #
    #     FIX Issue 14 — mixed Dirichlet sampling.
    #     Standard Dirichlet(alpha=1) concentrates near equal
    #     weight (1/n each). With 24 stocks this means most
    #     random portfolios cluster around 4.2% per stock.
    #     The frontier edges — where concentrated, high-return
    #     portfolios live — are severely undersampled, making
    #     the upper-left region of the frontier sparse.
    #
    #     Fix: 50/50 mix of two Dirichlet concentrations.
    #     alpha=1.0 → uniform, covers center of frontier
    #     alpha=0.5 → more concentrated, covers edges/extremes
    #     This gives a smooth, well-covered frontier cloud
    #     across the entire risk-return space.
    # -----------------------------------------------------------

    print(f"\nGenerating {N_PORTFOLIOS:,} unbounded portfolios...")
    np.random.seed(GLOBAL_SEED)

    half           = N_PORTFOLIOS // 2
    w_uniform      = np.random.dirichlet(
                        np.ones(n_stocks), size=half)
    w_concentrated = np.random.dirichlet(
                        0.5 * np.ones(n_stocks), size=N_PORTFOLIOS - half)
    raw_weights    = np.vstack([w_uniform, w_concentrated])

    # Shuffle to avoid systematic patterns in the scatter plot
    idx         = np.random.permutation(len(raw_weights))
    raw_weights = raw_weights[idx]

    # Fully vectorized computation — no loops
    # einsum handles w @ Cov @ w for all 110k portfolios at once
    sim_returns = raw_weights @ ret_vals
    sim_vols    = np.sqrt(np.einsum(
                    'ij,jk,ik->i', raw_weights, cov_vals, raw_weights))
    sim_sharpes = (sim_returns - RISK_FREE) / sim_vols

    print(f"Unbounded simulation complete.")
    print(f"  Return range : {sim_returns.min()*100:.1f}% "
        f"→ {sim_returns.max()*100:.1f}%")
    print(f"  Vol range    : {sim_vols.min()*100:.1f}% "
        f"→ {sim_vols.max()*100:.1f}%")
    print(f"  Sharpe range : {sim_sharpes.min():.2f} "
        f"→ {sim_sharpes.max():.2f}")

    # -----------------------------------------------------------
    # 4C. Generate bounded portfolios — validation layer
    #
    #     These respect WEIGHT_MIN and WEIGHT_MAX — same
    #     constraints as the optimizer. Used to:
    #     1. Visually show the constrained feasible region
    #     2. Numerically validate the optimizer found the best
    #        point within those constraints
    #
    #     METHOD: clip → renormalize → validate
    #     Faster than rejection sampling. Generates 5M raw
    #     portfolios and keeps only those that remain within
    #     bounds after renormalization.
    #
    #     WHY 5M BATCH?
    #     With 24 stocks and WEIGHT_MIN=2%, WEIGHT_MAX=15%,
    #     the probability of all 24 weights simultaneously
    #     falling within bounds after Dirichlet sampling is
    #     very small (~0.001-0.01%). Need a large batch to
    #     collect enough valid portfolios for a meaningful plot.
    # -----------------------------------------------------------

    print(f"\nGenerating bounded portfolios for validation (1M batch)...")
    N_BATCH    = 1_000_000
    raw_batch  = np.random.dirichlet(np.ones(n_stocks), size=N_BATCH)
    clipped    = np.clip(raw_batch, LOW, HIGH)
    row_sums   = clipped.sum(axis=1, keepdims=True)
    normed     = clipped / row_sums
    valid_mask = ((normed >= LOW - 1e-6) &
                (normed <= HIGH + 1e-6)).all(axis=1)
    bounded_w  = normed[valid_mask]

    n_valid     = len(bounded_w)
    accept_rate = n_valid / N_BATCH * 100
    print(f"  Generated {N_BATCH:,} → {n_valid:,} valid bounded portfolios")
    print(f"  Acceptance rate: {accept_rate:.3f}%")

    if n_valid >= 10:
        bounded_ret    = bounded_w @ ret_vals
        bounded_vol    = np.sqrt(np.einsum(
                            'ij,jk,ik->i', bounded_w, cov_vals, bounded_w))
        bounded_sharpe = (bounded_ret - RISK_FREE) / bounded_vol
        best_bnd_sh    = bounded_sharpe.max()
    else:
        best_bnd_sh = None
        print("  Too few bounded portfolios — visual check sufficient.")

    # -----------------------------------------------------------
    # 4D. Optimizer validation check
    #
    #     FIX Issue 15 — CML unit verification.
    #     port_sharpe computed in decimal units:
    #       (port_return - RISK_FREE) / port_volatility
    #     All three terms are decimal → port_sharpe is decimal.
    #     CML formula:
    #       y_cml = RISK_FREE*100 + port_sharpe * x_cml
    #     x_cml is in % → y_cml is in % → units consistent.
    #
    #     Validation logic:
    #     If bounded random portfolios find a Sharpe within 0.05
    #     of the optimizer — optimizer is confirmed correct.
    #     If unbounded portfolios beat the optimizer — expected,
    #     they ignore your investment constraints entirely.
    # -----------------------------------------------------------

    print(f"\n{'─'*55}")
    print(f"OPTIMIZER VALIDATION")
    print(f"{'─'*55}")
    print(f"  Max Sharpe optimizer       : {port_sharpe:.4f}")
    print(f"  Min Var optimizer          : {minvar_sharpe:.4f}")
    print(f"  Best unbounded Sharpe      : {sim_sharpes.max():.4f}")

    if best_bnd_sh is not None:
        diff = abs(best_bnd_sh - port_sharpe)
        print(f"  Best bounded Sharpe        : {best_bnd_sh:.4f}")
        print(f"  Difference (bounded-optim) : {diff:.4f}")
        if diff < OPTIMIZER_VALIDATION_TOLERANCE:
            print(f"  ✓ Optimizer confirmed correct within bounded region.")
        else:
            print(f"  ⚠ Difference > {OPTIMIZER_VALIDATION_TOLERANCE} — "
                    f"consider increasing N_STARTS in Module 3.")
    else:
        print(f"  Visual validation only.")
        print(f"  ✓ If ★ sits at bounded dots cluster top — optimizer correct.")

    unbounded_gap = sim_sharpes.max() - port_sharpe
    print(f"\n  Unbounded beats optimizer by : {unbounded_gap:.4f} Sharpe")
    print(f"  This is expected — unbounded portfolios concentrate")
    print(f"  30-50% in one stock, violating investment constraints.")
    print(f"{'─'*55}")

    # -----------------------------------------------------------
    # 4E. Plot — four layer approach
    #
    #     Layer 1: Unbounded cloud (full opportunity set)
    #     Layer 2: Bounded region (your constrained space)
    #     Layer 3: Optimizer portfolio markers
    #     Layer 4: Capital Market Line
    #     Layer 5: Explanatory annotation
    # -----------------------------------------------------------

    fig, ax = plt.subplots(figsize=(14, 9))
    fig.patch.set_facecolor('#0f0f1a')
    ax.set_facecolor('#0f0f1a')

    # Layer 1 — unbounded cloud coloured by Sharpe
    scatter = ax.scatter(
        sim_vols    * 100,
        sim_returns * 100,
        c          = sim_sharpes,
        cmap       = 'RdYlGn',
        alpha      = 0.2,
        s          = 2,
        linewidths = 0,
        zorder     = 1,
        vmin       = 0,
        vmax       = sim_sharpes.max()
    )
    cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label('Sharpe Ratio (unbounded)', fontsize=10, color='white')
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')

    # Layer 2 — bounded region overlay
    if n_valid >= 10:
        ax.scatter(
            bounded_vol * 100,
            bounded_ret * 100,
            c          = bounded_sharpe,
            cmap       = 'cool',
            alpha      = 0.6,
            s          = 8,
            linewidths = 0,
            zorder     = 2,
            label      = (f'Bounded region '
                        f'({LOW*100:.1f}–{HIGH*100:.1f}% per stock, '
                        f'n={n_valid:,})')
        )

    # Layer 3 — optimizer portfolio markers
    ax.scatter(
        port_volatility * 100,
        port_return     * 100,
        color     = '#4fc3f7',
        s         = 350,
        zorder    = 6,
        marker    = '*',
        edgecolor = 'white',
        linewidth = 1.0,
        label     = (f'Max Sharpe  |  SR={port_sharpe:.2f}  '
                    f'R={port_return*100:.1f}%  '
                    f'V={port_volatility*100:.1f}%')
    )
    ax.scatter(
        minvar_volatility * 100,
        minvar_return     * 100,
        color     = '#81c784',
        s         = 220,
        zorder    = 6,
        marker    = 'D',
        edgecolor = 'white',
        linewidth = 1.0,
        label     = (f'Min Variance  |  SR={minvar_sharpe:.2f}  '
                    f'R={minvar_return*100:.1f}%  '
                    f'V={minvar_volatility*100:.1f}%')
    )

    # Layer 4 — Capital Market Line
    # y = rf*100 + port_sharpe * x — all in % units, consistent
    x_max = sim_vols.max() * 100 * 1.1
    x_cml = np.linspace(0, x_max, 300)
    y_cml = RISK_FREE * 100 + port_sharpe * x_cml

    ax.plot(
        x_cml, y_cml,
        color     = '#4fc3f7',
        linewidth = 1.2,
        linestyle = '--',
        alpha     = 0.6,
        zorder    = 4,
        label     = f'Capital Market Line  |  Slope={port_sharpe:.2f}'
    )
    ax.scatter(
        0, RISK_FREE * 100,
        color     = 'white',
        s         = 80,
        zorder    = 6,
        marker    = 'o',
        edgecolor = '#aaaaaa',
        linewidth = 0.8,
        label     = f'Risk-Free Rate ({RISK_FREE*100:.1f}%)'
    )

    # Layer 5 — annotations
    ax.annotate(
        'Max Sharpe\n(bounded optimizer)',
        xy         = (port_volatility * 100, port_return * 100),
        xytext     = (port_volatility * 100 - 4, port_return * 100 + 2),
        fontsize   = 8.5,
        color      = '#4fc3f7',
        arrowprops = dict(arrowstyle='->', color='#4fc3f7', lw=1.2)
    )
    ax.annotate(
        'Min Variance\n(bounded optimizer)',
        xy         = (minvar_volatility * 100, minvar_return * 100),
        xytext     = (minvar_volatility * 100 - 4, minvar_return * 100 - 3),
        fontsize   = 8.5,
        color      = '#81c784',
        arrowprops = dict(arrowstyle='->', color='#81c784', lw=1.2)
    )
    ax.text(
        0.98, 0.05,
        'Cloud  = unbounded portfolios (full opportunity set)\n'
        'Dots   = bounded portfolios within weight limits\n'
        '★ / ◆  = bounded optimizer solutions\n'
        'Star sitting inside cloud is expected —\n'
        'unbounded portfolios ignore weight constraints.',
        transform  = ax.transAxes,
        fontsize   = 7.5,
        color      = '#aaaaaa',
        ha         = 'right',
        va         = 'bottom',
        bbox       = dict(boxstyle='round', facecolor='#1a1a2e',
                        alpha=0.8, edgecolor='#444444')
    )

    # -----------------------------------------------------------
    # 4F. Formatting
    # -----------------------------------------------------------

    ax.set_title(
        f'Efficient Frontier  —  {N_PORTFOLIOS:,} Random Portfolios\n'
        f'Unbounded Cloud + Bounded Region  |  '
        f'Risk-Free={RISK_FREE*100:.1f}%  |  '
        f'Bounds={LOW*100:.1f}–{HIGH*100:.1f}%',
        fontsize   = 12,
        fontweight = 'bold',
        color      = 'white',
        pad        = 15
    )
    ax.set_xlabel('Annual Volatility (%)', fontsize=11, color='white')
    ax.set_ylabel('Annual Return (%)',     fontsize=11, color='white')
    ax.tick_params(colors='white')
    ax.spines['bottom'].set_color('#444444')
    ax.spines['left'].set_color('#444444')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.12, linewidth=0.5, color='white')
    ax.set_xlim(left=0)
    ax.legend(
        loc        = 'upper left',
        fontsize   = 8.5,
        framealpha = 0.3,
        facecolor  = '#1a1a2e',
        edgecolor  = '#444444',
        labelcolor = 'white'
    )

    plt.tight_layout()
    plt.savefig(str(output_dir / 'efficient_frontier.png'), dpi=150,
                bbox_inches='tight', facecolor='#0f0f1a')
    plt.show()

    print("""
    HOW TO READ THIS PLOT:
    ──────────────────────────────────────────────────────────
    Cloud        : Unbounded portfolios — full opportunity
                    set. Colour shows Sharpe (green=good).

    Cyan dots    : Bounded portfolios respecting your
                    weight constraints. This is the feasible
                    region your optimizer works within.

    ★ Max Sharpe : Should sit at TOP of the bounded dots
                    cluster. If inside unbounded cloud only
                    — increase N_STARTS in Module 3.

    ◆ Min Var    : Should sit at LEFT edge of bounded dots.
                    Lowest volatility within constraints.

    Dashed line  : Capital Market Line. Starts at risk-free
                    rate through Max Sharpe. Slope = Sharpe.
                    Steeper = better risk-adjusted return.

    KEY INSIGHT:
    The gap between ★ and unbounded cloud top = cost of your
    investment discipline. That cost is worth paying —
    concentration is how portfolios blow up, not compound.
    ──────────────────────────────────────────────────────────
    """)
    # ---- publish (generated) ----
    _scope = locals()
    ctx.update({k: _scope[k] for k in _PUBLISHED if k in _scope})

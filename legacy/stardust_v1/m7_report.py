"""
Legacy Stardust 1.x - M7 - Excel report generator

Mechanically extracted from stardust_core.py lines 4659-6161 by
legacy/tools/split_monolith.py. The code inside run() is the original,
byte-for-byte; only the input/publish scaffolding around it is generated.
Do not hand-edit the extracted block - fix bugs in STAR2.0 (star/) instead.
"""
from ._bootstrap import ensure_bootstrap

ensure_bootstrap()

#: data this stage reads from earlier stages (its upstream contract)
INPUTS = (
    '_cov_method',
    '_factor_r2',
    'active_label',
    'all_drawdowns',
    'arith_bl',
    'arith_equal',
    'arith_ewma',
    'arith_factor',
    'arith_returns',
    'BENCHMARK',
    'betas',
    'clean_tickers',
    'cov_annual',
    'cov_daily',
    'CVaR_95',
    'CVaR_hist_1day',
    'CVaR_hist_annual',
    'final_values',
    'GLOBAL_SEED',
    'INITIAL_VALUE',
    'log_return',
    'LOOKBACK_Y',
    'MAX_ASSETS',
    'minvar_return',
    'minvar_sharpe',
    'minvar_volatility',
    'minvar_weights',
    'N_SIMULATIONS',
    'N_STARTS',
    'optimal_weights',
    'output_dir',
    'Pi_s',
    'POLISH_FLOOR',
    'port_return',
    'port_sharpe',
    'port_volatility',
    'prob_loss',
    'R2_MIN_THRESHOLD',
    'REBALANCE_FREQ',
    'REBALANCE_PERIODS_PER_YEAR',
    'RETURN_CAP',
    'RETURN_METHOD',
    'RISK_FREE',
    'ROLLING_WINDOW_YRS',
    'scenario_results',
    'SHRINK_FACTOR',
    'status_callback',
    'T_DIST_DF',
    'TICKERS',
    'TRADING_DAYS',
    'TRANSACTION_COST',
    'VaR_95',
    'VaR_hist_1day',
    'VaR_hist_annual',
    'WEIGHT_MAX',
    'WEIGHT_MIN',
    'wf_port_returns',
    'wf_summary',
    'wf_turnover',
    'wf_window_stats',
)

#: legacy loop variables that leak in from earlier stages; pulled only so the
#: split behaves exactly like the monolith's single shared scope
CARRYOVER = ()

#: data this stage produces that a later stage reads (downstream contract)
OUTPUTS = ()

#: everything bound at stage scope; all of it is published to ctx
_PUBLISHED = ('BORD', 'CHART_FILES', 'C_ALT', 'C_BLACK', 'C_BLUE', 'C_BORDER', 'C_BRAND', 'C_ERR_BG', 'C_HDR_BG', 'C_HDR_FG', 'C_INFO', 'C_INFO_BG', 'C_NEG', 'C_OK_BG', 'C_POS', 'C_SUB_BG', 'C_WARN', 'C_WARN_BG', 'OUTPUT_XLSX', '_DIAG', '_get', '_log', '_report_context', 'build_betas', 'build_charts', 'build_covariance', 'build_diagnostics', 'build_excel', 'build_returns', 'build_risk', 'build_simulation', 'build_stress', 'build_summary', 'build_universe', 'build_walkforward', 'build_weights', 'col_w', 'ctr', 'data_cell', 'fill', 'header_row', 'hf', 'lft', 'pct', 'rgt', 'rupee', 'section_title', 'thin')


def run(ctx: dict) -> None:
    # ---- inputs from upstream stages (generated) ----
    _cov_method = ctx['_cov_method']
    _factor_r2 = ctx['_factor_r2']
    active_label = ctx['active_label']
    all_drawdowns = ctx['all_drawdowns']
    arith_bl = ctx['arith_bl']
    arith_equal = ctx['arith_equal']
    arith_ewma = ctx['arith_ewma']
    arith_factor = ctx['arith_factor']
    arith_returns = ctx['arith_returns']
    BENCHMARK = ctx['BENCHMARK']
    betas = ctx['betas']
    clean_tickers = ctx['clean_tickers']
    cov_annual = ctx['cov_annual']
    cov_daily = ctx['cov_daily']
    CVaR_95 = ctx['CVaR_95']
    CVaR_hist_1day = ctx['CVaR_hist_1day']
    CVaR_hist_annual = ctx['CVaR_hist_annual']
    final_values = ctx['final_values']
    GLOBAL_SEED = ctx['GLOBAL_SEED']
    INITIAL_VALUE = ctx['INITIAL_VALUE']
    log_return = ctx['log_return']
    LOOKBACK_Y = ctx['LOOKBACK_Y']
    MAX_ASSETS = ctx['MAX_ASSETS']
    minvar_return = ctx['minvar_return']
    minvar_sharpe = ctx['minvar_sharpe']
    minvar_volatility = ctx['minvar_volatility']
    minvar_weights = ctx['minvar_weights']
    N_SIMULATIONS = ctx['N_SIMULATIONS']
    N_STARTS = ctx['N_STARTS']
    optimal_weights = ctx['optimal_weights']
    output_dir = ctx['output_dir']
    Pi_s = ctx['Pi_s']
    POLISH_FLOOR = ctx['POLISH_FLOOR']
    port_return = ctx['port_return']
    port_sharpe = ctx['port_sharpe']
    port_volatility = ctx['port_volatility']
    prob_loss = ctx['prob_loss']
    R2_MIN_THRESHOLD = ctx['R2_MIN_THRESHOLD']
    REBALANCE_FREQ = ctx['REBALANCE_FREQ']
    REBALANCE_PERIODS_PER_YEAR = ctx['REBALANCE_PERIODS_PER_YEAR']
    RETURN_CAP = ctx['RETURN_CAP']
    RETURN_METHOD = ctx['RETURN_METHOD']
    RISK_FREE = ctx['RISK_FREE']
    ROLLING_WINDOW_YRS = ctx['ROLLING_WINDOW_YRS']
    scenario_results = ctx['scenario_results']
    SHRINK_FACTOR = ctx['SHRINK_FACTOR']
    status_callback = ctx['status_callback']
    T_DIST_DF = ctx['T_DIST_DF']
    TICKERS = ctx['TICKERS']
    TRADING_DAYS = ctx['TRADING_DAYS']
    TRANSACTION_COST = ctx['TRANSACTION_COST']
    VaR_95 = ctx['VaR_95']
    VaR_hist_1day = ctx['VaR_hist_1day']
    VaR_hist_annual = ctx['VaR_hist_annual']
    WEIGHT_MAX = ctx['WEIGHT_MAX']
    WEIGHT_MIN = ctx['WEIGHT_MIN']
    wf_port_returns = ctx['wf_port_returns']
    wf_summary = ctx['wf_summary']
    wf_turnover = ctx['wf_turnover']
    wf_window_stats = ctx['wf_window_stats']
    # ---- original code: stardust_core.py lines 4659-6161 ----
    # ============================================================
    # ============================================================
    # MODULE 7: Excel Report Generator  (Stardust 1.0 — Final)
    # ------------------------------------------------------------
    # Compatible with BOTH notebook versions:
    #   - Original  : WEIGHT_MIN defined, all stocks held
    #   - Sparse    : MAX_ASSETS defined, top N stocks held
    #
    # Run AFTER Modules 0–6 (Module 8 optional but before this).
    #
    # Sheets produced:
    #   1.  Summary            — key metrics at a glance + branding
    #   2.  Stock Universe     — tickers, returns, std dev (M1/M2)
    #   3.  Expected Returns   — all return methods compared (M2)
    #   4.  Weights            — Max Sharpe vs Min Variance (M3)
    #   5.  Correlation        — top-N correlation heatmap (M2)
    #   6.  Simulation         — Monte Carlo path stats (M5)
    #   7.  Risk Metrics       — VaR, CVaR, drawdown (M5/M6)
    #   8.  Stress Tests       — scenario outcomes (M6)
    #   9.  Betas              — stock betas vs benchmark (M6)
    #  10.  Charts             — all PNGs embedded (M4–M6)
    #  11.  Walk-Fwd Backtest  — OOS walk-forward results (M8)
    #  12.  Diagnostics        — pipeline log, warnings, health checks
    # ============================================================

    import os, datetime, shutil, platform, subprocess, pathlib, warnings
    import numpy as np
    import pandas as pd
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    from openpyxl.utils import get_column_letter
    from openpyxl.drawing.image import Image as XLImage

    status_callback("Module 7: Building report...")

    warnings.filterwarnings('ignore')

    OUTPUT_XLSX = str(output_dir / 'portfolio_report.xlsx')

    CHART_FILES = {
        'Efficient Frontier'     : str(output_dir / 'efficient_frontier.png'),
        'Monte Carlo Simulation' : str(output_dir / 'monte_carlo_simulation.png'),
        'Risk & Stress Testing'  : str(output_dir / 'risk_stress_testing.png'),
    }

    # ── Palette ───────────────────────────────────────────────────
    C_HDR_BG  = '1a237e'
    C_HDR_FG  = 'FFFFFF'
    C_SUB_BG  = 'E8EAF6'
    C_ALT     = 'F5F5F5'
    C_POS     = '1B5E20'
    C_NEG     = 'B71C1C'
    C_BORDER  = 'B0BEC5'
    C_BLUE    = '0000FF'
    C_BLACK   = '000000'
    C_BRAND   = '4A148C'
    C_WARN    = 'E65100'   # amber — warnings
    C_INFO    = '01579B'   # steel blue — info
    C_OK_BG   = 'E8F5E9'   # light green background — ok rows
    C_WARN_BG = 'FFF3E0'   # light amber background — warn rows
    C_ERR_BG  = 'FFEBEE'   # light red background — error rows
    C_INFO_BG = 'E3F2FD'   # light blue background — info rows

    thin = Side(style='thin', color=C_BORDER)
    BORD = Border(left=thin, right=thin, top=thin, bottom=thin)

    # ── Diagnostics log ───────────────────────────────────────────
    # All build functions append here.
    # Sheet 12 reads this list at the end.
    _DIAG = []

    def _log(module, message, status='ok'):
        """
        Append a diagnostic entry.
        status: 'ok' | 'info' | 'warn' | 'error'
        """
        _DIAG.append({
            'module'    : module,
            'message'   : message,
            'status'    : status,
            'timestamp' : datetime.datetime.now().strftime('%H:%M:%S'),
        })

    # ── Style helpers ──────────────────────────────────────────────
    def hf(size=9, bold=False, color=C_BLACK):
        return Font(name='Arial', size=size, bold=bold, color=color)

    def fill(color):
        return PatternFill('solid', fgColor=color)

    def ctr():
        return Alignment(horizontal='center', vertical='center', wrap_text=True)

    def lft():
        return Alignment(horizontal='left', vertical='center')

    def rgt():
        return Alignment(horizontal='right', vertical='center')

    def col_w(ws, col, w):
        ws.column_dimensions[get_column_letter(col)].width = w

    def header_row(ws, row, headers, col_widths=None):
        for c, h in enumerate(headers, 1):
            cell           = ws.cell(row, c)
            cell.value     = h
            cell.font      = Font(name='Arial', size=9, bold=True, color=C_HDR_FG)
            cell.fill      = fill(C_HDR_BG)
            cell.alignment = ctr()
            cell.border    = BORD
        if col_widths:
            for c, w in enumerate(col_widths, 1):
                col_w(ws, c, w)

    def data_cell(ws, row, col, value, bold=False, color=C_BLACK,
                alt=False, align='left', num_fmt=None, bg=None):
        cell           = ws.cell(row, col)
        cell.value     = value
        cell.font      = Font(name='Arial', size=9, bold=bold, color=color)
        cell.fill      = fill(bg) if bg else (fill(C_ALT) if alt else PatternFill())
        cell.alignment = ctr() if align == 'center' else (rgt() if align == 'right' else lft())
        cell.border    = BORD
        if num_fmt:
            cell.number_format = num_fmt
        return cell

    def section_title(ws, row, text, ncols):
        ws.merge_cells(start_row=row, start_column=1,
                    end_row=row, end_column=ncols)
        c           = ws.cell(row, 1)
        c.value     = text
        c.font      = Font(name='Arial', size=10, bold=True, color=C_HDR_BG)
        c.fill      = fill(C_SUB_BG)
        c.alignment = lft()
        c.border    = BORD
        ws.row_dimensions[row].height = 16

    def rupee(v):
        return f'Rs.{v:,.0f}'

    def pct(v, decimals=2, sign=False):
        s = '+' if (sign and v >= 0) else ''
        return f'{s}{v:.{decimals}f}%'

    def _get(name, default=None):
        return _report_context.get(name, default)


    # ============================================================
    # SHEET 1 — Summary
    # ============================================================
    _report_context = {
                'INITIAL_VALUE'      : INITIAL_VALUE,
                'RISK_FREE'          : RISK_FREE,
                'RETURN_METHOD'      : RETURN_METHOD,
                'SHRINK_FACTOR'      : SHRINK_FACTOR,
                'T_DIST_DF'          : T_DIST_DF,
                'LOOKBACK_Y'         : LOOKBACK_Y,
                'BENCHMARK'          : BENCHMARK,
                'MAX_ASSETS'         : MAX_ASSETS,
                'WEIGHT_MIN'         : WEIGHT_MIN,
                'WEIGHT_MAX'         : WEIGHT_MAX,
                'N_STARTS'           : N_STARTS,
                'N_SIMULATIONS'      : N_SIMULATIONS,
                'GLOBAL_SEED'        : GLOBAL_SEED,
                'TRADING_DAYS'       : TRADING_DAYS,
                'REBALANCE_FREQ'     : REBALANCE_FREQ,
                'TRANSACTION_COST'   : TRANSACTION_COST,
                'ROLLING_WINDOW_YRS' : ROLLING_WINDOW_YRS,
                'RETURN_CAP'         : RETURN_CAP,
                'POLISH_FLOOR'       : POLISH_FLOOR,
                # module outputs
                'port_return'        : port_return,
                'port_volatility'    : port_volatility,
                'port_sharpe'        : port_sharpe,
                'minvar_return'      : minvar_return,
                'minvar_volatility'  : minvar_volatility,
                'minvar_sharpe'      : minvar_sharpe,
                'VaR_95'             : VaR_95,
                'CVaR_95'            : CVaR_95,
                'VaR_hist_1day'      : VaR_hist_1day,
                'CVaR_hist_1day'     : CVaR_hist_1day,
                'VaR_hist_annual'    : VaR_hist_annual,
                'CVaR_hist_annual'   : CVaR_hist_annual,
                'prob_loss'          : prob_loss,
                'final_values'       : final_values,
                'all_drawdowns'      : all_drawdowns,
                'scenario_results'   : scenario_results,
                'betas'              : betas,
                'N_SIMULATIONS'      : N_SIMULATIONS,
                '_cov_method'        : _cov_method,
                'active_label'       : active_label,
                '_factor_r2'         : _factor_r2,
                'arith_equal'        : arith_equal,
                'arith_ewma'         : arith_ewma,
                'arith_factor'       : arith_factor,
                'arith_bl'           : arith_bl,
                'TICKERS'            : TICKERS,
                'wf_summary'         : wf_summary,
                'wf_window_stats'    : wf_window_stats,
                'wf_turnover'        : wf_turnover,
                'wf_port_returns'    : wf_port_returns,
                'Pi_s'               : Pi_s,
            }

    def build_summary(wb):
        ws        = wb.active
        ws.title  = '1. Summary'
        ws.sheet_view.showGridLines = False

        today     = datetime.date.today().strftime('%d %B %Y')
        RISK_FREE = _get('RISK_FREE', 0.065)
        INIT      = _get('INITIAL_VALUE', 1_000_000)
        n         = len(arith_returns)
        is_sparse = _get('MAX_ASSETS') is not None
        version_tag = (f'Sparse (MAX_ASSETS={_get("MAX_ASSETS")})'
                    if is_sparse
                    else f'Dense floor (WEIGHT_MIN={_get("WEIGHT_MIN",0)*100:.2f}%)')

        # ── Branding cell ────────────────────────────────────────
        ws.merge_cells('E1:G1')
        brand           = ws['E1']
        brand.value     = 'Stardust 1.0 — designed by Yash Parikh'
        brand.font      = Font(name='Arial', size=9, bold=True,
                            italic=True, color=C_HDR_FG)
        brand.fill      = fill(C_BRAND)
        brand.alignment = ctr()
        brand.border    = BORD

        # ── Title ────────────────────────────────────────────────
        ws.merge_cells('A1:D1')
        c           = ws['A1']
        c.value     = 'PORTFOLIO ANALYSIS REPORT'
        c.font      = Font(name='Arial', size=16, bold=True, color=C_HDR_BG)
        c.alignment = ctr()
        c.fill      = fill(C_SUB_BG)
        ws.row_dimensions[1].height = 32

        ws.merge_cells('A2:G2')
        c       = ws['A2']
        c.value = (f'Indian Equity Portfolio — NSE | '
                f'Generated: {today} | Optimizer: {version_tag}')
        c.font      = Font(name='Arial', size=8, color='546e7a')
        c.alignment = ctr()

        # ── Parameters block ─────────────────────────────────────
        r = 4
        section_title(ws, r, 'Portfolio Parameters', 4); r += 1
        header_row(ws, r, ['Parameter', 'Value', 'Notes', ''],
                [28, 22, 30, 10]); r += 1

        date_from = log_return.index[0].strftime('%d %b %Y')
        date_to   = log_return.index[-1].strftime('%d %b %Y')

        params = [
            ('Universe size',      f'{n} stocks',
            'Passed Module 1 validation'),
            ('Data range',         f'{date_from} to {date_to}',
            '5-year lookback'),
            ('Trading days',       f'{len(log_return):,}',
            'NSE trading days'),
            ('Initial investment', rupee(INIT),
            'Used in VaR/stress calc'),
            ('Risk-free rate',     pct(RISK_FREE*100),
            'RBI 10Y Gsec proxy'),
            ('Return method',      str(_get('RETURN_METHOD','N/A')),
            'From Module 0'),
            ('Optimizer version',  version_tag, ''),
            ('Covariance method',  str(_get('_cov_method','N/A')),
            'From Module 2'),
            ('Simulation paths',   f"{_get('N_SIMULATIONS',0):,}",
            'Module 5'),
        ]
        for i, (k, v, note) in enumerate(params):
            alt = i % 2 == 0
            data_cell(ws, r, 1, k,    alt=alt)
            data_cell(ws, r, 2, v,    alt=alt, color=C_BLUE)
            data_cell(ws, r, 3, note, alt=alt, color='546e7a')
            r += 1

        # ── Key Metrics ──────────────────────────────────────────
        r += 1
        section_title(ws, r, 'Key Portfolio Metrics', 4); r += 1
        header_row(ws, r, ['Metric', 'Max Sharpe',
                            'Min Variance', 'Better']); r += 1

        pr   = _get('port_return',       0)
        pv   = _get('port_volatility',   0)
        ps   = _get('port_sharpe',       0)
        mr   = _get('minvar_return',     0)
        mv_v = _get('minvar_volatility', 0)
        ms_s = _get('minvar_sharpe',     0)

        metrics = [
            ('Expected Return p.a.',
            pct(pr*100, sign=True), pct(mr*100, sign=True),
            'Max Sharpe' if pr >= mr else 'Min Var'),
            ('Annual Volatility',
            pct(pv*100), pct(mv_v*100),
            'Min Var' if mv_v <= pv else 'Max Sharpe'),
            ('Sharpe Ratio',
            f'{ps:.4f}', f'{ms_s:.4f}',
            'Max Sharpe' if ps >= ms_s else 'Min Var'),
            ('Stocks held',
            str((optimal_weights > 0.001).sum()),
            str((minvar_weights  > 0.001).sum()), '—'),
        ]
        for i, (m, ms_v, mv_val, better) in enumerate(metrics):
            alt = i % 2 == 0
            data_cell(ws, r, 1, m,      alt=alt)
            data_cell(ws, r, 2, ms_v,   alt=alt, color=C_POS)
            data_cell(ws, r, 3, mv_val, alt=alt, color=C_POS)
            data_cell(ws, r, 4, better, alt=alt, align='center')
            r += 1

        # ── Risk Snapshot ────────────────────────────────────────
        r += 1
        section_title(ws, r, 'Risk Snapshot', 4); r += 1
        header_row(ws, r, ['Risk Metric', 'Value (Rs.)',
                            'Value (%)', 'Source']); r += 1

        risk_rows = [
            ('Simulation VaR 95% (1-day)',
            rupee(_get('VaR_95', 0)),
            pct(_get('VaR_95',0)/INIT*100, 3), 'Module 5'),
            ('Simulation CVaR 95% (1-day)',
            rupee(_get('CVaR_95', 0)),
            pct(_get('CVaR_95',0)/INIT*100, 3), 'Module 5'),
            ('Historical VaR 95% (1-day)',
            rupee(_get('VaR_hist_1day', 0)),
            pct(_get('VaR_hist_1day',0)/INIT*100, 3),
            'Module 6 — fat tails'),
            ('Historical CVaR 95% (1-day)',
            rupee(_get('CVaR_hist_1day', 0)),
            pct(_get('CVaR_hist_1day',0)/INIT*100, 3), 'Module 6'),
            ('Annual VaR 95% (simulation)',
            rupee(_get('VaR_hist_annual', 0)),
            pct(_get('VaR_hist_annual',0)/INIT*100, 2), 'Module 6'),
            ('Prob. of loss over 1 year',
            pct(_get('prob_loss', 0)), '—', 'Module 5'),
        ]
        for i, row_data in enumerate(risk_rows):
            alt = i % 2 == 0
            for c, v in enumerate(row_data, 1):
                data_cell(ws, r, c, v, alt=alt,
                        color=C_NEG if c in [2, 3] else C_BLACK)
            r += 1

        # ── Disclaimer ───────────────────────────────────────────
        r += 1
        ws.merge_cells(start_row=r, start_column=1, end_row=r, end_column=7)
        d       = ws.cell(r, 1)
        d.value = ('DISCLAIMER: For analytical purposes only. '
                'Not investment advice. Past performance does not '
                'guarantee future results. All figures based on '
                'historical data and Monte Carlo simulation.')
        d.font      = Font(name='Arial', size=8, italic=True, color='78909c')
        d.alignment = Alignment(wrap_text=True)
        ws.row_dimensions[r].height = 24

        _log('M7 Report', 'Sheet 1 — Summary written', 'ok')


    # ============================================================
    # SHEET 2 — Stock Universe
    # ============================================================
    def build_universe(wb):
        ws = wb.create_sheet('2. Stock Universe')
        ws.sheet_view.showGridLines = False

        headers = ['#', 'Ticker', 'Short Name', 'Ann. Return (%)',
                'R² (factor)', 'Daily Std Dev (%)', 'Ann. Std Dev (%)']
        header_row(ws, 1, headers, [4, 18, 15, 16, 12, 16, 16])

        factor_r2   = _get('_factor_r2', {})
        low_r2      = []

        for i, ticker in enumerate(clean_tickers, 1):
            r     = i + 1
            ret   = arith_returns.get(ticker, float('nan'))
            var   = (cov_daily.loc[ticker, ticker]
                    if ticker in cov_daily.index else float('nan'))
            std_d = np.sqrt(var) * 100   if not np.isnan(var) else float('nan')
            std_a = std_d * np.sqrt(TRADING_DAYS) if not np.isnan(std_d) else float('nan')
            r2    = factor_r2.get(ticker, float('nan'))
            alt   = i % 2 == 0

            if not np.isnan(r2) and r2 < R2_MIN_THRESHOLD:
                low_r2.append(ticker)

            data_cell(ws, r, 1, i,                        alt=alt, align='center')
            data_cell(ws, r, 2, ticker,                   alt=alt)
            data_cell(ws, r, 3, ticker.replace('.NS',''), alt=alt)
            data_cell(ws, r, 4,
                    pct(ret*100, sign=True) if not np.isnan(ret) else 'N/A',
                    alt=alt,
                    color=C_POS if (not np.isnan(ret) and ret >= 0) else C_NEG)
            data_cell(ws, r, 5,
                    f'{r2:.3f}' if not np.isnan(r2) else 'N/A',
                    alt=alt,
                    color=C_NEG if (not np.isnan(r2) and r2 < R2_MIN_THRESHOLD) else C_BLACK)
            data_cell(ws, r, 6,
                    f'{std_d:.3f}%' if not np.isnan(std_d) else 'N/A', alt=alt)
            data_cell(ws, r, 7,
                    f'{std_a:.2f}%' if not np.isnan(std_a) else 'N/A', alt=alt)

        nr = len(clean_tickers) + 3
        ws.merge_cells(start_row=nr, start_column=1, end_row=nr, end_column=7)
        c       = ws.cell(nr, 1)
        c.value = ('Note: R² < 0.15 (red) = return mostly idiosyncratic, '
                'factor estimate unreliable.')
        c.font  = Font(name='Arial', size=8, italic=True, color='78909c')

        _log('M7 Report', f'Sheet 2 — {len(clean_tickers)} tickers written', 'ok')
        if low_r2:
            _log('M2 Covariance',
                f'{len(low_r2)} tickers with R² < 0.15: {", ".join(low_r2)}',
                'warn')
        else:
            _log('M2 Covariance', 'All tickers R² ≥ 0.15 — factor fit acceptable', 'ok')


    # ============================================================
    # SHEET 3 — Expected Returns
    # ============================================================
    def build_returns(wb):
        ws = wb.create_sheet('3. Expected Returns')
        ws.sheet_view.showGridLines = False

        return_series = {'Final (active)': arith_returns}
        for name, var in [('Equal-weight hist', 'arith_equal'),
                        ('EWMA hist',         'arith_ewma'),
                        ('Factor model',      'arith_factor'),
                        ('Black-Litterman',   'arith_bl')]:
            v = _get(var)
            if v is not None:
                return_series[name] = v

        headers    = ['Ticker', 'Short Name'] + list(return_series.keys())
        col_widths = [18, 14] + [16] * len(return_series)
        header_row(ws, 1, headers, col_widths)

        active_label = _get('active_label', 'Final (active)')

        for i, ticker in enumerate(arith_returns.index, 1):
            r   = i + 1
            alt = i % 2 == 0
            data_cell(ws, r, 1, ticker,                   alt=alt)
            data_cell(ws, r, 2, ticker.replace('.NS',''), alt=alt)
            for c, (name, series) in enumerate(return_series.items(), 3):
                val       = series.get(ticker, float('nan'))
                is_active = (name == 'Final (active)' or name == active_label)
                data_cell(ws, r, c,
                        pct(val*100, sign=True) if not np.isnan(val) else 'N/A',
                        alt=alt, bold=is_active,
                        color=C_POS if (not np.isnan(val) and val >= 0) else C_NEG)

        nr = len(arith_returns) + 3
        ws.merge_cells(start_row=nr, start_column=1,
                    end_row=nr, end_column=len(headers))
        c       = ws.cell(nr, 1)
        c.value = (f'Active method: {active_label}  |  '
                f'All returns after James-Stein shrinkage.')
        c.font  = Font(name='Arial', size=8, italic=True, color='78909c')

        _log('M7 Report',
            f'Sheet 3 — Expected returns written ({len(return_series)} methods)', 'ok')
        _log('M2 Covariance',
            f'Active return method: {active_label}', 'info')


    # ============================================================
    # SHEET 4 — Weights
    # ============================================================
    def build_weights(wb):
        ws = wb.create_sheet('4. Weights')
        ws.sheet_view.showGridLines = False

        is_sparse = _get('MAX_ASSETS') is not None
        RISK_FREE = _get('RISK_FREE', 0.065)
        pr   = _get('port_return',       0)
        pv   = _get('port_volatility',   0)
        ps   = _get('port_sharpe',       0)
        mr   = _get('minvar_return',     0)
        mv_v = _get('minvar_volatility', 0)
        ms_s = _get('minvar_sharpe',     0)

        section_title(ws, 1, 'Optimizer Metrics — Module 3', 5)
        header_row(ws, 2,
                ['Metric', 'Max Sharpe', 'Min Variance', 'Notes', ''],
                [26, 18, 18, 32, 8])

        n_held_ms = int((optimal_weights > 0.001).sum())
        n_held_mv = int((minvar_weights  > 0.001).sum())

        opt_rows = [
            ('Expected Return p.a.',
            pct(pr*100, sign=True), pct(mr*100, sign=True),
            'After James-Stein shrinkage'),
            ('Annual Volatility',
            pct(pv*100), pct(mv_v*100), 'sqrt(w^T Sigma w)'),
            ('Sharpe Ratio',
            f'{ps:.4f}', f'{ms_s:.4f}',
            f'Risk-free = {RISK_FREE*100:.1f}%'),
            ('Stocks held',
            str(n_held_ms), str(n_held_mv), 'Stocks with w > 0.1%'),
            ('Optimizer',
            'SLSQP multi-start', 'SLSQP multi-start',
            f'{_get("N_STARTS","?")} starts per phase'),
            ('Weight cap',
            pct(_get('WEIGHT_MAX',0.40)*100),
            pct(_get('WEIGHT_MAX',0.40)*100), 'Concentration limit'),
            ('Weight floor',
            '0.00% (sparse)' if is_sparse else pct(_get('WEIGHT_MIN',0)*100),
            '0.00% (sparse)' if is_sparse else pct(_get('WEIGHT_MIN',0)*100),
            'Phase 1 floor' if is_sparse else 'Dynamic'),
        ]
        for i, (m, ms_v, mv_val, note) in enumerate(opt_rows):
            alt = i % 2 == 0
            data_cell(ws, 3+i, 1, m,      alt=alt)
            data_cell(ws, 3+i, 2, ms_v,   alt=alt, color=C_BLUE)
            data_cell(ws, 3+i, 3, mv_val, alt=alt, color=C_BLUE)
            data_cell(ws, 3+i, 4, note,   alt=alt, color='546e7a')

        r = 12
        section_title(ws, r, 'Weight Allocation — All Stocks', 5); r += 1
        header_row(ws, r,
                ['Ticker', 'Short Name', 'Max Sharpe (%)',
                    'Min Var (%)', 'Difference (%)']); r += 1

        sorted_idx = optimal_weights.sort_values(ascending=False).index
        for i, ticker in enumerate(sorted_idx):
            ms_w     = optimal_weights[ticker] * 100
            mv_w     = minvar_weights[ticker]  * 100
            diff     = ms_w - mv_w
            alt      = i % 2 == 0
            excluded = ms_w < 0.1

            data_cell(ws, r, 1, ticker,
                    alt=alt, color='78909c' if excluded else C_BLACK)
            data_cell(ws, r, 2, ticker.replace('.NS',''),
                    alt=alt, color='78909c' if excluded else C_BLACK)
            data_cell(ws, r, 3, pct(ms_w), alt=alt,
                    color='78909c' if excluded else (C_POS if ms_w > 0.1 else C_BLACK))
            data_cell(ws, r, 4, pct(mv_w), alt=alt,
                    color='78909c' if mv_w < 0.1 else C_POS)
            data_cell(ws, r, 5, pct(diff, sign=True), alt=alt,
                    color=C_POS if diff > 0.5 else (C_NEG if diff < -0.5 else C_BLACK))
            r += 1

        if is_sparse:
            ws.cell(r+1, 1).value = ('Grey rows = excluded by '
                                    'sparse optimizer (weight = 0)')
            ws.cell(r+1, 1).font  = Font(name='Arial', size=8,
                                        italic=True, color='78909c')

        _log('M7 Report', 'Sheet 4 — Weights written', 'ok')
        _log('M3 Optimizer',
            f'Max Sharpe: {n_held_ms} stocks held  |  '
            f'Sharpe = {ps:.4f}  |  Return = {pct(pr*100, sign=True)}', 'info')
        _log('M3 Optimizer',
            f'Min Variance: {n_held_mv} stocks held  |  '
            f'Sharpe = {ms_s:.4f}  |  Vol = {pct(mv_v*100)}', 'info')
        _log('M3 Optimizer', 'Safe to proceed to Module 4', 'ok')


    # ============================================================
    # SHEET 5 — Correlation
    # ============================================================
    def build_covariance(wb):
        ws = wb.create_sheet('5. Correlation')
        ws.sheet_view.showGridLines = False

        held = optimal_weights[optimal_weights > 0.001].index.tolist()
        N    = min(len(held), 30)
        held = held[:N]

        section_title(ws, 1,
                    f'Correlation Matrix — Top {N} Max Sharpe stocks', N+2)

        ws.cell(2, 1).value  = 'Ticker'
        ws.cell(2, 1).font   = Font(name='Arial', size=8, bold=True, color=C_HDR_FG)
        ws.cell(2, 1).fill   = fill(C_HDR_BG)
        ws.cell(2, 1).border = BORD
        col_w(ws, 1, 14)

        for c, t in enumerate(held, 2):
            cell           = ws.cell(2, c)
            cell.value     = t.replace('.NS','')
            cell.font      = Font(name='Arial', size=7, bold=True, color=C_HDR_FG)
            cell.fill      = fill(C_HDR_BG)
            cell.alignment = ctr()
            cell.border    = BORD
            col_w(ws, c, 8)

        cov_sub = cov_annual.loc[held, held]
        std_vec = np.sqrt(np.diag(cov_sub.values))
        corr    = cov_sub.values / np.outer(std_vec, std_vec)

        high_corr_pairs = []
        for i, ticker in enumerate(held):
            r    = i + 3
            cell = ws.cell(r, 1)
            cell.value     = ticker.replace('.NS','')
            cell.font      = Font(name='Arial', size=8, bold=True, color=C_HDR_FG)
            cell.fill      = fill(C_HDR_BG)
            cell.border    = BORD
            ws.row_dimensions[r].height = 14

            for j in range(len(held)):
                c    = j + 2
                val  = corr[i, j]
                cell = ws.cell(r, c)
                cell.value         = round(val, 2)
                cell.number_format = '0.00'
                cell.alignment     = ctr()
                cell.border        = BORD
                cell.font          = Font(name='Arial', size=7)
                if i == j:
                    cell.fill = fill('FFFFFF')
                elif val > 0.7:
                    cell.fill = fill('FFCDD2')
                    if j > i:
                        high_corr_pairs.append(
                            f'{held[i].replace(".NS","")}/'
                            f'{held[j].replace(".NS","")} ({val:.2f})')
                elif val > 0.4:
                    cell.fill = fill('FFF9C4')
                else:
                    cell.fill = fill('C8E6C9')

        _log('M7 Report', f'Sheet 5 — Correlation matrix written ({N}x{N})', 'ok')
        if high_corr_pairs:
            _log('M2 Covariance',
                f'{len(high_corr_pairs)} high-corr pairs (>0.70): '
                f'{", ".join(high_corr_pairs[:5])}'
                f'{"..." if len(high_corr_pairs) > 5 else ""}',
                'warn')
        else:
            _log('M2 Covariance',
                'No highly correlated pairs (>0.70) in held portfolio', 'ok')


    # ============================================================
    # SHEET 6 — Monte Carlo Simulation
    # ============================================================
    def build_simulation(wb):
        ws = wb.create_sheet('6. Simulation')
        ws.sheet_view.showGridLines = False

        INIT = _get('INITIAL_VALUE', 1_000_000)
        fv   = _get('final_values', np.array([INIT]))
        N    = _get('N_SIMULATIONS', len(fv))
        df   = _get('T_DIST_DF', 6)
        seed = _get('GLOBAL_SEED', 42)

        section_title(ws, 1, 'Monte Carlo Simulation — Module 5', 4)
        header_row(ws, 2, ['Metric', 'Value', 'Rs. Amount', 'Notes'],
                [32, 18, 20, 28])

        pcts  = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        stats = [
            ('Simulation paths',          f'{N:,}',           '—',
            f'Cholesky {"t-dist" if df != float("inf") else "Normal"} GBM'),
            ('Distribution df',
            'Normal' if df == float('inf') else str(df), '—',
            'Student-t degrees of freedom'),
            ('Random seed',               str(seed),          '—', ''),
            ('Initial investment',        '—',                rupee(INIT), ''),
            ('Median final value',        '—',                rupee(np.median(fv)),
            'Central outcome'),
            ('Mean final value',          '—',                rupee(np.mean(fv)), ''),
            ('Std dev of final values',   '—',                rupee(np.std(fv)),  ''),
            ('Prob. of loss (< initial)',
            pct(_get('prob_loss',0)), '—', 'Module 5'),
            ('Worst path (min)',           '—',                rupee(fv.min()),
            'Tail scenario'),
            ('Best path (max)',            '—',                rupee(fv.max()), ''),
        ]
        for p in pcts:
            val = np.percentile(fv, p)
            stats.append((f'{p}th percentile', '—', rupee(val),
                        'LOSS' if val < INIT else 'GAIN'))

        for i, (m, v, rs, note) in enumerate(stats):
            alt     = i % 2 == 0
            is_loss = (rs != '—' and 'Rs.' in rs and
                    float(rs.replace('Rs.','').replace(',','')) < INIT)
            data_cell(ws, 3+i, 1, m,    alt=alt)
            data_cell(ws, 3+i, 2, v,    alt=alt, color=C_BLUE)
            data_cell(ws, 3+i, 3, rs,   alt=alt,
                    color=C_NEG if is_loss else C_POS)
            data_cell(ws, 3+i, 4, note, alt=alt, color='546e7a')

        prob = _get('prob_loss', 0)
        _log('M7 Report', 'Sheet 6 — Simulation stats written', 'ok')
        _log('M5 Simulation',
            f'{N:,} paths completed  |  '
            f'Seed: {seed}  |  '
            f'Dist: {"Normal" if df == float("inf") else f"t-dist df={df}"}',
            'info')
        _log('M5 Simulation',
            f'Prob of loss: {pct(prob)}  |  '
            f'Median final: {rupee(np.median(fv))}',
            'info' if prob < 0.30 else 'warn')
        _log('M5 Simulation', 'Safe to proceed to Module 6', 'ok')


    # ============================================================
    # SHEET 7 — Risk Metrics
    # ============================================================
    def build_risk(wb):
        ws = wb.create_sheet('7. Risk Metrics')
        ws.sheet_view.showGridLines = False

        INIT  = _get('INITIAL_VALUE', 1_000_000)
        var5  = _get('VaR_95',           0)
        cvar5 = _get('CVaR_95',          0)
        hv1   = _get('VaR_hist_1day',    0)
        hc1   = _get('CVaR_hist_1day',   0)
        hva   = _get('VaR_hist_annual',  0)
        hca   = _get('CVaR_hist_annual', 0)

        section_title(ws, 1, 'VaR & CVaR — Modules 5 & 6', 5)
        header_row(ws, 2,
                ['Metric', 'Rs. Amount', 'As % of Portfolio',
                    'Annual Equivalent', 'Source'],
                [32, 18, 20, 22, 26])

        rows = [
            ('Simulation VaR 95% (1-day)',
            var5,  var5/INIT*100,  var5*np.sqrt(TRADING_DAYS),
            'Module 5 GBM paths'),
            ('Simulation CVaR 95% (1-day)',
            cvar5, cvar5/INIT*100, cvar5*np.sqrt(TRADING_DAYS),
            'Module 5 expected shortfall'),
            ('Historical VaR 95% (1-day)',
            hv1,   hv1/INIT*100,   hv1*np.sqrt(TRADING_DAYS),
            'Module 6 — actual returns, fat tails'),
            ('Historical CVaR 95% (1-day)',
            hc1,   hc1/INIT*100,   hc1*np.sqrt(TRADING_DAYS),
            'Module 6 expected shortfall'),
            ('Annual VaR 95% (simulation)',
            hva,   hva/INIT*100,   hva,
            'Module 6 — from path distribution'),
            ('Annual CVaR 95% (simulation)',
            hca,   hca/INIT*100,   hca, 'Module 6'),
        ]
        for i, (m, rs, pct_v, ann, src) in enumerate(rows):
            alt = i % 2 == 0
            data_cell(ws, 3+i, 1, m,              alt=alt)
            data_cell(ws, 3+i, 2, rupee(rs),      alt=alt, color=C_NEG)
            data_cell(ws, 3+i, 3, pct(pct_v, 3), alt=alt, color=C_NEG)
            data_cell(ws, 3+i, 4, rupee(ann),     alt=alt, color=C_NEG)
            data_cell(ws, 3+i, 5, src,            alt=alt, color='546e7a')

        r = 11
        section_title(ws, r, 'Drawdown Analysis — Module 6', 5); r += 1
        header_row(ws, r, ['Metric', 'Value', '', '', '']); r += 1

        all_dd = _get('all_drawdowns', np.array([0]))
        dd_rows = [
            ('Median max drawdown',      f'{np.median(all_dd)*100:.2f}%'),
            ('Worst max drawdown (5th)', f'{np.percentile(all_dd,5)*100:.2f}%'),
            ('Best max drawdown (95th)', f'{np.percentile(all_dd,95)*100:.2f}%'),
            ('Mean max drawdown',        f'{np.mean(all_dd)*100:.2f}%'),
        ]
        for i, (m, v) in enumerate(dd_rows):
            alt = i % 2 == 0
            data_cell(ws, r, 1, m, alt=alt)
            data_cell(ws, r, 2, v, alt=alt, color=C_NEG)
            r += 1

        ws.column_dimensions['A'].width = 32
        ws.column_dimensions['B'].width = 18

        _log('M7 Report', 'Sheet 7 — Risk metrics written', 'ok')
        _log('M6 Risk',
            f'VaR 95% (sim, 1-day): {rupee(var5)}  |  '
            f'CVaR 95% (sim, 1-day): {rupee(cvar5)}',
            'info')
        _log('M6 Risk',
            f'Historical VaR (1-day): {rupee(hv1)}  |  '
            f'Annual VaR (sim): {rupee(hva)}',
            'info')
        _log('M6 Risk', 'Safe to proceed to Module 8 / Module 7', 'ok')


    # ============================================================
    # SHEET 8 — Stress Tests
    # ============================================================
    def build_stress(wb):
        ws = wb.create_sheet('8. Stress Tests')
        ws.sheet_view.showGridLines = False

        section_title(ws, 1, 'Stress Test Scenarios — Module 6', 6)
        header_row(ws, 2,
                ['Scenario', 'Market Shock', 'Portfolio Loss (%)',
                    'Portfolio Loss (Rs.)', 'Final Value (Rs.)', 'Method'],
                [30, 14, 20, 22, 22, 20])

        sc = _get('scenario_results', {})
        if sc:
            for i, (name, res) in enumerate(sc.items()):
                alt  = i % 2 == 0
                r    = i + 3
                loss = res.get('portfolio_loss', 0)
                data_cell(ws, r, 1, name, alt=alt)
                data_cell(ws, r, 2, pct(res.get('market_drop',0)),
                        alt=alt, color=C_NEG)
                data_cell(ws, r, 3, pct(loss), alt=alt, color=C_NEG)
                data_cell(ws, r, 4, rupee(res.get('loss_rupees',0)),
                        alt=alt, color=C_NEG)
                data_cell(ws, r, 5, rupee(res.get('final_value', 0)),
                        alt=alt,
                        color=C_NEG
                        if res.get('final_value',0) < _get('INITIAL_VALUE',1e6)
                        else C_POS)
                data_cell(ws, r, 6, res.get('method','—'),
                        alt=alt, color='546e7a')
            _log('M7 Report',
                f'Sheet 8 — {len(sc)} stress scenarios written', 'ok')
            _log('M6 Risk',
                f'Scenarios run: {", ".join(sc.keys())}', 'info')
        else:
            ws.cell(3, 1).value = ('Run Module 6 first to populate '
                                    'stress test results.')
            ws.cell(3, 1).font  = Font(name='Arial', size=9,
                                        italic=True, color='78909c')
            _log('M6 Risk', 'No stress scenario results found — run Module 6', 'warn')


    # ============================================================
    # SHEET 9 — Betas
    # ============================================================
    def build_betas(wb):
        ws = wb.create_sheet('9. Betas')
        ws.sheet_view.showGridLines = False

        section_title(ws, 1, 'Stock Betas vs Benchmark — Module 6', 5)
        header_row(ws, 2,
                ['Ticker', 'Short Name', 'Beta',
                    'Max Sharpe Weight', 'Interpretation'],
                [18, 15, 10, 20, 36])

        betas_s = _get('betas', pd.Series(dtype=float))
        high_beta = []

        if betas_s is not None and len(betas_s) > 0:
            for i, (ticker, beta) in enumerate(
                    betas_s.sort_values(ascending=False).items()):
                r   = i + 3
                alt = i % 2 == 0
                w   = optimal_weights.get(ticker, 0) * 100

                if beta > 1.2:
                    interp = 'High — amplifies market moves'
                    high_beta.append(ticker)
                elif beta > 0.8:
                    interp = 'Market-like — moves with index'
                elif beta > 0.4:
                    interp = 'Moderate — partial market exposure'
                else:
                    interp = 'Defensive — low market sensitivity'

                data_cell(ws, r, 1, ticker,                   alt=alt)
                data_cell(ws, r, 2, ticker.replace('.NS',''),  alt=alt)
                data_cell(ws, r, 3, f'{beta:.4f}',            alt=alt,
                        color=C_NEG if beta>1.2 else
                                (C_POS if beta < 0.8 else C_BLACK))
                data_cell(ws, r, 4, pct(w), alt=alt,
                        color='78909c' if w < 0.1 else C_BLACK)
                data_cell(ws, r, 5, interp, alt=alt, color='546e7a')

            _log('M7 Report',
                f'Sheet 9 — {len(betas_s)} betas written', 'ok')
            bm = _get('BENCHMARK', '^NSEI')
            _log('M6 Risk',
                f'Benchmark used for beta: {bm}', 'info')
            if high_beta:
                _log('M6 Risk',
                    f'{len(high_beta)} high-beta stocks (>1.2): '
                    f'{", ".join([t.replace(".NS","") for t in high_beta])}',
                    'warn')
        else:
            ws.cell(3, 1).value = 'Run Module 6 first.'
            ws.cell(3, 1).font  = Font(name='Arial', size=9,
                                        italic=True, color='78909c')
            _log('M6 Risk', 'No beta data found — run Module 6', 'warn')


    # ============================================================
    # SHEET 10 — Charts
    # ============================================================
    def build_charts(wb):
        ws = wb.create_sheet('10. Charts')
        ws.sheet_view.showGridLines     = False
        ws.sheet_view.showRowColHeaders = False

        ws.merge_cells('A1:L1')
        c           = ws['A1']
        c.value     = 'Charts — Modules 4, 5, 6'
        c.font      = Font(name='Arial', size=13, bold=True, color=C_HDR_BG)
        c.alignment = ctr()
        c.fill      = fill(C_SUB_BG)
        ws.row_dimensions[1].height = 26

        row = 3
        for title, path in CHART_FILES.items():
            ws.merge_cells(start_row=row, start_column=1,
                        end_row=row, end_column=12)
            lbl           = ws.cell(row, 1)
            lbl.value     = title
            lbl.font      = Font(name='Arial', size=10, bold=True, color=C_HDR_BG)
            lbl.fill      = fill(C_SUB_BG)
            lbl.alignment = lft()
            ws.row_dimensions[row].height = 16
            row += 1

            if os.path.exists(path):
                try:
                    img        = XLImage(path)
                    scale      = min(1.0, 750 / img.width)
                    img.width  = int(img.width  * scale)
                    img.height = int(img.height * scale)
                    ws.add_image(img, f'A{row}')
                    rows_needed = max(20, int(img.height / 15) + 2)
                    for rr in range(row, row + rows_needed):
                        ws.row_dimensions[rr].height = 15
                    row += rows_needed + 2
                    _log('M7 Report', f'Chart embedded: {path}', 'ok')
                except Exception as e:
                    ws.cell(row, 1).value = f'[Embed failed: {e}]'
                    ws.cell(row, 1).font  = Font(name='Arial', size=8, color='78909c')
                    row += 2
                    _log('M7 Report', f'Chart embed failed: {path} — {e}', 'warn')
            else:
                ws.cell(row, 1).value = (f'[{path} not found — '
                                        'run Module 4/5/6 first]')
                ws.cell(row, 1).font  = Font(name='Arial', size=8, color='78909c')
                row += 2
                _log('M7 Report', f'Chart not found: {path}', 'warn')

        for col in range(1, 13):
            ws.column_dimensions[get_column_letter(col)].width = 12


    # ============================================================
    # SHEET 11 — Walk-Forward Backtest
    # ============================================================
    def build_walkforward(wb):
        wf_summary_data  = _get('wf_summary')
        wf_window_data   = _get('wf_window_stats')
        wf_turnover_data = _get('wf_turnover')
        wf_port_ret      = _get('wf_port_returns')

        if wf_summary_data is None or wf_port_ret is None:
            ws = wb.create_sheet('11. Walk-Fwd Backtest')
            ws.sheet_view.showGridLines = False
            ws.merge_cells('A1:G1')
            c           = ws['A1']
            c.value     = 'Walk-Forward Backtest (Module 8) — Not yet run'
            c.font      = Font(name='Arial', size=11, bold=True, color=C_NEG)
            c.alignment = ctr()
            c.fill      = fill(C_SUB_BG)
            ws.merge_cells('A2:G2')
            note           = ws['A2']
            note.value     = ('Run Module 8 first, then re-run Module 7 '
                            'to populate this sheet.')
            note.font      = Font(name='Arial', size=9, italic=True, color='546e7a')
            note.alignment = lft()
            _log('M8 Backtest', 'Module 8 not run — sheet skipped', 'warn')
            return

        ws = wb.create_sheet('11. Walk-Fwd Backtest')
        ws.sheet_view.showGridLines = False

        INIT      = _get('INITIAL_VALUE', 1_000_000)
        is_sh     = _get('port_sharpe', float('nan'))

        ws.merge_cells('A1:H1')
        c           = ws['A1']
        c.value     = 'Walk-Forward Out-of-Sample Backtest — Module 8'
        c.font      = Font(name='Arial', size=14, bold=True, color=C_HDR_BG)
        c.alignment = ctr()
        c.fill      = fill(C_SUB_BG)
        ws.row_dimensions[1].height = 30

        _s_dt = wf_port_ret.index[0].strftime('%d %b %Y')
        _e_dt = wf_port_ret.index[-1].strftime('%d %b %Y')
        _yrs  = (wf_port_ret.index[-1] - wf_port_ret.index[0]).days / 365.25

        ws.merge_cells('A2:H2')
        sub           = ws['A2']
        sub.value     = (f'Period: {_s_dt} → {_e_dt}  ({_yrs:.1f} years)  |  '
                        f'Rebalance: {_get("REBALANCE_FREQ","N/A")}  |  '
                        f'Tx Cost: {_get("TRANSACTION_COST",0)*100:.2f}% one-way')
        sub.font      = Font(name='Arial', size=8, color='546e7a')
        sub.alignment = ctr()

        r = 4

        # ── A. Aggregate Performance ─────────────────────────────
        section_title(ws, r, 'A. Aggregate Out-of-Sample Performance', 8); r += 1

        _strats = [
            ('max_sharpe', 'Max Sharpe (OOS)'),
            ('min_var',    'Min Variance (OOS)'),
            ('equal_wt',   'Equal Weight (OOS)'),
            ('benchmark',  'Nifty 50 (OOS)'),
        ]
        _metric_defs = [
            ('Final Value (Rs.)',       'final_value', False, True),
            ('Total Return (%)',        'total_ret',   True,  False),
            ('CAGR (%)',                'cagr',        True,  False),
            ('Annualised Vol (%)',      'ann_vol',      True,  False),
            ('Annualised Return (%)',   'ann_ret',      True,  False),
            ('Sharpe Ratio',            'sharpe',       False, False),
            ('Sortino Ratio',           'sortino',      False, False),
            ('Hit Rate (%)',            'hit_rate',     True,  False),
            ('Max Drawdown (%)',        'max_dd',       True,  False),
            ('Max DD Duration (days)',  'max_dd_dur',   False, False),
            ('Calmar Ratio',            'calmar',       False, False),
            ('VaR Breach Rate 95% (%)', 'var_breach',   True,  False),
            ('OOS Days',                'n_days',       False, False),
        ]

        col_labels = ['Metric'] + [lbl for _, lbl in _strats]
        header_row(ws, r, col_labels, [32] + [20]*4); r += 1

        def _fmt(val, is_pct, is_rupee):
            if val is None or (isinstance(val, float) and np.isnan(val)):
                return '—'
            if is_rupee:  return rupee(val)
            if is_pct:    return pct(val*100, 2, sign=(val >= 0))
            if isinstance(val, int): return str(val)
            return f'{val:.4f}'

        for i, (lbl, key, is_pct, is_rupee) in enumerate(_metric_defs):
            alt = i % 2 == 0
            data_cell(ws, r, 1, lbl, alt=alt, bold=True)
            for col_idx, (strat_key, _) in enumerate(_strats, 2):
                m   = wf_summary_data.get(strat_key, {})
                val = m.get(key)
                fmt = _fmt(val, is_pct, is_rupee)
                color = C_BLACK
                if val is not None and not (isinstance(val, float) and np.isnan(val)):
                    if key == 'final_value':
                        color = C_POS if val >= INIT else C_NEG
                    elif key in ('total_ret', 'cagr', 'ann_ret'):
                        color = C_POS if val >= 0 else C_NEG
                    elif key == 'max_dd':
                        color = C_NEG
                    elif key in ('sharpe', 'sortino', 'calmar'):
                        color = C_POS if val >= 0 else C_NEG
                data_cell(ws, r, col_idx, fmt, alt=alt, color=color)
            r += 1

        # ── B. IS → OOS Sharpe Gap ───────────────────────────────
        r += 1
        section_title(ws, r, 'B. In-Sample vs Out-of-Sample Sharpe Gap', 8); r += 1
        header_row(ws, r,
                ['Metric', 'Value', 'Interpretation', '', '', '', '', ''],
                [32, 20, 46]); r += 1

        oos_sh = wf_summary_data.get('max_sharpe', {}).get('sharpe', float('nan'))
        if (not np.isnan(oos_sh) and not np.isnan(is_sh)
                and abs(is_sh) > 1e-4):
            drop_pct = (1.0 - oos_sh / is_sh) * 100
            drop_pct = max(-999.0, min(drop_pct, 999.0))
            verdict  = ('Acceptable — < 50% IS Sharpe preserved'
                        if drop_pct < 50
                        else 'HIGH — consider increasing SHRINK_FACTOR')
        else:
            drop_pct = float('nan')
            verdict  = 'Indeterminate'

        gap_rows = [
            ('In-Sample Sharpe (Module 3)',
            f'{is_sh:.4f}' if not np.isnan(is_sh) else '—',
            'Trained on full historical data'),
            ('OOS Sharpe (Max Sharpe)',
            f'{oos_sh:.4f}' if not np.isnan(oos_sh) else '—',
            'Walk-forward out-of-sample'),
            ('IS → OOS Sharpe Drop',
            f'{drop_pct:+.1f}%' if not np.isnan(drop_pct) else '—',
            verdict),
        ]
        for i, (m, v, note) in enumerate(gap_rows):
            alt    = i % 2 == 0
            is_bad = ('HIGH' in note)
            data_cell(ws, r, 1, m,    alt=alt)
            data_cell(ws, r, 2, v,    alt=alt,
                    color=C_NEG if is_bad else C_BLUE)
            data_cell(ws, r, 3, note, alt=alt, color='546e7a')
            r += 1

        # ── C. Per-Window Table ───────────────────────────────────
        r += 1
        section_title(ws, r, 'C. Per-Window Performance', 8); r += 1
        header_row(ws, r,
                ['Rebalance Date', 'OOS Days',
                    'MS Sharpe', 'MV Sharpe',
                    'Turnover MS (%)', 'Turnover MV (%)',
                    'Cost Drag (%)', 'Stocks MS/MV'],
                [18, 10, 12, 12, 16, 16, 13, 16]); r += 1

        n_windows   = 0
        n_pos_ms    = 0
        n_pos_mv    = 0
        if wf_window_data:
            for i, ws_row in enumerate(wf_window_data):
                alt   = i % 2 == 0
                sh_ms = ws_row.get('sharpe_ms', float('nan'))
                sh_mv = ws_row.get('sharpe_mv', float('nan'))
                to_ms = ws_row.get('turnover_ms', 0)
                to_mv = ws_row.get('turnover_mv', 0)
                cost  = ws_row.get('cost_ms', 0)
                n_ms  = ws_row.get('n_stocks_ms', '—')
                n_mv  = ws_row.get('n_stocks_mv', '—')
                n_windows += 1
                if not np.isnan(sh_ms) and sh_ms >= 0: n_pos_ms += 1
                if not np.isnan(sh_mv) and sh_mv >= 0: n_pos_mv += 1

                data_cell(ws, r, 1, str(ws_row['date'].date()), alt=alt)
                data_cell(ws, r, 2, ws_row.get('n_days','—'),
                        alt=alt, align='center')
                data_cell(ws, r, 3,
                        f'{sh_ms:+.3f}' if not np.isnan(sh_ms) else '—',
                        alt=alt,
                        color=C_POS if (not np.isnan(sh_ms) and sh_ms >= 0)
                                else C_NEG)
                data_cell(ws, r, 4,
                        f'{sh_mv:+.3f}' if not np.isnan(sh_mv) else '—',
                        alt=alt,
                        color=C_POS if (not np.isnan(sh_mv) and sh_mv >= 0)
                                else C_NEG)
                data_cell(ws, r, 5, pct(to_ms*100), alt=alt)
                data_cell(ws, r, 6, pct(to_mv*100), alt=alt)
                data_cell(ws, r, 7, pct(cost*100, 3), alt=alt, color=C_NEG)
                data_cell(ws, r, 8, f'{n_ms} / {n_mv}', alt=alt, align='center')
                r += 1

        # ── D. Turnover Summary ───────────────────────────────────
        r += 1
        section_title(ws, r, 'D. Turnover & Transaction Cost Summary', 8); r += 1
        header_row(ws, r,
                ['Metric', 'Max Sharpe', 'Min Variance', '', '', '', '', ''],
                [32, 20, 20]); r += 1

        if wf_turnover_data:
            _to_ms_l    = [t[1] for t in wf_turnover_data]
            _to_mv_l    = [t[2] for t in wf_turnover_data]
            _RFREQ      = _get('REBALANCE_FREQ', '3ME')
            _periods_yr = REBALANCE_PERIODS_PER_YEAR.get(_RFREQ, 4)
            _tx         = _get('TRANSACTION_COST', 0.0035)
            avg_ms      = float(np.mean(_to_ms_l))
            avg_mv      = float(np.mean(_to_mv_l))
            ann_ms      = avg_ms * _periods_yr
            ann_mv      = avg_mv * _periods_yr

            for i, (m, v_ms, v_mv) in enumerate([
                ('Avg Turnover / Rebalance',
                pct(avg_ms*100), pct(avg_mv*100)),
                ('Annualised Turnover',
                pct(ann_ms*100), pct(ann_mv*100)),
                ('Est. Annual Cost Drag',
                pct(ann_ms*_tx*100,3), pct(ann_mv*_tx*100,3)),
                ('Tx Cost (one-way)', pct(_tx*100,2), '(same)'),
            ]):
                alt = i % 2 == 0
                data_cell(ws, r, 1, m,    alt=alt)
                data_cell(ws, r, 2, v_ms, alt=alt, color=C_BLUE)
                data_cell(ws, r, 3, v_mv, alt=alt, color=C_BLUE)
                r += 1

        # ── E. VaR Calibration ────────────────────────────────────
        r += 1
        section_title(ws, r, 'E. VaR Calibration Check (95%)', 8); r += 1
        header_row(ws, r,
                ['Metric', 'Value', 'Assessment', '', '', '', '', ''],
                [32, 20, 40]); r += 1

        var_95      = _get('VaR_95', 0)
        oos_breach  = wf_summary_data.get('max_sharpe',{}).get('var_breach',float('nan'))

        if not np.isnan(oos_breach):
            if   oos_breach <= 0.03: calib = 'Conservative — VaR overstates risk slightly'
            elif oos_breach <= 0.07: calib = 'Well-calibrated'
            elif oos_breach <= 0.10: calib = 'Slightly aggressive — VaR may understate'
            else:                    calib = 'Aggressive — VaR significantly understates risk'
        else:
            calib = 'Indeterminate'

        for i, (m, v, note) in enumerate([
            ('Module 5 Annual VaR 95%',
            rupee(var_95) if var_95 else '—',
            f'{var_95/INIT*100:.1f}% of capital' if var_95 and INIT else '—'),
            ('OOS Daily Breach Rate',
            pct(oos_breach*100) if not np.isnan(oos_breach) else '—', calib),
            ('Expected Breach Rate', pct(5.0), 'Theoretical target at 95% VaR'),
        ]):
            alt = i % 2 == 0
            data_cell(ws, r, 1, m,    alt=alt)
            data_cell(ws, r, 2, v,    alt=alt, color=C_BLUE)
            data_cell(ws, r, 3, note, alt=alt, color='546e7a')
            r += 1

        # ── F. Chart ──────────────────────────────────────────────
        r += 1
        section_title(ws, r, 'F. Walk-Forward Backtest Chart', 8); r += 1
        _wf_chart = str(output_dir / 'walkforward_backtest.png')
        if os.path.exists(_wf_chart):
            try:
                img        = XLImage(_wf_chart)
                scale      = min(1.0, 900 / img.width)
                img.width  = int(img.width  * scale)
                img.height = int(img.height * scale)
                ws.add_image(img, f'A{r}')
                rows_needed = max(25, int(img.height/15) + 2)
                for rr in range(r, r + rows_needed):
                    ws.row_dimensions[rr].height = 15
            except Exception as e:
                ws.cell(r, 1).value = f'[Chart embed failed: {e}]'
                ws.cell(r, 1).font  = Font(name='Arial', size=8, color='78909c')
        else:
            ws.cell(r, 1).value = (f'[{_wf_chart} not found — '
                                    'ensure Module 8 saved the chart]')
            ws.cell(r, 1).font  = Font(name='Arial', size=8, color='78909c')

        for col_idx, w in enumerate([18,10,12,12,16,16,13,16], 1):
            ws.column_dimensions[get_column_letter(col_idx)].width = w

        # ── Diagnostics for M8 ───────────────────────────────────
        ms_m = wf_summary_data.get('max_sharpe', {})
        _log('M8 Backtest',
            f'{n_windows} windows completed  |  '
            f'Period: {_s_dt} → {_e_dt}  ({_yrs:.1f} yrs)',
            'ok')
        _log('M8 Backtest',
            f'Max Sharpe OOS: Sharpe={ms_m.get("sharpe",float("nan")):.3f}  '
            f'CAGR={ms_m.get("cagr",float("nan"))*100:+.1f}%  '
            f'MaxDD={ms_m.get("max_dd",float("nan"))*100:.1f}%',
            'info')
        _log('M8 Backtest',
            f'MS positive-Sharpe windows: {n_pos_ms}/{n_windows}  |  '
            f'MV positive-Sharpe windows: {n_pos_mv}/{n_windows}',
            'info' if n_pos_ms >= n_windows//2 else 'warn')
        _log('M8 Backtest',
            f'IS→OOS drop: {drop_pct:+.1f}%  ({verdict})'
            if not np.isnan(drop_pct) else 'IS→OOS drop: indeterminate',
            'ok' if (not np.isnan(drop_pct) and drop_pct < 50) else 'warn')
        _log('M8 Backtest',
            f'VaR calibration: {calib}',
            'ok' if 'Well' in calib or 'Conservative' in calib else 'warn')
        _log('M8 Backtest', 'Safe to proceed to Module 7', 'ok')
        _log('M7 Report', 'Sheet 11 — Walk-forward backtest written', 'ok')


    # ============================================================
    # SHEET 12 — Diagnostics
    # ============================================================
    def build_diagnostics(wb):
        """
        Pipeline health log. Written last so it captures every
        message appended by all previous build functions.

        Columns:  Time | Module | Status | Message
        Rows colour-coded by status:
        ok    → light green background
        info  → light blue background
        warn  → light amber background
        error → light red background
        """
        ws = wb.create_sheet('12. Diagnostics')
        ws.sheet_view.showGridLines = False

        # ── Header ───────────────────────────────────────────────
        ws.merge_cells('A1:D1')
        c           = ws['A1']
        c.value     = 'Pipeline Diagnostics — Stardust 1.0'
        c.font      = Font(name='Arial', size=13, bold=True, color=C_HDR_BG)
        c.alignment = ctr()
        c.fill      = fill(C_SUB_BG)
        ws.row_dimensions[1].height = 28

        # Run summary line
        total   = len(_DIAG)
        n_ok    = sum(1 for d in _DIAG if d['status'] == 'ok')
        n_warn  = sum(1 for d in _DIAG if d['status'] == 'warn')
        n_err   = sum(1 for d in _DIAG if d['status'] == 'error')
        n_info  = sum(1 for d in _DIAG if d['status'] == 'info')

        ws.merge_cells('A2:D2')
        s           = ws['A2']
        s.value     = (f'Total entries: {total}   |   '
                    f'OK: {n_ok}   '
                    f'Info: {n_info}   '
                    f'Warnings: {n_warn}   '
                    f'Errors: {n_err}   |   '
                    f'Generated: {datetime.datetime.now().strftime("%d %b %Y %H:%M:%S")}')
        s.font      = Font(name='Arial', size=8, color='546e7a')
        s.alignment = ctr()

        # ── Column headers ───────────────────────────────────────
        header_row(ws, 3,
                ['Time', 'Module', 'Status', 'Message'],
                [10, 18, 10, 80])

        # ── Status config ────────────────────────────────────────
        STATUS_CONFIG = {
            'ok'   : ('✓ OK',    C_POS,   C_OK_BG),
            'info' : ('◎ Info',  C_INFO,  C_INFO_BG),
            'warn' : ('⚠ Warn',  C_WARN,  C_WARN_BG),
            'error': ('✗ Error', C_NEG,   C_ERR_BG),
        }

        # ── Rows ─────────────────────────────────────────────────
        for i, entry in enumerate(_DIAG):
            r      = i + 4
            status = entry.get('status', 'info')
            label, color, bg = STATUS_CONFIG.get(
                status, ('◎ Info', C_INFO, C_INFO_BG))

            data_cell(ws, r, 1, entry.get('timestamp', '—'),
                    align='center', bg=bg)
            data_cell(ws, r, 2, entry.get('module', '—'),
                    bold=True, bg=bg)
            data_cell(ws, r, 3, label,
                    color=color, bold=True, align='center', bg=bg)
            data_cell(ws, r, 4, entry.get('message', '—'),
                    bg=bg)
            ws.row_dimensions[r].height = 14

        # ── Legend ───────────────────────────────────────────────
        legend_r = len(_DIAG) + 6
        ws.merge_cells(start_row=legend_r, start_column=1,
                    end_row=legend_r, end_column=4)
        leg           = ws.cell(legend_r, 1)
        leg.value     = ('Legend:  '
                        '✓ OK = passed health check   '
                        '◎ Info = informational   '
                        '⚠ Warn = review recommended   '
                        '✗ Error = action required')
        leg.font      = Font(name='Arial', size=8, italic=True, color='546e7a')
        leg.alignment = lft()

        # ── Freeze top rows ──────────────────────────────────────
        ws.freeze_panes = 'A4'


    # ============================================================
    # MAIN
    # ============================================================
    def build_excel(output_path=OUTPUT_XLSX):

        # ── Pre-flight check ─────────────────────────────────────


        # ── Seed the diagnostics log ──────────────────────────────
        _DIAG.clear()
        _log('M7 Report',
            f'Report generation started — '
            f'{datetime.datetime.now().strftime("%d %b %Y %H:%M:%S")}',
            'info')

        # ── Capture M0 config provenance if available ─────────────
        rm = _get('RETURN_METHOD', 'N/A')
        rf = _get('RISK_FREE', 0.065)
        ly = _get('LOOKBACK_Y', 5)
        iv = _get('INITIAL_VALUE', 1_000_000)
        td = _get('T_DIST_DF', 6)
        bm = _get('BENCHMARK', '^NSEI')

        _log('M0 Config', f'Return method   : {rm}', 'info')
        _log('M0 Config', f'Risk-free rate  : {rf*100:.2f}%', 'info')
        _log('M0 Config', f'Lookback        : {ly} years', 'info')
        _log('M0 Config', f'Initial value   : Rs.{iv:,.0f}', 'info')
        _log('M0 Config', f'Sim distribution: '
            f'{"Normal" if td == float("inf") else f"t-dist df={td}"}', 'info')
        _log('M0 Config', f'Benchmark       : {bm}', 'info')

        # ── M1 diagnostics ────────────────────────────────────────
        requested = _get('TICKERS', [])
        fetched   = clean_tickers or []
        dropped   = [t for t in requested if t not in fetched]
        _log('M1 Data', f'{len(requested)} tickers requested', 'info')
        _log('M1 Data', f'{len(fetched)} tickers successfully fetched', 'ok')
        if dropped:
            _log('M1 Data',
                f'{len(dropped)} tickers dropped: {", ".join(dropped)}', 'warn')
        date_from = log_return.index[0].strftime('%d %b %Y')
        date_to   = log_return.index[-1].strftime('%d %b %Y')
        _log('M1 Data',
            f'Data range: {date_from} → {date_to}  '
            f'({len(log_return)} trading days)', 'info')
        _log('M1 Data', 'Safe to proceed to Module 2', 'ok')

        # ── M2 diagnostics ────────────────────────────────────────
        cov_m = _get('_cov_method', 'N/A')
        _log('M2 Covariance', f'Covariance method used: {cov_m}', 'info')
        _log('M2 Covariance', f'Active return label: {_get("active_label","N/A")}',
            'info')
        _log('M2 Covariance', 'Safe to proceed to Module 3', 'ok')

        print('Building Excel report...')
        wb = Workbook()

        # ── Build all sheets ──────────────────────────────────────
        steps = [
            ('1.  Summary',           build_summary),
            ('2.  Stock Universe',    build_universe),
            ('3.  Expected Returns',  build_returns),
            ('4.  Weights',           build_weights),
            ('5.  Correlation',       build_covariance),
            ('6.  Simulation',        build_simulation),
            ('7.  Risk Metrics',      build_risk),
            ('8.  Stress Tests',      build_stress),
            ('9.  Betas',             build_betas),
            ('10. Charts',            build_charts),
            ('11. Walk-Fwd Backtest', build_walkforward),
            ('12. Diagnostics',       build_diagnostics),   # always last
        ]
        for label, fn in steps:
            print(f'  [{label}]...')
            try:
                fn(wb)
            except Exception as e:
                print(f'    WARNING: {e}')
                _log('M7 Report', f'Sheet {label} failed: {e}', 'error')

        # ── Final diagnostics entries ─────────────────────────────
        n_sheets = len(wb.sheetnames)
        kb        = 0
        wb.save(output_path)
        kb = os.path.getsize(output_path) / 1024
        _log('M7 Report', f'{n_sheets} sheets written successfully', 'ok')
        _log('M7 Report', f'Output file: {output_path}  ({kb:.0f} KB)', 'info')
        _log('M7 Report', f'Report complete — Stardust 1.0', 'ok')

        # ── Re-save after diagnostics are fully populated ─────────
        # (Sheet 12 was built before the final log entries above,
        #  so we append them directly to the sheet now)
        try:
            ws12 = wb['12. Diagnostics']
            last_row = len(_DIAG) + 4   # 3 header rows + 1-based
            STATUS_CONFIG = {
                'ok'   : ('✓ OK',    C_POS,  C_OK_BG),
                'info' : ('◎ Info',  C_INFO, C_INFO_BG),
                'warn' : ('⚠ Warn',  C_WARN, C_WARN_BG),
                'error': ('✗ Error', C_NEG,  C_ERR_BG),
            }
            # Entries added after build_diagnostics() ran
            late_entries = _DIAG[last_row - 4:]
            for j, entry in enumerate(late_entries):
                r      = last_row + j
                status = entry.get('status', 'info')
                label2, color, bg = STATUS_CONFIG.get(
                    status, ('◎ Info', C_INFO, C_INFO_BG))
                data_cell(ws12, r, 1, entry.get('timestamp','—'),
                        align='center', bg=bg)
                data_cell(ws12, r, 2, entry.get('module','—'),
                        bold=True, bg=bg)
                data_cell(ws12, r, 3, label2,
                        color=color, bold=True, align='center', bg=bg)
                data_cell(ws12, r, 4, entry.get('message','—'), bg=bg)
                ws12.row_dimensions[r].height = 14
        except Exception:
            pass

        wb.save(output_path)
        print(f'\nSaved: {output_path}  ({kb:.0f} KB)')

        # ── Copy to Downloads ─────────────────────────────────────
        dest = pathlib.Path.home() / 'Downloads' / pathlib.Path(output_path).name
        shutil.copy2(output_path, dest)
        print(f'Copied to: {dest}')

        # ── Auto-open ─────────────────────────────────────────────
        try:
            p    = pathlib.Path(output_path).resolve()
            sys_ = platform.system()
            if   sys_ == 'Windows': os.startfile(str(p))
            elif sys_ == 'Darwin':  subprocess.Popen(['open', str(p)])
            else:                   subprocess.Popen(['xdg-open', str(p)])
            print('Excel opened.')
        except Exception as e:
            print(f'Could not auto-open: {e}')

        # ── Jupyter download link ─────────────────────────────────
        try:
            from IPython.display import display, HTML
            display(HTML(
                f'<a href="{output_path}" download target="_blank">'
                f'&#x2B07; Download {output_path}</a>'))
        except Exception:
            pass

        print('Done.')


    build_excel()
    status_callback("✓ Report Generated Successfully!")
    # ---- publish (generated) ----
    _scope = locals()
    ctx.update({k: _scope[k] for k in _PUBLISHED if k in _scope})

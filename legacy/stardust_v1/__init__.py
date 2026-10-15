"""
Stardust 1.x engine, split into one module per stage (Task 1 "firefighting").

The legacy stardust_core.run_stardust() was a single 6,000-line function whose
stages shared state through closure variables. Each stage now lives in its own
module with an explicit data contract (INPUTS / OUTPUTS) and the stages pass
state through a plain ``ctx`` dict. The numerical logic is unchanged, known
flaws included; see STAR2.0/docs/AUDIT_legacy.md for them. The corrected
engine is the ``star`` package.

Usage (same signature as the legacy function):

    from legacy.stardust_v1 import run_stardust
    ctx = run_stardust(params, tickers, output_dir, status_callback)
"""
from ._bootstrap import ensure_bootstrap

ensure_bootstrap()

from . import (  # noqa: E402  (bootstrap must run first: matplotlib backend)
    m0_config,
    m1_data,
    m2a_factor_cov,
    m2b_signals_views,
    m2c_bl_selection,
    m3_optimizer,
    m4_frontier,
    m5_simulation,
    m6_risk_stress,
    m8_backtest,
    m7_report,
)

#: execution order - identical to the legacy monolith (M8 ran before M7)
PIPELINE = (
    m0_config,
    m1_data,
    m2a_factor_cov,
    m2b_signals_views,
    m2c_bl_selection,
    m3_optimizer,
    m4_frontier,
    m5_simulation,
    m6_risk_stress,
    m8_backtest,
    m7_report,
)


def run_stardust(params, tickers, output_dir, status_callback, stop_after=None):
    """Run the legacy pipeline. Returns the final ctx (all stage outputs).

    stop_after: optional module name (e.g. "m6_risk_stress") to halt early.
    """
    ctx = {
        "params": params,
        "tickers": tickers,
        "output_dir": output_dir,
        "status_callback": status_callback,
    }
    for stage in PIPELINE:
        stage.run(ctx)
        if stop_after and stage.__name__.endswith(stop_after):
            break
    return ctx

"""
Two things this file proves:

1. the proposed weights are tested against the uncertainty in their own inputs;
2. the whole chain - real NSE prices, covariance, GARCH, equilibrium prior,
   Black-Litterman, mandate constraints - runs end to end and produces a
   portfolio that satisfies the mandate exactly.
"""
import numpy as np
import pandas as pd
import pytest

from star.config.schema import Mandate
from star.core.errors import OptimizationError
from star.data.fixture import FixtureProvider
from star.data.quality import assess
from star.data.view import DataView
from star.estimators import garch as G
from star.estimators.black_litterman import View, black_litterman
from star.estimators.covariance import estimate_covariance
from star.estimators.equilibrium import equilibrium_prior, reference_weights
from star.estimators.views import ICEstimate, quintile_view
from star.optimize.baselines import equal_weight, risk_contributions
from star.optimize.cardinality import limit_holdings
from star.optimize.constraints import build_constraints
from star.optimize.frontier import efficient_frontier
from star.optimize.mean_variance import OptimiserSettings, optimise
from star.optimize.stability import perturbation_sensitivity
from tests.test_optimize_core import NAMES, SECTORS, _cov, _mu


# ------------------------------------------------------------------ stability ---
def test_stability_is_measured_against_the_uncertainty_in_the_inputs():
    cov, mu = _cov(), _mu(seed=5)
    cs = build_constraints(NAMES, Mandate(max_stock_weight=0.20, max_sector_weight=0.45),
                           sectors=SECTORS)
    tiny = perturbation_sensitivity(mu, cov, cs, risk_aversion=3.0, tau=1e-8, draws=8)
    loose = perturbation_sensitivity(mu, cov, cs, risk_aversion=3.0, tau=0.25, draws=8)
    assert tiny.mean_turnover < 1e-3
    assert loose.mean_turnover > tiny.mean_turnover
    assert tiny.is_stable() and "stable" in tiny.headline
    assert loose.max_name_change >= loose.mean_abs_change


def test_an_unstable_portfolio_says_so_in_words():
    cov, mu = _cov(), _mu(seed=5, scale=0.01)      # barely any dispersion: noise dominates
    cs = build_constraints(NAMES, Mandate(max_stock_weight=0.20, max_sector_weight=0.45),
                           sectors=SECTORS)
    r = perturbation_sensitivity(mu, cov, cs, risk_aversion=3.0, tau=0.50, draws=10)
    assert not r.is_stable()
    assert "defensible answers" in r.headline


def test_stability_is_reproducible_and_every_draw_stays_within_the_mandate():
    cov, mu = _cov(), _mu(seed=5)
    cs = build_constraints(NAMES, Mandate(max_stock_weight=0.20, max_sector_weight=0.45),
                           sectors=SECTORS)
    a = perturbation_sensitivity(mu, cov, cs, risk_aversion=3.0, draws=6, seed=7)
    b = perturbation_sensitivity(mu, cov, cs, risk_aversion=3.0, draws=6, seed=7)
    assert a.mean_turnover == pytest.approx(b.mean_turnover)
    assert np.allclose(a.dispersion.to_numpy(), b.dispersion.to_numpy())
    assert (a.dispersion >= 0).all()
    with pytest.raises(OptimizationError, match="at least 2 draws"):
        perturbation_sensitivity(mu, cov, cs, risk_aversion=3.0, draws=1)


# ------------------------------------------------------------- the whole chain ---
@pytest.fixture(scope="module")
def real_inputs():
    p = FixtureProvider.default()
    if not (p.root / "nse_close.parquet").exists():
        pytest.skip("fixture not built")
    panel = p.prices(p.universe)
    kept = assess(panel.close, min_adv_inr=0.0).kept[:20]
    v = DataView.build(panel, panel.sessions.max())
    cov = estimate_covariance(v.ohlc(), tickers=kept, as_of=v.as_of)
    w_ref, notes = reference_weights(kept, allow_market_cap=False)   # backtest-safe: equal weight
    prior, delta = equilibrium_prior(cov, w_ref, degradations=notes)
    return v, kept, cov, prior, delta


def test_the_full_chain_produces_a_compliant_portfolio(real_inputs):
    v, kept, cov, prior, delta = real_inputs
    scores = pd.Series(np.linspace(1, -1, len(kept)), index=kept)     # a stand-in signal
    ic = ICEstimate(mean=0.05, std_error=0.018, observations=40, horizon_days=21,
                    raw_mean=0.05, clipped=False)
    view = quintile_view(scores, cov, ic)
    bl = black_litterman(prior, cov, [view], delta=delta)

    sectors = pd.Series({t: ["A", "B", "C", "D"][i % 4] for i, t in enumerate(kept)})
    evidence = pd.Series(view.confidence, index=kept)

    # First, the mandate a manager might reach for: a 10% cap shrunk by moderate
    # evidence leaves 15 names unable to reach 100%. That is refused, with the sum.
    tight = Mandate(max_stock_weight=0.10, max_sector_weight=0.35, max_holdings=15, min_holdings=8)
    with pytest.raises(OptimizationError, match="cannot reach the 100% budget"):
        limit_holdings(bl.mu, bl.covariance, build_constraints(kept, tight, sectors=sectors,
                                                               evidence=evidence),
                       risk_aversion=delta, max_holdings=15, min_holdings=8, min_position=0.02)

    mandate = Mandate(max_stock_weight=0.15, max_sector_weight=0.35, max_holdings=15, min_holdings=8)
    cs = build_constraints(kept, mandate, sectors=sectors, evidence=evidence)
    assert cs.infeasibilities() == ()

    p = limit_holdings(bl.mu, bl.covariance, cs, risk_aversion=delta,
                       max_holdings=mandate.max_holdings, min_holdings=mandate.min_holdings,
                       min_position=0.02, settings=OptimiserSettings(turnover_penalty=0.02))
    cs.assert_satisfied(p.weights, what="end-to-end weights")
    assert mandate.min_holdings <= len(p.holdings(0.02)) <= mandate.max_holdings
    assert p.weights.sum() == pytest.approx(1.0, abs=1e-9)
    vol = p.volatility(bl.covariance)
    assert 0.05 < vol < 0.40, vol                                    # a plausible Indian equity book
    assert p.expected_return(bl.mu) > 0


def test_the_conviction_cap_binds_when_the_signal_is_weak(real_inputs):
    v, kept, cov, prior, delta = real_inputs
    mandate = Mandate(max_stock_weight=0.15, max_sector_weight=1.0)
    weak = build_constraints(kept, mandate, evidence=pd.Series(0.0, index=kept))
    strong = build_constraints(kept, mandate, evidence=pd.Series(1.0, index=kept))
    assert weak.upper.max() < strong.upper.max()
    pw = optimise(prior, cov, weak, risk_aversion=delta)
    ps = optimise(prior, cov, strong, risk_aversion=delta)
    assert pw.weights.max() <= weak.upper.max() + 1e-9
    assert len(pw.holdings()) >= len(ps.holdings())               # weak evidence spreads the book


def test_a_real_frontier_and_the_baselines_line_up(real_inputs):
    v, kept, cov, prior, delta = real_inputs
    cs = build_constraints(kept, Mandate(max_stock_weight=0.15, max_sector_weight=1.0))
    f = efficient_frontier(prior, cov, cs, n_points=6)
    for point in f.points.index:
        cs.assert_satisfied(f.weights.loc[point], what=f"frontier point {point}")
    ew = equal_weight(cov, kept)
    assert f.points["volatility"].min() <= ew.volatility(cov) + 1e-9
    rc = risk_contributions(f.max_sharpe_weights()[lambda s: s > 1e-6], cov)
    assert rc.sum() == pytest.approx(1.0)


def test_the_garch_horizon_covariance_can_drive_the_optimiser(real_inputs):
    """A 21-day view of risk should produce a different, still-compliant portfolio."""
    v, kept, cov, prior, delta = real_inputs
    close = v.ohlc()["close"][kept]
    r = np.log(close / close.shift(1)).iloc[1:]
    fcs = G.forecast_panel(r, 21, tickers=kept, as_of=v.as_of)
    horizon_cov = G.scale_covariance_to_horizon(cov, fcs, 21)
    cs = build_constraints(kept, Mandate(max_stock_weight=0.15, max_sector_weight=1.0))
    a = optimise(prior, cov, cs, risk_aversion=delta)
    b = optimise(prior, horizon_cov, cs, risk_aversion=delta)
    cs.assert_satisfied(b.weights, what="horizon-risk weights")
    assert horizon_cov.is_psd()
    assert np.abs(a.weights - b.weights).max() > 1e-6

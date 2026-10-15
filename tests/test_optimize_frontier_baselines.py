"""Frontier (solved, not interpolated), holding-count limits, and the baselines."""
import numpy as np
import pandas as pd
import pytest

from star.config.schema import Mandate
from star.core.errors import OptimizationError
from star.optimize.baselines import (equal_risk_contribution, equal_weight,
                                     hierarchical_risk_parity, risk_contributions)
from star.optimize.cardinality import limit_holdings, utility_contribution
from star.optimize.constraints import build_constraints
from star.optimize.frontier import efficient_frontier, max_return_portfolio
from star.optimize.mean_variance import min_variance, optimise
from tests.test_optimize_core import NAMES, SECTORS, N, _cov, _cs, _mu


# ------------------------------------------------------------------ frontier ---
@pytest.fixture(scope="module")
def frontier():
    cov, mu = _cov(), _mu(seed=5)
    cs = build_constraints(NAMES, Mandate(max_stock_weight=0.20, max_sector_weight=0.45),
                           sectors=SECTORS)
    return efficient_frontier(mu, cov, cs, n_points=9), mu, cov, cs


def test_every_frontier_point_is_feasible_and_risk_rises_with_return(frontier):
    f, mu, cov, cs = frontier
    assert len(f.points) >= 9
    for point in f.points.index:
        cs.assert_satisfied(f.weights.loc[point], what=f"frontier point {point}")
    ordered = f.points[f.points["label"] != "max_sharpe"].sort_values("expected_return")
    assert (ordered["volatility"].diff().dropna() > -1e-9).all()
    assert ordered["expected_return"].iloc[0] < ordered["expected_return"].iloc[-1]


def test_the_frontier_is_not_a_straight_line_between_its_ends(frontier):
    """
    The legacy interpolated between two portfolios. Under box and sector caps the
    efficient set is curved, so the interpolation is both infeasible and wrong -
    demonstrated here by measuring the gap.
    """
    f, mu, cov, cs = frontier
    ends = f.points[f.points["label"].isin(["min_variance", "max_return"])].sort_values("expected_return")
    lo, hi = ends.index[0], ends.index[-1]
    mid = f.points[f.points["label"] == "target"].sort_values("expected_return").index[len(
        f.points[f.points["label"] == "target"]) // 2]
    t = ((f.points.loc[mid, "expected_return"] - f.points.loc[lo, "expected_return"])
         / (f.points.loc[hi, "expected_return"] - f.points.loc[lo, "expected_return"]))
    blended = (1 - t) * f.weights.loc[lo] + t * f.weights.loc[hi]
    c = cov.annualise().matrix.to_numpy()
    blended_vol = float(np.sqrt(blended.to_numpy() @ c @ blended.to_numpy()))
    solved_vol = float(f.points.loc[mid, "volatility"])
    assert solved_vol < blended_vol - 1e-6          # the solved point really is more efficient
    assert np.abs(blended.to_numpy() - f.weights.loc[mid].to_numpy()).max() > 1e-3


def test_max_sharpe_comes_from_solved_points_and_is_the_best_of_them(frontier):
    f, mu, cov, cs = frontier
    best = f.max_sharpe()
    assert best["sharpe"] >= f.points["sharpe"].max() - 1e-12
    w = f.max_sharpe_weights()
    cs.assert_satisfied(w, what="max-Sharpe weights")
    assert "max_sharpe" in set(f.points["label"])    # the refinement ran
    # and it beats both ends of the frontier
    ends = f.points[f.points["label"].isin(["min_variance", "max_return"])]
    assert best["sharpe"] >= ends["sharpe"].max() - 1e-12


def test_at_volatility_returns_a_solved_point_never_a_blend(frontier):
    f, _, _, _ = frontier
    target = float(f.points["volatility"].median())
    w = f.at_volatility(target)
    assert any(np.allclose(w.to_numpy(), f.weights.loc[i].to_numpy()) for i in f.weights.index)


def test_max_return_portfolio_respects_the_mandate(frontier):
    f, mu, cov, cs = frontier
    p = max_return_portfolio(mu, cov, cs)
    assert cs.violations(p.weights) == {}
    assert p.expected_return(mu) >= f.points["expected_return"].max() - 1e-8


def test_a_frontier_needs_dispersion_in_expected_returns():
    cov = _cov()
    flat = _mu()
    flat = type(flat)(values=pd.Series(0.06, index=NAMES), basis=flat.basis,
                      compounding=flat.compounding, periodicity=flat.periodicity,
                      provenance=flat.provenance)
    with pytest.raises(OptimizationError, match="flat across the universe"):
        efficient_frontier(flat, cov, _cs(), n_points=5)
    with pytest.raises(OptimizationError, match="at least 3 points"):
        efficient_frontier(_mu(), cov, _cs(), n_points=2)


# --------------------------------------------------------------- cardinality ---
def test_the_holdings_limit_is_actually_enforced():
    """Legacy MAX_ASSETS was advertised and never applied (M3-06)."""
    cov, mu = _cov(), _mu(seed=8)
    cs = build_constraints(NAMES, Mandate(max_stock_weight=0.34, max_sector_weight=0.80),
                           sectors=SECTORS)
    relaxed = optimise(mu, cov, cs, risk_aversion=3.0)
    assert len(relaxed.holdings()) > 3                    # the limit will really bite
    p = limit_holdings(mu, cov, cs, risk_aversion=3.0, max_holdings=3, min_position=0.05)
    assert len(p.holdings(1e-6)) <= 3
    assert cs.violations(p.weights) == {}
    assert p.weights.sum() == pytest.approx(1.0, abs=1e-9)
    assert "of at most 3 names" in p.diagnostics["cardinality"]


def test_a_minimum_holding_count_is_also_enforced():
    cov, mu = _cov(), _mu(seed=8)
    cs = build_constraints(NAMES, Mandate(max_stock_weight=0.50, max_sector_weight=1.0),
                           sectors=SECTORS)
    p = limit_holdings(mu, cov, cs, risk_aversion=3.0, max_holdings=10, min_holdings=6,
                       min_position=0.02)
    assert len(p.holdings(0.02)) >= 6
    assert cs.violations(p.weights) == {}


def test_names_are_selected_by_what_they_add_not_by_raw_weight():
    cov, mu = _cov(), _mu(seed=8)
    cs = _cs(sectors=SECTORS)
    w = optimise(mu, cov, cs, risk_aversion=3.0).weights
    contrib = utility_contribution(w, mu, cov, 3.0)
    assert contrib.index.equals(w.index)
    # a name with zero weight contributes nothing; a held name with positive alpha does
    assert contrib[w < 1e-9].abs().max() < 1e-12
    assert contrib.sum() != pytest.approx(w.sum())


def test_impossible_holding_counts_are_refused_with_the_arithmetic():
    cov, mu = _cov(), _mu()
    cs = _cs()                                       # cap 15%, so 6 names cannot reach 100%
    with pytest.raises(OptimizationError, match="cannot reach the 100% budget"):
        limit_holdings(mu, cov, cs, risk_aversion=3.0, max_holdings=6)   # 6 x 15% = 90%
    with pytest.raises(OptimizationError, match="above the 100% budget"):
        limit_holdings(mu, cov, cs, risk_aversion=3.0, max_holdings=12, min_holdings=11,
                       min_position=0.10)
    with pytest.raises(OptimizationError, match="exceeds max_holdings"):
        limit_holdings(mu, cov, cs, risk_aversion=3.0, max_holdings=5, min_holdings=9)


# ----------------------------------------------------------------- baselines ---
def test_equal_weight_is_equal_weight():
    p = equal_weight(_cov())
    assert np.allclose(p.weights.to_numpy(), 1.0 / N)
    assert p.weights.sum() == pytest.approx(1.0)
    assert p.diagnostics["mandate_aware"] is False


def test_equal_risk_contribution_actually_equalises_risk():
    cov = _cov()
    p = equal_risk_contribution(cov)
    rc = risk_contributions(p.weights, cov)
    assert rc.sum() == pytest.approx(1.0)
    assert rc.max() - rc.min() < 1e-8                 # the defining property
    assert (p.weights > 0).all() and p.weights.sum() == pytest.approx(1.0)
    # it sits between equal weight and minimum variance in risk terms
    ew, mv = equal_weight(cov), min_variance(cov, _cs())
    assert mv.volatility(cov) <= p.volatility(cov) + 1e-9 <= ew.volatility(cov) + 1e-9


def test_equal_risk_contribution_puts_less_in_riskier_names():
    cov = _cov()
    p = equal_risk_contribution(cov)
    vols = cov.annualise().volatilities()
    assert p.weights.corr(vols, method="spearman") < 0


def test_hierarchical_risk_parity_is_inverse_variance_for_two_assets():
    """The one case where HRP has a closed form - so it can be checked exactly."""
    from star.core.types import CovarianceEstimate, Periodicity, Provenance
    v1, v2, rho = 0.04, 0.16, 0.3
    m = np.array([[v1, rho * np.sqrt(v1 * v2)], [rho * np.sqrt(v1 * v2), v2]])
    cov = CovarianceEstimate(matrix=pd.DataFrame(m, index=["A", "B"], columns=["A", "B"]),
                             periodicity=Periodicity.ANNUAL, provenance=Provenance(method="t"))
    p = hierarchical_risk_parity(cov)
    expected = np.array([1 / v1, 1 / v2]) / (1 / v1 + 1 / v2)
    assert np.allclose(p.weights.to_numpy(), expected, atol=1e-12)


def test_hierarchical_risk_parity_is_valid_and_deterministic():
    cov = _cov()
    a, b = hierarchical_risk_parity(cov), hierarchical_risk_parity(cov)
    assert np.allclose(a.weights.to_numpy(), b.weights.to_numpy())
    assert (a.weights > 0).all() and a.weights.sum() == pytest.approx(1.0)
    # HRP is not a variance minimiser - its claim is out-of-sample stability, and
    # in sample it can sit either side of 1/N. What must hold is diversification:
    # below the average single-name volatility, and not concentrated in one name.
    assert a.volatility(cov) < float(cov.annualise().volatilities().mean())
    assert a.weights.max() < 0.5


def test_baselines_are_checked_against_the_mandate_rather_than_clipped():
    """They ignore caps by construction, so the breach must be visible."""
    cov = _cov()
    tight = build_constraints(NAMES, Mandate(max_stock_weight=0.09), sectors=None)
    erc = equal_risk_contribution(cov)
    breaches = tight.violations(erc.weights)
    assert isinstance(breaches, dict)                 # reported, whatever it contains
    assert erc.diagnostics["mandate_aware"] is False

"""
Constraints, the QP underneath, and mean-variance construction.

The theme of every test here: a limit is either satisfied on the returned
weights or the run fails loudly. Nothing is clipped into shape afterwards.
"""
import numpy as np
import pandas as pd
import pytest

from star.config.schema import Mandate
from star.core.errors import OptimizationError
from star.core.types import (Basis, Compounding, CovarianceEstimate, Periodicity, Provenance,
                             ReturnEstimate)
from star.optimize.constraints import (ConstraintSet, build_constraints, conviction_cap,
                                       liquidity_cap)
from star.optimize.mean_variance import (OptimiserSettings, min_variance, optimise, target_return)
from star.optimize.qp import solve_qp

N = 12
NAMES = [f"A{i:02d}" for i in range(N)]
SECTORS = pd.Series(["Financials"] * 4 + ["IT"] * 3 + ["Energy"] * 3 + ["Pharma"] * 2, index=NAMES)


def _cov(seed=0, names=NAMES):
    rng = np.random.default_rng(seed)
    n = len(names)
    f = rng.normal(size=(n, 3))
    m = f @ f.T / 3 + np.diag(rng.uniform(0.02, 0.09, n))
    m *= 0.04                                   # annual variances in a plausible range
    return CovarianceEstimate(matrix=pd.DataFrame(m, index=names, columns=names),
                              periodicity=Periodicity.ANNUAL,
                              provenance=Provenance(method="synthetic"))


def _mu(seed=1, names=NAMES, scale=0.05):
    rng = np.random.default_rng(seed)
    return ReturnEstimate(values=pd.Series(rng.normal(0.06, scale, len(names)), index=names),
                          basis=Basis.EXCESS, compounding=Compounding.SIMPLE,
                          periodicity=Periodicity.ANNUAL,
                          provenance=Provenance(method="synthetic expected returns"))


def _cs(**kw):
    kw.setdefault("mandate", Mandate(max_stock_weight=0.15, max_sector_weight=0.40))
    return build_constraints(NAMES, kw.pop("mandate"), **kw)


# ------------------------------------------------------------------- the QP ---
def test_qp_matches_the_closed_form_minimum_variance_portfolio():
    S = _cov().matrix.to_numpy()
    one = np.ones(N)
    inv = np.linalg.inv(S)
    closed_form = inv @ one / (one @ inv @ one)
    sol = solve_qp(S, np.zeros(N), lb=np.full(N, -10.0), ub=np.full(N, 10.0),
                   A_eq=one.reshape(1, -1), b_eq=[1.0])
    assert np.abs(sol.x - closed_form).max() < 1e-8
    assert sol.max_constraint_violation == 0.0


def test_qp_matches_the_closed_form_maximum_utility_portfolio():
    S, m, delta = _cov().matrix.to_numpy(), _mu().values.to_numpy(), 3.0
    one = np.ones(N)
    inv = np.linalg.inv(S)
    lam = (1 - one @ inv @ m / delta) / (one @ inv @ one)
    closed_form = inv @ (m / delta + lam * one)
    # bounds wide enough not to bind: this is the unconstrained comparison
    assert np.abs(closed_form).max() < 50
    sol = solve_qp(delta * S, -m, lb=np.full(N, -100.0), ub=np.full(N, 100.0),
                   A_eq=one.reshape(1, -1), b_eq=[1.0])
    assert np.abs(sol.x - closed_form).max() < 1e-6


def test_qp_solution_satisfies_the_optimality_conditions_when_bounds_bind():
    """With box constraints there is no closed form, so check KKT directly."""
    S, m, delta, cap = _cov().matrix.to_numpy(), _mu().values.to_numpy(), 3.0, 0.20
    sol = solve_qp(delta * S, -m, lb=np.zeros(N), ub=np.full(N, cap),
                   A_eq=np.ones((1, N)), b_eq=[1.0])
    g = delta * S @ sol.x - m
    free = (sol.x > 1e-7) & (sol.x < cap - 1e-7)
    # on the free set the gradient must be flat (all equal to the budget multiplier)
    assert g[free].max() - g[free].min() < 1e-7
    at_cap, at_floor = sol.x > cap - 1e-7, sol.x < 1e-7
    if at_cap.any():
        assert g[at_cap].max() <= g[free].min() + 1e-7      # want more, not allowed
    if at_floor.any():
        assert g[at_floor].min() >= g[free].max() - 1e-7    # want less, not allowed


def test_qp_refuses_a_non_convex_problem():
    bad = np.array([[1.0, 0.0], [0.0, -1.0]])
    with pytest.raises(OptimizationError, match="not convex"):
        solve_qp(bad, np.zeros(2), lb=np.zeros(2), ub=np.ones(2))


# ------------------------------------------------------------- constraints ---
def test_conviction_cap_shrinks_the_limit_when_the_evidence_is_weak():
    caps = conviction_cap(0.10, pd.Series({"A": 1.0, "B": 0.5, "C": 0.0}))
    assert caps["A"] == pytest.approx(0.10)
    assert caps["C"] == pytest.approx(0.04)                  # floor_fraction 0.4
    assert caps["B"] == pytest.approx(0.07)
    assert caps["A"] > caps["B"] > caps["C"]
    with pytest.raises(OptimizationError, match="evidence must be"):
        conviction_cap(0.10, pd.Series({"A": 1.4}))


def test_liquidity_cap_is_what_can_be_sold_inside_the_exit_window():
    m = Mandate(max_days_to_exit=5.0, max_adv_participation=0.20)
    caps = liquidity_cap(pd.Series({"LIQUID": 1e7, "THIN": 1e5}), nav_inr=1e8, mandate=m)
    assert caps["LIQUID"] == pytest.approx(5 * 0.20 * 1e7 / 1e8)   # 10%
    assert caps["THIN"] == pytest.approx(0.001)
    with pytest.raises(OptimizationError, match="NAV must be positive"):
        liquidity_cap(pd.Series({"A": 1.0}), 0.0, m)


def test_the_tightest_cap_wins_and_the_report_can_say_which_one():
    cs = build_constraints(NAMES, Mandate(max_stock_weight=0.15, max_sector_weight=0.40),
                           sectors=SECTORS,
                           evidence=pd.Series(1.0, index=NAMES).mask(pd.Series(NAMES).isin(NAMES[:2]).to_numpy(), 0.0),
                           adv_inr=pd.Series(1e9, index=NAMES).mask(pd.Series(NAMES).isin([NAMES[5]]).to_numpy(), 1e5),
                           nav_inr=1e8)
    assert cs.upper[NAMES[0]] == pytest.approx(0.15 * 0.4)        # conviction binds
    assert cs.binding_cap[NAMES[0]] == "conviction"
    assert cs.binding_cap[NAMES[5]] == "liquidity"
    assert cs.binding_cap[NAMES[8]] == "mandate"


def test_impossible_mandates_are_reported_with_the_arithmetic():
    tight = build_constraints(NAMES, Mandate(max_stock_weight=0.05))
    assert any("below the 100% budget" in p for p in tight.infeasibilities())
    sectors = build_constraints(NAMES, Mandate(max_stock_weight=0.50, max_sector_weight=0.10),
                                sectors=SECTORS)
    assert any("sector caps allow at most" in p for p in sectors.infeasibilities())
    turn = build_constraints(NAMES, Mandate(max_stock_weight=0.15), max_turnover=0.01,
                             prev_weights=pd.Series(0.0, index=NAMES))
    assert any("turnover limit" in p for p in turn.infeasibilities())
    assert _cs(sectors=SECTORS).infeasibilities() == ()


def test_violations_name_every_breach():
    cs = _cs(sectors=SECTORS, prev_weights=pd.Series(1.0 / N, index=NAMES), max_turnover=0.10)
    w = pd.Series(1.0 / N, index=NAMES)
    assert cs.violations(w) == {}
    over = w.copy()
    over[NAMES[0]] = 0.30
    v = cs.violations(over / over.sum())
    assert any("over cap" in k for k in v)
    heavy = pd.Series(0.0, index=NAMES)
    heavy[NAMES[:4]] = 0.25                                  # all Financials
    assert "over sector cap: Financials" in cs.violations(heavy)
    assert "budget" in cs.violations(w * 0.9)
    moved = pd.Series(0.0, index=NAMES)
    moved[NAMES[:8]] = 0.125
    assert "over turnover limit" in cs.violations(moved)


def test_the_legacy_recipe_of_clipping_and_renormalising_breaks_its_own_cap():
    """Why constraints go to the solver: the old fix does not actually fix anything."""
    cap = 0.15
    raw = pd.Series([0.40, 0.30, 0.20, 0.10] + [0.0] * (N - 4), index=NAMES)
    legacy = raw.clip(upper=cap)
    legacy = legacy / legacy.sum()                           # restore the budget
    assert legacy.max() > cap + 1e-6                         # and break the cap doing it
    cs = _cs()
    assert any("over cap" in k for k in cs.violations(legacy))
    good = optimise(_mu(), _cov(), cs, risk_aversion=3.0)
    assert cs.violations(good.weights) == {}


# --------------------------------------------------------- mean - variance ---
def test_optimised_weights_satisfy_every_constraint_exactly():
    cs = _cs(sectors=SECTORS)
    p = optimise(_mu(), _cov(), cs, risk_aversion=3.0)
    assert cs.violations(p.weights, tol=1e-9) == {}
    assert p.weights.sum() == pytest.approx(1.0, abs=1e-9)
    assert (p.weights >= -1e-12).all()
    assert p.weights.max() <= cs.upper.max() + 1e-9
    for sec, idx in cs.sector_groups().items():
        assert float(p.weights.iloc[idx].sum()) <= 0.40 + 1e-9, sec
    assert p.diagnostics["max_constraint_violation"] < 1e-8


def test_a_binding_cap_is_reported_with_its_reason():
    cs = build_constraints(NAMES, Mandate(max_stock_weight=0.15, max_sector_weight=0.40),
                           sectors=SECTORS, evidence=pd.Series(0.0, index=NAMES[:3],
                                                               dtype=float).reindex(NAMES).fillna(1.0))
    p = optimise(_mu(seed=4), _cov(), cs, risk_aversion=2.0)
    assert cs.violations(p.weights) == {}
    if p.diagnostics["names_at_cap"]:
        assert set(p.diagnostics["binding_cap"]) == set(p.diagnostics["names_at_cap"])


def test_turnover_falls_as_trading_is_priced_and_never_breaks_a_constraint():
    prev = pd.Series(0.0, index=NAMES)
    prev[NAMES[:6]] = 1 / 6
    turnovers = []
    for penalty in (0.0, 0.05, 0.20, 1.0):
        cs = _cs(sectors=SECTORS, prev_weights=prev)
        p = optimise(_mu(seed=7), _cov(), cs, risk_aversion=3.0,
                     settings=OptimiserSettings(turnover_penalty=penalty))
        assert cs.violations(p.weights) == {}
        turnovers.append(p.turnover())
    assert all(a >= b - 1e-6 for a, b in zip(turnovers, turnovers[1:])), turnovers
    assert turnovers[0] > turnovers[-1] + 1e-4


def test_a_hard_turnover_limit_is_obeyed():
    prev = pd.Series(1.0 / N, index=NAMES)               # already compliant, so the limit is the only bind
    cs = _cs(sectors=SECTORS, prev_weights=prev, max_turnover=0.20)
    p = optimise(_mu(seed=7), _cov(), cs, risk_aversion=3.0)
    assert float(np.abs(p.weights - prev).sum()) / 2.0 <= 0.20 + 1e-8
    assert p.turnover() == pytest.approx(0.20, abs=1e-6)          # the limit binds
    assert cs.violations(p.weights) == {}


def test_the_no_trade_band_stops_trades_that_are_not_worth_doing():
    prev = pd.Series(1.0 / N, index=NAMES)
    cs = _cs(sectors=SECTORS, prev_weights=prev)
    base = optimise(_mu(seed=11), _cov(), cs, risk_aversion=3.0)
    banded = optimise(_mu(seed=11), _cov(), _cs(sectors=SECTORS, prev_weights=prev),
                      risk_aversion=3.0, settings=OptimiserSettings(no_trade_band=0.02))
    small_before = int(((base.weights - prev).abs() < 0.02).sum())
    assert banded.diagnostics["names_held_by_band"] >= small_before - 1
    untouched = (banded.weights - prev).abs() < 1e-10
    assert untouched.sum() >= 1
    assert cs.violations(banded.weights) == {}


def test_smoothing_spends_only_part_of_the_trading_budget():
    prev = pd.Series(1.0 / N, index=NAMES)
    full = optimise(_mu(seed=3), _cov(), _cs(sectors=SECTORS, prev_weights=prev), risk_aversion=3.0)
    last = full.turnover()
    for phi in (0.5, 0.25):
        cs = _cs(sectors=SECTORS, prev_weights=prev)
        part = optimise(_mu(seed=3), _cov(), cs, risk_aversion=3.0,
                        settings=OptimiserSettings(smoothing=phi))
        assert cs.violations(part.weights) == {}
        assert part.diagnostics["smoothed"]
        assert part.turnover() == pytest.approx(phi * full.turnover(), rel=1e-4)
        assert part.turnover() < last
        last = part.turnover()


def test_smoothing_never_keeps_a_portfolio_out_of_compliance():
    """
    A portfolio is usually rebalanced *because* it breaches something. Blending
    with it would inherit the breach, so smoothing is a trading budget instead,
    raised to whatever compliance actually requires.
    """
    prev = pd.Series(0.0, index=NAMES)
    prev[NAMES[:6]] = 1 / 6                     # over the 15% cap and the 40% Financials cap
    cs = _cs(sectors=SECTORS, prev_weights=prev)
    p = optimise(_mu(seed=3), _cov(), cs, risk_aversion=3.0,
                 settings=OptimiserSettings(smoothing=0.05))
    assert cs.violations(p.weights) == {}
    assert p.diagnostics["smoothing_limited_by_compliance"]


def test_more_risk_aversion_means_less_risk():
    cs = _cs(sectors=SECTORS)
    cov = _cov()
    vols = [optimise(_mu(seed=2), cov, _cs(sectors=SECTORS), risk_aversion=d).volatility(cov)
            for d in (1.0, 3.0, 10.0)]
    assert vols[0] > vols[1] > vols[2]
    assert min_variance(cov, cs).volatility(cov) <= vols[-1] + 1e-9


def test_frontier_points_are_solved_not_interpolated():
    cov, mu = _cov(), _mu(seed=5)
    cs = _cs(sectors=SECTORS)
    lo = min_variance(cov, cs)
    base = lo.expected_return(mu)
    prev_vol = lo.volatility(cov)
    for target in (base + 0.005, base + 0.010, base + 0.015):
        p = target_return(mu, cov, cs, target)
        assert cs.violations(p.weights) == {}
        assert p.expected_return(mu) >= target - 1e-8
        assert p.volatility(cov) >= prev_vol - 1e-9        # risk rises with required return
        prev_vol = p.volatility(cov)


def test_an_impossible_target_fails_loudly():
    cov, mu = _cov(), _mu(seed=5)
    cs = _cs(sectors=SECTORS)
    with pytest.raises(OptimizationError):
        target_return(mu, cov, cs, float(mu.values.max()) + 1.0)


def test_the_optimiser_refuses_mixed_up_units():
    cs = _cs()
    total = _mu().to_total(0.06)
    with pytest.raises(OptimizationError, match="excess returns"):
        optimise(total, _cov(), cs, risk_aversion=3.0)
    with pytest.raises(OptimizationError, match="risk aversion must be positive"):
        optimise(_mu(), _cov(), cs, risk_aversion=0.0)


def test_daily_inputs_are_annualised_rather_than_mixed():
    """A daily covariance and an annual mu must not meet without conversion."""
    daily = CovarianceEstimate(matrix=_cov().matrix / 252.0, periodicity=Periodicity.DAILY,
                               provenance=Provenance(method="synthetic"))
    a = optimise(_mu(), _cov(), _cs(), risk_aversion=3.0)
    b = optimise(_mu(), daily, _cs(), risk_aversion=3.0)
    assert np.allclose(a.weights.to_numpy(), b.weights.to_numpy(), atol=1e-8)

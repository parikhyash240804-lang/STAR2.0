"""Historical replay, conditional scenarios, reverse stress, and the standing monitors."""
import numpy as np
import pandas as pd
import pytest

from star.core.errors import DataError, EstimationError
from star.core.types import CovarianceEstimate, Periodicity, Provenance
from star.data.fixture import FixtureProvider
from star.data.quality import assess
from star.risk.monitors import (average_pairwise_correlation, correlation_monitor,
                                diversification_ratio_series, gold_sleeve)
from star.stress.historical import EPISODES, Episode, replay, replay_all
from star.stress.hypothetical import apply_scenario, conditional_returns, sector_shock
from star.stress.reverse import loss_at_sigmas, reverse_stress

NAMES = ["A", "B", "C", "D"]


def _cov(rho=0.4, sd=(0.20, 0.25, 0.30, 0.15), names=NAMES):
    sd = np.asarray(sd, dtype=float)
    c = np.full((len(names), len(names)), rho)
    np.fill_diagonal(c, 1.0)
    return CovarianceEstimate(matrix=pd.DataFrame(c * np.outer(sd, sd), index=names, columns=names),
                              periodicity=Periodicity.ANNUAL, provenance=Provenance(method="synthetic"))


def _prices(names=NAMES, start="2007-01-01", periods=1200, seed=0, drift=0.0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=periods)
    r = rng.normal(drift, 0.012, (periods, len(names)))
    return pd.DataFrame(100 * np.exp(np.cumsum(r, axis=0)), index=idx, columns=names)


# ------------------------------------------------------------ historical ---
def test_replay_uses_the_actual_return_of_each_holding():
    idx = pd.bdate_range("2020-02-19", "2020-03-23")
    close = pd.DataFrame({"A": np.linspace(100, 60, len(idx)),      # -40%
                          "B": np.linspace(100, 90, len(idx))}, index=idx)   # -10%
    w = pd.Series({"A": 0.5, "B": 0.5})
    res = replay("covid_2020", close, w)
    assert res.per_holding.loc["A", "return"] == pytest.approx(-0.40, abs=1e-6)
    assert res.portfolio_return == pytest.approx(-0.25, abs=1e-6)
    assert res.coverage == pytest.approx(1.0)
    assert res.proxied == () and res.missing == ()
    assert "COVID" in res.headline and "-25" in res.headline.replace("−", "-")


def test_a_holding_without_history_is_proxied_through_beta_and_disclosed():
    idx = pd.bdate_range("2020-02-19", "2020-03-23")
    close = pd.DataFrame({"A": np.linspace(100, 60, len(idx))}, index=idx)
    bench = pd.Series(np.linspace(100, 62, len(idx)), index=idx)        # -38%
    w = pd.Series({"A": 0.5, "NEWCO": 0.5})
    res = replay("covid_2020", close, w, benchmark=bench, betas=pd.Series({"NEWCO": 1.2}))
    assert res.proxied == ("NEWCO",)
    assert res.per_holding.loc["NEWCO", "return"] == pytest.approx(1.2 * -0.38, abs=1e-6)
    assert res.per_holding.loc["NEWCO", "source"] == "beta-proxied"
    assert res.coverage == pytest.approx(0.5)
    assert "only 50% of the book has real data" in res.headline
    assert res.provenance.degradations


def test_a_proxied_holding_cannot_lose_more_than_everything():
    idx = pd.bdate_range("2008-01-08", "2009-03-09")
    close = pd.DataFrame({"A": np.linspace(100, 45, len(idx))}, index=idx)
    bench = pd.Series(np.linspace(100, 40, len(idx)), index=idx)        # -60%
    w = pd.Series({"A": 0.5, "HIGHBETA": 0.5})
    res = replay("gfc_2008", close, w, benchmark=bench, betas=pd.Series({"HIGHBETA": 2.5}))
    assert res.per_holding.loc["HIGHBETA", "return"] == -1.0            # not -150%
    # and no arbitrary 1.3x multiplier or -80% floor decides the answer (legacy M6-04)
    assert res.per_holding.loc["A", "return"] == pytest.approx(-0.55, abs=1e-6)


def test_a_holding_that_cannot_be_covered_is_excluded_and_named():
    idx = pd.bdate_range("2020-02-19", "2020-03-23")
    close = pd.DataFrame({"A": np.linspace(100, 60, len(idx))}, index=idx)
    res = replay("covid_2020", close, pd.Series({"A": 0.5, "GHOST": 0.5}))
    assert res.missing == ("GHOST",)
    assert res.portfolio_return == pytest.approx(-0.40)   # the covered part, not diluted by a fake zero
    assert "excluded" in res.provenance.degradations[-1]


def test_partial_windows_are_refused_rather_than_quietly_truncated():
    """The legacy accepted a window with as few as five rows and said nothing."""
    idx = pd.bdate_range("2020-03-16", "2020-03-23")      # only the last week of the crash
    close = pd.DataFrame({"A": np.linspace(100, 95, len(idx))}, index=idx)
    with pytest.raises(DataError, match="no holding has data or a proxy"):
        replay("covid_2020", close, pd.Series({"A": 1.0}))


def test_replay_all_ranks_episodes_and_keeps_coverage_beside_each_figure():
    close = _prices(periods=5000, seed=1)
    w = pd.Series(0.25, index=NAMES)
    table = replay_all(close, w)
    assert len(table) == len(EPISODES)
    assert list(table.columns) == ["episode", "start", "end", "portfolio_return",
                                   "benchmark_return", "coverage", "proxied", "note"]
    covered = table.dropna(subset=["portfolio_return"])
    assert (covered["portfolio_return"].diff().dropna() >= -1e-12).all()   # worst first
    assert (covered["coverage"] <= 1.0).all()


def test_episodes_are_dated_windows_a_reader_can_check():
    for ep in EPISODES:
        assert pd.Timestamp(ep.start) < pd.Timestamp(ep.end)
        assert ep.what_happened and len(ep.what_happened) > 20


# ---------------------------------------------------------- hypothetical ---
def test_a_shock_reaches_correlated_names_and_spares_independent_ones():
    names = ["SHOCKED", "TWIN", "LONER"]
    sd = np.array([0.2, 0.2, 0.2])
    c = np.array([[1.0, 0.9, 0.0], [0.9, 1.0, 0.0], [0.0, 0.0, 1.0]])
    cov = CovarianceEstimate(matrix=pd.DataFrame(c * np.outer(sd, sd), index=names, columns=names),
                             periodicity=Periodicity.ANNUAL, provenance=Provenance(method="t"))
    r = conditional_returns(cov, {"SHOCKED": -0.20})
    assert r["SHOCKED"] == pytest.approx(-0.20)
    assert r["TWIN"] == pytest.approx(-0.18)             # 0.9 beta to the shocked name
    assert r["LONER"] == pytest.approx(0.0, abs=1e-12)


def test_scenario_impact_splits_direct_damage_from_contagion():
    cov = _cov(rho=0.5)
    w = pd.Series(0.25, index=NAMES)
    res = apply_scenario(w, cov, {"A": -0.30}, name="A collapses")
    assert res.portfolio_return < res.contributions["A"]     # the rest moves too
    assert "from the names shocked directly" in res.headline
    assert res.returns["A"] == pytest.approx(-0.30)
    assert res.provenance.degradations                       # correlations rise in a crisis


def test_a_sector_shock_moves_every_member_of_the_sector():
    cov = _cov()
    w = pd.Series(0.25, index=NAMES)
    sectors = pd.Series({"A": "Fin", "B": "Fin", "C": "IT", "D": "IT"})
    res = sector_shock(w, cov, sectors, "Fin", -0.20)
    assert res.returns["A"] == res.returns["B"] == pytest.approx(-0.20)
    assert res.returns["C"] < 0                              # IT drags along
    assert set(res.shocked) == {"A", "B"}
    with pytest.raises(EstimationError, match="no holding is in sector"):
        sector_shock(w, cov, sectors, "Pharma", -0.1)


def test_scenarios_refuse_names_outside_the_universe():
    cov = _cov()
    with pytest.raises(EstimationError, match="not in the universe"):
        conditional_returns(cov, {"ZZZ": -0.1})
    with pytest.raises(EstimationError, match="at least one name"):
        conditional_returns(cov, {})
    with pytest.raises(EstimationError, match="finite"):
        conditional_returns(cov, {"A": np.nan})


def test_a_scenario_is_one_matrix_multiply_so_the_brief_can_recompute_it_live():
    """The property that makes the custom-scenario control possible offline."""
    cov = _cov()
    w = pd.Series(0.25, index=NAMES)
    a = apply_scenario(w, cov, {"A": -0.10}).portfolio_return
    b = apply_scenario(w, cov, {"A": -0.20}).portfolio_return
    assert b == pytest.approx(2 * a)                         # linear in the shock, as it must be


# --------------------------------------------------------------- reverse ---
def test_reverse_stress_reports_the_loss_in_standard_deviations():
    cov = _cov(rho=0.4)
    w = pd.Series(0.25, index=NAMES)
    rs = reverse_stress(w, cov, 0.20, horizon_days=252)
    assert rs.portfolio_volatility > 0
    assert rs.sigmas == pytest.approx(0.20 / rs.portfolio_volatility)
    assert float(w @ rs.scenario) == pytest.approx(-0.20)     # the scenario really costs the target
    assert rs.contributions.sum() == pytest.approx(-0.20)
    assert "sigma move" in rs.headline


def test_the_same_loss_is_a_bigger_shock_for_a_quieter_portfolio():
    w = pd.Series(0.25, index=NAMES)
    loud = reverse_stress(w, _cov(sd=(0.35, 0.35, 0.35, 0.35)), 0.20)
    quiet = reverse_stress(w, _cov(sd=(0.03, 0.03, 0.03, 0.03)), 0.20)
    assert quiet.sigmas > loud.sigmas
    assert quiet.probability_if_normal < loud.probability_if_normal
    # a 20% loss on a book with ~26% volatility is under one sigma: likely, not a stress
    assert loud.sigmas < 1.5 and "well within normal variation" in loud.plausibility
    assert "beyond anything" in quiet.plausibility
    # and the middle of the range reads as a crisis rather than as routine
    middling = reverse_stress(pd.Series(0.25, index=NAMES), _cov(sd=(0.07,) * 4), 0.20)
    assert 3.0 <= middling.sigmas < 5.0 and "genuine crisis" in middling.plausibility


def test_the_reverse_scenario_names_the_risk_positions_not_the_big_ones():
    names = ["BIG_QUIET", "SMALL_WILD"]
    sd = np.array([0.05, 0.60])
    cov = CovarianceEstimate(matrix=pd.DataFrame(np.diag(sd ** 2), index=names, columns=names),
                             periodicity=Periodicity.ANNUAL, provenance=Provenance(method="t"))
    w = pd.Series({"BIG_QUIET": 0.8, "SMALL_WILD": 0.2})
    rs = reverse_stress(w, cov, 0.20)
    assert abs(rs.contributions["SMALL_WILD"]) > abs(rs.contributions["BIG_QUIET"])


def test_horizon_scales_the_reverse_stress_and_the_helper_agrees():
    cov = _cov()
    w = pd.Series(0.25, index=NAMES)
    year = reverse_stress(w, cov, 0.20, horizon_days=252)
    month = reverse_stress(w, cov, 0.20, horizon_days=21)
    assert month.sigmas > year.sigmas                         # a month is less time for it to happen
    assert loss_at_sigmas(w, cov, year.sigmas, 252) == pytest.approx(0.20)
    for bad in (0.0, -0.1):
        with pytest.raises(EstimationError, match="target loss must be positive"):
            reverse_stress(w, cov, bad)


# -------------------------------------------------------------- monitors ---
def test_the_correlation_monitor_notices_when_holdings_converge():
    rng = np.random.default_rng(3)
    n = 600
    idx = pd.bdate_range("2022-01-03", periods=n)
    independent = rng.normal(0, 0.01, (n, 4))
    common = rng.normal(0, 0.01, n)
    together = independent.copy()
    together[400:] = 0.3 * independent[400:] + 0.9 * common[400:, None]   # they converge late
    R = pd.DataFrame(together, index=idx, columns=NAMES)
    s = average_pairwise_correlation(R, window=60)
    assert s.iloc[-1] > s.iloc[100] + 0.3
    mon = correlation_monitor(R, window=60)
    assert mon.percentile > 80 and mon.change_3m > 0
    assert "diversification is weaker than usual" in mon.headline


def test_the_diversification_ratio_falls_as_holdings_converge():
    rng = np.random.default_rng(4)
    n = 500
    idx = pd.bdate_range("2022-01-03", periods=n)
    common = rng.normal(0, 0.01, n)
    data = rng.normal(0, 0.01, (n, 4))
    data[300:] = 0.2 * data[300:] + 0.95 * common[300:, None]
    R = pd.DataFrame(data, index=idx, columns=NAMES)
    s = diversification_ratio_series(R, pd.Series(0.25, index=NAMES), window=60)
    assert s.iloc[-1] < s.iloc[50]
    assert (s > 1.0).all()


def test_the_gold_sleeve_table_shows_the_trade_off_without_recommending_it():
    rng = np.random.default_rng(6)
    n = 900
    idx = pd.bdate_range("2021-01-04", periods=n)
    equity = pd.Series(rng.normal(0.0004, 0.012, n), index=idx)
    gold = pd.Series(rng.normal(0.0002, 0.008, n), index=idx)      # independent and calmer
    table = gold_sleeve(equity, gold)
    assert list(table.index) == [0.0, 0.05, 0.10, 0.15]
    assert table.loc[0.0, "volatility_change"] == 0.0
    assert table.loc[0.15, "volatility"] < table.loc[0.0, "volatility"]
    assert table.loc[0.15, "shortfall_change"] < 0                  # less tail loss
    assert "return_change" in table.columns                         # and the cost is shown too
    with pytest.raises(EstimationError, match="at least 250"):
        gold_sleeve(equity.iloc[:100], gold.iloc[:100])


def test_monitors_run_on_the_real_universe():
    p = FixtureProvider.default()
    if not (p.root / "nse_close.parquet").exists():
        pytest.skip("fixture not built")
    panel = p.prices(p.universe)
    kept = assess(panel.close, min_adv_inr=0.0).kept[:12]
    close = panel.close[kept]
    R = np.log(close / close.shift(1)).dropna()
    mon = correlation_monitor(R, window=60)
    assert -1.0 < mon.current < 1.0 and 0 <= mon.percentile <= 100
    w = pd.Series(1 / len(kept), index=kept)
    dr = diversification_ratio_series(R, w, window=60)
    assert (dr > 1.0).all() and len(dr) > 100

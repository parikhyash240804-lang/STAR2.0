"""GJR-GARCH-t conditional volatility: recovery, term structure, degradation, real data."""
import numpy as np
import pandas as pd
import pytest
from arch.univariate import GARCH, ConstantMean, StudentsT

from star.core.errors import EstimationError
from star.data.fixture import FixtureProvider
from star.data.quality import assess
from star.data.view import DataView
from star.estimators import garch as G
from star.estimators.covariance import estimate_covariance

TRUE = {"mu": 0.05, "omega": 0.03, "alpha": 0.04, "gamma": 0.08, "beta": 0.90, "nu": 6.0}


def _simulate(n, *, gamma=TRUE["gamma"], beta=TRUE["beta"], seed=7):
    """Percent-scale GJR-GARCH-t path, returned as daily log returns."""
    assert TRUE["alpha"] + gamma / 2 + beta < 1, "simulate a stationary process or the test is meaningless"
    am = ConstantMean(None)
    am.volatility = GARCH(p=1, o=1, q=1)
    am.distribution = StudentsT(seed=np.random.default_rng(seed))
    sim = am.simulate([TRUE["mu"], TRUE["omega"], TRUE["alpha"], gamma, beta, TRUE["nu"]],
                      n, burn=1000)
    return pd.Series(sim["data"].to_numpy() / 100.0, index=pd.bdate_range("2010-01-04", periods=n))


@pytest.fixture(scope="module")
def sim_fit():
    r = _simulate(4000)
    return r, G.forecast_variance(r, 252, ticker="SIM", settings=G.GarchSettings(lookback=4000))


def test_recovers_the_parameters_it_was_simulated_from(sim_fit):
    _, f = sim_fit
    assert f.method == "gjr_garch_t"
    p = f.params
    assert p["alpha[1]"] == pytest.approx(TRUE["alpha"], abs=0.03)
    assert p["gamma[1]"] == pytest.approx(TRUE["gamma"], abs=0.05)
    assert p["beta[1]"] == pytest.approx(TRUE["beta"], abs=0.03)
    assert p["nu"] == pytest.approx(TRUE["nu"], rel=0.35)
    true_persistence = TRUE["alpha"] + TRUE["gamma"] / 2 + TRUE["beta"]
    assert f.persistence == pytest.approx(true_persistence, abs=0.02)


def test_leverage_effect_is_found_when_present_and_not_invented_when_absent():
    """The whole reason for GJR over plain GARCH: a fall must matter more than a rise."""
    asym = G.forecast_variance(_simulate(4000, gamma=0.14, beta=0.85, seed=3), 5,
                               settings=G.GarchSettings(lookback=4000))
    sym = G.forecast_variance(_simulate(4000, gamma=0.0, beta=0.89, seed=3), 5,
                              settings=G.GarchSettings(lookback=4000))
    assert asym.method == sym.method == "gjr_garch_t"
    assert asym.params["gamma[1]"] > 0.06
    assert abs(sym.params["gamma[1]"]) < 0.04
    # and the asymmetry is directional: a -3% day raises tomorrow's variance more than +3%
    p = asym.params
    shock = 3.0 ** 2                                        # percent scale, as fitted
    up = p["omega"] + p["alpha[1]"] * shock
    down = p["omega"] + (p["alpha[1]"] + p["gamma[1]"]) * shock
    assert down > up * 1.2


def test_one_day_ahead_matches_the_model_recursion(sim_fit):
    r, f = sim_fit
    s = G.GarchSettings(lookback=4000)
    res = G._fit("SIM", r, s, None)
    p = f.params
    e = r.iloc[-1] * s.scale - p["mu"]
    s2_t = res.conditional_volatility[-1] ** 2
    expected = (p["omega"] + (p["alpha[1]"] + p["gamma[1]"] * (e < 0)) * e ** 2
                + p["beta[1]"] * s2_t) / s.scale ** 2
    assert f.daily_variance[0] == pytest.approx(expected, rel=1e-10)


def test_term_structure_mean_reverts_towards_the_long_run_level(sim_fit):
    _, f = sim_fit
    lr = f.long_run_variance
    gap = np.abs(f.daily_variance - lr)
    assert np.all(np.diff(gap) < 1e-15)                     # monotonically closer, never further
    assert gap[-1] < gap[0]
    assert f.daily_variance[-1] == pytest.approx(lr, rel=0.25)
    # direction is the sign of the starting gap, not hard-coded
    assert np.all(np.sign(np.diff(f.daily_variance)) == np.sign(lr - f.daily_variance[0]))


def test_horizon_aggregation_is_consistent(sim_fit):
    _, f = sim_fit
    assert f.horizon == 252
    assert f.cumulative_variance(10) == pytest.approx(f.daily_variance[:10].sum())
    assert f.average_daily_variance(10) == pytest.approx(f.cumulative_variance(10) / 10)
    assert f.annual_vol(10) == pytest.approx(np.sqrt(f.average_daily_variance(10) * 252))
    for bad in (0, 253, -1):
        with pytest.raises(EstimationError):
            f.cumulative_variance(bad)


def test_horizon_beyond_the_extrapolation_cap_is_refused():
    r = _simulate(600)
    with pytest.raises(EstimationError, match="between 1 and 756"):
        G.forecast_variance(r, G.MAX_HORIZON + 1)


def test_short_history_degrades_to_ewma_and_says_so():
    r = _simulate(300)
    f = G.forecast_variance(r, 20, ticker="SHORT")
    assert f.method == "ewma"
    assert len(set(f.daily_variance)) == 1                  # flat: EWMA has no term structure
    assert f.provenance.degradations and "300 sessions" in f.provenance.degradations[0]
    assert "EWMA" in f.provenance.method


def test_a_failed_fit_degrades_instead_of_raising(monkeypatch):
    monkeypatch.setattr(G, "_fit", lambda *a, **k: None)
    f = G.forecast_variance(_simulate(600), 10, ticker="BADFIT")
    assert f.method == "ewma"
    assert "did not converge" in f.provenance.degradations[0]
    assert np.isfinite(f.daily_variance).all() and (f.daily_variance > 0).all()


def test_cache_avoids_refitting_the_same_window(sim_fit):
    r, _ = sim_fit
    cache = G.FitCache()
    s = G.GarchSettings(lookback=1000)
    a = G.forecast_variance(r, 30, ticker="SIM", settings=s, cache=cache)
    b = G.forecast_variance(r, 30, ticker="SIM", settings=s, cache=cache)
    assert (cache.misses, cache.hits) == (1, 1)
    assert np.allclose(a.daily_variance, b.daily_variance)
    # a different window is a different key
    G.forecast_variance(r.iloc[:-1], 30, ticker="SIM", settings=s, cache=cache)
    assert cache.misses == 2


@pytest.fixture(scope="module")
def view():
    p = FixtureProvider.default()
    if not (p.root / "nse_close.parquet").exists():
        pytest.skip("fixture not built")
    panel = p.prices(p.universe)
    rep = assess(panel.close, min_adv_inr=0.0)
    return DataView.build(panel, panel.sessions.max()), rep.kept


def _log_returns(v, names):
    close = v.ohlc()["close"][names]
    return np.log(close / close.shift(1)).iloc[1:]


def test_real_nse_forecasts_are_valid_and_plausible(view):
    v, kept = view
    names = kept[:8]
    fcs = G.forecast_panel(_log_returns(v, names), 252, tickers=names, as_of=v.as_of)
    assert list(fcs) == names
    for t, f in fcs.items():
        assert np.isfinite(f.daily_variance).all() and (f.daily_variance > 0).all()
        assert 0.08 < f.annual_vol(1) < 1.50, t            # a single day can be turbulent
        assert 0.08 < f.annual_vol(252) < 0.90, t          # the one-year average cannot be
        if f.method == "gjr_garch_t":
            assert 0.0 < f.persistence < 1.0
            assert f.nu > 2.1
            assert "Student-t" in f.provenance.method


def test_forecast_uses_nothing_after_as_of(view):
    v, kept = view
    as_of = pd.Timestamp("2025-03-28")
    names = kept[:3]
    early = _log_returns(DataView.build(v.panel, as_of), names)
    truncated = _log_returns(v, names).loc[:as_of]
    a = G.forecast_panel(early, 21, tickers=names, as_of=as_of)
    b = G.forecast_panel(truncated, 21, tickers=names, as_of=as_of)
    for t in names:
        assert np.allclose(a[t].daily_variance, b[t].daily_variance)


def test_forecast_panel_names_missing_tickers(view):
    v, kept = view
    with pytest.raises(EstimationError, match="NOPE"):
        G.forecast_panel(_log_returns(v, kept[:2]), 5, tickers=["NOPE.NS"])


def test_horizon_covariance_keeps_correlation_and_matches_the_conditional_one_day(view):
    v, kept = view
    names = kept[:8]
    cov = estimate_covariance(v.ohlc(), tickers=names, as_of=v.as_of)
    fcs = G.forecast_panel(_log_returns(v, names), 252, tickers=names, as_of=v.as_of)
    one = G.scale_covariance_to_horizon(cov, fcs, 1)
    year = G.scale_covariance_to_horizon(cov, fcs, 252)
    assert np.allclose(one.correlation().to_numpy(), cov.correlation().to_numpy(), atol=1e-12)
    assert one.is_psd() and year.is_psd()
    # h=1 reproduces the conditional one-day variance exactly, not the unconditional one
    for t in names:
        assert one.matrix.loc[t, t] == pytest.approx(fcs[t].daily_variance[0])
        assert year.matrix.loc[t, t] == pytest.approx(fcs[t].average_daily_variance(252))
    assert "ledoit_wolf" in year.corr_method

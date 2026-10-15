"""
The forward distribution: one run, every horizon, and the properties that make it
trustworthy.

The legacy's simulation is the benchmark for what to avoid: i.i.d. days, constant
covariance, a drift taken from the optimiser's own view, and a one-year figure
presented as a one-day one.
"""
import numpy as np
import pandas as pd
import pytest
from scipy import stats

from star.core.errors import EstimationError
from star.core.horizons import HorizonGrid
from star.data.fixture import FixtureProvider
from star.data.quality import assess
from star.forecast.engine import (DriftAssumption, MAX_HORIZON, _plan_chunks, drift_sensitivity,
                                  simulate)
from star.forecast.paths import (BootstrapGenerator, FilteredHistoricalGenerator, StudentTGenerator,
                                 iid_indices, regime_similarity_weights, stationary_block_indices)


def _history(n=1500, sd=0.01, mu=0.0003, seed=0, rho=0.8):
    """Portfolio and a correlated benchmark, as daily log returns."""
    rng = np.random.default_rng(seed)
    b = rng.normal(mu, sd, n)
    p = rho * b + np.sqrt(1 - rho ** 2) * rng.normal(mu, sd, n)
    return p, b


# ------------------------------------------------------------- generators ---
def test_block_bootstrap_keeps_consecutive_days_together():
    rng = np.random.default_rng(1)
    idx = stationary_block_indices(500, 200, 400, mean_block=10, rng=rng)
    steps = np.diff(idx, axis=0)
    consecutive = float(np.mean(steps == 1))
    assert 0.80 < consecutive < 0.95                      # about 1 - 1/10 of steps continue a block
    iid = iid_indices(500, 200, 400, rng=rng)
    assert float(np.mean(np.diff(iid, axis=0) == 1)) < 0.05
    assert idx.min() >= 0 and idx.max() < 500


def test_the_bootstrap_reproduces_the_history_it_was_given():
    p, b = _history(sd=0.012, mu=0.0004)
    gen = BootstrapGenerator(portfolio=p, benchmark=b, mean_block=10)
    port, bench = gen.generate(250, 4000, np.random.default_rng(2))
    assert port.shape == (250, 4000) and bench.shape == (250, 4000)
    assert np.std(port) == pytest.approx(np.std(p), rel=0.05)
    # quantiles, not the mean: with block resampling the mean of a finite draw is
    # too noisy to assert anything useful about
    for q in (5, 25, 50, 75, 95):
        assert np.percentile(port, q) == pytest.approx(np.percentile(p, q), abs=0.0015)
    # the benchmark is drawn on the same dates, so their relationship survives
    assert np.corrcoef(port.ravel(), bench.ravel())[0, 1] == pytest.approx(
        float(np.corrcoef(p, b)[0, 1]), abs=0.05)


def test_the_benchmark_is_not_simulated_independently():
    """If it were, "probability of underperforming the index" would be meaningless."""
    p, b = _history(rho=0.95)
    gen = BootstrapGenerator(portfolio=p, benchmark=b)
    port, bench = gen.generate(100, 2000, np.random.default_rng(3))
    assert np.corrcoef(port.ravel(), bench.ravel())[0, 1] > 0.85


def test_regime_weights_favour_similar_days_without_excluding_the_rest():
    p_stress = pd.Series(np.linspace(0, 1, 200))
    w = regime_similarity_weights(p_stress, current=0.9)
    assert w.sum() == pytest.approx(1.0)
    assert (w > 0).all()                                  # a calm forecast still sees bad days
    assert w[-1] > w[0] * 5
    with pytest.raises(EstimationError, match="must be in"):
        regime_similarity_weights(p_stress, current=1.4)


def test_filtered_historical_simulation_keeps_volatility_clustering_alive():
    """
    The reason FHS exists: a bad day makes the next day more volatile, so the
    252-day distribution keeps fat tails instead of converging to normal.
    """
    rng = np.random.default_rng(4)
    n_assets = 3
    resid = rng.standard_normal((800, n_assets))
    common = dict(weights=np.full(n_assets, 1 / n_assets),
                  omega=np.full(n_assets, 2e-6), alpha=np.full(n_assets, 0.10),
                  gamma=np.full(n_assets, 0.08), beta=np.full(n_assets, 0.85),
                  sigma0=np.full(n_assets, 0.012))
    gen = FilteredHistoricalGenerator(residuals=resid, **common)
    port, _ = gen.generate(252, 3000, np.random.default_rng(5))
    annual = port.sum(axis=0)
    assert stats.kurtosis(annual) > 0.3                   # excess kurtosis survives to one year
    # a flat-volatility generator on the same residuals loses it
    flat = FilteredHistoricalGenerator(residuals=resid, **{**common, "alpha": np.zeros(n_assets),
                                                           "gamma": np.zeros(n_assets),
                                                           "beta": np.zeros(n_assets),
                                                           "omega": np.full(n_assets, 0.012 ** 2)})
    flat_annual = flat.generate(252, 3000, np.random.default_rng(5))[0].sum(axis=0)
    assert stats.kurtosis(annual) > stats.kurtosis(flat_annual)


def test_generators_validate_their_inputs():
    with pytest.raises(EstimationError, match="at least 60 daily"):
        BootstrapGenerator(portfolio=np.zeros(10))
    with pytest.raises(EstimationError, match="same length"):
        BootstrapGenerator(portfolio=np.zeros(100), benchmark=np.zeros(50))
    with pytest.raises(EstimationError, match="tail index"):
        StudentTGenerator(cov=np.array([[1e-4]]), nu=1.5)
    with pytest.raises(EstimationError, match="one entry per asset"):
        FilteredHistoricalGenerator(residuals=np.zeros((100, 3)), weights=np.ones(2),
                                    omega=np.ones(3), alpha=np.ones(3), gamma=np.ones(3),
                                    beta=np.ones(3), sigma0=np.ones(3))


# ---------------------------------------------------------------- engine ---
@pytest.fixture(scope="module")
def run():
    p, b = _history(n=2000, sd=0.011, mu=0.0004, seed=7)
    gen = BootstrapGenerator(portfolio=p, benchmark=b, mean_block=10)
    return simulate(gen, horizon=252, n_paths=8000, seed=11), gen


def test_every_view_reads_the_same_stored_cell(run):
    """The structural cure for a 1-day and a 1-year figure that contradict."""
    ts, _ = run
    for h in (1, 37, 252):
        at = ts.at(h)
        assert at["quantiles"][5] == ts.quantiles[5][h - 1]
        assert at["p_loss"] == ts.p_loss[h - 1]
    band = ts.band(5, 95)
    assert band.loc[37, "low"] == ts.at(37)["quantiles"][5]
    assert band.loc[37, "median"] == ts.at(37)["quantiles"][50]


def test_any_horizon_is_a_lookup_not_a_rerun(run):
    ts, _ = run
    assert ts.max_days == 252
    assert ts.at(2)["days"] == 2 and ts.at(37)["days"] == 37
    assert len(ts.days) == 252
    with pytest.raises(EstimationError, match="outside this run"):
        ts.at(253)
    with pytest.raises(EstimationError, match="outside this run"):
        ts.at(0)


def test_the_distribution_widens_with_time_and_the_median_moves_with_drift(run):
    ts, _ = run
    widths = ts.quantiles[95] - ts.quantiles[5]
    # day-to-day the estimate wobbles by Monte Carlo noise, so check the trend on
    # a grid rather than asserting a monotonicity the estimator cannot deliver
    marks = [widths[d - 1] for d in (1, 21, 63, 126, 252)]
    assert all(b > a for a, b in zip(marks, marks[1:]))
    assert widths[-1] > widths[0] * 3
    assert ts.quantiles[50][-1] > ts.quantiles[50][0]     # positive assumed drift lifts the median
    assert (ts.quantiles[99] >= ts.quantiles[95]).all()
    assert (ts.quantiles[5] >= ts.quantiles[1]).all()


def test_drawdown_is_measured_along_the_path_not_sliced_from_the_end(run):
    ts, _ = run
    for q in (50, 95):
        assert (np.diff(ts.dar[q]) >= -1e-9).all()        # the worst-so-far can only grow
    assert ts.dar[95][-1] > ts.dar[50][-1]
    # a portfolio can end a year up and still have fallen hard on the way
    end_loss = ts.p_loss[-1]
    assert ts.dar[95][-1] > end_loss * 0.5


def test_probabilities_are_between_zero_and_one_and_behave_sensibly(run):
    ts, _ = run
    for arr in (ts.p_loss, ts.p_below_rf, ts.p_below_benchmark):
        assert arr is not None
        assert ((arr >= 0) & (arr <= 1)).all()
    assert ts.p_below_rf[-1] > ts.p_loss[-1]              # beating cash is harder than breaking even


def test_expected_shortfall_grows_with_the_horizon_and_with_confidence(run):
    ts, _ = run
    assert (ts.es[99] >= ts.es[95] - 1e-9).all()
    assert ts.es[95][-1] > ts.es[95][0]


def test_the_matching_closed_form_is_reproduced():
    """On i.i.d. normal input the engine must agree with the analytic answer."""
    sd = 0.01
    gen = StudentTGenerator(cov=np.array([[sd ** 2]]), nu=200)   # ~normal
    ts = simulate(gen, horizon=100, n_paths=40_000,
                  drift=DriftAssumption(kind="zero"), seed=3)
    h = 100
    expected_sd = sd * np.sqrt(h)
    log_q5 = np.log(ts.quantiles[5][h - 1])
    # zero drift means E[value] = 1, so the log median sits at -var/2
    assert log_q5 == pytest.approx(-0.5 * expected_sd ** 2 + stats.norm.ppf(0.05) * expected_sd,
                                   abs=0.004)
    assert np.log(ts.quantiles[50][h - 1]) == pytest.approx(-0.5 * expected_sd ** 2, abs=0.004)


def test_clustered_volatility_gives_a_smaller_annual_loss_than_the_square_root_rule():
    """
    A one-day number times sqrt(252) is not a one-year number - and on clustered
    data it is not even close. This is the quantitative form of legacy M6-02.
    """
    rng = np.random.default_rng(9)
    resid = rng.standard_normal((1000, 1))
    gen = FilteredHistoricalGenerator(residuals=resid, weights=np.ones(1), omega=np.array([2e-6]),
                                      alpha=np.array([0.10]), gamma=np.array([0.08]),
                                      beta=np.array([0.85]), sigma0=np.array([0.012]))
    ts = simulate(gen, horizon=252, n_paths=8000, drift=DriftAssumption(kind="zero"), seed=4)
    one_day_loss = 1.0 - ts.quantiles[5][0]
    year_loss = 1.0 - ts.quantiles[5][251]
    assert year_loss < one_day_loss * np.sqrt(252)


def test_drift_is_an_assumption_that_is_named_and_can_be_varied():
    p, b = _history(n=1200, mu=0.0012, seed=13)           # a history with strong past returns
    gen = BootstrapGenerator(portfolio=p, benchmark=b)
    default = simulate(gen, horizon=252, n_paths=4000, seed=5)
    historical = simulate(gen, horizon=252, n_paths=4000,
                          drift=DriftAssumption(kind="historical"), seed=5)
    assert "no alpha assumed" in default.drift.describe()
    assert historical.at(252)["median_return"] > default.at(252)["median_return"]
    assert historical.p_loss[-1] < default.p_loss[-1]     # assuming the past repeats flatters it
    assert "drift" in default.provenance.window


def test_the_drift_sensitivity_strip_shows_how_much_rests_on_the_assumption():
    p, b = _history(n=1200, mu=0.0010, seed=17)
    gen = BootstrapGenerator(portfolio=p, benchmark=b)
    table = drift_sensitivity(gen, horizon=252, n_paths=2000, drifts=(
        DriftAssumption(kind="zero"), DriftAssumption(kind="risk_free"),
        DriftAssumption(kind="rf_plus_erp"), DriftAssumption(kind="historical")))
    assert list(table.index) == ["zero", "risk_free", "rf_plus_erp", "historical"]
    assert table.loc["zero", "p_loss"] > table.loc["rf_plus_erp", "p_loss"]
    assert table["median_return"].is_monotonic_increasing


def test_the_expectation_lands_where_the_drift_says_it_should():
    sd = 0.01
    gen = StudentTGenerator(cov=np.array([[sd ** 2]]), nu=200)
    target = 0.10
    ts = simulate(gen, horizon=252, n_paths=40_000,
                  drift=DriftAssumption(kind="custom", custom_annual_return=target), seed=6)
    # the half-variance correction means the MEAN hits the target, not the median
    assert ts.quantiles[50][-1] < 1 + target
    assert np.log(ts.quantiles[50][-1]) == pytest.approx(
        np.log1p(target) - 0.5 * (sd ** 2 * 252), abs=0.01)


# ------------------------------------------------------------- guardrails ---
def test_memory_is_budgeted_rather_than_hoped_for():
    assert _plan_chunks(252, 20_000, 512e6) == 2_000
    assert _plan_chunks(252, 500, 512e6) == 500          # never more chunk than paths
    with pytest.raises(EstimationError, match="above the"):
        _plan_chunks(756, 500_000, 512e6)                # asks for 1.5 GB
    # and the refusal says what to do instead
    try:
        _plan_chunks(756, 500_000, 512e6)
    except EstimationError as e:
        assert "Reduce paths to about" in str(e)


def test_the_answer_does_not_depend_on_the_memory_budget():
    """Same seed, same answer - on a laptop or a server."""
    p, b = _history(n=900, seed=19)
    gen = BootstrapGenerator(portfolio=p, benchmark=b)
    big = simulate(gen, horizon=60, n_paths=6000, seed=2, max_bytes=512e6)
    small = simulate(gen, horizon=60, n_paths=6000, seed=2, max_bytes=20e6)
    assert np.allclose(big.quantiles[50], small.quantiles[50])
    assert np.allclose(big.p_loss, small.p_loss)
    assert np.allclose(big.dar[95], small.dar[95])
    # a budget too small to run is refused rather than silently changing the method
    with pytest.raises(EstimationError, match="above the"):
        simulate(gen, horizon=60, n_paths=6000, seed=2, max_bytes=1e6)


def test_long_horizons_are_capped_and_flagged():
    p, _ = _history(n=1500, seed=23)
    gen = BootstrapGenerator(portfolio=p)
    with pytest.raises(EstimationError, match="between 1 and 756"):
        simulate(gen, horizon=MAX_HORIZON + 1, n_paths=1000)
    long_run = simulate(gen, horizon=600, n_paths=2000, seed=1)
    assert any("extrapolation" in w for w in long_run.warnings)
    assert long_run.provenance.degradations
    short = simulate(gen, horizon=60, n_paths=2000, seed=1)
    assert not any("extrapolation" in w for w in short.warnings)


def test_too_few_paths_is_refused():
    p, _ = _history(n=900, seed=29)
    with pytest.raises(EstimationError, match="too few"):
        simulate(BootstrapGenerator(portfolio=p), horizon=21, n_paths=200)


def test_a_run_without_a_benchmark_says_so_instead_of_inventing_one():
    p, _ = _history(n=900, seed=31)
    ts = simulate(BootstrapGenerator(portfolio=p), horizon=21, n_paths=2000, seed=1)
    assert ts.p_below_benchmark is None
    assert ts.at(21)["p_below_benchmark"] is None
    assert any("not available" in w for w in ts.warnings)


# ------------------------------------------------------------- real data ---
def test_the_whole_thing_runs_on_the_real_universe():
    p = FixtureProvider.default()
    if not (p.root / "nse_close.parquet").exists():
        pytest.skip("fixture not built")
    panel = p.prices(p.universe)
    kept = assess(panel.close, min_adv_inr=0.0).kept
    close = panel.close[kept]
    r = np.log(close / close.shift(1)).dropna()
    w = pd.Series(1 / len(kept), index=kept)
    port = (r @ w).to_numpy()
    bench = np.log(p.benchmark("^NSEI") / p.benchmark("^NSEI").shift(1)).dropna()
    bench = bench.reindex(r.index).fillna(0.0).to_numpy()
    ts = simulate(BootstrapGenerator(portfolio=port, benchmark=bench, mean_block=10),
                  horizon=252, n_paths=8000, seed=3)
    table = ts.table()
    assert set(table.index) >= {"1D", "1M", "1Y"}
    assert (table["p_loss"].between(0, 1)).all()
    assert table.loc["1Y", "median_return"] > table.loc["1D", "median_return"]
    assert table.loc["1Y", "drawdown_at_risk_95"] > table.loc["1M", "drawdown_at_risk_95"]
    assert "%" in ts.headline(63) and "chance of ending below" in ts.headline(63)
    grid = HorizonGrid.default()
    assert all(d in grid.days for d in (1, 21, 252))

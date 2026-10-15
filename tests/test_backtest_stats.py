"""
Inference on a track record - including whether the inference itself works.

Every test here is really asking one question: would this machinery have stopped
somebody presenting noise as skill?
"""
import numpy as np
import pandas as pd
import pytest
from scipy import stats

from star.backtest.attribution import brinson, trade_statistics
from star.backtest.stats import (autocorrelation_factor, deflated_sharpe_ratio,
                                 minimum_track_record_length,
                                 probability_of_backtest_overfitting,
                                 probabilistic_sharpe_ratio, sharpe_confidence_interval,
                                 sharpe_difference_test, sharpe_ratio, sharpe_standard_error,
                                 stationary_bootstrap_interval)
from star.core.errors import DataError, EstimationError


def _returns(n=756, mu=0.0004, sd=0.01, seed=0):
    return pd.Series(np.random.default_rng(seed).normal(mu, sd, n))


# ------------------------------------------------------------------ Sharpe ---
def test_sharpe_and_its_standard_error_match_the_formulas():
    r = _returns(n=1000, mu=0.0005, sd=0.01, seed=1)
    sr = sharpe_ratio(r)
    assert sr == pytest.approx(r.mean() / r.std(ddof=1) * np.sqrt(252))
    se = sharpe_standard_error(r, adjust_autocorrelation=False)
    sr_period = r.mean() / r.std(ddof=1)
    assert se == pytest.approx(np.sqrt((1 + 0.5 * sr_period ** 2) / len(r)) * np.sqrt(252))


def test_the_confidence_interval_is_wide_on_a_short_record():
    """Three years of daily data cannot pin down a Sharpe ratio, and it says so."""
    three_years = _returns(n=756, mu=0.0006, sd=0.012, seed=2)
    sr, lo, hi = sharpe_confidence_interval(three_years)
    assert lo < sr < hi
    assert hi - lo > 1.0                              # the interval spans more than a full Sharpe
    ten_years = _returns(n=2520, mu=0.0006, sd=0.012, seed=2)
    _, lo10, hi10 = sharpe_confidence_interval(ten_years)
    assert (hi10 - lo10) < (hi - lo)


def test_smoothed_returns_are_penalised_rather_than_rewarded():
    """Serial correlation inflates a naive Sharpe; Lo's adjustment removes the flattery."""
    raw = _returns(n=1000, seed=3)
    smoothed = raw.rolling(3).mean().dropna()         # the classic illiquid-asset artefact
    assert autocorrelation_factor(smoothed) < autocorrelation_factor(raw)
    assert sharpe_standard_error(smoothed) > sharpe_standard_error(smoothed,
                                                                   adjust_autocorrelation=False)


def test_the_probabilistic_sharpe_discounts_a_flattering_shape():
    good = _returns(n=1000, mu=0.0006, sd=0.01, seed=4)
    # same mean and volatility, but occasional large losses
    skewed = good.copy()
    skewed.iloc[::100] -= 0.05
    skewed = skewed - skewed.mean() + good.mean()
    skewed = skewed / skewed.std(ddof=1) * good.std(ddof=1)
    assert stats.skew(skewed) < stats.skew(good)
    assert probabilistic_sharpe_ratio(skewed) < probabilistic_sharpe_ratio(good)
    assert 0.0 <= probabilistic_sharpe_ratio(good) <= 1.0


def test_minimum_track_record_length_is_the_sobering_number():
    weak = _returns(n=756, mu=0.0002, sd=0.012, seed=5)
    strong = _returns(n=756, mu=0.0012, sd=0.008, seed=5)
    assert minimum_track_record_length(strong) < minimum_track_record_length(weak)
    losing = _returns(n=756, mu=-0.0003, seed=6)
    assert minimum_track_record_length(losing) == float("inf")   # no record length fixes a loss
    years = minimum_track_record_length(weak) / 252
    assert years > 1


def test_the_deflated_sharpe_prices_in_how_many_things_were_tried():
    r = _returns(n=1000, mu=0.0005, sd=0.01, seed=7)
    once = deflated_sharpe_ratio(r, n_trials=1)
    many = deflated_sharpe_ratio(r, n_trials=50, trial_sharpe_variance=0.02)
    assert many < once
    assert deflated_sharpe_ratio(r, n_trials=5, trial_sharpe_variance=0.02) > many
    with pytest.raises(EstimationError, match="n_trials"):
        deflated_sharpe_ratio(r, n_trials=0)


# ------------------------------------------------------- difference test ---
def test_a_strategy_identical_to_its_benchmark_shows_no_difference():
    r = _returns(n=1000, seed=8)
    d = sharpe_difference_test(r, r)
    assert d.difference == pytest.approx(0.0, abs=1e-12)
    assert d.p_value > 0.99
    assert d.standard_error == 0.0
    assert "identical" in d.headline


def test_noise_is_not_mistaken_for_skill():
    """Size: two independent draws from the same distribution, repeatedly."""
    rejected = 0
    reps = 200
    for seed in range(reps):
        rng = np.random.default_rng(seed)
        a = pd.Series(rng.normal(0.0004, 0.01, 756))
        b = pd.Series(rng.normal(0.0004, 0.01, 756))
        rejected += sharpe_difference_test(a, b).p_value < 0.05
    assert 0.01 <= rejected / reps <= 0.12, rejected / reps


def test_a_real_edge_is_detected_when_it_is_large_enough():
    """Power: the same market plus a genuine margin."""
    detected = 0
    reps = 60
    for seed in range(reps):
        rng = np.random.default_rng(seed)
        market = rng.normal(0.0003, 0.011, 2520)
        edge = market + rng.normal(0.0006, 0.004, 2520)     # correlated, with real alpha
        detected += sharpe_difference_test(pd.Series(edge), pd.Series(market)).p_value < 0.05
    assert detected / reps > 0.5, detected / reps


def test_the_difference_test_needs_aligned_periods():
    r = _returns(n=100, seed=9)
    with pytest.raises(EstimationError, match="at least 20 aligned"):
        sharpe_difference_test(r.iloc[:10], r.iloc[:10])


# --------------------------------------------------------------- overfitting ---
def test_selecting_among_pure_noise_is_shown_to_be_worthless():
    rng = np.random.default_rng(11)
    noise = pd.DataFrame(rng.normal(0.0003, 0.01, (1000, 12)))
    res = probability_of_backtest_overfitting(noise, n_splits=8)
    assert 0.25 < res.pbo < 0.75                       # around a coin flip, as it must be
    assert res.n_configurations == 12


def test_a_genuinely_better_configuration_survives_out_of_sample():
    rng = np.random.default_rng(12)
    data = rng.normal(0.0002, 0.01, (1000, 8))
    data[:, 3] += 0.0012                               # one configuration really is better
    res = probability_of_backtest_overfitting(pd.DataFrame(data), n_splits=8)
    assert res.pbo < 0.2
    assert "carried over" in res.headline


def test_overfitting_needs_enough_data_and_more_than_one_configuration():
    rng = np.random.default_rng(13)
    with pytest.raises(EstimationError, match="two or more configurations"):
        probability_of_backtest_overfitting(pd.DataFrame(rng.normal(size=(500, 1))))
    with pytest.raises(EstimationError, match="even and at least 4"):
        probability_of_backtest_overfitting(pd.DataFrame(rng.normal(size=(500, 4))), n_splits=5)
    with pytest.raises(EstimationError, match="too few for"):
        probability_of_backtest_overfitting(pd.DataFrame(rng.normal(size=(50, 4))), n_splits=8)


def test_bootstrap_intervals_respect_serial_dependence():
    r = _returns(n=800, mu=0.0005, seed=14)
    point, lo, hi = stationary_bootstrap_interval(
        r, lambda x: float(x.mean() / x.std(ddof=1) * np.sqrt(252)), n_resamples=200)
    assert lo < point < hi
    assert point == pytest.approx(sharpe_ratio(r))


# -------------------------------------------------------------- attribution ---
def test_brinson_components_sum_exactly_to_the_difference():
    names = list("ABCD")
    groups = pd.Series({"A": "Fin", "B": "Fin", "C": "IT", "D": "IT"})
    wp = pd.Series({"A": 0.4, "B": 0.1, "C": 0.3, "D": 0.2})
    wb = pd.Series({"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25})
    rp = pd.Series({"A": 0.10, "B": -0.02, "C": 0.05, "D": 0.01})
    res = brinson(wp, rp, wb, rp, groups)
    assert res.allocation + res.selection + res.interaction == pytest.approx(res.total)
    assert res.total == pytest.approx(res.portfolio_return - res.benchmark_return)
    assert set(res.by_group.index) == {"Fin", "IT"}


def test_pure_allocation_and_pure_selection_are_separated():
    groups = pd.Series({"A": "Fin", "B": "Fin", "C": "IT", "D": "IT"})
    rb = pd.Series({"A": 0.10, "B": 0.10, "C": 0.00, "D": 0.00})
    wb = pd.Series({"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25})
    # same names, same returns, but overweight the winning sector
    over = pd.Series({"A": 0.40, "B": 0.40, "C": 0.10, "D": 0.10})
    alloc = brinson(over, rb, wb, rb, groups)
    assert alloc.selection == pytest.approx(0.0, abs=1e-12)
    assert alloc.allocation > 0

    # same weights, better names inside each sector
    rp = pd.Series({"A": 0.14, "B": 0.14, "C": 0.04, "D": 0.04})
    sel = brinson(wb, rp, wb, rb, groups)
    assert sel.allocation == pytest.approx(0.0, abs=1e-12)
    assert sel.interaction == pytest.approx(0.0, abs=1e-12)
    assert sel.selection == pytest.approx(0.04)


def test_brinson_refuses_a_holding_with_no_sector():
    groups = pd.Series({"A": "Fin"})
    w = pd.Series({"A": 0.5, "B": 0.5})
    r = pd.Series({"A": 0.1, "B": 0.1})
    with pytest.raises(DataError, match="no sector for"):
        brinson(w, r, w, r, groups)


# ------------------------------------------------------------ trade stats ---
def test_trade_statistics_score_completed_positions():
    idx = pd.bdate_range("2024-01-01", periods=10)
    prices = pd.DataFrame({"WIN": [100, 102, 104, 106, 108, 110, 110, 110, 110, 110],
                           "LOSE": [100, 98, 96, 94, 92, 90, 90, 90, 90, 90]}, index=idx)
    weights = pd.DataFrame({"WIN": [0, .5, .5, .5, .5, 0, 0, 0, 0, 0],
                            "LOSE": [0, .5, .5, .5, .5, 0, 0, 0, 0, 0]}, index=idx)
    ts = trade_statistics(weights, prices)
    assert len(ts.trades) == 2
    assert ts.hit_rate == pytest.approx(0.5)
    win = ts.trades.set_index("ticker").loc["WIN", "return"]
    lose = ts.trades.set_index("ticker").loc["LOSE", "return"]
    assert win == pytest.approx(108 / 102 - 1)
    assert lose == pytest.approx(92 / 98 - 1)
    assert ts.profit_factor == pytest.approx(win / -lose)
    assert "of 2 closed positions were profitable" in ts.headline


def test_the_worst_drawdown_while_held_is_measured_inside_the_holding_period():
    idx = pd.bdate_range("2024-01-01", periods=8)
    prices = pd.DataFrame({"A": [100, 100, 120, 60, 90, 130, 130, 130]}, index=idx)
    weights = pd.DataFrame({"A": [0, 1, 1, 1, 1, 1, 0, 0]}, index=idx)
    ts = trade_statistics(weights, prices)
    assert len(ts.trades) == 1
    assert ts.trades.iloc[0]["return"] == pytest.approx(130 / 100 - 1)   # ended up 30%
    assert ts.trades.iloc[0]["worst_drawdown_while_held"] == pytest.approx(0.5)  # fell 50% on the way


def test_positions_still_open_are_counted_separately_not_closed_at_the_last_price():
    idx = pd.bdate_range("2024-01-01", periods=6)
    prices = pd.DataFrame({"A": [100, 90, 80, 70, 60, 50]}, index=idx)
    weights = pd.DataFrame({"A": [0, 1, 1, 1, 1, 1]}, index=idx)
    ts = trade_statistics(weights, prices)
    assert ts.open_positions == 1
    assert len(ts.trades) == 0                       # the loser is not quietly marked as closed
    assert "No completed positions yet" in ts.headline

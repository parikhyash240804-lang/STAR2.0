"""
Testing the tests.

A validator is only useful if it does two things: leaves a correct model alone
(size), and rejects a wrong one (power). Both are checked here by simulation,
because a test that never rejects would pass every other test in this file while
being worthless.
"""
import numpy as np
import pandas as pd
import pytest
from scipy import stats

from star.core.errors import EstimationError
from star.risk.backtests import (acerbi_szekely_z2, basel_traffic_light, breaches,
                                 christoffersen_cc, christoffersen_independence, dynamic_quantile,
                                 kupiec_pof, validate_var)


def _normal(n, sd=0.01, seed=0):
    return pd.Series(np.random.default_rng(seed).normal(0, sd, n))


def _garch(n, seed=0, omega=2e-6, alpha=0.12, beta=0.86):
    """Volatility clustering without needing a fitted model."""
    rng = np.random.default_rng(seed)
    r = np.empty(n)
    s2 = omega / (1 - alpha - beta)
    for t in range(n):
        e = rng.normal(0, np.sqrt(s2))
        r[t] = e
        s2 = omega + alpha * e ** 2 + beta * s2
    return pd.Series(r)


# ------------------------------------------------------------------- size ---
def test_a_correct_model_is_not_rejected_more_often_than_it_should_be():
    """Size: with a true model, rejection at 5% should happen about 5% of the time."""
    c, sd, n, reps = 0.95, 0.01, 750, 300
    var = sd * stats.norm.ppf(c)
    rejects_uc = rejects_cc = 0
    for seed in range(reps):
        r = _normal(n, sd, seed)
        _, p_uc, _, _ = kupiec_pof(r, var, c)
        _, p_cc = christoffersen_cc(r, var, c)
        rejects_uc += p_uc < 0.05
        rejects_cc += p_cc < 0.05
    assert 0.01 <= rejects_uc / reps <= 0.12, rejects_uc / reps
    assert 0.01 <= rejects_cc / reps <= 0.12, rejects_cc / reps


# ------------------------------------------------------------------ power ---
def test_power_rises_with_sample_size_and_is_honestly_weak_on_short_histories():
    """
    Power: a constant VaR applied to clustered data gets the count roughly right
    but the *timing* badly wrong, which is what Christoffersen catches - eventually.
    The measured rejection rates (29% at 3 years, 84% at 10) are why the brief
    never reports "not rejected" as a clean bill of health.
    """
    c, reps = 0.95, 60
    power = {}
    for n in (750, 2500):
        rejected = 0
        for seed in range(reps):
            r = _garch(n, seed)
            var = float(-np.quantile(r, 1 - c))       # calibrated in-sample: the count looks fine
            rejected += christoffersen_cc(r, var, c)[1] < 0.05
        power[n] = rejected / reps
    assert power[2500] > power[750]                   # more data, more power
    assert power[2500] > 0.6, power
    assert power[750] < 0.6, power                    # three years cannot settle this


def test_clustered_breaches_are_what_gives_the_test_its_power():
    c, reps = 0.95, 60
    clustered = 0
    for seed in range(reps):
        r = _garch(1500, seed)
        var = float(-np.quantile(r, 1 - c))
        clustered += christoffersen_independence(r, var)[1] < 0.05
    assert clustered / reps > 0.5


def test_a_model_that_understates_risk_is_caught_by_the_count():
    r = _garch(1000, 3)
    honest = float(-np.quantile(r, 0.01))
    _, p_ok, x_ok, expected = kupiec_pof(r, honest, 0.99)
    _, p_bad, x_bad, _ = kupiec_pof(r, honest * 0.5, 0.99)     # half the risk it should carry
    assert p_ok > 0.05 and p_bad < 0.01
    assert x_bad > x_ok > 0
    assert expected == pytest.approx(10.0)


# ------------------------------------------------------ individual tests ---
def test_kupiec_counts_what_it_says_it_counts():
    n = 1000
    r = pd.Series(np.full(n, 0.01))
    r.iloc[:10] = -0.5                                  # exactly 10 breaches
    stat, p, x, expected = kupiec_pof(r, 0.2, 0.99)
    assert x == 10 and expected == pytest.approx(10.0)
    assert p > 0.9                                      # exactly as expected: no evidence against
    assert breaches(r, 0.2).sum() == 10


def test_christoffersen_separates_clustered_breaches_from_scattered_ones():
    n = 600
    spread = pd.Series(np.full(n, 0.01))
    spread.iloc[::20] = -0.5                            # every 20th day
    clustered = pd.Series(np.full(n, 0.01))
    clustered.iloc[100:130] = -0.5                      # all in one month
    _, p_spread = christoffersen_independence(spread, 0.2)
    _, p_clustered = christoffersen_independence(clustered, 0.2)
    assert p_clustered < 0.01
    assert p_clustered < p_spread


def test_dynamic_quantile_catches_predictable_breaches():
    n = 800
    predictable = pd.Series(np.full(n, 0.01))
    predictable.iloc[200:260] = -0.5
    _, p_bad = dynamic_quantile(predictable, 0.2, 0.95)
    r = _normal(n, 0.01, 7)
    _, p_ok = dynamic_quantile(r, 0.01 * stats.norm.ppf(0.95), 0.95)
    assert p_bad < 0.05 < p_ok


def test_basel_zones_match_the_published_thresholds():
    def with_breaches(k, n=250):
        r = pd.Series(np.full(n, 0.01))
        r.iloc[:k] = -0.5
        return r
    assert basel_traffic_light(with_breaches(3), 0.2, 0.99)[0] == "green"
    assert basel_traffic_light(with_breaches(4), 0.2, 0.99)[0] == "green"
    assert basel_traffic_light(with_breaches(5), 0.2, 0.99)[0] == "yellow"
    assert basel_traffic_light(with_breaches(9), 0.2, 0.99)[0] == "yellow"
    assert basel_traffic_light(with_breaches(10), 0.2, 0.99)[0] == "red"
    zone, raw, scaled = basel_traffic_light(with_breaches(10, 500), 0.2, 0.99)
    assert raw == 10 and scaled == 5 and zone == "yellow"       # scaled pro rata, raw shown too


# ----------------------------------------------- expected shortfall test ---
def test_acerbi_szekely_detects_an_expected_shortfall_that_is_too_small():
    c, n, sd = 0.975, 1000, 0.01
    rng = np.random.default_rng(11)
    r = pd.Series(rng.normal(0, sd, n))
    var = sd * stats.norm.ppf(c)
    es_true = sd * stats.norm.pdf(stats.norm.ppf(c)) / (1 - c)
    # the null must be simulated from the model under test: here, the true one
    null = rng.normal(0, sd, (400, n))
    z_ok, p_ok = acerbi_szekely_z2(r, var, es_true, c, null_returns=null)
    assert abs(z_ok) < 0.25 and p_ok > 0.05

    # a model claiming half the true tail loss: its own VaR and its own null draws
    half_var = 0.5 * sd * stats.norm.ppf(c)
    half_null = rng.normal(0, sd / 2, (400, n))
    z_bad, p_bad = acerbi_szekely_z2(r, half_var, es_true * 0.5, c, null_returns=half_null)
    assert z_bad < z_ok and p_bad < 0.05                        # tail losses worse than forecast

    z_only, p_none = acerbi_szekely_z2(r, var, es_true, c)
    assert p_none is None and z_only == pytest.approx(z_ok)


def test_a_null_drawn_from_the_wrong_model_is_refused_rather_than_trusted():
    """The mistake is invisible in the output, so it has to fail loudly instead."""
    c, n, sd = 0.975, 1000, 0.01
    rng = np.random.default_rng(11)
    r = pd.Series(rng.normal(0, sd, n))
    half_var = 0.5 * sd * stats.norm.ppf(c)
    wrong_null = rng.normal(0, sd, (200, n))                    # true model, narrow VaR
    with pytest.raises(EstimationError, match="must come from the model being tested"):
        acerbi_szekely_z2(r, half_var, sd, c, null_returns=wrong_null)


# -------------------------------------------------------------- the report ---
def test_the_verdict_says_which_way_the_model_was_wrong():
    n = 1000
    r = _normal(n, 0.01, 21)
    honest = 0.01 * stats.norm.ppf(0.99)
    clean = validate_var(r, honest, 0.99)
    assert clean.passed and "consistent with its own" in clean.headline
    assert clean.basel_zone in {"green", "yellow"}

    understated = validate_var(r, honest * 0.4, 0.99)
    assert not understated.passed
    assert "understated" in understated.headline and "breaches" in understated.headline

    overstated = validate_var(r, honest * 4, 0.99)
    assert "overstated" in overstated.headline
    assert "tighter than intended" in overstated.headline


def test_the_report_carries_every_p_value_it_used():
    r = _normal(800, 0.01, 5)
    v = 0.01 * stats.norm.ppf(0.975)
    es = 0.01 * stats.norm.pdf(stats.norm.ppf(0.975)) / 0.025
    rep = validate_var(r, v, 0.975, es_forecast=es)
    for p in (rep.kupiec_p, rep.independence_p, rep.conditional_coverage_p, rep.dq_p):
        assert 0.0 <= p <= 1.0
    assert rep.es_z2 is not None and rep.es_p is None           # no null supplied: not calibrated
    assert rep.observations == 800
    assert rep.verdict() in {"model health: acceptable", "model health: rejected"}


def test_backtests_refuse_malformed_input():
    r = _normal(100, 0.01, 1)
    with pytest.raises(EstimationError, match="positive losses"):
        kupiec_pof(r, -0.02, 0.99)
    with pytest.raises(EstimationError, match="at least 30"):
        kupiec_pof(r.iloc[:10], 0.02, 0.99)
    with pytest.raises(EstimationError, match="100 returns but 50"):
        kupiec_pof(r, np.full(50, 0.02), 0.99)


def test_a_time_varying_forecast_is_aligned_to_its_own_dates():
    idx = pd.bdate_range("2024-01-01", periods=400)
    r = pd.Series(np.random.default_rng(2).normal(0, 0.01, 400), index=idx)
    rolling_var = (r.rolling(60).std() * stats.norm.ppf(0.95)).bfill()
    rep = validate_var(r, rolling_var, 0.95)
    assert rep.observations == 400
    # the same forecast passed as a bare array must give the same answer
    assert validate_var(r, rolling_var.to_numpy(), 0.95).breaches == rep.breaches

"""Two-state Student-t HMM: recovery, determinism, no look-ahead, and the VIX cross-check."""
import numpy as np
import pandas as pd
import pytest

from star.core.errors import EstimationError
from star.data.fixture import FixtureProvider
from star.estimators import regime as R
from star.estimators.regime import CALM, STRESS, fit_regimes, filtered_probabilities, vix_bands

TRUE_A = np.array([[0.98, 0.02], [0.08, 0.92]])
TRUE_VOL = np.array([0.008, 0.025])          # daily: ~13% and ~40% annualised
TRUE_MU = np.array([0.0006, -0.0010])
TRUE_NU = np.array([12.0, 4.0])


def _simulate(n=2500, seed=4):
    rng = np.random.default_rng(seed)
    states = np.empty(n, dtype=int)
    states[0] = CALM
    for t in range(1, n):
        states[t] = rng.choice(2, p=TRUE_A[states[t - 1]])
    scale = TRUE_VOL * np.sqrt((TRUE_NU - 2) / TRUE_NU)        # so the sd is TRUE_VOL
    y = TRUE_MU[states] + scale[states] * rng.standard_t(TRUE_NU[states])
    idx = pd.bdate_range("2015-01-01", periods=n)
    return pd.Series(y, index=idx), pd.Series(states, index=idx)


@pytest.fixture(scope="module")
def fitted():
    y, states = _simulate()
    return fit_regimes(y), y, states


def test_recovers_the_regimes_it_was_simulated_from(fitted):
    fit, _, states = fitted
    assert fit.converged
    assert np.allclose(np.diag(fit.transition), np.diag(TRUE_A), atol=0.03)
    assert np.allclose(fit.volatility, TRUE_VOL, rtol=0.20)
    classified = fit.filtered["stress"].to_numpy() > 0.5
    accuracy = float((classified == (states.to_numpy() == STRESS)).mean())
    assert accuracy > 0.85, accuracy


def test_states_are_labelled_by_volatility_not_by_optimiser_order(fitted):
    fit, _, _ = fitted
    assert fit.volatility[CALM] < fit.volatility[STRESS]
    s = fit.summary()
    assert list(s.index) == ["calm", "stress"]
    assert s.loc["stress", "annual_volatility"] > s.loc["calm", "annual_volatility"]
    assert s["time_share"].sum() == pytest.approx(1.0)


def test_fat_tails_are_estimated_rather_than_assumed():
    """Gaussian data should not be handed a fat tail, and t data should be."""
    fat, _ = _simulate()
    fat_fit = fit_regimes(fat)
    rng = np.random.default_rng(1)
    n = 2500
    states = np.zeros(n, dtype=int)
    for t in range(1, n):
        states[t] = rng.choice(2, p=TRUE_A[states[t - 1]])
    gauss = pd.Series(TRUE_MU[states] + TRUE_VOL[states] * rng.standard_normal(n),
                      index=pd.bdate_range("2015-01-01", periods=n))
    gauss_fit = fit_regimes(gauss)
    assert fat_fit.nu[STRESS] < 8.0
    assert gauss_fit.nu.min() > fat_fit.nu[STRESS]


def test_the_same_window_always_gives_the_same_answer(fitted):
    """The legacy's estimates moved between runs on identical data."""
    _, y, _ = fitted
    a, b = fit_regimes(y), fit_regimes(y)
    assert np.allclose(a.transition, b.transition) and np.allclose(a.scale, b.scale)
    assert a.log_likelihood == b.log_likelihood


def test_filtered_probabilities_never_look_ahead(fitted):
    fit, y, _ = fitted
    cut = y.index[1800]
    full = filtered_probabilities(y, fit.mu, fit.scale, fit.nu, fit.transition)
    early = filtered_probabilities(y.loc[:cut], fit.mu, fit.scale, fit.nu, fit.transition)
    assert np.allclose(full.loc[:cut].to_numpy(), early.to_numpy(), atol=1e-12)


def test_smoothed_probabilities_are_not_reachable_at_all():
    """A backtest that uses smoothed probabilities is looking at the future."""
    public = [n for n in dir(R) if not n.startswith("_")]
    assert not [n for n in public if "smooth" in n.lower()]
    assert "smoothed" not in R.RegimeFit.__dataclass_fields__


def test_expected_duration_is_one_over_the_exit_probability(fitted):
    fit, _, _ = fitted
    for state in (CALM, STRESS):
        p = fit.transition[state, state]
        assert fit.expected_duration(state) == pytest.approx(1.0 / (1.0 - p))
    assert fit.expected_duration(STRESS) > 1.0


def test_too_little_history_is_refused_rather_than_guessed():
    y, _ = _simulate(n=120)
    with pytest.raises(EstimationError, match="at least 250 sessions"):
        fit_regimes(y)


# ------------------------------------------------------------------ VIX ---
def _vix(n=800, seed=2):
    rng = np.random.default_rng(seed)
    x = 15 + np.cumsum(rng.normal(0, 0.4, n))
    return pd.Series(np.clip(x, 8, 80), index=pd.bdate_range("2021-01-01", periods=n))


def test_vix_bands_are_percentiles_of_history_up_to_as_of():
    s = _vix()
    r = vix_bands(s)
    assert r.calm_below < r.stress_above
    assert r.band in {"calm", "normal", "stress"}
    assert 0 <= r.percentile <= 100
    # a spike today reads as stress; the same series ending earlier does not know about it
    spiked = s.copy()
    spiked.iloc[-1] = float(s.max() * 2)
    assert vix_bands(spiked).band == "stress"
    as_of = s.index[400]
    assert vix_bands(spiked, as_of=as_of).level == pytest.approx(float(s.loc[as_of]))
    assert vix_bands(spiked, as_of=as_of).stress_above == vix_bands(s.loc[:as_of]).stress_above


def test_vix_bands_refuse_nonsense():
    with pytest.raises(EstimationError, match="at least 60 sessions"):
        vix_bands(_vix(n=30))
    with pytest.raises(EstimationError, match="percentiles"):
        vix_bands(_vix(), calm_pct=90, stress_pct=50)


def test_disagreement_between_gauges_is_reported_not_hidden(fitted):
    fit, _, _ = fitted
    calm_vix = R.VixReading(level=11.0, band="calm", calm_below=12.0, stress_above=22.0,
                            percentile=10.0, as_of=pd.Timestamp("2025-01-01"), lookback_sessions=500)
    stressed = R.RegimeFit(mu=fit.mu, scale=fit.scale, nu=fit.nu, transition=fit.transition,
                           filtered=pd.DataFrame({"calm": [0.1], "stress": [0.9]},
                                                 index=[pd.Timestamp("2025-01-01")]),
                           log_likelihood=0.0, iterations=1, converged=True,
                           provenance=fit.provenance)
    op = R.combine(stressed, calm_vix)
    assert op.hmm_says == "stress" and op.vix_says == "calm"
    assert not op.agree and "disagrees" in op.headline and "uncertain" in op.headline
    assert f"{0.9:.0%}" in op.headline
    # and when they agree, no hedging language
    agreeing = R.combine(stressed, R.VixReading(level=30.0, band="stress", calm_below=12.0,
                                                stress_above=22.0, percentile=95.0,
                                                as_of=pd.Timestamp("2025-01-01"), lookback_sessions=500))
    assert agreeing.agree and "disagrees" not in agreeing.headline


def test_mixed_state_says_so_instead_of_picking_one(fitted):
    fit, _, _ = fitted
    mixed = R.RegimeFit(mu=fit.mu, scale=fit.scale, nu=fit.nu, transition=fit.transition,
                        filtered=pd.DataFrame({"calm": [0.55], "stress": [0.45]},
                                              index=[pd.Timestamp("2025-01-01")]),
                        log_likelihood=0.0, iterations=1, converged=True, provenance=fit.provenance)
    op = R.combine(mixed)
    assert op.hmm_says == "mixed" and "Neither reading is settled" in op.headline


# ------------------------------------------------------------ real data ---
@pytest.fixture(scope="module")
def nifty():
    p = FixtureProvider.default()
    if not (p.root / "nse_close.parquet").exists():
        pytest.skip("fixture not built")
    s = p.benchmark("^NSEI").dropna()
    return np.log(s / s.shift(1)).dropna()


def test_real_nifty_regimes_are_plausible_and_find_known_turbulence(nifty):
    fit = fit_regimes(nifty)
    assert fit.annual_volatility[CALM] < fit.annual_volatility[STRESS]
    assert 0.03 < fit.annual_volatility[CALM] < 0.25
    assert fit.annual_volatility[STRESS] < 1.2
    assert fit.expected_duration(STRESS) > 1.5            # regimes persist; they are not day labels
    stress = fit.filtered["stress"]
    # the Ukraine invasion week was not a calm market
    window = stress.loc["2022-02-21":"2022-03-11"]
    if len(window):
        assert window.mean() > stress.mean()

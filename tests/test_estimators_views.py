"""Signals to views: measured IC, Grinold sizing, and confidence that follows the evidence."""
import numpy as np
import pandas as pd
import pytest

from star.core.errors import EstimationError
from star.core.types import CovarianceEstimate, Periodicity, Provenance
from star.estimators.black_litterman import black_litterman
from star.estimators.views import (ICEstimate, ViewSettings, absolute_view, grinold_alpha,
                                   quantile_baskets, quintile_view, rank_ic, standardise)
from tests.test_estimators_black_litterman import HL_DELTA, hl_prior

NAMES = [f"S{i:02d}" for i in range(20)]


def _cov(vols=None, corr=0.3, names=NAMES):
    n = len(names)
    sd = np.full(n, 0.25) if vols is None else np.asarray(vols, dtype=float)
    c = np.full((n, n), corr)
    np.fill_diagonal(c, 1.0)
    return CovarianceEstimate(matrix=pd.DataFrame(c * np.outer(sd, sd), index=names, columns=names),
                              periodicity=Periodicity.ANNUAL, provenance=Provenance(method="synthetic"))


def _prices(ic_target, horizon=21, days=1500, seed=5, names=NAMES):
    """
    Prices whose forward `horizon`-day return is correlated with a fixed score.

    The score is constant per name, so the designed rank IC is the rank
    correlation between the score and each period's realised forward return.
    """
    rng = np.random.default_rng(seed)
    score = pd.Series(np.linspace(-1, 1, len(names)), index=names)
    z = (score - score.mean()) / score.std(ddof=1)
    drift = ic_target * 0.25 / np.sqrt(252) * z.to_numpy()       # Grinold-sized daily edge
    noise = rng.normal(0, 0.25 / np.sqrt(252), (days, len(names)))
    log_px = np.cumsum(drift + noise, axis=0)
    idx = pd.bdate_range("2019-01-01", periods=days)
    close = pd.DataFrame(100 * np.exp(log_px), index=idx, columns=names)
    scores = pd.DataFrame(np.tile(score.to_numpy(), (days, 1)), index=idx, columns=names)
    return scores, close


def test_standardise_demeans_so_a_view_can_be_negative():
    z = standardise(pd.Series({"A": 1.0, "B": 2.0, "C": 3.0}))
    assert z.mean() == pytest.approx(0.0)
    assert z.std(ddof=1) == pytest.approx(1.0)
    assert z["A"] < 0 < z["C"]                      # the legacy could only be bullish
    # an outlier is winsorised, not allowed to dominate: with 50 names its z-score
    # would be 49/sqrt(50) = 6.9 before clipping
    wild = standardise(pd.Series([0.0] * 49 + [1e6]))
    assert wild.max() == pytest.approx(3.0)
    with pytest.raises(EstimationError, match="no cross-sectional dispersion"):
        standardise(pd.Series({"A": 1.0, "B": 1.0}))


def test_grinold_alpha_is_ic_times_vol_times_score():
    scores = pd.Series(np.linspace(-1, 1, 20), index=NAMES)
    cov = _cov(vols=np.linspace(0.15, 0.45, 20))
    a = grinold_alpha(scores, cov, 0.05)
    z = standardise(scores)
    assert np.allclose(a.to_numpy(), 0.05 * cov.volatilities().to_numpy() * z.to_numpy())
    # twice the skill, twice the alpha; no skill, no alpha
    assert np.allclose(grinold_alpha(scores, cov, 0.10).to_numpy(), 2 * a.to_numpy())
    assert np.allclose(grinold_alpha(scores, cov, 0.0).to_numpy(), 0.0)
    # same score, more volatile stock, bigger view
    high, low = NAMES[-1], NAMES[10]
    assert cov.volatilities()[high] > cov.volatilities()[low]
    assert abs(a[high]) > abs(a[low])


def test_measured_ic_recovers_a_designed_edge_and_reports_zero_for_noise():
    scores, close = _prices(ic_target=0.06)
    est = rank_ic(scores, close, 21)
    assert est.raw_mean > 0.02
    assert est.horizon_days == 21
    noise_scores, noise_close = _prices(ic_target=0.0, seed=99)
    noise = rank_ic(noise_scores, noise_close, 21)
    assert abs(noise.raw_mean) < abs(est.raw_mean)
    assert noise.confidence() < est.confidence()


def test_overlapping_windows_are_not_counted_as_independent_evidence():
    """A 21-day IC measured daily has ~n/21 independent observations, not n."""
    scores, close = _prices(ic_target=0.06)
    daily = rank_ic(scores, close, 1)
    monthly = rank_ic(scores, close, 21)
    assert monthly.observations < daily.observations / 10
    # the same dispersion over fewer independent observations is a wider error bar
    assert monthly.std_error > 0


def test_an_implausible_ic_is_clipped_and_says_so():
    scores, close = _prices(ic_target=0.60, seed=2)
    est = rank_ic(scores, close, 21)
    assert est.clipped and abs(est.mean) <= ViewSettings().ic_max
    assert "clipped" in est.note
    assert abs(est.raw_mean) > abs(est.mean)


def test_confidence_follows_the_evidence_not_an_opinion():
    s = ViewSettings()
    weak = ICEstimate(mean=0.05, std_error=0.05, observations=30, horizon_days=21,
                      raw_mean=0.05, clipped=False)
    assert weak.t_stat == pytest.approx(1.0)
    assert weak.confidence() == 0.0                 # not distinguishable from noise: do not bet
    strong = ICEstimate(mean=0.05, std_error=0.01, observations=30, horizon_days=21,
                        raw_mean=0.05, clipped=False)
    assert strong.confidence() == pytest.approx(s.max_confidence)
    middling = ICEstimate(mean=0.05, std_error=0.05 / 2.75, observations=30, horizon_days=21,
                          raw_mean=0.05, clipped=False)
    assert 0 < middling.confidence() < s.max_confidence
    # a single cross-sectional signal never gets more than half the way there
    assert ICEstimate(mean=0.1, std_error=1e-6, observations=30, horizon_days=21,
                      raw_mean=0.1, clipped=False).confidence() == pytest.approx(s.max_confidence)


def test_quantile_baskets_split_by_score():
    scores = pd.Series(np.linspace(-1, 1, 20), index=NAMES)
    top, bottom = quantile_baskets(scores, 5)
    assert len(top) == len(bottom) == 4
    assert scores[top].min() > scores[bottom].max()
    with pytest.raises(EstimationError, match="at least 10 names"):
        quantile_baskets(scores.iloc[:6], 5)


def test_quintile_view_is_relative_and_sized_by_skill():
    scores = pd.Series(np.linspace(-1, 1, 20), index=NAMES)
    cov = _cov()
    ic = ICEstimate(mean=0.05, std_error=0.015, observations=40, horizon_days=21,
                    raw_mean=0.05, clipped=False)
    v = quintile_view(scores, cov, ic)
    assert v.is_relative                                  # coefficients sum to zero
    assert sum(v.picks.values()) == pytest.approx(0.0, abs=1e-12)
    assert v.q > 0
    assert v.confidence == pytest.approx(ic.confidence())
    # half the IC, half the spread
    half = quintile_view(scores, cov, ICEstimate(mean=0.025, std_error=0.015, observations=40,
                                                 horizon_days=21, raw_mean=0.025, clipped=False))
    assert half.q == pytest.approx(v.q / 2, rel=1e-9)
    # a negative IC reverses which basket is bought - the spread stays positive
    # because the baskets themselves swap, which is the honest way to say
    # "this signal has worked backwards"
    flipped = quintile_view(scores, cov, ICEstimate(mean=-0.05, std_error=0.015, observations=40,
                                                    horizon_days=21, raw_mean=-0.05, clipped=False))
    assert flipped.q == pytest.approx(v.q, rel=1e-9)
    bought_normally = {t for t, c in v.picks.items() if c > 0}
    bought_flipped = {t for t, c in flipped.picks.items() if c > 0}
    assert bought_normally.isdisjoint(bought_flipped)
    assert bought_flipped == {t for t, c in v.picks.items() if c < 0}


def test_a_signal_that_cannot_be_distinguished_from_noise_leaves_the_portfolio_alone():
    """The whole point of measuring IC: no evidence, no trade."""
    from star.estimators.equilibrium import implied_weights
    prior, cov, w_ref = hl_prior()
    scores = pd.Series([0.9, 0.5, 0.2, -0.2, -0.4, -0.7, -0.3],
                       index=list(cov.matrix.columns))
    weak = ICEstimate(mean=0.04, std_error=0.04, observations=20, horizon_days=21,
                      raw_mean=0.04, clipped=False)
    v = quintile_view(scores, cov, weak, settings=ViewSettings(quantiles=3))
    assert v.confidence == 0.0
    res = black_litterman(prior, cov, [v], delta=HL_DELTA)
    assert np.abs(implied_weights(res.mu, cov, HL_DELTA) - w_ref).max() < 1e-6


def test_a_credible_signal_does_move_the_portfolio():
    from star.estimators.equilibrium import implied_weights
    prior, cov, w_ref = hl_prior()
    names = list(cov.matrix.columns)
    scores = pd.Series([0.9, 0.5, 0.2, -0.2, -0.4, -0.7, -0.3], index=names)
    strong = ICEstimate(mean=0.08, std_error=0.02, observations=40, horizon_days=21,
                        raw_mean=0.08, clipped=False)
    v = quintile_view(scores, cov, strong, settings=ViewSettings(quantiles=3))
    res = black_litterman(prior, cov, [v], delta=HL_DELTA)
    tilt = implied_weights(res.mu, cov, HL_DELTA) - w_ref
    assert tilt[max(v.picks, key=v.picks.get)] > 0
    assert tilt[min(v.picks, key=v.picks.get)] < 0
    assert np.abs(tilt).max() > 1e-3


def test_ic_refuses_impossible_inputs():
    scores, close = _prices(ic_target=0.05)
    with pytest.raises(EstimationError, match="at least 1 day"):
        rank_ic(scores, close, 0)
    with pytest.raises(EstimationError, match="shared names"):
        rank_ic(scores[NAMES[:2]], close[NAMES[:2]], 21)
    with pytest.raises(EstimationError, match="not enough overlapping"):
        rank_ic(scores.iloc[:1], close.iloc[:1], 21)


def test_absolute_views_are_only_ever_explicit():
    v = absolute_view("RELIANCE.NS", 0.09, 0.4)
    assert not v.is_relative and v.picks == {"RELIANCE.NS": 1.0} and v.q == 0.09

"""
Equilibrium prior and Black-Litterman posterior.

The anchor is He & Litterman (1999): their seven-country example is the standard
known-answer test for this model.
"""
import numpy as np
import pandas as pd
import pytest

from star.core.errors import ConventionError, EstimationError
from star.core.types import Basis, Compounding, CovarianceEstimate, Periodicity, Provenance
from star.estimators.black_litterman import (DEFAULT_TAU, View, black_litterman, build_view_matrices,
                                             idzorek_omega)
from star.estimators.equilibrium import (EquilibriumSettings, equilibrium_prior, implied_weights,
                                         market_volatility, reference_weights, risk_aversion)

# He & Litterman (1999), Table 1: correlations, annual volatilities and equilibrium
# (market-capitalisation) weights for seven equity markets.
HL_NAMES = ["AUS", "CAN", "FRA", "GER", "JAP", "UK", "USA"]
HL_CORR = np.array([
    [1.000, 0.488, 0.478, 0.515, 0.439, 0.512, 0.491],
    [0.488, 1.000, 0.664, 0.655, 0.310, 0.608, 0.779],
    [0.478, 0.664, 1.000, 0.861, 0.355, 0.783, 0.668],
    [0.515, 0.655, 0.861, 1.000, 0.354, 0.777, 0.653],
    [0.439, 0.310, 0.355, 0.354, 1.000, 0.405, 0.306],
    [0.512, 0.608, 0.783, 0.777, 0.405, 1.000, 0.652],
    [0.491, 0.779, 0.668, 0.653, 0.306, 0.652, 1.000],
])
HL_VOL = np.array([16.0, 20.3, 24.8, 27.1, 21.0, 20.0, 18.7]) / 100
HL_W = np.array([1.6, 2.2, 5.2, 5.5, 11.6, 12.4, 61.5]) / 100
HL_DELTA = 2.5
# Table 2, equilibrium risk premia, as published (one decimal place, in percent).
HL_PI_PUBLISHED = np.array([3.9, 6.9, 8.4, 9.0, 4.3, 6.8, 7.6])


def hl_cov():
    m = HL_CORR * np.outer(HL_VOL, HL_VOL)
    return CovarianceEstimate(matrix=pd.DataFrame(m, index=HL_NAMES, columns=HL_NAMES),
                              periodicity=Periodicity.ANNUAL,
                              provenance=Provenance(method="He-Litterman 1999 Table 1"),
                              vol_method="published", corr_method="published")


def hl_prior():
    """Prior at the paper's own delta, not ours, so the comparison is like for like."""
    w = pd.Series(HL_W, index=HL_NAMES)
    cov = hl_cov()
    pi = pd.Series(HL_DELTA * (cov.matrix.to_numpy() @ w.to_numpy()), index=HL_NAMES)
    from star.core.types import ReturnEstimate
    return ReturnEstimate(values=pi, basis=Basis.EXCESS, compounding=Compounding.SIMPLE,
                          periodicity=Periodicity.ANNUAL,
                          provenance=Provenance(method="equilibrium prior")), cov, w


def test_reproduces_he_litterman_equilibrium_premia():
    """Pi = delta * Sigma * w_eq, to the precision the paper prints."""
    prior, _, _ = hl_prior()
    assert np.allclose(np.round(prior.values.to_numpy() * 100, 1), HL_PI_PUBLISHED)


def test_delta_is_derived_from_a_target_sharpe_not_typed_in():
    assert risk_aversion(0.15) == pytest.approx(0.35 / 0.15)
    # the legacy's delta = 0.5 implied a ~1% risk premium; this implies ~5%
    delta = risk_aversion(0.15)
    assert delta * 0.15 ** 2 == pytest.approx(0.0525, abs=1e-4)
    # a different opinion about the market moves it, and the clamps hold
    assert risk_aversion(0.15, EquilibriumSettings(target_market_sharpe=0.5)) > delta
    assert risk_aversion(0.01) == EquilibriumSettings().max_delta
    for bad in (0.0, -0.2, np.nan):
        with pytest.raises(EstimationError):
            risk_aversion(bad)


def test_prior_is_the_exact_inverse_of_an_unconstrained_optimisation():
    """Reverse-optimisation identity: feeding Pi back in must return w_ref."""
    cov = hl_cov()
    w_ref = pd.Series(HL_W, index=HL_NAMES)
    prior, delta = equilibrium_prior(cov, w_ref)
    assert np.allclose(implied_weights(prior, cov, delta).to_numpy(), HL_W, atol=1e-12)
    assert prior.units == "annual simple excess"
    assert delta == pytest.approx(risk_aversion(market_volatility(cov, w_ref)))


def test_reference_weights_fall_back_and_record_every_fallback():
    caps = pd.Series({"A": 300.0, "B": 100.0})
    w, notes = reference_weights(["A", "B"], market_caps=caps)
    assert w.tolist() == [0.75, 0.25] and notes == ()

    w, notes = reference_weights(["A", "B"], shares_outstanding=pd.Series({"A": 10.0, "B": 30.0}),
                                 price=pd.Series({"A": 30.0, "B": 10.0}))
    assert w.tolist() == [0.5, 0.5] and "shares outstanding" in notes[0]

    w, notes = reference_weights(["A", "B"], market_caps=pd.Series({"A": 300.0, "B": np.nan}))
    assert w.tolist() == [0.5, 0.5] and "no usable market cap for 1 of 2" in notes[0]

    # in a backtest, today's market cap is look-ahead
    w, notes = reference_weights(["A", "B"], market_caps=caps, allow_market_cap=False)
    assert w.tolist() == [0.5, 0.5] and "point-in-time unavailable" in notes[0]
    with pytest.raises(EstimationError):
        reference_weights([])


def test_degradations_travel_into_the_prior_provenance():
    w, notes = reference_weights(["AUS", "CAN"], market_caps=pd.Series({"AUS": 1.0, "CAN": np.nan}))
    cov = CovarianceEstimate(matrix=hl_cov().matrix.loc[["AUS", "CAN"], ["AUS", "CAN"]],
                             periodicity=Periodicity.ANNUAL, provenance=Provenance(method="t"))
    prior, _ = equilibrium_prior(cov, w, degradations=notes)
    assert prior.provenance.degradations == notes


def test_no_views_leaves_the_prior_untouched():
    prior, cov, _ = hl_prior()
    res = black_litterman(prior, cov, [], delta=HL_DELTA)
    assert np.allclose(res.mu.values.to_numpy(), prior.values.to_numpy(), atol=1e-12)
    # with no views the posterior covariance is (1 + tau) Sigma: parameter risk, kept
    assert np.allclose(res.covariance.matrix.to_numpy(),
                       (1 + DEFAULT_TAU) * cov.matrix.to_numpy(), atol=1e-12)
    assert res.covariance.is_psd()


def test_posterior_moves_only_in_the_direction_of_the_view_portfolio():
    """
    He & Litterman's central intuition, and an exact property: with one view the
    weights depart from equilibrium only along that view's portfolio.
    """
    prior, cov, w_ref = hl_prior()
    # Germany beats the rest of Europe by 5% a year (their example 1)
    view = View("GER vs EUR", {"GER": 1.0, "FRA": -0.295, "UK": -0.705}, 0.05, confidence=0.6)
    res = black_litterman(prior, cov, [view], delta=HL_DELTA)
    tilt = implied_weights(res.mu, cov, HL_DELTA) - w_ref
    p = res.P.loc["GER vs EUR"].to_numpy()
    scale = tilt.to_numpy() @ p / (p @ p)
    assert np.allclose(tilt.to_numpy(), scale * p, atol=1e-8)
    assert scale > 0                                     # a bullish view on Germany buys Germany
    assert tilt["GER"] > 0 > tilt["UK"]
    # note the asymmetry: the *weights* move purely along the view, but the
    # expected *returns* of correlated names move with it too (UK correlates 0.78
    # with Germany), which is the model working, not a leak.
    assert res.view_shift["GER"] > 0 and res.view_shift["UK"] > 0


def test_stated_confidence_moves_the_posterior_by_exactly_that_much():
    """Idzorek's defining property - and the cure for legacy defect M2B-11."""
    prior, cov, w_ref = hl_prior()
    picks = {"GER": 1.0, "FRA": -0.295, "UK": -0.705}
    full = implied_weights(black_litterman(prior, cov, [View("v", picks, 0.05, confidence=1.0)].copy(),
                                           delta=HL_DELTA).mu, cov, HL_DELTA) - w_ref
    for c in (0.25, 0.5, 0.75):
        tilt = implied_weights(black_litterman(prior, cov, [View("v", picks, 0.05, confidence=c)],
                                               delta=HL_DELTA).mu, cov, HL_DELTA) - w_ref
        # only where the certain tilt is materially non-zero: elsewhere the ratio is noise
        big = np.abs(full.to_numpy()) > 0.01 * np.abs(full.to_numpy()).max()
        moved = tilt.to_numpy()[big] / full.to_numpy()[big]
        assert np.allclose(moved, c, atol=1e-3), c
    zero = implied_weights(black_litterman(prior, cov, [View("v", picks, 0.05, confidence=0.0)],
                                           delta=HL_DELTA).mu, cov, HL_DELTA) - w_ref
    assert np.abs(zero).max() < 1e-6


def test_tau_cannot_change_the_answer_and_that_is_the_point():
    """
    With Omega calibrated from confidence (or the classical Omega = P tau Sigma P'),
    tau cancels out of the posterior exactly. The legacy exposed BL_TAU as a live
    control while this identity made it inert (M2B-11); here confidence is the
    control and tau is documented as a scale that cannot be identified.
    """
    prior, cov, _ = hl_prior()
    views = [View("v", {"GER": 1.0, "UK": -1.0}, 0.04, confidence=0.5)]
    a = black_litterman(prior, cov, views, delta=HL_DELTA, tau=0.01).mu.values
    b = black_litterman(prior, cov, views, delta=HL_DELTA, tau=0.25).mu.values
    assert np.allclose(a.to_numpy(), b.to_numpy(), atol=1e-9)
    classical = [View("v", {"GER": 1.0, "UK": -1.0}, 0.04)]
    c1 = black_litterman(prior, cov, classical, delta=HL_DELTA, tau=0.02).mu.values
    c2 = black_litterman(prior, cov, classical, delta=HL_DELTA, tau=0.40).mu.values
    assert np.allclose(c1.to_numpy(), c2.to_numpy(), atol=1e-9)
    # and it is general, not an artefact of having a single view
    two = [View("a", {"GER": 1.0, "UK": -1.0}, 0.04, confidence=0.5),
           View("b", {"CAN": 1.0, "USA": -1.0}, 0.02)]
    d1 = black_litterman(prior, cov, two, delta=HL_DELTA, tau=0.01).mu.values
    d2 = black_litterman(prior, cov, two, delta=HL_DELTA, tau=0.30).mu.values
    assert np.allclose(d1.to_numpy(), d2.to_numpy(), atol=1e-9)


def test_view_size_and_direction_both_reach_the_posterior():
    prior, cov, _ = hl_prior()
    def shift(q):
        v = View("v", {"GER": 1.0, "UK": -1.0}, q, confidence=0.6)
        return black_litterman(prior, cov, [v], delta=HL_DELTA).view_shift
    implied_gap = float(prior.values["GER"] - prior.values["UK"])
    assert shift(implied_gap + 0.05)["GER"] > 0                 # more bullish than equilibrium
    assert shift(implied_gap - 0.05)["GER"] < 0                 # a bearish view must be expressible
    assert shift(implied_gap + 0.10)["GER"] > shift(implied_gap + 0.05)["GER"]


def test_posterior_covariance_adds_parameter_risk_rather_than_discarding_it():
    prior, cov, _ = hl_prior()
    res = black_litterman(prior, cov, [View("v", {"GER": 1.0, "UK": -1.0}, 0.04, confidence=0.5)],
                          delta=HL_DELTA)
    extra = res.covariance.matrix.to_numpy() - cov.matrix.to_numpy()
    assert np.linalg.eigvalsh(extra).min() > -1e-12             # M is positive semi-definite
    assert res.covariance.is_psd()
    assert np.all(np.diag(extra) > 0)                           # every variance grows
    assert "bl_parameter_risk" in res.covariance.corr_method


def test_posterior_is_excess_and_cannot_be_converted_twice():
    prior, cov, _ = hl_prior()
    res = black_litterman(prior, cov, [View("v", {"GER": 1.0, "UK": -1.0}, 0.04, confidence=0.5)],
                          delta=HL_DELTA)
    assert res.mu.basis is Basis.EXCESS
    total = res.mu.to_total(0.06)
    assert total.basis is Basis.TOTAL
    with pytest.raises(ConventionError, match="M2-01"):
        total.to_total(0.06)


def test_views_are_validated_before_they_can_distort_anything():
    prior, cov, _ = hl_prior()
    with pytest.raises(EstimationError, match="outside the universe"):
        black_litterman(prior, cov, [View("v", {"NIFTY": 1.0}, 0.05)], delta=HL_DELTA)
    with pytest.raises(EstimationError, match="share the name"):
        build_view_matrices([View("v", {"GER": 1.0}, 0.05), View("v", {"UK": 1.0}, 0.05)], HL_NAMES)
    with pytest.raises(EstimationError, match="confidence"):
        View("v", {"GER": 1.0}, 0.05, confidence=1.4)
    with pytest.raises(EstimationError, match="picks no assets"):
        View("v", {}, 0.05)
    with pytest.raises(EstimationError, match="non-finite"):
        View("v", {"GER": 1.0}, np.nan)
    with pytest.raises(EstimationError, match="prior is not EXCESS"):
        black_litterman(prior.to_total(0.05), cov, [], delta=HL_DELTA)
    with pytest.raises(EstimationError, match="tau"):
        black_litterman(prior, cov, [], delta=HL_DELTA, tau=0.0)


def test_relative_and_absolute_views_are_distinguished():
    rel = View("a", {"GER": 1.0, "UK": -1.0}, 0.03)
    absolute = View("b", {"GER": 1.0}, 0.09)
    assert rel.is_relative and not absolute.is_relative
    P, q = build_view_matrices([rel, absolute], HL_NAMES)
    assert P.shape == (2, 7) and list(q) == [0.03, 0.09]
    assert P.loc["a", "GER"] == 1.0 and P.loc["a", "UK"] == -1.0
    assert P.loc["a"].sum() == pytest.approx(0.0)


def test_two_views_are_each_honoured_at_their_own_confidence():
    prior, cov, w_ref = hl_prior()
    views = [View("GER>UK", {"GER": 1.0, "UK": -1.0}, 0.05, confidence=0.8),
             View("CAN>USA", {"CAN": 1.0, "USA": -1.0}, 0.02, confidence=0.2)]
    res = black_litterman(prior, cov, views, delta=HL_DELTA)
    assert res.omega.shape == (2, 2)
    assert res.omega.iloc[0, 0] < res.omega.iloc[1, 1]       # more confidence, less uncertainty
    tilt = implied_weights(res.mu, cov, HL_DELTA) - w_ref
    assert tilt["GER"] > 0 > tilt["UK"]
    assert tilt["CAN"] > 0 > tilt["USA"]
    assert abs(tilt["GER"]) > abs(tilt["CAN"])


def test_omega_is_finite_positive_and_ordered_in_confidence():
    prior, cov, _ = hl_prior()
    sigma, pi = cov.matrix.to_numpy(), prior.values.to_numpy()
    P = np.zeros((1, 7)); P[0, 3] = 1.0; P[0, 5] = -1.0
    q = np.array([0.05])
    omegas = [idzorek_omega(sigma, pi, P, q, [c], DEFAULT_TAU, HL_DELTA)[0, 0]
              for c in (0.1, 0.3, 0.6, 0.9)]
    assert all(np.isfinite(o) and o > 0 for o in omegas)
    assert omegas == sorted(omegas, reverse=True)

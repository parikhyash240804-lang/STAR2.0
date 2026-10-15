"""Covariance estimator: Yang-Zhang volatility x Ledoit-Wolf correlation, repaired to be valid."""
import numpy as np
import pandas as pd
import pytest
from sklearn.covariance import LedoitWolf

from star.core.errors import EstimationError
from star.core.types import Periodicity
from star.data.fixture import FixtureProvider
from star.data.quality import assess
from star.data.view import DataView
from star.estimators.covariance import (CovarianceSettings, estimate_covariance, nearest_correlation,
                                        shrunk_correlation, yang_zhang_variance)


def _simulate_ohlc(n_days, daily_sd, rng, steps=78, overnight_share=0.2):
    """Intraday random walk: returns O, H, L, C with known daily log-return variance."""
    on_sd = daily_sd * np.sqrt(overnight_share)
    step_sd = daily_sd * np.sqrt((1 - overnight_share) / steps)
    o, h, l, c = [], [], [], []
    prev = 100.0
    for _ in range(n_days):
        op = prev * np.exp(rng.normal(0, on_sd))
        path = op * np.exp(np.cumsum(rng.normal(0, step_sd, steps)))
        o.append(op); h.append(max(op, path.max())); l.append(min(op, path.min())); c.append(path[-1])
        prev = path[-1]
    idx = pd.bdate_range("2024-01-01", periods=n_days)
    return [pd.Series(v, index=idx) for v in (o, h, l, c)]


def test_yang_zhang_is_unbiased_as_intraday_sampling_becomes_continuous():
    """
    Highs and lows sampled at a few points understate the true range, so a coarse
    simulation biases any range estimator low (measured: 0.86 at 78 steps, 0.94 at
    390, 0.98 at 2,000). Real NSE highs/lows come from continuous trading, i.e.
    the fine end - so unbiasedness is checked there, and convergence explicitly.
    """
    sd = 0.02
    def mean_ratio(steps, sims):
        rng = np.random.default_rng(11)
        return np.mean([yang_zhang_variance(*_simulate_ohlc(60, sd, rng, steps=steps), halflife=None)
                        for _ in range(sims)]) / sd ** 2
    coarse, fine = mean_ratio(78, 40), mean_ratio(2000, 40)
    assert fine == pytest.approx(1.0, abs=0.05)
    assert abs(1 - fine) < abs(1 - coarse)


def test_yang_zhang_is_more_efficient_than_close_to_close():
    rng = np.random.default_rng(11)
    sd = 0.02
    yz, cc = [], []
    for _ in range(150):
        o, h, l, c = _simulate_ohlc(60, sd, rng)
        yz.append(yang_zhang_variance(o, h, l, c, halflife=None) / sd ** 2)
        cc.append(np.log(c / c.shift(1)).dropna().var() / sd ** 2)
    # compare relative dispersion so the coarse-sampling level bias does not matter
    yz, cc = np.array(yz), np.array(cc)
    assert (cc.std() / cc.mean()) / (yz.std() / yz.mean()) > 1.4


def test_nearest_correlation_repairs_invalid_and_leaves_valid_alone():
    bad = np.array([[1.0, 0.9, 0.7], [0.9, 1.0, -0.4], [0.7, -0.4, 1.0]])   # not PSD
    assert np.linalg.eigvalsh(bad).min() < 0
    fixed = nearest_correlation(bad)
    assert np.linalg.eigvalsh(fixed).min() > -1e-10
    assert np.allclose(np.diag(fixed), 1.0)
    good = np.array([[1.0, 0.3], [0.3, 1.0]])
    assert np.allclose(nearest_correlation(good), good)


def test_shrunk_correlation_matches_sklearn_on_standardised_returns():
    rng = np.random.default_rng(3)
    idx = pd.bdate_range("2020-01-01", periods=400)
    r = pd.DataFrame(rng.multivariate_normal([0, 0, 0], [[1, .5, .2], [.5, 1, .3], [.2, .3, 1]], 400) * 0.01,
                     index=idx, columns=list("ABC"))
    corr, shrink = shrunk_correlation(r, standardise_halflife=60)
    vol = np.sqrt(r.pow(2).ewm(halflife=60).mean()).shift(1)
    vol = vol.clip(lower=vol.median() * 0.25, axis=1)
    z = (r / vol).dropna().clip(-8.0, 8.0)
    ref = LedoitWolf().fit(z.to_numpy())
    d = np.sqrt(np.diag(ref.covariance_))
    assert np.allclose(corr.to_numpy(), ref.covariance_ / np.outer(d, d), atol=1e-10)
    assert shrink == pytest.approx(ref.shrinkage_)


@pytest.fixture(scope="module")
def view():
    p = FixtureProvider.default()
    if not (p.root / "nse_close.parquet").exists():
        pytest.skip("fixture not built")
    panel = p.prices(p.universe)
    rep = assess(panel.close, min_adv_inr=0.0)
    return DataView.build(panel, panel.sessions.max()), rep.kept


def test_real_nse_covariance_is_valid_and_plausible(view):
    v, kept = view
    cov = estimate_covariance(v.ohlc(), tickers=kept, as_of=v.as_of)
    assert cov.periodicity is Periodicity.DAILY
    assert cov.is_psd()
    ann_vol = cov.annualise().volatilities()
    assert ann_vol.between(0.08, 0.90).all()                          # Indian large/mid caps
    corr = cov.correlation()
    assert np.allclose(np.diag(corr), 1.0) and corr.to_numpy().max() <= 1.0 + 1e-9
    assert "Ledoit-Wolf" in cov.provenance.method


def test_covariance_uses_nothing_after_as_of(view):
    v, kept = view
    as_of = pd.Timestamp("2025-03-28")
    early = DataView.build(v.panel, as_of)
    a = estimate_covariance(early.ohlc(), tickers=kept[:10], as_of=as_of)
    # rebuilding from a panel that never contained later data gives the same answer
    truncated = {k: f.loc[:as_of] for k, f in v.ohlc().items()}
    b = estimate_covariance(truncated, tickers=kept[:10], as_of=as_of)
    assert np.allclose(a.matrix.to_numpy(), b.matrix.to_numpy())


def test_method_does_not_switch_silently_with_the_data(view):
    """
    The legacy swapped estimator when many R^2 were low, and said nothing. Here the
    method is the same whatever the universe, and the one permitted fallback -
    too little history for the HAR regression - is always named in the provenance.
    """
    v, kept = view
    for names in (kept[:5], kept[:20]):
        cov = estimate_covariance(v.ohlc(), tickers=names, settings=CovarianceSettings())
        assert (cov.vol_method, cov.corr_method) == ("har", "ledoit_wolf")
        assert not cov.provenance.degradations
    short = {k: f.iloc[-260:] for k, f in v.ohlc().items()}
    fell_back = estimate_covariance(short, tickers=kept[:5])
    assert fell_back.vol_method == "yang_zhang_ewma"
    assert any("HAR volatility unavailable" in d for d in fell_back.provenance.degradations)


def test_the_correlation_level_tracks_the_sample_correlation(view):
    """
    The test that was missing, and it cost a 2x error.

    Valid, positive semi-definite and unit-diagonal were all checked; the *level*
    was not. A shrinkage intensity of 0.90 produced a perfectly well-formed matrix
    whose average correlation was 0.01 against a sample value of 0.21.
    """
    v, kept = view
    close = v.ohlc()["close"][kept]
    r = np.log(close / close.shift(1)).iloc[1:].iloc[-1260:]
    sample = r.corr().to_numpy()
    iu = np.triu_indices_from(sample, k=1)
    cov = estimate_covariance(v.ohlc(), tickers=kept, as_of=v.as_of)
    estimated = cov.correlation().to_numpy()
    assert abs(float(np.nanmean(estimated[iu])) - float(np.nanmean(sample[iu]))) < 0.05
    # shrinkage should be modest on 1,260 sessions of 37 names, not near-total
    shrink = float(cov.provenance.window.split("shrinkage ")[1])
    assert shrink < 0.25, shrink


def test_predicted_portfolio_volatility_matches_what_actually_happened(view):
    """
    Understated correlation shows up here first: diversification that is not real.
    This is the guard that would have caught the 2.25x error, and it is deliberately
    a band rather than a point - measured out-of-sample the model understates
    portfolio volatility by about 9% (docs/VALIDATION.md), which is a documented
    level bias, not licence for it to drift further.
    """
    v, kept = view
    close = v.ohlc()["close"][kept]
    r = np.log(close / close.shift(1)).iloc[1:].iloc[-1260:]
    w = pd.Series(1.0 / len(kept), index=kept)
    cov = estimate_covariance(v.ohlc(), tickers=kept, as_of=v.as_of)
    predicted = float(np.sqrt(w @ cov.annualise().matrix.loc[kept, kept] @ w))
    realised = float((r @ w).std(ddof=1) * np.sqrt(252))
    assert 0.75 < predicted / realised < 1.35, (predicted, realised)


def test_a_single_name_portfolio_needs_no_correlation_at_all(view):
    """The volatility path alone, isolated from the correlation estimate."""
    v, kept = view
    cov = estimate_covariance(v.ohlc(), tickers=kept, as_of=v.as_of)
    close = v.ohlc()["close"][kept]
    r = np.log(close / close.shift(1)).iloc[1:].iloc[-252:]
    realised = r.std(ddof=1) * np.sqrt(252)
    predicted = cov.annualise().volatilities()
    ratio = (realised / predicted).median()
    assert 0.7 < ratio < 1.4, ratio


def test_one_wild_observation_cannot_collapse_the_correlation_matrix():
    """
    Where it really comes from: the EWMA's burn-in. At the start of a window the
    volatility estimate has almost no data behind it, so if a stock's first
    observed move is near zero, the next ordinary day divides by almost nothing.
    On the real NSE universe this produced a standardised return of 137, which
    drove Ledoit-Wolf shrinkage to 0.90 and average correlation to 0.01.
    """
    rng = np.random.default_rng(17)
    idx = pd.bdate_range("2020-01-01", periods=800)
    common = rng.normal(0, 0.01, 800)
    r = pd.DataFrame({f"S{i}": 0.7 * common + 0.7 * rng.normal(0, 0.01, 800) for i in range(6)},
                     index=idx)
    clean_corr, clean_shrink = shrunk_correlation(r)
    iu = np.triu_indices(6, k=1)
    assert clean_shrink < 0.2

    contaminated = r.copy()
    contaminated.iloc[0, 0] = 1e-9          # a flat first session for one name
    contaminated.iloc[1, 0] = 0.02          # then an entirely ordinary move

    unguarded, unguarded_shrink = shrunk_correlation(contaminated, vol_floor_fraction=0.0,
                                                     max_standardised=1e9)
    assert unguarded_shrink > 0.5                                   # the pathology, reproduced
    assert float(unguarded.to_numpy()[iu].mean()) < 0.5 * float(clean_corr.to_numpy()[iu].mean())

    guarded, guarded_shrink = shrunk_correlation(contaminated)
    assert guarded_shrink < 0.2
    assert abs(float(guarded.to_numpy()[iu].mean())
               - float(clean_corr.to_numpy()[iu].mean())) < 0.05    # the estimate survives

    # either guard alone is enough
    floored, _ = shrunk_correlation(contaminated, max_standardised=1e9)
    clipped, _ = shrunk_correlation(contaminated, vol_floor_fraction=0.0)
    for c in (floored, clipped):
        assert abs(float(c.to_numpy()[iu].mean())
                   - float(clean_corr.to_numpy()[iu].mean())) < 0.05


def test_the_guards_validate_their_own_settings():
    rng = np.random.default_rng(3)
    r = pd.DataFrame(rng.normal(0, 0.01, (200, 3)), index=pd.bdate_range("2021-01-01", periods=200),
                     columns=list("ABC"))
    with pytest.raises(EstimationError, match="vol_floor_fraction"):
        shrunk_correlation(r, vol_floor_fraction=1.5)
    with pytest.raises(EstimationError, match="max_standardised"):
        shrunk_correlation(r, max_standardised=2.0)

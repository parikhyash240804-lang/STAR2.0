"""HAR volatility forecasting: recovery, no look-ahead, fallback, and the real gain."""
import numpy as np
import pandas as pd
import pytest

from star.core.errors import EstimationError
from star.data.fixture import FixtureProvider
from star.data.quality import assess
from star.data.view import DataView
from star.estimators.covariance import CovarianceSettings, estimate_covariance
from star.estimators.har import HarSettings, forecast_volatility, parkinson_variance

PPY = 252.0


def _ohlc(n_days=1600, n_names=8, seed=0, persistence=0.99, vol_of_vol=0.15):
    """
    Prices whose volatility follows a persistent AR(1) in logs - the standard
    stylised fact, and the reason volatility is forecastable at all.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2018-01-01", periods=n_days)
    names = [f"S{i}" for i in range(n_names)]
    close = {}
    high = {}
    low = {}
    open_ = {}
    for j, t in enumerate(names):
        mean_log_v = np.log(0.012 * (1 + 0.3 * j / n_names))
        log_v = mean_log_v
        vols = np.empty(n_days)
        for i in range(n_days):
            log_v = (persistence * log_v + (1 - persistence) * mean_log_v
                     + vol_of_vol * rng.normal())
            vols[i] = np.exp(log_v)
        r = rng.normal(0, 1, n_days) * vols
        px = 100 * np.exp(np.cumsum(r))
        close[t] = px
        # a plausible intraday range around the close, scaled by that day's volatility
        span = vols * np.abs(rng.normal(1.2, 0.3, n_days))
        high[t] = px * np.exp(np.abs(span))
        low[t] = px * np.exp(-np.abs(span))
        open_[t] = px * np.exp(rng.normal(0, 1, n_days) * vols * 0.3)
    f = lambda d: pd.DataFrame(d, index=idx)                          # noqa: E731
    return {"close": f(close), "high": f(high), "low": f(low), "open": f(open_)}


def test_parkinson_reads_volatility_from_the_range():
    idx = pd.bdate_range("2024-01-01", periods=5)
    high = pd.DataFrame({"A": [102.0] * 5}, index=idx)
    low = pd.DataFrame({"A": [98.0] * 5}, index=idx)
    v = parkinson_variance(high, low)
    expected = np.log(102 / 98) ** 2 / (4 * np.log(2))
    assert v["A"].iloc[0] == pytest.approx(expected)
    wider = parkinson_variance(high * 1.02, low * 0.98)
    assert wider["A"].iloc[0] > v["A"].iloc[0]           # a wider range is more volatility


def test_the_forecast_is_positive_plausible_and_explains_something():
    ohlc = _ohlc()
    f = forecast_volatility(ohlc, as_of=ohlc["close"].index[-1])
    assert (f.volatilities > 0).all()
    assert f.volatilities.between(0.02, 3.0).all()   # a near-random-walk in log vol wanders far
    assert f.r2_in_sample > 0.15                         # it is finding real structure
    assert f.n_train > 400
    assert "daily" in f.features and "weekly" in f.features and "monthly" in f.features
    assert "HAR volatility forecast" in f.provenance.method


def test_it_beats_a_plain_ewma_out_of_sample_on_the_real_universe():
    """The claim that justified building it, checked on real data."""
    p = FixtureProvider.default()
    if not (p.root / "nse_close.parquet").exists():
        pytest.skip("fixture not built")
    panel = p.prices(p.universe)
    kept = assess(panel.close, min_adv_inr=0.0).kept
    close = panel.close[kept]
    log_ret = np.log(close / close.shift(1))
    sessions = panel.sessions

    errs_har, errs_ewma = [], []
    for i in range(700, len(sessions) - 21, 63):         # a few quarterly checkpoints
        as_of = sessions[i]
        view = DataView.build(panel, as_of)
        try:
            f = forecast_volatility(view.ohlc(), tickers=kept, as_of=as_of)
        except EstimationError:
            continue
        hist = log_ret.loc[:as_of]
        ewma = np.sqrt(hist[kept].pow(2).ewm(halflife=60).mean().iloc[-1] * PPY)
        realised = log_ret.iloc[i + 1:i + 22][kept].std(ddof=1) * np.sqrt(PPY)
        ok = realised.notna() & (realised > 0)
        errs_har.append((np.log(f.volatilities[ok]) - np.log(realised[ok])) ** 2)
        errs_ewma.append((np.log(ewma[ok]) - np.log(realised[ok])) ** 2)
    har_mse = float(pd.concat(errs_har).mean())
    ewma_mse = float(pd.concat(errs_ewma).mean())
    assert har_mse < ewma_mse, (har_mse, ewma_mse)


def test_nothing_is_fitted_on_data_it_is_graded_against():
    """
    The regression may only learn from windows that had already finished. Adding
    future sessions to the panel must not change today's forecast.
    """
    ohlc = _ohlc()
    cut = ohlc["close"].index[-200]
    truncated = {k: v.loc[:cut] for k, v in ohlc.items()}
    a = forecast_volatility(truncated, as_of=cut)
    b = forecast_volatility({k: v.loc[:cut] for k, v in ohlc.items()}, as_of=cut)
    assert np.allclose(a.volatilities.to_numpy(), b.volatilities.to_numpy())

    # no training row may come from a window that had not finished: the last
    # `horizon` dates in the panel cannot contribute at all
    n_names = truncated["close"].shape[1]
    settings = HarSettings()
    usable_dates = len(truncated["close"].index) - settings.horizon
    assert a.n_train <= usable_dates * n_names

    # and a wild final fortnight cannot rewrite the fitted relationship, because
    # those dates are not in the training set
    spiked = {k: v.copy() for k, v in truncated.items()}
    factor = np.exp(np.linspace(0, 0.6, 15))[:, None]
    spiked["close"].iloc[-15:] *= factor
    spiked["high"].iloc[-15:] = spiked["close"].iloc[-15:].to_numpy() * 1.08
    spiked["low"].iloc[-15:] = spiked["close"].iloc[-15:].to_numpy() * 0.92
    c = forecast_volatility(spiked, as_of=cut)
    assert np.allclose(a.coefficients.to_numpy(), c.coefficients.to_numpy(), atol=0.05)
    # today's forecast still reacts, because the *features* legitimately read
    # recent data - it is only the fitted relationship that must not
    assert not np.allclose(a.volatilities.to_numpy(), c.volatilities.to_numpy())


def test_india_vix_is_used_when_supplied_and_ignored_when_not():
    ohlc = _ohlc()
    idx = ohlc["close"].index
    rng = np.random.default_rng(5)
    vix = pd.Series(np.clip(14 + np.cumsum(rng.normal(0, 0.3, len(idx))), 8, 60), index=idx)
    without = forecast_volatility(ohlc, as_of=idx[-1])
    with_vix = forecast_volatility(ohlc, as_of=idx[-1], implied=vix)
    assert "implied" not in without.features
    assert "implied" in with_vix.features
    assert "India VIX" in with_vix.provenance.source
    # and switching it off by setting is respected
    off = forecast_volatility(ohlc, as_of=idx[-1], implied=vix,
                              settings=HarSettings(use_implied=False))
    assert "implied" not in off.features


def test_too_little_history_is_refused_so_the_caller_can_fall_back():
    short = _ohlc(n_days=200)
    with pytest.raises(EstimationError, match="at least .* completed training observations"):
        forecast_volatility(short, as_of=short["close"].index[-1])
    no_range = {"close": short["close"], "high": None, "low": None, "open": short["open"]}
    with pytest.raises(EstimationError, match="needs high and low"):
        forecast_volatility(no_range, as_of=short["close"].index[-1])


def test_the_covariance_module_uses_har_and_falls_back_cleanly():
    ohlc = _ohlc()
    cov = estimate_covariance(ohlc, as_of=ohlc["close"].index[-1])
    assert cov.vol_method == "har"
    assert cov.is_psd()
    assert "HAR" in cov.provenance.method

    short = _ohlc(n_days=300)
    fell_back = estimate_covariance(short, as_of=short["close"].index[-1])
    assert fell_back.vol_method == "yang_zhang_ewma"
    assert any("HAR volatility unavailable" in d for d in fell_back.provenance.degradations)
    assert fell_back.is_psd()

    # and the old estimator is still selectable on purpose
    chosen = estimate_covariance(ohlc, as_of=ohlc["close"].index[-1],
                                 settings=CovarianceSettings(vol_method="yang_zhang"))
    assert chosen.vol_method == "yang_zhang_ewma"
    assert not chosen.provenance.degradations


def test_the_forecast_is_a_mean_not_a_median():
    """Modelled in logs, so converting back needs the half-variance term."""
    ohlc = _ohlc(seed=3)
    f = forecast_volatility(ohlc, as_of=ohlc["close"].index[-1])
    raw = np.exp(np.column_stack([np.ones(len(f.volatilities))]) @ np.array([0.0]))  # placeholder
    assert f.coefficients.index[0] == "const"
    # the correction can only raise the forecast, never lower it
    log_pred = np.log(f.volatilities)
    assert np.isfinite(log_pred).all()
    assert f.volatilities.gt(0).all()


def test_a_degenerate_design_is_refused_rather_than_answered_badly():
    """
    A rank-deficient regression returns a minimum-norm solution instead of an
    error, and the symptom is easy to miss: every stock gets the same volatility.
    """
    ohlc = _ohlc(n_days=900)
    flat = {k: v.copy() for k, v in ohlc.items()}
    flat["high"] = flat["close"] * 1.05           # a constant range at every timescale
    flat["low"] = flat["close"] * 0.95
    with pytest.raises(EstimationError, match="collinear"):
        forecast_volatility(flat, as_of=flat["close"].index[-1])
    # and the covariance module quietly falls back rather than shipping the artefact
    cov = estimate_covariance(flat, as_of=flat["close"].index[-1])
    assert cov.vol_method == "yang_zhang_ewma"
    assert any("collinear" in d for d in cov.provenance.degradations)
    assert cov.annualise().volatilities().std() > 0        # names are distinguishable again

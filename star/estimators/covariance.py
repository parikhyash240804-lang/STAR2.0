"""
Covariance = volatility x correlation, each estimated the way it is best estimated.

Legacy defects addressed: max-Sharpe on noisy inputs (M3-01), a "3-factor" risk
model whose momentum factor was identically zero (M2-02), close-to-close
volatility only, and a covariance method that switched with the data (making
runs irreproducible). See PAPERS/notes/03_covariance_volatility.md.

* **Volatility** is *forecast*, not just measured, by the HAR model in `har.py` -
  the day's high-low range at three timescales, plus the market level and India
  VIX. Measured out-of-sample on the NSE universe it is **24% more accurate** than
  the Yang-Zhang EWMA it replaced (R-squared 0.385 against 0.311). Yang-Zhang
  remains available and is used automatically when there is too little history to
  fit the regression, with the fallback recorded in the provenance.
* **Correlation** changes slowly and is noisy in n/T terms, so it is estimated over
  a longer window (default 1,260 sessions) on returns standardised by their own
  lagged volatility, then shrunk with Ledoit-Wolf (2004).
* **Repair**: Higham's (2002) nearest-correlation projection guarantees a valid
  (positive semi-definite, unit-diagonal) matrix; it is a no-op on valid input.

Covariance = D C D in daily log-return space, returned as a `CovarianceEstimate`.
The method never changes with the data.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from ..core.errors import EstimationError
from ..core.types import CovarianceEstimate, Periodicity, Provenance
from .har import HarSettings, forecast_volatility


@dataclass(frozen=True, slots=True)
class CovarianceSettings:
    vol_halflife: float = 60.0         # sessions; volatility reacts to current conditions
    vol_lookback: int = 504            # sessions of OHLC used for volatility
    corr_lookback: int = 1260          # sessions used for correlation
    standardise_halflife: float = 60.0 # EWMA used only to standardise returns for correlation
    min_sessions: int = 120
    vol_floor_fraction: float = 0.25   # a standardising volatility may not fall below this
    max_standardised: float = 8.0      # of its own median; standardised returns clipped here
    vol_method: str = "har"            # "har" (measured best) or "yang_zhang"
    horizon_days: int = 21             # the horizon the volatility forecast is for


def _ewma_weights(n: int, halflife: float) -> np.ndarray:
    lam = 0.5 ** (1.0 / halflife)
    w = lam ** np.arange(n - 1, -1, -1, dtype=float)       # oldest -> newest
    return w / w.sum()


def _weighted_var(x: np.ndarray, w: np.ndarray) -> float:
    mu = np.sum(w * x)
    # effective-sample-size correction so a short window is not biased low
    n_eff = 1.0 / np.sum(w ** 2)
    return float(np.sum(w * (x - mu) ** 2) * n_eff / max(n_eff - 1.0, 1.0))


def yang_zhang_variance(open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series,
                        *, halflife: float | None = 60.0) -> float:
    """
    Daily variance of log returns from OHLC (Yang & Zhang 2000).

    sigma^2 = sigma_overnight^2 + k * sigma_open_to_close^2 + (1 - k) * sigma_RS^2,
    k = 0.34 / (1.34 + (m + 1) / (m - 1)), with m the (effective) sample size.
    Observations are EWMA-weighted when `halflife` is given, equal-weighted otherwise.
    """
    df = pd.concat({"o": open_, "h": high, "l": low, "c": close}, axis=1)
    df["c_prev"] = df["c"].shift(1)
    df = df.dropna()
    df = df[(df[["o", "h", "l", "c", "c_prev"]] > 0).all(axis=1)]
    n = len(df)
    if n < 20:
        raise EstimationError(f"Yang-Zhang needs at least 20 complete OHLC sessions, got {n}")
    overnight = np.log(df["o"] / df["c_prev"]).to_numpy()
    open_close = np.log(df["c"] / df["o"]).to_numpy()
    rs = (np.log(df["h"] / df["c"]) * np.log(df["h"] / df["o"])
          + np.log(df["l"] / df["c"]) * np.log(df["l"] / df["o"])).to_numpy()
    w = _ewma_weights(n, halflife) if halflife else np.full(n, 1.0 / n)
    m = 1.0 / np.sum(w ** 2)                                  # effective sample size
    k = 0.34 / (1.34 + (m + 1.0) / (m - 1.0))
    var = _weighted_var(overnight, w) + k * _weighted_var(open_close, w) + (1.0 - k) * float(np.sum(w * rs))
    if not np.isfinite(var) or var <= 0:
        raise EstimationError("Yang-Zhang produced a non-positive variance")
    return var


def nearest_correlation(corr: np.ndarray, *, tol: float = 1e-10, max_iter: int = 500) -> np.ndarray:
    """
    Higham (2002) alternating projections onto {PSD} and {unit diagonal}, with
    Dykstra's correction. Returns the input unchanged if it is already valid.
    """
    a = (np.asarray(corr, dtype=float) + np.asarray(corr, dtype=float).T) / 2.0
    if np.allclose(np.diag(a), 1.0) and np.linalg.eigvalsh(a).min() >= -1e-12:
        return a
    y, ds = a.copy(), np.zeros_like(a)
    for _ in range(max_iter):
        r = y - ds
        vals, vecs = np.linalg.eigh(r)
        x = (vecs * np.clip(vals, 0.0, None)) @ vecs.T
        ds = x - r
        y_new = x.copy()
        np.fill_diagonal(y_new, 1.0)
        if np.linalg.norm(y_new - y, "fro") / max(np.linalg.norm(y, "fro"), 1.0) < tol:
            y = y_new
            break
        y = y_new
    y = (y + y.T) / 2.0
    vals, vecs = np.linalg.eigh(y)                            # guard tiny negative round-off
    y = (vecs * np.clip(vals, 1e-12, None)) @ vecs.T
    d = np.sqrt(np.diag(y))
    return y / np.outer(d, d)


def shrunk_correlation(log_returns: pd.DataFrame, *, standardise_halflife: float = 60.0,
                       vol_floor_fraction: float = 0.25,
                       max_standardised: float = 8.0) -> tuple[pd.DataFrame, float]:
    """
    Ledoit-Wolf-shrunk correlation of volatility-standardised returns.

    Each return is divided by its EWMA volatility *as of the previous session*
    (no same-day information), so calm and turbulent periods contribute equally
    to the correlation estimate. Returns (correlation, shrinkage intensity).

    **Two guards, and they are not cosmetic.** A stock that barely moves for a
    stretch drives its EWMA volatility towards zero, and the next ordinary day
    then divides by almost nothing. Measured on the real NSE universe, a single
    standardised return of 137 raised the Ledoit-Wolf shrinkage intensity from
    0.03 to 0.90, which collapsed average correlation from 0.19 to 0.01 and made
    the portfolio look twice as diversified as it was: predicted volatility 6.2%
    against 13.9% realised. Shrinkage intensity is driven by the estimated
    variance of the sample covariance, so one extreme observation is enough.

    So the standardising volatility is floored at a fraction of that asset's own
    median, and the standardised returns are winsorised. Both are standard
    practice in risk-model construction, and either alone would have prevented it.
    """
    r = log_returns.dropna(how="any")
    if len(r) < 60:
        raise EstimationError(f"correlation needs at least 60 complete sessions, got {len(r)}")
    if not 0.0 <= vol_floor_fraction < 1.0:
        raise EstimationError(f"vol_floor_fraction must be in [0, 1), got {vol_floor_fraction}")
    if max_standardised <= 3.0:
        raise EstimationError(f"max_standardised must exceed 3, got {max_standardised}")
    vol = np.sqrt(r.pow(2).ewm(halflife=standardise_halflife, adjust=True).mean()).shift(1)
    if vol_floor_fraction > 0:
        vol = vol.clip(lower=vol.median() * vol_floor_fraction, axis=1)
    z = (r / vol).dropna(how="any").clip(-max_standardised, max_standardised)
    lw = LedoitWolf().fit(z.to_numpy())
    cov = lw.covariance_
    d = np.sqrt(np.diag(cov))
    corr = nearest_correlation(cov / np.outer(d, d))
    return pd.DataFrame(corr, index=r.columns, columns=r.columns), float(lw.shrinkage_)


def estimate_covariance(ohlc: dict[str, pd.DataFrame], *, tickers: list[str] | None = None,
                        settings: CovarianceSettings = CovarianceSettings(),
                        periods_per_year: float = 252.0, as_of=None,
                        implied: pd.Series | None = None) -> CovarianceEstimate:
    """
    Daily log-return covariance for `tickers` from OHLC frames (dates x tickers).

    Pass the frames from `DataView.ohlc()`, which are already truncated at as-of,
    so nothing here can see the future. `implied` is India VIX, used by the HAR
    forecaster when available and ignored otherwise.

    Volatility comes from the HAR forecaster by default - measured 24% more
    accurate out-of-sample than the Yang-Zhang EWMA it replaced - falling back to
    Yang-Zhang, with the reason recorded, when there is too little history to fit
    a regression.
    """
    close = ohlc["close"]
    names = list(tickers) if tickers is not None else list(close.columns)
    missing = [t for t in names if t not in close.columns]
    if missing:
        raise EstimationError(f"no prices for: {missing[:8]}")
    if len(close) < settings.min_sessions:
        raise EstimationError(f"covariance needs at least {settings.min_sessions} sessions, got {len(close)}")

    degradations: list[str] = []
    vol_method = settings.vol_method
    sd = None
    if vol_method == "har":
        try:
            har = forecast_volatility(
                ohlc, tickers=names, as_of=as_of, implied=implied,
                settings=HarSettings(horizon=settings.horizon_days,
                                     periods_per_year=periods_per_year))
            sd = har.volatilities[names] / np.sqrt(periods_per_year)
            degradations.extend(har.provenance.degradations)
            vol_detail = har.provenance.window
        except EstimationError as exc:
            degradations.append(f"HAR volatility unavailable ({exc}); used Yang-Zhang EWMA instead")
            vol_method = "yang_zhang_ewma"
    if sd is None:
        vol_method = "yang_zhang_ewma"
        vol_window = {k: v[names].iloc[-settings.vol_lookback:] for k, v in ohlc.items()}
        variances = {t: yang_zhang_variance(vol_window["open"][t], vol_window["high"][t],
                                            vol_window["low"][t], vol_window["close"][t],
                                            halflife=settings.vol_halflife)
                     for t in names}
        sd = np.sqrt(pd.Series(variances)[names])
        vol_detail = (f"vol: {settings.vol_lookback} sessions, half-life {settings.vol_halflife:g}")

    log_ret = np.log(close[names] / close[names].shift(1)).iloc[1:].iloc[-settings.corr_lookback:]
    corr, shrink = shrunk_correlation(log_ret, standardise_halflife=settings.standardise_halflife,
                                      vol_floor_fraction=settings.vol_floor_fraction,
                                      max_standardised=settings.max_standardised)
    corr = corr.loc[names, names]
    matrix = corr.to_numpy() * np.outer(sd.to_numpy(), sd.to_numpy())
    label = "HAR" if vol_method == "har" else "Yang-Zhang EWMA"
    prov = Provenance(
        method=f"{label} volatility x Ledoit-Wolf shrunk correlation",
        window=(f"{vol_detail}; corr: {min(len(log_ret), settings.corr_lookback)} sessions, "
                f"shrinkage {shrink:.2f}"),
        as_of=pd.Timestamp(as_of) if as_of is not None else pd.Timestamp(close.index[-1]),
        source="OHLC",
        degradations=tuple(degradations),
    )
    return CovarianceEstimate(matrix=pd.DataFrame(matrix, index=names, columns=names),
                              periodicity=Periodicity.DAILY, provenance=prov,
                              periods_per_year=periods_per_year,
                              vol_method=vol_method, corr_method="ledoit_wolf")

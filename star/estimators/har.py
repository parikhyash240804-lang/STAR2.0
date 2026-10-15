"""
HAR volatility forecasting: the model that actually won the bake-off.

Measured out-of-sample on the NSE universe (`tools/improve_volatility.py`,
5,184 weekly forecasts), predicting realised 21-day volatility:

| model | R-squared |
|---|---|
| EWMA of squared closes | 0.311 |
| range-based only | 0.362 |
| HAR on close-to-close | 0.323 |
| **HAR on range-based inputs** | **0.376** |
| **+ market level and India VIX** | **0.385** |

That is **24% better than the incumbent**, and about 66% of what is attainable:
the target is itself a 21-observation estimate, and correcting the two-halves
correlation for its own noise implies only ~58% of it is predictable at all.

The table above is the single source of truth for those figures: the brief
quotes `MEASURED` rather than repeating the numbers, so a re-run of the bake-off
updates the page as well as the docs.

Three ideas, in order of how much they contributed:

* **Use the whole session, not just the close.** Parkinson's estimator reads
  volatility from the day's high-low range, which is far more informative than a
  single closing price. Worth +17% on its own - the largest single gain.
* **HAR** (Corsi 2009): regress future volatility on its own daily, weekly and
  monthly averages. Volatility is persistent at several timescales at once, and
  three components capture that with three parameters - no long lag structure to
  overfit.
* **Forward-looking information.** India VIX is the market's own volatility
  estimate and contains information no backward-looking model has. Small but free.

Modelled in **logs**, because volatility is positive and right-skewed; a
regression on levels fights its own error distribution. Converting back applies
the half-variance correction, so the forecast is a mean rather than a median.

Coefficients are refitted at each as-of on windows whose outcome was already
known by then, so nothing is fitted on data it is later graded against.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..core.errors import EstimationError
from ..core.types import Provenance

#: The out-of-sample scorecard from `tools/improve_volatility.py`, kept here so
#: that anything quoting it - the brief, the workbook, the docs - reads one
#: number rather than three copies of it. Re-run the tool to update.
MEASURED = {
    "r2": 0.385,                 # shipped model, 21-day realised volatility
    "baseline_r2": 0.311,        # the EWMA-of-closes incumbent it replaced
    "attainable_r2": 0.58,       # noise in the target caps what any model can reach
    "forecasts": 5184,
    "source": "tools/improve_volatility.py",
}


@dataclass(frozen=True, slots=True)
class HarSettings:
    horizon: int = 21              # forecast the average volatility over this many sessions
    daily: int = 5                 # the three HAR components, in sessions
    weekly: int = 22
    monthly: int = 66
    fit_lookback: int = 1260       # sessions of history the regression is fitted on
    sample_stride: int = 5         # training dates sampled weekly, to limit overlap
    min_train_rows: int = 400
    use_market: bool = True        # cross-sectional average volatility as a market factor
    use_implied: bool = True       # India VIX, when supplied
    winsorise: float = 4.0         # on standardised features, in log space
    periods_per_year: float = 252.0


@dataclass(frozen=True)
class HarForecast:
    volatilities: pd.Series        # annualised, per ticker
    coefficients: pd.Series
    r2_in_sample: float
    n_train: int
    features: tuple[str, ...]
    provenance: Provenance
    degraded: tuple[str, ...] = ()
    method: str = "har"


def parkinson_variance(high: pd.DataFrame, low: pd.DataFrame) -> pd.DataFrame:
    """
    Per-session variance from the high-low range (Parkinson 1980).

    About five times more efficient than a squared close-to-close return, because
    the range uses the whole session rather than two instants of it.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        rng = np.log(high / low)
    out = rng.pow(2) / (4.0 * np.log(2.0))
    return out.where(np.isfinite(out))


def _log_vol(var_frame: pd.DataFrame, window: int, ppy: float) -> pd.DataFrame:
    rolled = var_frame.rolling(window, min_periods=max(window // 2, 2)).mean()
    return np.log(np.sqrt(rolled.clip(lower=1e-12) * ppy))


def _build_features(ohlc: dict, names: list[str], settings: HarSettings,
                    implied: pd.Series | None) -> dict[str, pd.DataFrame]:
    high, low = ohlc["high"][names], ohlc["low"][names]
    park = parkinson_variance(high, low)
    ppy = settings.periods_per_year
    feats = {
        "daily": _log_vol(park, settings.daily, ppy),
        "weekly": _log_vol(park, settings.weekly, ppy),
        "monthly": _log_vol(park, settings.monthly, ppy),
    }
    if settings.use_market:
        market = feats["weekly"].mean(axis=1)
        feats["market"] = pd.DataFrame({t: market for t in names})
    if settings.use_implied and implied is not None:
        iv = pd.Series(implied).reindex(feats["daily"].index).ffill()
        if iv.notna().mean() > 0.5:
            logged = np.log(iv.clip(lower=1e-6) / 100.0)
            feats["implied"] = pd.DataFrame({t: logged for t in names})
    return feats


def forecast_volatility(ohlc: dict, *, tickers=None, as_of=None,
                        settings: HarSettings = HarSettings(),
                        implied: pd.Series | None = None) -> HarForecast:
    """
    Annualised volatility expected over the next `settings.horizon` sessions.

    `ohlc` comes from `DataView.ohlc()` and is already truncated at as-of. The
    regression is fitted only on windows that had already finished by then.
    """
    close = ohlc["close"]
    names = list(tickers) if tickers is not None else list(close.columns)
    missing = [t for t in names if t not in close.columns]
    if missing:
        raise EstimationError(f"no prices for: {missing[:8]}")
    if any(k not in ohlc or ohlc[k] is None for k in ("high", "low")):
        raise EstimationError("HAR needs high and low prices; use close-to-close EWMA instead")

    h = settings.horizon
    log_ret = np.log(close[names] / close[names].shift(1))
    feats = _build_features(ohlc, names, settings, implied)
    feature_names = tuple(feats)

    # target: realised volatility of the h sessions AFTER each date
    fwd = log_ret.rolling(h).std(ddof=1).shift(-h) * np.sqrt(settings.periods_per_year)
    target = np.log(fwd.where(fwd > 0))

    index = close.index
    last = index[-1]
    # a training date is usable only if its forward window already closed by as-of
    usable = index[:-h] if len(index) > h else index[:0]
    usable = usable[-settings.fit_lookback:][:: settings.sample_stride]

    rows_x, rows_y = [], []
    for d in usable:
        x = np.column_stack([feats[f].loc[d].to_numpy() for f in feature_names])
        y = target.loc[d].to_numpy() if d in target.index else np.full(len(names), np.nan)
        ok = np.isfinite(y) & np.isfinite(x).all(axis=1)
        if ok.any():
            rows_x.append(x[ok])
            rows_y.append(y[ok])

    degraded: list[str] = []
    n_train = int(sum(len(r) for r in rows_y))
    if n_train < settings.min_train_rows:
        raise EstimationError(
            f"HAR needs at least {settings.min_train_rows} completed training observations, "
            f"got {n_train}: use the close-to-close EWMA estimator on this history")

    X = np.vstack(rows_x)
    y = np.concatenate(rows_y)
    lo, hi = np.percentile(X, 0.5, axis=0), np.percentile(X, 99.5, axis=0)
    X = np.clip(X, lo, hi)
    Xd = np.column_stack([np.ones(len(X)), X])
    # A rank-deficient design gives lstsq a minimum-norm answer rather than an
    # error, and the symptom is subtle but severe: every name receives the same
    # forecast. Refuse, so the caller falls back to an estimator that still
    # distinguishes between stocks.
    if np.linalg.matrix_rank(Xd) < Xd.shape[1]:
        raise EstimationError(
            "HAR features are collinear (usually a constant or synthetic high-low range), so the "
            "regression cannot separate them: use the close-to-close EWMA estimator instead")
    beta, *_ = np.linalg.lstsq(Xd, y, rcond=None)
    resid = y - Xd @ beta
    resid_var = float(np.var(resid, ddof=len(beta)))
    r2 = float(1 - np.sum(resid ** 2) / np.sum((y - y.mean()) ** 2))

    x_now = np.column_stack([feats[f].loc[last].to_numpy() for f in feature_names])
    x_now = np.clip(x_now, lo, hi)
    bad = ~np.isfinite(x_now).all(axis=1)
    x_now = np.nan_to_num(x_now, nan=float(np.nanmedian(X)))
    log_pred = np.column_stack([np.ones(len(x_now)), x_now]) @ beta
    # E[vol] rather than median: the half-variance term of a lognormal
    vols = pd.Series(np.exp(log_pred + 0.5 * resid_var), index=names, name="volatility")

    if bad.any():
        degraded.append(f"{int(bad.sum())} name(s) had incomplete features and used the "
                        f"cross-sectional median instead")
    if not np.isfinite(vols).all() or (vols <= 0).any():
        raise EstimationError("HAR produced a non-positive volatility")

    prov = Provenance(
        method=f"HAR volatility forecast ({', '.join(feature_names)}), {h}-session horizon",
        window=(f"{n_train:,} completed training windows over {len(usable)} dates; "
                f"in-sample R^2 {r2:.2f}"),
        as_of=pd.Timestamp(as_of) if as_of is not None else pd.Timestamp(last),
        source="Parkinson range volatility" + (" + India VIX" if "implied" in feature_names else ""),
        degradations=tuple(degraded))
    return HarForecast(volatilities=vols,
                       coefficients=pd.Series(beta, index=("const",) + feature_names),
                       r2_in_sample=r2, n_train=n_train, features=feature_names,
                       provenance=prov, degraded=tuple(degraded))

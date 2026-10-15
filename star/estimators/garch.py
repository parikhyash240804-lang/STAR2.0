"""
Conditional volatility: GJR-GARCH(1,1) with Student-t innovations.

The covariance module estimates *today's* volatility well (Yang-Zhang EWMA).
This module answers the different question the forecast needs: **what does
volatility look like over the next h days?** Volatility is persistent and
mean-reverting, so a flat extrapolation of today's level is wrong in both
directions - it overstates risk after a shock and understates it in calm.

Model (Glosten-Jagannathan-Runkle 1993; Bollerslev 1987 for the innovations):

    r_t  = mu + e_t,  e_t = sigma_t z_t,  z_t ~ standardised Student-t(nu)
    s2_t = omega + (alpha + gamma * 1[e_{t-1} < 0]) e_{t-1}^2 + beta * s2_{t-1}

* `gamma` captures the leverage effect: falls raise tomorrow's volatility more
  than rises of equal size. Symmetric GARCH cannot express this, and Indian
  equity data shows it clearly.
* Student-t innovations stop the tail risk being understated. A Gaussian GARCH
  fitted to daily NSE returns produces VaR breaches more often than its
  confidence level allows.
* Persistence = alpha + gamma/2 + beta (the 1/2 is E[1[z<0]] for any symmetric
  innovation). Below 1 the process is stationary and forecasts mean-revert to
  omega / (1 - persistence); at or above 1 the model is not usable and we say so.

Multi-day forecasts are analytic, not simulated, so they are exact and cheap.
`cumulative_variance(h)` sums the per-day variances, which assumes returns are
serially uncorrelated - stated here because it is the assumption that would
matter if it failed.

Degradation, never silence: too little history, a failed optimisation, a
non-stationary fit or a degenerate tail index all fall back to EWMA and record
the reason in the provenance. The legacy switched method quietly; here the
caller can always see which one ran.

See PAPERS/notes/03_covariance_volatility.md.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from arch import arch_model

from ..core.errors import EstimationError
from ..core.types import CovarianceEstimate, Periodicity, Provenance

MAX_HORIZON = 756          # ~3 years; beyond this a 5-year sample is extrapolation


@dataclass(frozen=True, slots=True)
class GarchSettings:
    lookback: int = 1000         # sessions used to fit (~4 years)
    min_sessions: int = 500      # below this the likelihood is too flat to trust
    ewma_halflife: float = 60.0  # fallback estimator
    scale: float = 100.0         # fit in percent: the optimiser is ill-conditioned on ~1e-4 numbers
    max_persistence: float = 0.9995
    min_nu: float = 2.1          # variance of a Student-t exists only for nu > 2


@dataclass(frozen=True)
class VolForecast:
    """A per-day variance path in **daily log-return units**, plus how it was made."""

    ticker: str
    daily_variance: np.ndarray          # index 0 = one day ahead
    method: str                         # "gjr_garch_t" | "ewma"
    provenance: Provenance
    persistence: float | None = None
    long_run_variance: float | None = None
    nu: float | None = None
    params: dict[str, float] = field(default_factory=dict)
    periods_per_year: float = 252.0

    @property
    def horizon(self) -> int:
        return len(self.daily_variance)

    def _days(self, days: int | None) -> int:
        d = self.horizon if days is None else int(days)
        if not 1 <= d <= self.horizon:
            raise EstimationError(f"{self.ticker}: horizon {d} outside the forecast (1..{self.horizon})")
        return d

    def cumulative_variance(self, days: int | None = None) -> float:
        """Variance of the h-day cumulative log return (assumes no serial correlation)."""
        return float(self.daily_variance[: self._days(days)].sum())

    def average_daily_variance(self, days: int | None = None) -> float:
        d = self._days(days)
        return self.cumulative_variance(d) / d

    def annual_vol_path(self) -> np.ndarray:
        return np.sqrt(self.daily_variance * self.periods_per_year)

    def annual_vol(self, days: int | None = None) -> float:
        return float(np.sqrt(self.average_daily_variance(days) * self.periods_per_year))


class FitCache:
    """
    Fits are the compute hot spot (60 assets x 40 rebalances in a backtest), and
    the same (ticker, window) is refitted many times. Keyed by the window's end
    date and length, so a cache hit is always the same data.
    """

    def __init__(self) -> None:
        self._store: dict[tuple, object] = {}
        self.hits = 0
        self.misses = 0

    @staticmethod
    def key(ticker: str, r: pd.Series, settings: GarchSettings) -> tuple:
        return (ticker, str(r.index[0].date()), str(r.index[-1].date()), len(r),
                settings.lookback, settings.scale)

    def get(self, key):
        hit = self._store.get(key)
        if hit is None:
            self.misses += 1
        else:
            self.hits += 1
        return hit

    def put(self, key, value):
        self._store[key] = value
        return value


def _ewma_forecast(ticker: str, r: pd.Series, horizon: int, settings: GarchSettings,
                   as_of, reason: str) -> VolForecast:
    """RiskMetrics-style fallback: flat term structure at the current EWMA level."""
    if len(r) < 20:
        raise EstimationError(f"{ticker}: need at least 20 returns to estimate volatility, got {len(r)}")
    var = float(r.pow(2).ewm(halflife=settings.ewma_halflife, adjust=True).mean().iloc[-1])
    if not np.isfinite(var) or var <= 0:
        raise EstimationError(f"{ticker}: EWMA variance is not positive")
    prov = Provenance(
        method="EWMA variance (GJR-GARCH-t unavailable)",
        window=f"{len(r)} sessions, half-life {settings.ewma_halflife:g}",
        as_of=pd.Timestamp(as_of) if as_of is not None else pd.Timestamp(r.index[-1]),
        source="close-to-close log returns",
    ).degraded(reason)
    return VolForecast(ticker=ticker, daily_variance=np.full(horizon, var), method="ewma",
                       provenance=prov, long_run_variance=var)


def _fit(ticker: str, r: pd.Series, settings: GarchSettings, cache: FitCache | None):
    """Fit GJR-GARCH(1,1)-t on percent returns. Returns the arch result, or None."""
    key = FitCache.key(ticker, r, settings) if cache is not None else None
    if cache is not None:
        hit = cache.get(key)
        if hit is not None:
            return hit
    model = arch_model(r.to_numpy() * settings.scale, mean="Constant", vol="GARCH",
                       p=1, o=1, q=1, dist="t", rescale=False)
    try:
        res = model.fit(disp="off", show_warning=False, options={"maxiter": 500})
    except Exception:                                    # optimiser blew up: caller falls back
        res = None
    if cache is not None:
        cache.put(key, res)
    return res


def forecast_variance(log_returns: pd.Series, horizon: int, *, ticker: str = "",
                      settings: GarchSettings = GarchSettings(), as_of=None,
                      cache: FitCache | None = None, periods_per_year: float = 252.0) -> VolForecast:
    """
    Per-day variance forecast for one asset, 1..`horizon` days ahead.

    `log_returns` must already be truncated at as-of (use `DataView`). Falls back
    to EWMA - with the reason recorded - rather than failing or faking a fit.
    """
    if not 1 <= int(horizon) <= MAX_HORIZON:
        raise EstimationError(f"horizon must be between 1 and {MAX_HORIZON} days, got {horizon}")
    horizon = int(horizon)
    name = ticker or getattr(log_returns, "name", "") or "asset"
    r = pd.Series(log_returns).dropna()
    r = r[np.isfinite(r.to_numpy())].iloc[-settings.lookback:]

    if len(r) < settings.min_sessions:
        return _ewma_forecast(name, r, horizon, settings, as_of,
                              f"only {len(r)} sessions, GJR-GARCH needs {settings.min_sessions}")
    res = _fit(name, r, settings, cache)
    if res is None or getattr(res, "convergence_flag", 0) != 0:
        return _ewma_forecast(name, r, horizon, settings, as_of, "GJR-GARCH-t did not converge")

    p = {k: float(v) for k, v in res.params.items()}
    persistence = p["alpha[1]"] + 0.5 * p["gamma[1]"] + p["beta[1]"]
    if not np.isfinite(persistence) or persistence >= settings.max_persistence:
        return _ewma_forecast(name, r, horizon, settings, as_of,
                              f"GJR-GARCH-t is non-stationary (persistence {persistence:.4f})")
    if not np.isfinite(p.get("nu", np.nan)) or p["nu"] <= settings.min_nu:
        return _ewma_forecast(name, r, horizon, settings, as_of,
                              f"tail index nu={p.get('nu', float('nan')):.2f} implies infinite variance")

    fc = res.forecast(horizon=horizon, reindex=False, method="analytic")
    var = np.asarray(fc.variance.iloc[-1], dtype=float) / settings.scale ** 2
    if not np.all(np.isfinite(var)) or np.any(var <= 0):
        return _ewma_forecast(name, r, horizon, settings, as_of,
                              "GJR-GARCH-t produced a non-positive variance")

    long_run = p["omega"] / (1.0 - persistence) / settings.scale ** 2
    prov = Provenance(
        method="GJR-GARCH(1,1) with Student-t innovations",
        window=(f"{len(r)} sessions; persistence {persistence:.3f}, "
                f"long-run vol {np.sqrt(long_run * periods_per_year):.1%}, nu {p['nu']:.1f}"),
        as_of=pd.Timestamp(as_of) if as_of is not None else pd.Timestamp(r.index[-1]),
        source="close-to-close log returns",
    )
    return VolForecast(ticker=name, daily_variance=var, method="gjr_garch_t", provenance=prov,
                       persistence=persistence, long_run_variance=long_run, nu=p["nu"],
                       params=p, periods_per_year=periods_per_year)


def forecast_panel(log_returns: pd.DataFrame, horizon: int, *, tickers: list[str] | None = None,
                   settings: GarchSettings = GarchSettings(), as_of=None,
                   cache: FitCache | None = None) -> dict[str, VolForecast]:
    """One `VolForecast` per ticker, in the order given."""
    names = list(tickers) if tickers is not None else list(log_returns.columns)
    missing = [t for t in names if t not in log_returns.columns]
    if missing:
        raise EstimationError(f"no returns for: {missing[:8]}")
    return {t: forecast_variance(log_returns[t], horizon, ticker=t, settings=settings,
                                 as_of=as_of, cache=cache) for t in names}


def scale_covariance_to_horizon(cov: CovarianceEstimate, forecasts: dict[str, VolForecast],
                                days: int) -> CovarianceEstimate:
    """
    Re-scale a daily covariance to the *average daily* volatility expected over
    the next `days`, keeping the (slow-moving, well-estimated) correlation.

    This is the bridge between the two estimators: correlation from the long
    shrunk window, level from the conditional model. The result is still a daily
    covariance - multiply by `days` for the horizon, so h=1 reproduces the
    conditional one-day variance exactly.
    """
    names = list(cov.matrix.columns)
    missing = [t for t in names if t not in forecasts]
    if missing:
        raise EstimationError(f"no volatility forecast for: {missing[:8]}")
    sd = np.sqrt(np.array([forecasts[t].average_daily_variance(days) for t in names]))
    corr = cov.correlation().to_numpy()
    matrix = corr * np.outer(sd, sd)
    methods = sorted({forecasts[t].method for t in names})
    prov = Provenance(
        method=f"{'+'.join(methods)} volatility x {cov.corr_method} correlation",
        window=f"{days}-day average conditional variance; correlation from {cov.provenance.window}",
        as_of=cov.provenance.as_of,
        source=cov.provenance.source,
        degradations=tuple(sorted({d for t in names for d in forecasts[t].provenance.degradations})),
    )
    return CovarianceEstimate(matrix=pd.DataFrame(matrix, index=names, columns=names),
                              periodicity=Periodicity.DAILY, provenance=prov,
                              periods_per_year=cov.periods_per_year,
                              vol_method="+".join(methods), corr_method=cov.corr_method)

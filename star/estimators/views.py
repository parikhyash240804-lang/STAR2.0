"""
Turning a signal into views of the right size - Grinold (1994).

    alpha_i = IC * sigma_i * z_i

where `z` is the cross-sectionally standardised score and `IC` is the
information coefficient: the rank correlation between the signal and the returns
that followed. Three things follow from that one formula, and each fixes a
verified legacy defect:

* **The size of a view depends on how predictive the signal has been** (IC is
  *measured* from history, never asserted). The legacy mapped scores onto a fixed
  +1.5% to +15.5% band regardless of predictive power (M2B-05), and the band was
  bullish-only, so the model could not dislike a stock (M2B-01).
* **A volatile stock gets a bigger view than a quiet one for the same score**,
  because the same conviction is worth more return where there is more risk.
* **Scores are demeaned cross-sectionally**, so a signal that ranks stocks
  against each other produces relative views, not an accidental market call.

Default views are **relative quintile baskets** (top fifth minus bottom fifth):
a cross-sectional signal says which stocks beat which, and saying that directly
keeps stock selection separate from the market call.

**Confidence is derived from evidence, not typed in.** The IC's t-statistic sets
the Idzorek confidence, with the overlap of forward-return windows corrected for
(a 21-day IC measured daily has roughly n/21 independent observations, not n).
A signal that is not statistically distinguishable from noise gets confidence
zero, which leaves the posterior at the prior - the model declines to bet.

See PAPERS/notes/02_black_litterman.md.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..core.errors import EstimationError
from ..core.types import CovarianceEstimate
from .black_litterman import View


@dataclass(frozen=True, slots=True)
class ViewSettings:
    ic_min: float = 0.02          # a floor: below this the signal is not worth trading
    ic_max: float = 0.10          # a ceiling: a measured IC above this is almost always a data artifact
    quantiles: int = 5            # quintile baskets
    max_confidence: float = 0.50  # a single cross-sectional signal never gets more than half the way
    min_t_stat: float = 1.5       # below this, confidence is zero: the model declines to bet
    t_stat_for_max: float = 4.0   # the t-statistic at which confidence reaches its cap


@dataclass(frozen=True)
class ICEstimate:
    """A measured information coefficient with the honest uncertainty around it."""

    mean: float
    std_error: float
    observations: int             # independent, after correcting for window overlap
    horizon_days: int
    raw_mean: float               # before clipping, so the report can show what was measured
    clipped: bool
    note: str | None = None

    @property
    def t_stat(self) -> float:
        return 0.0 if self.std_error <= 0 else float(self.mean / self.std_error)

    def confidence(self, settings: ViewSettings = ViewSettings()) -> float:
        """Idzorek confidence from the t-statistic, linear between the two thresholds."""
        t = abs(self.t_stat)
        if t <= settings.min_t_stat:
            return 0.0
        span = max(settings.t_stat_for_max - settings.min_t_stat, 1e-9)
        return float(min((t - settings.min_t_stat) / span, 1.0) * settings.max_confidence)


def rank_ic(scores: pd.DataFrame, close: pd.DataFrame, horizon_days: int, *,
            settings: ViewSettings = ViewSettings(), min_names: int = 5) -> ICEstimate:
    """
    Rolling rank information coefficient of `scores` against the returns that
    followed them.

    `scores` is (dates x tickers), known at the close of each date. The IC on
    date t pairs the score at t with the return from t to t + horizon, so every
    pair uses information in the order it arrived. Both frames must already be
    truncated at as-of; pairs whose forward window extends past the data are
    dropped rather than shortened.
    """
    if horizon_days < 1:
        raise EstimationError(f"IC horizon must be at least 1 day, got {horizon_days}")
    common = scores.columns.intersection(close.columns)
    if len(common) < min_names:
        raise EstimationError(f"IC needs at least {min_names} shared names, got {len(common)}")
    s = scores[common].reindex(close.index).dropna(how="all")
    fwd = (close[common].shift(-horizon_days) / close[common]) - 1.0

    ics = []
    for date in s.index:
        if date not in fwd.index:
            continue
        pair = pd.concat([s.loc[date], fwd.loc[date]], axis=1, keys=["s", "r"]).dropna()
        if len(pair) < min_names or pair["s"].nunique() < 2:
            continue
        ics.append(pair["s"].corr(pair["r"], method="spearman"))
    ics = pd.Series(ics, dtype=float).dropna()
    if len(ics) < 2:
        raise EstimationError("not enough overlapping score/return pairs to measure an IC")

    # overlapping forward windows are not independent observations
    n_eff = max(len(ics) / horizon_days, 1.0)
    raw = float(ics.mean())
    se = float(ics.std(ddof=1) / np.sqrt(n_eff))

    magnitude = float(np.clip(abs(raw), settings.ic_min if abs(raw) > 0 else 0.0, settings.ic_max))
    clipped = not np.isclose(magnitude, abs(raw))
    note = None
    if clipped:
        note = (f"measured IC {raw:+.3f} clipped to {np.sign(raw) * magnitude:+.3f} "
                f"(plausible range {settings.ic_min}-{settings.ic_max})")
    return ICEstimate(mean=float(np.sign(raw) * magnitude), std_error=se, observations=int(n_eff),
                      horizon_days=int(horizon_days), raw_mean=raw, clipped=clipped, note=note)


def standardise(scores: pd.Series, *, winsorise: float = 3.0) -> pd.Series:
    """Cross-sectionally demeaned, unit-variance scores, winsorised at +/- 3 sd."""
    s = pd.Series(scores, dtype=float).dropna()
    if len(s) < 2:
        raise EstimationError("standardising a score needs at least two names")
    sd = s.std(ddof=1)
    if not np.isfinite(sd) or sd <= 0:
        raise EstimationError("scores have no cross-sectional dispersion; there is nothing to say")
    z = (s - s.mean()) / sd
    return z.clip(-winsorise, winsorise)


def grinold_alpha(scores: pd.Series, cov: CovarianceEstimate, ic: float) -> pd.Series:
    """
    alpha = IC * sigma * z, in the periodicity of `cov` (annualise it first for
    annual alphas). Signed, so the model can dislike a stock.
    """
    if not np.isfinite(ic):
        raise EstimationError("IC must be finite")
    z = standardise(scores)
    vols = cov.volatilities().reindex(z.index)
    if vols.isna().any():
        raise EstimationError(f"no volatility for: {list(vols.index[vols.isna()][:8])}")
    return (ic * vols * z).rename("alpha")


def quantile_baskets(scores: pd.Series, quantiles: int = 5) -> tuple[list[str], list[str]]:
    """Top and bottom baskets by score. Equal counts, ties broken by rank order."""
    s = pd.Series(scores, dtype=float).dropna()
    if len(s) < 2 * quantiles:
        raise EstimationError(f"{quantiles} baskets need at least {2 * quantiles} names, got {len(s)}")
    order = s.sort_values(ascending=False)
    size = max(len(order) // quantiles, 1)
    return list(order.index[:size]), list(order.index[-size:])


def quintile_view(scores: pd.Series, cov: CovarianceEstimate, ic: ICEstimate, *,
                  name: str = "signal spread", settings: ViewSettings = ViewSettings()) -> View:
    """
    The default view: the top basket beats the bottom basket by the difference in
    their Grinold alphas, at a confidence set by the IC's t-statistic.

    Equal-weighted within each basket, coefficients summing to zero, so this
    expresses selection only - no implicit view on the market.
    """
    annual = cov.annualise()
    alpha = grinold_alpha(scores, annual, ic.mean)
    top, bottom = quantile_baskets(alpha.reindex(scores.dropna().index), settings.quantiles)
    picks = {t: 1.0 / len(top) for t in top}
    for t in bottom:
        picks[t] = picks.get(t, 0.0) - 1.0 / len(bottom)
    q = float(alpha[top].mean() - alpha[bottom].mean())
    return View(name=name, picks=picks, q=q, confidence=ic.confidence(settings))


def absolute_view(ticker: str, annual_excess_return: float, confidence: float, *,
                  name: str | None = None) -> View:
    """A user-supplied absolute view. Only ever created on request, never inferred."""
    return View(name=name or f"{ticker} absolute", picks={ticker: 1.0},
                q=float(annual_excess_return), confidence=float(confidence))

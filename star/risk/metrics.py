"""
What can this portfolio lose, over what period, and which holdings carry it.

**Expected shortfall at 97.5% is the headline**, not value-at-risk. VaR says "you
will not lose more than X, 97.5% of the time" and is silent about the other 2.5%;
ES answers "when it does go wrong, how bad is the average case". ES is also
coherent - it can never say that splitting a portfolio in two increases its risk,
which VaR can (Artzner et al. 1999), and it is the Basel FRTB measure. VaR is
reported alongside because people are used to reading it.

**Every measure belongs to a horizon and is computed at that horizon.** There is
deliberately no function in this module that converts a one-day figure into a
ten-day one. That conversion (multiply by sqrt(10)) assumes independent,
identically distributed, symmetric returns - none of which holds in a crisis,
which is exactly when the number is read. The legacy did it and labelled a
one-year loss as one-day (M6-02).

Other things here:

* **Euler contributions** - risk split by holding, both for volatility and for
  expected shortfall, each summing exactly to the portfolio total. This is what
  supports "three holdings carry 41% of the risk" in the brief.
* **Drawdown-at-risk and conditional drawdown** (Chekhlov, Uryasev & Zabarankin
  2005) - computed along simulated paths, because a drawdown is a property of a
  path, not of an endpoint. The legacy sliced its one-year figure instead.
* **Honesty about the tail**: a 99% figure from 200 observations rests on two
  data points. Where the tail is too thin to support the number, the report says
  so rather than printing it with three decimals.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..core.errors import EstimationError
from ..core.types import CovarianceEstimate, Provenance

MIN_TAIL_OBSERVATIONS = 10


# --------------------------------------------------------------- the sample ---
@dataclass(frozen=True)
class HorizonSample:
    """
    A sample of portfolio returns **at one horizon**, and where it came from.

    Constructed from real overlapping windows, from simulated paths, or from a
    backtest. Carrying the horizon on the object is what makes a mislabelled
    figure hard to produce.
    """

    returns: pd.Series                 # simple returns over `horizon_days`
    horizon_days: int
    provenance: Provenance
    overlapping: bool = False

    def __post_init__(self) -> None:
        if self.horizon_days < 1:
            raise EstimationError(f"horizon must be at least 1 day, got {self.horizon_days}")
        if len(self.returns) < 30:
            raise EstimationError(f"a risk figure needs at least 30 observations, got {len(self.returns)}")

    @property
    def effective_observations(self) -> float:
        """Overlapping windows are not independent: n/h of them are."""
        n = len(self.returns)
        return n / self.horizon_days if self.overlapping else float(n)


def horizon_sample_from_daily(daily_returns: pd.Series, horizon_days: int, *,
                              overlapping: bool = True, as_of=None) -> HorizonSample:
    """
    Build an h-day sample by **compounding actual consecutive days** - never by
    scaling a one-day number.

    Overlapping windows use the history more efficiently but are serially
    dependent, so the effective sample size is recorded and used by every test
    downstream.
    """
    r = pd.Series(daily_returns).dropna()
    if horizon_days < 1:
        raise EstimationError(f"horizon must be at least 1 day, got {horizon_days}")
    log = np.log1p(r)
    if horizon_days == 1:
        agg = r
    elif overlapping:
        agg = np.expm1(log.rolling(horizon_days).sum().dropna())
    else:
        blocks = len(log) // horizon_days
        if blocks < 30:
            raise EstimationError(f"only {blocks} non-overlapping {horizon_days}-day windows available; "
                                  f"use overlapping=True and accept the dependence, or a longer history")
        trimmed = log.iloc[len(log) - blocks * horizon_days:]
        agg = np.expm1(trimmed.groupby(np.arange(len(trimmed)) // horizon_days).sum())
        agg.index = trimmed.index[horizon_days - 1::horizon_days][:len(agg)]
    prov = Provenance(
        method=f"{horizon_days}-day returns compounded from daily history",
        window=f"{len(agg)} {'overlapping' if overlapping and horizon_days > 1 else 'independent'} windows",
        as_of=pd.Timestamp(as_of) if as_of is not None else pd.Timestamp(r.index[-1]),
        source="realised daily returns",
    )
    return HorizonSample(returns=agg.rename(f"r_{horizon_days}d"), horizon_days=int(horizon_days),
                         provenance=prov, overlapping=overlapping and horizon_days > 1)


# ------------------------------------------------------------- the measures ---
def value_at_risk(sample: HorizonSample | pd.Series, confidence: float = 0.975) -> float:
    """
    Loss at the given confidence, as a **positive number** (0.08 means a loss of
    8% of capital). A profit at that quantile returns a negative VaR, which is
    the honest answer rather than a floor at zero.
    """
    r = _values(sample)
    _check_confidence(confidence)
    return float(-np.quantile(r, 1.0 - confidence, method="linear"))


def tail_weights(returns: np.ndarray, confidence: float) -> np.ndarray:
    """
    Weights that pick out exactly the worst (1 - confidence) of the sample,
    summing to one.

    Selecting "everything at or below the quantile" is wrong whenever the quantile
    lands on a repeated value: with a 4% chance of a large loss and a 5% tail, the
    cut sits on the common outcome and sweeps in the whole sample, understating
    expected shortfall by an order of magnitude. This is the Acerbi-Tasche
    integrated estimator instead - the worst k observations in full, plus a
    fraction of the next one - which is exact for discrete data and converges to
    the closed form for continuous data.
    """
    n = len(returns)
    k_exact = (1.0 - confidence) * n
    order = np.argsort(returns, kind="stable")
    w = np.zeros(n)
    k = int(np.floor(k_exact + 1e-12))
    if k > 0:
        w[order[:k]] = 1.0
    remainder = k_exact - k
    if remainder > 1e-12 and k < n:
        w[order[k]] = remainder
    total = w.sum()
    if total <= 0:                                   # fewer than one observation in the tail
        w[order[0]] = 1.0
        total = 1.0
    return w / total


def expected_shortfall(sample: HorizonSample | pd.Series, confidence: float = 0.975) -> float:
    """Average loss in the worst (1 - confidence) of cases. The headline measure."""
    r = _values(sample)
    _check_confidence(confidence)
    return float(-np.sum(tail_weights(r, confidence) * r))


def tail_reliability(sample: HorizonSample, confidence: float) -> tuple[bool, str | None]:
    """Is there enough data in the tail to support this number?"""
    n_eff = sample.effective_observations
    expected_tail = n_eff * (1.0 - confidence)
    if expected_tail < MIN_TAIL_OBSERVATIONS:
        return False, (f"the {confidence:.1%} figure rests on about {expected_tail:.0f} independent "
                       f"observations in the tail; treat it as indicative, not precise")
    return True, None


def drawdown_at_risk(paths: np.ndarray, confidence: float = 0.95) -> tuple[float, float]:
    """
    Worst peak-to-trough decline along a path, at the given confidence, plus the
    conditional drawdown (the average of the worst tail).

    `paths` is (steps, n_paths) of cumulative portfolio **values**. Computed
    inside the path, not sliced from an endpoint distribution - the legacy's
    error was treating a one-year number as if it contained the path.
    """
    p = np.asarray(paths, dtype=float)
    if p.ndim != 2 or p.shape[0] < 2:
        raise EstimationError("drawdown needs a (steps, paths) array with at least two steps")
    _check_confidence(confidence)
    running_max = np.maximum.accumulate(p, axis=0)
    dd = 1.0 - p / running_max
    worst = dd.max(axis=0)
    dar = float(np.quantile(worst, confidence, method="linear"))
    cdar = float(worst[worst >= dar].mean()) if np.any(worst >= dar) else dar
    return dar, cdar


# ------------------------------------------------------- risk by holding ---
def volatility_contributions(weights: pd.Series, cov: CovarianceEstimate) -> pd.DataFrame:
    """
    Euler decomposition of portfolio volatility: contributions sum to the total.

    `marginal` is the change in portfolio volatility per unit of weight, which is
    what tells you whether adding to a position raises or lowers total risk.
    """
    names = list(weights.index)
    S = cov.annualise().matrix.reindex(index=names, columns=names)
    if S.isna().to_numpy().any():
        raise EstimationError("covariance is missing entries for some holdings")
    w = weights.to_numpy()
    Sw = S.to_numpy() @ w
    vol = float(np.sqrt(w @ Sw))
    if vol <= 0:
        raise EstimationError("portfolio volatility is not positive")
    marginal = Sw / vol
    contribution = w * marginal
    return pd.DataFrame({"weight": w, "marginal": marginal, "contribution": contribution,
                         "share": contribution / vol}, index=names)


def shortfall_contributions(asset_returns: pd.DataFrame, weights: pd.Series,
                            confidence: float = 0.975) -> pd.DataFrame:
    """
    Each holding's share of expected shortfall: its average return in exactly
    those scenarios where the portfolio is in its worst tail.

    Unlike the volatility split, this uses the co-movement that actually shows up
    in bad states, which is where correlations behave differently from their
    long-run averages.
    """
    names = list(weights.index)
    R = asset_returns[names].dropna()
    if len(R) < 30:
        raise EstimationError(f"shortfall contributions need at least 30 scenarios, got {len(R)}")
    _check_confidence(confidence)
    port = R.to_numpy() @ weights.to_numpy()
    w_tail = tail_weights(port, confidence)
    if int(np.sum(w_tail > 0)) < 2:
        raise EstimationError("too few scenarios in the tail to attribute expected shortfall")
    # the same tail weighting as `expected_shortfall`, so the parts sum to the whole
    contribution = -weights.to_numpy() * (w_tail @ R.to_numpy())
    es = contribution.sum()
    return pd.DataFrame({"weight": weights.to_numpy(), "contribution": contribution,
                         "share": contribution / es if es != 0 else np.nan},
                        index=names).sort_values("contribution", ascending=False)


def diversification_ratio(weights: pd.Series, cov: CovarianceEstimate) -> float:
    """
    Weighted average volatility divided by portfolio volatility (Choueifaty &
    Coignard 2008). 1.0 means no diversification benefit at all; falling values
    over time are an early warning that holdings are starting to move together.
    """
    names = list(weights.index)
    annual = cov.annualise()
    S = annual.matrix.reindex(index=names, columns=names).to_numpy()
    w = weights.to_numpy()
    port_vol = float(np.sqrt(w @ S @ w))
    if port_vol <= 0:
        raise EstimationError("portfolio volatility is not positive")
    return float((w @ np.sqrt(np.diag(S))) / port_vol)


def beta_and_tracking_error(portfolio: pd.Series, benchmark: pd.Series,
                            periods_per_year: float = 252.0) -> tuple[float, float]:
    """Beta against the benchmark and annualised tracking error, on aligned dates only."""
    df = pd.concat([pd.Series(portfolio), pd.Series(benchmark)], axis=1, keys=["p", "b"]).dropna()
    if len(df) < 30:
        raise EstimationError(f"beta needs at least 30 aligned observations, got {len(df)}")
    var_b = float(df["b"].var(ddof=1))
    if var_b <= 0:
        raise EstimationError("the benchmark has no variance over this window")
    beta = float(df["p"].cov(df["b"]) / var_b)
    te = float((df["p"] - df["b"]).std(ddof=1) * np.sqrt(periods_per_year))
    return beta, te


# ------------------------------------------------------------- the report ---
@dataclass(frozen=True)
class RiskReport:
    horizon_days: int
    expected_shortfall: dict[float, float]     # confidence -> loss
    value_at_risk: dict[float, float]
    volatility: float                          # of the horizon distribution
    annual_volatility: float | None
    observations: int
    effective_observations: float
    provenance: Provenance
    warnings: tuple[str, ...] = ()
    contributions: pd.DataFrame | None = None

    @property
    def headline(self) -> str:
        es = self.expected_shortfall.get(0.975)
        h = self.horizon_days
        period = "day" if h == 1 else f"{h} days"
        return (f"Over {period}, the worst 2.5% of outcomes average a loss of {es:.1%}"
                + (f" ({self.warnings[0]})" if self.warnings else "."))


def risk_report(sample: HorizonSample, *, weights: pd.Series | None = None,
                cov: CovarianceEstimate | None = None,
                confidences: tuple[float, ...] = (0.95, 0.975, 0.99),
                periods_per_year: float = 252.0) -> RiskReport:
    """Every risk figure for one horizon, with the caveats attached to it."""
    r = sample.returns.to_numpy()
    es = {c: expected_shortfall(sample, c) for c in confidences}
    var = {c: value_at_risk(sample, c) for c in confidences}
    warnings = []
    for c in confidences:
        ok, note = tail_reliability(sample, c)
        if not ok:
            warnings.append(note)
    vol = float(np.std(r, ddof=1))
    annual = float(vol * np.sqrt(periods_per_year / sample.horizon_days)) if sample.horizon_days else None
    contributions = None
    if weights is not None and cov is not None:
        contributions = volatility_contributions(weights, cov)
    prov = Provenance(
        method=f"expected shortfall and VaR from the {sample.horizon_days}-day distribution "
               f"(no sqrt-time scaling)",
        window=f"{len(r)} observations, {sample.effective_observations:.0f} effective",
        as_of=sample.provenance.as_of, source=sample.provenance.method,
        degradations=tuple(warnings))
    return RiskReport(horizon_days=sample.horizon_days, expected_shortfall=es, value_at_risk=var,
                      volatility=vol, annual_volatility=annual, observations=len(r),
                      effective_observations=sample.effective_observations, provenance=prov,
                      warnings=tuple(warnings), contributions=contributions)


# ------------------------------------------------------------------ helpers ---
def _values(sample) -> np.ndarray:
    r = sample.returns if isinstance(sample, HorizonSample) else pd.Series(sample)
    a = np.asarray(pd.Series(r).dropna(), dtype=float)
    if a.size < 2:
        raise EstimationError("need at least two observations")
    return a


def _check_confidence(c: float) -> None:
    if not 0.5 < c < 1.0:
        raise EstimationError(f"confidence must be between 0.5 and 1, got {c}")

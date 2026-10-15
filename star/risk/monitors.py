"""
Standing monitors: the things worth watching between rebalances.

Selected by the owner from `docs/INDUSTRY_METRICS_ASSESSMENT.md`.

* **Correlation monitor.** Diversification is not a property of how many names you
  hold; it is a property of how they move together, and that changes. Average
  pairwise correlation rising through its own historical range is the earliest
  warning that a portfolio which looks spread out has quietly become one bet.
* **Diversification ratio** (Choueifaty & Coignard 2008) - the same idea as a
  single number for the portfolio actually held, tracked over time.
* **Gold sleeve.** What a 5/10/15% allocation to gold would have done to risk and
  return. Reported as a diagnostic, never as a recommendation: it is here to
  answer "would this have helped?" with a number instead of an opinion.

Each monitor compares today against the portfolio's own history rather than a
fixed threshold, because "0.6 is high" is meaningless without knowing what is
normal for these holdings.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..core.errors import EstimationError
from ..core.types import Provenance
from .metrics import expected_shortfall


def average_pairwise_correlation(returns: pd.DataFrame, window: int = 60) -> pd.Series:
    """Rolling mean of the off-diagonal correlations. One number for "how together"."""
    R = returns.dropna(how="all")
    n = R.shape[1]
    if n < 2:
        raise EstimationError("correlation needs at least two holdings")
    if len(R) < window + 5:
        raise EstimationError(f"need more than {window} sessions to roll a {window}-day correlation")
    out = {}
    for end in range(window, len(R) + 1):
        block = R.iloc[end - window:end].dropna(axis=1, how="any")
        if block.shape[1] < 2:
            continue
        c = np.corrcoef(block.to_numpy(), rowvar=False)
        iu = np.triu_indices_from(c, k=1)
        out[R.index[end - 1]] = float(np.nanmean(c[iu]))
    return pd.Series(out, name=f"avg_corr_{window}d")


@dataclass(frozen=True)
class CorrelationMonitor:
    current: float
    percentile: float              # against this portfolio's own history
    median: float
    change_3m: float
    series: pd.Series
    provenance: Provenance

    @property
    def headline(self) -> str:
        direction = "rising" if self.change_3m > 0.02 else ("falling" if self.change_3m < -0.02
                                                            else "stable")
        base = (f"Holdings are moving together {self.current:.2f} on average, "
                f"{self.percentile:.0f}th percentile of their own history and {direction}")
        if self.percentile >= 80:
            return base + " - diversification is weaker than usual, so shocks will hit more of the book."
        if self.percentile <= 20:
            return base + " - unusually independent, so diversification is doing more work than normal."
        return base + "."


def correlation_monitor(returns: pd.DataFrame, *, window: int = 60) -> CorrelationMonitor:
    s = average_pairwise_correlation(returns, window)
    if len(s) < 30:
        raise EstimationError(f"only {len(s)} rolling correlation points; need at least 30 to say "
                              f"what is normal for this portfolio")
    current = float(s.iloc[-1])
    lookback = min(63, len(s) - 1)
    return CorrelationMonitor(
        current=current,
        percentile=float((s.to_numpy() <= current).mean() * 100),
        median=float(s.median()),
        change_3m=float(current - s.iloc[-1 - lookback]),
        series=s,
        provenance=Provenance(method=f"{window}-session average pairwise correlation",
                              window=f"{len(s)} rolling observations",
                              as_of=pd.Timestamp(s.index[-1]), source="daily returns"))


def diversification_ratio_series(returns: pd.DataFrame, weights: pd.Series,
                                 window: int = 60) -> pd.Series:
    """
    Weighted average volatility over portfolio volatility, rolled through time.

    Falling means the holdings are converging - the same warning as the
    correlation monitor, but weighted by what is actually held.
    """
    names = [t for t in weights.index if t in returns.columns]
    if len(names) < 2:
        raise EstimationError("diversification needs at least two holdings with returns")
    w = weights[names].to_numpy()
    R = returns[names]
    out = {}
    for end in range(window, len(R) + 1):
        block = R.iloc[end - window:end]
        if block.isna().to_numpy().any():
            continue
        S = np.cov(block.to_numpy(), rowvar=False)
        pv = float(w @ S @ w)
        if pv <= 0:
            continue
        out[R.index[end - 1]] = float((w @ np.sqrt(np.diag(S))) / np.sqrt(pv))
    if not out:
        raise EstimationError("no complete window was available for the diversification ratio")
    return pd.Series(out, name=f"diversification_{window}d")


@dataclass(frozen=True)
class SleeveImpact:
    allocation: float
    volatility: float
    volatility_change: float
    expected_shortfall: float
    shortfall_change: float
    mean_return: float
    return_change: float


def gold_sleeve(portfolio_returns: pd.Series, gold_returns: pd.Series, *,
                allocations: tuple[float, ...] = (0.05, 0.10, 0.15),
                confidence: float = 0.975, periods_per_year: float = 252.0) -> pd.DataFrame:
    """
    What funding a gold sleeve out of the equity book would have done.

    A diagnostic: it answers "would this have helped, on this history?" and
    nothing more. Gold's diversification benefit is regime-dependent and this is
    one sample, so the table is shown with the period it covers.
    """
    df = pd.concat([pd.Series(portfolio_returns), pd.Series(gold_returns)], axis=1,
                   keys=["p", "g"]).dropna()
    if len(df) < 250:
        raise EstimationError(f"a sleeve comparison needs at least 250 aligned sessions, got {len(df)}")
    if any(a < 0 or a >= 1 for a in allocations):
        raise EstimationError("allocations must be between 0 and 1")
    base_vol = float(df["p"].std(ddof=1) * np.sqrt(periods_per_year))
    base_es = expected_shortfall(df["p"], confidence)
    base_ret = float(df["p"].mean() * periods_per_year)
    rows = [SleeveImpact(0.0, base_vol, 0.0, base_es, 0.0, base_ret, 0.0)]
    for a in allocations:
        blend = (1 - a) * df["p"] + a * df["g"]
        vol = float(blend.std(ddof=1) * np.sqrt(periods_per_year))
        es = expected_shortfall(blend, confidence)
        ret = float(blend.mean() * periods_per_year)
        rows.append(SleeveImpact(a, vol, vol - base_vol, es, es - base_es, ret, ret - base_ret))
    return pd.DataFrame([r.__dict__ for r in rows]).set_index("allocation")

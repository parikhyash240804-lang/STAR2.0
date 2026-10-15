"""
Named episodes, replayed with the real returns of the real holdings.

A stress test is only credible if the reader can check it: "Financials fell 41%
between January 2008 and March 2009" is auditable, "a 2008-style shock costs 38%"
is not. So each episode is a dated window, each holding contributes its own
actual return over that window, and anything that could not be measured is
disclosed rather than quietly filled in.

Where a stock was not listed at the time - which is most of them for 2008 - its
return is **propagated from the market move through its beta**, and the result
says how much of the portfolio was handled that way. That is an estimate and is
labelled as one. The legacy applied a flat 1.3x multiplier to every holding and
then clipped losses at -80%, which meant that for high-beta names the clip, not
the data, decided the answer (M6-04).

Coverage is the number to read first. A 2008 replay where 80% of the book is
proxied is a statement about beta, not about 2008.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..core.errors import DataError
from ..core.types import Provenance


@dataclass(frozen=True)
class Episode:
    """A dated window that a reader can look up."""

    key: str
    name: str
    start: str
    end: str
    what_happened: str


EPISODES: tuple[Episode, ...] = (
    Episode("gfc_2008", "Global financial crisis", "2008-01-08", "2009-03-09",
            "Nifty fell around 60% peak to trough as global credit markets froze."),
    Episode("taper_2013", "Taper tantrum", "2013-05-22", "2013-08-28",
            "The rupee hit a record low and yields spiked after the Fed signalled tapering."),
    Episode("demon_2016", "Demonetisation", "2016-11-08", "2016-12-26",
            "86% of currency in circulation was withdrawn overnight; consumption names fell hardest."),
    Episode("covid_2020", "COVID-19 crash", "2020-02-19", "2020-03-23",
            "Nifty fell roughly 38% in 23 sessions - the fastest drawdown on record."),
    Episode("rates_2022", "2022 rate shock and Ukraine", "2022-01-17", "2022-06-17",
            "Inflation, the invasion of Ukraine and the fastest global tightening cycle in decades."),
    Episode("election_2024", "2024 election result", "2024-06-03", "2024-06-04",
            "Nifty fell around 6% in one session when the result defied exit polls."),
)

EPISODE_BY_KEY = {e.key: e for e in EPISODES}


@dataclass(frozen=True)
class StressResult:
    episode: Episode
    portfolio_return: float
    benchmark_return: float | None
    per_holding: pd.DataFrame          # weight, return, contribution, source
    coverage: float                    # share of weight with real data
    proxied: tuple[str, ...]
    missing: tuple[str, ...]
    provenance: Provenance

    @property
    def headline(self) -> str:
        loss = f"{self.portfolio_return:+.1%}"
        base = f"{self.episode.name} ({self.episode.start} to {self.episode.end}): {loss}"
        if self.coverage < 0.999:
            base += f", but only {self.coverage:.0%} of the book has real data for that window"
        return base + "."

    def worst(self, n: int = 5) -> pd.DataFrame:
        return self.per_holding.nsmallest(n, "contribution")


def _window_return(series: pd.Series, start, end, min_coverage: float) -> tuple[float | None, float]:
    """Return over the window and the share of the window the data actually covers."""
    s = series.dropna().loc[str(start):str(end)]
    if len(s) < 2:
        return None, 0.0
    span = (pd.Timestamp(end) - pd.Timestamp(start)).days
    covered = (s.index[-1] - s.index[0]).days
    share = 1.0 if span <= 0 else min(covered / span, 1.0)
    if share < min_coverage:
        return None, share
    return float(s.iloc[-1] / s.iloc[0] - 1.0), share


def replay(episode: Episode | str, close: pd.DataFrame, weights: pd.Series, *,
           benchmark: pd.Series | None = None, betas: pd.Series | None = None,
           min_window_coverage: float = 0.80) -> StressResult:
    """
    Replay one episode over the given holdings.

    A holding contributes its own return when it has data covering at least
    `min_window_coverage` of the window. Otherwise it is proxied through its beta
    to the benchmark, if both are available, and listed as proxied. If neither is
    possible the name is listed as missing and excluded - never silently treated
    as flat, which would understate the loss.
    """
    ep = EPISODE_BY_KEY[episode] if isinstance(episode, str) else episode
    w = pd.Series(weights, dtype=float)
    if w.empty:
        raise DataError("cannot stress an empty portfolio")

    bench_return = None
    if benchmark is not None:
        bench_return, bench_share = _window_return(pd.Series(benchmark), ep.start, ep.end, 0.5)
        if bench_return is None:
            bench_return = None

    rows, proxied, missing = [], [], []
    for ticker, weight in w.items():
        r = share = None
        if ticker in close.columns:
            r, share = _window_return(close[ticker], ep.start, ep.end, min_window_coverage)
        if r is not None:
            rows.append((ticker, weight, r, "actual"))
        elif bench_return is not None and betas is not None and ticker in betas.index \
                and np.isfinite(betas.get(ticker, np.nan)):
            # propagate the market move through beta; a stock cannot lose more than everything
            est = max(float(betas[ticker]) * bench_return, -1.0)
            rows.append((ticker, weight, est, "beta-proxied"))
            proxied.append(ticker)
        else:
            missing.append(ticker)

    if not rows:
        raise DataError(f"no holding has data or a proxy for {ep.name} ({ep.start} to {ep.end})")
    df = pd.DataFrame(rows, columns=["ticker", "weight", "return", "source"]).set_index("ticker")
    df["contribution"] = df["weight"] * df["return"]
    invested = float(df["weight"].sum())
    portfolio = float(df["contribution"].sum() / invested) if invested > 0 else float("nan")
    coverage = float(df.loc[df["source"] == "actual", "weight"].sum() / w.sum())

    degradations = []
    if proxied:
        degradations.append(f"{len(proxied)} holding(s) had no price history in this window and were "
                            f"estimated from the market move through their beta")
    if missing:
        degradations.append(f"{len(missing)} holding(s) could not be covered at all and are excluded: "
                            f"{missing[:5]}")
    prov = Provenance(
        method=f"historical replay of {ep.name}",
        window=f"{ep.start} to {ep.end}; {coverage:.0%} of weight from actual prices",
        source="realised prices over the episode",
        degradations=tuple(degradations))
    return StressResult(episode=ep, portfolio_return=portfolio, benchmark_return=bench_return,
                        per_holding=df, coverage=coverage, proxied=tuple(proxied),
                        missing=tuple(missing), provenance=prov)


def replay_all(close: pd.DataFrame, weights: pd.Series, *, benchmark: pd.Series | None = None,
               betas: pd.Series | None = None, episodes=EPISODES,
               min_window_coverage: float = 0.80) -> pd.DataFrame:
    """Every episode as one table, worst first, with coverage beside each figure."""
    rows = []
    for ep in episodes:
        try:
            res = replay(ep, close, weights, benchmark=benchmark, betas=betas,
                         min_window_coverage=min_window_coverage)
        except DataError:
            rows.append({"episode": ep.name, "start": ep.start, "end": ep.end,
                         "portfolio_return": np.nan, "benchmark_return": np.nan,
                         "coverage": 0.0, "proxied": 0, "note": "no data for this window"})
            continue
        rows.append({"episode": ep.name, "start": ep.start, "end": ep.end,
                     "portfolio_return": res.portfolio_return,
                     "benchmark_return": res.benchmark_return, "coverage": res.coverage,
                     "proxied": len(res.proxied),
                     "note": "; ".join(res.provenance.degradations) or ""})
    return pd.DataFrame(rows).sort_values("portfolio_return", na_position="last").reset_index(drop=True)

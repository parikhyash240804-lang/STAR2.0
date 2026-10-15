"""
The trading calendar, derived from observed sessions.

The legacy engine hard-coded `TRADING_DAYS = 252` and used it for three different
jobs: annualising volatility, de-annualising the risk-free rate, and setting the
simulation length. It is not configurable, it is not measured, and the NSE does
not actually trade 252 days in most years (2022: 248, 2023: 247, 2024: 250).

Here the annualisation factor is *measured* from the sessions the data actually
contains, and the forecast length is a separate concept entirely (`HorizonGrid`).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .errors import DataError


@dataclass(frozen=True, slots=True)
class TradingCalendar:
    """Observed trading sessions for a market, plus what follows from them."""

    sessions: pd.DatetimeIndex

    def __post_init__(self) -> None:
        if len(self.sessions) < 2:
            raise DataError("a trading calendar needs at least two sessions")
        if not self.sessions.is_monotonic_increasing:
            raise DataError("trading sessions must be sorted")
        if self.sessions.has_duplicates:
            raise DataError("trading calendar contains duplicate sessions")

    @classmethod
    def from_sessions(cls, dates) -> "TradingCalendar":
        idx = pd.DatetimeIndex(pd.to_datetime(pd.Index(dates))).sort_values().unique()
        return cls(sessions=pd.DatetimeIndex(idx))

    # ---- the annualisation factor, measured rather than assumed ----
    @property
    def periods_per_year(self) -> float:
        """
        Median sessions per complete calendar year, falling back to a rate when
        the history is shorter than one year.

        The median over complete years is robust to a partial first or last year,
        which a naive "sessions / years" ratio is not.
        """
        years = self.sessions.year
        counts = pd.Series(1, index=self.sessions).groupby(years).sum()
        if len(counts) >= 3:
            complete = counts.iloc[1:-1]          # drop partial first/last years
            if len(complete) >= 1:
                return float(np.median(complete.values))
        span_days = (self.sessions[-1] - self.sessions[0]).days
        if span_days < 90:
            raise DataError(
                f"only {span_days} calendar days of sessions: too short to measure sessions "
                f"per year reliably (need at least 90)")
        # n sessions span n-1 intervals; dividing n by the span overstates the rate
        return float((len(self.sessions) - 1) / (span_days / 365.25))

    @property
    def first(self) -> pd.Timestamp:
        return self.sessions[0]

    @property
    def last(self) -> pd.Timestamp:
        return self.sessions[-1]

    def __len__(self) -> int:
        return len(self.sessions)

    # ---- navigation ----
    def nearest(self, date, *, side: str = "nearest") -> pd.Timestamp:
        """
        Snap a date to a session.

        `side='previous'` is the only safe choice for a rebalance date: snapping
        forward uses a session that had not happened yet.
        """
        ts = pd.Timestamp(date)
        if side == "previous":
            pos = self.sessions.searchsorted(ts, side="right") - 1
            if pos < 0:
                raise DataError(f"no trading session on or before {ts.date()}")
            return self.sessions[pos]
        if side == "next":
            pos = self.sessions.searchsorted(ts, side="left")
            if pos >= len(self.sessions):
                raise DataError(f"no trading session on or after {ts.date()}")
            return self.sessions[pos]
        pos = int(self.sessions.get_indexer([ts], method="nearest")[0])
        return self.sessions[pos]

    def sessions_between(self, start, end, *, inclusive: str = "both") -> pd.DatetimeIndex:
        s, e = pd.Timestamp(start), pd.Timestamp(end)
        left = self.sessions >= s if inclusive in ("both", "left") else self.sessions > s
        right = self.sessions <= e if inclusive in ("both", "right") else self.sessions < e
        return self.sessions[left & right]

    def count_between(self, start, end, *, inclusive: str = "both") -> int:
        return len(self.sessions_between(start, end, inclusive=inclusive))

    def shift(self, date, sessions: int) -> pd.Timestamp:
        """
        Move `sessions` trading days from the session in effect on `date`.

        A non-trading date (a weekend, a holiday) always anchors on the last
        session on or before it - the one whose close is actually known - in both
        directions, so shifting back from a Saturday never snaps forward first.
        """
        anchor = self.nearest(date, side="previous")
        pos = int(self.sessions.searchsorted(anchor, side="left")) + sessions
        if not 0 <= pos < len(self.sessions):
            raise DataError(
                f"{sessions:+d} sessions from {pd.Timestamp(date).date()} falls outside "
                f"the calendar ({self.first.date()}..{self.last.date()})")
        return self.sessions[pos]

    def horizon_end(self, start, horizon_days: int) -> pd.Timestamp:
        """The session a `horizon_days` forecast from `start` would end on."""
        return self.shift(start, horizon_days)

    def rebalance_dates(self, start, end, freq: str) -> pd.DatetimeIndex:
        """
        Rebalance sessions between two dates.

        Each calendar date is snapped *backwards* to a real session, because a
        decision can only use information from a session that has closed. The
        legacy snapped to the nearest session in either direction, which let a
        rebalance land on a day that had not happened yet.
        """
        wanted = pd.date_range(start=pd.Timestamp(start), end=pd.Timestamp(end), freq=freq)
        snapped = []
        for d in wanted:
            try:
                snapped.append(self.nearest(d, side="previous"))
            except DataError:
                continue
        return pd.DatetimeIndex(pd.Index(snapped).unique())

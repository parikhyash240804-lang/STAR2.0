"""
Data provider protocols, and the look-ahead canary.

Legacy defect M8-02 has two halves. The look-ahead half — using data that did not
exist at the simulated decision time — is an engineering problem, and this module
solves it structurally: while the guard is armed, *any* request for data past the
cutoff raises `LookAheadError`. A backtest must run to completion with the guard
armed at every rebalance, which is asserted by a test. Discipline is not relied
upon, because the legacy had discipline and still leaked.

The survivorship half cannot be fixed with free Yahoo data (there is no delisted
history and no index-membership history), so `UniverseProvider.members()` returns
`None` for Yahoo and the report says plainly what the backtest can and cannot
answer. `FundamentalsProvider.is_point_in_time` exists for the same reason: a
vendor with real point-in-time data can be plugged in without touching callers.
"""
from __future__ import annotations

import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator, Protocol, Sequence, runtime_checkable

import pandas as pd

from ..core.errors import LookAheadError


@dataclass
class AsOfGuard:
    """
    Refuses data requests that reach past a cutoff.

    Thread-local so a GUI thread and a worker cannot disarm each other. Every
    violation is counted even when it raises, so a test can assert zero.
    """

    _state: threading.local = field(default_factory=threading.local, repr=False)
    violations: list[str] = field(default_factory=list)

    @property
    def cutoff(self) -> pd.Timestamp | None:
        return getattr(self._state, "cutoff", None)

    @contextmanager
    def armed(self, cutoff) -> Iterator[None]:
        """Arm the guard at `cutoff` for the duration of the block."""
        previous = self.cutoff
        new = pd.Timestamp(cutoff)
        # nested arming can only tighten the cutoff, never loosen it
        self._state.cutoff = new if previous is None else min(previous, new)
        try:
            yield
        finally:
            self._state.cutoff = previous

    def check(self, requested_end, *, what: str) -> None:
        cutoff = self.cutoff
        if cutoff is None:
            return
        if requested_end is None:
            # an open-ended request while armed means "up to whatever exists", which
            # is exactly the future a backtest must not see
            msg = f"{what}: open-ended data request while the as-of guard is armed at {cutoff.date()}"
            self.violations.append(msg)
            raise LookAheadError(msg)
        end = pd.Timestamp(requested_end)
        if end.tzinfo is not None:
            end = end.tz_localize(None)
        if end > cutoff:
            msg = (f"{what}: requested data to {end.date()} but the as-of cutoff is "
                   f"{cutoff.date()} - this would use information that did not exist yet")
            self.violations.append(msg)
            raise LookAheadError(msg)

    def clamp(self, frame: pd.DataFrame | pd.Series, *, what: str):
        """
        Defence in depth: drop any row dated after the cutoff.

        A provider may legitimately return more than was asked for (a cache hit
        covering a wider range). Clamping keeps that from leaking, and records it.
        """
        cutoff = self.cutoff
        if cutoff is None or not isinstance(frame.index, pd.DatetimeIndex):
            return frame
        # compare wall-clock dates: converting an IST midnight to UTC would move it
        # to the previous day, so drop the zone rather than convert it
        idx = frame.index.tz_localize(None) if frame.index.tz is not None else frame.index
        late = idx > cutoff
        if late.any():
            self.violations.append(
                f"{what}: {int(late.sum())} row(s) after {cutoff.date()} were clamped")
            return frame.loc[~late]
        return frame


#: the process-wide guard; providers consult it, the backtest arms it
GUARD = AsOfGuard()


@dataclass(frozen=True, slots=True)
class PricePanel:
    """
    Daily OHLCV for a universe, plus where it came from.

    `close` is split- and dividend-adjusted (total return). `close_raw` is the
    unadjusted close, needed for share counts and for the price-only benchmark
    reconciliation. Keeping both is what lets the report state honestly whether a
    comparison is total-return or price-only - the legacy mixed the two (M0-02).
    """

    close: pd.DataFrame
    close_raw: pd.DataFrame | None
    high: pd.DataFrame | None
    low: pd.DataFrame | None
    open_: pd.DataFrame | None
    volume: pd.DataFrame | None
    source: str
    as_of: pd.Timestamp

    @property
    def tickers(self) -> list[str]:
        return list(self.close.columns)

    @property
    def sessions(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(self.close.index)

    def log_returns(self) -> pd.DataFrame:
        """Daily log returns. No forward-filling, no row dropping - see data.clean."""
        import numpy as np
        return pd.DataFrame(np.log(self.close / self.close.shift(1)),
                            index=self.close.index, columns=self.close.columns).iloc[1:]

    def has_ohlc(self) -> bool:
        return all(x is not None for x in (self.high, self.low, self.open_))


@runtime_checkable
class PriceProvider(Protocol):
    """Daily prices for a set of tickers."""

    name: str

    def prices(self, tickers: Sequence[str], start, end, *, adjusted: bool = True) -> PricePanel:
        ...


@runtime_checkable
class FundamentalsProvider(Protocol):
    """
    Fundamental fields as of a date.

    `is_point_in_time` is the honesty flag. Yahoo returns *restated current*
    values, so a backtest using them is biased no matter what lag is applied;
    STAR2.0 therefore defaults to price-only signals in backtests and measures
    the fundamentals-mode difference as an upper bound on the attainable edge.
    """

    name: str
    is_point_in_time: bool

    def snapshot(self, tickers: Sequence[str], field: str, as_of) -> pd.Series:
        ...


@runtime_checkable
class UniverseProvider(Protocol):
    """Index membership as of a date. Returns None when history is unavailable."""

    name: str

    def members(self, index: str, as_of) -> list[str] | None:
        ...


@runtime_checkable
class RiskFreeProvider(Protocol):
    """
    The risk-free curve as a dated series, not one constant.

    The legacy used a single 6.5% for both today's discount rate and five years of
    history (M0-11, M2-15), which is simply wrong for 2019-2022.
    """

    name: str

    def series(self, start, end) -> pd.Series:
        """Annual simple rates, indexed by date."""
        ...

    def rate(self, date) -> float:
        ...

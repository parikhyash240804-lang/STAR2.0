"""
A point-in-time view of market data.

`DataView` is what the strategy sees. It holds data and an `as_of` date, and every
accessor refuses to return anything dated after it. The same object serves the
live run (as_of = today) and each backtest rebalance (as_of = that date), which is
what lets one strategy implementation be used in both places — closing legacy
defect M8-01, where the backtest quietly ran a different model.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..core.calendar import TradingCalendar
from ..core.errors import DataError, LookAheadError
from .provider import GUARD, PricePanel


@dataclass(frozen=True, slots=True)
class DataView:
    """Everything a strategy may look at, truncated at `as_of`."""

    panel: PricePanel
    as_of: pd.Timestamp
    calendar: TradingCalendar
    benchmark_close: pd.Series | None = None
    risk_free: pd.Series | None = None

    @classmethod
    def build(cls, panel: PricePanel, as_of, *, benchmark_close: pd.Series | None = None,
              risk_free: pd.Series | None = None) -> "DataView":
        """
        Build a view that physically contains nothing dated after `as_of`.

        Truncating in the accessors alone is not enough: `view.panel.close` would
        still hand a strategy the future. So every stored frame is cut here, once.
        """
        from dataclasses import replace
        ts = pd.Timestamp(as_of)
        sessions = panel.sessions[panel.sessions <= ts]
        if len(sessions) < 2:
            raise DataError(f"no usable price history on or before {ts.date()}")

        def cut(frame):
            return None if frame is None else frame.loc[frame.index <= ts]

        past = replace(panel, close=cut(panel.close), close_raw=cut(panel.close_raw),
                       high=cut(panel.high), low=cut(panel.low), open_=cut(panel.open_),
                       volume=cut(panel.volume), as_of=min(pd.Timestamp(panel.as_of), ts))
        return cls(panel=past, as_of=ts,
                   calendar=TradingCalendar.from_sessions(sessions),
                   benchmark_close=cut(benchmark_close), risk_free=cut(risk_free))

    # ---- accessors: all truncated, all explicit about the window ----
    def _truncate(self, frame, what: str):
        if frame is None:
            return None
        out = frame.loc[frame.index <= self.as_of]
        if out.empty:
            raise DataError(f"{what}: nothing on or before {self.as_of.date()}")
        return out

    @staticmethod
    def _last(frame, n: int | None):
        if n is None:
            return frame
        if int(n) < 1:
            raise DataError(f"lookback_sessions must be at least 1, got {n}")
        return frame.iloc[-int(n):]

    def closes(self, *, lookback_sessions: int | None = None) -> pd.DataFrame:
        return self._last(self._truncate(self.panel.close, "closes"), lookback_sessions)

    def log_returns(self, *, lookback_sessions: int | None = None) -> pd.DataFrame:
        return self._last(self._truncate(self.panel.log_returns(), "log returns"), lookback_sessions)

    def ohlc(self, *, lookback_sessions: int | None = None) -> dict[str, pd.DataFrame]:
        """Open/high/low/close, for range-based volatility (Yang-Zhang)."""
        if not self.panel.has_ohlc():
            raise DataError("this panel has no OHLC data; range-based volatility needs it")
        out = {k: self._truncate(v, k) for k, v in (
            ("open", self.panel.open_), ("high", self.panel.high),
            ("low", self.panel.low), ("close", self.panel.close))}
        return {k: self._last(v, lookback_sessions) for k, v in out.items()}

    def benchmark_log_returns(self, *, lookback_sessions: int | None = None) -> pd.Series:
        import numpy as np
        if self.benchmark_close is None:
            raise DataError("no benchmark series in this view")
        px = self._truncate(self.benchmark_close, "benchmark")
        r = pd.Series(np.log(px / px.shift(1)), index=px.index).iloc[1:]
        return self._last(r, lookback_sessions)

    def risk_free_rate(self) -> float:
        """The annual simple risk-free rate in effect at `as_of`."""
        if self.risk_free is None:
            raise DataError("no risk-free series in this view")
        s = self._truncate(self.risk_free, "risk-free")
        return float(s.iloc[-1])

    def risk_free_series(self) -> pd.Series:
        return self._truncate(self.risk_free, "risk-free")

    # ---- the canary ----
    def assert_no_future_data(self) -> None:
        """
        Verify nothing in this view postdates `as_of`.

        Cheap enough to call in tests and after every backtest window.
        """
        stored = (("close", self.panel.close), ("close_raw", self.panel.close_raw),
                  ("high", self.panel.high), ("low", self.panel.low), ("open", self.panel.open_),
                  ("volume", self.panel.volume), ("benchmark", self.benchmark_close),
                  ("risk_free", self.risk_free))
        for name, frame in stored:
            if frame is None:
                continue
            idx = frame.index
            if isinstance(idx, pd.DatetimeIndex) and (idx > self.as_of).any():
                raise LookAheadError(f"view holds {name} data after {self.as_of.date()}")
        if self.panel.as_of > self.as_of:
            raise LookAheadError(f"panel is stamped {self.panel.as_of.date()}, after the view's "
                                 f"as-of {self.as_of.date()}")

    def armed(self):
        """Arm the process-wide provider guard at this view's as-of date."""
        return GUARD.armed(self.as_of)

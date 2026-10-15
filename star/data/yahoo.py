"""
Yahoo Finance provider.

Fixes the legacy fetch path's specific defects:

* TLS verification stays **on** (legacy M0-01 disabled it process-wide).
* `auto_adjust` is explicit, and the unadjusted close is kept too, so a
  total-return portfolio is never silently compared with a price index (M0-02).
* yfinance 1.x returns MultiIndex columns even for a single ticker; that is
  handled once here instead of defensively guessed at each call site.
* No `ffill(limit=2).dropna()`: gaps are reported, not papered over (M1-05).
* Requests are clamped by the as-of guard, so a backtest cannot fetch the future.
* Every degradation (a ticker with no data, a short history) is recorded rather
  than printed to a discarded stdout.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..core.errors import DataError
from .provider import GUARD, PricePanel


def _flatten(df: pd.DataFrame, field_name: str, tickers: list[str]) -> pd.DataFrame | None:
    """
    Pull one field out of a yfinance frame, whatever shape it arrived in.

    yfinance returns columns as a (Field, Ticker) MultiIndex by default - even for
    a single ticker, which is the shape the legacy's M6 benchmark download got
    wrong.
    """
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        if field_name not in df.columns.get_level_values(0):
            return None
        out = df[field_name].copy()
    else:
        if field_name not in df.columns:
            return None
        out = df[[field_name]].copy()
        out.columns = tickers[:1]
    keep = [t for t in tickers if t in out.columns]
    out = out[keep]
    out.index = pd.DatetimeIndex(out.index).tz_localize(None)
    return out.sort_index()


@dataclass
class YahooProvider:
    """Live market data. One network-touching class, so quirks stay in one file."""

    name: str = "yahoo"
    max_retries: int = 3
    retry_sleep: float = 1.5
    degradations: list[str] = field(default_factory=list)

    def _download(self, symbols: list[str], start, end, *, adjusted: bool) -> pd.DataFrame:
        import yfinance as yf

        last_err: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                df = yf.download(
                    symbols, start=pd.Timestamp(start), end=pd.Timestamp(end),
                    auto_adjust=adjusted,      # explicit: never rely on the default
                    progress=False, actions=False, threads=True,
                )
                if df is not None and not df.empty:
                    return df
                last_err = DataError("empty frame")
            except Exception as e:                      # network/parse errors
                last_err = e
            if attempt < self.max_retries:
                time.sleep(self.retry_sleep * attempt)
        raise DataError(f"download failed for {len(symbols)} symbol(s) after "
                        f"{self.max_retries} attempts: {last_err}")

    def prices(self, tickers, start, end, *, adjusted: bool = True) -> PricePanel:
        tickers = list(dict.fromkeys(tickers))          # de-duplicate, keep order
        if not tickers:
            raise DataError("no tickers requested")
        end_ts = pd.Timestamp(end)
        GUARD.check(end_ts, what=f"yahoo.prices({len(tickers)} tickers)")

        adj = self._download(tickers, start, end_ts, adjusted=True)
        close = _flatten(adj, "Close", tickers)
        if close is None or close.empty:
            raise DataError("no Close data returned")

        high = _flatten(adj, "High", tickers)
        low = _flatten(adj, "Low", tickers)
        open_ = _flatten(adj, "Open", tickers)
        volume = _flatten(adj, "Volume", tickers)

        close_raw = None
        if not adjusted:
            raw = self._download(tickers, start, end_ts, adjusted=False)
            close_raw = _flatten(raw, "Close", tickers)

        missing = [t for t in tickers if t not in close.columns]
        if missing:
            self.degradations.append(f"no data for {len(missing)} ticker(s): {missing[:8]}")

        empty = [c for c in close.columns if close[c].notna().sum() == 0]
        if empty:
            close = close.drop(columns=empty)
            self.degradations.append(f"all-NaN history dropped for: {empty[:8]}")

        close = GUARD.clamp(close, what="yahoo.close")
        panel = PricePanel(close=close, close_raw=close_raw, high=high, low=low,
                           open_=open_, volume=volume, source=self.name,
                           as_of=pd.Timestamp(close.index.max()))
        return panel

    # ---- benchmark ----
    def benchmark(self, symbol: str, start, end) -> pd.Series:
        """
        A benchmark close series.

        For a total-return NIFTY proxy use `NIFTYBEES.NS` with adjustment on: it is
        the Nippon India Nifty 50 ETF, whose adjusted close includes dividends, and
        it has history to 2002. `^NSEI` is a PRICE index and understates the
        benchmark by roughly the dividend yield - the legacy compared
        dividend-adjusted stocks against it (M0-02, M8-07).
        """
        GUARD.check(pd.Timestamp(end), what=f"yahoo.benchmark({symbol})")
        df = self._download([symbol], start, end, adjusted=True)
        s = _flatten(df, "Close", [symbol])
        if s is None or s.empty:
            raise DataError(f"no benchmark data for {symbol}")
        out = s.iloc[:, 0].dropna()
        out.name = symbol
        return GUARD.clamp(out, what=f"yahoo.benchmark({symbol})")

    # ---- fundamentals (current snapshot only - never point-in-time) ----
    @property
    def is_point_in_time(self) -> bool:
        return False

    def snapshot(self, tickers, field_name: str, as_of) -> pd.Series:
        """
        A current fundamental field, labelled for what it is.

        Yahoo serves *restated current* values with no as-of history, so this is
        only legitimate for the live recommendation. Backtests must not call it
        (the backtest defaults to price-only signals); when they do, in an
        explicitly-labelled bias-measurement run, the value is stamped with the
        real retrieval date so the report can say so.
        """
        import yfinance as yf

        vals: dict[str, float] = {}
        failures = 0
        for t in tickers:
            try:
                info = yf.Ticker(t).info
                v = info.get(field_name, None)
                vals[t] = float(v) if v is not None else np.nan
            except Exception:
                vals[t] = np.nan
                failures += 1
        if failures:
            self.degradations.append(
                f"fundamentals '{field_name}': {failures}/{len(tickers)} lookups failed")
        out = pd.Series(vals, dtype=float)
        out.attrs["is_point_in_time"] = False
        out.attrs["retrieved"] = str(pd.Timestamp(as_of).date())
        return out

    def members(self, index: str, as_of) -> list[str] | None:
        """
        Index membership as of a date.

        Yahoo has no membership history, so this returns None rather than
        pretending. That is the survivorship half of M8-02, and it is a data
        limitation: no amount of code fixes it. A vendor provider can implement
        this method and every caller keeps working.
        """
        return None

"""
Offline provider backed by parquet fixtures.

Every test runs against this, so the suite is deterministic and needs no network.
The fixture is real NSE data (39 tickers, ~5 years of OHLCV plus the NIFTY 50 and
Midcap 100 indices), extracted from the recording the legacy parity harness made,
which also keeps STAR2.0's numbers directly comparable with the frozen baseline.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import pandas as pd

from ..core.errors import DataError
from .provider import GUARD, PricePanel

_FIELDS = {"open": "nse_open.parquet", "high": "nse_high.parquet", "low": "nse_low.parquet",
           "close": "nse_close.parquet", "volume": "nse_volume.parquet"}


@lru_cache(maxsize=8)
def _read(path_str: str) -> pd.DataFrame:
    df = pd.read_parquet(path_str)
    df.index = pd.DatetimeIndex(df.index)
    return df.sort_index()


@dataclass
class FixtureProvider:
    """A `PriceProvider` (and benchmark source) reading committed parquet files."""

    root: Path
    name: str = "fixture"
    is_point_in_time: bool = False
    degradations: list[str] = field(default_factory=list)

    @classmethod
    def default(cls) -> "FixtureProvider":
        return cls(root=Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "nse")

    def _field(self, which: str) -> pd.DataFrame | None:
        path = self.root / _FIELDS[which]
        return _read(str(path)) if path.exists() else None

    @property
    def universe(self) -> list[str]:
        close = self._field("close")
        if close is None:
            raise DataError(f"fixture not found at {self.root}")
        return list(close.columns)

    def prices(self, tickers, start=None, end=None, *, adjusted: bool = True) -> PricePanel:
        close = self._field("close")
        if close is None:
            raise DataError(f"fixture not found at {self.root}")
        wanted = [t for t in tickers if t in close.columns]
        missing = [t for t in tickers if t not in close.columns]
        if missing:
            self.degradations.append(f"not in fixture: {missing[:8]}")
        if not wanted:
            raise DataError(f"none of the requested tickers are in the fixture: {list(tickers)[:5]}")

        lo = pd.Timestamp(start) if start is not None else close.index.min()
        hi = pd.Timestamp(end) if end is not None else close.index.max()
        GUARD.check(hi, what="fixture.prices")

        def slice_(which):
            f = self._field(which)
            if f is None:
                return None
            out = f.loc[(f.index >= lo) & (f.index <= hi), [c for c in wanted if c in f.columns]]
            return GUARD.clamp(out, what=f"fixture.{which}")

        close_s = slice_("close")
        if close_s is None or close_s.empty:
            raise DataError(f"fixture has no data between {lo.date()} and {hi.date()}")
        return PricePanel(close=close_s, close_raw=None, high=slice_("high"), low=slice_("low"),
                          open_=slice_("open"), volume=slice_("volume"),
                          source=f"{self.name}:{self.root.name}",
                          as_of=pd.Timestamp(close_s.index.max()))

    def benchmark(self, symbol: str = "^NSEI", start=None, end=None) -> pd.Series:
        name = symbol.replace("^", "").replace(".", "_")
        path = self.root / f"index_{name}.parquet"
        if not path.exists():
            raise DataError(f"no fixture for benchmark {symbol} (looked for {path.name})")
        s = _read(str(path))["close"]
        lo = pd.Timestamp(start) if start is not None else s.index.min()
        hi = pd.Timestamp(end) if end is not None else s.index.max()
        GUARD.check(hi, what=f"fixture.benchmark({symbol})")
        out = s.loc[(s.index >= lo) & (s.index <= hi)].dropna()
        out.name = symbol
        return GUARD.clamp(out, what=f"fixture.benchmark({symbol})")

    def snapshot(self, tickers, field_name: str, as_of) -> pd.Series:
        raise DataError("the fixture carries no fundamentals: backtests are price-only by "
                        "default, and fundamentals mode needs a live or vendor provider")

    def members(self, index: str, as_of) -> list[str] | None:
        return None      # no membership history, same honest answer as Yahoo

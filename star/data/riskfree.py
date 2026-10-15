"""
The risk-free rate as a dated curve, not one constant.

Legacy defects M0-11 / M2-15 (verified): a single 6.5% stood in for both today's
discount rate and five years of history. The RBI repo rate was 4.0% until May
2022, so every early backtest window was charged the wrong hurdle.

`RiskFreeCurve` holds annual simple rates as dated observations and answers
"what rate was in effect on date d" as-of: only observations on or before d are
used, so a backtest can never borrow a future rate. A constant curve is still
available, but it records a degradation that the report must show.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..core.errors import DataError
from ..core.types import Provenance
from .provider import GUARD

#: annual simple rates outside this band are almost certainly a unit error
PLAUSIBLE_RATE = (-0.05, 0.30)


@dataclass(frozen=True, slots=True)
class RiskFreeCurve:
    """Dated annual simple risk-free rates (a step function between observations)."""

    rates: pd.Series                 # annual simple rates, sorted by date
    name: str
    source: str
    is_constant: bool = False
    degradations: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()          # conversions applied at load (not degradations)

    def __post_init__(self) -> None:
        r = self.rates
        if r.empty:
            raise DataError("risk-free curve has no observations")
        if not isinstance(r.index, pd.DatetimeIndex):
            raise DataError("risk-free curve must be indexed by date")
        if r.index.has_duplicates or not r.index.is_monotonic_increasing:
            raise DataError("risk-free curve dates must be unique and increasing")
        if not np.isfinite(r.values).all():
            raise DataError("risk-free curve contains missing or infinite rates")
        lo, hi = PLAUSIBLE_RATE
        if (r < lo).any() or (r > hi).any():
            raise DataError(f"risk-free rates outside {lo:.0%}..{hi:.0%}: check the units")

    # ---- construction ----
    @classmethod
    def from_series(cls, rates: pd.Series, *, name: str, source: str, units: str = "auto",
                    quote: str = "effective", tenor_days: int = 91) -> "RiskFreeCurve":
        """
        Build a curve from dated rates.

        units: "decimal" (0.065), "percent" (6.5) or "auto". Auto reads anything
            above 1 as percent; when every value is <= 1 it assumes decimals and
            records that assumption, because a percent-quoted series of low rates
            (0.14, 0.08) is indistinguishable from decimals by value alone.
        quote: "effective" (an effective annual rate) or "tbill_simple" (a money-
            market yield annualised simply on 365 days, the usual quote for
            Indian Treasury bills), converted as (1 + y*t/365)^(365/t) - 1.
        """
        s = pd.Series(rates, dtype="float64").dropna()
        s.index = pd.DatetimeIndex(s.index)
        s = s.sort_index()
        notes: list[str] = []
        degradations: list[str] = []
        if units not in ("auto", "decimal", "percent"):
            raise DataError(f"units must be auto, decimal or percent, got {units!r}")
        if units == "percent" or (units == "auto" and s.abs().max() > 1.0):
            s = s / 100.0
            notes.append("rates quoted in percent, converted to decimals")
        elif units == "auto":
            degradations.append("rate units ambiguous (every value <= 1): read as decimals; "
                                 "pass units='percent' if they are percentages")
        if quote == "tbill_simple":
            t = float(tenor_days)
            s = (1.0 + s * t / 365.0) ** (365.0 / t) - 1.0
            notes.append(f"{tenor_days}-day simple money-market yields converted to effective annual rates")
        elif quote != "effective":
            raise DataError(f"quote must be effective or tbill_simple, got {quote!r}")
        return cls(rates=s, name=name, source=source, degradations=tuple(degradations), notes=tuple(notes))

    #: tried in order; the first that parses EVERY row wins. Month-first US formats
    #: are deliberately absent: 01-02-2020 is 1 February in Indian sources.
    DATE_FORMATS = ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%d-%b-%Y", "%d %b %Y", "%d-%b-%y", "%Y/%m/%d")

    @classmethod
    def _parse_dates(cls, values: pd.Series, date_format: str | None, filename: str) -> pd.DatetimeIndex:
        text = values.astype(str).str.strip()
        formats = (date_format,) if date_format else cls.DATE_FORMATS
        for fmt in formats:
            parsed = pd.to_datetime(text, format=fmt, errors="coerce")
            if parsed.notna().all():
                return pd.DatetimeIndex(parsed)
        bad = text[pd.to_datetime(text, format=formats[0], errors="coerce").isna()].iloc[0]
        raise DataError(
            f"{filename}: cannot read the dates consistently (e.g. {bad!r}). Supported formats: "
            f"{', '.join(formats)}; pass date_format=... for anything else. Dates are never guessed, "
            f"because a misread date moves a future rate into the past.")

    @classmethod
    def from_csv(cls, path: str | Path, *, date_col: str | None = None, rate_col: str | None = None,
                 name: str = "risk-free", date_format: str | None = None, units: str = "auto",
                 quote: str = "tbill_simple", tenor_days: int = 91) -> "RiskFreeCurve":
        """
        Load a CSV of dates and rates.

        With no column names given, the first column is the date and the first
        numeric column is the rate. The default quote is the simple 91-day T-bill
        yield; pass quote="effective" for series that are already effective rates.
        """
        p = Path(path)
        if not p.exists():
            raise DataError(f"risk-free file not found: {p}")
        df = pd.read_csv(p, dtype=str)
        if df.shape[1] < 2:
            raise DataError(f"{p.name}: expected a date column and a rate column")
        dcol = date_col or df.columns[0]
        df = df[df[dcol].notna() & (df[dcol].str.strip() != "")]
        if rate_col is None:
            numeric = [c for c in df.columns if c != dcol
                       and pd.to_numeric(df[c], errors="coerce").notna().mean() > 0.5]
            if not numeric:
                raise DataError(f"{p.name}: no numeric rate column found")
            rate_col = numeric[0]
        values = pd.to_numeric(df[rate_col], errors="coerce")
        keep = values.notna()                     # e.g. FRED writes "." for a missing month
        dates = cls._parse_dates(df.loc[keep, dcol], date_format, p.name)
        if len(dates) == 0:
            raise DataError(f"{p.name}: no valid date/rate rows")
        s = pd.Series(values[keep].values, index=dates)
        if s.index.has_duplicates:
            raise DataError(f"{p.name}: duplicate dates")
        return cls.from_series(s, name=name, source=f"csv:{p.name}", units=units,
                               quote=quote, tenor_days=tenor_days)

    @classmethod
    def constant(cls, rate: float, *, since="1990-01-01", reason: str) -> "RiskFreeCurve":
        """A flat curve. Always carries a degradation, because history was not flat."""
        return cls(rates=pd.Series([float(rate)], index=pd.DatetimeIndex([pd.Timestamp(since)])),
                   name="constant risk-free", source="constant", is_constant=True,
                   degradations=(f"constant risk-free rate {rate:.2%} used for all dates: {reason}",))

    # ---- queries (as-of: never uses a later observation) ----
    def rate(self, date) -> float:
        """The annual simple rate in effect on `date`."""
        d = pd.Timestamp(date)
        GUARD.check(d, what="risk_free.rate")
        eligible = self.rates.loc[self.rates.index <= d]
        if eligible.empty:
            raise DataError(f"no risk-free observation on or before {d.date()} "
                            f"(curve starts {self.rates.index[0].date()})")
        return float(eligible.iloc[-1])

    def series(self, start, end) -> pd.Series:
        """Observations in [start, end], plus the one already in effect at `start`."""
        s, e = pd.Timestamp(start), pd.Timestamp(end)
        GUARD.check(e, what="risk_free.series")
        before = self.rates.loc[self.rates.index <= s]
        inside = self.rates.loc[(self.rates.index > s) & (self.rates.index <= e)]
        head = before.iloc[-1:] if not before.empty else before
        return pd.concat([head, inside])

    def on_sessions(self, sessions: pd.DatetimeIndex) -> pd.Series:
        """The annual simple rate in effect on each session (as-of step function)."""
        idx = pd.DatetimeIndex(sessions)
        if len(idx) == 0:
            return pd.Series(dtype="float64")
        GUARD.check(idx.max(), what="risk_free.on_sessions")
        aligned = self.rates.reindex(self.rates.index.union(idx)).ffill().reindex(idx)
        if aligned.isna().any():
            first = idx[aligned.isna()][0]
            raise DataError(f"no risk-free observation on or before {first.date()}")
        return aligned

    def daily_simple(self, sessions: pd.DatetimeIndex, *, periods_per_year: float = 252.0) -> pd.Series:
        """Per-session simple rate, de-annualised by compounding (not by division)."""
        annual = self.on_sessions(sessions)
        return (1.0 + annual) ** (1.0 / periods_per_year) - 1.0

    @property
    def provenance(self) -> Provenance:
        return Provenance(method=self.name, source=self.source,
                          as_of=self.rates.index[-1], degradations=self.degradations)

    def describe(self) -> str:
        last = self.rates.index[-1].date()
        extra = "; ".join(self.notes + self.degradations)
        return f"{self.name} ({self.source}, last observation {last})" + (f": {extra}" if extra else "")

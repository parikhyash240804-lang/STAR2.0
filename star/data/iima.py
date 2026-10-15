"""
The IIMA Indian factor library: Carhart four factors and the RBI T-bill rate.

Source: Agarwalla, Jacob & Varma, "Four factor model in Indian equities market",
IIMA W.P. 2013-09-05, data library at faculty.iima.ac.in/iffm/Indian-Fama-French-Momentum
(new series, computed from CMIE Prowess DX, NSE and BSE, micro-caps and penny
stocks excluded). What the library provides, verified 2026-09-19:

* **Four factors, not five.** MF (market minus risk-free), SMB (size), HML (value),
  WML (momentum) - the Carhart (1997) model. There are no profitability or
  investment factors, so STAR2.0's attribution for India is four-factor.
* **The risk-free rate.** Per the library's FAQ, RF is derived from "the
  annualized yields of new issuances of 91-day Treasury Bills, provided by the
  Reserve Bank of India". This is the dated T-bill series the risk-free curve
  needs (FRED's IMF mirror for India no longer exists).
* **Holding-period returns in percent.** Each row is the return since the
  previous row, so Monday's RF covers three calendar days. The annual yield is
  recovered as RF/100 x 365 / (calendar days since the previous row).
* **Revisions and lag.** Files are re-issued three times a year and past values
  can change when CMIE revises its database. The data ends at the release's last
  month, so later dates carry the last rate forward - reported as a degradation.

Use the survivorship-bias-adjusted files (the library's own recommendation).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..core.errors import DataError
from ..core.types import Provenance
from .riskfree import RiskFreeCurve

FACTOR_COLUMNS = ("MF", "SMB", "HML", "WML")
#: carrying the last published rate forward beyond this many days is flagged
STALE_AFTER_DAYS = 45


def default_path() -> Path:
    """The bundled release: STAR2.0/data/reference/iima/<release>_FourFactors..._Daily_SurvivorshipBiasAdjusted.csv"""
    folder = Path(__file__).resolve().parents[2] / "data" / "reference" / "iima"
    files = sorted(folder.glob("*_FourFactors_and_Market_Returns_Daily_SurvivorshipBiasAdjusted.csv"))
    if not files:
        raise DataError(f"no IIMA daily factor file found in {folder}")
    return files[-1]                       # latest release by name (YYYY-MM prefix)


@dataclass(frozen=True, slots=True)
class IimaFactors:
    """Daily Indian factor returns (decimals) and the RBI 91-day T-bill yield."""

    factors: pd.DataFrame          # MF, SMB, HML, WML - daily simple returns, decimal
    rf_period: pd.Series           # daily holding-period risk-free return, decimal
    tbill_yield: pd.Series         # annualised simple 91-day T-bill yield, decimal
    release: str                   # e.g. "2025-12"
    survivorship_adjusted: bool
    source: str

    @classmethod
    def load(cls, path: str | Path | None = None) -> "IimaFactors":
        p = Path(path) if path else default_path()
        if not p.exists():
            raise DataError(f"IIMA factor file not found: {p}")
        df = pd.read_csv(p, na_values=["NA"])
        missing = {"Date", "RF", *FACTOR_COLUMNS} - set(df.columns)
        if missing:
            raise DataError(f"{p.name}: expected columns {sorted(missing)} are missing")
        dates = pd.to_datetime(df["Date"], format="%Y-%m-%d", errors="coerce")
        if dates.isna().any():
            raise DataError(f"{p.name}: unreadable dates, e.g. {df['Date'][dates.isna()].iloc[0]!r}")
        df.index = pd.DatetimeIndex(dates)
        if df.index.has_duplicates or not df.index.is_monotonic_increasing:
            raise DataError(f"{p.name}: dates must be unique and increasing")

        factors = df[list(FACTOR_COLUMNS)].astype("float64") / 100.0     # percent -> decimal
        rf = (df["RF"].astype("float64") / 100.0).dropna()
        gap_days = rf.index.to_series().diff().dt.days
        # each RF value accrues over the calendar days since the previous row
        tbill = (rf * 365.0 / gap_days).dropna()
        tbill = tbill[(tbill > 0) & np.isfinite(tbill)]
        release = re.match(r"(\d{4}-\d{2})", p.name)
        return cls(factors=factors, rf_period=rf, tbill_yield=tbill,
                   release=release.group(1) if release else "unknown",
                   survivorship_adjusted="SurvivorshipBiasAdjusted" in p.name,
                   source=f"IIMA factor library {release.group(1) if release else ''} ({p.name})")

    @property
    def last_date(self) -> pd.Timestamp:
        return pd.Timestamp(self.tbill_yield.index[-1])

    def risk_free_curve(self, *, monthly: bool = True) -> RiskFreeCurve:
        """
        The RBI 91-day T-bill curve, as effective annual rates.

        Daily yields change only at weekly auctions, so month-end sampling keeps
        the curve compact without losing information a monthly-rebalanced
        backtest could use; `monthly=False` keeps every observation.
        """
        y = self.tbill_yield
        if monthly:
            y = y.groupby(y.index.to_period("M")).last()
            y.index = y.index.to_timestamp(how="end").normalize()
            # the month-end label must not precede the observation it carries
            y.index = pd.DatetimeIndex([min(d, self.last_date) for d in y.index])
        return RiskFreeCurve.from_series(y, name="RBI 91-day T-bill (via IIMA)", source=self.source,
                                         units="decimal", quote="tbill_simple", tenor_days=91)

    def staleness_note(self, as_of) -> str | None:
        """A degradation to report when `as_of` is well past the data's last date."""
        gap = (pd.Timestamp(as_of) - self.last_date).days
        if gap > STALE_AFTER_DAYS:
            return (f"IIMA data ends {self.last_date.date()} ({gap} days before {pd.Timestamp(as_of).date()}): "
                    f"the last published T-bill rate and factor returns are carried forward")
        return None

    def provenance(self, as_of=None) -> Provenance:
        notes = (self.staleness_note(as_of),) if as_of is not None and self.staleness_note(as_of) else ()
        return Provenance(method="Carhart four-factor (IIMA)", source=self.source,
                          as_of=self.last_date, degradations=notes)

"""
The NIFTY 50 total-return benchmark, built from free data and calibrated honestly.

The problem (legacy defects M0-02 / M8-07, verified): the legacy compared
dividend-adjusted stocks against the NIFTY 50 *price* index, handicapping the
benchmark by the index dividend yield (~1.2-1.4% a year) and flattering every
reported outperformance. There is no free NIFTY 50 total-return index.

What was measured before designing this (2026-09-16, live Yahoo data):

* NIFTYBEES.NS (an ETF that retains index dividends inside its NAV) beat the
  NIFTY 50 price index by +1.04%, +1.23%, +1.20% and +1.26% a year over 1, 3, 5
  and 10 years - consistent with the dividend yield less the ETF fee. So it is a
  sound *long-run* total-return proxy.
* Its *daily* series is not: daily correlation with the index is only ~0.97
  (tracking error ~3% a year from market-price noise), and Yahoo's series has a
  broken unit split in Dec 2019 (-89.9% then +897% three sessions later) that
  collapses the 10-year daily correlation to 0.13.

So the benchmark uses each source for what it is good at: the price index's
clean daily moves, plus a constant per-session dividend accrual equal to the
NIFTYBEES-minus-index log growth gap per session, measured over a trailing
window that ends at `as_of` (so the built index reproduces the proxy's growth
over that window exactly).
Calibrating only on data up to `as_of` means a backtest never uses a dividend
yield estimated from its own future.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..core.errors import DataError
from ..core.types import Provenance
from .quality import PriceEvent, repair_price_artifacts

#: used only when NIFTYBEES is unavailable or implausible; always flagged
FALLBACK_DIVIDEND_YIELD = 0.012
#: a calibrated gap outside this band means the proxy is broken, not informative
PLAUSIBLE_GAP = (0.0, 0.03)


@dataclass(frozen=True, slots=True)
class BenchmarkSeries:
    """A daily NIFTY 50 total-return proxy with the evidence for how it was built."""

    tri_log_returns: pd.Series          # daily log total return (proxy)
    price_log_returns: pd.Series        # daily log price return of the index
    dividend_accrual_annual: float      # annual simple rate the accrual corresponds to
    accrual_log_per_session: float      # what is actually added to each daily log price return
    calibrated: bool                    # False when the fallback yield was used
    calibration_start: pd.Timestamp | None
    calibration_end: pd.Timestamp | None
    etf_daily_correlation: float | None
    etf_tracking_error: float | None    # annualised, ETF vs index daily log returns
    repaired_events: tuple[PriceEvent, ...]
    as_of: pd.Timestamp
    provenance: Provenance
    periods_per_year: float = 252.0

    def tri_level(self, base: float = 100.0) -> pd.Series:
        """Total-return index level, starting at `base`."""
        return base * np.exp(self.tri_log_returns.cumsum())

    def price_level(self, base: float = 100.0) -> pd.Series:
        return base * np.exp(self.price_log_returns.cumsum())

    def describe(self) -> str:
        """One line for the report, stating exactly what the benchmark is."""
        how = (f"calibrated on NIFTYBEES {self.calibration_start.date()} to {self.calibration_end.date()}"
               if self.calibrated else "fallback estimate - NIFTYBEES unavailable or implausible")
        return (f"NIFTY 50 total return = index price return + "
                f"{self.dividend_accrual_annual:.2%}/yr dividend accrual ({how})")


def build_benchmark(
    index_close: pd.Series,
    tri_proxy_close: pd.Series | None,
    *,
    as_of,
    window_years: float = 5.0,
    min_years: float = 2.0,
    periods_per_year: float = 252.0,
) -> BenchmarkSeries:
    """
    Build the total-return benchmark up to `as_of`.

    Parameters
    ----------
    index_close : NIFTY 50 price index closes (e.g. ^NSEI)
    tri_proxy_close : NIFTYBEES.NS closes, or None
    as_of : nothing after this date is used, for the series or the calibration
    window_years : trailing window for measuring the dividend gap
    min_years : shorter overlapping history than this uses the flagged fallback
    """
    cutoff = pd.Timestamp(as_of)
    idx = index_close.dropna()
    idx = idx.loc[idx.index <= cutoff]
    if len(idx) < 2:
        raise DataError(f"no benchmark index data on or before {cutoff.date()}")
    price_lr = np.log(idx / idx.shift(1)).iloc[1:]
    price_lr.name = "price"

    notes: list[str] = []
    events: tuple[PriceEvent, ...] = ()
    gap, calibrated, cal_start, cal_end = FALLBACK_DIVIDEND_YIELD, False, None, None
    accrual_log = float(np.log1p(FALLBACK_DIVIDEND_YIELD) / periods_per_year)
    corr = te = None

    if tri_proxy_close is None or tri_proxy_close.dropna().empty:
        notes.append("NIFTYBEES unavailable: fallback dividend accrual used")
    else:
        # truncate BEFORE repairing: detecting a spike's reversal must not use
        # sessions after as_of, or the backtest would repair with future data
        raw = tri_proxy_close.dropna()
        raw = raw.loc[raw.index <= cutoff].rename("proxy")
        cleaned = repair_price_artifacts(raw)
        events = tuple(cleaned.repaired)
        proxy = cleaned.close.iloc[:, 0]
        start = cutoff - pd.Timedelta(days=365.25 * window_years)
        common = proxy.index.intersection(idx.index)
        common = common[common >= start]
        unresolved = [e for e in cleaned.flagged
                      if abs(e.move) >= 0.45 and e.start >= (common[0] if len(common) else cutoff)]
        if len(common) < 2 or (common[-1] - common[0]).days / 365.25 < min_years:
            notes.append(f"under {min_years:g} years of NIFTYBEES overlap before "
                         f"{cutoff.date()}: fallback dividend accrual used")
        elif unresolved:
            notes.append(f"NIFTYBEES has an unresolved implausible move on "
                         f"{unresolved[0].start.date()} ({unresolved[0].move:+.1%}) inside the "
                         f"calibration window: fallback dividend accrual used")
        else:
            # gap measured in log terms per actual session: adding it to every
            # daily index log return reproduces the proxy's growth over the window
            # exactly (a difference of CAGRs over an assumed 252 days does not)
            n_int = len(common) - 1
            g_log = (np.log(proxy.loc[common[-1]] / proxy.loc[common[0]])
                     - np.log(idx.loc[common[-1]] / idx.loc[common[0]])) / n_int
            sessions_per_year = n_int / ((common[-1] - common[0]).days / 365.25)
            measured = float(np.expm1(g_log * sessions_per_year))
            if PLAUSIBLE_GAP[0] <= measured <= PLAUSIBLE_GAP[1]:
                gap, calibrated, accrual_log = measured, True, float(g_log)
                cal_start, cal_end = pd.Timestamp(common[0]), pd.Timestamp(common[-1])
            else:
                notes.append(f"measured NIFTYBEES gap {measured:+.2%}/yr is implausible: "
                             f"fallback dividend accrual used")
            pr = np.log(proxy.loc[common] / proxy.loc[common].shift(1)).iloc[1:]
            ir = np.log(idx.loc[common] / idx.loc[common].shift(1)).iloc[1:]
            corr = float(pr.corr(ir))
            te = float((pr - ir).std() * np.sqrt(periods_per_year))
        for e in events:
            notes.append(f"NIFTYBEES {e.kind.value} {e.start.date()} to {e.end.date()} "
                         f"({e.move:+.1%}) repaired before calibration")

    tri_lr = (price_lr + accrual_log).rename("tri")

    prov = Provenance(
        method="NIFTY 50 price return + calibrated dividend accrual",
        window=f"{window_years:g}y trailing",
        as_of=cutoff,
        source="^NSEI + NIFTYBEES.NS",
        degradations=tuple(n for n in notes if "fallback" in n or "implausible" in n),
    )
    return BenchmarkSeries(
        tri_log_returns=tri_lr, price_log_returns=price_lr, dividend_accrual_annual=float(gap),
        accrual_log_per_session=accrual_log,
        calibrated=calibrated, calibration_start=cal_start, calibration_end=cal_end,
        etf_daily_correlation=corr, etf_tracking_error=te, repaired_events=events,
        as_of=cutoff, provenance=prov, periods_per_year=periods_per_year,
    )

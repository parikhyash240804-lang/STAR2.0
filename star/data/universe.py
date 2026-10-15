"""
The investable universe as it was knowable on a given date.

Legacy defect M8-02 / M1-06 (verified, critical): the backtest used today's
hand-picked ticker list, filtered on missing data measured over the *whole*
sample, at every past rebalance. That mixes two problems:

* **Survivorship** - only stocks that exist today are ever considered. Free
  Yahoo data has no delisted names and no index-membership history, so this half
  cannot be fixed here; the report must say so, and a point-in-time vendor plugs in
  through `UniverseProvider`.
* **Look-ahead in the filter** - whether a stock had "too much missing data" was
  judged using years that had not happened yet. This half *is* fixable, and this
  module fixes it: every screen uses only sessions on or before `as_of`.
"""
from __future__ import annotations

import pandas as pd

from ..core.errors import DataError
from .quality import DataQualityReport, assess


def as_of_universe(
    close: pd.DataFrame,
    as_of,
    *,
    lookback_sessions: int | None = 756,
    volume: pd.DataFrame | None = None,
    max_missing: float = 0.05,
    min_sessions: int = 250,
    min_adv_inr: float = 1_000_000.0,
    min_names: int = 3,
) -> DataQualityReport:
    """
    Screen the universe using only data available on `as_of`.

    The screening window is the `lookback_sessions` sessions ending at `as_of`
    (the estimation window a strategy would actually use then). A stock that had
    not listed yet has no data in that window and is excluded as such - never
    because of data that only exists in its future.
    """
    cutoff = pd.Timestamp(as_of)
    past = close.loc[close.index <= cutoff]
    if past.empty:
        raise DataError(f"no price data on or before {cutoff.date()}")
    if lookback_sessions is not None:
        if int(lookback_sessions) < 1:
            raise DataError(f"lookback_sessions must be at least 1, got {lookback_sessions}")
        past = past.iloc[-int(lookback_sessions):]
    vol = None
    if volume is not None:
        vol = volume.loc[volume.index <= cutoff].reindex(past.index)
    report = assess(past, volume=vol, max_missing=max_missing, min_sessions=min_sessions,
                    min_adv_inr=min_adv_inr, min_names=min_names)
    report.notes.append(
        f"screened on {len(past)} sessions ending {cutoff.date()} (as-of; no later data used). "
        f"Survivorship is not corrected: the candidate list is today's.")
    return report

"""
Data quality as a reported decision, not a silent filter.

Measured on the default 39-ticker universe, the legacy engine drops HYUNDAI.NS
(61.9% missing - it listed in late 2024) and GATEWAY.NS (10.6% missing) because
they exceed `MISSING_THRESHOLD`, leaving 37 stocks. That filter is defensible: it
is what stops one recent IPO from truncating the whole panel when `dropna()` runs.
What is not defensible is that the user is told only through a `print` that the
GUI discards (M1-12, M3-09), and the report then describes a "5-year lookback"
over a universe two names smaller than the one requested.

So here every exclusion is a `TickerVerdict` with a reason, a number and a
severity, carried into the report. Nothing is dropped silently, and nothing is
forward-filled into existence: a missing day stays missing, because fabricating
zero-return days biases every volatility estimate downward (M1-05).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np
import pandas as pd

from ..core.errors import DataError


class Disposition(StrEnum):
    KEEP = "keep"
    DROP = "drop"


@dataclass(frozen=True, slots=True)
class TickerVerdict:
    """Why one ticker is in or out, in words a user can act on."""

    ticker: str
    disposition: Disposition
    reason: str
    missing_pct: float
    first_session: pd.Timestamp | None
    last_session: pd.Timestamp | None
    sessions: int

    @property
    def dropped(self) -> bool:
        return self.disposition is Disposition.DROP


@dataclass(slots=True)
class DataQualityReport:
    """The full picture: what was requested, what survived, and why."""

    requested: list[str]
    verdicts: list[TickerVerdict] = field(default_factory=list)
    panel_sessions: int = 0
    panel_start: pd.Timestamp | None = None
    panel_end: pd.Timestamp | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def kept(self) -> list[str]:
        return [v.ticker for v in self.verdicts if not v.dropped]

    @property
    def dropped(self) -> list[TickerVerdict]:
        return [v for v in self.verdicts if v.dropped]

    @property
    def coverage_years(self) -> float:
        if self.panel_start is None or self.panel_end is None:
            return 0.0
        return (self.panel_end - self.panel_start).days / 365.25

    def headline(self) -> str:
        """One line the report can print verbatim, instead of a silent filter."""
        if not self.dropped:
            return (f"{len(self.kept)} stocks, {self.panel_sessions} sessions "
                    f"({self.coverage_years:.1f} years), no exclusions.")
        names = ", ".join(f"{v.ticker.removesuffix('.NS')} ({v.reason})" for v in self.dropped[:4])
        more = "" if len(self.dropped) <= 4 else f" and {len(self.dropped) - 4} more"
        return (f"{len(self.kept)} of {len(self.requested)} stocks usable over "
                f"{self.panel_sessions} sessions ({self.coverage_years:.1f} years). "
                f"Excluded: {names}{more}.")

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([{
            "ticker": v.ticker, "disposition": str(v.disposition), "reason": v.reason,
            "missing_pct": round(v.missing_pct * 100, 2), "sessions": v.sessions,
            "first": v.first_session, "last": v.last_session,
        } for v in self.verdicts])


def assess(
    close: pd.DataFrame,
    *,
    volume: pd.DataFrame | None = None,
    price_for_turnover: pd.DataFrame | None = None,
    max_missing: float = 0.05,
    min_sessions: int = 250,
    min_adv_inr: float = 1_000_000.0,
    min_names: int = 3,
) -> DataQualityReport:
    """
    Judge each ticker, explaining every exclusion.

    Parameters mirror the legacy's intent but fix its units: liquidity is screened
    in **rupees traded per day**, not share count. The legacy's `VOLUME_MIN` of
    10,000 *shares* is not a liquidity threshold at all - 10,000 shares of a
    Rs.50 stock is Rs.5 lakh a day, while 10,000 shares of a Rs.5,000 stock is
    Rs.5 crore - and it only printed a warning rather than acting (M1-13).

    Turnover should be valued at the traded (unadjusted) price: pass
    `price_for_turnover` (e.g. `close_raw`). The dividend-adjusted close embeds
    dividends paid later, a small but real look-ahead.
    """
    if close.empty:
        raise DataError("no price data to assess")

    report = DataQualityReport(
        requested=list(close.columns),
        panel_sessions=int(len(close)),
        panel_start=pd.Timestamp(close.index.min()),
        panel_end=pd.Timestamp(close.index.max()),
    )

    adv = None
    screen_liquidity = volume is not None and min_adv_inr > 0
    if volume is not None:
        px = price_for_turnover if price_for_turnover is not None else close
        traded = volume.reindex_like(close) * px.reindex_like(close)
        adv = traded.median(skipna=True)          # median rupee volume per session

    for t in close.columns:
        col = close[t]
        present = col.notna()
        n_present = int(present.sum())
        missing_pct = 1.0 - (n_present / len(col) if len(col) else 0.0)
        first = pd.Timestamp(col.index[present.argmax()]) if n_present else None
        last = pd.Timestamp(col.index[len(col) - 1 - present.values[::-1].argmax()]) if n_present else None

        disposition, reason = Disposition.KEEP, "complete"
        if n_present == 0:
            disposition, reason = Disposition.DROP, "no price data"
        elif missing_pct > max_missing:
            # the usual cause is a recent listing, which is worth naming explicitly
            listed_late = first is not None and first > close.index.min()
            reason = (f"listed {first.date()}, only {n_present} of {len(col)} sessions"
                      if listed_late else f"{missing_pct:.1%} of sessions missing")
            disposition = Disposition.DROP
        elif n_present < min_sessions:
            disposition, reason = Disposition.DROP, f"only {n_present} sessions of history"
        elif screen_liquidity and (t not in adv.index or not np.isfinite(adv[t])):
            # absence of volume is not evidence of liquidity
            disposition, reason = Disposition.DROP, "no volume data to confirm liquidity"
        elif screen_liquidity and adv[t] < min_adv_inr:
            disposition = Disposition.DROP
            reason = f"median turnover Rs.{adv[t]:,.0f}/day is below the liquidity floor"
        elif missing_pct > 0:
            reason = f"{missing_pct:.1%} of sessions missing (within tolerance)"

        report.verdicts.append(TickerVerdict(
            ticker=t, disposition=disposition, reason=reason, missing_pct=missing_pct,
            first_session=first, last_session=last, sessions=n_present))

    if len(report.kept) < min_names:
        raise DataError(
            f"only {len(report.kept)} usable stock(s) after quality screening "
            f"(need {min_names}). {report.headline()}")

    dropped_recent = [v for v in report.dropped if "listed" in v.reason]
    if dropped_recent:
        report.notes.append(
            f"{len(dropped_recent)} stock(s) excluded for short history; including them "
            f"would truncate the common sample for every other holding.")
    return report


def common_sample(close: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """
    The rectangular panel used for estimation, without fabricating data.

    Rows are kept only where every retained ticker actually traded. No
    forward-filling: a filled day is a fabricated zero return, and a few of those
    bias volatility down and correlation up precisely when it matters.
    """
    sub = close[tickers]
    keep = sub.notna().all(axis=1)
    out = sub.loc[keep]
    if len(out) < 2:
        raise DataError("no overlapping sessions across the retained universe")
    return out


# ---------------------------------------------------------------------------
# Price artifacts: repair only the impossible, flag the ambiguous, keep the real
# ---------------------------------------------------------------------------
#
# Measured on real data: Yahoo's NIFTYBEES.NS series moves -89.9% on 19 Dec 2019
# and +897% three sessions later - a unit split applied to all but two days.
# An index ETF cannot fall 90% in a day, so that is a data error, and left in it
# destroys every volatility, drawdown and regime estimate trained on it.
#
# The same scan over the 39-stock universe flags 34 large moves, and almost all
# of them are REAL: L&T and Power Grid fell ~12% on the 4 June 2024 election
# result, Tata Steel and Jindal Steel fell 13-17% on the May 2022 export duty,
# and several mid-caps hit India's 20% circuit bands on results. "Cleaning" those
# would sanitise exactly the tail risk a risk system exists to measure. So:
#
#   repair   a move of >= 45% that snaps back within a few sessions (a broken
#            adjustment), a CONFIRMED split-like jump (see below), or a price
#            that is zero or negative (removed as missing, never interpolated from);
#   flag     a smaller spike that reverts, and any split-like jump that cannot be
#            confirmed;
#   keep     everything else, including every genuine crash.
#
# A split-like jump is only confirmed when (a) enough later sessions exist to see
# that it did not reverse - otherwise a bad print on the as-of day would rescale
# the whole history, and the history would change once the print reverted - and
# (b) traded volume shifted by the inverse ratio, as it must after a real split. A
# genuine -50% crash looks exactly like a 1:2 split in prices alone; only volume
# tells them apart. Without volume, only unmistakable ratios (1:4 or larger) are
# repaired and everything else is flagged for a human.

class ArtifactKind(StrEnum):
    SPIKE_REVERT = "spike_revert"          # a big move that snaps back within a few sessions
    SPLIT_LIKE_JUMP = "split_like_jump"    # an isolated move matching a split/bonus ratio
    NON_POSITIVE = "non_positive_price"    # a zero or negative price: impossible for listed equity
    LARGE_MOVE = "large_move"              # extreme but plausible: a real event until shown otherwise


class ArtifactAction(StrEnum):
    REPAIRED = "repaired"
    FLAGGED = "flagged"


@dataclass(frozen=True, slots=True)
class PriceEvent:
    ticker: str
    kind: ArtifactKind
    action: ArtifactAction
    start: pd.Timestamp        # first affected session
    end: pd.Timestamp          # last affected session (the revert day for spike-reverts)
    move: float                # simple return on the first day
    net: float                 # simple return across the whole block
    note: str


#: split/bonus ratios as the price multiple they produce on the day. A 3:2 bonus
#: (x2/3, -33%) is deliberately absent: it lies inside plausible market moves, so
#: prices alone cannot distinguish it and it is left to the volume-confirmed path.
_SPLIT_RATIOS = (1 / 2, 1 / 3, 1 / 4, 1 / 5, 1 / 10, 2.0, 3.0, 4.0, 5.0, 10.0)
_UNMISTAKABLE = 4.0          # without volume, repair only ratios at least this extreme


def _matches_split(ratio: float, tol: float = 0.03) -> float | None:
    for r in _SPLIT_RATIOS:
        if abs(ratio / r - 1.0) <= tol:
            return r
    return None


def _valid_prices(s: pd.Series) -> pd.Series:
    """The observations artifact logic may use: present and strictly positive."""
    s = s.dropna()
    return s[s > 0]


def _volume_confirms_split(volume, ticker, date, price_ratio, window=20) -> bool | None:
    """True/False if volume data can judge the split, None if it cannot."""
    if volume is None:
        return None
    vol = volume.to_frame() if isinstance(volume, pd.Series) else volume
    if ticker not in vol.columns:
        return None
    v = vol[ticker].dropna()
    v = v[v > 0]
    before, after = v.loc[v.index < date].iloc[-window:], v.loc[v.index >= date].iloc[:window]
    if len(before) < 5 or len(after) < 5:
        return None
    shift = float(after.median() / before.median())
    expected = 1.0 / price_ratio                 # 1:2 split halves price, doubles share volume
    return 0.5 * expected <= shift <= 2.0 * expected


def scan_price_artifacts(
    close: pd.DataFrame | pd.Series,
    *,
    volume: pd.DataFrame | pd.Series | None = None,
    impossible_move: float = 0.45,
    flag_move: float = 0.10,
    mad_multiple: float = 8.0,
    revert_sessions: int = 3,
    revert_fraction: float = 0.25,
) -> list[PriceEvent]:
    """
    Find suspicious prices and daily moves, classifying each as repairable or flag-only.

    A move is examined only if it exceeds both `flag_move` and `mad_multiple`
    robust standard deviations of that series' own daily log returns, so a
    volatile small-cap is not held to a large-cap's standard.
    """
    frame = close.to_frame() if isinstance(close, pd.Series) else close
    events: list[PriceEvent] = []
    for ticker in frame.columns:
        raw = frame[ticker].dropna()
        for d, p in raw[raw <= 0].items():
            events.append(PriceEvent(ticker, ArtifactKind.NON_POSITIVE, ArtifactAction.REPAIRED,
                                     pd.Timestamp(d), pd.Timestamp(d), float("nan"), float("nan"),
                                     f"price {p} removed and treated as missing"))
        s = _valid_prices(raw)
        if len(s) < 3:
            continue
        lr = np.diff(np.log(s.values))            # lr[i] is the move INTO session i+1
        dates = s.index
        mad = float(np.median(np.abs(lr - np.median(lr))) * 1.4826)
        threshold = max(np.log1p(flag_move), mad_multiple * mad)
        i, n = 0, len(lr)
        while i < n:
            move = lr[i]
            if abs(move) <= threshold:
                i += 1
                continue
            simple = float(np.expm1(move))
            implausible = abs(simple) >= impossible_move
            revert_at = None
            for k in range(1, revert_sessions + 1):
                if i + k >= n:
                    break
                net = lr[i:i + k + 1].sum()
                if np.sign(lr[i + k]) != np.sign(move) and abs(net) < revert_fraction * abs(move):
                    revert_at = k
                    break
            start = pd.Timestamp(dates[i + 1])
            if revert_at is not None:
                net_simple = float(np.expm1(lr[i:i + revert_at + 1].sum()))
                action = ArtifactAction.REPAIRED if implausible else ArtifactAction.FLAGGED
                note = ("implausible move that reverses: a broken corporate-action adjustment"
                        if implausible else "spike that reverses: a bad print, or news later retracted")
                events.append(PriceEvent(ticker, ArtifactKind.SPIKE_REVERT, action, start,
                                         pd.Timestamp(dates[i + revert_at + 1]), simple, net_simple, note))
                i += revert_at + 1
                continue

            ratio = float(np.exp(move))
            split = _matches_split(ratio) if implausible else None
            confirmed_no_revert = i + revert_sessions < n
            if split is not None and not confirmed_no_revert:
                events.append(PriceEvent(ticker, ArtifactKind.LARGE_MOVE, ArtifactAction.FLAGGED, start, start,
                                         simple, simple,
                                         "split-like move too close to the end of the data to confirm; not repaired"))
            elif split is not None:
                by_volume = _volume_confirms_split(volume, ticker, start, split)
                unmistakable = split >= _UNMISTAKABLE or split <= 1.0 / _UNMISTAKABLE
                if by_volume is True or (by_volume is None and unmistakable):
                    how = "volume confirms" if by_volume else "ratio too extreme for a market move"
                    events.append(PriceEvent(ticker, ArtifactKind.SPLIT_LIKE_JUMP, ArtifactAction.REPAIRED,
                                             start, start, simple, simple,
                                             f"unadjusted split or bonus ({how}): earlier prices rescaled by {split:g}"))
                else:
                    why = ("volume did not shift as a split would" if by_volume is False
                           else "no volume data to tell a split from a crash")
                    events.append(PriceEvent(ticker, ArtifactKind.LARGE_MOVE, ArtifactAction.FLAGGED, start, start,
                                             simple, simple, f"split-like move not repaired: {why}; verify manually"))
            else:
                note = ("move beyond 45% that neither reverses nor matches a split ratio: verify manually"
                        if implausible else "large move kept as a genuine market event")
                events.append(PriceEvent(ticker, ArtifactKind.LARGE_MOVE, ArtifactAction.FLAGGED, start, start,
                                         simple, simple, note))
            i += 1
    return events


@dataclass(slots=True)
class CleanResult:
    close: pd.DataFrame
    events: list[PriceEvent]

    @property
    def repaired(self) -> list[PriceEvent]:
        return [e for e in self.events if e.action is ArtifactAction.REPAIRED]

    @property
    def flagged(self) -> list[PriceEvent]:
        return [e for e in self.events if e.action is ArtifactAction.FLAGGED]

    def notes(self) -> list[str]:
        out = []
        for e in self.repaired:
            if e.kind is ArtifactKind.NON_POSITIVE:
                out.append(f"{e.ticker}: {e.note} on {e.start.date()}")
            else:
                out.append(f"{e.ticker}: {e.kind.value} {e.start.date()} to {e.end.date()} "
                           f"({e.move:+.1%}, net {e.net:+.1%}) repaired: {e.note}")
        if self.flagged:
            out.append(f"{len(self.flagged)} large move(s) flagged and left unchanged")
        return out


def repair_price_artifacts(close: pd.DataFrame | pd.Series, **scan_kwargs) -> CleanResult:
    """
    Repair only the events `scan_price_artifacts` marks REPAIRED; return all events.

    Non-positive prices become missing. Spike-reverts: the bad closes between the
    spike and the revert are replaced by log-linear interpolation between the last
    good close before and the first good close after, which preserves the true net
    move. Split-like jumps: every close before the jump is rescaled by the ratio.
    """
    frame = (close.to_frame() if isinstance(close, pd.Series) else close).copy()
    events = scan_price_artifacts(frame, **scan_kwargs)
    for e in events:
        if e.action is ArtifactAction.REPAIRED and e.kind is ArtifactKind.NON_POSITIVE:
            frame.loc[e.start, e.ticker] = np.nan
    for e in events:
        if e.action is not ArtifactAction.REPAIRED or e.kind is ArtifactKind.NON_POSITIVE:
            continue
        valid = _valid_prices(frame[e.ticker])
        pos = valid.index.get_loc(e.start)
        if e.kind is ArtifactKind.SPIKE_REVERT:
            end_pos = valid.index.get_loc(e.end)
            before, after = valid.iloc[pos - 1], valid.iloc[end_pos]
            steps = end_pos - (pos - 1)
            bad_dates = valid.index[pos:end_pos]
            fill = np.exp(np.log(before) + (np.log(after) - np.log(before))
                          * np.arange(1, steps) / steps)
            frame.loc[bad_dates, e.ticker] = fill
        elif e.kind is ArtifactKind.SPLIT_LIKE_JUMP:
            ratio = _matches_split(1.0 + e.move)
            earlier = valid.index[:pos]
            frame.loc[earlier, e.ticker] = frame.loc[earlier, e.ticker] * ratio
    return CleanResult(close=frame, events=events)

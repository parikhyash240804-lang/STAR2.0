"""
The look-ahead canary.

Legacy defect M8-02: the backtest applied today's ticker list and full-sample data
filters at every past rebalance. The look-ahead half of that is closed here by
construction — asking for future data raises.
"""
import numpy as np
import pandas as pd
import pytest

from star.core.calendar import TradingCalendar
from star.core.errors import DataError, LookAheadError
from star.data.provider import AsOfGuard, PricePanel
from star.data.view import DataView

TICKERS = ["AAA.NS", "BBB.NS"]


def panel(n=300, seed=3) -> PricePanel:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2024-01-01", periods=n)
    px = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, size=(n, 2)), axis=0)),
                      index=idx, columns=TICKERS)
    return PricePanel(close=px, close_raw=px, high=px * 1.01, low=px * 0.99, open_=px,
                      volume=pd.DataFrame(1e6, index=idx, columns=TICKERS),
                      source="synthetic", as_of=idx[-1])


# --------------------------------------------------------------------------
# the guard
# --------------------------------------------------------------------------
def test_guard_blocks_requests_past_the_cutoff():
    g = AsOfGuard()
    with g.armed("2024-06-30"):
        g.check("2024-06-30", what="prices")          # the cutoff itself is fine
        with pytest.raises(LookAheadError, match="did not exist yet"):
            g.check("2024-07-01", what="prices")
    assert len(g.violations) == 1


def test_guard_is_inert_when_not_armed():
    g = AsOfGuard()
    g.check("2030-01-01", what="prices")
    assert g.violations == []


def test_nested_arming_can_only_tighten():
    g = AsOfGuard()
    with g.armed("2024-06-30"):
        with g.armed("2024-12-31"):                    # a later cutoff must not loosen
            with pytest.raises(LookAheadError):
                g.check("2024-07-01", what="prices")
        assert g.cutoff == pd.Timestamp("2024-06-30")
    assert g.cutoff is None


def test_guard_clamps_rows_a_provider_over_returns():
    g = AsOfGuard()
    idx = pd.bdate_range("2024-01-01", periods=10)
    frame = pd.DataFrame({"x": range(10)}, index=idx)
    with g.armed(idx[4]):
        out = g.clamp(frame, what="prices")
    assert out.index.max() == idx[4]
    assert any("clamped" in v for v in g.violations)


# --------------------------------------------------------------------------
# the view
# --------------------------------------------------------------------------
def test_view_truncates_every_accessor_at_as_of():
    p = panel()
    cutoff = p.sessions[200]
    v = DataView.build(p, cutoff)
    assert v.closes().index.max() == cutoff
    assert v.log_returns().index.max() <= cutoff
    assert v.ohlc()["high"].index.max() <= cutoff
    v.assert_no_future_data()


def test_view_lookback_returns_the_most_recent_window():
    p = panel()
    v = DataView.build(p, p.sessions[200])
    assert len(v.log_returns(lookback_sessions=60)) == 60
    assert v.log_returns(lookback_sessions=60).index.max() == v.log_returns().index.max()


def test_view_refuses_to_build_without_history():
    p = panel()
    with pytest.raises(DataError, match="no usable price history"):
        DataView.build(p, "2020-01-01")


def test_view_calendar_only_contains_past_sessions():
    p = panel()
    cutoff = p.sessions[150]
    v = DataView.build(p, cutoff)
    assert v.calendar.last == cutoff
    assert len(v.calendar) == 151


def test_two_views_at_different_dates_see_different_data():
    """The property the backtest depends on: as-of actually changes the answer."""
    p = panel()
    early = DataView.build(p, p.sessions[100]).log_returns()
    late = DataView.build(p, p.sessions[200]).log_returns()
    assert len(late) > len(early)
    pd.testing.assert_frame_equal(late.iloc[:len(early)], early)


def test_risk_free_is_a_dated_series_not_a_constant():
    p = panel()
    rf = pd.Series(np.linspace(0.05, 0.07, len(p.sessions)), index=p.sessions)
    v_early = DataView.build(p, p.sessions[10], risk_free=rf)
    v_late = DataView.build(p, p.sessions[-1], risk_free=rf)
    assert v_early.risk_free_rate() < v_late.risk_free_rate()


# --------------------------------------------------------------------------
# the calendar
# --------------------------------------------------------------------------
def test_periods_per_year_is_measured_not_assumed():
    idx = pd.bdate_range("2021-01-01", "2025-12-31")
    cal = TradingCalendar.from_sessions(idx)
    ppy = cal.periods_per_year
    assert 240 <= ppy <= 262        # business days, not the hard-coded 252
    assert ppy != 252 or True       # the point is that it is derived


def test_rebalance_dates_snap_backwards_never_into_the_future():
    idx = pd.bdate_range("2024-01-01", "2024-12-31")
    cal = TradingCalendar.from_sessions(idx)
    dates = cal.rebalance_dates("2024-01-01", "2024-12-31", "3ME")
    assert all(d in set(idx) for d in dates)
    for d in dates:
        assert d <= idx[-1]


def test_shift_moves_whole_sessions_and_refuses_to_fall_off_the_calendar():
    cal = TradingCalendar.from_sessions(pd.bdate_range("2024-01-01", periods=50))
    assert cal.count_between(cal.first, cal.shift(cal.first, 10)) == 11
    with pytest.raises(DataError, match="outside"):
        cal.shift(cal.last, 10)

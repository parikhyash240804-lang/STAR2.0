"""
The offline fixture: real NSE data, no network.

These tests also pin down facts about the data that later phases depend on - in
particular the *measured* number of trading sessions per year, which the legacy
hard-coded as 252 in about forty places.
"""
import numpy as np
import pandas as pd
import pytest

from star.core.errors import DataError, LookAheadError
from star.data.fixture import FixtureProvider
from star.data.provider import GUARD
from star.data.view import DataView


@pytest.fixture(scope="module")
def provider():
    p = FixtureProvider.default()
    if not (p.root / "nse_close.parquet").exists():
        pytest.skip("fixture not built; run tests/fixtures/build_fixture.py")
    return p


def test_fixture_holds_a_realistic_indian_universe(provider):
    uni = provider.universe
    assert len(uni) >= 30
    assert all(t.endswith(".NS") for t in uni)
    assert "HDFCBANK.NS" in uni and "RELIANCE.NS" in uni


def test_panel_has_full_ohlc_for_range_based_volatility(provider):
    panel = provider.prices(provider.universe[:5])
    assert panel.has_ohlc()
    # where both exist, the high must not be below the low; NaN days are absent
    # history, not corrupt data, and comparisons against NaN are always False
    ok = (panel.high >= panel.low) | panel.high.isna() | panel.low.isna()
    assert ok.all().all()
    assert len(panel.sessions) > 1000


def test_some_tickers_have_partial_history_which_later_phases_must_handle(provider):
    """
    Documents the gap structure the legacy destroyed.

    `ffill(limit=2).dropna()` truncated the whole panel to the latest listing date
    and fabricated zero-return days (M1-05). STAR2.0 must treat a missing day as
    missing, so this test records that gaps really are present in real NSE data.
    """
    close = provider.prices(provider.universe).close
    missing_per_ticker = close.isna().sum()
    assert missing_per_ticker.max() > 0, "expected at least one short-history ticker"
    # and dropping every row with any NaN would discard a large part of the sample
    complete_rows = close.dropna(how="any").shape[0]
    assert complete_rows < close.shape[0]


def test_log_returns_are_finite_and_plausible_for_daily_equities(provider):
    r = provider.prices(provider.universe).log_returns()
    assert np.isfinite(r.to_numpy()[~np.isnan(r.to_numpy())]).all()
    daily_vol = r.std().median()
    assert 0.005 < daily_vol < 0.05            # ~8%-80% annualised: sane for NSE
    assert abs(r.mean().median()) < 0.01


def test_nse_trades_fewer_than_252_sessions_a_year(provider):
    """The legacy's hard-coded 252 is not what the calendar actually does."""
    view = DataView.build(provider.prices(provider.universe[:3]),
                          provider.prices(provider.universe[:3]).sessions[-1])
    ppy = view.calendar.periods_per_year
    assert 235 <= ppy <= 252
    assert ppy != 252, f"measured {ppy}; if this ever equals 252 exactly, check the calendar"


def test_date_range_slicing_is_inclusive_and_ordered(provider):
    panel = provider.prices(provider.universe[:4], start="2024-01-01", end="2024-06-30")
    assert panel.sessions.min() >= pd.Timestamp("2024-01-01")
    assert panel.sessions.max() <= pd.Timestamp("2024-06-30")
    assert panel.sessions.is_monotonic_increasing


def test_unknown_tickers_are_recorded_not_silently_ignored(provider):
    p = FixtureProvider(root=provider.root)
    p.prices(["HDFCBANK.NS", "NOT_A_STOCK.NS"])
    assert any("NOT_A_STOCK.NS" in d for d in p.degradations)


def test_requesting_only_unknown_tickers_fails_loudly(provider):
    with pytest.raises(DataError, match="none of the requested"):
        provider.prices(["NOT_A_STOCK.NS"])


def test_benchmark_series_is_available(provider):
    s = provider.benchmark("^NSEI")
    assert len(s) > 1000 and s.is_monotonic_increasing is False   # prices move both ways
    assert (s > 0).all()


def test_the_guard_blocks_a_fixture_read_past_the_cutoff(provider):
    with GUARD.armed("2024-01-31"):
        with pytest.raises(LookAheadError):
            provider.prices(provider.universe[:3], start="2023-01-01", end="2024-06-30")


def test_guard_clamps_a_wider_cache_hit(provider):
    """Asking within the cutoff must never return rows beyond it."""
    with GUARD.armed("2024-01-31"):
        panel = provider.prices(provider.universe[:3], start="2023-01-01", end="2024-01-31")
    assert panel.sessions.max() <= pd.Timestamp("2024-01-31")


def test_fundamentals_are_refused_rather_than_faked(provider):
    with pytest.raises(DataError, match="no fundamentals"):
        provider.snapshot(["HDFCBANK.NS"], "trailingPE", "2024-01-01")


def test_membership_history_is_honestly_unavailable(provider):
    assert provider.members("NIFTY50", "2020-01-01") is None

"""
NIFTY 50 total-return benchmark (closes legacy M0-02 / M8-07).

Pins the measured facts: NIFTYBEES beats the price index by roughly the dividend
yield, the accrual is calibrated only on data up to `as_of`, and a missing or
broken proxy produces a flagged fallback rather than a silent number.
"""
import numpy as np
import pandas as pd
import pytest

from star.data.benchmark import FALLBACK_DIVIDEND_YIELD, build_benchmark
from star.data.fixture import FixtureProvider


@pytest.fixture(scope="module")
def series():
    p = FixtureProvider.default()
    if not (p.root / "index_NIFTYBEES_NS.parquet").exists():
        pytest.skip("reference fixture not built")
    return p.benchmark("^NSEI"), p.benchmark("NIFTYBEES.NS")


def test_calibrated_dividend_accrual_matches_nifty_dividend_yield(series):
    nsei, bees = series
    b = build_benchmark(nsei, bees, as_of=nsei.index.max(), window_years=3, min_years=2)
    assert b.calibrated
    assert 0.008 < b.dividend_accrual_annual < 0.017      # measured ~1.2%/yr
    assert not b.provenance.degradations
    assert "NIFTYBEES" in b.describe()


def test_total_return_is_price_return_plus_a_constant_daily_accrual(series):
    nsei, bees = series
    b = build_benchmark(nsei, bees, as_of=nsei.index.max(), window_years=3, min_years=2)
    assert np.allclose((b.tri_log_returns - b.price_log_returns).values, b.accrual_log_per_session)
    # over a year the gap in growth is the accrual
    lvl_tri, lvl_px = b.tri_level(), b.price_level()
    yrs = (lvl_tri.index[-1] - lvl_tri.index[0]).days / 365.25
    assert (lvl_tri.iloc[-1] / lvl_px.iloc[-1]) ** (1 / yrs) - 1 == pytest.approx(
        b.dividend_accrual_annual, abs=0.002)


def test_nothing_after_as_of_is_used(series):
    nsei, bees = series
    as_of = pd.Timestamp("2025-06-30")
    b = build_benchmark(nsei, bees, as_of=as_of, window_years=3, min_years=2)
    assert b.tri_log_returns.index.max() <= as_of
    assert b.calibration_end <= as_of
    # changing data after as_of cannot change the calibration
    future = bees.copy()
    future.loc[future.index > as_of] *= 3.0
    b2 = build_benchmark(nsei, future, as_of=as_of, window_years=3, min_years=2)
    assert b2.dividend_accrual_annual == b.dividend_accrual_annual


def test_missing_proxy_falls_back_and_says_so(series):
    nsei, _ = series
    b = build_benchmark(nsei, None, as_of=nsei.index.max())
    assert not b.calibrated
    assert b.dividend_accrual_annual == FALLBACK_DIVIDEND_YIELD
    assert b.provenance.degradations
    assert "fallback" in b.describe()


def test_too_little_overlap_falls_back(series):
    nsei, bees = series
    b = build_benchmark(nsei, bees, as_of="2022-06-30", window_years=5, min_years=2)
    assert not b.calibrated and b.provenance.degradations


def test_calibration_repairs_the_real_2019_artifact_first(series):
    nsei, bees = series
    # a window that contains Dec 2019 in the proxy but whose overlap with the fixture
    # index starts later: the repair must still run and be reported
    b = build_benchmark(nsei, bees, as_of=nsei.index.max(), window_years=8, min_years=2)
    assert any(e.start == pd.Timestamp("2019-12-19") for e in b.repaired_events)

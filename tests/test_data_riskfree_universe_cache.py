"""Risk-free curve (M0-11), point-in-time universe (M8-02 look-ahead half), data cache (M0-15)."""
import numpy as np
import pandas as pd
import pytest

from star.core.errors import DataError, LookAheadError
from star.data.cache import CachedPriceProvider, DataCache
from star.data.fixture import FixtureProvider
from star.data.provider import GUARD
from star.data.riskfree import RiskFreeCurve
from star.data.universe import as_of_universe


# ---------------------------------------------------------------- risk-free
def _curve():
    return RiskFreeCurve.from_series(
        pd.Series([4.0, 4.9, 6.5], index=pd.to_datetime(["2020-05-22", "2022-05-04", "2023-02-08"])),
        name="repo", source="test")


def test_percent_quotes_are_converted_and_noted():
    c = _curve()
    assert c.rates.max() == pytest.approx(0.065)
    assert any("percent" in n for n in c.notes)
    assert not c.degradations


def test_rate_is_as_of_never_a_later_observation():
    c = _curve()
    assert c.rate("2021-01-01") == pytest.approx(0.040)     # the legacy used 6.5% here
    assert c.rate("2022-05-04") == pytest.approx(0.049)
    assert c.rate("2024-01-01") == pytest.approx(0.065)
    with pytest.raises(DataError):
        c.rate("2019-01-01")


def test_on_sessions_is_a_step_function():
    c = _curve()
    days = pd.bdate_range("2022-05-02", "2022-05-06")
    s = c.on_sessions(days)
    assert list(np.round(s.values, 3)) == [0.040, 0.040, 0.049, 0.049, 0.049]


def test_daily_rate_compounds_back_to_the_annual_rate():
    c = RiskFreeCurve.constant(0.065, reason="test")
    d = c.daily_simple(pd.bdate_range("2024-01-01", periods=3), periods_per_year=252)
    assert (1 + d.iloc[0]) ** 252 - 1 == pytest.approx(0.065)


def test_constant_curve_always_carries_a_degradation():
    c = RiskFreeCurve.constant(0.065, reason="no RBI file supplied")
    assert c.is_constant and c.provenance.degradations


def test_implausible_units_are_rejected():
    with pytest.raises(DataError):
        RiskFreeCurve(rates=pd.Series([0.65], index=pd.DatetimeIndex(["2024-01-01"])), name="x", source="x")


def test_csv_loading(tmp_path):
    p = tmp_path / "tbill.csv"
    p.write_text("DATE,INTGSTINM193N\n2021-01-01,3.2\n2022-01-01,3.8\n2023-01-01,6.4\n")
    c = RiskFreeCurve.from_csv(p, quote="effective")
    assert c.rate("2022-06-30") == pytest.approx(0.038)
    assert c.source == "csv:tbill.csv"


def test_guard_blocks_future_rate_requests():
    c = _curve()
    with GUARD.armed("2022-01-01"):
        c.rate("2021-12-31")
        with pytest.raises(LookAheadError):
            c.rate("2022-06-01")


# ---------------------------------------------------------------- universe
@pytest.fixture(scope="module")
def panel():
    p = FixtureProvider.default()
    if not (p.root / "nse_close.parquet").exists():
        pytest.skip("fixture not built")
    return p.prices(p.universe)


def test_a_stock_not_yet_listed_is_excluded_as_unlisted(panel):
    rep = as_of_universe(panel.close, "2023-06-30", min_adv_inr=0.0)
    hyundai = next(v for v in rep.verdicts if v.ticker == "HYUNDAI.NS")
    assert hyundai.dropped and hyundai.reason == "no price data"


def test_screening_ignores_data_after_as_of(panel):
    close = panel.close.copy()
    as_of = pd.Timestamp("2024-03-28")
    base = as_of_universe(close, as_of, min_adv_inr=0.0)
    # wreck a kept stock's data AFTER as_of: the as-of decision must not change
    keep = base.kept[0]
    close.loc[close.index > as_of, keep] = np.nan
    again = as_of_universe(close, as_of, min_adv_inr=0.0)
    assert again.kept == base.kept
    assert any("as-of" in n for n in again.notes)


# ---------------------------------------------------------------- cache
def test_cache_round_trip_and_corruption_detection(tmp_path):
    cache = DataCache(tmp_path)
    df = pd.DataFrame({"A": [1.0, 2.0]}, index=pd.bdate_range("2024-01-01", periods=2))
    digest = cache.put("k", df, meta={"source": "t"})
    got, manifest = cache.get("k")
    assert got.equals(df) and manifest["sha256"] == digest
    (tmp_path / "k.parquet").write_bytes(b"corrupted")
    assert cache.get("k") is None


def test_cached_provider_replays_without_refetching(tmp_path, panel):
    calls = []

    class Counting:
        name = "fixture"

        def prices(self, tickers, start, end, *, adjusted=True):
            calls.append(1)
            return FixtureProvider.default().prices(tickers, start, end)

    cp = CachedPriceProvider(Counting(), DataCache(tmp_path))
    tick = ["LT.NS", "ITC.NS", "INFY.NS"]
    a = cp.prices(tick, "2023-01-01", "2024-01-01")
    fp1 = cp.data_fingerprint()
    b = cp.prices(tick, "2023-01-01", "2024-01-01")
    assert len(calls) == 1
    assert a.close.equals(b.close)
    assert cp.data_fingerprint() == fp1

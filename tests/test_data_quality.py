"""
Data quality: every exclusion is explained.

These tests run on the real fixture, so they also pin the *measured* behaviour of
the default NSE universe: two stocks are excluded, and the reason for the big one
is a 2024 listing rather than bad data.
"""
import numpy as np
import pandas as pd
import pytest

from star.core.errors import DataError
from star.data.fixture import FixtureProvider
from star.data.quality import Disposition, assess, common_sample


@pytest.fixture(scope="module")
def panel():
    p = FixtureProvider.default()
    if not (p.root / "nse_close.parquet").exists():
        pytest.skip("fixture not built")
    return p.prices(p.universe)


def test_the_default_universe_loses_exactly_the_short_history_names(panel):
    rep = assess(panel.close, volume=panel.volume, min_adv_inr=0.0)
    dropped = {v.ticker for v in rep.dropped}
    assert "HYUNDAI.NS" in dropped        # listed late 2024
    assert len(rep.kept) == len(rep.requested) - len(dropped)
    assert len(rep.kept) >= 35


def test_a_recent_listing_is_named_as_such_not_called_bad_data(panel):
    rep = assess(panel.close, min_adv_inr=0.0)
    hyundai = next(v for v in rep.verdicts if v.ticker == "HYUNDAI.NS")
    assert hyundai.disposition is Disposition.DROP
    assert "listed" in hyundai.reason
    assert hyundai.first_session > panel.sessions.min()


def test_headline_is_a_sentence_a_user_can_read(panel):
    rep = assess(panel.close, min_adv_inr=0.0)
    line = rep.headline()
    assert "stocks usable" in line and "Excluded:" in line
    assert "HYUNDAI" in line


def test_verdict_table_covers_every_requested_ticker(panel):
    rep = assess(panel.close, min_adv_inr=0.0)
    assert len(rep.to_frame()) == len(panel.close.columns)
    assert set(rep.to_frame()["disposition"]) <= {"keep", "drop"}


def test_liquidity_is_screened_in_rupees_not_share_count():
    """
    The legacy's 10,000-*share* floor is not a liquidity test: 10,000 shares of a
    Rs.50 stock is Rs.5 lakh a day, and of a Rs.5,000 stock is Rs.5 crore.
    """
    idx = pd.bdate_range("2024-01-01", periods=300)
    close = pd.DataFrame({"CHEAP.NS": 50.0, "DEAR.NS": 5000.0}, index=idx)
    volume = pd.DataFrame({"CHEAP.NS": 10_000.0, "DEAR.NS": 10_000.0}, index=idx)
    rep = assess(close, volume=volume, min_adv_inr=1_000_000, min_names=1)
    by = {v.ticker: v for v in rep.verdicts}
    assert by["CHEAP.NS"].dropped        # Rs.5 lakh/day: below the floor
    assert not by["DEAR.NS"].dropped     # Rs.5 crore/day: fine
    assert "turnover" in by["CHEAP.NS"].reason


def test_common_sample_never_fabricates_a_day():
    idx = pd.bdate_range("2024-01-01", periods=10)
    close = pd.DataFrame({"A.NS": np.arange(10.0), "B.NS": np.arange(10.0)}, index=idx)
    close.loc[idx[5], "B.NS"] = np.nan
    out = common_sample(close, ["A.NS", "B.NS"])
    assert len(out) == 9
    assert idx[5] not in out.index
    assert out.notna().all().all()


def test_too_few_usable_names_fails_loudly_with_the_reason():
    idx = pd.bdate_range("2024-01-01", periods=300)
    close = pd.DataFrame({"A.NS": 100.0, "B.NS": np.nan, "C.NS": np.nan}, index=idx)
    with pytest.raises(DataError, match="usable stock"):
        assess(close, min_names=3)


def test_short_history_exclusions_are_explained_in_the_notes(panel):
    rep = assess(panel.close, min_adv_inr=0.0)
    assert any("truncate the common sample" in n for n in rep.notes)

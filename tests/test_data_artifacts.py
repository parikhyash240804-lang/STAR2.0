"""
Price artifacts: repair only the impossible, flag the ambiguous, keep real crashes.

The regression cases are real: Yahoo's NIFTYBEES series (a broken Dec-2019 unit
split) and the 4 June 2024 election-day falls in L&T and Power Grid.
"""
import numpy as np
import pandas as pd
import pytest

from star.data.fixture import FixtureProvider
from star.data.quality import ArtifactAction, ArtifactKind, repair_price_artifacts, scan_price_artifacts


@pytest.fixture(scope="module")
def provider():
    p = FixtureProvider.default()
    if not (p.root / "index_NIFTYBEES_NS.parquet").exists():
        pytest.skip("reference fixture not built")
    return p


def _series(values, start="2024-01-01"):
    return pd.Series(values, index=pd.bdate_range(start, periods=len(values)), name="X.NS", dtype=float)


def test_real_niftybees_split_artifact_is_repaired(provider):
    bees = provider.benchmark("NIFTYBEES.NS")
    res = repair_price_artifacts(bees)
    repaired = res.repaired
    assert len(repaired) == 1
    e = repaired[0]
    assert e.kind is ArtifactKind.SPIKE_REVERT
    assert e.start == pd.Timestamp("2019-12-19") and e.end == pd.Timestamp("2019-12-23")
    fixed = res.close.iloc[:, 0]
    assert fixed.pct_change().abs().max() < 0.15          # was +897%
    # the repair preserves the true move across the whole history
    assert fixed.iloc[-1] / fixed.iloc[0] == pytest.approx(bees.iloc[-1] / bees.iloc[0], rel=1e-12)


def test_genuine_election_day_crash_is_kept(provider):
    close = provider.prices(["LT.NS", "POWERGRID.NS"]).close
    res = repair_price_artifacts(close)
    assert not res.repaired
    assert (res.close.fillna(-1) == close.fillna(-1)).all().all()
    election = [e for e in res.flagged if e.start == pd.Timestamp("2024-06-04")]
    assert {e.ticker for e in election} == {"LT.NS", "POWERGRID.NS"}
    assert all(e.kind is ArtifactKind.LARGE_MOVE for e in election)


def test_whole_stock_universe_is_left_unchanged(provider):
    close = provider.prices(provider.universe).close
    res = repair_price_artifacts(close)
    assert res.repaired == []
    assert (res.close.fillna(-1) == close.fillna(-1)).all().all()


def test_unadjusted_split_is_rescaled_not_left_as_a_crash():
    rng = np.random.default_rng(3)
    px = 1000 * np.exp(np.cumsum(rng.normal(0, 0.01, 300)))
    px[150:] /= 10.0                                          # 1:10 split never adjusted
    res = repair_price_artifacts(_series(px))
    assert [e.kind for e in res.repaired] == [ArtifactKind.SPLIT_LIKE_JUMP]
    assert res.close.iloc[:, 0].pct_change().abs().max() < 0.10


def test_small_spike_that_reverts_is_flagged_not_changed():
    rng = np.random.default_rng(5)
    px = 500 * np.exp(np.cumsum(rng.normal(0, 0.005, 200)))
    px[100] *= 1.14                                           # +14% for one day, then back
    s = _series(px)
    res = repair_price_artifacts(s)
    assert not res.repaired
    assert any(e.kind is ArtifactKind.SPIKE_REVERT and e.action is ArtifactAction.FLAGGED for e in res.flagged)
    assert np.allclose(res.close.iloc[:, 0].values, s.values)


def test_quiet_series_produces_no_events():
    rng = np.random.default_rng(9)
    s = _series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 500))))
    assert scan_price_artifacts(s) == []

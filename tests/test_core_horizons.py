"""Horizons are configuration, not a hard-coded 252."""
import pytest

from star.core import HorizonGrid, HorizonSpec
from star.core.errors import ConfigError


@pytest.mark.parametrize("token,days", [
    ("1d", 1), ("2d", 2), ("1w", 5), ("2w", 10),
    ("1m", 21), ("3m", 63), ("6m", 126), ("1y", 252),
    ("45", 45), (45, 45), ("1Y", 252), (" 3m ", 63),
])
def test_parse_covers_every_horizon_the_owner_asked_for(token, days):
    assert HorizonSpec.parse(token).days == days


@pytest.mark.parametrize("token,label", [
    ("1d", "1D"), ("1w", "1W"), ("1m", "1M"), ("3m", "3M"), ("1y", "1Y"), ("45", "45D"),
])
def test_labels_are_human_readable(token, label):
    assert HorizonSpec.parse(token).label == label


def test_a_week_is_five_trading_days_not_a_calendar_fraction():
    assert HorizonSpec.parse("1w").days == 5


def test_bad_token_is_a_clear_error():
    with pytest.raises(ConfigError, match="cannot read horizon"):
        HorizonSpec.parse("next quarter")


def test_zero_or_negative_horizon_is_rejected():
    with pytest.raises(ConfigError):
        HorizonSpec(days=0, label="0D")


def test_default_grid_is_the_seven_requested_horizons():
    g = HorizonGrid.default()
    assert g.days == (1, 2, 5, 21, 63, 126, 252)
    assert g.labels == ("1D", "2D", "1W", "1M", "3M", "6M", "1Y")
    assert g.max_days == 252
    assert len(g) == 7


def test_grid_sorts_and_deduplicates():
    g = HorizonGrid.parse("1y,1d,1m,21,1d")
    assert g.days == (1, 21, 252)


def test_grid_accepts_arbitrary_custom_days():
    g = HorizonGrid.default().with_custom(37)
    assert 37 in g.days
    assert g.label_for(37) == "37D"
    assert g.max_days == 252        # a custom horizon below the max costs nothing


def test_custom_horizon_beyond_the_max_extends_the_simulation():
    g = HorizonGrid.parse("1d,1y").with_custom(500)
    assert g.max_days == 500


def test_horizons_past_two_years_are_refused_as_extrapolation():
    with pytest.raises(ConfigError, match="cap"):
        HorizonGrid.parse("1d,4y")


def test_cap_can_be_raised_deliberately():
    g = HorizonGrid.parse("1d,4y", cap_days=1260)
    assert g.max_days == 1008


def test_empty_grid_is_rejected():
    with pytest.raises(ConfigError):
        HorizonGrid.parse("")


def test_grid_is_iterable_and_ordered():
    assert [s.days for s in HorizonGrid.default()] == sorted(HorizonGrid.default().days)
